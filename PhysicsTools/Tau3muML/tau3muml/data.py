"""Reading the training set written by make_training_set.py, and event weights."""
import glob
import json
import os
import queue
import threading

import numpy as np
import torch

META = ["label", "role", "sample", "fold", "mass", "nPVtx", "gen_weight", "sig_prod", "run", "event", "cand_idx"]


def load_stats(data_dir):
    with open(os.path.join(data_dir, "stats.json")) as fh:
        return json.load(fh)


def shard_dirs(data_dir):
    return sorted(glob.glob(os.path.join(data_dir, "shard_*")))


def input_names(stats):
    out = ["cand"]
    for t in stats["features"]["tokens"]:
        out += [f"{t}_feats", f"{t}_mask"]
    return out


def load_meta(data_dir):
    """Meta columns of all shards (small), plus the shard number of each row."""
    parts, shard = [], []
    for i, d in enumerate(shard_dirs(data_dir)):
        m = {k: np.load(os.path.join(d, f"{k}.npy")) for k in META}
        parts.append(m)
        shard.append(np.full(len(m["label"]), i, dtype=np.int32))
    out = {k: np.concatenate([p[k] for p in parts]) for k in META}
    out["shard"] = np.concatenate(shard)
    return out


def fold_split(test_fold, n_folds=5):
    val = (test_fold + 1) % n_folds
    train = [f for f in range(n_folds) if f not in (test_fold, val)]
    return train, [val], [test_fold]


class Weighter:
    """Training weights.

    signal:     gen weight sign x production-mode weight x nPV reweighting to the background (data)
    background: 1
    SS3MU:      0 (not trained on)
    Then signal is normalised to the background sum (balanced classes).

    prod_frac: {sig_prod code: fraction} (1 Ds, 2 B->Ds, 3 B->tau, 4 D+, 5 W). None keeps the MC mix.
    """

    def __init__(self, meta, rows, prod_frac=None, npv_reweight=True, npv_bins=np.arange(0, 101, 2)):
        sig = (meta["role"] == 0) & rows
        bkg = (meta["role"] == 1) & rows
        self.npv_bins = npv_bins
        self.npv_ratio = None
        base = self._base(meta, prod_frac=None, mask=sig)
        if prod_frac:
            codes, counts = np.unique(meta["sig_prod"][sig], return_counts=True)
            tot = sum(prod_frac.get(int(c), 0.0) for c in codes)
            self.prod_w = {int(c): prod_frac.get(int(c), 0.0) / tot / (n / sig.sum()) for c, n in zip(codes, counts)}
            missing = [c for c in prod_frac if c not in self.prod_w]
            if missing:
                print(f"Weighter: production modes {missing} have no signal rows")
        else:
            self.prod_w = None
        if npv_reweight:
            w_sig = base * self._prod(meta)
            hs, _ = np.histogram(meta["nPVtx"][sig], self.npv_bins, weights=w_sig[sig])
            hb, _ = np.histogram(meta["nPVtx"][bkg], self.npv_bins)
            hs, hb = hs / max(hs.sum(), 1e-9), hb / max(hb.sum(), 1e-9)
            self.npv_ratio = np.where(hs > 0, hb / np.maximum(hs, 1e-12), 0.0)
            self.npv_ratio = np.clip(self.npv_ratio, 0, 10)
        w = self.raw(meta)
        self.sig_norm = w[bkg].sum() / max(w[sig].sum(), 1e-9)

    @staticmethod
    def _base(meta, prod_frac=None, mask=None):
        return np.sign(meta["gen_weight"]).astype(np.float64) + (meta["gen_weight"] == 0)

    def _prod(self, meta):
        if self.prod_w is None:
            return np.ones(len(meta["role"]))
        return np.array([self.prod_w.get(int(c), 0.0) for c in range(8)])[np.clip(meta["sig_prod"], 0, 7)]

    def raw(self, meta):
        role = meta["role"]
        w = np.where(role == 1, 1.0, 0.0)
        ws = self._base(meta) * self._prod(meta)
        if self.npv_ratio is not None:
            idx = np.clip(np.digitize(meta["nPVtx"], self.npv_bins) - 1, 0, len(self.npv_ratio) - 1)
            ws = ws * self.npv_ratio[idx]
        return np.where(role == 0, ws, w)

    def __call__(self, meta):
        w = self.raw(meta)
        return np.where(meta["role"] == 0, w * self.sig_norm, w).astype(np.float32)


def _load_shard(d, names, folds, roles):
    meta = {k: np.load(os.path.join(d, f"{k}.npy")) for k in META}
    sel = np.isin(meta["fold"], folds) & np.isin(meta["role"], roles)
    idx = np.nonzero(sel)[0]
    meta = {k: v[idx] for k, v in meta.items()}
    x = {k: np.load(os.path.join(d, f"{k}.npy"), mmap_mode="r")[idx] for k in names}
    return x, meta


def to_torch(x, device):
    """numpy batch -> torch, trimming each token type to the longest sequence in the batch."""
    out = {}
    for k, v in x.items():
        if k.endswith("_mask"):
            n = max(1, int(v.sum(1).max())) if len(v) else 1
            t = k[:-5]
            out[k] = torch.from_numpy(np.ascontiguousarray(v[:, :n])).to(device, non_blocking=True)
            out[f"{t}_feats"] = torch.from_numpy(np.ascontiguousarray(x[f"{t}_feats"][:, :n])).to(device, non_blocking=True)
        elif k == "cand":
            out[k] = torch.from_numpy(np.ascontiguousarray(v)).to(device, non_blocking=True)
    return out


class ShardLoader:
    """Iterates over batches: shards in random order, rows shuffled inside a shard; the next
    shard is read by a background thread while the current one is used."""

    def __init__(self, data_dir, folds, roles=(0, 1), batch_size=256, shuffle=True, weighter=None,
                 max_rows=None, seed=0, drop_last=False):
        self.stats = load_stats(data_dir)
        self.names = input_names(self.stats)
        self.dirs = shard_dirs(data_dir)
        self.folds, self.roles = list(folds), list(roles)
        self.batch_size, self.shuffle, self.weighter = batch_size, shuffle, weighter
        self.max_rows, self.drop_last = max_rows, drop_last
        self.rng = np.random.default_rng(seed)

    def _producer(self, order, q):
        for d in order:
            q.put(_load_shard(d, self.names, self.folds, self.roles))
        q.put(None)

    def __iter__(self):
        order = list(self.dirs)
        if self.shuffle:
            self.rng.shuffle(order)
        q = queue.Queue(maxsize=1)
        th = threading.Thread(target=self._producer, args=(order, q), daemon=True)
        th.start()
        seen = 0
        while True:
            item = q.get()
            if item is None:
                break
            x, meta = item
            n = len(meta["label"])
            perm = self.rng.permutation(n) if self.shuffle else np.arange(n)
            w = self.weighter(meta) if self.weighter else np.ones(n, np.float32)
            for i in range(0, n, self.batch_size):
                j = perm[i : i + self.batch_size]
                if self.drop_last and len(j) < self.batch_size:
                    break
                if self.max_rows is not None and seen >= self.max_rows:
                    break
                seen += len(j)
                yield {k: v[j] for k, v in x.items()}, {k: v[j] for k, v in meta.items()}, w[j]
            if self.max_rows is not None and seen >= self.max_rows:
                break
        # drain so the producer thread finishes
        while th.is_alive():
            try:
                q.get(timeout=0.1)
            except queue.Empty:
                pass
