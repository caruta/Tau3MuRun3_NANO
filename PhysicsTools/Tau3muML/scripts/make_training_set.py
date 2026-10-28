#!/usr/bin/env python3
"""Build the transformer training set from the flattened ntuples (run at Bari, inside cmsenv).

Roles (meta "role"):
  0 signal      MC, channel 0 (|Q| = 1), label_sig == 1
  1 background  data, channel 0, sidebands (the flattener already removed 1.74-1.82 GeV)
  2 SS3MU       data, channel 2 (|Q| = 3), all masses: not trained on, used for the sculpting check

Files of each role are read in random order (so all eras and streams contribute) and a random
fraction --row-frac of the selected rows is kept, until the role's target is reached. Rows are
assigned to random shards, so every shard mixes roles, eras and folds; each shard is shuffled.

Output (copy the whole directory to EOS for SWAN):
  <out>/stats.json                   feature names, normalisation, sample ids, counts
  <out>/shard_00000/<array>.npy      cand, <tok>_feats (float16), <tok>_mask (bool), meta arrays

  python3 make_training_set.py --flat-dir /lustre/cms/store/user/<you>/flat_Transformer_v1 \\
      --out /lustre/cms/store/user/<you>/train_Transformer_v1
"""
import argparse
import fnmatch
import glob
import json
import os
import resource
import shutil
import sys
import time

import awkward as ak
import numpy as np
import uproot

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tau3muml import features as F  # noqa: E402

ROLES = {"signal": 0, "background": 1, "ss3mu": 2}
META = ["label", "role", "sample", "fold", "mass", "nPVtx", "gen_weight", "sig_prod", "run", "event", "cand_idx"]
META_DTYPES = {"label": np.int8, "role": np.int8, "sample": np.int16, "fold": np.int8, "mass": np.float32,
               "nPVtx": np.float32, "gen_weight": np.float32, "sig_prod": np.int8, "run": np.int32,
               "event": np.int64, "cand_idx": np.int16}


def list_samples(flat_dir, channel, patterns):
    cdir = os.path.join(flat_dir, channel)
    out = []
    for s in sorted(os.listdir(cdir)):
        if any(fnmatch.fnmatch(s, p) for p in patterns):
            files = sorted(glob.glob(os.path.join(cdir, s, "*_fold*.root")))
            if files:
                out.append((s, files))
    return out


def select(arr, role):
    ch = ak.to_numpy(arr["cand_channel"])
    data = ak.to_numpy(arr["isData"]) == 1
    if role == "signal":
        return (~data) & (ch == 0) & (ak.to_numpy(arr["label_sig"]) == 1)
    if role == "background":
        return data & (ch == 0)
    return data & (ch == 2)


def meta_arrays(arr, role, sample_id):
    n = len(arr)

    def get(name, default):
        return ak.to_numpy(arr[name]) if name in arr.fields else np.full(n, default)

    return {
        "label": np.full(n, 1 if role == "signal" else 0),
        "role": np.full(n, ROLES[role]),
        "sample": np.full(n, sample_id),
        "fold": get("fold", 0),
        "mass": F.fit_mass(arr),
        "nPVtx": get("nPVtx", 0),
        "gen_weight": get("Generator_weight", 1.0),
        "sig_prod": get("sig_prod", 0),
        "run": get("run", 0),
        "event": get("event", 0),
        "cand_idx": get("cand_idx", 0),
    }


class ShardWriter:
    """Stage 1: rows go to random shards, buffered and flushed to temporary parts.
    Stage 2: each shard's parts are joined, shuffled and written as .npy arrays."""

    def __init__(self, out, n_shards, rng, flush_rows=50_000):
        self.out, self.n_shards, self.rng, self.flush_rows = out, n_shards, rng, flush_rows
        self.tmp = os.path.join(out, "_tmp")
        os.makedirs(self.tmp, exist_ok=True)
        self.buf, self.nbuf, self.nparts = [], 0, 0

    def add(self, block):
        self.buf.append(block)
        self.nbuf += len(block["label"])
        if self.nbuf >= self.flush_rows:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        cat = {k: np.concatenate([b[k] for b in self.buf]) for k in self.buf[0]}
        shard = self.rng.integers(0, self.n_shards, len(cat["label"]))
        for s in range(self.n_shards):
            sel = shard == s
            if sel.any():
                np.savez(os.path.join(self.tmp, f"s{s:05d}_p{self.nparts:05d}.npz"), **{k: v[sel] for k, v in cat.items()})
        self.nparts += 1
        self.buf, self.nbuf = [], 0

    def finalize(self, stats_rows=300_000):
        """Write shards; returns running sums for the normalisation (from the first shards)."""
        self.flush()
        acc, seen, written = {}, 0, 0
        for s in range(self.n_shards):
            parts = sorted(glob.glob(os.path.join(self.tmp, f"s{s:05d}_p*.npz")))
            if not parts:
                continue
            loaded = [dict(np.load(p)) for p in parts]
            cat = {k: np.concatenate([l[k] for l in loaded]) for k in loaded[0]}
            perm = self.rng.permutation(len(cat["label"]))
            d = os.path.join(self.out, f"shard_{written:05d}")
            os.makedirs(d, exist_ok=True)
            for k, v in cat.items():
                v = v[perm]
                np.save(os.path.join(d, f"{k}.npy"), v)
            if seen < stats_rows:
                accumulate(acc, cat, train_only=cat["role"] != ROLES["ss3mu"])
                seen += len(cat["label"])
            written += 1
            del cat, loaded
            for p in parts:
                os.remove(p)
        shutil.rmtree(self.tmp, ignore_errors=True)
        return acc, written


def accumulate(acc, cat, train_only):
    """Sums for mean/std over real tokens (and over the signal+background rows)."""
    for k, v in cat.items():
        if k == "cand":
            x, m = v[train_only], None
        elif k.endswith("_feats"):
            x = v[train_only]
            m = cat[k.replace("_feats", "_mask")][train_only]
            x = x[m]
        else:
            continue
        a = acc.setdefault(k, [0.0, np.zeros(x.shape[-1]), np.zeros(x.shape[-1])])
        for i in range(0, len(x), 200_000):
            xi = x[i : i + 200_000].astype(np.float64)
            a[0] += len(xi)
            a[1] += xi.sum(0)
            a[2] += (xi ** 2).sum(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flat-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--signal", nargs="+", default=["*_MC_Ds", "*_MC_Bd", "*_MC_Bu"],
                    help="Tau3Mu sample patterns used as signal")
    ap.add_argument("--data", nargs="+", default=["*era*"], help="Tau3Mu sample patterns used as data")
    ap.add_argument("--n-signal", type=float, default=1.0e6)
    ap.add_argument("--n-background", type=float, default=1.0e6)
    ap.add_argument("--n-ss3mu", type=float, default=2.0e5)
    ap.add_argument("--row-frac-data", type=float, default=0.2,
                    help="fraction of selected data rows kept per file (spreads the sample over more files)")
    ap.add_argument("--row-frac-mc", type=float, default=1.0)
    ap.add_argument("--shard-size", type=int, default=50_000)
    ap.add_argument("--use-activity", action="store_true",
                    help="also use track/neutral/SV counts (pileup dependent, off by default)")
    ap.add_argument("--step", type=int, default=20_000, help="candidates read per chunk (memory ~ 30 kB x step)")
    ap.add_argument("--flush-rows", type=int, default=50_000,
                    help="rows buffered before writing temporary parts (memory ~ 10 kB x flush-rows)")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    rng = np.random.default_rng(args.seed)
    if os.path.exists(os.path.join(args.out, "stats.json")):
        sys.exit(f"{args.out} already contains a training set; choose another --out")
    os.makedirs(args.out, exist_ok=True)

    samples = {"signal": list_samples(args.flat_dir, "Tau3Mu", args.signal),
               "background": list_samples(args.flat_dir, "Tau3Mu", args.data)}
    samples["ss3mu"] = samples["background"]
    for role in ("signal", "background"):
        if not samples[role]:
            sys.exit(f"no {role} samples matching {getattr(args, 'signal' if role == 'signal' else 'data')} "
                     f"in {args.flat_dir}/Tau3Mu")
    sample_ids = {s: i for i, s in enumerate(sorted({s for v in samples.values() for s, _ in v}))}

    first = samples["signal"][0][1][0]
    with uproot.open(first) as fh:
        fields = fh["Events"].keys()
    names = F.default_feature_names(fields, use_activity=args.use_activity)
    branches = F.required_branches(names)
    print("features:", json.dumps({"cand": names["cand"], **names["tokens"]}))
    if names["skipped"]:
        print("skipped (branches missing):", names["skipped"])

    targets = {"signal": args.n_signal, "background": args.n_background, "ss3mu": args.n_ss3mu}
    n_shards = max(1, int(round(sum(targets.values()) / args.shard_size)))
    writer = ShardWriter(args.out, n_shards, rng, flush_rows=args.flush_rows)
    counts = {}
    for role, target in targets.items():
        target = int(target)
        if target <= 0:
            continue
        files = [(s, f) for s, fl in samples[role] for f in fl]
        rng.shuffle(files)
        frac = args.row_frac_mc if role == "signal" else args.row_frac_data
        n, per_sample, t0 = 0, {}, time.time()
        for i, (sample, path) in enumerate(files):
            if n >= target:
                break
            try:
                it = uproot.iterate(f"{path}:Events", filter_name=branches, step_size=args.step)
                for arr in it:
                    sel = select(arr, role)
                    sel &= rng.random(len(sel)) < frac
                    if not sel.any():
                        continue
                    arr = arr[sel][: target - n]
                    block = F.compute(arr, names, dtype=np.float16)
                    block.update({k: v.astype(META_DTYPES[k]) for k, v in meta_arrays(arr, role, sample_ids[sample]).items()})
                    writer.add(block)
                    n += len(arr)
                    per_sample[sample] = per_sample.get(sample, 0) + len(arr)
                    if n >= target:
                        break
            except Exception as e:
                print(f"  skipping {path}: {type(e).__name__}: {str(e)[:150]}")
            if i % 20 == 0:
                print(f"{role}: file {i + 1}/{len(files)}, {n}/{target} rows, {time.time() - t0:.0f} s, "
                      f"max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB")
        print(f"{role}: {n} rows from {len(per_sample)} samples: {per_sample}")
        if n < target:
            print(f"  WARNING: only {n} of {target} requested rows available"
                  + (" (raise --row-frac-data)" if role != "signal" else ""))
        counts[role] = {"total": n, "per_sample": per_sample}

    print("writing shards ...")
    acc, written = writer.finalize()
    norm = {}
    for k, (cnt, s1, s2) in acc.items():
        mean = s1 / max(cnt, 1)
        std = np.sqrt(np.maximum(s2 / max(cnt, 1) - mean ** 2, 0))
        norm[k] = {"mean": mean.tolist(), "std": np.where(std < 1e-3, 1.0, std).tolist()}
    stats = {"features": {"cand": names["cand"], "tokens": names["tokens"], "token_max": names["token_max"],
                          "geom_index": F.geom_index(names)},
             "norm": norm, "samples": sample_ids, "roles": ROLES, "counts": counts, "n_shards": written,
             "use_activity": args.use_activity, "flat_dir": args.flat_dir, "seed": args.seed}
    with open(os.path.join(args.out, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=1)
    print(f"wrote {written} shards to {args.out}; max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB")


if __name__ == "__main__":
    main()
