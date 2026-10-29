#!/usr/bin/env python3
"""What does a trained model rely on? Permutation tests on a held-out sample.

For one fold model, takes a sample of its test fold (signal and sidebands), then for each group of
inputs shuffles them across candidates (or removes a whole token type) and measures:
  auc            signal vs sidebands
  bkg_eff90      sideband efficiency at 90 % signal efficiency
  near_far       sculpting: among the top 1 % sideband candidates, (1.70-1.74 + 1.82-1.86 GeV) /
                 (1.50-1.62 + 1.94-2.06 GeV), divided by the same ratio for all sidebands (1 = no sculpting)
  early_late     data-taking dependence: sideband efficiency at the top-1 % cut for runs < 383000
                 divided by runs >= 383000 (1 = no dependence)
A group whose shuffling lowers near_far or early_late towards 1 is where the mass or the
data-taking conditions enter; a group whose shuffling costs much AUC is what the model uses.

  python3 -m tau3muml.diagnose --data <DATA> --out <OUT> --fold 0 --n-sig 30000 --n-bkg 60000
Writes <OUT>/eval/diagnose_fold<k>.csv and prints a table.
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

from . import data as D
from .train import device_of, load_model, weighted_auc

NEAR = ((1.70, 1.74), (1.82, 1.86))
FAR = ((1.50, 1.62), (1.94, 2.06))
EARLY_RUN = 383000

# feature groups by name (applied when the feature exists)
TOKEN_GROUPS = {
    "mu": {"kinematics": ["log_ptfrac", "eta", "deta", "dphi"],
           "id": ["isGlobal", "isTracker", "isPF", "isMedium", "isSoft", "isStandalone", "softMva", "softMvaRun3",
                  "mvaId", "segmComp", "caloComp", "nStations"],
           "track_quality": ["log_trkKink", "log_glbKink", "log_chi2LocPos", "log_chi2LocMom", "log_innerChi2",
                             "validFrac", "log_glbChi2", "nPixHits", "nTrkLayers"],
           "timing": ["time"], "ip": ["log_ip_pv_sig", "log_ip_sv_sig"]},
    "trk": {"pt": ["log_ptrel"], "angles": ["deta", "dphi", "log_dR"],
            "ip_displacement": ["slog_dxy", "slog_dz", "log_ip_sv", "log_ip_sv_sig", "log_ip_pv", "log_ip_pv_sig",
                                "log_dcaFlight"],
            "quality_hits": ["isLost", "highPurity", "hasDetails", "nPixHits", "nHits", "lostInner", "log_chi2"],
            "pileup_pv": ["fromPV", "pvQual", "puppi"],
            "region": ["inCone", "inCylinder"], "muon_flags": ["isMuon", "isLooseMuon", "muSoftMva", "hcalFrac"],
            "sv_fit": ["inSV", "has_dchi2", "slog_dchi2"],
            "leg_pairs": ["log_dca1", "log_dca2", "log_dca3", "vprob1", "vprob2", "vprob3"], "charge": ["charge"]},
    "neu": {"pt": ["log_ptrel"], "angles": ["deta", "dphi", "log_dR", "log_dR_flight"],
            "other": ["inCone", "inFlightCone", "isPhoton", "puppi", "hcalFrac"]},
    "sv": {"pt": ["log_ptrel"], "all_but_pt": None},
    "jet": {"pt": ["log_ptrel"], "all_but_pt": None},
}


def load_sample(data_dir, fold, n_sig, n_bkg, seed=0):
    stats = D.load_stats(data_dir)
    names = D.input_names(stats)
    rng = np.random.default_rng(seed)
    dirs = D.shard_dirs(data_dir)
    rng.shuffle(dirs)
    xs, ms, ns, nb = [], [], 0, 0
    for d in dirs:
        x, m = D._load_shard(d, names, [fold], [0, 1])
        keep = ((m["role"] == 0) & (np.cumsum(m["role"] == 0) + ns <= n_sig)) | \
               ((m["role"] == 1) & (np.cumsum(m["role"] == 1) + nb <= n_bkg))
        xs.append({k: v[keep] for k, v in x.items()})
        ms.append({k: v[keep] for k, v in m.items()})
        ns += int((ms[-1]["role"] == 0).sum())
        nb += int((ms[-1]["role"] == 1).sum())
        if ns >= n_sig and nb >= n_bkg:
            break
    x = {k: np.concatenate([a[k] for a in xs]) for k in xs[0]}
    m = {k: np.concatenate([a[k] for a in ms]) for k in ms[0]}
    return stats, x, m


@torch.no_grad()
def predict(model, x, device, bs=2048):
    out = []
    n = len(x["cand"])
    for i in range(0, n, bs):
        xb = D.to_torch({k: v[i:i + bs] for k, v in x.items()}, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(torch.sigmoid(model(xb).float()).cpu().numpy())
    return np.concatenate(out)


def in_any(m, ranges):
    return np.any([(m >= a) & (m < b) for a, b in ranges], axis=0)


def metrics(s, meta, w):
    y = meta["label"]
    sig, bkg = y == 1, meta["role"] == 1
    o = np.argsort(-s[sig])
    c = np.cumsum(w[sig][o]) / w[sig].sum()
    cut90 = s[sig][o][min(np.searchsorted(c, 0.9), sig.sum() - 1)]
    top = np.quantile(s[bkg], 0.99)
    mb = meta["mass"]
    near, far = bkg & in_any(mb, NEAR), bkg & in_any(mb, FAR)
    r_all = near.sum() / max(far.sum(), 1)
    r_top = (near & (s >= top)).sum() / max((far & (s >= top)).sum(), 1)
    early = bkg & (meta["run"] < EARLY_RUN)
    late = bkg & (meta["run"] >= EARLY_RUN)
    e_early = (s[early] >= top).mean() if early.any() else np.nan
    e_late = (s[late] >= top).mean() if late.any() else np.nan
    return dict(auc=weighted_auc(s, y, w), bkg_eff90=float((s[bkg] >= cut90).mean()),
                near_far=float(r_top / r_all), early_late=float(e_early / max(e_late, 1e-9)))


def run(data_dir, out_dir, fold, n_sig, n_bkg, which="best.pt", seed=0):
    device = device_of()
    model = load_model(os.path.join(out_dir, f"fold{fold}", which), device)
    stats, x, meta = load_sample(data_dir, fold, n_sig, n_bkg, seed)
    full_meta = D.load_meta(data_dir)
    tr_f, _, _ = D.fold_split(fold)
    w = D.Weighter(full_meta, np.isin(full_meta["fold"], tr_f) & np.isin(full_meta["role"], [0, 1]))(meta)
    print(f"fold {fold}: {int((meta['role'] == 0).sum())} signal, {int((meta['role'] == 1).sum())} sideband "
          f"candidates, device {device}")
    rng = np.random.default_rng(seed + 1)
    t0 = time.time()
    base = metrics(predict(model, x, device), meta, w)
    rows = [dict(group="(none)", **base)]
    print(f"baseline: {base}  ({time.time() - t0:.0f} s per pass)")

    feats = stats["features"]
    tests = [(f"cand:{n}", "cand", [i]) for i, n in enumerate(feats["cand"])]
    for t, names in feats["tokens"].items():
        if t != "mu":
            tests.append((f"{t}:REMOVED", t, None))
        for g, members in TOKEN_GROUPS.get(t, {}).items():
            if members is None:
                idx = [i for i, n in enumerate(names) if n != "log_ptrel"]
            else:
                idx = [names.index(n) for n in members if n in names]
            if idx:
                tests.append((f"{t}:{g}", t, idx))

    for label, t, idx in tests:
        xp = dict(x)
        if idx is None:  # remove the token type
            xp[f"{t}_mask"] = np.zeros_like(x[f"{t}_mask"])
            xp[f"{t}_feats"] = np.zeros_like(x[f"{t}_feats"])
        elif t == "cand":
            v = x["cand"].copy()
            v[:, idx] = v[rng.permutation(len(v))][:, idx]
            xp["cand"] = v
        else:
            v = x[f"{t}_feats"].copy()
            perm = rng.permutation(len(v))
            mask = x[f"{t}_mask"]
            # values from another candidate's token in the same slot (0 where that slot is empty)
            v[:, :, idx] = v[perm][:, :, idx] * mask[:, :, None]
            xp[f"{t}_feats"] = v
        r = metrics(predict(model, xp, device), meta, w)
        rows.append(dict(group=label, **r))
        print(f"{label:28s} auc {r['auc']:.5f}  bkg_eff90 {r['bkg_eff90']:.2e}  near_far {r['near_far']:.2f}  "
              f"early_late {r['early_late']:.2f}", flush=True)

    path = os.path.join(out_dir, "eval", f"diagnose_fold{fold}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {path}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-sig", type=int, default=30000)
    ap.add_argument("--n-bkg", type=int, default=60000)
    ap.add_argument("--which", default="best.pt")
    a = ap.parse_args()
    run(a.data, a.out, a.fold, a.n_sig, a.n_bkg, a.which)


if __name__ == "__main__":
    main()
