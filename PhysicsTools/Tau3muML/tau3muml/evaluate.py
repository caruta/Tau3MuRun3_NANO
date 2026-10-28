#!/usr/bin/env python3
"""Out-of-fold evaluation of the five fold models, with mass-sculpting checks.

Every candidate in the training set (signal, sidebands, SS3MU) is scored by the model whose
test fold contains it. Outputs in <out>/eval/:
  scores.npz            score + meta for every candidate
  summary.json          AUC, background efficiency at fixed signal efficiency, SS3MU checks
  roc.png, score.png    performance
  bkg_mass.png          sideband mass shape in score bins (should not change with the score)
  ss3mu_mass.png        SS3MU (|Q| = 3) mass after score cuts: a peak at 1.777 GeV means the
                        model has learnt the mass (SS3MU contains no signal, but is not blinded)
  npv.png               efficiencies vs number of PVs (data/MC pileup shortcut)

  python3 -m tau3muml.evaluate --data /eos/.../train_Transformer_v1 --out runs/v1
"""
import argparse
import json
import os

import numpy as np

from . import data as D
from .train import bkg_eff_at, device_of, load_model, predict_loader, weighted_auc

SR = (1.74, 1.82)
SB = ((1.62, 1.74), (1.82, 1.94))


def score_all(data_dir, out_dir, folds=range(5), which="best.pt", batch_size=1024):
    device = device_of()
    meta_all = D.load_meta(data_dir)
    parts = []
    for k in folds:
        path = os.path.join(out_dir, f"fold{k}", which)
        if not os.path.exists(path):
            print(f"no model for fold {k} ({path}), skipped")
            continue
        model = load_model(path, device)
        tr_f, _, _ = D.fold_split(k)
        rows_tr = np.isin(meta_all["fold"], tr_f) & np.isin(meta_all["role"], [0, 1])
        cfg_path = os.path.join(out_dir, f"fold{k}", "config.json")
        cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
        prod = {int(a): float(b) for a, b in cfg["prod_frac"].items()} if cfg.get("prod_frac") else None
        weighter = D.Weighter(meta_all, rows_tr, prod_frac=prod, npv_reweight=cfg.get("npv_reweight", True))
        loader = D.ShardLoader(data_dir, [k], roles=(0, 1, 2), batch_size=batch_size, shuffle=False, weighter=weighter)
        s, m, w = predict_loader(model, loader, device)
        m["score"], m["weight"], m["model_fold"] = s, w, np.full(len(s), k)
        parts.append(m)
        print(f"fold {k}: scored {len(s)} candidates")
    out = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    return out


def in_range(m, lo, hi):
    return (m >= lo) & (m < hi)


def sculpting_numbers(r):
    """SR/SB ratio of SS3MU before and after score cuts (should stay constant)."""
    sig = r["role"] == 0
    ss = r["role"] == 2
    res = {}
    for eff in (1.0, 0.8, 0.5, 0.2):
        if eff < 1.0:
            o = np.sort(r["score"][sig])[::-1]
            cut = o[min(int(eff * len(o)), len(o) - 1)]
        else:
            cut = -1.0
        sel = ss & (r["score"] >= cut)
        n_sr = int(in_range(r["mass"][sel], *SR).sum())
        n_sb = int(sum(in_range(r["mass"][sel], *b).sum() for b in SB))
        ratio = n_sr / n_sb if n_sb else float("nan")
        err = ratio * np.sqrt(1 / max(n_sr, 1) + 1 / max(n_sb, 1)) if n_sb else float("nan")
        res[f"sig_eff_{eff:.1f}"] = dict(cut=float(cut), ss3mu_SR=n_sr, ss3mu_SB=n_sb, ratio=ratio, ratio_err=err)
    return res


def make_plots(r, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sig, bkg, ss = r["role"] == 0, r["role"] == 1, r["role"] == 2
    w = r["weight"]
    # ROC
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for k in np.unique(r["model_fold"]):
        f = r["model_fold"] == k
        sel = f & (sig | bkg)
        s, y, ww = r["score"][sel], r["label"][sel], w[sel]
        o = np.argsort(-s)
        tp = np.cumsum(ww[o] * (y[o] == 1)); fp = np.cumsum(ww[o] * (y[o] == 0))
        ax.plot(tp / tp[-1], np.maximum(fp / fp[-1], 1e-6), label=f"fold {k}: AUC {weighted_auc(s, y, ww):.4f}")
    ax.set_yscale("log"); ax.set_xlabel("signal efficiency"); ax.set_ylabel("background (sideband) efficiency")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(os.path.join(out, "roc.png"), dpi=120)
    plt.close(fig)
    # scores
    fig, ax = plt.subplots(figsize=(5, 4.5))
    bins = np.linspace(0, 1, 51)
    for sel, lab in ((sig, "signal MC"), (bkg, "sidebands"), (ss, "SS3MU")):
        if sel.any():
            ax.hist(r["score"][sel], bins, weights=w[sel] if lab == "signal MC" else None, density=True, histtype="step", label=lab)
    ax.set_yscale("log"); ax.set_xlabel("score"); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out, "score.png"), dpi=120); plt.close(fig)
    # background mass in score quantiles
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    mb = np.linspace(1.4, 2.2, 41)
    qs = np.quantile(r["score"][bkg], [0, 0.5, 0.9, 0.99, 1.0]) if bkg.any() else []
    for lo, hi, lab in zip(qs[:-1], qs[1:], ["0-50%", "50-90%", "90-99%", "top 1%"]):
        sel = bkg & (r["score"] >= lo) & (r["score"] <= hi)
        if sel.sum() > 10:
            ax.hist(r["mass"][sel], mb, density=True, histtype="step", label=f"sideband score quantile {lab}")
    ax.axvspan(*SR, color="grey", alpha=0.2, label="blinded SR")
    ax.set_xlabel("m(3μ) [GeV]"); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(out, "bkg_mass.png"), dpi=120); plt.close(fig)
    # SS3MU
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    o = np.sort(r["score"][sig])[::-1] if sig.any() else np.array([0.0])
    for eff in (1.0, 0.5, 0.2):
        cut = -1 if eff == 1.0 else o[min(int(eff * len(o)), len(o) - 1)]
        sel = ss & (r["score"] >= cut)
        if sel.sum():
            ax.hist(r["mass"][sel], mb, density=True, histtype="step", label=f"SS3MU, cut at sig eff {eff:.0%} ({sel.sum()})")
    ax.axvspan(*SR, color="grey", alpha=0.2)
    ax.set_xlabel("m(3μ) [GeV]"); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(out, "ss3mu_mass.png"), dpi=120); plt.close(fig)
    # nPV
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cut = o[min(int(0.5 * len(o)), len(o) - 1)]
    pb = np.arange(0, 81, 5)
    for sel, lab in ((sig, "signal eff"), (bkg, "sideband eff (x10)")):
        if not sel.any():
            continue
        tot, _ = np.histogram(r["nPVtx"][sel], pb, weights=w[sel])
        pas, _ = np.histogram(r["nPVtx"][sel & (r["score"] >= cut)], pb, weights=w[sel & (r["score"] >= cut)])
        eff = np.where(tot > 0, pas / np.maximum(tot, 1e-12), np.nan) * (10 if "x10" in lab else 1)
        ax.step(pb[:-1], eff, where="post", label=lab)
    ax.set_xlabel("number of PVs"); ax.set_ylabel("efficiency at 50% signal efficiency cut"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out, "npv.png"), dpi=120); plt.close(fig)


def evaluate(data_dir, out_dir, which="best.pt", plots=True):
    ev = os.path.join(out_dir, "eval")
    os.makedirs(ev, exist_ok=True)
    r = score_all(data_dir, out_dir, which=which)
    np.savez(os.path.join(ev, "scores.npz"), **r)
    tb = np.isin(r["role"], [0, 1])
    summary = {"auc": weighted_auc(r["score"][tb], r["label"][tb], r["weight"][tb]),
               "bkg_eff_at_sig_eff": {str(e): bkg_eff_at(r["score"][tb], r["label"][tb], r["weight"][tb], e)
                                      for e in (0.8, 0.5, 0.2, 0.1)},
               "per_fold_auc": {int(k): weighted_auc(r["score"][tb & (r["model_fold"] == k)], r["label"][tb & (r["model_fold"] == k)],
                                                     r["weight"][tb & (r["model_fold"] == k)]) for k in np.unique(r["model_fold"])},
               "ss3mu_SR_over_SB": sculpting_numbers(r),
               "n": {"signal": int((r["role"] == 0).sum()), "sidebands": int((r["role"] == 1).sum()),
                     "ss3mu": int((r["role"] == 2).sum())}}
    with open(os.path.join(ev, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))
    if plots:
        make_plots(r, ev)
    return summary, r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True, help="training output directory (with fold<k>/)")
    ap.add_argument("--which", default="best.pt")
    a = ap.parse_args()
    evaluate(a.data, a.out, a.which)


if __name__ == "__main__":
    main()
