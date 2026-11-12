"""Transformer vs BDTs on an analysis-like sample: base cuts and one candidate per event.

No retraining: the scores of tau3muml.compare_bdt (eval/compare_scores.npz) are reused, and the
candidates are selected with the flags of scripts/make_basecut_table.py (<data>/basecuts.npz):

  loose        the training-set preselection, all triplets (as in compare_bdt)
  base         analysis base selection (trigger, acceptance, global+medium, |Q|, dR/dz, flight
               significance, 1.62-2.0 GeV, trigger matching)
  base_best    base + one candidate per event (lowest vertex chi2), as in the analysis
  base_best_veto  + Run 2 phi(1020) veto per resolution category and omega(782) veto

For each selection and method: selection efficiency, sideband efficiency at fixed signal efficiency
(relative to the selected signal), near/far and early/late, and the expected limit (inclusive and in the
Run 2 categories) with the background fitted in 1.62-2.0 GeV for every selection, signal yields as a
fraction of the loose signal, ratio to bdt_run2 within the same selection.

  python3 -m tau3muml.compare_basecuts --data <DATA> --out <OUT>
or in the notebook:  from tau3muml.compare_basecuts import compare_basecuts; compare_basecuts(DATA, OUT)
"""
import argparse
import json
import os

import numpy as np

from .compare_bdt import (BLIND, COLORS, bkg_eff_at, expected_limits, join, plot_fits)

FIT = (1.62, 2.0)
NEAR = ((1.70, 1.74), (1.82, 1.86))
FAR = ((1.62, 1.66), (1.90, 2.00))
PHI_VETO = {"A": (0.994, 1.044), "B": (0.985, 1.053), "C": (0.974, 1.064)}   # AN section 7
OMEGA_VETO = (0.772, 0.792)


def _key(m):
    return np.rec.fromarrays([np.asarray(m[k], np.int64) for k in ("sample", "run", "event", "cand_idx")])


def selections(meta, extra, bc):
    ss = meta["role"] == 2
    base = np.where(ss, bc["base_ss3mu"], bc["base"]).astype(bool)
    best = base & bc["best"].astype(bool)
    res = np.nan_to_num(extra.get("massres", np.full(len(base), 0.01)), nan=1.0)
    cat = np.where(res < 0.007, "A", np.where(res <= 0.0105, "B", "C"))
    veto = np.zeros(len(base), bool)
    for m in (bc["m_os1"], bc["m_os2"]):
        m = np.nan_to_num(m, nan=-1)
        for c, (lo, hi) in PHI_VETO.items():
            veto |= (cat == c) & (m >= lo) & (m <= hi)
        veto |= (m >= OMEGA_VETO[0]) & (m <= OMEGA_VETO[1])
    return {"loose": np.ones(len(base), bool), "base": base, "base_best": best, "base_best_veto": best & ~veto}


def near_far(score, meta, effs=(1.0, 0.1, 0.01)):
    b = meta["role"] == 1
    m = meta["mass"]
    near = b & np.any([(m >= lo) & (m < hi) for lo, hi in NEAR], axis=0)
    far = b & np.any([(m >= lo) & (m < hi) for lo, hi in FAR], axis=0)
    out = {}
    for e in effs:
        cut = -np.inf if e >= 1 else np.quantile(score[b], 1 - e)
        n, f = int((near & (score >= cut)).sum()), int((far & (score >= cut)).sum())
        r = n / f if f else float("nan")
        out[f"{e:g}"] = dict(near=n, far=f, ratio=r, err=r * np.sqrt(1 / max(n, 1) + 1 / max(f, 1)) if f else float("nan"))
    return out


def early_late(score, meta, effs=(0.01,), split=383000):
    b = meta["role"] == 1
    e_, l_ = b & (meta["run"] < split), b & (meta["run"] >= split)
    out = {}
    for e in effs:
        cut = np.quantile(score[b], 1 - e)
        pe, pl = (score[e_] >= cut).mean(), (score[l_] >= cut).mean()
        ne, nl = (score[e_] >= cut).sum(), (score[l_] >= cut).sum()
        r = pe / pl if pl > 0 else float("nan")
        out[f"{e:g}"] = dict(ratio=r, err=r * np.sqrt(1 / max(ne, 1) + 1 / max(nl, 1)))
    return out


def compare_basecuts(data_dir, out_dir, bkg_scale=None, plots=True):
    ev = os.path.join(out_dir, "eval")
    z = dict(np.load(os.path.join(ev, "compare_scores.npz")))
    if bkg_scale is None:
        bkg_scale = json.load(open(os.path.join(ev, "compare_bdt.json")))["bkg_scale"]
    w = z.pop("weight")
    scores = {k[6:]: z.pop(k) for k in list(z) if k.startswith("score_")}
    scores = {k: scores[k] for k in sorted(scores, key=lambda k: list(COLORS).index(k) if k in COLORS else 99)}
    extra = {k: z.pop(k) for k in ("massres", "is3glb") if k in z}
    meta = z
    bc = dict(np.load(os.path.join(data_dir, "basecuts.npz")))
    idx = join(_key(bc), _key(meta))
    found = idx >= 0
    print(f"base-cut flags found for {found.sum()} / {len(found)} candidates; background scale {bkg_scale:.0f}")
    if (~found).any():
        print("  candidates without flags are dropped from the base selections")
    flags = {k: np.where(found, v[np.maximum(idx, 0)], 0) for k, v in bc.items() if v.ndim == 1 and k not in
             ("run", "event", "cand_idx", "sample")}
    for k in ("m_os1", "m_os2"):
        flags[k] = np.where(found, bc[k][np.maximum(idx, 0)], np.nan)
    sels = selections(meta, extra, flags)
    for k in ("base", "base_best", "base_best_veto"):
        sels[k] &= found

    sig, bkg = meta["role"] == 0, meta["role"] == 1
    w_sig_loose = w[sig].sum()
    ref = "bdt_run2" if "bdt_run2" in scores else "bdt_run2like"
    res = {"bkg_scale": bkg_scale, "reference": ref, "fit_range": FIT, "selections": {}}
    for sn, sel in sels.items():
        m = {k: v[sel] for k, v in meta.items()}
        ex = {k: v[sel] for k, v in extra.items()}
        ws = w[sel]
        tb = np.isin(m["role"], [0, 1])
        r = {"eff": {"signal": float(w[sig & sel].sum() / w_sig_loose), "sidebands": float((bkg & sel).sum() / bkg.sum()),
                     "ss3mu": float(((meta["role"] == 2) & sel).sum() / max((meta["role"] == 2).sum(), 1))},
             "n": {"signal": int((sig & sel).sum()), "sidebands": int((bkg & sel).sum())},
             "methods": {}}
        for name, s in scores.items():
            s_ = s[sel]
            lim = expected_limits(s_, m, ws, ex, bkg_scale, sig_total=w_sig_loose, incl_fit=FIT)
            r["methods"][name] = {
                "bkg_eff": {f"{e:.1f}": bkg_eff_at(s_[tb], m["label"][tb], ws[tb], e) for e in (0.8, 0.5, 0.2)},
                "near_far": near_far(s_, m), "early_late": early_late(s_, m), "expected_ul": lim}
            print(f"{sn}: {name} done", flush=True)
        for name, rm in r["methods"].items():
            for c, v in rm["expected_ul"].items():
                v["ratio"] = v["ul"] / r["methods"][ref]["expected_ul"][c]["ul"]
        res["selections"][sn] = r
        if plots and sn == "base_best":
            plot_fits({k: v[sel] for k, v in scores.items()}, m, ws,
                      {"methods": {k: {"expected_ul": v["expected_ul"]} for k, v in r["methods"].items()}},
                      ev, fit=FIT, tag="_base_best")
    with open(os.path.join(ev, "compare_basecuts.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print_table(res)
    if plots:
        plot_roc(scores, meta, w, sels, ev)
    return res


def print_table(res):
    for sn, r in res["selections"].items():
        meth = list(r["methods"])
        print(f"\n=== {sn}: signal eff {r['eff']['signal']:.3f}, sideband eff {r['eff']['sidebands']:.4f} "
              f"({r['n']['sidebands']} sideband candidates)")
        print(f"{'':>24s}" + "".join(f"{m:>14s}" for m in meth))
        for e in ("0.8", "0.5", "0.2"):
            print(f"{'bkg eff @ sig ' + e:>24s}" + "".join(f"{r['methods'][m]['bkg_eff'][e]:14.2e}" for m in meth))
        for e in ("1", "0.1", "0.01"):
            print(f"{'near/far @ ' + e:>24s}" + "".join(f"{r['methods'][m]['near_far'][e]['ratio']:14.3f}" for m in meth))
        print(f"{'early/late @ 0.01':>24s}" + "".join(f"{r['methods'][m]['early_late']['0.01']['ratio']:14.2f}" for m in meth))
        for c in r["methods"][meth[0]]["expected_ul"]:
            print(f"{'UL ratio ' + c:>24s}" + "".join(f"{r['methods'][m]['expected_ul'][c]['ratio']:14.3f}" for m in meth))


def plot_roc(scores, meta, w, sels, ev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["loose", "base", "base_best"]
    fig, axs = plt.subplots(1, len(names), figsize=(5.5 * len(names), 4.8))
    for ax, sn in zip(axs, names):
        tb = np.isin(meta["role"], [0, 1]) & sels[sn]
        y = meta["label"][tb]
        for name, s in scores.items():
            s_, ww = s[tb], w[tb]
            o = np.argsort(-s_)
            tp = np.cumsum(ww[o] * (y[o] == 1)); fp = np.cumsum(y[o] == 0)
            ax.plot(tp / tp[-1], np.maximum(fp / max(fp[-1], 1), 1e-7), label=name, color=COLORS.get(name))
        ax.set_yscale("log"); ax.set_xlabel("signal efficiency (within selection)"); ax.set_ylabel("sideband efficiency")
        ax.set_title(sn); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ev, "compare_roc_basecuts.png"), dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="training-set directory (with basecuts.npz)")
    ap.add_argument("--out", required=True, help="transformer training directory (with eval/compare_scores.npz)")
    ap.add_argument("--bkg-scale", type=float, default=None,
                    help="default: the scale used by compare_bdt (eval/compare_bdt.json)")
    a = ap.parse_args()
    compare_basecuts(a.data, a.out, a.bkg_scale)


if __name__ == "__main__":
    main()
