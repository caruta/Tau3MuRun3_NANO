#!/usr/bin/env python3
"""Quick comparison of the transformer with BDT baselines, on the same candidates and folds.

Trains, per fold (same train / validation / test split and weights as the transformer):
  bdt_run2like  ~20 candidate-level variables in the spirit of the Run 2 HF BDT
  bdt_candmu    every candidate and muon feature the transformer sees (no context tokens)
and compares their out-of-fold scores with the transformer's (<out>/eval/scores.npz from evaluate.py):

  * background (sideband) efficiency at fixed signal efficiency, ROC curves
  * SS3MU SR/SB ratio vs cut (mass sculpting) for each method
  * a rough expected 95% CL upper limit on the number of signal candidates (before the classifier),
    from a counting experiment in the signal region with 1 and 2 score categories each. Background
    in the SR: exponential fit to the sidebands passing the cut, scaled to the full dataset with
    --bkg-scale. Only the RATIO between methods is meaningful; it is what the expected limit on
    the branching fraction improves by, as long as the signal normalisation is the same.

Rough on purpose: no trigger matching or analysis base cuts, one 80 MeV SR window for all mass
resolutions, Asimov approximation. Runs in minutes (XGBoost, GPU if available).

  python3 -m tau3muml.compare_bdt --data <DATA> --out <OUT> --bkg-scale 124
"""
import argparse
import json
import os
import time

import numpy as np

from . import data as D

SR = (1.74, 1.82)
FIT_RANGE = (1.40, 2.20)
NEAR_SB = ((1.62, 1.74), (1.82, 1.94))

RUN2LIKE_CAND = ["log_Lsig", "log_LxySig", "pointing", "prob", "kin_prob", "vprob12", "vprob13", "vprob23",
                 "log_d0sig", "log_d0maxsig", "dr_max", "dz_max", "rel_iso", "iso03_rel", "iso05_rel",
                 "mindca_iso", "n_vtxcompat", "dimu_extra_prob", "eta"]
RUN2LIKE_MU = ["log_ptfrac", "softMvaRun3", "log_ip_sv_sig", "isGlobal", "segmComp"]


# ---------------------------------------------------------------- data
def load_inputs(data_dir):
    stats = D.load_stats(data_dir)
    cand, mu, meta = [], [], []
    for d in D.shard_dirs(data_dir):
        cand.append(np.load(os.path.join(d, "cand.npy")))
        mu.append(np.load(os.path.join(d, "mu_feats.npy")))
        meta.append({k: np.load(os.path.join(d, f"{k}.npy")) for k in D.META})
    meta = {k: np.concatenate([m[k] for m in meta]) for k in D.META}
    return stats, np.concatenate(cand), np.concatenate(mu), meta


def feature_sets(stats, cand, mu):
    cn = stats["features"]["cand"]
    mn = stats["features"]["tokens"]["mu"]
    X_all = np.concatenate([cand, mu.reshape(len(mu), -1)], axis=1).astype(np.float32)
    names_all = cn + [f"mu{i + 1}_{n}" for i in range(mu.shape[1]) for n in mn]

    ci = [cn.index(n) for n in RUN2LIKE_CAND if n in cn]
    cols = [cand[:, ci]]
    names = [cn[i] for i in ci]
    for n in RUN2LIKE_MU:
        if n in mn:
            v = mu[:, :, mn.index(n)].astype(np.float32)
            # order-independent summaries of the three legs
            cols += [v.min(1, keepdims=True), v.max(1, keepdims=True)]
            names += [f"min_mu_{n}", f"max_mu_{n}"]
    X_r2 = np.concatenate(cols, axis=1).astype(np.float32)
    return {"bdt_run2like": (X_r2, names), "bdt_candmu": (X_all, names_all)}


# ---------------------------------------------------------------- BDTs
def train_bdt(X, meta, fold, seed=0, prod_frac=None, npv_reweight=True):
    tr_f, va_f, te_f = D.fold_split(fold)
    rows_tr = np.isin(meta["fold"], tr_f) & np.isin(meta["role"], [0, 1])
    rows_va = np.isin(meta["fold"], va_f) & np.isin(meta["role"], [0, 1])
    rows_te = np.isin(meta["fold"], te_f)
    w = D.Weighter(meta, rows_tr, prod_frac=prod_frac, npv_reweight=npv_reweight)(meta)
    y = meta["label"].astype(np.float32)
    try:
        import xgboost as xgb
        dev = "cuda" if _has_cuda() else "cpu"
        clf = xgb.XGBClassifier(n_estimators=2000, max_depth=6, learning_rate=0.1, subsample=0.8,
                                colsample_bytree=0.8, min_child_weight=10, tree_method="hist", device=dev,
                                early_stopping_rounds=50, eval_metric="logloss", random_state=seed)
        clf.fit(X[rows_tr], y[rows_tr], sample_weight=w[rows_tr],
                eval_set=[(X[rows_va], y[rows_va])], sample_weight_eval_set=[w[rows_va]], verbose=False)
        info = f"xgboost ({dev}), {clf.best_iteration + 1} trees"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(max_iter=1000, learning_rate=0.1, max_depth=6,
                                             early_stopping=True, validation_fraction=0.1, random_state=seed)
        clf.fit(X[rows_tr], y[rows_tr], sample_weight=w[rows_tr])
        info = f"sklearn HistGradientBoosting, {clf.n_iter_} iterations (xgboost not available)"
    return rows_te, clf.predict_proba(X[rows_te])[:, 1], w, info


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------- metrics
def key_of(m):
    return np.rec.fromarrays([m["role"].astype(np.int64), m["sample"].astype(np.int64), m["run"].astype(np.int64),
                              m["event"].astype(np.int64), m["cand_idx"].astype(np.int64)])


def bkg_eff_at(s, y, w, eff):
    sig = y == 1
    o = np.sort(s[sig])[::-1]
    ws = w[sig][np.argsort(-s[sig])]
    c = np.cumsum(ws) / ws.sum()
    cut = o[min(np.searchsorted(c, eff), len(o) - 1)]
    b = y == 0
    return float((s[b] >= cut).mean()), float(cut)


def exp_fit_sr(m, lo=FIT_RANGE[0], hi=FIT_RANGE[1], sr=SR):
    """Events expected in the SR from an exponential fitted to the sidebands (unbinned, grid MLE)."""
    m = m[(m >= lo) & (m <= hi) & ((m < sr[0]) | (m > sr[1]))]
    n = len(m)
    if n < 3:
        return n * (sr[1] - sr[0]) / ((hi - lo) - (sr[1] - sr[0]))  # flat
    lam = np.linspace(-8, 8, 801)

    def integ(l, a, b):
        return np.where(np.abs(l) < 1e-9, b - a, (np.exp(l * b) - np.exp(l * a)) / np.where(np.abs(l) < 1e-9, 1, l))

    norm = integ(lam, lo, sr[0]) + integ(lam, sr[1], hi)
    ll = lam * m.sum() - n * np.log(norm)
    best = lam[np.argmax(ll)]
    return float(n * integ(best, sr[0], sr[1]) / (integ(best, lo, sr[0]) + integ(best, sr[1], hi)))


def asimov_ul(s, b):
    """Expected 95% CL UL on the signal scale mu (signal yields mu*s_i), background-only Asimov,
    CLs+b-like approximation sqrt(q_mu) = 1.645."""
    s, b = np.asarray(s, float), np.maximum(np.asarray(b, float), 1e-3)

    def q(mu):
        return 2 * np.sum(mu * s - b * np.log1p(mu * s / b))

    lo, hi = 0.0, 1.0
    while q(hi) < 1.645 ** 2:
        hi *= 2
        if hi > 1e12:
            return np.inf
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if q(mid) < 1.645 ** 2 else (lo, mid)
    return hi


def category_scan(score, meta, w, bkg_scale, n_grid=40):
    """Best 1- and 2-category splits. Signal: weighted fraction of all signal candidates in the SR
    passing; background: exponential sideband fit x bkg_scale. Returns expected UL on N_signal."""
    sig = meta["role"] == 0
    bkg = meta["role"] == 1
    m = meta["mass"]
    ws = w[sig] / w[sig].sum()
    s_sig, m_sig = score[sig], m[sig]
    in_sr = (m_sig >= SR[0]) & (m_sig <= SR[1])
    qs = np.unique(np.quantile(s_sig, np.linspace(0, 0.995, n_grid)))

    def sb(lo, hi):
        s = float(ws[in_sr & (s_sig >= lo) & (s_sig < hi)].sum())
        sel = bkg & (score >= lo) & (score < hi)
        return s, exp_fit_sr(m[sel]) * bkg_scale

    best1 = (np.inf, None)
    cache = {}
    for c in qs:
        cache[(c, np.inf)] = sb(c, np.inf)
        s, b = cache[(c, np.inf)]
        ul = asimov_ul([s], [b])
        if ul < best1[0]:
            best1 = (ul, dict(cut=float(c), sig_frac=s, bkg_sr=b))
    best2 = (np.inf, None)
    for i, c1 in enumerate(qs):
        for c2 in qs[i + 1:]:
            s1, b1 = sb(c1, c2)
            s2, b2 = cache[(c2, np.inf)]
            ul = asimov_ul([s1, s2], [b1, b2])
            if ul < best2[0]:
                best2 = (ul, dict(cuts=[float(c1), float(c2)], sig_frac=[s1, s2], bkg_sr=[b1, b2]))
    return {"1cat": {"ul_nsig": best1[0], **best1[1]}, "2cat": {"ul_nsig": best2[0], **best2[1]}}


def ss3mu_ratio(score, meta, w, effs=(1.0, 0.5, 0.2)):
    sig, ss = meta["role"] == 0, meta["role"] == 2
    out = {}
    for e in effs:
        cut = -np.inf if e >= 1 else bkg_eff_at(score[(sig | (meta["role"] == 1))],
                                               meta["label"][(sig | (meta["role"] == 1))],
                                               w[(sig | (meta["role"] == 1))], e)[1]
        m = meta["mass"][ss & (score >= cut)]
        n_sr = int(((m >= SR[0]) & (m < SR[1])).sum())
        n_sb = int(sum(((m >= a) & (m < b)).sum() for a, b in NEAR_SB))
        r = n_sr / n_sb if n_sb else float("nan")
        out[f"{e:.1f}"] = dict(SR=n_sr, SB=n_sb, ratio=r,
                               err=r * np.sqrt(1 / max(n_sr, 1) + 1 / max(n_sb, 1)) if n_sb else float("nan"))
    return out


# ---------------------------------------------------------------- main
def compare(data_dir, out_dir, bkg_scale, folds=range(5), plots=True):
    t0 = time.time()
    stats, cand, mu, meta = load_inputs(data_dir)
    print(f"loaded {len(meta['role'])} candidates in {time.time() - t0:.0f} s")
    sets = feature_sets(stats, cand, mu)
    del cand, mu
    scores = {k: np.full(len(meta["role"]), np.nan, dtype=np.float32) for k in sets}
    w_all = np.zeros(len(meta["role"]), dtype=np.float32)
    info = {}

    cfg_path = os.path.join(out_dir, "fold0", "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    prod = {int(a): float(b) for a, b in cfg["prod_frac"].items()} if cfg.get("prod_frac") else None
    for name, (X, names) in sets.items():
        for k in folds:
            t1 = time.time()
            rows, s, w, inf = train_bdt(X, meta, k, prod_frac=prod, npv_reweight=cfg.get("npv_reweight", True))
            scores[name][rows] = s
            w_all[rows] = w[rows]
            print(f"{name} ({len(names)} inputs) fold {k}: {inf}, {time.time() - t1:.0f} s", flush=True)
            info[name] = {"n_inputs": len(names), "inputs": names, "trainer": inf}

    # transformer scores, joined on (role, sample, run, event, cand_idx)
    path = os.path.join(out_dir, "eval", "scores.npz")
    if os.path.exists(path):
        t = dict(np.load(path))
        kt, km = key_of(t), key_of(meta)
        ot, om = np.argsort(kt), np.argsort(km)
        pos = np.searchsorted(kt[ot], km)
        pos = np.clip(pos, 0, len(kt) - 1)
        ok = kt[ot][pos] == km
        tr = np.full(len(km), np.nan, dtype=np.float32)
        tr[ok] = t["score"][ot][pos[ok]]
        scores = {"transformer": tr, **scores}
        print(f"transformer scores matched for {ok.sum()} / {len(km)} candidates")
    else:
        print(f"no {path}: run tau3muml.evaluate first to include the transformer")

    have = np.all([np.isfinite(v) for v in scores.values()], axis=0)
    meta = {k: v[have] for k, v in meta.items()}
    scores = {k: v[have] for k, v in scores.items()}
    w = w_all[have]
    tb = np.isin(meta["role"], [0, 1])
    y = meta["label"]

    res = {"n": {r: int((meta["role"] == i).sum()) for i, r in enumerate(["signal", "sidebands", "ss3mu"])},
           "bkg_scale": bkg_scale, "methods": {}}
    for name, s in scores.items():
        r = {"bkg_eff": {f"{e:.1f}": bkg_eff_at(s[tb], y[tb], w[tb], e)[0] for e in (0.8, 0.5, 0.3, 0.2, 0.1)},
             "ss3mu_SR_over_SB": ss3mu_ratio(s, meta, w),
             "expected_ul": category_scan(s, meta, w, bkg_scale)}
        if name in info:
            r.update(n_inputs=info[name]["n_inputs"], trainer=info[name]["trainer"])
        res["methods"][name] = r
    ref = "bdt_run2like"
    for name, r in res["methods"].items():
        for c in ("1cat", "2cat"):
            r["expected_ul"][c]["ratio_to_" + ref] = r["expected_ul"][c]["ul_nsig"] / res["methods"][ref]["expected_ul"][c]["ul_nsig"]

    ev = os.path.join(out_dir, "eval")
    os.makedirs(ev, exist_ok=True)
    with open(os.path.join(ev, "compare_bdt.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    np.savez(os.path.join(ev, "compare_scores.npz"), **{f"score_{k}": v for k, v in scores.items()},
             **meta, weight=w)
    print_table(res)
    if plots:
        make_plots(scores, meta, w, ev)
    return res


def print_table(res):
    meth = list(res["methods"])
    print("\nbackground (sideband) efficiency at fixed signal efficiency")
    print(f"{'sig eff':>8s} " + " ".join(f"{m:>14s}" for m in meth))
    for e in ("0.8", "0.5", "0.3", "0.2", "0.1"):
        print(f"{e:>8s} " + " ".join(f"{res['methods'][m]['bkg_eff'][e]:14.2e}" for m in meth))
    print("\nSS3MU SR/SB ratio (constant = no sculpting)")
    for e in ("1.0", "0.5", "0.2"):
        print(f"{e:>8s} " + " ".join(f"{res['methods'][m]['ss3mu_SR_over_SB'][e]['ratio']:8.3f}±"
                                    f"{res['methods'][m]['ss3mu_SR_over_SB'][e]['err']:<5.3f}" for m in meth))
    print(f"\nexpected 95% UL on N(signal), relative to bdt_run2like (lower is better), bkg scale {res['bkg_scale']}")
    for c in ("1cat", "2cat"):
        print(f"{c:>8s} " + " ".join(f"{res['methods'][m]['expected_ul'][c]['ratio_to_bdt_run2like']:14.3f}" for m in meth))


def make_plots(scores, meta, w, ev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tb = np.isin(meta["role"], [0, 1])
    y = meta["label"][tb]
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    for name, s in scores.items():
        s, ww = s[tb], w[tb]
        o = np.argsort(-s)
        tp = np.cumsum(ww[o] * (y[o] == 1)); fp = np.cumsum(y[o] == 0)
        ax.plot(tp / tp[-1], np.maximum(fp / fp[-1], 1e-7), label=name)
    ax.set_yscale("log"); ax.set_xlabel("signal efficiency"); ax.set_ylabel("sideband efficiency")
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(ev, "compare_roc.png"), dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True, help="transformer training directory (with eval/scores.npz)")
    ap.add_argument("--bkg-scale", type=float, required=True,
                    help="full-dataset sideband candidates / sideband candidates in the training set")
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    a = ap.parse_args()
    compare(a.data, a.out, a.bkg_scale, a.folds)


if __name__ == "__main__":
    main()
