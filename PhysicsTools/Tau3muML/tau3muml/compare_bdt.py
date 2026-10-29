#!/usr/bin/env python3
"""Quick comparison of the transformer with BDT baselines, on the same candidates and folds.

BDTs trained per fold (same train / validation / test split and weights as the transformer):
  bdt_run2      the Run 2 HF BDT inputs (CMS AN sec. 11.1, arXiv:2312.02371) and XGBoost settings
                (AN table 18). Needs <data>/bdt_run2.npz from scripts/make_bdt_table.py. One BDT for
                all mass-resolution categories (Run 2: one per category); softMvaRun3 replaces the
                Run 2 global-muon MVA.
  bdt_run2like  ~20 transformer candidate-level features in the same spirit (no absolute pT)
  bdt_candmu    every candidate and muon feature of the transformer (no context tokens)
and compares their out-of-fold scores with the transformer's (<out>/eval/scores.npz from evaluate.py):

  * sideband efficiency at fixed signal efficiency (all candidates and three-global-muon only), ROC
  * SS3MU SR/SB ratio vs cut (mass sculpting)
  * rough expected 95% CL upper limit on the signal yield, as a ratio between methods:
      - "inclusive": all candidates, one 1.74-1.82 GeV window, best 2 score categories
      - "run2_categories" (needs bdt_run2.npz): three global muons, mass-resolution categories
        A/B/C (sigma_m/m < 0.7 %, < 1.05 %, above) with the Run 2 signal windows, and in each the best
        3 score categories above a discarded region, as in Run 2
    Background in a window: exponential fit to the sidebands passing the cut (1.62-2.0 GeV for the
    Run 2 categories, 1.40-2.20 GeV otherwise, blinded region excluded), times --bkg-scale.
    Asimov approximation, counting per category. Only ratios between methods are meaningful.

  python3 -m tau3muml.compare_bdt --data <DATA> --out <OUT> --bkg-scale 120
"""
import argparse
import json
import os
import time

import numpy as np

from . import data as D

BLIND = (1.74, 1.82)
NEAR_SB = ((1.62, 1.74), (1.82, 1.94))
RUN2_CATS = {"A": (0.0, 0.007, (1.753, 1.801)), "B": (0.007, 0.0105, (1.739, 1.815)),
             "C": (0.0105, 9.0, (1.727, 1.827))}

RUN2LIKE_CAND = ["log_Lsig", "log_LxySig", "pointing", "prob", "kin_prob", "vprob12", "vprob13", "vprob23",
                 "log_d0sig", "log_d0maxsig", "dr_max", "dz_max", "rel_iso", "iso03_rel", "iso05_rel",
                 "mindca_iso", "n_vtxcompat", "dimu_extra_prob", "eta"]
RUN2LIKE_MU = ["log_ptfrac", "softMvaRun3", "log_ip_sv_sig", "isGlobal", "segmComp"]

XGB_DEFAULT = dict(n_estimators=2000, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                   min_child_weight=10, early_stopping_rounds=50)
XGB_RUN2 = dict(n_estimators=1000, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.9,
                min_child_weight=10, gamma=10, reg_alpha=5, reg_lambda=5, early_stopping_rounds=100)


# ---------------------------------------------------------------- data
def key_of(m):
    return np.rec.fromarrays([np.asarray(m["role"], np.int64), np.asarray(m["sample"], np.int64),
                              np.asarray(m["run"], np.int64), np.asarray(m["event"], np.int64),
                              np.asarray(m["cand_idx"], np.int64)])


def join(src_keys, dst_keys):
    """Index into src for every dst row (-1 if missing)."""
    o = np.argsort(src_keys)
    pos = np.clip(np.searchsorted(src_keys[o], dst_keys), 0, len(src_keys) - 1)
    ok = src_keys[o][pos] == dst_keys
    return np.where(ok, o[pos], -1)


def load_inputs(data_dir):
    stats = D.load_stats(data_dir)
    cand, mu, meta = [], [], []
    for d in D.shard_dirs(data_dir):
        cand.append(np.load(os.path.join(d, "cand.npy")))
        mu.append(np.load(os.path.join(d, "mu_feats.npy")))
        meta.append({k: np.load(os.path.join(d, f"{k}.npy")) for k in D.META})
    meta = {k: np.concatenate([m[k] for m in meta]) for k in D.META}
    return stats, np.concatenate(cand), np.concatenate(mu), meta


def feature_sets(stats, cand, mu, meta, data_dir):
    cn = stats["features"]["cand"]
    mn = stats["features"]["tokens"]["mu"]
    sets, extra = {}, {}
    path = os.path.join(data_dir, "bdt_run2.npz")
    if os.path.exists(path):
        t = dict(np.load(path))
        idx = join(key_of(t), key_of(meta))
        names = [str(v) for v in t["variables"]]
        X = np.full((len(idx), len(names)), np.nan, dtype=np.float32)
        ok = idx >= 0
        for i, n in enumerate(names):
            X[ok, i] = t[n][idx[ok]]
        sets["bdt_run2"] = (X, names, XGB_RUN2)
        extra["massres"] = np.where(ok, t["massres"][np.maximum(idx, 0)], np.nan)
        print(f"bdt_run2 inputs found for {ok.sum()} / {len(ok)} candidates")
    else:
        print(f"no {path}: bdt_run2 skipped (run scripts/make_bdt_table.py at Bari)")

    ci = [cn.index(n) for n in RUN2LIKE_CAND if n in cn]
    cols, names = [cand[:, ci]], [cn[i] for i in ci]
    for n in RUN2LIKE_MU:
        if n in mn:
            v = mu[:, :, mn.index(n)].astype(np.float32)
            cols += [v.min(1, keepdims=True), v.max(1, keepdims=True)]
            names += [f"min_mu_{n}", f"max_mu_{n}"]
    sets["bdt_run2like"] = (np.concatenate(cols, axis=1).astype(np.float32), names, XGB_DEFAULT)
    X_all = np.concatenate([cand, mu.reshape(len(mu), -1)], axis=1).astype(np.float32)
    sets["bdt_candmu"] = (X_all, cn + [f"mu{i + 1}_{n}" for i in range(mu.shape[1]) for n in mn], XGB_DEFAULT)
    if "isGlobal" in mn:
        extra["is3glb"] = (mu[:, :, mn.index("isGlobal")].astype(np.float32) > 0.5).all(1)
    return sets, extra


# ---------------------------------------------------------------- BDTs
def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def train_bdt(X, meta, fold, params, prod_frac=None, npv_reweight=True, seed=0):
    tr_f, va_f, te_f = D.fold_split(fold)
    rows_tr = np.isin(meta["fold"], tr_f) & np.isin(meta["role"], [0, 1])
    rows_va = np.isin(meta["fold"], va_f) & np.isin(meta["role"], [0, 1])
    rows_te = np.isin(meta["fold"], te_f)
    w = D.Weighter(meta, rows_tr, prod_frac=prod_frac, npv_reweight=npv_reweight)(meta)
    y = meta["label"].astype(np.float32)
    try:
        import xgboost as xgb
        dev = "cuda" if _has_cuda() else "cpu"
        clf = xgb.XGBClassifier(**params, tree_method="hist", device=dev, eval_metric="logloss", random_state=seed)
        clf.fit(X[rows_tr], y[rows_tr], sample_weight=w[rows_tr],
                eval_set=[(X[rows_va], y[rows_va])], sample_weight_eval_set=[w[rows_va]], verbose=False)
        info = f"xgboost ({dev}), {clf.best_iteration + 1} trees"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(max_iter=1000, learning_rate=params["learning_rate"],
                                             max_depth=params["max_depth"], early_stopping=True,
                                             validation_fraction=0.1, random_state=seed)
        clf.fit(X[rows_tr], y[rows_tr], sample_weight=w[rows_tr])
        info = f"sklearn HistGradientBoosting, {clf.n_iter_} iterations (xgboost not available)"
    return rows_te, clf.predict_proba(X[rows_te])[:, 1], w, info


# ---------------------------------------------------------------- metrics
def cut_at_sig_eff(s, y, w, eff):
    sig = y == 1
    o = np.argsort(-s[sig])
    c = np.cumsum(w[sig][o]) / w[sig].sum()
    return s[sig][o][min(np.searchsorted(c, eff), sig.sum() - 1)]


def bkg_eff_at(s, y, w, eff):
    cut = cut_at_sig_eff(s, y, w, eff)
    return float((s[y == 0] >= cut).mean())


def _integ(l, a, b):
    small = np.abs(l) < 1e-9
    ls = np.where(small, 1.0, l)
    return np.where(small, b - a, (np.exp(l * b) - np.exp(l * a)) / ls)


def exp_fit(m, fit, target, excl):
    """Background expected in `target` from an exponential fitted to m in `fit` minus `excl` windows."""
    pieces = [fit]
    for e in excl:
        new = []
        for a, b in pieces:
            if e[1] <= a or e[0] >= b:
                new.append((a, b))
            else:
                if e[0] > a:
                    new.append((a, e[0]))
                if e[1] < b:
                    new.append((e[1], b))
        pieces = new
    inside = np.zeros(len(m), bool)
    for a, b in pieces:
        inside |= (m >= a) & (m < b)
    m = m[inside]
    n = len(m)
    width = sum(b - a for a, b in pieces)
    if n < 3:
        return max(n, 1.0) * (target[1] - target[0]) / width  # flat, at least one event's worth
    lam = np.linspace(-10, 10, 1001)
    norm = sum(_integ(lam, a, b) for a, b in pieces)
    ll = lam * m.sum() - n * np.log(norm)
    best = lam[np.argmax(ll)]
    return float(n * _integ(best, *target) / sum(_integ(best, a, b) for a, b in pieces))


def asimov_ul(s, b):
    """Expected 95% CL UL on the signal scale mu (yields mu*s_i), background-only Asimov, sqrt(q_mu) = 1.645."""
    s, b = np.asarray(s, float), np.maximum(np.asarray(b, float), 1e-3)
    if s.sum() <= 0:
        return np.inf

    def q(mu):
        return 2 * np.sum(mu * s - b * np.log1p(mu * s / b))

    lo, hi = 0.0, 1.0
    while q(hi) < 1.645 ** 2:
        hi *= 2
        if hi > 1e15:
            return np.inf
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if q(mid) < 1.645 ** 2 else (lo, mid)
    return hi


def interval_table(score, mass, ws, sig, bkg, window, fit, excl, bkg_scale, n_grid):
    """(s, b) for every score interval [q_i, q_j) of a signal-quantile grid (q_G = +inf)."""
    in_win = sig & (mass >= window[0]) & (mass < window[1])
    q = np.unique(np.quantile(score[sig], np.linspace(0, 0.995, n_grid)))
    edges = np.append(q, np.inf)
    G = len(edges)
    S = np.zeros((G, G))
    B = np.full((G, G), np.inf)
    for i in range(G - 1):
        for j in range(i + 1, G):
            sel_s = in_win & (score >= edges[i]) & (score < edges[j])
            S[i, j] = ws[sel_s].sum()
            sel_b = bkg & (score >= edges[i]) & (score < edges[j])
            B[i, j] = exp_fit(mass[sel_b], fit, window, excl) * bkg_scale
    return edges, S, B


def best_split(edges, S, B, n_cat):
    """Best n_cat contiguous categories at the top of the score range (lower scores discarded)."""
    G = len(edges)
    best = (np.inf, None)
    import itertools
    for cuts in itertools.combinations(range(G - 1), n_cat):
        bounds = list(cuts) + [G - 1]
        s = [S[bounds[k], bounds[k + 1]] for k in range(n_cat)]
        b = [B[bounds[k], bounds[k + 1]] for k in range(n_cat)]
        ul = asimov_ul(s, b)
        if ul < best[0]:
            best = (ul, dict(cuts=[float(edges[c]) for c in cuts], sig=s, bkg=b))
    return best


def expected_limits(score, meta, w, extra, bkg_scale, n_grid=25):
    sig, bkg = meta["role"] == 0, meta["role"] == 1
    ws = np.where(sig, w, 0) / w[sig].sum()   # fraction of all signal candidates in the training set
    mass = meta["mass"]
    out = {}
    e, S, B = interval_table(score, mass, ws, sig, bkg, BLIND, (1.40, 2.20), [BLIND], bkg_scale, n_grid)
    ul, d = best_split(e, S, B, 2)
    out["inclusive"] = {"ul": ul, **d}
    if "massres" in extra and "is3glb" in extra:
        res, glb = extra["massres"], extra["is3glb"]
        s_all, b_all, cats = [], [], {}
        for name, (lo, hi, win) in RUN2_CATS.items():
            c = glb & (np.nan_to_num(res, nan=1.0) >= lo) & (np.nan_to_num(res, nan=1.0) < hi)
            e, S, B = interval_table(score, mass, ws, sig & c, bkg & c, win, (1.62, 2.0), [BLIND, win],
                                     bkg_scale, n_grid)
            ul_c, d = best_split(e, S, B, 3)
            cats[name] = {"ul": ul_c, **d}
            s_all += d["sig"]
            b_all += d["bkg"]
        out["run2_categories"] = {"ul": asimov_ul(s_all, b_all), "categories": cats}
    return out


def sideband_near_far(score, meta, effs=(1.0, 0.1, 0.01, 0.001)):
    """Sidebands near the SR (1.70-1.74, 1.82-1.86 GeV) over far (1.50-1.62, 1.94-2.06 GeV) above cuts
    at fixed sideband efficiency. A rise with tighter cuts: sculpting, or a background that peaks near
    the SR at high score; if the BDTs show the same rise at the same efficiency, it is the background."""
    b = meta["role"] == 1
    m = meta["mass"]
    near = b & (((m >= 1.70) & (m < 1.74)) | ((m >= 1.82) & (m < 1.86)))
    far = b & (((m >= 1.50) & (m < 1.62)) | ((m >= 1.94) & (m < 2.06)))
    out = {}
    for e in effs:
        cut = -np.inf if e >= 1 else np.quantile(score[b], 1 - e)
        n, f = int((near & (score >= cut)).sum()), int((far & (score >= cut)).sum())
        r = n / f if f else float("nan")
        out[f"{e:g}"] = dict(near=n, far=f, ratio=r, err=r * np.sqrt(1 / max(n, 1) + 1 / max(f, 1)) if f else float("nan"))
    return out


def ss3mu_ratio(score, meta, w, effs=(1.0, 0.5, 0.2)):
    tb = np.isin(meta["role"], [0, 1])
    ss = meta["role"] == 2
    out = {}
    for e in effs:
        cut = -np.inf if e >= 1 else cut_at_sig_eff(score[tb], meta["label"][tb], w[tb], e)
        m = meta["mass"][ss & (score >= cut)]
        n_sr = int(((m >= BLIND[0]) & (m < BLIND[1])).sum())
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
    sets, extra = feature_sets(stats, cand, mu, meta, data_dir)
    del cand, mu
    cfg_path = os.path.join(out_dir, "fold0", "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    prod = {int(a): float(b) for a, b in cfg["prod_frac"].items()} if cfg.get("prod_frac") else None

    scores, info = {}, {}
    w_all = np.zeros(len(meta["role"]), dtype=np.float32)
    for name, (X, names, params) in sets.items():
        scores[name] = np.full(len(meta["role"]), np.nan, dtype=np.float32)
        for k in folds:
            t1 = time.time()
            rows, s, w, inf = train_bdt(X, meta, k, params, prod_frac=prod,
                                        npv_reweight=cfg.get("npv_reweight", True))
            scores[name][rows] = s
            w_all[rows] = w[rows]
            print(f"{name} ({len(names)} inputs) fold {k}: {inf}, {time.time() - t1:.0f} s", flush=True)
        info[name] = {"n_inputs": len(names), "inputs": names, "trainer": inf}

    path = os.path.join(out_dir, "eval", "scores.npz")
    if os.path.exists(path):
        t = dict(np.load(path))
        idx = join(key_of(t), key_of(meta))
        tr = np.where(idx >= 0, t["score"][np.maximum(idx, 0)], np.nan).astype(np.float32)
        scores = {"transformer": tr, **scores}
        print(f"transformer scores matched for {(idx >= 0).sum()} / {len(idx)} candidates")
    else:
        print(f"no {path}: run tau3muml.evaluate first to include the transformer")

    have = np.all([np.isfinite(v) for v in scores.values()], axis=0)
    # --bkg-scale is full data / training set; if only some folds are scored, scale up accordingly
    n_bkg_all = int((meta["role"] == 1).sum())
    n_bkg_used = int(((meta["role"] == 1) & have).sum())
    bkg_scale_eff = bkg_scale * n_bkg_all / max(n_bkg_used, 1)
    print(f"background scale: {bkg_scale} x {n_bkg_all}/{n_bkg_used} sideband candidates used = {bkg_scale_eff:.0f}")
    meta = {k: v[have] for k, v in meta.items()}
    extra = {k: v[have] for k, v in extra.items()}
    scores = {k: v[have] for k, v in scores.items()}
    w = w_all[have]
    tb = np.isin(meta["role"], [0, 1])
    y = meta["label"]
    subsets = {"all": np.ones(len(y), bool)}
    if "is3glb" in extra:
        subsets["3glb"] = extra["is3glb"]

    res = {"n": {r: int((meta["role"] == i).sum()) for i, r in enumerate(["signal", "sidebands", "ss3mu"])},
           "bkg_scale": bkg_scale_eff, "methods": {}}
    for name, s in scores.items():
        r = {"bkg_eff": {sub: {f"{e:.1f}": bkg_eff_at(s[tb & m], y[tb & m], w[tb & m], e)
                               for e in (0.8, 0.6, 0.5, 0.3, 0.2, 0.1)} for sub, m in subsets.items()},
             "ss3mu_SR_over_SB": ss3mu_ratio(s, meta, w),
             "sideband_near_far": sideband_near_far(s, meta),
             "expected_ul": expected_limits(s, meta, w, extra, bkg_scale_eff)}
        if name in info:
            r.update(n_inputs=info[name]["n_inputs"], trainer=info[name]["trainer"])
        res["methods"][name] = r
        print(f"{name}: limits done", flush=True)
    ref = "bdt_run2" if "bdt_run2" in res["methods"] else "bdt_run2like"
    res["reference"] = ref
    for name, r in res["methods"].items():
        for c, v in r["expected_ul"].items():
            v["ratio"] = v["ul"] / res["methods"][ref]["expected_ul"][c]["ul"]

    ev = os.path.join(out_dir, "eval")
    os.makedirs(ev, exist_ok=True)
    with open(os.path.join(ev, "compare_bdt.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    np.savez(os.path.join(ev, "compare_scores.npz"), **{f"score_{k}": v for k, v in scores.items()},
             **meta, **extra, weight=w)
    print_table(res, list(subsets))
    if plots:
        make_plots(scores, meta, w, subsets, ev)
        plot_fits(scores, meta, w, res, ev)
    return res


def print_table(res, subsets):
    meth = list(res["methods"])
    head = " ".join(f"{m:>13s}" for m in meth)
    for sub in subsets:
        print(f"\nsideband efficiency at fixed signal efficiency ({sub})")
        print(f"{'sig eff':>8s} {head}")
        for e in ("0.8", "0.6", "0.5", "0.3", "0.2", "0.1"):
            print(f"{e:>8s} " + " ".join(f"{res['methods'][m]['bkg_eff'][sub][e]:13.2e}" for m in meth))
    print("\nSS3MU SR/SB ratio vs cut at signal efficiency (constant = no sculpting)")
    for e in ("1.0", "0.5", "0.2"):
        print(f"{e:>8s} " + " ".join(f"{res['methods'][m]['ss3mu_SR_over_SB'][e]['ratio']:6.3f}±"
                                    f"{res['methods'][m]['ss3mu_SR_over_SB'][e]['err']:<6.3f}" for m in meth))
    print("\nsidebands near/far the SR at fixed sideband efficiency (constant = no sculpting)")
    for e in ("1", "0.1", "0.01", "0.001"):
        print(f"{e:>8s} " + " ".join(f"{res['methods'][m]['sideband_near_far'][e]['ratio']:6.3f}±"
                                    f"{res['methods'][m]['sideband_near_far'][e]['err']:<6.3f}" for m in meth))
    print(f"\nexpected 95% CL UL on the signal yield, ratio to {res['reference']} (lower is better)")
    for c in res["methods"][meth[0]]["expected_ul"]:
        print(f"{c:>16s} " + " ".join(f"{res['methods'][m]['expected_ul'][c]['ratio']:13.3f}" for m in meth))


def plot_fits(scores, meta, w, res, ev):
    """Sideband mass and exponential fit in each inclusive score category, per method: fits_<method>.png"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sig, bkg = meta["role"] == 0, meta["role"] == 1
    m = meta["mass"]
    pieces = [(1.40, BLIND[0]), (BLIND[1], 2.20)]
    bins = np.linspace(1.40, 2.20, 41)
    bw = bins[1] - bins[0]
    c = 0.5 * (bins[1:] + bins[:-1])
    for name, s in scores.items():
        cuts = res["methods"][name]["expected_ul"]["inclusive"]["cuts"] + [np.inf]
        fig, axs = plt.subplots(1, len(cuts) - 1, figsize=(6.5 * (len(cuts) - 1), 5), squeeze=False)
        for k, ax in enumerate(axs[0]):
            sb = bkg & (s >= cuts[k]) & (s < cuts[k + 1])
            sg = sig & (s >= cuts[k]) & (s < cuts[k + 1])
            mm = m[sb]
            mm = mm[(mm >= 1.40) & (mm <= 2.20) & ((mm < BLIND[0]) | (mm >= BLIND[1]))]
            h, _ = np.histogram(mm, bins)
            ok = (c < BLIND[0]) | (c > BLIND[1])
            ax.errorbar(c[ok], h[ok], yerr=np.sqrt(h[ok]), fmt="o", color="k", ms=4, label=f"sidebands ({len(mm)})")
            xx = np.linspace(1.40, 2.20, 300)
            if len(mm) >= 3:
                lam = np.linspace(-10, 10, 1001)
                norm = sum(_integ(lam, a, b) for a, b in pieces)
                best = lam[np.argmax(lam * mm.sum() - len(mm) * np.log(norm))]
                ax.plot(xx, len(mm) * bw * np.exp(best * xx) / sum(_integ(best, a, b) for a, b in pieces), "r-",
                        label=f"exponential, slope {best:.1f}/GeV")
            ax.axvspan(*BLIND, color="grey", alpha=0.2, label=f"SR: {exp_fit(m[sb], (1.40, 2.20), BLIND, [BLIND]):.2f} expected")
            hs, _ = np.histogram(m[sg], bins, weights=w[sg])
            if hs.max() > 0:
                ax.step(bins[:-1], hs / hs.max() * max(h.max(), 3) * 0.9, where="post", color="b", label="signal MC (shape)")
            hi = "" if not np.isfinite(cuts[k + 1]) else f" < {cuts[k + 1]:.5f}"
            ax.set_title(f"{name}: {cuts[k]:.5f} <= score{hi}", fontsize=11)
            ax.set_xlabel("m(3mu) [GeV]")
            ax.set_ylim(0, max(h.max(), 3) * 1.6)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(ev, f"fits_{name}.png"), dpi=120)
        plt.close(fig)


def make_plots(scores, meta, w, subsets, ev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, len(subsets), figsize=(5.5 * len(subsets), 4.8), squeeze=False)
    for ax, (sub, msk) in zip(axs[0], subsets.items()):
        tb = np.isin(meta["role"], [0, 1]) & msk
        y = meta["label"][tb]
        for name, s in scores.items():
            s_, ww = s[tb], w[tb]
            o = np.argsort(-s_)
            tp = np.cumsum(ww[o] * (y[o] == 1)); fp = np.cumsum(y[o] == 0)
            ax.plot(tp / tp[-1], np.maximum(fp / fp[-1], 1e-7), label=name)
        ax.set_yscale("log"); ax.set_xlabel("signal efficiency"); ax.set_ylabel("sideband efficiency")
        ax.set_title(sub); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
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
