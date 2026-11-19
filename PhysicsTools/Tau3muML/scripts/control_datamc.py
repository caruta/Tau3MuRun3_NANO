#!/usr/bin/env python3
"""Data/MC agreement of the classifier inputs on Ds -> phi(mumu) pi, per data-taking era.

Question: are the inputs of the transformer and of the BDTs well modelled by MC? If not, a
classifier trained on MC signal vs data background can learn "MC-like" instead of "signal-like",
and its signal efficiency in data is lower than MC says. Ds -> phi(mumu) pi is a real-signal-like
sample in both data and MC.

Method (sideband subtraction, as in section 8.3 of the Run 2 analysis note, with sidebands on both
sides of the peak): fit of the mumupi mass in 1.88-2.08 GeV (quadratic background, double-Gaussian
Ds, D+ tail); data in the Ds peak window (the one with the best S/sqrt(S+B)) minus the sidebands 1.905-1.925 and 2.015-2.055
GeV scaled to the fitted background under the peak; compared with Ds -> phi pi MC reweighted in nPV
to the background-subtracted data. A per-input check of the subtraction: the separation between the
left and the right sideband (column sideband_left_vs_right): large values mean the background shape
changes across the peak, and the subtracted distribution of that input is not reliable. Everything is done separately for each era, for early / late 2024
(run < / >= 383000) and for the whole year.

Inputs compared: exactly the transformer features (tau3muml.features, names from the training set's
stats.json), as the networks see them before standardisation:
  cand:*        candidate features
  mu:*          the two muon legs (leg 1 and 2)
  pi:*          the track leg, for the muon-token features that are defined for a track
  trk/neu/sv/jet:*   every real context token, and the token multiplicities n_<type>
For each input and group: separation = 2 |AUC(data vs MC) - 0.5| (0 = identical, 1 = disjoint) and
chi2/ndf of the normalised shapes.

Three stages (run at Bari inside cmsenv, from any directory):
  python3 control_datamc.py fit    --flat-dir FLAT --train-set TRAINSET --work WORK --jobs 16
  python3 control_datamc.py hist   --flat-dir FLAT --train-set TRAINSET --work WORK --jobs 16
  python3 control_datamc.py report --work WORK [--model-config OUT/fold0/config.json]
fit:    selection, mass fits, sideband scale factors and nPV weights per group, input binning
hist:   histograms of every input (peak, sideband, MC) per group
report: WORK/report/{separation.csv, summary.json, mass_fits.png, separation_vs_era.png,
        worst_inputs_<group>.png, inputs_<family>.png}
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import pickle
import re
import sys
import time
import zlib

import awkward as ak
import numpy as np
import uproot

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tau3muml import features as F  # noqa: E402

FIT_RANGE = (1.70, 2.10)        # selected mass range (and the mass histogram)
LOCAL_FIT = (1.88, 2.08)        # range of the Ds fit
SBL = (1.905, 1.925)            # sidebands next to the Ds peak (right of the D+ peak)
SBR = (2.015, 2.055)
SLICES = np.round(np.arange(LOCAL_FIT[0], LOCAL_FIT[1] + 1e-9, 0.005), 4)   # 5 MeV slices for nPV
RUN_SPLIT = 383000
MU_MASS = 0.1056584
N_BINS = 40
MC_SAMPLE_MAX = 300000   # MC values per input used to choose the bins


# ---------------------------------------------------------------- inputs and selection

def list_files(flat_dir, data_pat, mc_pat):
    out = []
    for pat, is_data in ((data_pat, True), (mc_pat, False)):
        for d in sorted(glob.glob(os.path.join(flat_dir, "DsPhiPi", pat))):
            out += [(f, os.path.basename(d), is_data) for f in sorted(glob.glob(os.path.join(d, "*.root")))]
    return out


def era_of(sample):
    m = re.search(r"era([A-Z])", sample)
    return m.group(1) if m else "MC"


def leg(a, name, i, fill=np.nan):
    x = np.asarray(ak.to_numpy(a[name][:, i]), dtype=np.float64)
    return np.where((x <= -998) | ~np.isfinite(x), fill, x)


def selection(a, is_data, cfg):
    """Ds -> phi(mumu) pi selection (Run 2 AN section 8.1, with the Run 3 muon acceptance)."""
    pt = [leg(a, "mu_pt", i, 0) for i in range(3)]
    eta = [leg(a, "mu_eta", i, 9) for i in range(3)]
    phi = [leg(a, "mu_phi", i, 0) for i in range(3)]
    px = [p * np.cos(f) for p, f in zip(pt, phi)]
    py = [p * np.sin(f) for p, f in zip(pt, phi)]
    pz = [p * np.sinh(e) for p, e in zip(pt, eta)]
    e = [np.sqrt(px[i] ** 2 + py[i] ** 2 + pz[i] ** 2 + MU_MASS ** 2) for i in range(2)]
    mmumu = np.sqrt(np.maximum((e[0] + e[1]) ** 2 - (px[0] + px[1]) ** 2 - (py[0] + py[1]) ** 2 - (pz[0] + pz[1]) ** 2, 0))
    ok = (mmumu > cfg["phi_window"][0]) & (mmumu < cfg["phi_window"][1])
    for i in range(2):
        aeta = np.abs(eta[i])
        ok &= ((aeta <= 1.2) & (pt[i] > 3.5)) | ((aeta > 1.2) & (aeta < 2.4) & (pt[i] > 2.0))
        ok &= leg(a, "mu_isGlobal", i, 0) > 0.5
    ok &= (pt[2] > cfg["trk_pt_min"]) & (np.abs(eta[2]) < 2.4)
    ok &= np.nan_to_num(np.asarray(ak.to_numpy(a["cand_flightDistSig"]), np.float64), nan=-1) >= cfg["fsig_min"]
    m = np.asarray(ak.to_numpy(a["cand_mass"]), np.float64)
    ok &= (m >= FIT_RANGE[0]) & (m <= FIT_RANGE[1])
    if cfg["hlt"]:
        b = "Trigger_" + cfg["hlt"]
        ok &= np.asarray(ak.to_numpy(a[b])) > 0.5 if b in a.fields else np.zeros(len(m), bool)
    if not is_data and "label_dsphipi" in a.fields:
        ok &= np.asarray(ak.to_numpy(a["label_dsphipi"])) == 1
    return ok, m


SEL_BRANCHES = ["run", "nPVtx", "cand_mass", "cand_flightDistSig", "mu_pt", "mu_eta", "mu_phi", "mu_isGlobal", "label_dsphipi"]


def groups_of(sample, run):
    """Group labels of each candidate: era, early/late, all (MC: 'MC')."""
    era = era_of(sample)
    if era == "MC":
        return [("MC", np.ones(len(run), bool))]
    return [(f"era{era}", np.ones(len(run), bool)), ("early", run < RUN_SPLIT), ("late", run >= RUN_SPLIT),
            ("all", np.ones(len(run), bool))]


# ---------------------------------------------------------------- feature evaluation

def variables(a, names):
    """{variable name: (values [n_entries], candidate index [n_entries])} for one chunk."""
    out = {}
    n = len(a)
    idx = np.arange(n)
    for k in names["cand"]:
        try:
            out[f"cand:{k}"] = (np.asarray(ak.to_numpy(F.CAND_FEATURES[k](a)), np.float64), idx)
        except Exception:
            pass
    for t, feats in names["tokens"].items():
        nmax = names["token_max"][t]
        if t == "mu":
            for k in feats:
                try:
                    col = F._pad(F.TOKEN_FEATURES["mu"][k](a), 3).astype(np.float64)
                except Exception:
                    continue
                out[f"mu:{k}"] = (col[:, :2].ravel(), np.repeat(idx, 2))
                out[f"pi:{k}"] = (col[:, 2], idx)
            continue
        key = f"{t}_eta" if f"{t}_eta" in a.fields else f"{t}_pt"
        if key not in a.fields:
            continue
        counts = np.minimum(ak.to_numpy(ak.num(a[key], axis=1)), nmax)
        out[f"{t}:n"] = (counts.astype(np.float64), idx)
        mask = np.arange(nmax)[None, :] < counts[:, None]
        cidx = np.broadcast_to(idx[:, None], mask.shape)[mask]
        for k in feats:
            try:
                col = F._pad(F.TOKEN_FEATURES[t][k](a), nmax).astype(np.float64)
            except Exception:
                continue
            out[f"{t}:{k}"] = (col[mask], cidx)
    return {k: (np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0), i) for k, (v, i) in out.items()}


def read_chunks(path, branches, step):
    with uproot.open(path) as fh:
        t = fh["Events"]
        have = set(t.keys())
        names = [b for b in branches if b in have or any(re.fullmatch(b.replace("*", ".*"), h) for h in have)]
        for a in t.iterate(filter_name=names, step_size=step):
            yield a


# ---------------------------------------------------------------- stage 1: selection, mass fit, binning

def stage1_file(args):
    path, sample, is_data, cfg, names, mc_sample_frac = args
    edges_m = np.linspace(*FIT_RANGE, 201)
    edges_pv = np.arange(0, 101)
    res = {"mass": {}, "pv2d": {}, "pv_mc": None, "mc_sample": None, "n": 0}
    rng = np.random.default_rng(zlib.crc32(path.encode()))
    try:
        for a in read_chunks(path, SEL_BRANCHES + ["Trigger_" + cfg["hlt"]] if cfg["hlt"] else SEL_BRANCHES, cfg["step"]):
            ok, m = selection(a, is_data, cfg)
            if cfg["data_frac"] < 1 and is_data:
                ok &= rng.random(len(ok)) < cfg["data_frac"]
            run = ak.to_numpy(a["run"])[ok]
            pv = ak.to_numpy(a["nPVtx"])[ok]
            m = m[ok]
            res["n"] += int(ok.sum())
            if is_data:
                for g, sel in groups_of(sample, run):
                    res["mass"][g] = res["mass"].get(g, 0) + np.histogram(m[sel], edges_m)[0]
                    res["pv2d"][g] = res["pv2d"].get(g, 0) + np.histogram2d(m[sel], pv[sel], [SLICES, edges_pv])[0]
            else:
                res["mass"]["MC"] = res["mass"].get("MC", 0) + np.histogram(m, edges_m)[0]
                res["pv_mc"] = (0 if res["pv_mc"] is None else res["pv_mc"]) + np.histogram(pv, edges_pv)[0]
    except Exception as e:
        return f"{path}: {type(e).__name__}: {str(e)[:200]}"
    if not is_data and rng.random() < mc_sample_frac:
        # a sample of MC inputs for the binning
        vals = {}
        for a in read_chunks(path, F.required_branches(names) + ["label_dsphipi", "mu_isGlobal"], cfg["step"]):
            ok, _ = selection(a, False, cfg)
            for k, (v, _) in variables(a[ok], names).items():
                vals.setdefault(k, []).append(v[:5000].astype(np.float32))
            break
        res["mc_sample"] = {k: np.concatenate(v) for k, v in vals.items()}
    return res


def _window_sum(lo, hi, f, step=0.0002):
    x = np.arange(lo + step / 2, hi, step)
    return float(np.sum(f(x)) * step)


def fit_mass(counts, edges):
    """Binned Poisson fit in LOCAL_FIT: quadratic background + double-Gaussian Ds + Gaussian D+ tail.
    Returns the Ds peak window (best S/sqrt(S+B), on the 5 MeV slices) and the sideband scale
    alpha = fitted background under the peak / data in the two sidebands."""
    from scipy.optimize import minimize
    xa = 0.5 * (edges[1:] + edges[:-1])
    sel = (xa >= LOCAL_FIT[0]) & (xa <= LOCAL_FIT[1])
    x, n = xa[sel], counts[sel].astype(float)
    w = edges[1] - edges[0]
    T = n.sum()
    xs = lambda m: (m - 1.98) / 0.1
    gauss = lambda m, mu, sg: np.exp(-0.5 * ((m - mu) / sg) ** 2) / (np.sqrt(2 * np.pi) * sg)

    def parts(p, m):
        fN, mu, s1, f, k, fP, mP, b0, b1, b2 = p
        u = xs(m)
        bkg = T * np.maximum(b0 + b1 * u + b2 * u * u, 0) / (LOCAL_FIT[1] - LOCAL_FIT[0])
        ds = T * fN * (f * gauss(m, mu, s1) + (1 - f) * gauss(m, mu, k * s1))
        dp = T * fP * gauss(m, mP, s1 * mP / mu)
        return bkg, ds, dp

    def nll(p):
        b, d, q = parts(p, x)
        mu = np.maximum((b + d + q) * w, 1e-9)
        return float(np.sum(mu - n * np.log(mu)))

    side = ((x >= 1.90) & (x <= 1.935)) | (x >= 2.01)
    c = np.polyfit(xs(x[side]), n[side] / w / T * (LOCAL_FIT[1] - LOCAL_FIT[0]), 2)
    bkg0 = np.polyval(c, xs(x)) * T / (LOCAL_FIT[1] - LOCAL_FIT[0]) * w
    fN0 = max(np.sum((n - bkg0)[(x > 1.945) & (x < 1.99)]) / T, 1e-3)
    p0 = [fN0, 1.968, 0.009, 0.6, 2.0, 0.005, 1.8695, c[2], c[1], c[0]]
    bounds = [(0, 1), (1.955, 1.98), (0.003, 0.03), (0.05, 1), (1.1, 4), (0, 0.5), (1.862, 1.876),
              (None, None), (None, None), (None, None)]
    r = minimize(nll, p0, bounds=bounds, method="L-BFGS-B", options={"maxiter": 5000})
    r = minimize(nll, r.x, bounds=bounds, method="Powell", options={"maxiter": 20000})
    r = minimize(nll, r.x, bounds=bounds, method="L-BFGS-B", options={"maxiter": 5000})
    p = r.x
    fN, mu, s1, f, k = p[:5]
    s_eff = float(np.sqrt(f * s1 ** 2 + (1 - f) * (k * s1) ** 2))
    bfun = lambda m: parts(p, m)[0]
    dsfun = lambda m: parts(p, m)[1]
    # peak window on the 5 MeV slices: the one with the best S/sqrt(S+B), i.e. the most precise
    # subtracted distribution (the double-Gaussian tails make mean +- 2 sigma_eff far too wide)
    best = (-1, SLICES[0], SLICES[-1])
    for lo_ in SLICES[(SLICES >= SBL[1]) & (SLICES < mu)]:
        for hi_ in SLICES[(SLICES <= SBR[0]) & (SLICES > mu)]:
            if hi_ - lo_ < 0.015:
                continue
            S_, B_ = _window_sum(lo_, hi_, dsfun, 0.001), _window_sum(lo_, hi_, bfun, 0.001)
            fom = S_ / np.sqrt(S_ + B_) if S_ + B_ > 0 else 0
            if fom > best[0]:
                best = (fom, lo_, hi_)
    lo, hi = float(best[1]), float(best[2])
    b_peak = _window_sum(lo, hi, bfun)
    s_peak = _window_sum(lo, hi, dsfun)
    allfun = lambda m: np.sum(parts(p, m), axis=0)
    # expected data in each sideband (all components, exact window integrals)
    m_l, m_r = _window_sum(*SBL, allfun), _window_sum(*SBR, allfun)
    # sideband weights that interpolate the background linearly in mass to the centre of the peak
    # window: w_l * centre(SBL) + w_r * centre(SBR) = centre(peak)
    c, cl, cr = 0.5 * (lo + hi), 0.5 * sum(SBL), 0.5 * sum(SBR)
    w_l = (cr - c) / (cr - cl)
    full = np.zeros(len(xa))
    full[sel] = np.sum(parts(p, x), axis=0) * w
    inside = lambda a, b: (edges[:-1] >= a - 1e-9) & (edges[1:] <= b + 1e-9)   # bins fully inside
    closure = lambda a, b: float(counts[inside(a, b)].sum() / max(full[inside(a, b)].sum(), 1))
    return {"params": dict(zip(["frac_Ds", "m_Ds", "sigma1", "f_core", "k_tail", "frac_Dplus", "m_Dplus", "b0", "b1", "b2"],
                               map(float, p))),
            "N_Ds": float(fN * T), "N_Ds_in_peak": s_peak, "sigma_eff": s_eff, "peak": [float(lo), float(hi)],
            "sbl": list(SBL), "sbr": list(SBR), "bkg_in_peak": b_peak, "expected_in_sbl": m_l, "expected_in_sbr": m_r,
            "alpha": b_peak / (m_l + m_r), "alpha_l": w_l * b_peak / m_l, "alpha_r": (1 - w_l) * b_peak / m_r,
            "w_left": w_l, "purity": float(s_peak / (s_peak + b_peak)),
            "data_over_fit": {"peak": closure(lo, hi), "sbl": closure(*SBL), "sbr": closure(*SBR)},
            "chi2_ndf": float(np.sum((n - full[sel]) ** 2 / np.maximum(n, 1)) / max(len(n) - 10, 1)),
            "converged": bool(r.success), "model": full.tolist()}


def stage1(args, cfg, names, files):
    tasks = [(f, s, d, cfg, names, args.mc_sample_frac) for f, s, d in files]
    acc = {"mass": {}, "pv2d": {}, "pv_mc": 0, "n": {}}
    mc_vals, errors, t0 = {}, [], time.time()
    with mp.Pool(args.jobs) as pool:
        for i, ((f, s, d), r) in enumerate(zip(files, pool.imap(stage1_file, tasks, chunksize=2))):
            if isinstance(r, str):
                errors.append(r)
                continue
            for k in ("mass", "pv2d"):
                for g, h in r[k].items():
                    acc[k][g] = acc[k].get(g, 0) + h
            if r["pv_mc"] is not None:
                acc["pv_mc"] = acc["pv_mc"] + r["pv_mc"]
            acc["n"][s] = acc["n"].get(s, 0) + r["n"]
            if r["mc_sample"]:
                for k, v in r["mc_sample"].items():
                    mc_vals.setdefault(k, []).append(v)
            if i % 50 == 0:
                print(f"  fit: {i + 1}/{len(files)} files, {time.time() - t0:.0f} s", flush=True)
    for e in errors:
        print("  unreadable:", e)
    if args.nparts > 1:
        path = os.path.join(args.work, "parts", f"fit_part{args.part:04d}.pkl")
        with open(path, "wb") as fh:
            pickle.dump({"acc": acc, "mc_vals": mc_vals, "cfg": cfg, "errors": errors}, fh)
        print(f"wrote {path}; when all parts are done: control_datamc.py fit-merge --work {args.work}")
        return
    stage1_finish(args, cfg, acc, mc_vals)


def stage1_merge(args):
    parts = sorted(glob.glob(os.path.join(args.work, "parts", "fit_part*.pkl")))
    acc = {"mass": {}, "pv2d": {}, "pv_mc": 0, "n": {}}
    mc_vals, cfg, nerr = {}, None, 0
    rng = np.random.default_rng(1)
    for i, p in enumerate(parts):
        if i % 20 == 0:
            print(f"  merging part {i + 1}/{len(parts)}", flush=True)
        r = pickle.load(open(p, "rb"))
        cfg, a = r["cfg"], r["acc"]
        nerr += len(r["errors"])
        for k in ("mass", "pv2d"):
            for g, h in a[k].items():
                acc[k][g] = acc[k].get(g, 0) + h
        acc["pv_mc"] = acc["pv_mc"] + a["pv_mc"]
        for s_, n in a["n"].items():
            acc["n"][s_] = acc["n"].get(s_, 0) + n
        for k, v in r["mc_vals"].items():
            # keep a bounded random MC sample per input, so the merge fits in a few hundred MB
            x = np.concatenate([np.asarray(a_, np.float32) for a_ in v]) if v else np.zeros(0, np.float32)
            if k in mc_vals:
                x = np.concatenate([mc_vals[k], x])
            if len(x) > MC_SAMPLE_MAX:
                x = rng.choice(x, MC_SAMPLE_MAX, replace=False)
            mc_vals[k] = x
        del r, a
    print(f"merged {len(parts)} parts ({nerr} unreadable files in total)")
    if args.expect_parts and len(parts) != args.expect_parts:
        sys.exit(f"expected {args.expect_parts} parts, found {len(parts)}: some jobs are missing")
    stage1_finish(args, cfg, acc, {k: [v] for k, v in mc_vals.items()})


def stage1_finish(args, cfg, acc, mc_vals):
    edges_m = np.linspace(*FIT_RANGE, 201)
    fits, pv_w = {}, {}
    for g, h in sorted(acc["mass"].items()):
        if g == "MC" or h.sum() < 500:
            continue
        fits[g] = fit_mass(h, edges_m)
        p = fits[g]["params"]
        F_ = fits[g]
        print(f"  {g:7s} Ds {F_['N_Ds']:10.0f}  mean {1000 * p['m_Ds']:.1f}  sigma core {1000 * p['sigma1']:.1f} "
              f"(core frac {p['f_core']:.2f}) eff {1000 * F_['sigma_eff']:.1f} MeV  Ds in peak {F_['N_Ds_in_peak']:.0f}  "
              f"peak {F_['peak'][0]:.3f}-{F_['peak'][1]:.3f}  purity {F_['purity']:.3f}  alpha L/R {F_['alpha_l']:.3f}/{F_['alpha_r']:.3f}  "
              f"data/fit peak {F_['data_over_fit']['peak']:.4f} SB {F_['data_over_fit']['sbl']:.4f}/{F_['data_over_fit']['sbr']:.4f}  "
              f"chi2/ndf {F_['chi2_ndf']:.2f}{'' if F_['converged'] else '  (fit did not converge)'}")
        sl = lambda lo, hi: (SLICES[:-1] >= lo - 1e-6) & (SLICES[1:] <= hi + 1e-6)
        pv2 = acc["pv2d"][g]
        sig_pv = pv2[sl(*F_["peak"])].sum(0) - F_["alpha"] * (pv2[sl(*SBL)].sum(0) + pv2[sl(*SBR)].sum(0))
        mc = acc["pv_mc"].astype(float)
        ratio = np.where(mc > 0, np.maximum(sig_pv, 0) / max(sig_pv.sum(), 1) / np.maximum(mc / max(mc.sum(), 1), 1e-12), 1.0)
        pv_w[g] = np.where(mc > 20, ratio, 1.0).tolist()
    edges = {}
    for k, v in mc_vals.items():
        v = np.concatenate(v)
        q = np.unique(np.quantile(v, np.linspace(0, 1, N_BINS + 1)))
        if len(q) < 3:
            q = np.unique(np.r_[v.min() - 0.5, np.unique(v) + 0.5]) if len(np.unique(v)) < 50 else np.linspace(v.min(), v.max() + 1e-6, N_BINS + 1)
        q[0], q[-1] = -np.inf, np.inf
        edges[k] = q.tolist()
    out = {"cfg": cfg, "fits": fits, "pv_weights": pv_w, "edges": edges, "n_selected": acc["n"],
           "mass_hist": {g: h.tolist() for g, h in acc["mass"].items()}, "mass_edges": edges_m.tolist()}
    with open(os.path.join(args.work, "fit.json"), "w") as fh:
        json.dump(out, fh)
    print(f"wrote {args.work}/fit.json ({len(edges)} inputs binned)")


# ---------------------------------------------------------------- stage 2: histograms

def stage2_file(args):
    path, sample, is_data, cfg, names, fit = args
    edges = {k: np.asarray(v) for k, v in fit["edges"].items()}
    pvw = {g: np.asarray(w) for g, w in fit["pv_weights"].items()}
    H = {}   # (group, kind) -> {var: (sumw, sumw2)}
    rng = np.random.default_rng(zlib.crc32(path.encode()))

    def fill(group, kind, vals, w):
        d = H.setdefault((group, kind), {})
        for k, (v, ci) in vals.items():
            if k not in edges:
                continue
            ww = w[ci]
            s = ww != 0
            if not s.any():
                continue
            h = np.histogram(v[s], edges[k], weights=ww[s])[0]
            h2 = np.histogram(v[s], edges[k], weights=ww[s] ** 2)[0]
            if k in d:
                d[k][0] += h
                d[k][1] += h2
            else:
                d[k] = [h, h2]

    try:
        br = F.required_branches(names) + SEL_BRANCHES + (["Trigger_" + cfg["hlt"]] if cfg["hlt"] else [])
        for a in read_chunks(path, br, cfg["step"]):
            ok, m = selection(a, is_data, cfg)
            if cfg["data_frac"] < 1 and is_data:
                ok &= rng.random(len(ok)) < cfg["data_frac"]
            if is_data:
                ok &= (m >= LOCAL_FIT[0]) & (m <= LOCAL_FIT[1])
            if not ok.any():
                continue
            a = a[ok]
            m = m[ok]
            vals = variables(a, names)
            run = ak.to_numpy(a["run"])
            pv = np.clip(ak.to_numpy(a["nPVtx"]).astype(int), 0, 99)
            if is_data:
                for g, sel in groups_of(sample, run):
                    if g not in fit["fits"]:
                        continue
                    for kind in ("peak", "sbl", "sbr"):
                        lo, hi = fit["fits"][g][kind]
                        fill(g, kind, vals, (sel & (m >= lo) & (m < hi)).astype(float))
            else:
                for g, w in pvw.items():
                    fill(g, "mc", vals, w[pv])
                fill("raw", "mc", vals, np.ones(len(a)))
    except Exception as e:
        return f"{path}: {type(e).__name__}: {str(e)[:200]}"
    return H


def stage2(args, cfg, names, files):
    fit = json.load(open(os.path.join(args.work, "fit.json")))
    tasks = [(f, s, d, cfg, names, fit) for f, s, d in files]
    tot, errors, t0 = {}, [], time.time()
    with mp.Pool(args.jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(stage2_file, tasks, chunksize=1)):
            if isinstance(r, str):
                errors.append(r)
                continue
            for key, d in r.items():
                T = tot.setdefault(key, {})
                for k, (h, h2) in d.items():
                    if k in T:
                        T[k][0] += h
                        T[k][1] += h2
                    else:
                        T[k] = [h.copy(), h2.copy()]
            if i % 50 == 0:
                print(f"  hist: {i + 1}/{len(files)} files, {time.time() - t0:.0f} s", flush=True)
    for e in errors:
        print("  unreadable:", e)
    flat = {f"{g}|{kind}|{k}|{j}": v[j] for (g, kind), d in tot.items() for k, v in d.items() for j in (0, 1)}
    if args.nparts > 1:
        path = os.path.join(args.work, "parts", f"hists_part{args.part:04d}.npz")
        np.savez_compressed(path, **flat)
        print(f"wrote {path}; when all parts are done: control_datamc.py hist-merge --work {args.work}")
        return
    np.savez_compressed(os.path.join(args.work, "hists.npz"), **flat)
    print(f"wrote {args.work}/hists.npz")


def stage2_merge(args):
    parts = sorted(glob.glob(os.path.join(args.work, "parts", "hists_part*.npz")))
    if args.expect_parts and len(parts) != args.expect_parts:
        sys.exit(f"expected {args.expect_parts} parts, found {len(parts)}: some jobs are missing")
    tot = {}
    for p in parts:
        z = np.load(p)
        for k in z.files:
            tot[k] = tot[k] + z[k] if k in tot else z[k].copy()
    np.savez_compressed(os.path.join(args.work, "hists.npz"), **tot)
    print(f"merged {len(parts)} parts into {args.work}/hists.npz")


# ---------------------------------------------------------------- stage 3: report

def alphas(f):
    """Sideband scales: linear interpolation in mass if available, else one common scale."""
    return (f["alpha_l"], f["alpha_r"]) if "alpha_l" in f else (f["alpha"], f["alpha"])


def separation(d, m):
    """2|AUC - 0.5| between two (non-negative, normalised) binned distributions."""
    d = np.maximum(d, 0); m = np.maximum(m, 0)
    if d.sum() <= 0 or m.sum() <= 0:
        return np.nan
    d, m = d / d.sum(), m / m.sum()
    auc = np.sum(d * (np.cumsum(m) - 0.5 * m))
    return float(2 * abs(auc - 0.5))


def report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fit = json.load(open(os.path.join(args.work, "fit.json")))
    z = np.load(os.path.join(args.work, "hists.npz"))
    H = {}
    for key in z.files:
        g, kind, k, j = key.split("|")
        H.setdefault((g, kind, k), [None, None])[int(j)] = z[key]
    dropped = set()
    if args.model_config:
        c = json.load(open(args.model_config))
        drop = c.get("drop") or []
        if isinstance(drop, dict):           # {"trk": [names]} form
            drop = [f"{t}:{v}" for t, lst in drop.items() for v in lst]
        dropped |= set(drop)                 # model config: ["trk:nPixHits", ...]
        for t in c.get("drop_tokens") or []:
            dropped |= {f"{t}:*"}
        print(f"inputs ignored by the model: {sorted(dropped) or 'none'}")
    groups = [g for g in sorted(fit["fits"], key=lambda g: (g not in ("all", "early", "late"), g))]
    rep = os.path.join(args.work, "report")
    os.makedirs(rep, exist_ok=True)
    rows, sep = [], {}
    variables_ = sorted({k for (_, kind, k) in H if kind == "mc"})
    for k in variables_:
        for g in groups:
            if (g, "peak", k) not in H or (g, "mc", k) not in H:
                continue
            al, ar = alphas(fit["fits"][g])
            pk, pk2 = H[(g, "peak", k)]
            s = 1.0
            sb = al * H[(g, "sbl", k)][0] + ar * H[(g, "sbr", k)][0]
            sb2 = al * al * H[(g, "sbl", k)][1] + ar * ar * H[(g, "sbr", k)][1]
            mc, mc2 = H[(g, "mc", k)]
            d = pk - s * sb
            dv = pk2 + s * s * sb2
            if d.sum() <= 0 or mc.sum() <= 0:
                continue
            dn, mn = d / d.sum(), mc / mc.sum()
            var = dv / d.sum() ** 2 + mc2 / mc.sum() ** 2
            ok = var > 0
            chi2 = float(np.sum((dn[ok] - mn[ok]) ** 2 / var[ok]))
            sp = separation(d, mc)
            sep[(k, g)] = sp
            closure = separation(H[(g, "sbl", k)][0], H[(g, "sbr", k)][0])
            # background-subtraction uncertainty: background under the peak varied by +-2 %
            syst = max(abs(separation(pk - s * f_ * sb, mc) - sp) for f_ in (0.98, 1.02))
            rows.append(dict(variable=k, group=g, separation=sp, separation_syst=syst,
                             chi2_ndf=chi2 / max(ok.sum() - 1, 1), sideband_left_vs_right=closure,
                             ds_yield=float(d.sum()) if k.startswith("cand:") else np.nan, dropped=k in dropped))
    import csv
    with open(os.path.join(rep, "separation.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    eras = [g for g in groups if g.startswith("era")]
    worst = sorted(variables_, key=lambda k: -np.nan_to_num(sep.get((k, "all"), 0)))
    summary = {"fits": {g: {k: v for k, v in f.items() if k != "model"} for g, f in fit["fits"].items()},
               "n_selected": fit["n_selected"],
               "worst_all": [(k, sep.get((k, "all")), sep.get((k, "early")), sep.get((k, "late"))) for k in worst[:40]],
               "family_mean_separation": {}}
    fam = sorted({k.split(":")[0] for k in variables_})
    for f_ in fam:
        summary["family_mean_separation"][f_] = {g: float(np.nanmean([sep.get((k, g), np.nan) for k in variables_
                                                                     if k.startswith(f_ + ":")])) for g in groups}
    with open(os.path.join(rep, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)

    # mass fits
    me = np.asarray(fit["mass_edges"]); x = 0.5 * (me[1:] + me[:-1])
    nG = len(groups); nc = min(nG, 4); nr = int(np.ceil(nG / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.4 * nr), squeeze=False)
    for ax, g in zip(axs.ravel(), groups):
        h = np.asarray(fit["mass_hist"][g]); f = fit["fits"][g]
        ax.errorbar(x, h, np.sqrt(h), fmt=".", ms=2, color="k")
        ax.plot(x, f["model"], "r-", lw=1)
        for lo, hi, c in (tuple(f["peak"]) + ("C2",), SBL + ("C0",), SBR + ("C0",)):
            ax.axvspan(lo, hi, color=c, alpha=0.12)
        ax.set_xlim(*LOCAL_FIT)
        ax.set_ylim(0, 1.15 * h[(x >= LOCAL_FIT[0]) & (x <= LOCAL_FIT[1])].max())
        ax.set_title(f"{g}: Ds {f['N_Ds']:.3g}, σ {1000 * f['sigma_eff']:.1f} MeV, purity {f['purity']:.2f}", fontsize=9)
        ax.set_xlabel("m(μμπ) [GeV]", fontsize=8)
    for ax in axs.ravel()[nG:]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(rep, "mass_fits.png"), dpi=110); plt.close(fig)

    # separation vs era for the worst inputs
    top = worst[:30]
    cols = ["all", "early", "late"] + eras
    M = np.array([[sep.get((k, g), np.nan) for g in cols] for k in top])
    fig, ax = plt.subplots(figsize=(1.0 * len(cols) + 4, 0.32 * len(top) + 1.5))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=max(0.2, np.nanmax(M) if M.size else 0.2))
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45)
    ax.set_yticks(range(len(top))); ax.set_yticklabels([k + ("  (dropped)" if k in dropped else "") for k in top], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6, color="w" if M[i, j] < 0.6 * im.get_clim()[1] else "k")
    fig.colorbar(im, ax=ax, label="separation 2|AUC−0.5|, data vs MC")
    fig.tight_layout(); fig.savefig(os.path.join(rep, "separation_vs_era.png"), dpi=110); plt.close(fig)

    # family averages vs era
    fig, ax = plt.subplots(figsize=(7, 4))
    for f_ in fam:
        ax.plot(eras, [summary["family_mean_separation"][f_].get(g, np.nan) for g in eras], "o-", label=f_)
    ax.set_ylabel("mean separation data vs MC"); ax.set_xlabel("era"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(rep, "separation_by_family.png"), dpi=110); plt.close(fig)

    # distributions of the worst inputs: early, late and MC
    def draw(keys, fname):
        nc = 4; nr = int(np.ceil(len(keys) / nc))
        fig, axs = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.2 * nr), squeeze=False)
        for ax, k in zip(axs.ravel(), keys):
            for g, c in (("early", "C0"), ("late", "C3")):
                if (g, "peak", k) not in H:
                    continue
                al, ar = alphas(fit["fits"][g])
                d = H[(g, "peak", k)][0] - al * H[(g, "sbl", k)][0] - ar * H[(g, "sbr", k)][0]
                ax.step(range(len(d)), d / max(d.sum(), 1), where="mid", color=c, label=f"data {g}")
                mc = H[(g, "mc", k)][0]
                ax.step(range(len(mc)), mc / max(mc.sum(), 1), where="mid", color=c, ls="--", label=f"MC (nPV of {g})")
            ax.set_title(f"{k}  sep {sep.get((k, 'all'), np.nan):.2f}", fontsize=9)
            ax.set_xlabel("MC-quantile bin", fontsize=8)
            ax.legend(fontsize=6)
        for ax in axs.ravel()[len(keys):]:
            ax.axis("off")
        fig.tight_layout(); fig.savefig(os.path.join(rep, fname), dpi=100); plt.close(fig)

    draw(worst[:16], "worst_inputs.png")
    for f_ in fam:
        keys = [k for k in variables_ if k.startswith(f_ + ":")]
        for i in range(0, len(keys), 16):
            draw(keys[i:i + 16], f"inputs_{f_}_{i // 16}.png")
    print(f"\n{'input':38s} {'all':>6s} {'early':>6s} {'late':>6s}   (separation 2|AUC-0.5|)")
    for k in worst[:25]:
        print(f"{k + (' (dropped)' if k in dropped else ''):38s} " + " ".join(f"{np.nan_to_num(sep.get((k, g), np.nan)):6.3f}" for g in ("all", "early", "late")))
    print(f"\nwrote {rep}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["fit", "fit-merge", "hist", "hist-merge", "report"])
    ap.add_argument("--work", required=True, help="output directory")
    ap.add_argument("--flat-dir")
    ap.add_argument("--train-set", help="training set (its stats.json gives the transformer inputs)")
    ap.add_argument("--data", default="2024era*", help="DsPhiPi data sample pattern")
    ap.add_argument("--mc", default="*_MC_DsPhiPi", help="DsPhiPi MC sample pattern")
    ap.add_argument("--data-frac", type=float, default=1.0, help="random fraction of data candidates used")
    ap.add_argument("--mc-sample-frac", type=float, default=0.3, help="fraction of MC files sampled for the binning")
    ap.add_argument("--phi-window", nargs=2, type=float, default=[1.00, 1.04])
    ap.add_argument("--trk-pt-min", type=float, default=1.2)
    ap.add_argument("--fsig-min", type=float, default=3.5)
    ap.add_argument("--hlt", default="", help="optional event trigger bit, e.g. HLT_DoubleMu3_Trk_Tau3mu")
    ap.add_argument("--step", type=int, default=20000)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--model-config", help="config.json of a trained model, to flag its dropped inputs")
    ap.add_argument("--part", type=int, default=0, help="batch mode: this job processes files part, part+nparts, ...")
    ap.add_argument("--nparts", type=int, default=1, help="batch mode: number of jobs (outputs go to WORK/parts/)")
    ap.add_argument("--expect-parts", type=int, default=0, help="merge stages: stop if fewer parts are found")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    os.makedirs(args.work, exist_ok=True)
    if args.stage == "report":
        return report(args)
    if args.stage == "fit-merge":
        return stage1_merge(args)
    if args.stage == "hist-merge":
        return stage2_merge(args)
    if args.nparts > 1:
        os.makedirs(os.path.join(args.work, "parts"), exist_ok=True)
    cfg = dict(phi_window=args.phi_window, trk_pt_min=args.trk_pt_min, fsig_min=args.fsig_min, hlt=args.hlt,
               step=args.step, data_frac=args.data_frac)
    stats = json.load(open(os.path.join(args.train_set, "stats.json")))
    names = stats["features"]
    files = list_files(args.flat_dir, args.data, args.mc)
    samples = sorted({s for _, s, _ in files})
    print(f"{len(files)} files from {len(samples)} samples: {samples}")
    if not any(not d for _, _, d in files):
        sys.exit("no MC files found (--mc)")
    if args.nparts > 1:
        files = files[args.part::args.nparts]
        print(f"part {args.part} of {args.nparts}: {len(files)} files")
    (stage1 if args.stage == "fit" else stage2)(args, cfg, names, files)


if __name__ == "__main__":
    main()
