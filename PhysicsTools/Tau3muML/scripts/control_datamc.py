#!/usr/bin/env python3
"""Data/MC agreement of the classifier inputs on Ds -> phi(mumu) pi, per data-taking era.

Question: are the inputs of the transformer and of the BDTs well modelled by MC? If not, a
classifier trained on MC signal vs data background can learn "MC-like" instead of "signal-like",
and its signal efficiency in data is lower than MC says. Ds -> phi(mumu) pi is a real-signal-like
sample in both data and MC.

Method (as in section 8.3 of the Run 2 analysis note): data in the Ds peak [1.93, 2.01] GeV minus
the sideband [1.72, 1.80] GeV scaled to the background under the peak from a fit of the mumupi mass
(exponential + Ds and D+ Gaussians), compared with Ds -> phi pi MC reweighted in nPV to the
background-subtracted data. Everything is done separately for each era, for early / late 2024
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

FIT_RANGE = (1.70, 2.10)
PEAK = (1.93, 2.01)
SB = (1.72, 1.80)
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
    res = {"mass": {}, "pv_peak": {}, "pv_sb": {}, "pv_mc": None, "mc_sample": None, "n": 0}
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
                    pk = sel & (m >= PEAK[0]) & (m <= PEAK[1])
                    sb = sel & (m >= SB[0]) & (m <= SB[1])
                    res["pv_peak"][g] = res["pv_peak"].get(g, 0) + np.histogram(pv[pk], edges_pv)[0]
                    res["pv_sb"][g] = res["pv_sb"].get(g, 0) + np.histogram(pv[sb], edges_pv)[0]
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


def fit_mass(counts, edges):
    """Binned Poisson fit: exponential + Gaussian Ds + Gaussian D+ (width scaled by mass)."""
    from scipy.optimize import minimize
    x = 0.5 * (edges[1:] + edges[:-1])
    w = edges[1] - edges[0]
    n = counts.astype(float)

    def model(p):
        nds, mds, sds, ndp, mdp, nb, lam = p
        sdp = sds * mdp / mds
        g = lambda N, mu, s: N * w * np.exp(-0.5 * ((x - mu) / s) ** 2) / (np.sqrt(2 * np.pi) * s)
        bk = np.exp(lam * (x - FIT_RANGE[0]))
        bk = nb * bk / (bk.sum())
        return g(nds, mds, sds) + g(ndp, mdp, sdp) + bk

    def nll(p):
        mu = np.maximum(model(p), 1e-9)
        return float(np.sum(mu - n * np.log(mu)))

    tot = n.sum()
    side = n[(x < 1.82) | (x > 2.05)].mean() * len(x)
    p0 = [max(tot - side, 1) * 0.6, 1.968, 0.012, max(tot - side, 1) * 0.3, 1.869, side, -1.0]
    bounds = [(0, None), (1.95, 1.99), (0.004, 0.04), (0, None), (1.85, 1.89), (0, None), (-20, 20)]
    r = minimize(nll, p0, bounds=bounds, method="L-BFGS-B")
    r = minimize(nll, r.x, bounds=bounds, method="L-BFGS-B")
    p = r.x
    lam, nb = p[6], p[5]
    shape = lambda lo, hi: (np.exp(lam * (hi - FIT_RANGE[0])) - np.exp(lam * (lo - FIT_RANGE[0]))) / lam if abs(lam) > 1e-9 else hi - lo
    norm = nb / shape(*FIT_RANGE)
    b_peak = norm * shape(*PEAK)
    n_sb = n[(x >= SB[0]) & (x <= SB[1])].sum()
    n_peak = n[(x >= PEAK[0]) & (x <= PEAK[1])].sum()
    return {"params": dict(zip(["N_Ds", "m_Ds", "sigma_Ds", "N_Dplus", "m_Dplus", "N_bkg", "slope"], map(float, p))),
            "bkg_in_peak": float(b_peak), "n_peak": float(n_peak), "n_sb": float(n_sb),
            "sb_scale": float(b_peak / n_sb) if n_sb else 0.0, "purity": float(1 - b_peak / max(n_peak, 1)),
            "converged": bool(r.success), "model": model(p).tolist()}


def stage1(args, cfg, names, files):
    tasks = [(f, s, d, cfg, names, args.mc_sample_frac) for f, s, d in files]
    acc = {"mass": {}, "pv_peak": {}, "pv_sb": {}, "pv_mc": 0, "n": {}}
    mc_vals, errors, t0 = {}, [], time.time()
    with mp.Pool(args.jobs) as pool:
        for i, ((f, s, d), r) in enumerate(zip(files, pool.imap(stage1_file, tasks, chunksize=2))):
            if isinstance(r, str):
                errors.append(r)
                continue
            for k in ("mass", "pv_peak", "pv_sb"):
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
    acc = {"mass": {}, "pv_peak": {}, "pv_sb": {}, "pv_mc": 0, "n": {}}
    mc_vals, cfg, nerr = {}, None, 0
    rng = np.random.default_rng(1)
    for i, p in enumerate(parts):
        if i % 20 == 0:
            print(f"  merging part {i + 1}/{len(parts)}", flush=True)
        r = pickle.load(open(p, "rb"))
        cfg, a = r["cfg"], r["acc"]
        nerr += len(r["errors"])
        for k in ("mass", "pv_peak", "pv_sb"):
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
        print(f"  {g:7s} Ds {p['N_Ds']:10.0f}  D+ {p['N_Dplus']:9.0f}  sigma {1000 * p['sigma_Ds']:.1f} MeV  "
              f"purity in peak {fits[g]['purity']:.2f}  SB scale {fits[g]['sb_scale']:.3f}")
        sig_pv = acc["pv_peak"][g] - fits[g]["sb_scale"] * acc["pv_sb"][g]
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
                ok &= ((m >= PEAK[0]) & (m <= PEAK[1])) | ((m >= SB[0]) & (m <= SB[1]))
            if not ok.any():
                continue
            a = a[ok]
            m = m[ok]
            vals = variables(a, names)
            run = ak.to_numpy(a["run"])
            pv = np.clip(ak.to_numpy(a["nPVtx"]).astype(int), 0, 99)
            if is_data:
                for g, sel in groups_of(sample, run):
                    pk = sel & (m >= PEAK[0]) & (m <= PEAK[1])
                    sb = sel & (m >= SB[0]) & (m <= SB[1])
                    fill(g, "peak", vals, pk.astype(float))
                    fill(g, "sb", vals, sb.astype(float))
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
            s = fit["fits"][g]["sb_scale"]
            pk, pk2 = H[(g, "peak", k)]
            sb, sb2 = H[(g, "sb", k)]
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
            rows.append(dict(variable=k, group=g, separation=sp, chi2_ndf=chi2 / max(ok.sum() - 1, 1),
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
        for lo, hi, c in (PEAK + ("C2",), SB + ("C0",)):
            ax.axvspan(lo, hi, color=c, alpha=0.12)
        ax.set_title(f"{g}: Ds {f['params']['N_Ds']:.3g}, purity {f['purity']:.2f}", fontsize=9)
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
                s = fit["fits"][g]["sb_scale"]
                d = H[(g, "peak", k)][0] - s * H[(g, "sb", k)][0]
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
    ap.add_argument("--phi-window", nargs=2, type=float, default=[0.97, 1.07])
    ap.add_argument("--trk-pt-min", type=float, default=1.2)
    ap.add_argument("--fsig-min", type=float, default=2.0)
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
