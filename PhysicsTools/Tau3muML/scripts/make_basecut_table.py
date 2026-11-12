#!/usr/bin/env python3
"""Analysis base cuts and best-candidate choice for the candidates of a training set (run at Bari, inside cmsenv).

The training set uses the loose preselection and keeps every triplet of an event. This script reads the
flattened ntuples and, for every training-set candidate, records whether it passes the analysis base
selection and whether it is the candidate the analysis would keep in its event, so that
tau3muml.compare_basecuts can redo the BDT/transformer comparison on an analysis-like sample without
retraining anything.

Base selection, as in PhysicsTools/Tau3muANA/tau3mu_analyser.py (process_tau3mu_events):
  hlt       event: any L1 seed of the list below AND HLT_DoubleMu3_TkMu_DsTau3Mu_v
  acc       each muon: |eta| <= 1.2 and pT > 3.5 GeV, or 1.2 < |eta| < 2.4 and pT > 2.0 GeV
  muid      each muon: global and medium
  charge    |Q| = 1 (|Q| = 3 for the SS3MU control candidates)
  drdz      dR < 0.6 and dz < 2.5 for each muon pair             (--dr-max, --dz-max)
  fsig      3D flight-distance significance >= 3.5               (--fsig-min)
  mass      1.62 <= m(3mu) <= 2.0 GeV
  trgmatch  each muon matched to one of the HLT paths below with dR < 0.03 and |dpT/pT| < 0.1
  base      all of the above
  best      base, and the lowest vertex chi2 among the base candidates of its event (and charge class)

Also stored: the two opposite-sign dimuon masses (for the Run 2 phi / omega vetoes), and how many
candidates of the event are missing from the flattened file (outside the flattening mass window, or
in the blinded signal region for data), since those cannot take part in the best-candidate choice.

  python3 make_basecut_table.py --train-set /lustre/.../train_Transformer_v1 \\
      --flat-dir /lustre/.../flat_Transformer_v1 --jobs 8

Output: <train-set>/basecuts.npz (and basecuts.json with the cut flow).
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time

import awkward as ak
import numpy as np
import uproot

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tau3muml import data as D  # noqa: E402

L1_SEEDS = ["L1_DoubleMu0er1p5_SQ_OS_dR_Max1p4", "L1_DoubleMu0er1p4_SQ_OS_dR_Max1p4",
            "L1_DoubleMu4_SQ_OS_dR_Max1p2", "L1_DoubleMu4p5_SQ_OS_dR_Max1p2",
            "L1_TripleMu_5SQ_3SQ_0OQ_DoubleMu_5_3_SQ_OS_Mass_Max9",
            "L1_TripleMu_5SQ_3SQ_0_DoubleMu_5_3_SQ_OS_Mass_Max9",
            "L1_DoubleMu0er2p0_SQ_OS_dEta_Max1p6", "L1_DoubleMu0er2p0_SQ_OS_dEta_Max1p5",
            "L1_TripleMu_2SQ_1p5SQ_0OQ_Mass_Max12", "L1_TripleMu_3SQ_2p5SQ_0OQ_Mass_Max12"]
HLT_REQUIRED = "HLT_DoubleMu3_TkMu_DsTau3Mu_v"
HLT_MATCH = ["HLT_DoubleMu3_TkMu_DsTau3Mu_v", "HLT_DoubleMu3_Trk_Tau3mu", "HLT_DoubleMu3_Trk_Tau3mu_NoL1Mass_v",
             "HLT_DoubleMu4_3_LowMass_v", "HLT_DoubleMu4_LowMass_Displaced_v"]
MU_MASS = 0.1056584
FLAGS = ["hlt", "acc", "muid", "charge", "drdz", "fsig", "mass", "trgmatch", "base", "best"]


def keyof(run, event, cidx):
    return (run.astype(np.uint64) << np.uint64(44)) | (event.astype(np.uint64) << np.uint64(10)) | cidx.astype(np.uint64)


def evkey(run, event):
    return (run.astype(np.uint64) << np.uint64(34)) ^ event.astype(np.uint64)


def branches(keys):
    want = ["run", "event", "cand_idx", "nCand", "isData", "cand_channel", "cand_charge", "cand_mass", "cand_chi2",
            "cand_flightDistSig", "cand_dr12", "cand_dr13", "cand_dr23", "cand_dz12", "cand_dz13", "cand_dz23",
            "mu_pt", "mu_eta", "mu_phi", "mu_charge", "mu_isGlobal", "mu_isMedium", "mu_trgDR", "mu_trgDPT"]
    want += ["Trigger_" + s for s in L1_SEEDS + [HLT_REQUIRED]] + ["mu_" + p for p in HLT_MATCH]
    return [b for b in dict.fromkeys(want) if b in keys], [b for b in dict.fromkeys(want) if b not in keys]


def np_(a, name, fill=np.nan):
    x = np.asarray(ak.to_numpy(a[name]), dtype=np.float64)
    return np.where((x <= -998) | ~np.isfinite(x), fill, x)


def os_masses(a):
    pt, eta, phi, q = (np_(a, f"mu_{v}") for v in ("pt", "eta", "phi", "charge"))
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    e = np.sqrt(px ** 2 + py ** 2 + pz ** 2 + MU_MASS ** 2)
    out = np.full((len(pt), 2), np.nan)
    for row_pairs in [(0, 1), (0, 2), (1, 2)]:
        i, j = row_pairs
        m = np.sqrt(np.maximum((e[:, i] + e[:, j]) ** 2 - (px[:, i] + px[:, j]) ** 2 - (py[:, i] + py[:, j]) ** 2
                               - (pz[:, i] + pz[:, j]) ** 2, 0))
        os_ = q[:, i] * q[:, j] < 0
        first = os_ & np.isnan(out[:, 0])
        second = os_ & ~first & np.isnan(out[:, 1])
        out[first, 0] = m[first]
        out[second, 1] = m[second]
    return out


def flags_of(a, missing, cfg):
    n = len(a)
    ones = np.ones(n, bool)
    f = {}
    # event trigger
    if f"Trigger_{HLT_REQUIRED}" in missing or all(f"Trigger_{s}" in missing for s in L1_SEEDS):
        f["hlt"] = ones.copy()
    else:
        l1 = np.zeros(n, bool)
        for s in L1_SEEDS:
            if f"Trigger_{s}" not in missing:
                l1 |= np_(a, f"Trigger_{s}", 0) > 0.5
        f["hlt"] = l1 & (np_(a, f"Trigger_{HLT_REQUIRED}", 0) > 0.5)
    pt, aeta = np_(a, "mu_pt"), np.abs(np_(a, "mu_eta"))
    f["acc"] = (((aeta <= 1.2) & (pt > 3.5)) | ((aeta > 1.2) & (aeta < 2.4) & (pt > 2.0))).all(1)
    f["muid"] = ((np_(a, "mu_isGlobal", 0) > 0.5) & (np_(a, "mu_isMedium", 0) > 0.5)).all(1) \
        if "mu_isMedium" not in missing else (np_(a, "mu_isGlobal", 0) > 0.5).all(1)
    q = np.abs(np_(a, "cand_charge", 0))
    f["charge"] = q == 1          # the SS3MU role uses |Q| = 3, see below
    dr = np.stack([np_(a, f"cand_dr{p}", np.inf) for p in ("12", "13", "23")], 1) if "cand_dr12" not in missing else None
    dz = np.stack([np.abs(np_(a, f"cand_dz{p}", np.inf)) for p in ("12", "13", "23")], 1) if "cand_dz12" not in missing else None
    f["drdz"] = (ones if dr is None else (dr < cfg["dr_max"]).all(1)) & (ones if dz is None else (dz < cfg["dz_max"]).all(1))
    f["fsig"] = np_(a, "cand_flightDistSig", -1) >= cfg["fsig_min"]
    m = np_(a, "cand_mass", 0)
    f["mass"] = (m >= 1.62) & (m <= 2.0)
    if "mu_trgDR" in missing:
        f["trgmatch"] = ones.copy()
    else:
        path = np.zeros((n, 3), bool)
        for p in HLT_MATCH:
            if "mu_" + p not in missing:
                path |= np_(a, "mu_" + p, 0) > 0.5
        f["trgmatch"] = (path & (np.abs(np_(a, "mu_trgDR", np.inf)) < 0.03)
                         & (np.abs(np_(a, "mu_trgDPT", np.inf)) < 0.1)).all(1)
    return f, q


def process(args):
    path, sample_id, wanted, cfg = args
    try:
        with uproot.open(path) as fh:
            t = fh["Events"]
            k = t.arrays(["run", "event", "cand_idx"], library="np")
            key = keyof(k["run"], k["event"], k["cand_idx"])
            sel_cand = np.isin(key, wanted)
            if not sel_cand.any():
                return None
            # all candidates of the events that hold a wanted candidate take part in the best choice
            ev = evkey(k["run"], k["event"])
            sel = np.isin(ev, np.unique(ev[sel_cand]))
            names, missing = branches(t.keys())
            a = t.arrays(names)[sel]
    except Exception as e:
        return f"{path}: {type(e).__name__}: {str(e)[:150]}"
    f, q = flags_of(a, set(missing), cfg)
    base_core = f["hlt"] & f["acc"] & f["muid"] & f["drdz"] & f["fsig"] & f["mass"] & f["trgmatch"]
    f["base"] = base_core & (q == 1)
    base_ss = base_core & (q == 3)
    # best candidate: lowest chi2 among base candidates of the event, separately for |Q| = 1 and |Q| = 3
    run, event, cidx = (ak.to_numpy(a[b]) for b in ("run", "event", "cand_idx"))
    ev = evkey(run, event)
    chi2 = np_(a, "cand_chi2", np.inf)
    best = np.zeros(len(a), bool)
    for passing in (f["base"], base_ss):
        idx = np.flatnonzero(passing)
        if len(idx):
            o = idx[np.lexsort((chi2[idx], ev[idx]))]
            first = np.r_[True, ev[o][1:] != ev[o][:-1]]
            best[o[first]] = True
    f["best"] = best
    f["base_ss3mu"] = base_ss
    # how many candidates of the event are not in the file (cannot compete for best)
    _, inv, cnt = np.unique(ev, return_inverse=True, return_counts=True)
    n_missing = ak.to_numpy(a["nCand"]).astype(np.int64) - cnt[inv] if "nCand" in a.fields else np.zeros(len(a), np.int64)
    keep = np.isin(keyof(run, event, cidx), wanted)
    res = {k: v[keep] for k, v in f.items()}
    mos = os_masses(a)[keep]
    res.update(m_os1=mos[:, 0].astype(np.float32), m_os2=mos[:, 1].astype(np.float32),
               chi2=chi2[keep].astype(np.float32), n_missing_in_event=n_missing[keep].astype(np.int16),
               run=run[keep], event=event[keep], cand_idx=cidx[keep],
               sample=np.full(int(keep.sum()), sample_id, np.int16))
    return res, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-set", required=True)
    ap.add_argument("--flat-dir", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--dr-max", type=float, default=0.6)
    ap.add_argument("--dz-max", type=float, default=2.5)
    ap.add_argument("--fsig-min", type=float, default=3.5)
    ap.add_argument("--allow-missing", action="store_true",
                    help="skip cuts whose branches are missing instead of stopping (for tests)")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    cfg = dict(dr_max=args.dr_max, dz_max=args.dz_max, fsig_min=args.fsig_min)

    stats = D.load_stats(args.train_set)
    meta = D.load_meta(args.train_set)
    id_to_name = {v: k for k, v in stats["samples"].items()}
    tasks = []
    for sid in np.unique(meta["sample"]):
        name = id_to_name[int(sid)]
        m = meta["sample"] == sid
        wanted = np.unique(keyof(meta["run"][m], meta["event"][m], meta["cand_idx"][m]))
        files = sorted(glob.glob(os.path.join(args.flat_dir, "Tau3Mu", name, "*_fold*.root")))
        tasks += [(f, int(sid), wanted, cfg) for f in files]
    print(f"{len(meta['role'])} candidates to find in {len(tasks)} files, {args.jobs} processes")

    parts, errors, missing_all, t0 = [], [], set(), time.time()
    with mp.Pool(args.jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(process, tasks, chunksize=4)):
            if isinstance(r, str):
                errors.append(r)
            elif r is not None:
                parts.append(r[0])
                missing_all |= set(r[1])
            if i % 200 == 0:
                n = sum(len(p["run"]) for p in parts)
                print(f"  {i + 1}/{len(tasks)} files, {n} candidates found, {time.time() - t0:.0f} s")
    for e in errors:
        print("  unreadable:", e)
    if missing_all:
        print("branches missing in the flattened files:", sorted(missing_all))
        print("  -> the cuts that need them are NOT applied")
        if not args.allow_missing:
            sys.exit("stopping: rerun with --allow-missing to accept this")
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    n_found = len(out["run"])
    print(f"found {n_found} of {len(meta['role'])} training-set candidates")

    # cut flow per role, with the role taken from the training set
    idx = {k: i for i, k in enumerate(keyof(meta["run"], meta["event"], meta["cand_idx"]) ^
                                     (meta["sample"].astype(np.uint64) << np.uint64(58)))}
    rk = keyof(out["run"], out["event"], out["cand_idx"]) ^ (out["sample"].astype(np.uint64) << np.uint64(58))
    pos = np.array([idx.get(k, -1) for k in rk])
    role = np.where(pos >= 0, meta["role"][np.maximum(pos, 0)], -1)
    flow = {}
    for r, rn in enumerate(["signal", "sidebands", "ss3mu"]):
        m = role == r
        cum = np.ones(m.sum(), bool)
        flow[rn] = {}
        for fl in ["hlt", "acc", "muid", "drdz", "fsig", "mass", "trgmatch"]:
            cum &= out[fl][m]
            flow[rn][fl] = float(cum.mean()) if m.any() else float("nan")
        flow[rn]["base"] = float((out["base_ss3mu"] if r == 2 else out["base"])[m].mean()) if m.any() else float("nan")
        flow[rn]["best"] = float(out["best"][m].mean()) if m.any() else float("nan")
        flow[rn]["events_with_missing_candidates"] = float((out["n_missing_in_event"][m] > 0).mean()) if m.any() else float("nan")
        print(f"  {rn:10s} " + "  ".join(f"{k} {v:.3f}" for k, v in flow[rn].items()))
    path = os.path.join(args.train_set, "basecuts.npz")
    np.savez(path, **out)
    with open(os.path.join(args.train_set, "basecuts.json"), "w") as fh:
        json.dump({"cuts": cfg, "found": n_found, "total": int(len(meta["role"])), "missing_branches": sorted(missing_all),
                   "cumulative_efficiency": flow}, fh, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
