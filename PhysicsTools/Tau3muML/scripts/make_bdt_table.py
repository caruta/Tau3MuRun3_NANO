#!/usr/bin/env python3
"""Run 2 HF BDT input variables for the candidates of a training set (run at Bari, inside cmsenv).

Variables of the Run 2 three-global-muon BDT (CMS AN, section 11.1; JHEP 2024, arXiv:2312.02371),
computed from the flattened ntuples:

  Pt_mu3         lowest muon pT
  cLP_max        max chi2LocalPosition of the three muons
  tKink_max      max trkKink
  segmComp_min   min segmentCompatibility
  mvaId_mu1/2/3  muon ID MVA per muon, pT-ordered. Run 2 used a dedicated global-muon BDT;
                 softMvaRun3 stands in for it
  fv_nC          3mu vertex chi2 / ndof
  fv_dphi3D      angle between the 3mu momentum and the PV->SV vector
  fv_d3Dsig      3D flight-distance significance
  d0sig_min/max  min / max transverse IP significance of the muons wrt the PV
  mindca_iso     min distance of a track with pT > 1 GeV to the 3mu vertex (from the track tokens,
                 which keep the tracks closest to the SV first)
  trkrel         max over muons of sum pT(tracks: pT > 1 GeV, dR(track, mu) < 0.3, DCA(track, mu) < 1 mm) / pT(mu)

plus for categorisation: sigma_m/m (kinematic fit) and whether all three muons are global.

Only the candidates in the training set are kept (matched on sample, run, event, cand_idx), so the
BDT can be compared with the transformer on exactly the same candidates. Output: <train-set>/bdt_run2.npz

  python3 make_bdt_table.py --train-set /lustre/.../train_Transformer_v1 \\
      --flat-dir /lustre/.../flat_Transformer_v1 --jobs 8
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

BRANCHES = ["run", "event", "cand_idx", "cand_channel", "isData",
            "cand_chi2", "cand_ndof", "cand_pointingAngle", "cand_cosPointingAngle", "cand_flightDistSig",
            "cand_d0_sig", "cand_d0max_sig", "cand_kin_ok", "cand_kin_mass", "cand_kin_massErr", "cand_mass",
            "mu_pt", "mu_eta", "mu_phi", "mu_chi2LocalPosition", "mu_trkKink", "mu_segmComp", "mu_softMvaRun3",
            "mu_isGlobal", "trk_pt", "trk_eta", "trk_phi", "trk_ip3d_sv",
            "trk_pair_leg1_dca", "trk_pair_leg2_dca", "trk_pair_leg3_dca"]
VARS = ["Pt_mu3", "cLP_max", "tKink_max", "segmComp_min", "mvaId_mu1", "mvaId_mu2", "mvaId_mu3", "fv_nC",
        "fv_dphi3D", "fv_d3Dsig", "d0sig_min", "d0sig_max", "mindca_iso", "trkrel"]
EXTRA = ["massres", "is3glb"]


def keyof(run, event, cidx):
    return (run.astype(np.uint64) << np.uint64(44)) | (event.astype(np.uint64) << np.uint64(10)) | cidx.astype(np.uint64)


def clean(x, fill=np.nan):
    if x is None:
        return np.nan
    x = np.asarray(x, dtype=np.float64)
    return np.where((x <= -998) | ~np.isfinite(x), fill, x)


def compute(a):
    def get(name):
        return a[name] if name in a.fields else None

    mu_pt = ak.to_numpy(a["mu_pt"]).astype(np.float64)          # (N, 3), legs as in the candidate
    order = np.argsort(-mu_pt, axis=1)

    def mu(name):
        return clean(ak.to_numpy(a[name]))

    mva = np.take_along_axis(mu("mu_softMvaRun3"), order, axis=1)
    out = {
        "Pt_mu3": mu_pt.min(1),
        "cLP_max": np.nanmax(mu("mu_chi2LocalPosition"), 1),
        "tKink_max": np.nanmax(mu("mu_trkKink"), 1),
        "segmComp_min": np.nanmin(mu("mu_segmComp"), 1),
        "mvaId_mu1": mva[:, 0], "mvaId_mu2": mva[:, 1], "mvaId_mu3": mva[:, 2],
        "fv_nC": clean(get("cand_chi2")) / np.maximum(clean(get("cand_ndof"), 1.0), 1e-3) * np.ones(len(a)),
        "fv_dphi3D": (clean(a["cand_pointingAngle"]) if "cand_pointingAngle" in a.fields
                      else np.arccos(np.clip(clean(a["cand_cosPointingAngle"]), -1, 1))),
        "fv_d3Dsig": clean(get("cand_flightDistSig")),
        "d0sig_min": clean(get("cand_d0_sig")),
        "d0sig_max": clean(get("cand_d0max_sig")),
    }
    # isolation from the track tokens
    tpt = a["trk_pt"]
    hi = tpt > 1.0
    ipsv = ak.where(a["trk_ip3d_sv"] > -998, a["trk_ip3d_sv"], 999.0)
    out["mindca_iso"] = ak.to_numpy(ak.fill_none(ak.min(ak.where(hi, ipsv, 999.0), axis=1), 999.0)).astype(np.float64)
    rel = np.zeros((len(a), 3))
    for i in range(3):
        meta_eta, meta_phi = a["mu_eta"][:, i], a["mu_phi"][:, i]
        dphi = (a["trk_phi"] - meta_phi + np.pi) % (2 * np.pi) - np.pi
        dr = np.sqrt((a["trk_eta"] - meta_eta) ** 2 + dphi ** 2)
        dca = a[f"trk_pair_leg{i + 1}_dca"]
        sel = hi & (dr < 0.3) & (dca > -998) & (dca < 0.1)
        rel[:, i] = ak.to_numpy(ak.sum(ak.where(sel, tpt, 0.0), axis=1)) / np.maximum(mu_pt[:, i], 1e-3)
    out["trkrel"] = rel.max(1)
    km = clean(get("cand_kin_mass")) * np.ones(len(a))
    ke = clean(get("cand_kin_massErr")) * np.ones(len(a))
    ok = (ak.to_numpy(a["cand_kin_ok"]) > 0) & np.isfinite(ke) & (km > 0)
    out["massres"] = np.where(ok, ke / np.where(ok, km, 1.0), np.nan)
    out["is3glb"] = (ak.to_numpy(a["mu_isGlobal"]) > 0.5).all(1).astype(np.float64)
    return out


def role_of(a):
    ch = ak.to_numpy(a["cand_channel"])
    data = ak.to_numpy(a["isData"]) == 1
    return np.where(~data, 0, np.where(ch == 2, 2, 1))


def process(args):
    path, sample_id, wanted = args
    try:
        with uproot.open(path) as fh:
            t = fh["Events"]
            k = t.arrays(["run", "event", "cand_idx"], library="np")
            key = keyof(k["run"], k["event"], k["cand_idx"])
            sel = np.isin(key, wanted)
            if not sel.any():
                return None
            names = [b for b in BRANCHES if b in t.keys()]
            a = t.arrays(names)[sel]
    except Exception as e:
        return f"{path}: {type(e).__name__}: {str(e)[:150]}"
    v = compute(a)
    res = {n: v[n].astype(np.float32) for n in VARS + EXTRA}
    res.update(run=ak.to_numpy(a["run"]), event=ak.to_numpy(a["event"]), cand_idx=ak.to_numpy(a["cand_idx"]),
               sample=np.full(len(a), sample_id, np.int16), role=role_of(a).astype(np.int8))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-set", required=True)
    ap.add_argument("--flat-dir", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    stats = D.load_stats(args.train_set)
    meta = D.load_meta(args.train_set)
    id_to_name = {v: k for k, v in stats["samples"].items()}
    tasks = []
    for sid in np.unique(meta["sample"]):
        name = id_to_name[int(sid)]
        m = meta["sample"] == sid
        wanted = np.unique(keyof(meta["run"][m], meta["event"][m], meta["cand_idx"][m]))
        files = sorted(glob.glob(os.path.join(args.flat_dir, "Tau3Mu", name, "*_fold*.root")))
        tasks += [(f, int(sid), wanted) for f in files]
    print(f"{len(meta['role'])} candidates to find in {len(tasks)} files, {args.jobs} processes")

    parts, errors, t0 = [], [], time.time()
    with mp.Pool(args.jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(process, tasks, chunksize=4)):
            if isinstance(r, str):
                errors.append(r)
            elif r is not None:
                parts.append(r)
            if i % 200 == 0:
                n = sum(len(p["run"]) for p in parts)
                print(f"  {i + 1}/{len(tasks)} files, {n} candidates found, {time.time() - t0:.0f} s")
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    # one row per training-set candidate (same sample/run/event/cand_idx can appear once per role)
    n_found = len(out["run"])
    print(f"found {n_found} of {len(meta['role'])} training-set candidates")
    for e in errors:
        print("  unreadable:", e)
    for v in VARS + EXTRA:
        x = out[v]
        print(f"  {v:14s} median {np.nanmedian(x):10.4g}  missing {np.mean(~np.isfinite(x)):.1%}")
    path = os.path.join(args.train_set, "bdt_run2.npz")
    np.savez(path, **out, variables=np.array(VARS))
    with open(os.path.join(args.train_set, "bdt_run2.json"), "w") as fh:
        json.dump({"variables": VARS, "found": n_found, "total": int(len(meta["role"]))}, fh, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
