#!/usr/bin/env python3
"""Synthetic flattened files (same layout as flatten_for_transformer.py) to test the training
chain without the real ntuples: python3 make_synthetic_flat.py <out-dir> [--n 4000]

Signal has a slightly displaced, isolated topology; background is prompt-ish with more
activity. A small mass dependence is put in the background isolation, to see DisCo at work.
"""
import argparse
import os

import awkward as ak
import numpy as np
import uproot


def jag(rng, counts, gen):
    flat = gen(int(counts.sum()))
    return ak.unflatten(flat.astype(np.float32), counts)


def make(rng, n, kind):
    sig = kind == "sig"
    is_data = kind != "sig"
    out = {}
    out["run"] = rng.integers(379000, 380000, n).astype(np.int32)
    out["luminosityBlock"] = rng.integers(1, 500, n).astype(np.int32)
    out["event"] = rng.integers(1, 10**9, n).astype(np.int64)
    out["nCand"] = np.ones(n, np.int32)
    out["cand_idx"] = np.zeros(n, np.int32)
    if sig:
        mass = rng.normal(1.777, 0.018, n)
    else:
        mass = rng.uniform(1.40, 2.20, n)
        if kind == "bkg":
            mass = mass[(mass < 1.74) | (mass > 1.82)]
            mass = np.concatenate([mass, rng.uniform(1.40, 1.74, n - len(mass))])
    pt = rng.gamma(3.0, 4.0, n) + 5
    out["cand_pt"] = pt.astype(np.float32)
    out["cand_eta"] = rng.uniform(-2.4, 2.4, n).astype(np.float32)
    out["cand_phi"] = rng.uniform(-np.pi, np.pi, n).astype(np.float32)
    out["cand_mass"] = mass.astype(np.float32)
    out["cand_kin_mass"] = (mass + rng.normal(0, 0.003, n)).astype(np.float32)
    out["cand_kin_ok"] = np.ones(n, np.float32)
    out["cand_charge"] = np.where(kind == "ss", 3, 1) * rng.choice([-1, 1], n)
    out["cand_channel"] = np.full(n, 2 if kind == "ss" else 0, np.int32)
    lsig = rng.exponential(8 if sig else 3, n)
    out["cand_flightDistSig"] = lsig.astype(np.float32)
    out["cand_flightDistXYSig"] = (lsig * 0.9).astype(np.float32)
    out["cand_cosPointingAngle"] = (1 - rng.exponential(1e-4 if sig else 1e-2, n)).astype(np.float32)
    for k in ["prob", "kin_prob", "vprob12", "vprob13", "vprob23", "diMuVtxFit_bestProb"]:
        out[f"cand_{k}"] = rng.uniform(0.0 if not sig else 0.1, 1, n).astype(np.float32)
    iso = rng.exponential(0.3 if sig else 1.0, n) * (1 + (0 if sig else 0.5 * (mass - 1.4)))
    out["cand_rel_iso"] = iso.astype(np.float32)
    out["cand_iso_trk_dR03"] = (iso * pt * 0.7).astype(np.float32)
    out["cand_iso_trk_dR05"] = (iso * pt).astype(np.float32)
    for k in ["d0_sig", "d0max_sig", "dr_max", "dz_max", "mindca_iso", "n_trk_vtxcompat"]:
        out[f"cand_{k}"] = rng.exponential(1.0, n).astype(np.float32)
    ntrk = rng.poisson(15 if sig else 30, n)
    out["cand_nTrk_total"] = ntrk.astype(np.float32)
    out["cand_nNeu_total"] = rng.poisson(8, n).astype(np.float32)
    out["cand_nSV_total"] = rng.poisson(1, n).astype(np.float32)
    out["fold"] = rng.integers(0, 5, n).astype(np.int32)
    out["isData"] = np.full(n, int(is_data), np.int32)
    out["label_sig"] = np.full(n, int(sig), np.int32)
    out["sig_prod"] = np.where(sig, rng.choice([1, 2, 3], n, p=[0.7, 0.1, 0.2]), 0).astype(np.int32)
    out["nPVtx"] = rng.poisson(35 if sig else 40, n).astype(np.int32)
    if sig:
        out["Generator_weight"] = np.ones(n, np.float32)

    mu = {}
    c3 = np.full(n, 3)
    for f in ["pt", "eta", "phi", "charge", "isGlobal", "isTracker", "isPF", "isMedium", "isSoft", "isStandalone",
              "softMva", "softMvaRun3", "mvaId", "segmComp", "caloComp", "trkKink", "glbKink", "chi2LocalPosition",
              "chi2LocalMomentum", "innerTrk_normChi2", "innerTrk_ValidFraction", "GLnormChi2", "nValidPixelHits",
              "trkLayersWMeas", "nMatchedStations", "timeAtIpInOut", "ip3d_pv_sig", "ip3d_sv_sig"]:
        mu[f] = jag(rng, c3, lambda m: rng.uniform(0, 1, m))
    mu["pt"] = ak.unflatten((np.repeat(pt, 3) * rng.dirichlet([2, 2, 2], n).ravel()).astype(np.float32), c3)
    mu["eta"] = ak.unflatten((np.repeat(out["cand_eta"], 3) + rng.normal(0, 0.1, 3 * n)).astype(np.float32), c3)
    mu["phi"] = ak.unflatten((np.repeat(out["cand_phi"], 3) + rng.normal(0, 0.1, 3 * n)).astype(np.float32), c3)

    nt = np.minimum(ntrk, 128)
    trk = {f: jag(rng, nt, lambda m: rng.uniform(0, 1, m)) for f in [
        "charge", "isLostTrack", "highPurity", "hasTrackDetails", "inCone", "inCylinder", "dxy", "dz", "nPixelHits",
        "nHits", "lostInnerHits", "normChi2", "fromPV", "pvAssocQuality", "puppiWeight", "hcalFraction", "isMuon",
        "isLooseMuon", "muon_softMva", "svIdx", "sv_dchi2", "pair_leg1_dca", "pair_leg2_dca", "pair_leg3_dca",
        "pair_leg1_vprob", "pair_leg2_vprob", "pair_leg3_vprob"]}
    trk["pt"] = jag(rng, nt, lambda m: rng.exponential(1.5, m) + 0.5)
    trk["eta"] = jag(rng, nt, lambda m: rng.uniform(-2.5, 2.5, m))
    trk["deta"] = jag(rng, nt, lambda m: rng.normal(0, 0.5, m))
    trk["dphi"] = jag(rng, nt, lambda m: rng.normal(0, 0.5, m))
    trk["dR"] = np.sqrt(trk["deta"] ** 2 + trk["dphi"] ** 2)
    trk["sv_dchi2"] = ak.where(trk["sv_dchi2"] > 0.5, trk["sv_dchi2"] * 20, -999.0)
    for f in ["ip3d_sv", "ip3d_sv_sig", "ip3d_pv", "ip3d_pv_sig", "dcaFlight"]:
        trk[f] = jag(rng, nt, lambda m: rng.exponential(0.05 if "sig" not in f else 3.0, m))

    nn_ = rng.poisson(6, n)
    neu = {f: jag(rng, nn_, lambda m: rng.uniform(0, 1, m)) for f in [
        "inCone", "inFlightCone", "puppiWeight", "hcalFraction"]}
    neu["pt"] = jag(rng, nn_, lambda m: rng.exponential(1.0, m) + 0.3)
    neu["pdgId"] = jag(rng, nn_, lambda m: rng.choice([22, 130], m))
    for f in ["deta", "dphi"]:
        neu[f] = jag(rng, nn_, lambda m: rng.normal(0, 0.3, m))
    neu["dR"] = np.sqrt(neu["deta"] ** 2 + neu["dphi"] ** 2)
    neu["dR_flight"] = neu["dR"] * 1.1

    ns = rng.poisson(1, n)
    sv = {f: jag(rng, ns, lambda m: rng.exponential(1.0, m)) for f in [
        "mass", "pt", "ntracks", "chi2", "ndof", "dlen", "dlenSig", "dxy", "dxySig", "distToCandSV", "distToCandSVSig"]}
    sv["eta"] = jag(rng, ns, lambda m: rng.uniform(-2.4, 2.4, m))
    sv["phi"] = jag(rng, ns, lambda m: rng.uniform(-np.pi, np.pi, m))
    sv["cosPAngle"] = jag(rng, ns, lambda m: rng.uniform(0.9, 1, m))
    sv["sharesCandTrack"] = jag(rng, ns, lambda m: rng.integers(0, 2, m).astype(float))

    nj = rng.poisson(2, n)
    jet = {f: jag(rng, nj, lambda m: rng.uniform(0, 1, m)) for f in ["btagPNetB", "btagUParTB"]}
    jet["pt"] = jag(rng, nj, lambda m: rng.exponential(20, m) + 15)
    jet["eta"] = jag(rng, nj, lambda m: rng.uniform(-2.4, 2.4, m))
    jet["phi"] = jag(rng, nj, lambda m: rng.uniform(-np.pi, np.pi, m))
    jet["mass"] = jag(rng, nj, lambda m: rng.uniform(1, 10, m))
    jet["nConstituents"] = jag(rng, nj, lambda m: rng.integers(2, 40, m).astype(float))

    for name, d in (("mu", mu), ("trk", trk), ("neu", neu), ("sv", sv), ("jet", jet)):
        out[name] = ak.zip({k: ak.values_astype(v, np.float32) for k, v in d.items()})
    return out


def write(path, out):
    with uproot.recreate(path) as f:
        f.mktree("Events", {k: (v.type if isinstance(v, ak.Array) else v.dtype) for k, v in out.items()},
                 field_name=lambda outer, inner: f"{outer}_{inner}", counter_name=lambda c: f"n{c}")
        f["Events"].extend(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=4000)
    a = ap.parse_args()
    rng = np.random.default_rng(1)
    layout = {"2024_MC_Ds": "sig", "2024_MC_Bd": "sig", "2024eraC_stream0": "data", "2024eraD_stream1": "data"}
    for sample, kind in layout.items():
        d = os.path.join(a.out, "Tau3Mu", sample)
        os.makedirs(d, exist_ok=True)
        for part in range(2):
            if kind == "sig":
                write(os.path.join(d, f"{sample}_part{part}_fold0.root"), make(rng, a.n, "sig"))
            else:
                bk, ss = make(rng, a.n, "bkg"), make(rng, a.n // 4, "ss")
                write(os.path.join(d, f"{sample}_part{part}_fold0.root"),
                      {k: (ak.concatenate([bk[k], ss[k]]) if isinstance(bk[k], ak.Array) else np.concatenate([bk[k], ss[k]]))
                       for k in bk})
    print("synthetic flattened files in", a.out)


if __name__ == "__main__":
    main()
