#!/usr/bin/env python3
"""Flatten Tau3muNANO output to one row per candidate, for transformer training.

The CMSSW step writes per-event tables (NanoAOD); this script regroups them into one
row per candidate with jagged token collections:

  cand_*   candidate-level features (all <Cand>_* columns, incl. the context extension)
  mu_*     the 3 candidate legs (Muon table joined through Muon_srcIdx; for DsPhiPi the
           third leg comes from the Track table), plus leg-level candidate columns
           (refit_<leg>_*, <leg>_ip3d_*) moved onto the leg
  trk_*    charged-track tokens   (<Cand>Trk table, rows grouped by candIdx)
  neu_*    neutral tokens         (<Cand>Neu table)
  sv_*     IVF secondary vertices (<Cand>SV table)
  jet_*    event jets             (CtxJet table, copied to every candidate of the event)

plus bookkeeping (run, luminosityBlock, event, cand_idx, nCand, fold), event-level
trigger bits, Generator_weight / Pileup_nTrueInt (MC), and MC labels.

Data candidates in the signal region are dropped here (blinding), for |Q|=1 TAU3MU
candidates only. Output is split by fold = hash(run, event) mod --nfolds.

Example:
  python3 flatten_for_transformer.py --channel Tau3Mu --data -o out/data2024 nano_*.root
  python3 flatten_for_transformer.py --channel Tau3Mu -o out/sig_Ds tau3mu_output_MC.root
"""
import argparse
import os

import awkward as ak
import numpy as np
import uproot

CHANNELS = {
    "Tau3Mu": dict(cand="Tau3Mu", legs=[("mu1", "mu1_idx", False), ("mu2", "mu2_idx", False), ("mu3", "mu3_idx", False)]),
    "DsPhiPi": dict(cand="Cand2MuTrk", legs=[("mu1", "mu1_idx", False), ("mu2", "mu2_idx", False), ("tr", "tr_idx", True)]),
}
MISSING = -999.0
EVENT_BRANCHES = ["Generator_weight", "Pileup_nTrueInt", "nPVtx"]

# genFlags bits written by MuonMatchEmbedder
FROM_TAU, FROM_DS, FROM_DPLUS, FROM_B, FROM_W, FROM_PHI = 1, 2, 4, 8, 16, 32
# sig_prod codes
PROD_UNKNOWN, PROD_DS, PROD_B_DS, PROD_B_TAU, PROD_DPLUS, PROD_W = 0, 1, 2, 3, 4, 5


def collection(arrays, prefix):
    """{field: jagged array} for a NanoAOD collection (branches named <prefix>_<field>)."""
    pre = prefix + "_"
    return {k[len(pre):]: arrays[k] for k in arrays.fields if k.startswith(pre)}


def as_float(a):
    return ak.values_astype(a, np.float32)


def fold_of(run, event, nfolds):
    """Deterministic fold from (run, event); all candidates of an event share it."""
    with np.errstate(over="ignore"):
        h = event.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
        h ^= run.astype(np.uint64) * np.uint64(0xBF58476D1CE4E5B9)
        h ^= h >> np.uint64(31)
    return (h % np.uint64(nfolds)).astype(np.int32)


def leg_tokens(arr, cand, ch):
    """The 3 legs as fixed-length tokens: dict field -> (ncand, 3) float32 numpy array."""
    muon = collection(arr, "Muon")
    track = collection(arr, "Track")
    if "srcIdx" not in muon:
        raise RuntimeError("Muon_srcIdx missing: the input was not produced with the transformer branch")

    leg_level = ["refit_{leg}_pt", "refit_{leg}_eta", "refit_{leg}_phi",
                 "{leg}_ip3d_pv", "{leg}_ip3d_pv_sig", "{leg}_ip3d_sv", "{leg}_ip3d_sv_sig"]
    fields = sorted((set(muon) | set(track)) - {"srcIdx"})
    per_leg = []
    for leg, idx_name, is_track in ch["legs"]:
        idx = cand[idx_name]
        if is_track:
            if not track:
                raise RuntimeError("the DsPhiPi track leg needs the Track table")
            ntrk = ak.num(track["pt"])
            valid = (idx >= 0) & (idx < ntrk)
            src, pos = track, ak.mask(idx, valid)
        else:
            match = idx[:, :, None] == muon["srcIdx"][:, None, :]
            src, pos = muon, ak.mask(ak.argmax(match, axis=2), ak.any(match, axis=2))
        feats = {}
        for f in fields:
            if f in src:
                feats[f] = ak.to_numpy(ak.flatten(ak.fill_none(as_float(src[f][pos]), MISSING)))
            else:
                feats[f] = None  # filled below once the size is known
        for tmpl in leg_level:
            name = tmpl.format(leg=leg)
            if name in cand:
                # refit_mu1_eta -> refit_eta, mu1_ip3d_pv -> ip3d_pv
                generic = tmpl.format(leg="").replace("__", "_").strip("_")
                feats[generic] = ak.to_numpy(ak.flatten(as_float(cand[name])))
        n = len(ak.flatten(idx))
        for f, v in feats.items():
            if v is None:
                feats[f] = np.full(n, MISSING, dtype=np.float32)
        feats["isTrackLeg"] = np.full(n, float(is_track), dtype=np.float32)
        per_leg.append(feats)
    names = sorted(set().union(*per_leg))
    return {f: np.stack([leg.get(f, np.full(len(per_leg[0]["isTrackLeg"]), MISSING, np.float32)) for leg in per_leg], axis=1)
            for f in names}


def grouped_tokens(arr, name, ncand):
    """Token table rows regrouped per candidate: dict field -> jagged (ncand_total, var)."""
    tok = collection(arr, name)
    total = int(ak.sum(ncand))
    if not tok:
        return {}
    first_cand = np.concatenate([[0], np.cumsum(ak.to_numpy(ncand))[:-1]]).astype(np.int64)
    gid = ak.to_numpy(ak.flatten(tok["candIdx"] + first_cand))
    if len(gid) and np.any(np.diff(gid) < 0):
        raise RuntimeError(f"{name} rows are not ordered by candidate")
    counts = np.bincount(gid, minlength=total) if len(gid) else np.zeros(total, dtype=np.int64)
    return {f: ak.unflatten(ak.to_numpy(ak.flatten(as_float(v))), counts)
            for f, v in tok.items() if f != "candIdx"}


def mc_labels(legs, channel):
    """label_sig (TAU3MU: all 3 legs from the same gen tau), sig_prod, label_dsphipi."""
    out = {}
    if "genTauIdx" not in legs:
        return out
    tau = legs["genTauIdx"]
    flags = legs["genFlags"].astype(np.int64)
    if channel == "Tau3Mu":
        out["label_sig"] = ((tau[:, 0] >= 0) & (tau[:, 0] == tau[:, 1]) & (tau[:, 0] == tau[:, 2])).astype(np.int32)
        f = flags[:, 0]
        prod = np.full(len(f), PROD_UNKNOWN, dtype=np.int32)
        prod = np.where(f & FROM_B, PROD_B_TAU, prod)
        prod = np.where(f & FROM_DS, PROD_DS, prod)
        prod = np.where(f & FROM_DPLUS, PROD_DPLUS, prod)
        prod = np.where((f & FROM_B) & (f & FROM_DS), PROD_B_DS, prod)
        prod = np.where(f & FROM_W, PROD_W, prod)
        out["sig_prod"] = np.where(out["label_sig"] == 1, prod, PROD_UNKNOWN).astype(np.int32)
    else:
        phi_ds = (flags[:, :2] & FROM_PHI).astype(bool) & (flags[:, :2] & FROM_DS).astype(bool)
        out["label_dsphipi"] = np.all(phi_ds, axis=1).astype(np.int32)
    return out


def process_chunk(arr, channel, is_data, sr, mass_window, nfolds):
    ch = CHANNELS[channel]
    cand = collection(arr, ch["cand"])
    ncand = ak.num(cand["mass"])
    nev = len(arr)
    ev_of_cand = np.repeat(np.arange(nev), ak.to_numpy(ncand))
    if len(ev_of_cand) == 0:
        return None

    out = {}
    for b in ["run", "luminosityBlock", "event"]:
        out[b] = ak.to_numpy(arr[b])[ev_of_cand]
    for b in EVENT_BRANCHES + [f for f in arr.fields if f.startswith("Trigger_")]:
        if b in arr.fields:
            out[b] = ak.to_numpy(arr[b])[ev_of_cand]
    out["nCand"] = ak.to_numpy(ncand)[ev_of_cand].astype(np.int32)
    out["cand_idx"] = ak.to_numpy(ak.flatten(ak.local_index(cand["mass"]))).astype(np.int32)
    for k, v in cand.items():
        out["cand_" + k] = ak.to_numpy(ak.flatten(v))
    out["fold"] = fold_of(out["run"], out["event"], nfolds)
    out["isData"] = np.full(len(ev_of_cand), int(is_data), dtype=np.int32)

    legs = leg_tokens(arr, cand, ch)
    if not is_data:
        out.update(mc_labels(legs, channel))
    elif channel == "Tau3Mu":
        # same columns in data and MC, so one training config reads both
        out["label_sig"] = np.zeros(len(ev_of_cand), dtype=np.int32)
        out["sig_prod"] = np.zeros(len(ev_of_cand), dtype=np.int32)

    tokens = {
        "mu": {f: ak.from_regular(v) for f, v in legs.items()},
        "trk": grouped_tokens(arr, ch["cand"] + "Trk", ncand),
        "neu": grouped_tokens(arr, ch["cand"] + "Neu", ncand),
        "sv": grouped_tokens(arr, ch["cand"] + "SV", ncand),
    }
    jets = collection(arr, "CtxJet")
    if jets:
        tokens["jet"] = {f: as_float(v)[ev_of_cand] for f, v in jets.items()}

    # selection: mass window and blinding (data, |Q|=1 TAU3MU only)
    mass = out["cand_mass"]
    keep = (mass >= mass_window[0]) & (mass <= mass_window[1])
    if is_data and channel == "Tau3Mu":
        in_sr = (mass >= sr[0]) & (mass <= sr[1])
        if "cand_kin_mass" in out:
            km = out["cand_kin_mass"]
            in_sr |= (km >= sr[0]) & (km <= sr[1])
        is_tau3mu = out["cand_channel"] == 0 if "cand_channel" in out else np.abs(out["cand_charge"]) == 1
        keep &= ~(in_sr & is_tau3mu)

    flat = {k: v[keep] for k, v in out.items()}
    for name, fields in tokens.items():
        if fields:
            flat[name] = ak.zip({f: v[keep] for f, v in fields.items()})
    return flat


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="Tau3muNANO output files")
    ap.add_argument("-o", "--output", required=True, help="output prefix; writes <prefix>_fold<k>.root")
    ap.add_argument("--channel", choices=sorted(CHANNELS), default="Tau3Mu")
    ap.add_argument("--data", action="store_true", help="input is data: apply blinding, no MC labels")
    ap.add_argument("--sr", nargs=2, type=float, default=[1.74, 1.82], help="blinded signal region [GeV]")
    ap.add_argument("--mass-window", nargs=2, type=float, default=[1.40, 2.20], help="kept 3-body mass range [GeV]")
    ap.add_argument("--nfolds", type=int, default=5)
    ap.add_argument("--step-size", default="200 MB")
    args = ap.parse_args()

    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    files, ncands = {}, 0
    try:
        for chunk in uproot.iterate([f + ":Events" for f in args.inputs], step_size=args.step_size):
            flat = process_chunk(chunk, args.channel, args.data, args.sr, args.mass_window, args.nfolds)
            if flat is None:
                continue
            for k in range(args.nfolds):
                sel = flat["fold"] == k
                if not np.any(sel):
                    continue
                part = {name: v[sel] for name, v in flat.items()}
                if k not in files:
                    files[k] = uproot.recreate(f"{args.output}_fold{k}.root")
                    # NanoAOD-style names: trk_pt with counter ntrk (what weaver expects)
                    files[k].mktree("Events", {n: (v.type if isinstance(v, ak.Array) else v.dtype) for n, v in part.items()},
                                    field_name=lambda outer, inner: f"{outer}_{inner}",
                                    counter_name=lambda counted: f"n{counted}")
                files[k]["Events"].extend(part)
            ncands += len(flat["run"])
    finally:
        for f in files.values():
            f.close()
    print(f"wrote {ncands} candidates to {len(files)} fold files with prefix {args.output}")


if __name__ == "__main__":
    main()
