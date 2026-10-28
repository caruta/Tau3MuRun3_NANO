"""Input features for the tau -> 3mu transformer, computed from the flattened ntuples.

Mass safety: the training sees signal only near m(tau) and background only in the
sidebands, so any input that lets the network reconstruct m(3mu) teaches it "peak =
signal". The 3-body mass follows from the leg momenta, so no absolute momentum scale
enters the inputs:
  * every momentum is a ratio to the candidate momentum (no candidate pT itself),
  * no pair or 3-body masses, no ctau,
  * pair features for the attention use angles and momentum ratios only.
Residual information (e.g. through track-parameter resolutions) is handled by the
DisCo term in the loss and checked with the sculpting tests in evaluate.py.

Each token type gives two padded arrays per candidate:
  feats [n_max, n_features]   transformed features (float32); columns GEOM[t] = (deta, dphi,
                              log ptrel) also feed the pairwise attention bias
  mask  [n_max]               True for real tokens
plus one global candidate vector.
"""
import awkward as ak
import numpy as np

MISSING = -998.0  # flattener sentinel is -999

TOKEN_MAX = {"mu": 3, "trk": 128, "neu": 32, "sv": 8, "jet": 6}

# features that grow with event activity; off by default because data and MC
# differ in pileup and the background comes from data, the signal from MC
ACTIVITY_FEATURES = {"cand_nTrk_total", "cand_nNeu_total", "cand_nSV_total"}


# ---------- small helpers (work on awkward or numpy) ----------
def _clean(x, fill=0.0):
    x = ak.values_astype(x, np.float32) if isinstance(x, ak.Array) else np.asarray(x, dtype=np.float32)
    bad = (x <= MISSING) | ~np.isfinite(x)
    return ak.where(bad, fill, x) if isinstance(x, ak.Array) else np.where(bad, fill, x)


def _log1p_pos(x, scale=1.0):
    x = _clean(x)
    return np.log1p(np.maximum(x, 0.0) / scale)


def _slog(x, scale):
    x = _clean(x)
    return np.sign(x) * np.log1p(np.abs(x) / scale)


def _clip(x, lo, hi):
    return np.minimum(np.maximum(_clean(x), lo), hi)


def _dphi(a, b):
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi


def _pointing(cos):
    """-log10(1 - cos(angle)): large for well-pointing candidates."""
    c = _clip(_clean(cos, fill=-1.0), -1.0, 1.0)
    return -np.log10(1.0 - c + 1e-7)


def _flag(x):
    return ak.values_astype(_clean(x) > 0.5, np.float32) if isinstance(x, ak.Array) else (np.asarray(x) > 0.5).astype(np.float32)


# ---------- feature tables: name -> function(record) ----------
# r is an awkward record with the flattened branches (cand_*, mu_*, trk_*, ...).
def _ptrel(r, pt):
    return np.log(np.maximum(_clean(pt), 1e-3) / np.maximum(r["cand_pt"], 1e-3))


CAND_FEATURES = {
    "eta": lambda r: _clean(r.cand_eta),
    "log_Lsig": lambda r: _log1p_pos(r.cand_flightDistSig),
    "log_LxySig": lambda r: _log1p_pos(r.cand_flightDistXYSig),
    "pointing": lambda r: _pointing(r.cand_cosPointingAngle),
    "prob": lambda r: _clean(r.cand_prob),
    "kin_prob": lambda r: _clean(r.cand_kin_prob),
    "vprob12": lambda r: np.maximum(_clean(r.cand_vprob12), 0),
    "vprob13": lambda r: np.maximum(_clean(r.cand_vprob13), 0),
    "vprob23": lambda r: np.maximum(_clean(r.cand_vprob23), 0),
    "log_d0sig": lambda r: _log1p_pos(r.cand_d0_sig),
    "log_d0maxsig": lambda r: _log1p_pos(r.cand_d0max_sig),
    "dr_max": lambda r: _clean(r.cand_dr_max),
    "dz_max": lambda r: _log1p_pos(r.cand_dz_max, 0.01),
    "rel_iso": lambda r: _log1p_pos(r.cand_rel_iso),
    "iso03_rel": lambda r: _log1p_pos(_clean(r.cand_iso_trk_dR03) / np.maximum(r.cand_pt, 1e-3)),
    "iso05_rel": lambda r: _log1p_pos(_clean(r.cand_iso_trk_dR05) / np.maximum(r.cand_pt, 1e-3)),
    "mindca_iso": lambda r: _log1p_pos(np.minimum(_clean(r.cand_mindca_iso, 10.0), 10.0), 0.001),
    "n_vtxcompat": lambda r: _clean(r.cand_n_trk_vtxcompat),
    "dimu_extra_prob": lambda r: np.maximum(_clean(r.cand_diMuVtxFit_bestProb), 0),
    # activity (optional)
    "cand_nTrk_total": lambda r: np.log1p(np.maximum(_clean(r.cand_nTrk_total), 0)),
    "cand_nNeu_total": lambda r: np.log1p(np.maximum(_clean(r.cand_nNeu_total), 0)),
    "cand_nSV_total": lambda r: np.maximum(_clean(r.cand_nSV_total), 0),
}

TOKEN_FEATURES = {
    "mu": {
        "log_ptfrac": lambda r: _ptrel(r, r.mu_pt),
        "eta": lambda r: _clean(r.mu_eta),
        "deta": lambda r: _clean(r.mu_eta) - r.cand_eta,
        "dphi": lambda r: _dphi(_clean(r.mu_phi), r.cand_phi),
        "charge": lambda r: _clean(r.mu_charge),
        "isGlobal": lambda r: _flag(r.mu_isGlobal),
        "isTracker": lambda r: _flag(r.mu_isTracker),
        "isPF": lambda r: _flag(r.mu_isPF),
        "isMedium": lambda r: _flag(r.mu_isMedium),
        "isSoft": lambda r: _flag(r.mu_isSoft),
        "isStandalone": lambda r: _flag(r.mu_isStandalone),
        "softMva": lambda r: _clip(r.mu_softMva, -1, 1),
        "softMvaRun3": lambda r: _clip(r.mu_softMvaRun3, -1, 1),
        "mvaId": lambda r: _clip(r.mu_mvaId, -1, 1),
        "segmComp": lambda r: _clip(r.mu_segmComp, 0, 1),
        "caloComp": lambda r: _clip(r.mu_caloComp, 0, 1),
        "log_trkKink": lambda r: _log1p_pos(r.mu_trkKink),
        "log_glbKink": lambda r: _log1p_pos(r.mu_glbKink),
        "log_chi2LocPos": lambda r: _log1p_pos(r.mu_chi2LocalPosition),
        "log_chi2LocMom": lambda r: _log1p_pos(r.mu_chi2LocalMomentum),
        "log_innerChi2": lambda r: _log1p_pos(r.mu_innerTrk_normChi2),
        "validFrac": lambda r: _clip(r.mu_innerTrk_ValidFraction, 0, 1),
        "log_glbChi2": lambda r: _log1p_pos(r.mu_GLnormChi2),
        "nPixHits": lambda r: _clip(r.mu_nValidPixelHits, 0, 10),
        "nTrkLayers": lambda r: _clip(r.mu_trkLayersWMeas, 0, 20),
        "nStations": lambda r: _clip(r.mu_nMatchedStations, 0, 5),
        "time": lambda r: _clip(r.mu_timeAtIpInOut, -50, 50),
        "log_ip_pv_sig": lambda r: _log1p_pos(r.mu_ip3d_pv_sig),
        "log_ip_sv_sig": lambda r: _log1p_pos(r.mu_ip3d_sv_sig),
    },
    "trk": {
        "log_ptrel": lambda r: _ptrel(r, r.trk_pt),
        "deta": lambda r: _clean(r.trk_deta),
        "dphi": lambda r: _clean(r.trk_dphi),
        "log_dR": lambda r: np.log(np.maximum(_clean(r.trk_dR), 1e-4)),
        "charge": lambda r: _clean(r.trk_charge),
        "isLost": lambda r: _flag(r.trk_isLostTrack),
        "highPurity": lambda r: _flag(r.trk_highPurity),
        "hasDetails": lambda r: _flag(r.trk_hasTrackDetails),
        "inCone": lambda r: _flag(r.trk_inCone),
        "inCylinder": lambda r: _flag(r.trk_inCylinder),
        "slog_dxy": lambda r: _slog(r.trk_dxy, 0.001),
        "slog_dz": lambda r: _slog(r.trk_dz, 0.001),
        "log_ip_sv": lambda r: _log1p_pos(r.trk_ip3d_sv, 0.001),
        "log_ip_sv_sig": lambda r: _log1p_pos(r.trk_ip3d_sv_sig),
        "log_ip_pv": lambda r: _log1p_pos(r.trk_ip3d_pv, 0.001),
        "log_ip_pv_sig": lambda r: _log1p_pos(r.trk_ip3d_pv_sig),
        "log_dcaFlight": lambda r: _log1p_pos(r.trk_dcaFlight, 0.001),
        "nPixHits": lambda r: _clip(r.trk_nPixelHits, 0, 10),
        "nHits": lambda r: _clip(r.trk_nHits, 0, 40),
        "lostInner": lambda r: _clip(r.trk_lostInnerHits, -1, 5),
        "log_chi2": lambda r: _log1p_pos(r.trk_normChi2),
        "fromPV": lambda r: _clip(r.trk_fromPV, -1, 3),
        "pvQual": lambda r: _clip(r.trk_pvAssocQuality, 0, 7),
        "puppi": lambda r: _clip(r.trk_puppiWeight, 0, 1),
        "hcalFrac": lambda r: _clip(r.trk_hcalFraction, 0, 1),
        "isMuon": lambda r: _flag(r.trk_isMuon),
        "isLooseMuon": lambda r: _flag(r.trk_isLooseMuon),
        "muSoftMva": lambda r: _clip(r.trk_muon_softMva, -1, 1),
        "inSV": lambda r: ak.values_astype(_clean(r.trk_svIdx, -1) >= 0, np.float32),
        "has_dchi2": lambda r: ak.values_astype(r.trk_sv_dchi2 > MISSING, np.float32),
        "slog_dchi2": lambda r: _slog(r.trk_sv_dchi2, 1.0),
        "log_dca1": lambda r: _log1p_pos(r.trk_pair_leg1_dca, 0.001),
        "log_dca2": lambda r: _log1p_pos(r.trk_pair_leg2_dca, 0.001),
        "log_dca3": lambda r: _log1p_pos(r.trk_pair_leg3_dca, 0.001),
        "vprob1": lambda r: np.maximum(_clean(r.trk_pair_leg1_vprob), 0),
        "vprob2": lambda r: np.maximum(_clean(r.trk_pair_leg2_vprob), 0),
        "vprob3": lambda r: np.maximum(_clean(r.trk_pair_leg3_vprob), 0),
    },
    "neu": {
        "log_ptrel": lambda r: _ptrel(r, r.neu_pt),
        "deta": lambda r: _clean(r.neu_deta),
        "dphi": lambda r: _clean(r.neu_dphi),
        "log_dR": lambda r: np.log(np.maximum(_clean(r.neu_dR), 1e-4)),
        "log_dR_flight": lambda r: np.log(np.maximum(_clean(r.neu_dR_flight), 1e-4)),
        "inCone": lambda r: _flag(r.neu_inCone),
        "inFlightCone": lambda r: _flag(r.neu_inFlightCone),
        "isPhoton": lambda r: ak.values_astype(abs(_clean(r.neu_pdgId)) == 22, np.float32),
        "puppi": lambda r: _clip(r.neu_puppiWeight, 0, 1),
        "hcalFrac": lambda r: _clip(r.neu_hcalFraction, 0, 1),
    },
    "sv": {
        "log_ptrel": lambda r: _ptrel(r, r.sv_pt),
        "deta": lambda r: _clean(r.sv_eta) - r.cand_eta,
        "dphi": lambda r: _dphi(_clean(r.sv_phi), r.cand_phi),
        "mass": lambda r: _clip(r.sv_mass, 0, 10),
        "ntracks": lambda r: _clip(r.sv_ntracks, 0, 20),
        "log_chi2ndof": lambda r: _log1p_pos(_clean(r.sv_chi2) / np.maximum(_clean(r.sv_ndof), 1)),
        "log_dlenSig": lambda r: _log1p_pos(r.sv_dlenSig),
        "log_dxySig": lambda r: _log1p_pos(r.sv_dxySig),
        "pointing": lambda r: _pointing(r.sv_cosPAngle),
        "log_dist": lambda r: _log1p_pos(r.sv_distToCandSV, 0.001),
        "log_distSig": lambda r: _log1p_pos(r.sv_distToCandSVSig),
        "sharesLeg": lambda r: _flag(r.sv_sharesCandTrack),
    },
    "jet": {
        "log_ptrel": lambda r: _ptrel(r, r.jet_pt),
        "deta": lambda r: _clean(r.jet_eta) - r.cand_eta,
        "dphi": lambda r: _dphi(_clean(r.jet_phi), r.cand_phi),
        "mass_over_pt": lambda r: _clip(_clean(r.jet_mass) / np.maximum(_clean(r.jet_pt), 1), 0, 1),
        "nConst": lambda r: _clip(r.jet_nConstituents, 0, 60),
        "btagPNetB": lambda r: _clip(r.jet_btagPNetB, 0, 1),
        "btagUParTB": lambda r: _clip(r.jet_btagUParTB, 0, 1),
    },
}

# (deta, dphi, log ptrel) per token type, for the pair bias
GEOM = {
    "mu": ("deta", "dphi", "log_ptfrac"),
    "trk": ("deta", "dphi", "log_ptrel"),
    "neu": ("deta", "dphi", "log_ptrel"),
    "sv": ("deta", "dphi", "log_ptrel"),
    "jet": ("deta", "dphi", "log_ptrel"),
}

# branches needed besides the feature inputs
META_BRANCHES = ["run", "event", "cand_idx", "fold", "isData", "cand_channel", "cand_mass", "cand_kin_mass",
                 "cand_pt", "cand_eta", "cand_phi", "label_sig", "sig_prod", "nPVtx", "Generator_weight"]


def required_branches(feature_names):
    """Branch-name globs to read for the chosen features (uproot filter_name)."""
    pats = set(META_BRANCHES) | {"cand_*"}
    for t in feature_names["tokens"]:
        pats.add(f"{t}_*")
        pats.add(f"n{t}")
    return sorted(pats)


def default_feature_names(fields, use_activity=False):
    """Feature lists actually available in the files (a feature whose branches are missing is skipped)."""
    have = set(fields)

    def available(fn):
        try:
            import inspect
            src = inspect.getsource(fn)
        except OSError:
            return True
        import re
        names = set(re.findall(r"r\.([A-Za-z0-9_]+)", src)) | set(re.findall(r'r\["([A-Za-z0-9_]+)"\]', src))
        return names <= have

    cand = [k for k, f in CAND_FEATURES.items()
            if available(f) and (use_activity or k not in ACTIVITY_FEATURES)]
    tokens = {}
    for t, feats in TOKEN_FEATURES.items():
        names = [k for k, f in feats.items() if available(f)]
        if names:
            tokens[t] = names
    skipped = {"cand": [k for k in CAND_FEATURES if k not in cand and k not in ACTIVITY_FEATURES],
               **{t: [k for k in TOKEN_FEATURES[t] if k not in tokens.get(t, [])] for t in TOKEN_FEATURES}}
    return {"cand": cand, "tokens": tokens, "token_max": {t: TOKEN_MAX[t] for t in tokens},
            "skipped": {k: v for k, v in skipped.items() if v}}


def _pad(x, n):
    return ak.to_numpy(ak.fill_none(ak.pad_none(x, n, clip=True, axis=1), 0.0)).astype(np.float32)


def geom_index(feature_names):
    """Column indices of (deta, dphi, log ptrel) in each token feature array (for the pair bias)."""
    return {t: [names.index(g) for g in GEOM[t]] for t, names in feature_names["tokens"].items()}


def compute(r, feature_names):
    """Feature arrays for a chunk of flattened candidates.

    r: awkward record array (one entry per candidate) with the flattened branches.
    Returns dict: cand [N,Fc]; for each token type t: t_feats [N,n,F] (zero for padding), t_mask [N,n].
    """
    out = {"cand": np.stack([ak.to_numpy(CAND_FEATURES[k](r)).astype(np.float32) for k in feature_names["cand"]], axis=1)}
    for t, names in feature_names["tokens"].items():
        n = feature_names["token_max"][t]
        counts = ak.to_numpy(ak.num(r[f"{t}_eta"] if f"{t}_eta" in r.fields else r[f"{t}_pt"], axis=1))
        mask = np.arange(n)[None, :] < np.minimum(counts, n)[:, None]
        feats = np.stack([_pad(TOKEN_FEATURES[t][k](r), n) for k in names], axis=2)
        feats[~np.isfinite(feats)] = 0.0
        out[f"{t}_feats"] = feats * mask[:, :, None]
        out[f"{t}_mask"] = mask
    out["cand"][~np.isfinite(out["cand"])] = 0.0
    return out


def fit_mass(r):
    """Mass used for decorrelation and sculpting checks: kinematic fit, else Kalman fit."""
    km = ak.to_numpy(_clean(r.cand_kin_mass, -1.0)) if "cand_kin_mass" in r.fields else None
    m = ak.to_numpy(_clean(r.cand_mass, -1.0))
    return np.where(km > 0, km, m) if km is not None else m
