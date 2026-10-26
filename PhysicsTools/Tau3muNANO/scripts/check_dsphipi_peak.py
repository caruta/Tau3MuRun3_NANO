#!/usr/bin/env python3
"""Ds -> phi(mumu) pi peak in the DsPhiPi NanoAOD output (Cand2MuTrk table).

Applies the standard selection step by step (phi mass, trigger, vertex probability,
displacement, pointing), keeps the best candidate per event (highest vertex probability),
fits D+ and Ds Gaussians on an exponential background in the kinematic-fit mass and
saves a plot. Prints the cut flow and the fitted yields.

Run inside cmsenv (uproot, awkward, numpy, matplotlib; scipy for the fit):
  python3 check_dsphipi_peak.py -o ds_2024C.png \
      "/lustre/cms/store/user/<you>/ParkingDoubleMuonLowMass*/SkimDsPhiPi_2024eraC_stream*_Transformer_v1/*/*/*.root"
Quote the glob so the shell does not expand it. Without scipy, a sideband subtraction
replaces the fit.
"""
import argparse
import glob

import awkward as ak
import numpy as np
import uproot

DP_MASS, DS_MASS, PHI_MASS = 1.86966, 1.96835, 1.019461
RANGE = (1.70, 2.10)
TRIGGERS = ["Trigger_HLT_DoubleMu3_Trk_Tau3mu", "Trigger_HLT_DoubleMu3_TkMu_DsTau3Mu_v"]
CAND = "Cand2MuTrk"
BRANCHES = [f"{CAND}_{b}" for b in ["kin_mass", "kin_ok", "m12", "prob", "flightDistSig", "cosPointingAngle"]]


def model(m, n_dp, mu_dp, s_dp, n_ds, mu_ds, s_ds, a, k, binw):
    g = lambda n, mu, s: n * binw / (np.sqrt(2 * np.pi) * s) * np.exp(-0.5 * ((m - mu) / s) ** 2)
    return g(n_dp, mu_dp, s_dp) + g(n_ds, mu_ds, s_ds) + a * np.exp(-k * (m - RANGE[0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="NanoAOD files or quoted globs")
    ap.add_argument("-o", "--output", default="dsphipi_peak.png", help="plot file")
    ap.add_argument("--max-files", type=int, default=0, help="use only the first N files (0 = all)")
    ap.add_argument("--bins", type=int, default=80, help="bins in 1.70-2.10 GeV (80 = 5 MeV)")
    ap.add_argument("--phi-window", type=float, default=0.020, help="|m(mumu) - m(phi)| [GeV]")
    ap.add_argument("--min-prob", type=float, default=0.01)
    ap.add_argument("--min-lsig", type=float, default=3.0, help="flight-distance significance")
    ap.add_argument("--min-cos", type=float, default=0.99, help="cosine of the pointing angle")
    ap.add_argument("--no-trigger", action="store_true", help="do not require the tau3mu/Ds triggers")
    ap.add_argument("--all-cands", action="store_true", help="keep every candidate, not the best per event")
    args = ap.parse_args()

    files = sorted({f for pat in args.inputs for f in (glob.glob(pat) or [pat])})
    if args.max_files:
        files = files[: args.max_files]
    print(f"{len(files)} files")
    if not files:
        raise SystemExit("no input files")

    steps = ["in mass range", "phi mass", "+ trigger", "+ vertex prob", "+ displaced", "+ pointing"]
    counts = np.zeros(len(steps), dtype=np.int64)
    masses = []
    n_events = 0
    for arr in uproot.iterate([f + ":Events" for f in files], filter_name=BRANCHES + TRIGGERS, step_size="200 MB"):
        n_events += len(arr)
        m = arr[f"{CAND}_kin_mass"]
        trig = np.zeros(len(arr), dtype=bool)
        if not args.no_trigger:
            missing = [t for t in TRIGGERS if t not in arr.fields]
            if missing:
                raise SystemExit(f"trigger branches missing: {missing} (use --no-trigger)")
            for t in TRIGGERS:
                trig |= ak.to_numpy(arr[t]).astype(bool)
        else:
            trig[:] = True
        trig_c = ak.broadcast_arrays(trig, m)[0]
        cuts = [(arr[f"{CAND}_kin_ok"] == 1) & (m > RANGE[0]) & (m < RANGE[1])]
        cuts.append(cuts[-1] & (abs(arr[f"{CAND}_m12"] - PHI_MASS) < args.phi_window))
        cuts.append(cuts[-1] & trig_c)
        cuts.append(cuts[-1] & (arr[f"{CAND}_prob"] > args.min_prob))
        cuts.append(cuts[-1] & (arr[f"{CAND}_flightDistSig"] > args.min_lsig))
        cuts.append(cuts[-1] & (arr[f"{CAND}_cosPointingAngle"] > args.min_cos))
        for i, c in enumerate(cuts):
            counts[i] += int(ak.sum(c))
        sel = cuts[-1]
        if args.all_cands:
            masses.append(ak.to_numpy(ak.flatten(m[sel])))
        else:
            prob = ak.where(sel, arr[f"{CAND}_prob"], -1.0)
            best = ak.argmax(prob, axis=1, keepdims=True)
            has = ak.any(sel, axis=1)
            masses.append(ak.to_numpy(ak.flatten(m[best][has])))

    print(f"{n_events} events")
    print("cut flow (candidates):")
    for s, c in zip(steps, counts):
        print(f"  {s:15s} {c:12d}")
    mass = np.concatenate(masses) if masses else np.array([])
    print(f"{len(mass)} {'candidates' if args.all_cands else 'events (best candidate)'} after all cuts")

    y, edges = np.histogram(mass, bins=args.bins, range=RANGE)
    x = 0.5 * (edges[1:] + edges[:-1])
    binw = edges[1] - edges[0]

    fit = None
    try:
        from scipy.optimize import curve_fit
        f = lambda m, n_dp, mu_dp, s_dp, n_ds, mu_ds, s_ds, a, k: model(m, n_dp, mu_dp, s_dp, n_ds, mu_ds, s_ds, a, k, binw)
        side = y[(x < 1.83) | (x > 2.03)]
        a0 = max(float(side.mean()) if len(side) else 1.0, 1.0)
        p0 = [max(y.sum() * 0.05, 1), DP_MASS, 0.015, max(y.sum() * 0.1, 1), DS_MASS, 0.015, a0, 1.0]
        lo = [0, 1.84, 0.004, 0, 1.94, 0.004, 0, -20]
        hi = [np.inf, 1.90, 0.05, np.inf, 2.00, 0.05, np.inf, 20]
        popt, pcov = curve_fit(f, x, y, p0=p0, sigma=np.sqrt(np.maximum(y, 1)), bounds=(lo, hi), maxfev=20000)
        perr = np.sqrt(np.diag(pcov))
        fit = (f, popt, perr)
        n_ds, mu_ds, s_ds = popt[3:6]
        win = (x > mu_ds - 2 * s_ds) & (x < mu_ds + 2 * s_ds)
        bkg = float(np.sum(popt[6] * np.exp(-popt[7] * (x[win] - RANGE[0]))))
        print(f"fit: Ds  N = {popt[3]:.0f} +- {perr[3]:.0f}, mean = {1000 * mu_ds:.1f} MeV, sigma = {1000 * s_ds:.1f} MeV, "
              f"S/sqrt(B) in +-2 sigma = {0.954 * n_ds / np.sqrt(max(bkg, 1)):.1f}")
        print(f"     D+  N = {popt[0]:.0f} +- {perr[0]:.0f}, mean = {1000 * popt[1]:.1f} MeV, sigma = {1000 * popt[2]:.1f} MeV")
    except ImportError:
        sig = (mass > 1.945) & (mass < 1.995)
        sb = ((mass > 1.905) & (mass < 1.930)) | ((mass > 2.010) & (mass < 2.035))
        n_sig = int(sig.sum()) - int(sb.sum())  # sidebands have the same total width as the window
        print(f"scipy not available: sideband-subtracted Ds yield in 1.945-1.995 GeV = {n_sig} "
              f"(window {int(sig.sum())}, sidebands {int(sb.sum())})")
    except RuntimeError as e:
        print(f"fit did not converge ({e}); look at the plot")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(x, y, yerr=np.sqrt(y), fmt="o", ms=3, color="#14213D", label="data")
    if fit:
        f, popt, _ = fit
        xx = np.linspace(*RANGE, 400)
        ax.plot(xx, f(xx, *popt) * 1.0, color="#E8833A", lw=2, label="fit")
        ax.plot(xx, popt[6] * np.exp(-popt[7] * (xx - RANGE[0])), color="#9AA5B8", ls="--", label="background")
    ax.set_xlabel("m(μμπ), kinematic fit [GeV]")
    ax.set_ylabel(f"{'candidates' if args.all_cands else 'events'} / {1000 * binw:.0f} MeV")
    ax.set_xlim(*RANGE)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"plot saved to {args.output}")


if __name__ == "__main__":
    main()
