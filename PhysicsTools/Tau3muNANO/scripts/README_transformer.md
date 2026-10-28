# Transformer ntuples for the HF channel

The custom NanoAOD now carries everything needed to train a ParT/UParT-style classifier on
τ→3μ candidates: candidate-level fit features, the three legs, and context tokens (charged
tracks, neutrals, IVF secondary vertices, jets) around each candidate. Production is two steps:

1. **CMSSW** (`cmsRun test/test_Tau3Mu_cfg.py maxEvents=1000`, or CRAB as before): per-event tables, as usual.
2. **Flattening** (`scripts/flatten_for_transformer.py`): one row per candidate, jagged token
   branches, blinding, k-fold split. Training: `PhysicsTools/Tau3muML` (standalone PyTorch; the weaver
   config `scripts/weaver_tau3mu.yaml` is kept for reference).

Everything is switched on by default. `setupTau3Mu(process, isMC, transformer=False)` (or the
same for `setupDsPhiPi`) gives the previous content back, except for the bug fixes and the
same-sign triplets.

## What changed in the NanoAOD

| Table | New content |
| --- | --- |
| `Tau3Mu`, `Cand2MuTrk` | `channel` (0 TAU3MU, 1 DSPHIPI, 2 SS3MU), `pv_idx`, SV and PV covariances (`sv_cxx` … `pv_czz`), kinematic-fit `kin_mass`, `kin_massErr`, `kin_prob`, pair masses `m12/m13/m23` and 2-track vertex probabilities `vprob12/13/23`, `ctau`, per-leg `refit_<leg>_eta/phi` and 3D IPs wrt PV and SV, context columns `iso_trk_dR03/05`, `n_trk_vtxcompat`, `nTrk_total`, `nTrk_cone`, `nTrk_cyl`, `nNeu_total`, `nSV_total` |
| `Muon` | `srcIdx` (join key for `mu*_idx`), `isStandalone`, `nMatchedStations`, `inTimeMuon`, `timeAtIpInOut(Err)`, `softMvaRun3`, `sip3d`; MC: `genFlags`, `genMotherPdgId`, `genTauIdx`, `genHadronPdgId` |
| `<Cand>Trk` | charged tracks (packed PF + lost tracks), pT > 0.5 GeV, `inCone` (ΔR < 1.0) or `inCylinder` (distance to the PV→SV segment < 1 mm within ΔR < 1.5, 3D IP significance wrt the PV > 3), max 128 per candidate ordered by IP significance wrt the SV; IPs wrt PV and SV, hit counts, PV association, muon match, IVF SV membership, `sv_dchi2` (top 16), leg × track `pair_leg<i>_m_pi/m_K/dca/vprob` |
| `<Cand>Neu` | photons and neutral hadrons, pT > 0.3 GeV, ΔR < 0.5 to the candidate momentum or to the flight direction, max 32; leg × neutral masses |
| `<Cand>SV` | IVF SVs ordered by distance to the candidate SV, max 8 |
| `CtxJet` | AK4 Puppi jets, pT > 15 GeV, with PNet and UParT b scores (−1000 if absent in the MiniAOD) |
| `Generator` | `weight` (MC) |

Tracks without stored track details in the MiniAOD (no covariance) are kept with their kinematics,
`dxy` and `dz`, but their errors, IPs, DCAs and fit-based features are −999 and they sort last.

Tokens are built only for candidates with Kalman or kinematic-fit mass in 1.2–2.5 GeV
(`ctxMassWindow` in `addTransformerTables`; `(0, 0)` builds them for every candidate). The candidate
and muon tables still contain all candidates; those outside the window have `nTrk_total = nNeu_total =
nSV_total = -1`, isolation −999 and no token rows, and the flattener skips them. In 2024 data, 98 % of
candidates are outside this window (mostly above 6 GeV), so this saves most of the token storage and CPU.

Token rows carry `candIdx`, the candidate row in `Tau3Mu` / `Cand2MuTrk`. Missing values are −999.

`genFlags` bits: 1 τ, 2 D<sub>s</sub>, 4 D⁺, 8 B hadron, 16 W, 32 φ, 64 η/η′/ρ/ω, 128 decay in
flight (no gen muon matched, but a gen π/K is). Legs from the same τ share `genTauIdx`.

### Fixes to the existing content

- `Tau3Mu_diMuVtxFit_bestProb` / `_bestMass` were written as `int` (truncated); now `float`.
- `MuonTriggerSelector` can skip a muon, which would shift `Muon` rows relative to `mu*_idx`.
  Join through `Muon_srcIdx` (the flattener does). `Tau3muANA` still indexes directly; it is
  correct whenever no muon is skipped, which is the usual case.
- `Muon_ip3d` was documented as a significance; it is the value. `Muon_sip3d` added.
- `PVRefitter` printed every removed track to stdout; now `LogDebug`.
- `Tau3Mu` now also contains |Q| = 3 triplets (SS3MU). `Tau3muANA/tau3mu_analyser.py` keeps
  |Q| = 1 only, so its results are unchanged.

## Flattening

```bash
pip install uproot awkward   # if not available
python3 scripts/flatten_for_transformer.py --channel Tau3Mu --data -o flat/data2024 nano_data_*.root
python3 scripts/flatten_for_transformer.py --channel Tau3Mu        -o flat/sig_DsTau  nano_sig_*.root
python3 scripts/flatten_for_transformer.py --channel DsPhiPi --data -o flat/dsphipi    nano_ds_*.root
```

Writes `<prefix>_fold<k>.root` (tree `Events`) with `cand_*`, `mu_*` (3 legs), `trk_*`, `neu_*`,
`sv_*`, `jet_*`, `fold`, `isData`, and MC labels `label_sig`, `sig_prod` (1 D<sub>s</sub>,
2 B→D<sub>s</sub>, 3 B→τ, 4 D⁺, 5 W) or `label_dsphipi`.

- **Blinding:** in data, |Q| = 1 candidates with Kalman or kinematic-fit mass in
  1.74–1.82 GeV are not written (`--sr` to change). SS3MU and DsPhiPi are not blinded.
- **Folds:** `fold = hash(run, event) mod 5`, so all candidates of an event share a fold.
- **Mass window:** 1.40–2.20 GeV by default (`--mass-window`).

## After the CRAB production

All commands inside `cmsenv`, from `$CMSSW_BASE/src/PhysicsTools/Tau3muNANO/scripts`; `<you>` is your
storage user name (outputs under `/lustre/cms/store/user/<you>` at T2_IT_Bari).

**Ds → φπ peak in data** (step 2): cut flow, best candidate per event, fit of D⁺ and Dₛ on an
exponential background (needs scipy; without it, a sideband subtraction).

```bash
python3 check_dsphipi_peak.py -o ds_2024C.png \
    "/lustre/cms/store/user/<you>/ParkingDoubleMuonLowMass*/SkimDsPhiPi_2024eraC_stream*_Transformer_v1/*/*/*.root"
```

**Flattening on HTCondor** (step 3):

```bash
python3 make_flatten_jobs.py --base /lustre/cms/store/user/<you> --tag Transformer_v1 --year 2024 \
    --out-dir /lustre/cms/store/user/<you>/flat_Transformer_v1 --work-dir ~/flatten_jobs
cd ~/flatten_jobs
for f in jobs_part*.txt; do condor_submit cmssw=$CMSSW_BASE jobs=$f $CMSSW_BASE/src/PhysicsTools/Tau3muNANO/scripts/submit_flatten.sub; done
# when the jobs are done:
python3 $CMSSW_BASE/src/PhysicsTools/Tau3muNANO/scripts/check_flat_production.py \
    --work-dir ~/flatten_jobs --out-dir /lustre/cms/store/user/<you>/flat_Transformer_v1
# failed jobs, if any:
for f in jobs_failed*.txt; do condor_submit cmssw=$CMSSW_BASE jobs=$f $CMSSW_BASE/src/PhysicsTools/Tau3muNANO/scripts/submit_flatten.sub; done
```

Outputs: `<out-dir>/<Tau3Mu|DsPhiPi>/<sample>/<sample>_part<k>_fold<f>.root`, one sample per data era
and stream (`2024eraC_stream0`) or MC sample (`2024_MC_Bd`). `make_flatten_jobs.py` only picks up
directories of the given tag, and the latest CRAB timestamp of each.

## Things to check on the first test production

The code has not been compiled here. Before a large production:

1. `scram b` and run `test/test_Tau3Mu_cfg.py` and `test/test_DsPhiPi_cfg.py` on MC and data.
2. **CPU:** the context producer builds transient tracks for every track near each candidate
   and runs pair and 4-track fits. Knobs in `transformer_cff.addTransformerTables`:
   `doPairFits` (2-track vertex probabilities), `nDchi2` (tracks with `sv_dchi2`), `maxTrk`.
3. **Release-dependent names:** `softMvaRun3` is on by default (pass `addSoftMvaRun3=False` if
   your release's `pat::Muon` lacks `softMvaRun3Value()`); the UParT jet tag name may not exist in
   2022–2023 MiniAOD (the column is then −1000).
4. **NanoAOD defaults:** the jet and generator-weight tables clone
   `simplePATJetFlatTableProducer_cfi` and `simpleGenEventFlatTableProducer_cfi`, like the
   existing `GenPart` table does with its cfi. If your release lacks either, define the module
   by type instead.
5. **Occupancy:** look at `nTrk_total`, `nNeu_total`, `nSV_total` in signal MC and sidebands and
   set `maxTrk`, `maxNeu`, `maxSV` (and the weaver `length`s) to their 99th percentiles.
6. **Size:** measure kB per candidate on a test file before submitting everything.
