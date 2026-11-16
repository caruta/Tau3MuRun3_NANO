# Tau3muML: transformer for the τ → 3μ HF channel

Standalone PyTorch (no weaver). Input: the flattened ntuples from
`Tau3muNANO/scripts/flatten_for_transformer.py`. Three steps:

1. **Training set** at Bari (`scripts/make_training_set.py`): selects, subsamples, computes features and
   writes shuffled shards of `.npy` arrays.
2. **Training** on a SWAN GPU (`notebooks/train_swan.ipynb`, or `python3 -m tau3muml.train`): one model
   per fold, k-fold with 5 folds.
3. **Evaluation** (`tau3muml.evaluate`): out-of-fold scores, ROC, mass-sculpting and pileup checks; then
   `tau3muml.apply_model` scores any flattened file.

## Samples and labels

| role | sample | used for |
| --- | --- | --- |
| 0 signal | Tau3Mu MC (`*_MC_Ds`, `*_MC_Bd`, `*_MC_Bu`), |Q| = 1, `label_sig == 1` | training |
| 1 background | Tau3Mu data, |Q| = 1, sidebands (SR 1.74–1.82 GeV removed by the flattener) | training |
| 2 SS3MU | Tau3Mu data, |Q| = 3, all masses | sculpting check only |

Folds come from `hash(run, event)`. Model k is tested on fold k, validated on fold (k+1) % 5 and trained
on the other three, so each candidate gets a score from a model that never saw its event.

## Why the inputs look the way they do

**Mass.** Signal exists only at m(τ) and background only in the sidebands, so a network that can
reconstruct m(3μ) learns "peak = signal" and sculpts a peak into the background. The inputs therefore
contain no absolute momentum: the candidate pT is not used, every momentum is a fraction of it, and no
pair or 3-body masses or ctau enter. Some mass information still leaks: the three leg pT fractions and
angles give m/pT. The loss adds a DisCo term (distance correlation between score and mass on the
sideband candidates of each batch, weight `disco`), and the evaluation checks the result.

**Data vs MC.** Signal is MC and background is data, so anything that differs between data and MC
separates them without being physics. The main known one is pileup: signal MC is reweighted to the
data nPV distribution, and the per-candidate track, neutral and SV counts are off by default
(`--use-activity` in `make_training_set.py` turns them on). The nPV plot in the evaluation shows what is
left. The Dₛ → φπ control channel is the place to validate the classifier on data vs MC, once the
third-leg treatment for that channel is settled.

## Features (`tau3muml/features.py`)

| input | features |
| --- | --- |
| candidate (CLS token) | η, flight-distance significances, pointing, vertex and kinematic-fit probabilities, 2-leg vertex probabilities, IP significances, ΔR / Δz between legs, isolations relative to the candidate pT, min DCA of isolation tracks, compatible-track count, best extra dimuon vertex probability |
| muons (3) | pT fraction, η, Δη, Δφ, charge, ID flags, soft MVAs, compatibilities, kinks, χ², hits, stations, timing, IP significances wrt PV and SV |
| tracks (≤128) | log pT fraction, Δη, Δφ, ln ΔR, charge, quality flags, cone/cylinder flags, dxy, dz, IPs and significances wrt PV and SV, distance to the flight segment, hits, χ², PV association, PUPPI weight, muon flags, IVF SV membership, Δχ² of adding it to the SV, leg × track DCA and vertex probability |
| neutrals (≤32) | log pT fraction, Δη, Δφ, ln ΔR to the candidate and to the flight direction, photon flag, PUPPI, HCAL fraction |
| SVs (≤8) | log pT fraction, Δη, Δφ, mass, tracks, χ²/ndof, displacement significances, pointing, distance to the candidate SV, shared-leg flag |
| jets (≤6) | log pT fraction, Δη, Δφ, m/pT, constituents, PNet and UParT b tags |

Sentinels (−999) become 0; everything is standardised inside the model with the means and widths in
`stats.json`. Features whose branches are missing are skipped and listed when the set is made.

## Model (`tau3muml/model.py`)

Per-type embedding MLPs and type embeddings; the candidate features form the CLS token; pre-LN
transformer blocks (default d = 128, 8 heads, 6 layers, ≈1.3 M parameters). As in ParT, the attention
gets a per-head bias from pair features of every token pair: ln ΔR, ln z = ln(min(pTᵢ, pTⱼ)/(pTᵢ + pTⱼ)),
ln(min(pTᵢ, pTⱼ)·ΔR/pT(cand)): angles and momentum ratios only. Each batch is cut to its longest
sequence, so the 128-track limit costs memory only when a batch has such a candidate.

Weights: signal gets sign(gen weight) × production-mode weight (`prod_frac`, default: the MC mix) ×
nPV reweighting, then is normalised to the background sum.

## 1. Training set (Bari)

```bash
cd $CMSSW_BASE/src/PhysicsTools/Tau3muML && cmsenv
python3 scripts/make_training_set.py \
    --flat-dir /lustre/cms/store/user/<you>/flat_Transformer_v1 \
    --out      /lustre/cms/store/user/<you>/train_Transformer_v1
```

Defaults: 1 M signal, 1 M sideband and 0.2 M SS3MU candidates; data files are read in random order and
20 % of each file's candidates kept (`--row-frac-data`), so all eras and streams contribute. If a role
has fewer candidates than requested, the script says so. About 10 kB per candidate, so ≈ 22 GB for the
defaults. Peak memory is ≈ 2–3 GB (printed as "max RSS"; `--step` and `--flush-rows` lower it).

It takes hours (it reads ≈ 5 M data candidates), longer than interactive nodes allow, so run it as an
HTCondor job:

```bash
mkdir -p ~/trainset_job/logs && cd ~/trainset_job
condor_submit cmssw=$CMSSW_BASE flat=/lustre/cms/store/user/<you>/flat_Transformer_v1 \
    out=/lustre/cms/store/user/<you>/train_Transformer_v1 \
    $CMSSW_BASE/src/PhysicsTools/Tau3muML/scripts/submit_make_training_set.sub
tail -f logs/make_training_set.*.out
```

If a job dies half way, just resubmit: the leftovers of the interrupted run are removed first.

Copy it to your EOS space (SWAN sees `/eos/user/<initial>/<user>`):

```bash
kinit <cern-user>@CERN.CH          # at Bari; or copy via lxplus
xrdcp -r --parallel 8 /lustre/cms/store/user/<you>/train_Transformer_v1 \
    root://eosuser.cern.ch//eos/user/<i>/<cern-user>/tau3mu/
```

## 2. Training (SWAN)

Start a SWAN session with a GPU software stack, open a terminal and clone the branch:

```bash
git clone -b transformer-ntuples https://github.com/caruta/Tau3MuRun3_NANO ~/Tau3MuRun3_NANO
```

Open `PhysicsTools/Tau3muML/notebooks/train_swan.ipynb`, set `DATA` and `OUT` (on EOS, so checkpoints
survive the session) and run it. Equivalent from the terminal:

```bash
cd ~/Tau3MuRun3_NANO/PhysicsTools/Tau3muML
python3 -m tau3muml.train --data /eos/user/<i>/<u>/tau3mu/train_Transformer_v1 --out /eos/user/<i>/<u>/tau3mu/runs/v1 --fold 0 1 2 3 4
```

Every epoch writes `fold<k>/last.pt` (resume point), `best.pt` (lowest validation loss) and a line in
`metrics.csv` with BCE, DisCo, AUC and background efficiency at 50 % and 20 % signal efficiency.
Rerunning the same command resumes. For a quick first pass, `max_train_rows=200000` and `epochs=5`.

## 3. Evaluation

```bash
python3 -m tau3muml.evaluate --data <DATA> --out <OUT>
```

Writes `<OUT>/eval/`: `scores.npz`, `summary.json` and plots. What to look at:

- **`bkg_mass.png`**: the sideband mass shape in score quantiles should not change.
- **`ss3mu_SR_over_SB`** in `summary.json` and `ss3mu_mass.png`: SS3MU has no signal and is not blinded;
  its SR/SB ratio should stay constant as the cut tightens. A rise means mass leakage: increase `disco`.
- **`npv.png`**: signal and background efficiencies vs nPV.
- per-fold AUCs should agree within their spread.

Scores for any flattened file (friend tree with `run`, `event`, `cand_idx`, `score`):

```bash
python3 -m tau3muml.apply_model --models <OUT> --out-dir scores/ /lustre/.../flat_Transformer_v1/Tau3Mu/2024eraC_stream0/*.root
```

## Quick comparison with BDT baselines

**Run 2 BDT inputs** (once per training set, at Bari): the Run 2 three-global-muon BDT variables
(AN section 11.1, arXiv:2312.02371) for exactly the training-set candidates, read from the flattened
ntuples. About an hour with 8 processes:

```bash
mkdir -p ~/bdt_job/logs && cd ~/bdt_job
condor_submit cmssw=$CMSSW_BASE flat=/lustre/cms/store/user/<you>/flat_Transformer_v1 \
    trainset=/lustre/cms/store/user/<you>/train_Transformer_v1 \
    $CMSSW_BASE/src/PhysicsTools/Tau3muML/scripts/submit_make_bdt_table.sub
```

It writes `bdt_run2.npz` into the training-set directory; copy it next to the shards on EOS. Variables:
lowest muon pT, max chi2LocalPosition, max trkKink, min segment compatibility, a muon ID MVA per muon
(softMvaRun3 instead of the Run 2 global-muon BDT), vertex χ²/ndof, pointing angle, 3D flight
significance, min/max d0 significance, min DCA of a pT > 1 GeV track to the vertex and the max relative
track isolation of the muons (both from the track tokens), plus σm/m for the categories.

**Comparison** (SWAN, after the evaluation):

```bash
python3 -m tau3muml.compare_bdt --data <DATA> --out <OUT> --bkg-scale 120
```

BDTs trained per fold on the same candidates, folds and weights as the transformer:

| name | inputs | what the comparison shows |
| --- | --- | --- |
| `bdt_run2` | Run 2 variables and XGBoost settings (AN table 18); one BDT for all resolution categories | the reference |
| `bdt_run2like` | ~20 transformer candidate features, no absolute pT | |
| `bdt_candmu` | all candidate and muon features of the transformer | gain from more candidate information |
| transformer | + tracks, neutrals, SVs, jets | gain from the context and the architecture |

Outputs (printed and in `eval/compare_bdt.json`, `compare_roc.png`): sideband efficiency at fixed signal
efficiency for all candidates and for three global muons; the SS3MU SR/SB ratio vs cut; and a rough
expected 95 % CL upper limit on the signal yield, as a ratio to `bdt_run2`:

- `inclusive`: all candidates, 1.74–1.82 GeV, best 2 score categories;
- `run2_categories`: three global muons, resolution categories A/B/C (σm/m < 0.7 %, < 1.05 %, above) with
  the Run 2 signal windows, best 3 score categories above a discarded region in each, as in Run 2.

Background in each window: exponential fit to the sidebands passing the cut, times `--bkg-scale`
(sideband candidates in the full data / in the training set); signal: fraction of the signal MC in the
window; Asimov approximation. Not included: trigger matching, analysis base cuts, the 2 global + 1
tracker muon category. Only ratios between methods mean something.

## Comparison after the analysis base cuts, one candidate per event

No retraining: the scores of `compare_bdt` are reused on an analysis-like subset.

1. At Bari, inside cmsenv (about an hour): flags for every training-set candidate, from the flattened
   ntuples. The base selection is the one of `PhysicsTools/Tau3muANA/tau3mu_analyser.py` (L1 OR and
   HLT_DoubleMu3_TkMu_DsTau3Mu, muon acceptance, 3 global + medium muons, |Q| = 1, dR < 0.6, dz < 2.5,
   flight significance >= 3.5, 1.62-2.0 GeV, trigger matching of the three muons); the best candidate
   is the lowest vertex chi2 among the base candidates of the event. It stops if a needed branch is
   missing from the flattened files.

```bash
mkdir -p ~/basecut_job/logs && cd ~/basecut_job
condor_submit cmssw=$CMSSW_BASE flat=/lustre/cms/store/user/<you>/flat_Transformer_v1 \
    trainset=/lustre/cms/store/user/<you>/train_Transformer_v1 \
    $CMSSW_BASE/src/PhysicsTools/Tau3muML/scripts/submit_make_basecut_table.sub
```

   It writes `basecuts.npz` and `basecuts.json` (cut flow) into the training-set directory; copy both
   next to the shards on EOS.

2. On SWAN, after `compare_bdt` (needs `eval/compare_scores.npz` and `eval/compare_bdt.json`):

```python
from tau3muml.compare_basecuts import compare_basecuts
res = compare_basecuts(DATA, OUT)
```

Selections: `loose`, `base`, `base_best`, `base_best_veto` (Run 2 phi and omega vetoes). Outputs in
`eval/`: `compare_basecuts.json`, `compare_roc_basecuts.png`, `fits_<method>_base_best.png`. All
limits here fit the background in 1.62-2.0 GeV, so the `loose` numbers differ slightly from those of
`compare_bdt`, which fits 1.40-2.20 GeV.

## Data/MC check of the inputs on Ds -> phi(mumu) pi

`scripts/control_datamc.py` compares every transformer input (candidate, the two muon legs, the
pion leg, all context tokens and their multiplicities) between background-subtracted Ds -> phi pi
data and nPV-reweighted Ds -> phi pi MC, per era, for early/late 2024 (run 383000) and for the year.
Background subtraction as in the Run 2 AN (section 8.3): peak 1.93-2.01 GeV minus the 1.72-1.80 GeV
sideband scaled by an exponential + Ds + D+ fit. Metric: separation 2|AUC(data vs MC) - 0.5|.

```bash
cd ~/control_job && mkdir -p logs
S=$CMSSW_BASE/src/PhysicsTools/Tau3muML/scripts
A="--flat-dir /lustre/.../flat_Transformer_v1 --train-set /lustre/.../train_Transformer_v1 --work /lustre/.../control_datamc --jobs 16"
python3 $S/control_datamc.py fit  $A
python3 $S/control_datamc.py hist $A     # the long one
python3 $S/control_datamc.py report --work /lustre/.../control_datamc --model-config <OUT>/fold0/config.json
```

On a slow machine, split `fit` and `hist` over many HTCondor jobs with
`scripts/submit_control_datamc.sub` (`--part i --nparts N`, outputs in `WORK/parts/`), then merge with
`fit-merge` / `hist-merge` before the next stage; the commands are in the file header.

Selection (Run 2 AN section 8.1 with the Run 3 muon acceptance): two global muons, OS mass
0.97-1.07 GeV, track pT > 1.2 GeV, flight significance >= 2; options `--phi-window`, `--fsig-min`,
`--trk-pt-min`, `--hlt <trigger bit>`, `--data-frac` (to use a fraction of the data).

## Testing without the real ntuples

```bash
python3 test/make_synthetic_flat.py /tmp/synth
python3 scripts/make_training_set.py --flat-dir /tmp/synth --out /tmp/trainset --n-signal 1e4 --n-background 8e3 \
    --n-ss3mu 2e3 --row-frac-data 1 --shard-size 4000
python3 -m tau3muml.train --data /tmp/trainset --out /tmp/runs --fold 0 --epochs 2 --layers 2 --d 32 --heads 4
```

## When the full statistics are in

Rerun the missing flattening jobs, then `make_training_set.py` with a new `--out` (it refuses to
overwrite) and larger targets, and train into a new `OUT`.
