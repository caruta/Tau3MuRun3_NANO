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

If a run dies half way, delete its `--out` directory before resubmitting.

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
