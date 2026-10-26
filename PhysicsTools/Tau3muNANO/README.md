# Tau3Mu NanoAOD Analysis Workspace

This workspace is dedicated to the production of **custom NanoAOD** for the $\tau \to 3\mu$ decay analysis for the HF channel.

## Testing Production

If you want to test the config file and produce the NanoAOD output from a small sample:

1.  Ensure you have a valid grid proxy (`voms-proxy-init`):
    ```bash
    voms-proxy-init --voms cms
    ```
2.  Run the job:
    ```bash
    cmsRun PhysicsTools/Tau3muNANO/test/test_{analysis_type}_cfg.py isMC=True/False
    ```
    analysis_type = ["Tau3Mu", "DsPhiPi"]
---

## Running Production

The main informations on data and MC samples are stored in the `Runs.json` file and in files in the `datasets` folder.
### How to add new years (or modify existing ones)
1. Apply your changes in `Runs.json`;
2. If your changes affect data (not required if changes are only in MC), run `getDatset.py` to download information from DAS (as in the example below);
    - `voms-proxy-init -voms cms -rfc`
    - `python3 getDatset.py --year 2025` 

<p>&nbsp;</p>


### Run ntuplizer on a full dataset for one specific year:
```
cd crab_submission
voms-proxy-init --voms cms --rfc --valid 192:00
python3 getDatset.py --year 2024          # once per year: writes datasets/datasets_<year>_<era>.txt from DAS
source submitAllJobs.sh [year] [isMC] [channel] [tag]
```
* `[year]` = `2022`, `2023`, `2024` or `2025`; `[isMC]` = `true` or `false`; `[channel]` = `signal` (Tau3Mu) or `control` (DsPhiPi); `[tag]` defaults to `Transformer_v1`.
* Data: one directory per era, `<year>_era<era>_<channel>_<tag>`, with one CRAB task per parking stream. MC: one task per sample in `Runs.json`; samples whose `MC_channel` differs from `[channel]` are skipped.
* An MC dataset name with `*` in `Runs.json` is resolved with DAS at submission and must match exactly one dataset.
* Jobs process all events (`maxEvents` defaults to -1); for local tests pass e.g. `maxEvents=1000`.

To check or resubmit, use `status.sh`, `report.sh` and `resubmit.sh` in each data directory, or for a whole production:
```
source resubmitAll.sh [year] [channel] [tag]
```

<p>&nbsp;</p>

Let's move now to the analysis directory Tau3MuANA!
<p>&nbsp;</p>
