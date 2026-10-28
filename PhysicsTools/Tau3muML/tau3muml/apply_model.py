#!/usr/bin/env python3
"""Score flattened files with the five fold models (each candidate gets the model whose test fold
contains it, so the score is never from a model trained on that event).

Writes, per input file, <out-dir>/<input name>_score.root with run, event, cand_idx, fold, score,
in the same row order as the input (usable as a friend tree).

  python3 -m tau3muml.apply_model --models runs/v1 --out-dir scores/ flat/Tau3Mu/2024_MC_Ds/*.root
"""
import argparse
import os

import awkward as ak
import numpy as np
import torch
import uproot

from . import features as F
from .data import to_torch
from .train import device_of, load_model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--models", required=True, help="training output directory with fold<k>/best.pt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--which", default="best.pt")
    ap.add_argument("--batch-size", type=int, default=2048)
    a = ap.parse_args()
    device = device_of()
    models = {k: load_model(os.path.join(a.models, f"fold{k}", a.which), device) for k in range(5)
              if os.path.exists(os.path.join(a.models, f"fold{k}", a.which))}
    if not models:
        raise SystemExit(f"no models in {a.models}")
    stats = torch.load(os.path.join(a.models, f"fold{min(models)}", a.which), map_location="cpu", weights_only=False)["stats"]
    names = {"cand": stats["features"]["cand"], "tokens": stats["features"]["tokens"],
             "token_max": stats["features"]["token_max"]}
    os.makedirs(a.out_dir, exist_ok=True)
    for path in a.inputs:
        res = {k: [] for k in ("run", "event", "cand_idx", "fold", "score")}
        for arr in uproot.iterate(f"{path}:Events", filter_name=F.required_branches(names), step_size="200 MB"):
            x = F.compute(arr, names)
            fold = ak.to_numpy(arr["fold"])
            score = np.full(len(fold), -1.0, dtype=np.float32)
            with torch.no_grad():
                for k, model in models.items():
                    idx = np.nonzero(fold == k)[0]
                    for i in range(0, len(idx), a.batch_size):
                        j = idx[i : i + a.batch_size]
                        xb = to_torch({n: v[j] for n, v in x.items()}, device)
                        score[j] = torch.sigmoid(model(xb).float()).cpu().numpy()
            for b in ("run", "event", "cand_idx"):
                res[b].append(ak.to_numpy(arr[b]))
            res["fold"].append(fold)
            res["score"].append(score)
        out = os.path.join(a.out_dir, os.path.basename(path).replace(".root", "_score.root"))
        with uproot.recreate(out) as fh:
            fh["Events"] = {k: np.concatenate(v) for k, v in res.items()}
        print(f"{path} -> {out}")


if __name__ == "__main__":
    main()
