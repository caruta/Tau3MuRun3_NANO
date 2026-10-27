#!/usr/bin/env python3
"""Check a flattened production made with make_flatten_jobs.py + submit_flatten.sub.

1. Jobs (with --work-dir): finds jobs with no log saying "wrote N candidates" (matched by
   output prefix, so resubmissions count) and writes them to <work-dir>/jobs_failed*.txt
   (at most 1000 jobs each), with the command to resubmit them.
2. Content (with --out-dir): per channel/sample, candidates per fold, label fractions (MC)
   and token occupancy before truncation against the CMSSW limits.

  python3 check_flat_production.py --work-dir flatten_jobs --out-dir /lustre/cms/store/user/<you>/flat_Transformer_v1
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import awkward as ak
import numpy as np
import uproot

LIMITS = {"cand_nTrk_total": 128, "cand_nNeu_total": 32, "cand_nSV_total": 8}


def check_jobs(work_dir):
    jobs = [l for l in open(os.path.join(work_dir, "jobs.txt")).read().splitlines() if l.strip()]
    done = {}
    for out in glob.glob(os.path.join(work_dir, "logs", "flatten.*.out")):
        text = open(out, errors="replace").read()
        p = re.search(r"output=(\S+)", text)
        w = re.search(r"wrote (\d+) candidates", text)
        if p and w:
            done[p.group(1)] = int(w.group(1))
    failed = [j for j in jobs if j.split()[2] not in done]
    print(f"jobs: {len(jobs)} total, {len(jobs) - len(failed)} finished, {len(failed)} failed or still running")
    for stale in glob.glob(os.path.join(work_dir, "jobs_failed*.txt")):
        os.remove(stale)
    if failed:
        n_sub = 1000
        parts = [failed[i : i + n_sub] for i in range(0, len(failed), n_sub)]
        for k, part in enumerate(parts):
            with open(os.path.join(work_dir, f"jobs_failed{k:03d}.txt"), "w") as fh:
                fh.write("\n".join(part) + "\n")
        print(f"  written to jobs_failed*.txt in {work_dir} ({len(parts)} file(s)); resubmit with:")
        print("  for f in jobs_failed*.txt; do condor_submit cmssw=$CMSSW_BASE jobs=$f "
              "$CMSSW_BASE/src/PhysicsTools/Tau3muNANO/scripts/submit_flatten.sub; done")
    return done


def check_content(out_dir):
    for channel in sorted(os.listdir(out_dir)):
        cdir = os.path.join(out_dir, channel)
        if not os.path.isdir(cdir):
            continue
        print(f"\n== {channel}")
        print(f"{'sample':24s} {'outputs':>7s} {'cands':>10s}  {'per fold':32s} {'labels / occupancy'}")
        totals = defaultdict(int)
        for sample in sorted(os.listdir(cdir)):
            files = sorted(glob.glob(os.path.join(cdir, sample, "*_fold*.root")))
            if not files:
                continue
            per_fold = np.zeros(5, dtype=np.int64)
            occ = {k: [] for k in LIMITS}
            lab = defaultdict(float)
            n = 0
            want = ["fold", "label_sig", "label_dsphipi", "isData"] + list(LIMITS)
            for arr in uproot.iterate([f + ":Events" for f in files], filter_name=want, step_size="200 MB"):
                n += len(arr)
                f = ak.to_numpy(arr.fold)
                per_fold += np.bincount(f, minlength=5)[:5]
                for k in LIMITS:
                    if k in arr.fields:
                        occ[k].append(ak.to_numpy(arr[k]))
                is_mc = "isData" in arr.fields and not bool(ak.any(arr.isData == 1))
                for lname in ("label_sig", "label_dsphipi"):
                    if is_mc and lname in arr.fields:
                        lab[lname] += float(ak.sum(arr[lname]))
            totals[channel] += n
            info = []
            for lname, v in lab.items():
                info.append(f"{lname} {v / max(n, 1):.2f}")
            for k, lim in LIMITS.items():
                if occ[k]:
                    v = np.concatenate(occ[k])
                    info.append(f"{k[5:-6]} p99 {np.percentile(v, 99):.0f}/{lim} ({np.mean(v > lim):.1%} over)")
            print(f"{sample:24s} {len(files):7d} {n:10d}  {str(per_fold.tolist()):32s} {'; '.join(info)}")
        print(f"{'total':24s} {'':7s} {totals[channel]:10d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", help="directory with jobs.txt and logs/")
    ap.add_argument("--out-dir", help="flattened output directory (<out-dir>/<channel>/<sample>/)")
    args = ap.parse_args()
    if not (args.work_dir or args.out_dir):
        ap.error("give --work-dir and/or --out-dir")
    if args.work_dir:
        check_jobs(args.work_dir)
    if args.out_dir:
        check_content(args.out_dir)


if __name__ == "__main__":
    main()
