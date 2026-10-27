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
import sys
import time
from collections import defaultdict

import awkward as ak
import numpy as np
import uproot

LIMITS = {"cand_nTrk_total": 128, "cand_nNeu_total": 32, "cand_nSV_total": 8}
HMAX = 2000  # occupancy histograms up to this count


def check_jobs(work_dir):
    if not os.path.isfile(os.path.join(work_dir, "jobs.txt")):
        sys.exit(f"no jobs.txt in {work_dir}")
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


def percentile_from_hist(h, q):
    c = np.cumsum(h)
    return int(np.searchsorted(c, q / 100.0 * c[-1])) if c[-1] else 0


def check_content(out_dir, only=None, max_files=0):
    """Per sample: candidates, folds, labels, token occupancy. Constant memory: occupancies
    are accumulated as histograms, files are read one at a time, unreadable files are listed."""
    if not os.path.isdir(out_dir):
        sys.exit(f"output directory not found: {out_dir}")
    want = ["fold", "label_sig", "label_dsphipi", "isData"] + list(LIMITS)
    bad_files = []
    for channel in sorted(os.listdir(out_dir)):
        cdir = os.path.join(out_dir, channel)
        if not os.path.isdir(cdir):
            continue
        print(f"\n== {channel}")
        print(f"{'sample':24s} {'outputs':>7s} {'cands':>10s}  {'per fold':32s} {'labels / occupancy'}")
        total = 0
        for sample in sorted(os.listdir(cdir)):
            if only and not re.search(only, f"{channel}/{sample}"):
                continue
            files = sorted(glob.glob(os.path.join(cdir, sample, "*_fold*.root")))
            if max_files:
                files = files[:max_files]
            if not files:
                continue
            t0 = time.time()
            per_fold = np.zeros(5, dtype=np.int64)
            hist = {k: np.zeros(HMAX + 1, dtype=np.int64) for k in LIMITS}
            lab = defaultdict(float)
            n = n_mc = 0
            for i, path in enumerate(files):
                if i % 10 == 0:
                    print(f"  {channel}/{sample}: file {i + 1}/{len(files)}, {n} candidates, "
                          f"{time.time() - t0:.0f} s", file=sys.stderr, flush=True)
                try:
                    with uproot.open(path) as fh:
                        tree = fh["Events"]
                        arr = tree.arrays([b for b in want if b in tree.keys()], library="np")
                except Exception as e:  # unreadable or incomplete file
                    bad_files.append(f"{path}: {type(e).__name__}: {str(e)[:120]}")
                    continue
                m = len(arr["fold"])
                n += m
                per_fold += np.bincount(arr["fold"].astype(np.int64), minlength=5)[:5]
                for k in LIMITS:
                    if k in arr:
                        v = np.clip(arr[k].astype(np.int64), 0, HMAX)
                        hist[k] += np.bincount(v, minlength=HMAX + 1)
                if "isData" in arr and not np.any(arr["isData"] == 1):
                    n_mc += m
                    for lname in ("label_sig", "label_dsphipi"):
                        if lname in arr:
                            lab[lname] += float(np.sum(arr[lname]))
            total += n
            info = [f"{lname} {v / max(n_mc, 1):.2f}" for lname, v in lab.items()]
            for k, lim in LIMITS.items():
                h = hist[k]
                if h.sum():
                    over = h[lim + 1 :].sum() / h.sum()
                    info.append(f"{k[5:-6]} p99 {percentile_from_hist(h, 99)}/{lim} ({over:.1%} over)")
            print(f"{sample:24s} {len(files):7d} {n:10d}  {str(per_fold.tolist()):32s} {'; '.join(info)}")
        print(f"{'total':24s} {'':7s} {total:10d}")
    if bad_files:
        print(f"\n{len(bad_files)} unreadable file(s): rerun their flattening jobs")
        for b in bad_files:
            print("  " + b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", help="directory with jobs.txt and logs/")
    ap.add_argument("--out-dir", help="flattened output directory (<out-dir>/<channel>/<sample>/)")
    ap.add_argument("--only", default=None, help="regex on '<channel>/<sample>', e.g. 'Tau3Mu/2024eraC'")
    ap.add_argument("--max-files", type=int, default=0, help="read at most N files per sample (quick look)")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # show results as they come, also through a pipe
    if not (args.work_dir or args.out_dir):
        ap.error("give --work-dir and/or --out-dir")
    if args.work_dir:
        check_jobs(args.work_dir)
    if args.out_dir:
        check_content(args.out_dir, args.only, args.max_files)


if __name__ == "__main__":
    main()
