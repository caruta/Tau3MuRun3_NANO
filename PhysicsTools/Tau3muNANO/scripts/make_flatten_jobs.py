#!/usr/bin/env python3
"""Build HTCondor jobs that flatten a CRAB production for transformer training.

Finds the CRAB output directories of one production tag under your storage area
(<base>/<primary dataset>/Skim<Channel>_<sample>_<tag>/<timestamp>/<NNNN>/*.root),
splits each sample's files into jobs and writes, in --work-dir:
  jobs.txt            one line per job: channel isData output_prefix filelist
  filelists/*.txt     the input files of each job
Outputs then go to <out-dir>/<channel>/<sample>/<sample>_part<k>_fold<f>.root.

  python3 make_flatten_jobs.py --base /lustre/cms/store/user/<you> --tag Transformer_v1 --year 2024 \
      --out-dir /lustre/cms/store/user/<you>/flat_Transformer_v1 --work-dir flatten_jobs
  cd flatten_jobs
  for f in jobs_part*.txt; do condor_submit cmssw=$CMSSW_BASE jobs=$f $CMSSW_BASE/src/PhysicsTools/Tau3muNANO/scripts/submit_flatten.sub; done

If a sample has several CRAB timestamps (a task submitted twice), the latest is used and
the others are listed; pass --all-timestamps to use them all.
"""
import argparse
import glob
import os
import re
import sys

CHANNELS = {"SkimTau3mu": "Tau3Mu", "SkimDsPhiPi": "DsPhiPi"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="your CRAB output area, e.g. /lustre/cms/store/user/<you>")
    ap.add_argument("--tag", default="Transformer_v1", help="production tag")
    ap.add_argument("--year", required=True)
    ap.add_argument("--out-dir", required=True, help="where the flattened files go")
    ap.add_argument("--work-dir", default="flatten_jobs", help="where jobs.txt, file lists and logs go")
    ap.add_argument("--files-per-job", type=int, default=50)
    ap.add_argument("--max-per-submit", type=int, default=1000,
                    help="jobs per condor_submit (site limit MAX_JOBS_PER_SUBMISSION, see condor_config_val)")
    ap.add_argument("--only", default=None, help="regex on '<channel>/<sample>' to restrict the samples")
    ap.add_argument("--all-timestamps", action="store_true")
    args = ap.parse_args()

    task_dirs = sorted(glob.glob(os.path.join(args.base, "*", f"Skim*_{args.year}*_{args.tag}")))
    if not task_dirs:
        sys.exit(f"no CRAB output directories match {args.base}/*/Skim*_{args.year}*_{args.tag}")

    os.makedirs(os.path.join(args.work_dir, "filelists"), exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, "logs"), exist_ok=True)
    jobs, summary = [], []
    for d in task_dirs:
        name = os.path.basename(d)
        prefix = next((p for p in CHANNELS if name.startswith(p + "_")), None)
        if prefix is None:
            continue
        channel = CHANNELS[prefix]
        sample = name[len(prefix) + 1 : -(len(args.tag) + 1)]
        if args.only and not re.search(args.only, f"{channel}/{sample}"):
            continue
        is_data = "_MC_" not in f"_{sample}_"
        stamps = sorted(glob.glob(os.path.join(d, "*")))
        if not stamps:
            continue
        if len(stamps) > 1 and not args.all_timestamps:
            print(f"WARNING {channel}/{sample}: {len(stamps)} timestamps, using {os.path.basename(stamps[-1])}; "
                  f"ignored: {[os.path.basename(s) for s in stamps[:-1]]}")
            stamps = stamps[-1:]
        files = sorted(f for s in stamps for f in glob.glob(os.path.join(s, "*", "*.root")))
        if not files:
            continue
        n = args.files_per_job
        chunks = [files[i : i + n] for i in range(0, len(files), n)]
        for k, chunk in enumerate(chunks):
            tag = f"{channel}_{sample}_part{k}"
            flist = os.path.abspath(os.path.join(args.work_dir, "filelists", tag + ".txt"))
            with open(flist, "w") as fh:
                fh.write("\n".join(chunk) + "\n")
            out_prefix = os.path.join(os.path.abspath(args.out_dir), channel, sample, f"{sample}_part{k}")
            jobs.append(f"{channel} {int(is_data)} {out_prefix} {flist}")
        summary.append((channel, sample, "data" if is_data else "MC", len(files), len(chunks)))

    with open(os.path.join(args.work_dir, "jobs.txt"), "w") as fh:
        fh.write("\n".join(jobs) + "\n")
    # one list per submission, each within the site limit; jobs.txt keeps the full list for checks
    for old in glob.glob(os.path.join(args.work_dir, "jobs_part*.txt")):
        os.remove(old)
    n_sub = args.max_per_submit
    parts = [jobs[i : i + n_sub] for i in range(0, len(jobs), n_sub)]
    for k, part in enumerate(parts):
        with open(os.path.join(args.work_dir, f"jobs_part{k:03d}.txt"), "w") as fh:
            fh.write("\n".join(part) + "\n")

    print(f"{'channel':8s} {'sample':24s} {'type':5s} {'files':>6s} {'jobs':>5s}")
    for row in summary:
        print(f"{row[0]:8s} {row[1]:24s} {row[2]:5s} {row[3]:6d} {row[4]:5d}")
    print(f"\n{len(jobs)} jobs written to {os.path.join(args.work_dir, 'jobs.txt')}, "
          f"split into {len(parts)} submissions of at most {n_sub} (jobs_part*.txt). Submit with:")
    print(f"  cd {os.path.abspath(args.work_dir)}")
    print("  for f in jobs_part*.txt; do condor_submit cmssw=$CMSSW_BASE jobs=$f "
          "$CMSSW_BASE/src/PhysicsTools/Tau3muNANO/scripts/submit_flatten.sub; done")


if __name__ == "__main__":
    main()
