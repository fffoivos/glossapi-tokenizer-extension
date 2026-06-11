#!/usr/bin/env python3
"""Collect the GreekMMLU adaptation trajectory for one or more curriculum runs.

GreekMMLU overall = sum(correct)/sum(n) over the per-subject rows (benchmark=='greekmmlu') in each
checkpoint's *_native_mcq_summary.csv (header: benchmark,subject,n,accuracy,correct). The summary also
has one greekmmlu,__all__ row, which we SKIP and recompute from the per-subject rows (round(acc*n) ==
the `correct` column, so our sum-of-subjects reproduces the __all__ accuracy exactly).
Walks $RUN_ROOT/eval_<run_tag>/iter_*/native_mcq/*_native_mcq_summary.csv. Emits a tidy CSV:
  run_tag, iter, tokens, greekmmlu_acc, n
so the replay/LR sweep grids can be compared directly.

Usage: python collect_greekmmlu.py <RUN_ROOT> [out.csv]
"""
import csv, glob, os, re, sys

RUN_ROOT = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(RUN_ROOT, "greekmmlu_trajectory.csv")
TPI = 4194304  # tokens/iter


def greekmmlu_overall(summary_csv):
    tot_n = tot_c = 0
    for r in csv.DictReader(open(summary_csv)):
        if (r.get("benchmark") or r.get("task")) != "greekmmlu":
            continue
        subj = r.get("subject", "")
        if subj == "__all__":
            continue
        try:
            n = int(float(r.get("n") or r.get("samples") or 0))
            acc = float(r.get("accuracy"))
            tot_n += n
            tot_c += round(acc * n)
        except Exception:
            pass
    return (tot_c / tot_n, tot_n) if tot_n else (None, 0)


rows = []
for ev in sorted(glob.glob(os.path.join(RUN_ROOT, "eval_*"))):
    run_tag = os.path.basename(ev)[len("eval_"):]
    for d in sorted(glob.glob(os.path.join(ev, "iter_*"))):
        m = re.search(r"iter_(\d+)", os.path.basename(d))
        if not m:
            continue
        it = int(m.group(1))
        cands = glob.glob(os.path.join(d, "native_mcq", "*_native_mcq_summary.csv"))
        if not cands:
            continue
        acc, n = greekmmlu_overall(sorted(cands)[-1])
        if acc is not None:
            rows.append((run_tag, it, it * TPI, round(acc, 4), n))

rows.sort()
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run_tag", "iter", "tokens", "greekmmlu_acc", "n"])
    w.writerows(rows)
print(f"wrote {len(rows)} points -> {OUT}")
for r in rows:
    print("  ", r)
