#!/usr/bin/env python3
"""Collect the per-set held-out LOSS trajectories from a curriculum run's .out logs.

Reads the trainer's printed lines (reliable; the TensorBoard scalar collides across sets — the M1
bug — unless UPSTREAM EDIT for the TB key is applied):
    validation loss at iteration <N> [<name>] | lm loss value: <X> | lm loss PPL: <Y>
for name in the 9 extra-valid sets:
  ADAPTATION (new-Greek):  hplt openarchives greek_phd
  FORGETTING (old-data):   english de ru zh code old_greek
Emits forgetting_loss.csv: run_tag, iter, tokens, name, kind, lm_loss, ppl, delta_from_first
where delta_from_first = lm_loss - (this set's first recorded loss), the within-arm forgetting signal.
NOTE: per-token loss is NOT comparable across arms (ext vs base tokenizer) — compare within a single
arm/sweep, or convert to bits-per-byte externally.

Usage: python collect_forgetting_loss.py <OUTPUT_DIR ...>   (one or more run dirs under RUN_ROOT)
"""
import csv, glob, os, re, sys

NEW_GREEK = {"hplt", "openarchives", "greek_phd"}
FORGET = {"english", "de", "ru", "zh", "code", "old_greek"}
TPI = 4194304
LINE = re.compile(r"validation loss at iteration (\d+) \[(\w+)\] \| lm loss value: ([\d.E+-]+) \| lm loss PPL: ([\d.E+-]+)")

rows = []
for run_dir in sys.argv[1:]:
    run_tag = os.path.basename(run_dir.rstrip("/"))
    seen = {}
    for fp in sorted(glob.glob(os.path.join(run_dir, "*.out"))):
        for line in open(fp, errors="ignore"):
            m = LINE.search(line)
            if not m:
                continue
            it, name, loss, ppl = int(m[1]), m[2], float(m[3]), float(m[4])
            kind = "adapt" if name in NEW_GREEK else ("forget" if name in FORGET else "other")
            first = seen.setdefault((run_tag, name), loss)
            rows.append([run_tag, it, it * TPI, name, kind, loss, ppl, round(loss - first, 5)])

rows.sort(key=lambda r: (r[0], r[3], r[1]))
out = "forgetting_loss.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run_tag", "iter", "tokens", "name", "kind", "lm_loss", "ppl", "delta_from_first"])
    w.writerows(rows)
print(f"wrote {len(rows)} rows -> {out}")
# quick summary: final forgetting delta per set
final = {}
for r in rows:
    final[(r[0], r[3])] = r
for (rt, name), r in sorted(final.items()):
    if r[4] == "forget":
        print(f"  {rt}  {name:9s} final lm_loss {r[5]:.4f}  delta_from_first {r[7]:+.4f}")
