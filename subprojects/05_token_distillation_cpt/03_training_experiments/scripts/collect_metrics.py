#!/usr/bin/env python3
"""Parse Megatron per-iteration training logs into a tidy CSV + cross-arm summary.

The trainer already logs every iteration to tensorboard (--log-interval 1,
--log-throughput, --log-params-norm, --log-memory-to-tensorboard) AND to the
.out files. This is the meticulous, durable record. This script turns the .out
lines into a CSV for quick CLI review and side-by-side comparison of the two
arms (vanilla vs modern-greek-td) — it does NOT replace tensorboard.

Usage:
  collect_metrics.py --arm vanilla:/capstor/.../cpt13b_vanilla_*/*.out \
                     --arm td:/capstor/.../cpt13b_td_*/*.out \
                     --out metrics.csv
  # or a single run:
  collect_metrics.py run.out                      # prints summary, writes run.out.csv
"""
import argparse, glob, os, re, sys, csv

# One Megatron training line, e.g.:
#  iteration  321/  476 | consumed samples: 328704 | consumed tokens: 1.346B |
#  elapsed time per iteration (ms): 139102.6 | ... | learning rate: 3.92E-06 |
#  global batch size: 1024 | lm loss: 2.629754E+00 | loss scale: 1.0 |
#  grad norm: 1.302 | params norm: 7140.780 | number of skipped iterations: 0 |
#  number of nan iterations: 0 |
TRAIN_FIELDS = [
    ("iteration",      r"iteration\s+(\d+)\s*/"),
    ("consumed_tokens", r"consumed tokens:\s*([\d.]+)\s*([BMK]?)"),
    ("ms_per_iter",    r"elapsed time per iteration \(ms\):\s*([\d.]+)"),
    ("tok_s_gpu",      r"tokens/sec/gpu:\s*([\d.]+)"),
    ("tflops_gpu",     r"TFLOP/s/GPU\):\s*([\d.]+)"),
    ("lr",             r"learning rate:\s*([\d.E+\-]+)"),
    ("lm_loss",        r"lm loss:\s*([\d.E+\-]+)"),
    ("grad_norm",      r"grad norm:\s*([\d.E+\-]+)"),
    ("params_norm",    r"params norm:\s*([\d.E+\-]+)"),
    ("skipped",        r"number of skipped iterations:\s*(\d+)"),
    ("nan",            r"number of nan iterations:\s*(\d+)"),
]
_SCALE = {"B": 1e9, "M": 1e6, "K": 1e3, "": 1.0}
VALID_RE = re.compile(
    r"validation loss at iteration\s+(\d+)"
    r"(?:[^\[]*\[([^\]]+)\])?"
    r".*?lm loss value:\s*([\d.E+\-]+)"
)
COLS = [
    "arm", "metric_type", "validation_set", "iteration", "consumed_tokens",
    "lm_loss", "lr", "grad_norm", "tflops_gpu", "tok_s_gpu", "ms_per_iter",
    "params_norm", "skipped", "nan",
]


def parse_training_line(line):
    if "lm loss:" not in line or "iteration" not in line:
        return None
    row = {}
    for name, pat in TRAIN_FIELDS:
        m = re.search(pat, line)
        if not m:
            return None
        if name == "consumed_tokens":
            row[name] = float(m.group(1)) * _SCALE.get(m.group(2), 1.0)
        elif name in ("iteration", "skipped", "nan"):
            row[name] = int(m.group(1))
        else:
            row[name] = float(m.group(1))
    row["metric_type"] = "train"
    row["validation_set"] = ""
    return row


def parse_validation_line(line):
    if "validation loss at iteration" not in line or "lm loss value:" not in line:
        return None
    m = VALID_RE.search(line)
    if not m:
        return None
    return {
        "metric_type": "valid",
        "validation_set": m.group(2) or "default",
        "iteration": int(m.group(1)),
        "consumed_tokens": "",
        "lm_loss": float(m.group(3)),
        "lr": "",
        "grad_norm": "",
        "tflops_gpu": "",
        "tok_s_gpu": "",
        "ms_per_iter": "",
        "params_norm": "",
        "skipped": "",
        "nan": "",
    }


def parse_line(line):
    return parse_training_line(line) or parse_validation_line(line)


def parse_files(paths):
    rows, last_train_iter = [], -1
    for p in sorted(paths):
        for line in open(p, encoding="utf-8", errors="replace"):
            r = parse_line(line)
            if not r:
                continue
            if r["metric_type"] == "train":
                if r["iteration"] == last_train_iter:  # dedup resume-overlap repeats
                    continue
                last_train_iter = r["iteration"]
            rows.append(r)
    return rows


def summarize(arm, rows):
    train_rows = [r for r in rows if r["metric_type"] == "train"]
    valid_rows = [r for r in rows if r["metric_type"] == "valid"]
    if not train_rows:
        print(f"  {arm}: no iteration lines found"); return
    losses = [r["lm_loss"] for r in train_rows]
    nans = sum(r["nan"] for r in train_rows); skips = sum(r["skipped"] for r in train_rows)
    last = train_rows[-1]
    valid_sets = sorted({r["validation_set"] for r in valid_rows})
    valid_note = f" | valid sets {','.join(valid_sets)}" if valid_sets else ""
    print(f"  {arm:14s} iters {train_rows[0]['iteration']}–{last['iteration']} | "
          f"last loss {last['lm_loss']:.4f} | min loss {min(losses):.4f} | "
          f"lr {last['lr']:.3e} | {last['tflops_gpu']:.0f} TFLOP/s/gpu | "
          f"nan {nans} skip {skips}{valid_note}" + ("  ⚠ NaN/skip!" if (nans or skips) else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*", help="one or more .out files (single-arm mode)")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME:GLOB",
                    help="repeatable; e.g. --arm vanilla:/path/*.out")
    ap.add_argument("--out", default=None, help="combined CSV output path")
    a = ap.parse_args()

    arms = {}
    for spec in a.arm:
        name, _, g = spec.partition(":")
        arms[name] = parse_files(glob.glob(g))
    if a.logs:
        arms.setdefault("run", []).extend(parse_files([p for g in a.logs for p in glob.glob(g)]))
    if not arms:
        ap.error("provide .out file(s) or --arm NAME:GLOB")

    out = a.out or (a.logs[0] + ".csv" if a.logs else "metrics.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
        for name, rows in arms.items():
            for r in rows:
                w.writerow({"arm": name, **r})
    print(f"wrote {sum(len(r) for r in arms.values())} rows -> {out}\n")
    print("Summary (per arm):")
    for name, rows in arms.items():
        summarize(name, rows)


if __name__ == "__main__":
    main()
