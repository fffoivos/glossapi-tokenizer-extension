#!/usr/bin/env python3
"""Summarize live bakeoff training logs into JSON/CSV/Markdown.

The bakeoff writes one Slurm stdout file per arm, with Megatron progress lines
like:

  iteration       81/     476 | consumed tokens: 0.340B | ... | lm loss: ...

This helper keeps the loss/throughput evidence reproducible. Raw Megatron
`lm loss` is per-target-token CE in nats, so it is a health/within-arm metric
only when comparing different tokenizers. When patched logs contain tokenizer-
fair fields (`bpb`, `bpt`, `base_loss`, `new_loss`, `n_new`), this script
captures them and surfaces BPB as the dense cross-tokenizer training signal.
Checkpoint heldout BPC/BPB remains the selection metric.

Example:
  python3 summarize_training_logs.py \
    --run-tag bakeoff_1node_chain_20260522_005620 \
    --log-root /capstor/scratch/cscs/fffoivos/runs/bakeoff \
    --json-out /capstor/.../training_summary.json \
    --csv-out /capstor/.../training_curve.csv \
    --md-out /capstor/.../training_summary.md
"""
import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path


PROGRESS_RE = re.compile(
    r"\[(?P<timestamp>[^\]]+)\]\s+iteration\s+(?P<iteration>\d+)\s*/\s*(?P<total>\d+)"
    r".*?consumed samples:\s+(?P<consumed_samples>\d+)"
    r".*?consumed tokens:\s+(?P<consumed_tokens>[0-9.]+)B"
    r".*?elapsed time per iteration \(ms\):\s+(?P<elapsed_ms>[0-9.]+)"
    r".*?tokens/sec/gpu:\s+(?P<tokens_sec_gpu>[0-9.]+)"
    r".*?learning rate:\s+(?P<learning_rate>[0-9.Ee+-]+)"
    r".*?lm loss:\s+(?P<lm_loss>[0-9.Ee+-]+)"
    r".*?grad norm:\s+(?P<grad_norm>[0-9.Ee+-]+)"
    r".*?params norm:\s+(?P<params_norm>[0-9.Ee+-]+)"
    r".*?number of skipped iterations:\s+(?P<skipped>\d+)"
    r".*?number of nan iterations:\s+(?P<nan>\d+)"
)

FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?|nan|NaN|inf|Inf|-inf|-Inf"


def _float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _extract_float(line, *names):
    for name in names:
        match = re.search(r"\b%s:\s*(%s)" % (re.escape(name), FLOAT_RE), line)
        if match:
            return _float(match.group(1))
    return None


def _extract_int(line, name):
    match = re.search(r"\b%s:\s*(\d+)" % re.escape(name), line)
    return int(match.group(1)) if match else None


def _parse_log(path, arm):
    rows = []
    try:
        fp = path.open("r", encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return rows
    with fp:
        for line in fp:
            match = PROGRESS_RE.search(line)
            if not match:
                continue
            g = match.groupdict()
            rows.append({
                "arm": arm,
                "timestamp": g["timestamp"],
                "iteration": int(g["iteration"]),
                "total_iterations": int(g["total"]),
                "consumed_samples": int(g["consumed_samples"]),
                "consumed_tokens_b": _float(g["consumed_tokens"]),
                "elapsed_ms": _float(g["elapsed_ms"]),
                "tokens_sec_gpu": _float(g["tokens_sec_gpu"]),
                "learning_rate": _float(g["learning_rate"]),
                "lm_loss": _float(g["lm_loss"]),
                "bpb": _extract_float(line, "bpb"),
                "bpt": _extract_float(line, "bpt", "bytes_per_token_batch"),
                "base_loss": _extract_float(line, "base_loss"),
                "new_loss": _extract_float(line, "new_loss"),
                "n_new": _extract_int(line, "n_new"),
                "grad_norm": _float(g["grad_norm"]),
                "params_norm": _float(g["params_norm"]),
                "skipped_iterations": int(g["skipped"]),
                "nan_iterations": int(g["nan"]),
                "log_path": str(path),
            })
    return rows


def _default_log_path(log_root, arm, job_id):
    if job_id:
        return log_root / ("bakeoff_%s-%s.out" % (arm, job_id))
    matches = sorted(log_root.glob("bakeoff_%s-*.out" % arm))
    return matches[-1] if matches else log_root / ("bakeoff_%s-UNKNOWN.out" % arm)


def _summarize_rows(rows):
    if not rows:
        return {
            "n_points": 0,
            "latest": None,
            "min_lm_loss": None,
            "max_skipped_iterations": None,
            "max_nan_iterations": None,
        }
    losses = [r["lm_loss"] for r in rows if r["lm_loss"] is not None]
    bpb = [r["bpb"] for r in rows if r.get("bpb") is not None]
    skipped = [r["skipped_iterations"] for r in rows]
    nans = [r["nan_iterations"] for r in rows]
    return {
        "n_points": len(rows),
        "latest": rows[-1],
        "min_lm_loss": min(losses) if losses else None,
        "min_bpb": min(bpb) if bpb else None,
        "has_tokenizer_fair_training_metrics": bool(bpb),
        "max_skipped_iterations": max(skipped) if skipped else None,
        "max_nan_iterations": max(nans) if nans else None,
    }


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "arm", "timestamp", "iteration", "total_iterations", "consumed_tokens_b",
        "lm_loss", "bpb", "bpt", "base_loss", "new_loss", "n_new",
        "tokens_sec_gpu", "learning_rate", "grad_norm", "params_norm",
        "skipped_iterations", "nan_iterations", "log_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _fmt(value, fmt):
    if value is None:
        return "-"
    try:
        return fmt % value
    except TypeError:
        return str(value)


def _write_md(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Bakeoff Training Summary",
        "",
        "- run tag: `%s`" % payload["run_tag"],
        "- generated at: `%s`" % payload["generated_at_utc"],
        "- selection note: raw `lm_loss` is tokenizer-dependent; use heldout BPC/BPB, or dense `bpb` when present, for cross-tokenizer comparisons.",
        "",
        "| arm | points | latest iter | tokens B | lm loss | bpb | bpt | new targets | tok/s/gpu | skipped | nan |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in payload["arm_order"]:
        summary = payload["arms"][arm]
        latest = summary.get("latest")
        lines.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            arm,
            summary["n_points"],
            latest["iteration"] if latest else "-",
            _fmt(latest.get("consumed_tokens_b") if latest else None, "%.3f"),
            _fmt(latest.get("lm_loss") if latest else None, "%.4f"),
            _fmt(latest.get("bpb") if latest else None, "%.4f"),
            _fmt(latest.get("bpt") if latest else None, "%.3f"),
            latest.get("n_new") if latest and latest.get("n_new") is not None else "-",
            _fmt(latest.get("tokens_sec_gpu") if latest else None, "%.0f"),
            summary["max_skipped_iterations"] if latest else "-",
            summary["max_nan_iterations"] if latest else "-",
        ))
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--log-root", type=Path, default=Path("/capstor/scratch/cscs/fffoivos/runs/bakeoff"))
    ap.add_argument("--arms", nargs="+", default=["vanilla", "retok", "centroid"])
    ap.add_argument("--job-ids", nargs="*", default=[],
                    help="optional arm=jobid entries, e.g. vanilla=2341822")
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--csv-out", type=Path)
    ap.add_argument("--md-out", type=Path)
    args = ap.parse_args()

    job_ids = {}
    for item in args.job_ids:
        if "=" not in item:
            print("ERROR: --job-ids entries must look like arm=jobid: %s" % item, file=sys.stderr)
            return 2
        arm, job_id = item.split("=", 1)
        job_ids[arm] = job_id

    all_rows = []
    arms = {}
    for arm in args.arms:
        path = _default_log_path(args.log_root, arm, job_ids.get(arm))
        rows = _parse_log(path, arm)
        all_rows.extend(rows)
        arms[arm] = _summarize_rows(rows)

    payload = {
        "run_tag": args.run_tag,
        "generated_at_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "log_root": str(args.log_root),
        "arm_order": args.arms,
        "arms": arms,
        "rows": all_rows,
    }

    if args.json_out:
        _write_json(args.json_out, payload)
    if args.csv_out:
        _write_csv(args.csv_out, all_rows)
    if args.md_out:
        _write_md(args.md_out, payload)

    _write_md(Path("/tmp/bakeoff_training_summary.md"), payload)
    print(Path("/tmp/bakeoff_training_summary.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
