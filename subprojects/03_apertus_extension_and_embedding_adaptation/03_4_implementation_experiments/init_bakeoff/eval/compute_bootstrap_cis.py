"""Bootstrap CIs over lm-eval-harness per-sample outputs.

Per cpt_plan.md v0.7 § 6.1: most benchmark runs are deterministic, so
"run 3×" doesn't establish variance. The honest variance signal comes
from bootstrap-resampling the eval items.

Reads one or more `samples_<task>.jsonl` files (from lm-eval-harness's
--log_samples flag), computes per-task mean + 95 % bootstrap CI over
1000 resamples of the sample set with replacement.

Usage:
    python3 compute_bootstrap_cis.py <eval-output-dir>
    python3 compute_bootstrap_cis.py path/to/samples_arc.jsonl path/to/samples_hellaswag.jsonl
    # or write a JSON report:
    python3 compute_bootstrap_cis.py <dir> --out report.json
"""
import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


_TIMESTAMP_SUFFIX_RE = re.compile(r"_20\d{2}-\d{2}-\d{2}T.*$")


def _task_name_from_path(path: Path) -> str:
    task_name = path.stem.replace("samples_", "", 1)
    return _TIMESTAMP_SUFFIX_RE.sub("", task_name)


def _metric_priority(task_name: str) -> Tuple[str, ...]:
    if "xquad" in task_name:
        return ("f1", "exact_match", "acc_norm", "acc", "pass@1", "score", "is_correct", "correct")
    return ("acc_norm", "acc", "f1", "exact_match", "pass@1", "score", "is_correct", "correct")


def _load_samples(path: Path) -> Tuple[str, List[float], Dict[str, object]]:
    """Read one samples_<task>.jsonl. Returns (task_name, per_sample_metric, extras).

    Per-sample metric is the binary 0/1 correctness (or per-sample log-prob)
    depending on what the task records. We try a few well-known keys and pick
    the first that's numeric.

    extras carries summary info (n_samples, metric_kind).
    """
    task_name = _task_name_from_path(path)
    values = []  # type: List[float]
    metric_kind = "unknown"
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Try known metric keys in order of preference
            metric_val = None
            for key in _metric_priority(task_name):
                if key in row:
                    metric_val = row[key]
                    metric_kind = key
                    break
            # Fallbacks: some tasks log per-target probs
            if metric_val is None and "loglikelihood" in row:
                # average log-likelihood; not great for bootstrap on accuracy, but better than nothing
                metric_val = float(row["loglikelihood"])
                metric_kind = "loglikelihood"
            if metric_val is None:
                continue
            try:
                values.append(float(metric_val))
            except (TypeError, ValueError):
                continue
    return task_name, values, {"n_samples": len(values), "metric_kind": metric_kind}


def _bootstrap_ci(
    values: List[float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 20_260_520,
) -> Tuple[float, float, float, float]:
    """Return (mean, ci_low, ci_high, std_of_means)."""
    if not values:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(arr)
    boot_means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = arr[idx].mean()
    mean = float(arr.mean())
    alpha = (1.0 - confidence) / 2.0
    ci_low = float(np.quantile(boot_means, alpha))
    ci_high = float(np.quantile(boot_means, 1.0 - alpha))
    std = float(boot_means.std(ddof=1))
    return (mean, ci_low, ci_high, std)


def _gather_paths(inputs: List[Path]) -> List[Path]:
    paths = []  # type: List[Path]
    for p in inputs:
        if p.is_dir():
            paths.extend(sorted(p.glob("samples_*.jsonl")))
        elif p.is_file():
            paths.append(p)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path,
                    help="one or more eval-output dirs OR samples_<task>.jsonl files")
    ap.add_argument("--n-resamples", type=int, default=1000)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--out", type=Path, default=None,
                    help="if given, write JSON report there; otherwise print to stdout")
    ap.add_argument("--seed", type=int, default=20_260_520)
    args = ap.parse_args()

    paths = _gather_paths(args.inputs)
    if not paths:
        print(f"ERROR: no samples_*.jsonl files found in inputs", file=sys.stderr)
        return 2

    report = {
        "n_resamples": args.n_resamples,
        "confidence": args.confidence,
        "seed": args.seed,
        "tasks": {},
    }  # type: Dict[str, object]

    table_rows = []  # type: List[Tuple[str, int, str, float, float, float, float]]
    for path in paths:
        task, values, extras = _load_samples(path)
        if not values:
            print(f"  [SKIP] {path}: no numeric per-sample metric found", file=sys.stderr)
            continue
        mean, lo, hi, std = _bootstrap_ci(values, args.n_resamples, args.confidence, args.seed)
        report["tasks"][task] = {
            "n_samples": extras["n_samples"],
            "metric_kind": extras["metric_kind"],
            "mean": mean,
            "ci_low": lo,
            "ci_high": hi,
            "ci_halfwidth": (hi - lo) / 2.0,
            "boot_std": std,
        }
        table_rows.append((task, extras["n_samples"], extras["metric_kind"], mean, lo, hi, std))

    # Pretty-print
    table_rows.sort()
    print(f"\nBootstrap CIs (n_resamples={args.n_resamples}, confidence={args.confidence})")
    print(f"  {'task':<28} {'N':>6} {'metric':>18} {'mean':>8} {'ci_low':>8} {'ci_high':>8} {'±halfwidth':>10}")
    for task, n, kind, mean, lo, hi, std in table_rows:
        hw = (hi - lo) / 2.0
        print(f"  {task:<28} {n:>6} {kind:>18} {mean:>8.4f} {lo:>8.4f} {hi:>8.4f} {hw:>10.4f}")

    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
