"""Path-A 0.5B geometry probe bootstrap CIs.

Methodology IDENTICAL to V4 v3 (reports/v4_bootstrap_cis_native_mcq.json):
  - per-task item-level resampling (independent within each task),
  - macro-mean across the 3 headline tasks per resample,
  - 1000 resamples, 95% percentile, rng_seed=20260529.
  - Paired deltas use the SAME resample indices on both models of a pair
    so the diff is paired.

Inputs (workspace-relative, copied from Clariden into predictions_tmp/):
  - Vanilla-Path-A-0.5B            -- the probe iter 119 (model under test)
  - iter-119-Vanilla-0.5B-Path-B   -- Path B iter 119 (Task 1 matched-tokens)
  - Apertus-Base                   -- released Path A base (geometry baseline)
  - Apertus-Base-matched-Path-B-perturbed
                                   -- matched-config Path-B-perturbed bookend

Output:
  reports/v4_workspace_path_a/path_a_probe_bootstrap_cis.json
"""

from __future__ import annotations

import json
import datetime as _dt
from pathlib import Path
from collections import defaultdict

import numpy as np

WORKSPACE = Path(__file__).resolve().parent
PRED_DIR = WORKSPACE / "predictions_tmp"
ARTIFACT = WORKSPACE / "path_a_probe_bootstrap_cis.json"

HEADLINE_TASKS = ["greekmmlu", "ilsp_medical_mcqa", "ilsp_mcqa_asep"]
ALL_TASKS = ["greekmmlu", "ilsp_medical_mcqa", "ilsp_mcqa_asep", "plutus_qa"]

N_RESAMPLES = 1000
CI_LEVEL = 0.95
RNG_SEED = 20260529

MODEL_UNDER_TEST = "Vanilla-Path-A-0.5B"

# Local copies and the source paths to record in the artifact.
MODELS_LOCAL = {
    "Vanilla-Path-A-0.5B": PRED_DIR / "Vanilla-Path-A-0.5B_native_mcq_predictions.jsonl",
    "Vanilla-Path-B-0.5B-iter119": PRED_DIR / "iter119_Vanilla-0.5B_PathB_native_mcq_predictions.jsonl",
    "Apertus-Base": PRED_DIR / "Apertus-Base_native_mcq_predictions.jsonl",
    "Apertus-Base-matched-Path-B-perturbed": PRED_DIR / "Apertus-Base-matched-Path-B-perturbed_native_mcq_predictions.jsonl",
}

MODELS_SOURCE_PATHS = {
    "Vanilla-Path-A-0.5B": (
        "/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/"
        "eval_04a_vanilla_path_a_probe_20260531T103924Z/iter_0000119/native_mcq/"
        "Vanilla-Path-A-0.5B_native_mcq_predictions.jsonl"
    ),
    "Vanilla-Path-B-0.5B-iter119": (
        "/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/"
        "eval_04_vanilla_goldfish_5b_20260528T112539Z/iter_0000119/native_mcq/"
        "Vanilla-0.5B_native_mcq_predictions.jsonl"
    ),
    "Apertus-Base": (
        "/capstor/scratch/cscs/fffoivos/runs/eval/native_greek_suite_20260526/"
        "mcq_all_checkpoints/chunk_1/Apertus-Base_/"
        "Apertus-Base_native_mcq_predictions.jsonl"
    ),
    "Apertus-Base-matched-Path-B-perturbed": (
        "/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/"
        "eval_apertus_base_matched_rope500k_seq4096/native_mcq/"
        "Apertus-Base-matched-rope500k-seq4096_native_mcq_predictions.jsonl"
    ),
}

PAIRS = [
    ("Vanilla-Path-A-0.5B", "Vanilla-Path-B-0.5B-iter119"),
    ("Vanilla-Path-A-0.5B", "Apertus-Base"),
    ("Vanilla-Path-A-0.5B", "Apertus-Base-matched-Path-B-perturbed"),
]
METRICS = ["headline_3task"] + ALL_TASKS


def load_correctness(path: Path) -> dict[str, np.ndarray]:
    """Return {benchmark: 1-D int array of 0/1 in example_id order}."""
    rows_by_bench: dict[str, list[tuple[str, int]]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            rows_by_bench[d["benchmark"]].append((d["example_id"], int(bool(d["correct"]))))
    out = {}
    for b, rows in rows_by_bench.items():
        # Predictions are written in fixed eval order; example_ids align across
        # files (verified upstream). Preserve insertion order.
        out[b] = np.array([r[1] for r in rows], dtype=np.int32)
    return out


def percentile_ci(samples: np.ndarray, level: float) -> tuple[float, float]:
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def main() -> None:
    correctness: dict[str, dict[str, np.ndarray]] = {}
    for name, p in MODELS_LOCAL.items():
        correctness[name] = load_correctness(p)
        print(f"loaded {name}: " + ", ".join(
            f"{b}=n{len(v)}/acc{v.mean():.4f}" for b, v in correctness[name].items()
        ))

    # Per-task resample index matrices, identical across models -> paired.
    rng = np.random.default_rng(RNG_SEED)
    task_index_matrices: dict[str, np.ndarray] = {}
    n_items_per_task: dict[str, int] = {}
    for task in ALL_TASKS:
        n_items = len(next(iter(correctness.values()))[task])
        n_items_per_task[task] = n_items
        task_index_matrices[task] = rng.integers(0, n_items, size=(N_RESAMPLES, n_items))

    def resampled_task_acc(model: str, task: str) -> np.ndarray:
        arr = correctness[model][task]
        idx = task_index_matrices[task]
        return arr[idx].mean(axis=1)

    def model_block(model: str) -> dict:
        per_task = {}
        for task in ALL_TASKS:
            point = float(correctness[model][task].mean())
            samples = resampled_task_acc(model, task)
            lo, hi = percentile_ci(samples, CI_LEVEL)
            per_task[task] = {
                "point": point,
                "lo_95": lo,
                "hi_95": hi,
                "n_items": n_items_per_task[task],
            }
        headline3_samples = np.mean(
            np.stack([resampled_task_acc(model, t) for t in HEADLINE_TASKS], axis=0),
            axis=0,
        )
        h3_point = float(np.mean([correctness[model][t].mean() for t in HEADLINE_TASKS]))
        h3_lo, h3_hi = percentile_ci(headline3_samples, CI_LEVEL)
        headline_3task = {
            "point": h3_point,
            "lo_95": h3_lo,
            "hi_95": h3_hi,
            "n_items_per_task": [n_items_per_task[t] for t in HEADLINE_TASKS],
            "tasks": list(HEADLINE_TASKS),
        }
        return {
            "label": model,
            "source_path": MODELS_SOURCE_PATHS[model],
            "headline_3task": headline_3task,
            "per_task": per_task,
        }

    def metric_samples(model: str, metric: str) -> np.ndarray:
        if metric == "headline_3task":
            return np.mean(
                np.stack([resampled_task_acc(model, t) for t in HEADLINE_TASKS], axis=0),
                axis=0,
            )
        return resampled_task_acc(model, metric)

    def metric_point(model: str, metric: str) -> float:
        if metric == "headline_3task":
            return float(np.mean([correctness[model][t].mean() for t in HEADLINE_TASKS]))
        return float(correctness[model][metric].mean())

    def metric_n_items(metric: str) -> list[int] | int:
        if metric == "headline_3task":
            return [n_items_per_task[t] for t in HEADLINE_TASKS]
        return n_items_per_task[metric]

    delta_rows = []
    for a, b in PAIRS:
        for metric in METRICS:
            a_samples = metric_samples(a, metric)
            b_samples = metric_samples(b, metric)
            diff = a_samples - b_samples  # paired by index construction
            delta = metric_point(a, metric) - metric_point(b, metric)
            lo, hi = percentile_ci(diff, CI_LEVEL)
            outside = (lo > 0) or (hi < 0)
            delta_rows.append({
                "a": a,
                "b": b,
                "metric": metric,
                "delta": delta,
                "delta_ci_lo": lo,
                "delta_ci_hi": hi,
                "outside_zero_ci": bool(outside),
                "n_items_per_task": metric_n_items(metric),
            })

    # Assemble model_under_test block + baseline blocks (baselines included as
    # context so the artifact is self-contained).
    artifact = {
        "schema": "path-a-probe-bootstrap-cis-v1",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "n_resamples": N_RESAMPLES,
        "ci_level": CI_LEVEL,
        "rng_seed": RNG_SEED,
        "methodology": (
            "per-task item-level paired bootstrap, same as "
            "reports/v4_bootstrap_cis_native_mcq.json v3 (1000 resamples, "
            "95% percentile, rng_seed=20260529, paired delta uses same "
            "per-task indices for both sides)."
        ),
        "headline_tasks": HEADLINE_TASKS,
        "all_tasks": ALL_TASKS,
        "model_under_test": model_block(MODEL_UNDER_TEST),
        "baselines": {
            name: model_block(name)
            for name in MODELS_LOCAL
            if name != MODEL_UNDER_TEST
        },
        "delta_table": delta_rows,
    }

    ARTIFACT.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {ARTIFACT}")

    # Print all delta rows for verdict reading
    print("\nDelta table (all rows):")
    for row in delta_rows:
        flag = "*" if row["outside_zero_ci"] else " "
        print(
            f"  {flag} {row['a']} vs {row['b']} | {row['metric']:>20s} "
            f"delta={row['delta']:+.4f} CI=[{row['delta_ci_lo']:+.4f},{row['delta_ci_hi']:+.4f}]"
        )

    # Print headline marginal
    h3 = artifact["model_under_test"]["headline_3task"]
    print(
        f"\nPath A 0.5B headline_3task: point={h3['point']:.4f} "
        f"CI=[{h3['lo_95']:.4f},{h3['hi_95']:.4f}]"
    )


if __name__ == "__main__":
    main()
