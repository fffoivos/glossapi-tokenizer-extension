"""V4 bootstrap re-emit (v3).

Adds iter-834-Vanilla-3.5B + iter-1192-Vanilla-5B model entries and the
seven new paired comparisons requested. Methodology identical to v1/v2:
  per-task item-level resampling (independent within each task), macro-mean
  across 3 headline tasks per resample, 1000 resamples, 95% percentile,
  rng_seed=20260529. Paired deltas use the same resample indices on both
  models of a pair so the diff is paired.

For each pair we emit 5 metric rows: headline_3task + 4 per-task.
"""

from __future__ import annotations

import json
import datetime as _dt
from pathlib import Path
from collections import defaultdict

import numpy as np

WORKSPACE = Path(__file__).resolve().parent
REPORTS = WORKSPACE.parent
ARTIFACT = REPORTS / "v4_bootstrap_cis_native_mcq.json"

HEADLINE_TASKS = ["greekmmlu", "ilsp_medical_mcqa", "ilsp_mcqa_asep"]
ALL_TASKS = ["greekmmlu", "ilsp_medical_mcqa", "ilsp_mcqa_asep", "plutus_qa"]

N_RESAMPLES = 1000
CI_LEVEL = 0.95
RNG_SEED = 20260529

NEW_MODELS = {
    "iter-834-Vanilla-3.5B": WORKSPACE / "iter834_Vanilla-3.5B_native_mcq_predictions.jsonl",
    "iter-1192-Vanilla-5B": WORKSPACE / "iter1192_Vanilla-5B_native_mcq_predictions.jsonl",
}

# Source paths to write into model entries (Clariden originals, since v3 is
# delete-after-use locally). Keep canonical original paths in the artifact.
NEW_SOURCE_PATHS = {
    "iter-834-Vanilla-3.5B": "/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/eval_04_vanilla_goldfish_5b_20260528T112539Z/iter_0000834/native_mcq/Vanilla-3.5B_native_mcq_predictions.jsonl",
    "iter-1192-Vanilla-5B": "/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/eval_04_vanilla_goldfish_5b_20260528T112539Z/iter_0001192/native_mcq/Vanilla-5B_native_mcq_predictions.jsonl",
}

# All models needed for paired delta computation this round.
PAIRED_MODELS = {
    "iter-834-Vanilla-3.5B": NEW_MODELS["iter-834-Vanilla-3.5B"],
    "iter-1192-Vanilla-5B": NEW_MODELS["iter-1192-Vanilla-5B"],
    "bakeoff-Vanilla-3.5B": WORKSPACE / "bakeoff_Vanilla-3.5B_native_mcq_predictions.jsonl",
    "bakeoff-Vanilla-5B": WORKSPACE / "bakeoff_Vanilla-5B_native_mcq_predictions.jsonl",
    "Apertus-Base": WORKSPACE / "Apertus-Base_native_mcq_predictions.jsonl",
    "Apertus-Base-matched-Path-B-perturbed": WORKSPACE / "Apertus-Base-matched-Path-B-perturbed_native_mcq_predictions.jsonl",
    "iter-477-Vanilla-2B": WORKSPACE / "iter477_Vanilla-2B_native_mcq_predictions.jsonl",
}

PAIRS = [
    ("iter-834-Vanilla-3.5B", "bakeoff-Vanilla-3.5B"),
    ("iter-1192-Vanilla-5B", "bakeoff-Vanilla-5B"),
    ("iter-1192-Vanilla-5B", "Apertus-Base"),
    ("iter-1192-Vanilla-5B", "Apertus-Base-matched-Path-B-perturbed"),
    ("iter-1192-Vanilla-5B", "iter-477-Vanilla-2B"),
    ("iter-1192-Vanilla-5B", "iter-834-Vanilla-3.5B"),
    ("iter-834-Vanilla-3.5B", "iter-477-Vanilla-2B"),
]
METRICS = ["headline_3task"] + ALL_TASKS  # 5 metrics per pair


def load_correctness(path: Path) -> dict[str, np.ndarray]:
    rows_by_bench: dict[str, list[tuple[str, int]]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            rows_by_bench[d["benchmark"]].append((d["example_id"], int(bool(d["correct"]))))
    return {b: np.array([r[1] for r in rows], dtype=np.int32) for b, rows in rows_by_bench.items()}


def percentile_ci(samples: np.ndarray, level: float) -> tuple[float, float]:
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def main() -> None:
    artifact = json.loads(ARTIFACT.read_text())

    correctness: dict[str, dict[str, np.ndarray]] = {}
    for name, p in PAIRED_MODELS.items():
        correctness[name] = load_correctness(p)
        print(f"loaded {name}: " + ", ".join(
            f"{b}=n{len(v)}/acc{v.mean():.4f}" for b, v in correctness[name].items()
        ))

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
        headline4_samples = np.mean(
            np.stack([resampled_task_acc(model, t) for t in ALL_TASKS], axis=0),
            axis=0,
        )
        h4_point = float(np.mean([correctness[model][t].mean() for t in ALL_TASKS]))
        h4_lo, h4_hi = percentile_ci(headline4_samples, CI_LEVEL)
        headline_4task = {
            "point": h4_point,
            "lo_95": h4_lo,
            "hi_95": h4_hi,
            "n_items_per_task": [n_items_per_task[t] for t in ALL_TASKS],
            "tasks": list(ALL_TASKS),
        }
        return {
            "per_task": per_task,
            "source_path": NEW_SOURCE_PATHS[model],
            "headline_3task": headline_3task,
            "headline_4task_with_plutus": headline_4task,
        }

    # ---- New model entries -------------------------------------------------
    for model in NEW_MODELS:
        artifact["models"][model] = model_block(model)
        print(
            f"{model} headline_3task: point={artifact['models'][model]['headline_3task']['point']:.4f} "
            f"lo={artifact['models'][model]['headline_3task']['lo_95']:.4f} "
            f"hi={artifact['models'][model]['headline_3task']['hi_95']:.4f}"
        )

    # ---- Delta table extensions -------------------------------------------
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

    new_rows = []
    for a, b in PAIRS:
        for metric in METRICS:
            a_samples = metric_samples(a, metric)
            b_samples = metric_samples(b, metric)
            diff = a_samples - b_samples
            delta = metric_point(a, metric) - metric_point(b, metric)
            lo, hi = percentile_ci(diff, CI_LEVEL)
            outside = (lo > 0) or (hi < 0)
            new_rows.append({
                "a": a,
                "b": b,
                "metric": metric,
                "delta": delta,
                "delta_ci_lo": lo,
                "delta_ci_hi": hi,
                "outside_zero_ci": bool(outside),
            })

    artifact["delta_table"].extend(new_rows)

    # ---- Top-level updates -------------------------------------------------
    artifact["generated_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    artifact["revision"] = (
        "v3 — added iter-834-Vanilla-3.5B + iter-1192-Vanilla-5B "
        "+ 7 paired comparisons (35 delta_table rows)"
    )
    artifact["pending"] = {}

    ARTIFACT.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {ARTIFACT}")

    # ---- Print headline_3task summary --------------------------------------
    print("\nNew delta_table rows (headline_3task only):")
    for row in new_rows:
        if row["metric"] != "headline_3task":
            continue
        print(
            f"  {row['a']} vs {row['b']}: delta={row['delta']:+.4f}  "
            f"CI=[{row['delta_ci_lo']:+.4f}, {row['delta_ci_hi']:+.4f}]  "
            f"outside_zero={row['outside_zero_ci']}"
        )


if __name__ == "__main__":
    main()
