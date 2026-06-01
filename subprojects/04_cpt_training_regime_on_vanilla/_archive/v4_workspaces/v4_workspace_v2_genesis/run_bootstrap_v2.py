"""V4 bootstrap re-emit (v2).

Re-emits reports/v4_bootstrap_cis_native_mcq.json with:
- Added model entries iter-238-Vanilla-1B, iter-477-Vanilla-2B,
  Apertus-Base-matched-Path-B-perturbed.
- Extended delta_table with 6 new paired comparisons (headline_3task + 4 tasks).
- generated_utc refreshed; revision tag added; pending block cleared.

Methodology kept identical to v1:
  per-task item-level resampling (independent within each task), macro-mean
  across 3 headline tasks per resample, 1000 resamples, 95% percentile,
  rng_seed=20260529. Paired deltas use the same resample indices on both
  models of a pair so the diff is paired.
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

# Map model name -> prediction JSONL path (workspace-relative for new ones,
# source_path string as it lives in the v1 artifact for existing ones).
NEW_MODELS = {
    "iter-238-Vanilla-1B": WORKSPACE / "iter238_Vanilla-1B_native_mcq_predictions.jsonl",
    "iter-477-Vanilla-2B": WORKSPACE / "iter477_Vanilla-2B_native_mcq_predictions.jsonl",
    "Apertus-Base-matched-Path-B-perturbed": (
        WORKSPACE / "Apertus-Base-matched-Path-B-perturbed_native_mcq_predictions.jsonl"
    ),
}

# Model files needed (locally) for paired delta computation:
PAIRED_MODELS = {
    "Apertus-Base": WORKSPACE / "Apertus-Base_native_mcq_predictions.jsonl",
    "bakeoff-Vanilla-2B": WORKSPACE / "Vanilla-2B_native_mcq_predictions.jsonl",
    "iter-119-Vanilla-0.5B": WORKSPACE / "iter119_Vanilla-0.5B_native_mcq_predictions.jsonl",
    "iter-238-Vanilla-1B": NEW_MODELS["iter-238-Vanilla-1B"],
    "iter-477-Vanilla-2B": NEW_MODELS["iter-477-Vanilla-2B"],
    "Apertus-Base-matched-Path-B-perturbed": NEW_MODELS["Apertus-Base-matched-Path-B-perturbed"],
}


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
    artifact = json.loads(ARTIFACT.read_text())

    # Load correctness arrays for all models we touch this round.
    correctness: dict[str, dict[str, np.ndarray]] = {}
    for name, p in PAIRED_MODELS.items():
        correctness[name] = load_correctness(p)
        print(f"loaded {name}: " + ", ".join(
            f"{b}=n{len(v)}/acc{v.mean():.4f}" for b, v in correctness[name].items()
        ))

    # Build per-task resample index matrices, identical across models. Shape
    # (N_RESAMPLES, n_items) per task. This is the paired bootstrap key.
    rng = np.random.default_rng(RNG_SEED)
    task_index_matrices: dict[str, np.ndarray] = {}
    n_items_per_task: dict[str, int] = {}
    for task in ALL_TASKS:
        # Use n from any model (all aligned)
        n_items = len(next(iter(correctness.values()))[task])
        n_items_per_task[task] = n_items
        task_index_matrices[task] = rng.integers(0, n_items, size=(N_RESAMPLES, n_items))

    def resampled_task_acc(model: str, task: str) -> np.ndarray:
        arr = correctness[model][task]
        idx = task_index_matrices[task]
        # mean over resampled items along axis=1
        return arr[idx].mean(axis=1)

    def model_block(model: str) -> dict:
        per_task = {}
        # Compute per-task point + per-task CI (matches v1 schema)
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
        # headline_3task = per-resample macro mean across 3 headline tasks
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
        # headline_4task_with_plutus
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
            "source_path": str(NEW_MODELS[model] if model in NEW_MODELS else PAIRED_MODELS[model]),
            "headline_3task": headline_3task,
            "headline_4task_with_plutus": headline_4task,
        }

    # ---- New model entries -------------------------------------------------
    for model in NEW_MODELS:
        artifact["models"][model] = model_block(model)
        # Path for matched-config gets a perturbation note.
        if model == "Apertus-Base-matched-Path-B-perturbed":
            artifact["models"][model]["perturbation_note"] = (
                "rope_theta=500K override on weights pretrained at rope_theta=12M. "
                "Static config-only override; the model is *perturbed*, not re-anchored. "
                "Used as DIAGNOSTIC of rope-sensitivity, NOT as a clean baseline. "
                "See subprojects/04_cpt_training_regime_on_vanilla/cpt-plan.md "
                "§2.1 'Training-time positional geometry override (Path B)' for full discussion."
            )
        # Refresh display of point for headline_3task to match the explicit
        # request that it reflects the corrected/native aggregate.
        print(
            f"{model} headline_3task: point={artifact['models'][model]['headline_3task']['point']:.4f} "
            f"lo={artifact['models'][model]['headline_3task']['lo_95']:.4f} "
            f"hi={artifact['models'][model]['headline_3task']['hi_95']:.4f}"
        )

    # ---- Delta table extensions -------------------------------------------
    PAIRS = [
        ("iter-477-Vanilla-2B", "bakeoff-Vanilla-2B"),
        ("iter-477-Vanilla-2B", "Apertus-Base"),
        ("iter-477-Vanilla-2B", "Apertus-Base-matched-Path-B-perturbed"),
        ("iter-238-Vanilla-1B", "iter-119-Vanilla-0.5B"),
        ("iter-477-Vanilla-2B", "iter-238-Vanilla-1B"),
        ("iter-477-Vanilla-2B", "iter-119-Vanilla-0.5B"),
    ]
    METRICS = ["headline_3task"] + ALL_TASKS  # 5 metrics per pair

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
            diff = a_samples - b_samples  # paired by index construction
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
        "v2 — added iter-238/477 corrected + matched-config Apertus-Base "
        "+ extended delta table"
    )
    artifact["pending"] = {}

    ARTIFACT.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {ARTIFACT}")

    # ---- Final printout of the 6 new headline_3task deltas ----------------
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
