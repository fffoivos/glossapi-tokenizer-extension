"""Paired bootstrap for iter-477 vs iter-834 native MCQ.

Mirrors the v2 methodology exactly:
  - per-task item-level resampling (independent within each task)
  - 1000 resamples, 95% percentile CI, rng_seed=20260529
  - paired indices: same RNG draws on both models -> diff is paired
  - macro-mean across 3 headline tasks per resample
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

WORKSPACE = Path(__file__).resolve().parent
ITER477 = WORKSPACE / "iter477_predictions.jsonl"
ITER834 = WORKSPACE / "iter834_predictions.jsonl"

HEADLINE_TASKS = ["greekmmlu", "ilsp_medical_mcqa", "ilsp_mcqa_asep"]
ALL_TASKS = ["greekmmlu", "ilsp_medical_mcqa", "ilsp_mcqa_asep", "plutus_qa"]

N_RESAMPLES = 1000
CI_LEVEL = 0.95
RNG_SEED = 20260529


def load_correctness(path: Path):
    rows_by_bench = defaultdict(list)
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            rows_by_bench[d["benchmark"]].append((d["example_id"], int(bool(d["correct"]))))
    # preserve insertion order; assert alignment can be checked later
    return {b: np.array([r[1] for r in rows], dtype=np.int32) for b, rows in rows_by_bench.items()}, \
           {b: [r[0] for r in rows] for b, rows in rows_by_bench.items()}


def percentile_ci(samples, level):
    a = (1.0 - level) / 2.0
    return float(np.quantile(samples, a)), float(np.quantile(samples, 1.0 - a))


def main():
    a_corr, a_ids = load_correctness(ITER477)
    b_corr, b_ids = load_correctness(ITER834)

    # Sanity: same example_ids in same order for each task
    for t in ALL_TASKS:
        assert a_ids[t] == b_ids[t], f"example_id misalignment in task {t}"
        assert len(a_corr[t]) == len(b_corr[t])
    print("[ok] example_ids align across iter 477 and iter 834 for every task.")

    print(f"iter-477 point per task: " + ", ".join(f"{t}={a_corr[t].mean():.4f}" for t in ALL_TASKS))
    print(f"iter-834 point per task: " + ", ".join(f"{t}={b_corr[t].mean():.4f}" for t in ALL_TASKS))

    rng = np.random.default_rng(RNG_SEED)
    task_idx = {}
    for t in ALL_TASKS:
        n = len(a_corr[t])
        task_idx[t] = rng.integers(0, n, size=(N_RESAMPLES, n))

    def resampled_task_acc(corr_arr, t):
        return corr_arr[t][task_idx[t]].mean(axis=1)

    def headline3_samples(corr_arr):
        return np.mean(np.stack([resampled_task_acc(corr_arr, t) for t in HEADLINE_TASKS], axis=0), axis=0)

    a_h3 = headline3_samples(a_corr)
    b_h3 = headline3_samples(b_corr)
    diff_h3 = b_h3 - a_h3  # iter834 - iter477

    point_a = float(np.mean([a_corr[t].mean() for t in HEADLINE_TASKS]))
    point_b = float(np.mean([b_corr[t].mean() for t in HEADLINE_TASKS]))
    delta = point_b - point_a
    lo, hi = percentile_ci(diff_h3, CI_LEVEL)
    outside = (lo > 0) or (hi < 0)
    print()
    print(f"=== Paired bootstrap: iter-834 vs iter-477 (3-task headline) ===")
    print(f"  iter-477 point: {point_a:.6f}")
    print(f"  iter-834 point: {point_b:.6f}")
    print(f"  delta (834 - 477): {delta:+.6f}")
    print(f"  95% CI: [{lo:+.6f}, {hi:+.6f}]")
    print(f"  outside zero: {outside}")

    # Per-task paired
    print()
    print("=== Per-task paired deltas (834 - 477) ===")
    per_task_rows = []
    for t in ALL_TASKS:
        diff = resampled_task_acc(b_corr, t) - resampled_task_acc(a_corr, t)
        d = float(b_corr[t].mean() - a_corr[t].mean())
        lo_t, hi_t = percentile_ci(diff, CI_LEVEL)
        out_t = (lo_t > 0) or (hi_t < 0)
        print(f"  {t}: delta={d:+.6f}, CI=[{lo_t:+.6f}, {hi_t:+.6f}], outside_zero={out_t}")
        per_task_rows.append({
            "task": t, "delta": d, "ci_lo": lo_t, "ci_hi": hi_t, "outside_zero": bool(out_t),
        })

    # Also compute marginal per-task CIs for iter 834 (for record)
    print()
    print("=== iter-834 marginal CIs ===")
    marg = {}
    for t in ALL_TASKS:
        samples = resampled_task_acc(b_corr, t)
        lo_t, hi_t = percentile_ci(samples, CI_LEVEL)
        marg[t] = {"point": float(b_corr[t].mean()), "lo_95": lo_t, "hi_95": hi_t, "n_items": int(len(b_corr[t]))}
        print(f"  {t}: {marg[t]['point']:.4f} [{lo_t:.4f}, {hi_t:.4f}] (n={marg[t]['n_items']})")
    lo_h3, hi_h3 = percentile_ci(b_h3, CI_LEVEL)
    print(f"  headline_3task: {point_b:.4f} [{lo_h3:.4f}, {hi_h3:.4f}]")

    # 4-task with plutus
    a_h4 = np.mean(np.stack([resampled_task_acc(a_corr, t) for t in ALL_TASKS], axis=0), axis=0)
    b_h4 = np.mean(np.stack([resampled_task_acc(b_corr, t) for t in ALL_TASKS], axis=0), axis=0)
    diff_h4 = b_h4 - a_h4
    d4 = float(np.mean([b_corr[t].mean() for t in ALL_TASKS]) - np.mean([a_corr[t].mean() for t in ALL_TASKS]))
    lo4, hi4 = percentile_ci(diff_h4, CI_LEVEL)
    print()
    print(f"=== Headline_4task_with_plutus paired (834 - 477) ===")
    print(f"  delta={d4:+.6f}, CI=[{lo4:+.6f}, {hi4:+.6f}], outside_zero={(lo4>0)or(hi4<0)}")

    # Save a compact JSON record
    out = {
        "n_resamples": N_RESAMPLES,
        "ci_level": CI_LEVEL,
        "rng_seed": RNG_SEED,
        "resampling_scheme": "Per-task item-level paired bootstrap (independent resampling within each task). Headline = macro-mean across tasks per resample. Paired by shared resample indices between iter-477 and iter-834.",
        "iter_477_point_headline_3task": point_a,
        "iter_834_point_headline_3task": point_b,
        "paired_headline_3task": {"delta": delta, "ci_lo": lo, "ci_hi": hi, "outside_zero": bool(outside)},
        "paired_headline_4task_with_plutus": {"delta": d4, "ci_lo": lo4, "ci_hi": hi4, "outside_zero": bool((lo4>0)or(hi4<0))},
        "paired_per_task": per_task_rows,
        "iter_834_marginal_3task_headline": {"point": point_b, "lo_95": lo_h3, "hi_95": hi_h3},
        "iter_834_marginal_per_task": marg,
    }
    (WORKSPACE / "paired_iter477_vs_iter834.json").write_text(json.dumps(out, indent=2))
    print()
    print(f"wrote {WORKSPACE / 'paired_iter477_vs_iter834.json'}")


if __name__ == "__main__":
    main()
