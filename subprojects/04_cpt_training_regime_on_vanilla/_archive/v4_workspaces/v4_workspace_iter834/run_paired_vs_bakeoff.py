"""Paired bootstrap iter-834 vs bakeoff-Vanilla-3.5B.

Same methodology as run_paired_bootstrap.py.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

WORKSPACE = Path(__file__).resolve().parent
ITER834 = WORKSPACE / "iter834_predictions.jsonl"
BAKEOFF = WORKSPACE / "bakeoff_Vanilla-3.5B_predictions.jsonl"

HEADLINE_TASKS = ["greekmmlu", "ilsp_medical_mcqa", "ilsp_mcqa_asep"]
ALL_TASKS = ["greekmmlu", "ilsp_medical_mcqa", "ilsp_mcqa_asep", "plutus_qa"]
N_RESAMPLES = 1000
CI_LEVEL = 0.95
RNG_SEED = 20260529


def load_correctness(path):
    rows_by_bench = defaultdict(list)
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            rows_by_bench[d["benchmark"]].append((d["example_id"], int(bool(d["correct"]))))
    return {b: np.array([r[1] for r in rows], dtype=np.int32) for b, rows in rows_by_bench.items()}, \
           {b: [r[0] for r in rows] for b, rows in rows_by_bench.items()}


def percentile_ci(samples, level):
    a = (1.0 - level) / 2.0
    return float(np.quantile(samples, a)), float(np.quantile(samples, 1.0 - a))


def main():
    a_corr, a_ids = load_correctness(ITER834)
    b_corr, b_ids = load_correctness(BAKEOFF)
    for t in ALL_TASKS:
        ok = a_ids[t] == b_ids[t]
        if not ok:
            print(f"[warn] example_id mismatch in {t} between iter-834 and bakeoff-3.5B")
        assert len(a_corr[t]) == len(b_corr[t])
    rng = np.random.default_rng(RNG_SEED)
    task_idx = {t: rng.integers(0, len(a_corr[t]), size=(N_RESAMPLES, len(a_corr[t]))) for t in ALL_TASKS}

    def resamp(corr, t):
        return corr[t][task_idx[t]].mean(axis=1)
    def h3(corr):
        return np.mean(np.stack([resamp(corr, t) for t in HEADLINE_TASKS], axis=0), axis=0)

    a_h3 = h3(a_corr); b_h3 = h3(b_corr)
    diff = a_h3 - b_h3  # iter-834 minus bakeoff-3.5B
    point_a = float(np.mean([a_corr[t].mean() for t in HEADLINE_TASKS]))
    point_b = float(np.mean([b_corr[t].mean() for t in HEADLINE_TASKS]))
    delta = point_a - point_b
    lo, hi = percentile_ci(diff, CI_LEVEL)
    out = (lo > 0) or (hi < 0)
    print(f"iter-834 headline_3task: {point_a:.6f}")
    print(f"bakeoff-Vanilla-3.5B headline_3task: {point_b:.6f}")
    print(f"delta (834 - bakeoff): {delta:+.6f}")
    print(f"95% CI: [{lo:+.6f}, {hi:+.6f}]")
    print(f"outside zero: {out}")

    # Per-task
    print("\nper-task:")
    rows = []
    for t in ALL_TASKS:
        d = float(a_corr[t].mean() - b_corr[t].mean())
        diff_t = resamp(a_corr, t) - resamp(b_corr, t)
        lo_t, hi_t = percentile_ci(diff_t, CI_LEVEL)
        out_t = (lo_t > 0) or (hi_t < 0)
        print(f"  {t}: delta={d:+.6f}, CI=[{lo_t:+.6f}, {hi_t:+.6f}], outside_zero={out_t}")
        rows.append({"task": t, "delta": d, "ci_lo": lo_t, "ci_hi": hi_t, "outside_zero": bool(out_t)})

    out_obj = {
        "n_resamples": N_RESAMPLES,
        "rng_seed": RNG_SEED,
        "iter_834_point": point_a,
        "bakeoff_Vanilla_3_5B_point": point_b,
        "paired_headline_3task": {"delta": delta, "ci_lo": lo, "ci_hi": hi, "outside_zero": bool(out)},
        "paired_per_task": rows,
    }
    (WORKSPACE / "paired_iter834_vs_bakeoff_3p5B.json").write_text(json.dumps(out_obj, indent=2))


if __name__ == "__main__":
    main()
