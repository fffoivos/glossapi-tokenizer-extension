"""Bootstrap CIs for iter 1192 (Vanilla-5B).

Computes the five load-bearing CIs requested by the Vanilla-5B adversarial review:

  1. iter-1192 marginal 3-task headline 95% CI
  2. paired iter-1192 vs iter-477 headline_3task CI (full post-warmup gain)
  3. paired iter-1192 vs iter-834 headline_3task CI (slope after plateau)
  4. paired iter-1192 vs Apertus-Base headline_3task CI
  5. paired iter-1192 vs iter-834 Plutus QA CI (n=225) — noise vs signal

Methodology mirrors `reports/v4_workspace/run_bootstrap_v2.py` exactly:
  - Per-task item-level resampling (independent within each task).
  - 1000 resamples, 95% percentile CI, rng_seed=20260529.
  - Paired by shared resample indices across models in any pair (so the diff is
    a paired diff under the same RNG draws).
  - Headline = macro-mean across 3 headline tasks per resample.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

WS = Path(__file__).resolve().parent

PATHS = {
    "iter-1192-Vanilla-5B": WS / "iter1192_predictions.jsonl",
    "iter-834-Vanilla-3.5B": WS / "iter834_predictions.jsonl",
    "iter-477-Vanilla-2B":   WS / "iter477_predictions.jsonl",
    "Apertus-Base":          WS / "Apertus-Base_predictions.jsonl",
}

HEADLINE_TASKS = ["greekmmlu", "ilsp_medical_mcqa", "ilsp_mcqa_asep"]
ALL_TASKS = ["greekmmlu", "ilsp_medical_mcqa", "ilsp_mcqa_asep", "plutus_qa"]

N_RESAMPLES = 1000
CI_LEVEL = 0.95
RNG_SEED = 20260529


def load(path: Path):
    rows = defaultdict(list)
    ids = defaultdict(list)
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            b = d["benchmark"]
            rows[b].append(int(bool(d["correct"])))
            ids[b].append(d["example_id"])
    return ({b: np.array(v, dtype=np.int32) for b, v in rows.items()},
            {b: list(v) for b, v in ids.items()})


def pctci(s, lvl=CI_LEVEL):
    a = (1.0 - lvl) / 2.0
    return float(np.quantile(s, a)), float(np.quantile(s, 1.0 - a))


def main():
    corr = {}
    ids = {}
    for name, p in PATHS.items():
        c, i = load(p)
        corr[name] = c
        ids[name] = i
        print(f"loaded {name}: " + ", ".join(
            f"{b}=n{len(c[b])}/acc{c[b].mean():.4f}" for b in ALL_TASKS if b in c
        ))

    # Verify example_id alignment across all models (paired bootstrap is
    # only valid if rows index the same items).
    for t in ALL_TASKS:
        ref = ids["iter-1192-Vanilla-5B"][t]
        for name in PATHS:
            if name == "iter-1192-Vanilla-5B":
                continue
            assert ids[name][t] == ref, f"example_id misalignment for {name} task {t}"
    print("[ok] example_ids align across all four models for all tasks.")

    rng = np.random.default_rng(RNG_SEED)
    task_idx = {}
    for t in ALL_TASKS:
        n = len(corr["iter-1192-Vanilla-5B"][t])
        task_idx[t] = rng.integers(0, n, size=(N_RESAMPLES, n))

    def resampled(model: str, task: str) -> np.ndarray:
        return corr[model][task][task_idx[task]].mean(axis=1)

    def h3_samples(model: str) -> np.ndarray:
        return np.mean(np.stack([resampled(model, t) for t in HEADLINE_TASKS], axis=0), axis=0)

    def h3_point(model: str) -> float:
        return float(np.mean([corr[model][t].mean() for t in HEADLINE_TASKS]))

    # ---- 1) iter-1192 marginal 3-task headline CI ------------------------------
    iter1192_h3 = h3_samples("iter-1192-Vanilla-5B")
    iter1192_p = h3_point("iter-1192-Vanilla-5B")
    lo, hi = pctci(iter1192_h3)
    marginal_h3 = {"point": iter1192_p, "lo_95": lo, "hi_95": hi}
    print(f"\n[1] iter-1192 marginal headline_3task: {iter1192_p:.4f} [{lo:.4f}, {hi:.4f}]")

    # Also compute marginal per-task CIs for the record
    iter1192_per_task_marg = {}
    for t in ALL_TASKS:
        s = resampled("iter-1192-Vanilla-5B", t)
        p = float(corr["iter-1192-Vanilla-5B"][t].mean())
        l, h = pctci(s)
        iter1192_per_task_marg[t] = {"point": p, "lo_95": l, "hi_95": h,
                                      "n_items": int(len(corr["iter-1192-Vanilla-5B"][t]))}
        print(f"    {t}: {p:.4f} [{l:.4f}, {h:.4f}] (n={len(corr['iter-1192-Vanilla-5B'][t])})")

    # 4-task with plutus marginal CI
    iter1192_h4 = np.mean(
        np.stack([resampled("iter-1192-Vanilla-5B", t) for t in ALL_TASKS], axis=0),
        axis=0,
    )
    h4_p = float(np.mean([corr["iter-1192-Vanilla-5B"][t].mean() for t in ALL_TASKS]))
    h4_lo, h4_hi = pctci(iter1192_h4)
    print(f"    headline_4task_with_plutus: {h4_p:.4f} [{h4_lo:.4f}, {h4_hi:.4f}]")

    # ---- 2) paired iter-1192 vs iter-477 headline_3task ----------------------
    a_h3 = h3_samples("iter-477-Vanilla-2B")
    b_h3 = iter1192_h3
    diff = b_h3 - a_h3
    pt_a = h3_point("iter-477-Vanilla-2B")
    pt_b = iter1192_p
    d = pt_b - pt_a
    lo2, hi2 = pctci(diff)
    print(f"\n[2] paired iter-1192 vs iter-477 headline_3task:")
    print(f"    iter-477={pt_a:.6f} iter-1192={pt_b:.6f} delta={d:+.6f}")
    print(f"    95% CI=[{lo2:+.6f},{hi2:+.6f}] outside_zero={(lo2>0) or (hi2<0)}")

    # ---- 3) paired iter-1192 vs iter-834 headline_3task ----------------------
    a_h3 = h3_samples("iter-834-Vanilla-3.5B")
    diff = b_h3 - a_h3
    pt_a = h3_point("iter-834-Vanilla-3.5B")
    d = pt_b - pt_a
    lo3, hi3 = pctci(diff)
    print(f"\n[3] paired iter-1192 vs iter-834 headline_3task:")
    print(f"    iter-834={pt_a:.6f} iter-1192={pt_b:.6f} delta={d:+.6f}")
    print(f"    95% CI=[{lo3:+.6f},{hi3:+.6f}] outside_zero={(lo3>0) or (hi3<0)}")

    # ---- 4) paired iter-1192 vs Apertus-Base headline_3task -----------------
    a_h3 = h3_samples("Apertus-Base")
    diff = b_h3 - a_h3
    pt_a = h3_point("Apertus-Base")
    d = pt_b - pt_a
    lo4, hi4 = pctci(diff)
    print(f"\n[4] paired iter-1192 vs Apertus-Base headline_3task:")
    print(f"    Apertus-Base={pt_a:.6f} iter-1192={pt_b:.6f} delta={d:+.6f}")
    print(f"    95% CI=[{lo4:+.6f},{hi4:+.6f}] outside_zero={(lo4>0) or (hi4<0)}")

    # ---- 5) paired iter-1192 vs iter-834 Plutus QA ---------------------------
    a_p = resampled("iter-834-Vanilla-3.5B", "plutus_qa")
    b_p = resampled("iter-1192-Vanilla-5B", "plutus_qa")
    diff = b_p - a_p
    pt_a = float(corr["iter-834-Vanilla-3.5B"]["plutus_qa"].mean())
    pt_b = float(corr["iter-1192-Vanilla-5B"]["plutus_qa"].mean())
    d = pt_b - pt_a
    lo5, hi5 = pctci(diff)
    print(f"\n[5] paired iter-1192 vs iter-834 Plutus QA:")
    print(f"    iter-834={pt_a:.6f} iter-1192={pt_b:.6f} delta={d:+.6f}")
    print(f"    95% CI=[{lo5:+.6f},{hi5:+.6f}] outside_zero={(lo5>0) or (hi5<0)}")

    # ---- Bonus: per-task paired deltas iter-1192 vs iter-834 ---------------
    per_task_834 = []
    print(f"\n[bonus] iter-1192 vs iter-834 per-task paired deltas:")
    for t in ALL_TASKS:
        ax = resampled("iter-834-Vanilla-3.5B", t)
        bx = resampled("iter-1192-Vanilla-5B", t)
        diff = bx - ax
        d_ = float(corr["iter-1192-Vanilla-5B"][t].mean() - corr["iter-834-Vanilla-3.5B"][t].mean())
        lo_, hi_ = pctci(diff)
        out_ = (lo_ > 0) or (hi_ < 0)
        per_task_834.append({"task": t, "delta": d_, "ci_lo": lo_, "ci_hi": hi_,
                              "outside_zero": bool(out_)})
        print(f"    {t}: Δ={d_:+.4f}, CI=[{lo_:+.4f},{hi_:+.4f}], outside={out_}")

    # iter-1192 vs iter-477 per-task
    per_task_477 = []
    print(f"\n[bonus] iter-1192 vs iter-477 per-task paired deltas:")
    for t in ALL_TASKS:
        ax = resampled("iter-477-Vanilla-2B", t)
        bx = resampled("iter-1192-Vanilla-5B", t)
        diff = bx - ax
        d_ = float(corr["iter-1192-Vanilla-5B"][t].mean() - corr["iter-477-Vanilla-2B"][t].mean())
        lo_, hi_ = pctci(diff)
        out_ = (lo_ > 0) or (hi_ < 0)
        per_task_477.append({"task": t, "delta": d_, "ci_lo": lo_, "ci_hi": hi_,
                              "outside_zero": bool(out_)})
        print(f"    {t}: Δ={d_:+.4f}, CI=[{lo_:+.4f},{hi_:+.4f}], outside={out_}")

    # iter-1192 vs Apertus-Base per-task
    per_task_AB = []
    print(f"\n[bonus] iter-1192 vs Apertus-Base per-task paired deltas:")
    for t in ALL_TASKS:
        ax = resampled("Apertus-Base", t)
        bx = resampled("iter-1192-Vanilla-5B", t)
        diff = bx - ax
        d_ = float(corr["iter-1192-Vanilla-5B"][t].mean() - corr["Apertus-Base"][t].mean())
        lo_, hi_ = pctci(diff)
        out_ = (lo_ > 0) or (hi_ < 0)
        per_task_AB.append({"task": t, "delta": d_, "ci_lo": lo_, "ci_hi": hi_,
                             "outside_zero": bool(out_)})
        print(f"    {t}: Δ={d_:+.4f}, CI=[{lo_:+.4f},{hi_:+.4f}], outside={out_}")

    # ---- Bonus: headline_4task_with_plutus paired vs iter-834 / iter-477 / AB
    def h4(m):
        return np.mean(np.stack([resampled(m, t) for t in ALL_TASKS], axis=0), axis=0)
    h4_1192 = h4("iter-1192-Vanilla-5B")
    pt_1192_4 = float(np.mean([corr["iter-1192-Vanilla-5B"][t].mean() for t in ALL_TASKS]))

    def paired_h4(other):
        h4_o = h4(other)
        diff = h4_1192 - h4_o
        pt_o = float(np.mean([corr[other][t].mean() for t in ALL_TASKS]))
        d_ = pt_1192_4 - pt_o
        lo_, hi_ = pctci(diff)
        return {"delta": d_, "ci_lo": lo_, "ci_hi": hi_,
                "outside_zero": bool((lo_ > 0) or (hi_ < 0))}

    h4_paired = {
        "vs_iter834": paired_h4("iter-834-Vanilla-3.5B"),
        "vs_iter477": paired_h4("iter-477-Vanilla-2B"),
        "vs_AB":      paired_h4("Apertus-Base"),
    }
    print(f"\n[bonus] headline_4task_with_plutus paired diffs:")
    for k, v in h4_paired.items():
        print(f"    {k}: Δ={v['delta']:+.4f}, CI=[{v['ci_lo']:+.4f},{v['ci_hi']:+.4f}],"
              f" outside={v['outside_zero']}")

    out = {
        "n_resamples": N_RESAMPLES,
        "ci_level": CI_LEVEL,
        "rng_seed": RNG_SEED,
        "methodology": ("Per-task item-level paired bootstrap (independent resampling within each task)."
                        " Headline = macro-mean across 3 headline tasks per resample."
                        " Paired by shared resample indices across all models."),
        "iter_1192_marginal_3task_headline": marginal_h3,
        "iter_1192_marginal_4task_with_plutus": {"point": h4_p, "lo_95": h4_lo, "hi_95": h4_hi},
        "iter_1192_marginal_per_task": iter1192_per_task_marg,
        "paired_iter1192_vs_iter477_headline_3task": {
            "iter_477_point": h3_point("iter-477-Vanilla-2B"),
            "iter_1192_point": iter1192_p,
            "delta": iter1192_p - h3_point("iter-477-Vanilla-2B"),
            "ci_lo": lo2, "ci_hi": hi2,
            "outside_zero": bool((lo2 > 0) or (hi2 < 0)),
        },
        "paired_iter1192_vs_iter834_headline_3task": {
            "iter_834_point": h3_point("iter-834-Vanilla-3.5B"),
            "iter_1192_point": iter1192_p,
            "delta": iter1192_p - h3_point("iter-834-Vanilla-3.5B"),
            "ci_lo": lo3, "ci_hi": hi3,
            "outside_zero": bool((lo3 > 0) or (hi3 < 0)),
        },
        "paired_iter1192_vs_ApertusBase_headline_3task": {
            "ApertusBase_point": h3_point("Apertus-Base"),
            "iter_1192_point": iter1192_p,
            "delta": iter1192_p - h3_point("Apertus-Base"),
            "ci_lo": lo4, "ci_hi": hi4,
            "outside_zero": bool((lo4 > 0) or (hi4 < 0)),
        },
        "paired_iter1192_vs_iter834_plutus": {
            "iter_834_point": float(corr["iter-834-Vanilla-3.5B"]["plutus_qa"].mean()),
            "iter_1192_point": float(corr["iter-1192-Vanilla-5B"]["plutus_qa"].mean()),
            "delta": float(corr["iter-1192-Vanilla-5B"]["plutus_qa"].mean()
                           - corr["iter-834-Vanilla-3.5B"]["plutus_qa"].mean()),
            "ci_lo": lo5, "ci_hi": hi5,
            "outside_zero": bool((lo5 > 0) or (hi5 < 0)),
        },
        "paired_per_task_iter1192_vs_iter834": per_task_834,
        "paired_per_task_iter1192_vs_iter477": per_task_477,
        "paired_per_task_iter1192_vs_ApertusBase": per_task_AB,
        "headline_4task_with_plutus_paired": h4_paired,
    }
    (WS / "iter1192_bootstrap_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {WS / 'iter1192_bootstrap_results.json'}")


if __name__ == "__main__":
    main()
