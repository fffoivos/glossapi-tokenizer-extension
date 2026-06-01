"""Plutus QA marginal + paired bootstrap CIs across iter-477 / iter-834 / iter-1192.

Methodology mirrors v4 v2:
  - per-task item-level resampling (independent within task)
  - 1000 resamples, 95% percentile CI, rng_seed=20260529
  - paired: shared resample indices across model pair -> paired diff
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

WS = Path(__file__).resolve().parent
ITERS = {
    "iter-477": WS / "iter477_predictions.jsonl",
    "iter-834": WS / "iter834_predictions.jsonl",
    "iter-1192": WS / "iter1192_predictions.jsonl",
}
N_RESAMPLES = 1000
CI_LEVEL = 0.95
RNG_SEED = 20260529


def load_plutus(path: Path):
    """Return list of (example_id, correct, num_choices, pred_index, answer_index, choice_scores)."""
    out = []
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            if d["benchmark"] != "plutus_qa":
                continue
            out.append({
                "example_id": d["example_id"],
                "correct": int(bool(d["correct"])),
                "num_choices": d["num_choices"],
                "pred_index": d["pred_index"],
                "answer_index": d["answer_index"],
                "choice_scores": d["choice_scores"],
            })
    out.sort(key=lambda r: int(r["example_id"].split(":")[1]))
    return out


def percentile_ci(samples, level):
    a = (1.0 - level) / 2.0
    return float(np.quantile(samples, a)), float(np.quantile(samples, 1.0 - a))


def main():
    data = {k: load_plutus(p) for k, p in ITERS.items()}
    for k, recs in data.items():
        print(f"{k}: n={len(recs)}, acc={sum(r['correct'] for r in recs) / len(recs):.4f}")

    # Verify alignment
    ids = [r["example_id"] for r in data["iter-477"]]
    for k, recs in data.items():
        assert [r["example_id"] for r in recs] == ids, f"id misalignment in {k}"
    print(f"[ok] example_ids aligned across all three iters (n={len(ids)}).")

    corr = {k: np.array([r["correct"] for r in recs], dtype=np.int32)
            for k, recs in data.items()}
    n_items = len(corr["iter-477"])

    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, n_items, size=(N_RESAMPLES, n_items))

    def resampled_acc(arr):
        return arr[idx].mean(axis=1)

    # ---- Marginal CIs --------------------------------------------------
    marginal = {}
    print("\n=== Plutus marginal 95% CIs (1000-resample percentile bootstrap) ===")
    for k, arr in corr.items():
        samples = resampled_acc(arr)
        lo, hi = percentile_ci(samples, CI_LEVEL)
        marginal[k] = {
            "point": float(arr.mean()),
            "lo_95": lo,
            "hi_95": hi,
            "n_items": int(n_items),
        }
        print(f"  {k}: {marginal[k]['point']:.4f}  CI=[{lo:.4f}, {hi:.4f}]")

    # ---- Paired CIs ----------------------------------------------------
    PAIRS = [
        ("iter-1192", "iter-834"),
        ("iter-1192", "iter-477"),
        ("iter-834", "iter-477"),
    ]
    paired = []
    print("\n=== Plutus paired 95% CIs (a - b, paired indices) ===")
    for a, b in PAIRS:
        a_samples = resampled_acc(corr[a])
        b_samples = resampled_acc(corr[b])
        diff = a_samples - b_samples
        delta = float(corr[a].mean() - corr[b].mean())
        lo, hi = percentile_ci(diff, CI_LEVEL)
        outside = (lo > 0) or (hi < 0)
        paired.append({
            "a": a, "b": b, "delta": delta,
            "ci_lo": lo, "ci_hi": hi,
            "outside_zero": bool(outside),
        })
        print(f"  {a} - {b}: delta={delta:+.4f}  CI=[{lo:+.4f}, {hi:+.4f}]  outside_zero={outside}")

    # ---- Item-level diff: 834-right & 1192-wrong (the drop) ------------
    a834 = corr["iter-834"]
    a1192 = corr["iter-1192"]
    a477 = corr["iter-477"]
    flips_834right_1192wrong = []
    flips_834wrong_1192right = []
    flips_477right_1192wrong = []
    flips_477wrong_1192right = []
    for i, (r477, r834, r1192) in enumerate(zip(a477, a834, a1192)):
        rec1192 = data["iter-1192"][i]
        rec834 = data["iter-834"][i]
        rec477 = data["iter-477"][i]
        if r834 == 1 and r1192 == 0:
            flips_834right_1192wrong.append({
                "example_id": rec1192["example_id"],
                "num_choices": rec1192["num_choices"],
                "answer_index": rec1192["answer_index"],
                "pred_index_1192": rec1192["pred_index"],
                "pred_index_834": rec834["pred_index"],
                "pred_index_477": rec477["pred_index"],
                "r477": int(r477),
            })
        if r834 == 0 and r1192 == 1:
            flips_834wrong_1192right.append({
                "example_id": rec1192["example_id"],
                "num_choices": rec1192["num_choices"],
                "answer_index": rec1192["answer_index"],
                "pred_index_1192": rec1192["pred_index"],
                "pred_index_834": rec834["pred_index"],
                "pred_index_477": rec477["pred_index"],
                "r477": int(r477),
            })
        if r477 == 1 and r1192 == 0:
            flips_477right_1192wrong.append(rec1192["example_id"])
        if r477 == 0 and r1192 == 1:
            flips_477wrong_1192right.append(rec1192["example_id"])

    print(f"\n=== Item-level diff ===")
    print(f"  iter-834 right & iter-1192 wrong: {len(flips_834right_1192wrong)}")
    print(f"  iter-834 wrong & iter-1192 right: {len(flips_834wrong_1192right)}")
    print(f"  iter-477 right & iter-1192 wrong: {len(flips_477right_1192wrong)}")
    print(f"  iter-477 wrong & iter-1192 right: {len(flips_477wrong_1192right)}")

    # num_choices distribution
    nc_counts = defaultdict(int)
    for r in data["iter-1192"]:
        nc_counts[r["num_choices"]] += 1
    print(f"  Plutus num_choices distribution: {dict(nc_counts)}")

    # Drop breakdown by num_choices
    drop_by_nc = defaultdict(int)
    for item in flips_834right_1192wrong:
        drop_by_nc[item["num_choices"]] += 1
    gain_by_nc = defaultdict(int)
    for item in flips_834wrong_1192right:
        gain_by_nc[item["num_choices"]] += 1
    print(f"  834-right→1192-wrong by num_choices: {dict(drop_by_nc)}")
    print(f"  834-wrong→1192-right by num_choices: {dict(gain_by_nc)}")

    # Pred 1192 picks for drops: did 1192 switch to a specific wrong index?
    pred_drops = defaultdict(int)
    for item in flips_834right_1192wrong:
        # offset from correct
        off = (item["pred_index_1192"] - item["answer_index"]) % item["num_choices"]
        pred_drops[(item["num_choices"], off)] += 1
    print(f"  drop-pred offsets (num_choices, offset_from_answer): {dict(pred_drops)}")

    # Pred 477 vs 1192 trajectory on the drops
    consistent_with_477 = sum(1 for it in flips_834right_1192wrong if it["pred_index_1192"] == it["pred_index_477"])
    print(f"  of drops, iter-1192 matches iter-477 prediction: {consistent_with_477}/{len(flips_834right_1192wrong)}")

    # Logprob margin tightness on drops vs gains: 1192 confidence
    def margin_stats(records, key):
        if not records:
            return None
        margins = []
        for rec in records:
            example_id = rec["example_id"]
            # Find the full record in data[key]
            for r in data[key]:
                if r["example_id"] == example_id:
                    scores = r["choice_scores"]
                    sorted_avg = sorted([s["avg_logprob"] for s in scores], reverse=True)
                    if len(sorted_avg) >= 2:
                        margins.append(sorted_avg[0] - sorted_avg[1])
                    break
        if margins:
            return {"mean": float(np.mean(margins)), "median": float(np.median(margins)), "n": len(margins)}
        return None

    print(f"  iter-1192 avg_logprob top-margin on drops: {margin_stats(flips_834right_1192wrong, 'iter-1192')}")
    print(f"  iter-1192 avg_logprob top-margin on gains: {margin_stats(flips_834wrong_1192right, 'iter-1192')}")
    print(f"  iter-1192 avg_logprob top-margin on all items: {margin_stats([{'example_id': r['example_id']} for r in data['iter-1192']], 'iter-1192')}")

    out = {
        "schema": "plutus-investigation-v1",
        "n_resamples": N_RESAMPLES,
        "ci_level": CI_LEVEL,
        "rng_seed": RNG_SEED,
        "n_plutus_items": n_items,
        "marginal_ci_per_checkpoint": marginal,
        "paired_ci": paired,
        "item_flips": {
            "iter834_right_and_iter1192_wrong": flips_834right_1192wrong,
            "iter834_wrong_and_iter1192_right": flips_834wrong_1192right,
            "n_iter477_right_iter1192_wrong": len(flips_477right_1192wrong),
            "n_iter477_wrong_iter1192_right": len(flips_477wrong_1192right),
        },
        "num_choices_distribution": dict(nc_counts),
        "drop_by_num_choices": dict(drop_by_nc),
        "gain_by_num_choices": dict(gain_by_nc),
    }
    (WS / "plutus_investigation_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nwrote {WS / 'plutus_investigation_results.json'}")


if __name__ == "__main__":
    main()
