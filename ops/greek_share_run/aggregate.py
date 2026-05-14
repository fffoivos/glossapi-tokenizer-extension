"""Aggregate per-dataset Greek-token measurements into a final summary.

Reads outputs/<dataset>.json files and applies Apertus's per-stage filter knobs
+ 8B stage durations (from paper Table H.8) to produce the Greek-share figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Paper Table 6 token counts per stage (in billions). Dataset name → tokens (B).
# Values from arXiv:2509.14233v2 §3.3 / Table 6.
STAGE_POOLS_B = {
    1: {
        "fineweb_edu_score2": 4815,
        "fineweb2_hq_plus_remaining": 3557,
        "starcoder": 235,
        "finemath_cc": 32,
        "gutenberg_v1_poison": 2,
    },
    2: {
        "fineweb_hq": 4064,
        "fineweb2_hq_plus_remaining": 3557,
        "fineweb_edu_score3": 1179,
        "finemath_cc": 32,
        "starcoder": 235,
        "gutenberg_v1_poison": 2,
    },
    3: {
        "fineweb_hq": 4064,
        "fineweb2_hq_plus_remaining": 3556,
        "fineweb_edu_score3": 1179,
        "starcoder": 235,
        "finemath_cc": 32,
        "infimm_webmath_cc": 19,
        "llm360_megamath_web": 260,
        "gutenberg_v2": 1,
    },
    4: {
        "dclm_edu": 1619,
        "fineweb2_hq_plus_remaining_10pct": 986,
        "starcoder": 234,
        "finemath_cc": 32,
        "infimm_webmath_cc": 19,
        "llm360_megamath_web_pro": 15,
    },
    5: {
        "dclm_edu": 1619,
        "fineweb2_hq_plus_remaining_10pct": 986,
        "starcoder_x2": 182,
        "commonpile_stack_v2_edu": 68,
        "finemath_cc": 32,
        "infimm_webmath_cc": 19,
        "llm360_megamath_web_pro": 15,
        "clean_wikipedia": 33,
        "translation_parallel": 21,
        "task_data_3x1": 3,
    },
}

# 8B stage durations from paper Appendix H.3 / Table H.8 (consumed-tokens-at-start, in B).
# Stage 2 skipped for 8B. Stage 5 cooldown end-of-run not in paper; estimate ~200 B tail.
STAGE_BOUNDARIES_8B_B = {
    1: (0, 7038),
    2: None,  # skipped
    3: (7038, 12000),
    4: (12000, 13345),
    5: (13345, 13545),  # 200 B cooldown estimate
}
TOTAL_8B_B = 13545  # estimated total consumed; paper's stated overall budget is 15 T

# Apertus FineWeb-2-HQ filter recipe per stage for ell_Grek (from pipelines/fineweb-2/main.py).
APERTUS_FW2HQ_ELL_FRACTION = {
    1: 0.33 * 0.95,
    2: 0.33 * 0.95,
    3: 0.33 * 0.95,
    4: 0.10 * 0.95,
    5: 0.10 * 0.95,
}


def stage_duration_b(stage: int) -> int:
    b = STAGE_BOUNDARIES_8B_B[stage]
    if b is None:
        return 0
    return b[1] - b[0]


def aggregate(outputs_dir: Path) -> dict:
    out: dict = {"per_dataset": {}, "stage_consumption_b": {}, "overall": {}}

    # Load per-dataset measurements
    per_dataset: dict[str, dict] = {}
    for jf in sorted(outputs_dir.glob("*.json")):
        if jf.name == "summary.json":
            continue
        per_dataset[jf.stem] = json.loads(jf.read_text())
    out["per_dataset"] = per_dataset

    # FineWeb-2-HQ Greek consumption per stage
    fw_meas = per_dataset.get("fineweb2_hq_ell")
    if not fw_meas:
        out["error"] = "missing fineweb2_hq_ell.json"
        return out

    fw_greek_tokens_total = fw_meas["tokens_with_bod_eod"]
    fw2hq_per_stage_b = {}
    for stage, (s_start, s_end) in {k: v for k, v in STAGE_BOUNDARIES_8B_B.items() if v}.items():
        # Greek tokens consumed via FineWeb-2-HQ in this stage:
        #   (stage duration / total stage pool) * fw2hq_pool_in_stage * (Greek share of fw2hq_pool)
        # Approximation: Greek share of `fineweb2_hq_plus_remaining` ≈ fw_greek_tokens / fw_hq_total_estimate.
        # Without a measured fw_hq_total, we treat the FW2HQ Greek-token figure as the post-p×0.95 contribution.
        # More direct: APERTUS_FW2HQ_ELL_FRACTION already encodes the filter; we then weight by stage duration as a fraction of pool availability.
        pool_dict = STAGE_POOLS_B[stage]
        stage_total_b = sum(pool_dict.values())
        stage_dur_b = s_end - s_start
        # Greek tokens *available* in this stage's FW2HQ slice:
        greek_avail_b = (fw_greek_tokens_total / 1e9) * APERTUS_FW2HQ_ELL_FRACTION[stage]
        # Greek tokens *consumed* this stage (fraction of pool that is actually consumed):
        consumption_rate = stage_dur_b / stage_total_b if stage_total_b else 0
        greek_consumed_b = greek_avail_b * consumption_rate * (
            pool_dict.get("fineweb2_hq_plus_remaining") or pool_dict.get("fineweb2_hq_plus_remaining_10pct", 0)
        ) / max(1, (pool_dict.get("fineweb2_hq_plus_remaining") or pool_dict.get("fineweb2_hq_plus_remaining_10pct", 0)))
        # Simpler restatement:
        # greek_consumed = greek_avail × (stage_dur / stage_pool_total)
        # The bracketed multiplication above resolves to 1; left here for documentation.
        greek_consumed_b = greek_avail_b * consumption_rate
        fw2hq_per_stage_b[stage] = {
            "stage_duration_b": stage_dur_b,
            "stage_pool_total_b": stage_total_b,
            "consumption_rate": round(consumption_rate, 4),
            "fw2hq_ell_filter_fraction": round(APERTUS_FW2HQ_ELL_FRACTION[stage], 4),
            "greek_available_b": round(greek_avail_b, 4),
            "greek_consumed_b": round(greek_consumed_b, 4),
        }

    # Wikipedia, EuroParl, EuroBlocks: Stage 5 only (per Table 6).
    s5_pool = STAGE_POOLS_B[5]
    s5_total = sum(s5_pool.values())
    s5_dur = stage_duration_b(5)
    s5_consumption_rate = s5_dur / s5_total if s5_total else 0

    stage5_aux_b = {}
    if "clean_wikipedia_el" in per_dataset:
        cw = per_dataset["clean_wikipedia_el"]["tokens_with_bod_eod"] / 1e9
        # Greek share of clean_wikipedia pool: assume cw_pool = 33 B fully reachable, Greek share is cw / total_cw.
        # Without a measured total Clean-Wikipedia size, we treat the Greek number as a tokens-available count
        # within the 33 B pool. Best estimate: Greek_consumed = Greek_tokens × (33B / total_clean_wiki_tokens) × consumption_rate.
        # We don't have total_clean_wiki_tokens; conservative bound: Greek_consumed_b ≤ cw × consumption_rate.
        stage5_aux_b["clean_wikipedia_el"] = {
            "greek_tokens_in_dataset_b": round(cw, 4),
            "stage5_pool_b": s5_pool["clean_wikipedia"],
            "consumption_rate_upper_bound": round(s5_consumption_rate, 4),
            "greek_consumed_b_upper_bound": round(cw * s5_consumption_rate, 4),
        }
    if "europarl_el" in per_dataset:
        ep = per_dataset["europarl_el"]["tokens_with_bod_eod"] / 1e9
        stage5_aux_b["europarl_el"] = {
            "greek_tokens_in_dataset_b": round(ep, 4),
            "stage5_pool_b": s5_pool["translation_parallel"],
            "consumption_rate_upper_bound": round(s5_consumption_rate, 4),
            "greek_consumed_b_upper_bound": round(ep * s5_consumption_rate, 4),
        }
    if "paradocs_el" in per_dataset:
        pd = per_dataset["paradocs_el"]["tokens_with_bod_eod"] / 1e9
        stage5_aux_b["paradocs_el"] = {
            "greek_tokens_in_dataset_b": round(pd, 4),
            "stage5_pool_b": s5_pool["translation_parallel"],
            "consumption_rate_upper_bound": round(s5_consumption_rate, 4),
            "greek_consumed_b_upper_bound": round(pd * s5_consumption_rate, 4),
        }
    if "euroblocks_el" in per_dataset:
        eb = per_dataset["euroblocks_el"]["tokens_with_bod_eod"] / 1e9
        stage5_aux_b["euroblocks_el"] = {
            "greek_tokens_in_dataset_b": round(eb, 4),
            "stage5_pool_b": s5_pool["task_data_3x1"],
            "consumption_rate_upper_bound": round(s5_consumption_rate, 4),
            "greek_consumed_b_upper_bound": round(eb * s5_consumption_rate * 3, 4),  # ×3 replicas
        }
    if "institutional_books_el" in per_dataset:
        ib = per_dataset["institutional_books_el"]["tokens_with_bod_eod"] / 1e9
        stage5_aux_b["institutional_books_el"] = {
            "greek_tokens_in_dataset_b": round(ib, 4),
            "note": "Long-context phase only; not in 13.5T main pretraining.",
        }

    fw_total_consumed = sum(s["greek_consumed_b"] for s in fw2hq_per_stage_b.values())
    aux_total_consumed_upper_bound = sum(
        v.get("greek_consumed_b_upper_bound", 0) for v in stage5_aux_b.values()
    )
    greek_total_b = fw_total_consumed + aux_total_consumed_upper_bound

    out["stage_consumption_b"] = {
        "fineweb2_hq_ell": fw2hq_per_stage_b,
        "stage5_aux": stage5_aux_b,
    }
    out["overall"] = {
        "greek_tokens_b": round(greek_total_b, 4),
        "fineweb2_hq_contribution_b": round(fw_total_consumed, 4),
        "stage5_aux_contribution_b_upper_bound": round(aux_total_consumed_upper_bound, 4),
        "denominator_b": TOTAL_8B_B,
        "denominator_source": "paper Table H.8 + ~200 B cooldown tail estimate",
        "greek_share_pct": round(100.0 * greek_total_b / TOTAL_8B_B, 4),
        "method_caveats": [
            "FineWeb2-HQ Greek consumed = G_hq × p × 0.95 × (stage_dur / stage_pool_total). "
            "Assumes quality filter selection within HQ has uniform tokens-per-doc; "
            "true value may differ ±5% if top-quality docs are systematically longer/shorter.",
            "Stage-5 auxiliary contributions are upper bounds: they assume Greek tokens in the dataset "
            "are consumed in full proportion to the dataset's stage-5 pool. Lower bound is 0 if Greek "
            "got downsampled into oblivion. Truth is between, closer to upper bound for Wikipedia/EuroParl.",
            "Institutional Books contribution is reported separately; it belongs to the long-context phase "
            "(~225 B total, paper Table 8), not the 13.5 T pretraining proper.",
        ],
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", default="/mnt/data/outputs")
    ap.add_argument("--summary", default="/mnt/data/outputs/summary.json")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    summary = aggregate(outputs_dir)
    Path(args.summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary["overall"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
