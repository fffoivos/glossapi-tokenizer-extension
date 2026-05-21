#!/usr/bin/env python3
"""Build the CPT-facing dedup overlay for Apertus continuation training.

This is intentionally narrower than the audit report:

* source side: the published nanochat dataset only
  (fffoivos/glossapi-greek-nanochat-pretraining-dataset), including its HPLT
  shard files
* excluded source side: the duplicate/separate hplt_clean60 repo that the
  audit also pulled for diagnostics
* Apertus side: the overlap parquet outputs from steps 06/07

The output is a hard-drop list of nanochat docs seen by Apertus pretraining.
For the final CPT build, apply this hard-drop list first, then replay the
latest nanochat builder dedup bundle on the remaining rows. That ordering lets
duplicate families choose a fresh representative when the old representative
is Apertus-overlapping.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

SUB = Path(__file__).resolve().parents[2]
RUN_ID = (SUB / "manifests/CURRENT_RUN_ID").read_text(encoding="utf-8").strip()
ART = SUB / "artifacts" / RUN_ID
OVERLAP = ART / "overlap"
OUT = ART / "cpt_final_overlay"

STAGE_PRIORITY = {
    "strict_exact": 0,
    "relaxed_exact": 1,
    "near": 2,
}

NANOCHAT_REPO_ID = "fffoivos/glossapi-greek-nanochat-pretraining-dataset"
NANOCHAT_DEDUP_RUN_ID = "wave2_20260426_builder_metadata_v2_latest_cleaner_20260507"
NANOCHAT_DEDUP_ROOT = (
    "dedup_metadata/"
    "wave2_20260426_builder_metadata_v2_latest_cleaner_20260507/"
    "builder_metadata"
)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    frames: list[pl.DataFrame] = []
    for stage, priority in STAGE_PRIORITY.items():
        stage_dir = OVERLAP / stage
        if not stage_dir.exists():
            continue
        for path in sorted(stage_dir.glob("*_x_glossapi_nanochat.parquet")):
            df = pl.read_parquet(path)
            if df.height == 0:
                continue
            jaccard_expr = (
                pl.col("estimated_jaccard")
                if "estimated_jaccard" in df.columns
                else pl.lit(None, dtype=pl.Float64)
            )
            frames.append(
                df.select(
                    pl.col("c_doc_key").alias("doc_key"),
                    pl.col("c_source_dataset").alias("source_dataset"),
                    pl.col("c_source_doc_id").alias("source_doc_id"),
                    pl.col("a_source_dataset").alias("apertus_source_dataset"),
                    pl.lit(stage).alias("overlap_stage"),
                    pl.lit(priority).alias("stage_priority"),
                    pl.lit(path.stem).alias("pair"),
                    jaccard_expr.alias("estimated_jaccard"),
                )
            )

    if not frames:
        raise SystemExit(f"No *_x_glossapi_nanochat overlap parquets found under {OVERLAP}")

    raw = pl.concat(frames, how="vertical")
    per_doc = (
        raw.sort(["doc_key", "source_dataset", "stage_priority"])
        .group_by(["doc_key", "source_dataset"])
        .agg(
            pl.col("source_doc_id").first(),
            pl.col("overlap_stage").first().alias("best_overlap_stage"),
            pl.col("stage_priority").min().alias("best_stage_priority"),
            pl.col("apertus_source_dataset").first().alias("best_apertus_source_dataset"),
            pl.col("pair").first().alias("best_pair"),
            pl.col("estimated_jaccard").max().alias("best_estimated_jaccard"),
            pl.len().alias("raw_match_rows"),
        )
        .sort(["source_dataset", "doc_key"])
    )

    drop_list_path = OUT / "apertus_overlap_drop_docs.parquet"
    per_doc.write_parquet(drop_list_path, compression="zstd")

    by_source = per_doc.group_by("source_dataset").len().sort("len", descending=True)
    by_stage = per_doc.group_by("best_overlap_stage").len().sort("len", descending=True)
    summary = {
        "artifact": "cpt_final_overlay",
        "created_from_run": RUN_ID,
        "source_pool_used": NANOCHAT_REPO_ID,
        "source_pool_excluded": "fffoivos/hplt-greek-ge8-no-mt-clean60-wave4",
        "drop_list_path": str(drop_list_path),
        "raw_overlap_rows": raw.height,
        "unique_apertus_overlap_docs": per_doc.height,
        "by_source_dataset": dict(
            zip(by_source["source_dataset"].to_list(), map(int, by_source["len"].to_list()))
        ),
        "by_best_overlap_stage": dict(
            zip(by_stage["best_overlap_stage"].to_list(), map(int, by_stage["len"].to_list()))
        ),
        "internal_nanochat_dedup_bundle_to_use": {
            "repo_id": NANOCHAT_REPO_ID,
            "latest_run_id": NANOCHAT_DEDUP_RUN_ID,
            "builder_metadata_root": NANOCHAT_DEDUP_ROOT,
            "greek_diacritic_policy": "preserve",
            "builder_metadata_version": "builder_metadata_v2",
        },
        "correct_application_order": [
            "start from published nanochat source rows",
            "hard-exclude docs in apertus_overlap_drop_docs.parquet",
            "apply nanochat builder dedup bundle with drop_intra_and_inter on the remaining rows",
            "then resolve CPT source/token mix for Apertus",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
