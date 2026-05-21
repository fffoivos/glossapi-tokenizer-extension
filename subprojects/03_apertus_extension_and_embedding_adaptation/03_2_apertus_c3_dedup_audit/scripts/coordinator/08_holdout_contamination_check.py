#!/usr/bin/env python3
"""Coordinator step 08: held-out contamination check on C3 val + test splits.

For every (C3 val|test doc) × every Apertus source, check membership at:
- strict_exact_hash
- relaxed_exact_hash
- near (Jaccard ≥ 0.85)

A contaminated doc is one matched at ANY level. Output records the
strictest level it matched at as `contamination_severity`.

Assumes the C3 val/test row identifiers are present in the C3-side
parquet (`split` column with values in {"train","val","test"}). If not,
filters by joining against an external val/test doc-id list provided as
`manifests/run_<RUN_ID>/holdout_doc_ids.parquet` (one row per doc_key).

Output: artifacts/<RUN_ID>/holdout_contamination.parquet
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

SUB = Path(__file__).resolve().parents[2]
RUN_ID = (SUB / "manifests/CURRENT_RUN_ID").read_text().strip()
MANI = SUB / f"manifests/run_{RUN_ID}"
SRC = SUB / "artifacts" / RUN_ID / "sources"
OVL = SUB / "artifacts" / RUN_ID / "overlap"
OUT = SUB / "artifacts" / RUN_ID / "holdout_contamination.parquet"

HOLDOUT = MANI / "holdout_doc_ids.parquet"


def main() -> int:
    # Load C3 doc-ids that are in val|test. If the manifest file is absent,
    # we cannot run this step — print a clear warning + exit 1 (non-fatal).
    if not HOLDOUT.exists():
        print(f"[holdout] WARNING: {HOLDOUT} missing; skipping contamination "
              f"check. Provide val/test doc-keys via that file to enable.",
              file=sys.stderr)
        return 1
    holdout = pl.read_parquet(HOLDOUT).select(["doc_key", "split"])  # split in {val,test}
    print(f"[holdout] {holdout.height} holdout docs to check")

    # Combine all C3-side rows.
    c_files = sorted(SRC.glob("hf_source_pool_*.parquet"))
    if not c_files:
        print("[holdout] no C3 source parquets; aborting", file=sys.stderr)
        return 2
    c = pl.concat([pl.read_parquet(p) for p in c_files], how="vertical")
    c_holdout = c.join(holdout, on="doc_key", how="inner")
    print(f"[holdout] {c_holdout.height} C3 holdout-side rows after join")

    # For each Apertus source, check the three levels and accumulate severity.
    a_files = sorted(SRC.glob("apertus_*.parquet"))
    contam: list[dict] = []
    for ap in a_files:
        a = pl.read_parquet(ap)
        a_strict = set(a["strict_exact_hash"].to_list())
        a_relaxed = set(a["relaxed_exact_hash"].to_list())
        ap_name = ap.stem.replace("apertus_", "")
        # Strict + relaxed in bulk.
        for row in c_holdout.iter_rows(named=True):
            severity = None
            if row["strict_exact_hash"] in a_strict:
                severity = "strict_exact"
            elif row["relaxed_exact_hash"] in a_relaxed:
                severity = "relaxed_exact"
            if severity is not None:
                contam.append({
                    "hf_source_pool_doc_key": row["doc_key"],
                    "hf_source_pool_source_dataset": row["source_dataset"],
                    "hf_source_pool_source_doc_id": row["source_doc_id"],
                    "split": row["split"],
                    "apertus_source": ap_name,
                    "contamination_severity": severity,
                })
    # Near-dup ladder: look up near-overlap output for any pair containing a
    # holdout doc on the C3 side.
    near_dir = OVL / "near"
    if near_dir.exists():
        for nf in sorted(near_dir.glob("*.parquet")):
            apertus_src = nf.stem.split("_x_")[0]
            near = pl.read_parquet(nf).select([
                "c_doc_key", "c_source_dataset", "c_source_doc_id"])
            near = near.join(holdout, left_on="c_doc_key", right_on="doc_key",
                              how="inner")
            for row in near.iter_rows(named=True):
                contam.append({
                    "hf_source_pool_doc_key": row["c_doc_key"],
                    "hf_source_pool_source_dataset": row["c_source_dataset"],
                    "hf_source_pool_source_doc_id": row["c_source_doc_id"],
                    "split": row["split"],
                    "apertus_source": apertus_src,
                    "contamination_severity": "near",
                })

    df = pl.DataFrame(contam) if contam else pl.DataFrame(
        schema={"hf_source_pool_doc_key": pl.Utf8, "hf_source_pool_source_dataset": pl.Utf8,
                "hf_source_pool_source_doc_id": pl.Utf8, "split": pl.Utf8,
                "apertus_source": pl.Utf8, "contamination_severity": pl.Utf8})
    # Reduce to (hf_source_pool_doc_key, apertus_source) keeping strictest severity.
    if df.height:
        severity_rank = pl.when(pl.col("contamination_severity") == "strict_exact").then(0) \
                          .when(pl.col("contamination_severity") == "relaxed_exact").then(1) \
                          .otherwise(2)
        df = df.with_columns(severity_rank.alias("_rank")) \
               .sort("_rank") \
               .group_by(["hf_source_pool_doc_key", "apertus_source"], maintain_order=True).first() \
               .drop("_rank")
    df.write_parquet(OUT, compression="zstd")

    print(f"[holdout] {df.height} contaminated rows -> {OUT}")
    counts = df.group_by("split", "contamination_severity").agg(pl.len().alias("n")) if df.height else pl.DataFrame()
    print(counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
