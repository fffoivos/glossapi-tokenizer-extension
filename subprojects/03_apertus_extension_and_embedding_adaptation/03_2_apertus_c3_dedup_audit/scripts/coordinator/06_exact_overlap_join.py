#!/usr/bin/env python3
"""Coordinator step 06: strict_exact + relaxed_exact INNER JOIN per
(Apertus source × C3 source) pair.

For each Apertus source A and each C3 source C:
- strict_exact join: A.strict_exact_hash ∩ C.strict_exact_hash
- relaxed_exact join: A.relaxed_exact_hash ∩ C.relaxed_exact_hash

Output: artifacts/<RUN_ID>/overlap/{strict_exact,relaxed_exact}/<A>_x_<C>.parquet
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

SUB = Path(__file__).resolve().parents[2]
RUN_ID = (SUB / "manifests/CURRENT_RUN_ID").read_text().strip()
SRC = SUB / "artifacts" / RUN_ID / "sources"
OUT = SUB / "artifacts" / RUN_ID / "overlap"
(OUT / "strict_exact").mkdir(parents=True, exist_ok=True)
(OUT / "relaxed_exact").mkdir(parents=True, exist_ok=True)


def join_pair(a_path: Path, c_path: Path, hash_col: str, out_path: Path) -> int:
    a = pl.scan_parquet(a_path).select(
        ["doc_key", "source_dataset", "source_doc_id", hash_col]
    ).rename({"doc_key": "a_doc_key", "source_dataset": "a_source_dataset",
              "source_doc_id": "a_source_doc_id"})
    c = pl.scan_parquet(c_path).select(
        ["doc_key", "source_dataset", "source_doc_id", hash_col]
    ).rename({"doc_key": "c_doc_key", "source_dataset": "c_source_dataset",
              "source_doc_id": "c_source_doc_id"})
    j = a.join(c, on=hash_col, how="inner")
    df = j.collect()
    if df.height:
        df.write_parquet(out_path, compression="zstd")
    return df.height


def main() -> int:
    a_files = sorted(SRC.glob("apertus_*.parquet"))
    c_files = sorted(SRC.glob("hf_source_pool_*.parquet"))
    if not a_files or not c_files:
        print(f"[exact] missing sources under {SRC}", file=sys.stderr)
        return 2
    rows: list[dict] = []
    for stage, hash_col in (("strict_exact", "strict_exact_hash"),
                             ("relaxed_exact", "relaxed_exact_hash")):
        for ap in a_files:
            for cp in c_files:
                pair = f"{ap.stem.replace('apertus_', '')}_x_{cp.stem.replace('hf_source_pool_', '')}"
                op = OUT / stage / f"{pair}.parquet"
                n = join_pair(ap, cp, hash_col, op)
                rows.append({"stage": stage, "pair": pair, "matched_pairs": n})
                print(f"[exact] {stage}  {pair:60s}  matches={n}")
    summary = pl.DataFrame(rows)
    summary.write_parquet(OUT.parent / "exact_overlap_summary.parquet")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
