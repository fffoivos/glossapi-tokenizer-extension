#!/usr/bin/env python3
"""Parallel strict + relaxed exact dedup stages built directly off the
`run_docs_inventory.parquet` produced by Stage 1, bypassing the
serial SQLite rebuild documented as "many hours" in
`subprojects/01_1_corpus_dedup/DEDUP_SCRIPT_REPAIR_PLAN.md`.

The output parquet files match the schemas the dedup library expects
(`text_dedup.EXACT_GROUP_SCHEMA`, `EXACT_DROP_SCHEMA`,
`EXACT_RESULT_SCHEMA`). When all 3 files for a stage exist, the
library's `build_stage_results` reuses them via the
`{stage}:reuse_existing_parquet` fast-path and skips the rebuild
entirely.

Selection priority replicates `selection_priority_tuple` /
`representative_score` from `glossapi_corpus_cli/text_dedup.py`:

  (invalid_ocr_rank,
   needs_ocr_rank,
   -representative_score,
   greek_badness_score (inf if NULL),
   mojibake_badness_score (inf if NULL),
   ocr_rank,
   title_rank,
   author_rank,
   source_dataset,
   source_doc_id)

Smallest tuple wins.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import duckdb


def now_iso() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{now_iso()}] {msg}", flush=True)


# Stage-output schemas mirror text_dedup.py
# EXACT_RESULT_SCHEMA (membership): doc_key, group_hash, group_size, kept_doc_key, dropped(int64)
# EXACT_GROUP_SCHEMA (groups, multi-member only):
#   group_hash, group_size, kept_doc_key, member_doc_key,
#   member_source_dataset, member_source_doc_id, dropped(bool)
# EXACT_DROP_SCHEMA (drop list, dropped-only from multi-member):
#   doc_key, source_dataset, source_doc_id, kept_doc_key, group_hash, reason


def build_stage(
    con: duckdb.DuckDBPyConnection,
    *,
    stage: str,
    hash_column: str,
    text_length_field: str,
    inventory_path: Path,
    output_dir: Path,
    survivors_membership_path: Path | None,
) -> dict[str, int]:
    """Build memberships, groups, drops parquets for one exact stage.

    For RELAXED_STAGE, pass `survivors_membership_path` pointing at the
    strict membership parquet — only docs with strict.dropped == 0 are
    eligible.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    membership_path = output_dir / f"{stage}_memberships.parquet"
    groups_path = output_dir / f"{stage}_groups.parquet"
    drops_path = output_dir / f"{stage}_drop_list.parquet"

    if survivors_membership_path is not None:
        survivors_filter = (
            f"WHERE doc_key IN (SELECT doc_key FROM read_parquet('{survivors_membership_path}') "
            "WHERE dropped = 0)"
        )
    else:
        survivors_filter = ""

    # Compute priority tuple (smallest wins) and stash hash + survivor pool.
    # Use STRUCT (DuckDB struct comparison is field-order lex), aligning with
    # selection_priority_tuple semantics. We cast NULL float ranks to a
    # sentinel large value (1e308) — Python uses float('inf') but DuckDB
    # struct sort handles 1e308 fine without infinities-in-floats hazards.
    INF = "1e308"
    # representative_rank = -representative_score
    # selection_length_value = COALESCE(len_greek, <text_length_field>, 0)
    repr_rank_sql = f"""
        CASE
          WHEN COALESCE(len_greek, {text_length_field}, 0) <= 0 THEN 0.0
          WHEN greek_badness_score IS NULL THEN 0.0
          ELSE -GREATEST(0.0,
                        CAST(COALESCE(len_greek, {text_length_field}) AS DOUBLE)
                        * (1.0 - greek_badness_score / 10.0))
        END
    """
    log(f"{stage}: building ranked rows from inventory")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE _ranked AS
        SELECT
          doc_key,
          source_dataset,
          source_doc_id,
          {hash_column} AS group_hash,
          STRUCT_PACK(
            inv_ocr := CASE WHEN needs_ocr = 1 AND ocr_success <> 1 THEN 1.0 ELSE 0.0 END,
            no_ocr  := CASE WHEN needs_ocr = 0 THEN 0.0
                            WHEN needs_ocr = 1 THEN 1.0
                            ELSE 0.5 END,
            repr    := {repr_rank_sql},
            greek   := COALESCE(greek_badness_score, {INF}),
            moji    := COALESCE(mojibake_badness_score, {INF}),
            ocr     := CASE WHEN ocr_success = 1 THEN 0.0
                            WHEN ocr_success = 0 THEN 1.0
                            ELSE 0.5 END,
            ti      := CASE WHEN title  IS NOT NULL THEN 0 ELSE 1 END,
            au      := CASE WHEN author IS NOT NULL THEN 0 ELSE 1 END,
            sd      := source_dataset,
            si      := source_doc_id
          ) AS prio
        FROM read_parquet('{inventory_path}')
        {survivors_filter}
        """
    )

    log(f"{stage}: computing keepers per group")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE _keepers AS
        SELECT
          group_hash,
          COUNT(*) AS group_size,
          arg_min(doc_key, prio) AS kept_doc_key
        FROM _ranked
        GROUP BY group_hash
        """
    )

    log(f"{stage}: writing memberships → {membership_path.name}")
    con.execute(
        f"""
        COPY (
          SELECT
            r.doc_key,
            k.group_hash,
            CAST(k.group_size AS BIGINT) AS group_size,
            k.kept_doc_key,
            CAST(CASE WHEN r.doc_key = k.kept_doc_key THEN 0 ELSE 1 END AS BIGINT) AS dropped
          FROM _ranked r
          JOIN _keepers k USING (group_hash)
        ) TO '{membership_path}' (FORMAT 'parquet', COMPRESSION 'zstd')
        """
    )

    log(f"{stage}: writing groups (multi-member) → {groups_path.name}")
    con.execute(
        f"""
        COPY (
          SELECT
            r.group_hash,
            CAST(k.group_size AS BIGINT) AS group_size,
            k.kept_doc_key,
            r.doc_key AS member_doc_key,
            r.source_dataset AS member_source_dataset,
            r.source_doc_id AS member_source_doc_id,
            r.doc_key <> k.kept_doc_key AS dropped
          FROM _ranked r
          JOIN _keepers k USING (group_hash)
          WHERE k.group_size > 1
        ) TO '{groups_path}' (FORMAT 'parquet', COMPRESSION 'zstd')
        """
    )

    log(f"{stage}: writing drop list → {drops_path.name}")
    con.execute(
        f"""
        COPY (
          SELECT
            r.doc_key,
            r.source_dataset,
            r.source_doc_id,
            k.kept_doc_key,
            r.group_hash,
            '{stage}' AS reason
          FROM _ranked r
          JOIN _keepers k USING (group_hash)
          WHERE k.group_size > 1
            AND r.doc_key <> k.kept_doc_key
        ) TO '{drops_path}' (FORMAT 'parquet', COMPRESSION 'zstd')
        """
    )

    # Tally counts.
    counts = con.execute(
        """
        SELECT
          (SELECT COUNT(*) FROM _ranked) AS total_rows,
          (SELECT COUNT(*) FROM _keepers) AS total_groups,
          (SELECT COUNT(*) FROM _keepers WHERE group_size > 1) AS dup_groups,
          (SELECT SUM(group_size) FROM _keepers WHERE group_size > 1) AS dup_rows
        """
    ).fetchone()
    total_rows, total_groups, dup_groups, dup_rows = counts
    dropped = (dup_rows or 0) - (dup_groups or 0)
    kept = (total_rows or 0) - dropped
    log(
        f"{stage}: total_rows={total_rows} total_groups={total_groups} "
        f"dup_groups={dup_groups} dup_rows={dup_rows} dropped={dropped} kept={kept}"
    )
    return {
        "stage": stage,
        "total_rows": int(total_rows or 0),
        "total_groups": int(total_groups or 0),
        "dup_groups": int(dup_groups or 0),
        "dup_rows": int(dup_rows or 0),
        "dropped": int(dropped),
        "kept": int(kept),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("/home/foivos/runs/wave2_20260426/dedup_run/run_docs_inventory.parquet"),
    )
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=Path("/home/foivos/runs/wave2_20260426/dedup_run/stage_01_exact"),
    )
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument(
        "--memory-limit",
        default="700GB",
        help="DuckDB memory limit (instance has 960 GB).",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=Path("/home/foivos/runs/wave2_20260426/duckdb_tmp"),
    )
    args = parser.parse_args()

    args.temp_dir.mkdir(parents=True, exist_ok=True)
    args.stage_dir.mkdir(parents=True, exist_ok=True)

    if not args.inventory.exists():
        log(f"FATAL: inventory not found: {args.inventory}")
        return 1

    log(f"opening DuckDB threads={args.threads} memory_limit={args.memory_limit}")
    con = duckdb.connect(":memory:")
    con.execute(f"PRAGMA threads={args.threads}")
    con.execute(f"PRAGMA memory_limit='{args.memory_limit}'")
    con.execute(f"PRAGMA temp_directory='{args.temp_dir}'")

    started = time.time()

    strict_summary = build_stage(
        con,
        stage="strict_exact",
        hash_column="exact_strict_hash",
        text_length_field="strict_text_chars",
        inventory_path=args.inventory,
        output_dir=args.stage_dir,
        survivors_membership_path=None,
    )
    elapsed_strict = time.time() - started
    log(f"strict_exact done in {elapsed_strict:.1f}s")

    strict_membership = args.stage_dir / "strict_exact_memberships.parquet"
    relaxed_summary = build_stage(
        con,
        stage="relaxed_exact",
        hash_column="exact_relaxed_hash",
        text_length_field="relaxed_text_chars",
        inventory_path=args.inventory,
        output_dir=args.stage_dir,
        survivors_membership_path=strict_membership,
    )
    elapsed_relaxed = time.time() - started - elapsed_strict
    log(f"relaxed_exact done in {elapsed_relaxed:.1f}s")

    log(f"all done; strict={strict_summary} relaxed={relaxed_summary}")
    log(f"total wall: {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
