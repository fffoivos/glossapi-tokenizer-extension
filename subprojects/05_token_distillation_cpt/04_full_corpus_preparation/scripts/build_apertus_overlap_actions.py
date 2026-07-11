#!/usr/bin/env python3
"""Bind the validated Apertus-overlap drop overlay to canonical Nanochat rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from finalization_io import (
    configure_duckdb,
    discover_parquet,
    parquet_file_receipt,
    read_json_object,
    sha256_file,
    sql_path_list,
    sql_string,
    utc_now,
    write_json_atomic,
)


SCHEMA_VERSION = "full_cpt_apertus_overlap_actions_v1"
ACTION_SCHEMA = "full_cpt_lineage_document_action_v1"
OVERLAY_FILENAME = "apertus_overlap_drop_docs.parquet"


def overlay_from_receipt(sources_path: Path, receipt_path: Path) -> Path:
    sources = read_json_object(sources_path)
    receipt = read_json_object(receipt_path)
    if receipt.get("schema_version") != "full_cpt_acquisition_receipt_v1":
        raise ValueError("unsupported acquisition receipt")
    expected = sources.get("apertus_overlap_overlay", {})
    matches = [
        row
        for row in receipt.get("sources", [])
        if isinstance(row, dict) and row.get("source_id") == "apertus_overlap_overlay"
    ]
    if len(matches) != 1:
        raise ValueError("acquisition receipt must contain exactly one Apertus overlap source")
    source = matches[0]
    if source.get("repo_id") != expected.get("repo_id"):
        raise ValueError("Apertus overlap repository differs from source registry")
    if source.get("revision") != expected.get("revision"):
        raise ValueError("Apertus overlap revision differs from source registry")
    files = [
        Path(str(row["local_path"])).resolve()
        for row in source.get("files", [])
        if Path(str(row.get("local_path", ""))).name == OVERLAY_FILENAME
    ]
    if len(files) != 1 or not files[0].is_file():
        raise FileNotFoundError(f"receipt does not resolve exactly one {OVERLAY_FILENAME}")
    return files[0]


def build(args: argparse.Namespace) -> dict[str, Any]:
    import duckdb

    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("refusing to overwrite immutable overlay actions/manifest")
    normalization = read_json_object(args.normalization_manifest)
    if normalization.get("schema_version") != "full_cpt_normalization_manifest_v1":
        raise ValueError("unsupported normalization manifest")
    if Path(str(normalization.get("output", ""))).resolve() != args.canonical_root.resolve():
        raise ValueError("normalization manifest is bound to a different canonical root")
    overlay = overlay_from_receipt(args.sources, args.acquisition_receipt)
    base_root = args.canonical_root / "nanochat_base"
    base_files = discover_parquet(base_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    configure_duckdb(
        connection,
        temporary_directory=args.temporary_directory,
        memory_limit=args.memory_limit,
        threads=args.threads,
    )
    try:
        connection.execute(
            f"CREATE TEMP VIEW base AS SELECT * FROM read_parquet({sql_path_list(base_files)}, union_by_name=true)"
        )
        connection.execute(
            f"CREATE TEMP VIEW overlay AS SELECT * FROM read_parquet({sql_string(overlay)})"
        )
        missing_columns: list[str] = []
        for view, required in (
            (
                "base",
                {
                    "stable_uid",
                    "source_dataset",
                    "source_doc_id",
                    "normalized_text_sha256",
                },
            ),
            (
                "overlay",
                {
                    "doc_key",
                    "source_dataset",
                    "source_doc_id",
                    "best_overlap_stage",
                    "best_estimated_jaccard",
                },
            ),
        ):
            observed = {str(row[0]) for row in connection.execute(f"DESCRIBE {view}").fetchall()}
            missing_columns.extend(f"{view}.{name}" for name in sorted(required - observed))
        if missing_columns:
            raise ValueError(f"Apertus overlay join columns are missing: {missing_columns}")
        counts = connection.execute(
            """
            WITH overlay_keys AS (
              SELECT source_dataset, source_doc_id, count(*) AS rows
              FROM overlay GROUP BY 1, 2
            ), base_keys AS (
              SELECT source_dataset, source_doc_id, count(*) AS rows
              FROM base GROUP BY 1, 2
            )
            SELECT
              (SELECT count(*) FROM overlay),
              (SELECT count(*) FROM overlay_keys),
              (SELECT count(*) FROM overlay_keys o JOIN base_keys b USING (source_dataset, source_doc_id)),
              (SELECT count(*) FROM overlay_keys o LEFT JOIN base_keys b USING (source_dataset, source_doc_id)
                 WHERE b.source_doc_id IS NULL),
              (SELECT count(*) FROM base),
              (SELECT count(*) - count(DISTINCT stable_uid) FROM base)
            """
        ).fetchone()
        if int(counts[3]):
            raise ValueError(f"Apertus overlap contains {counts[3]} natural keys absent from Nanochat")
        if int(counts[5]):
            raise ValueError("canonical Nanochat stable_uid values are not unique")
        output_sql = sql_string(args.output.resolve())
        connection.execute(
            f"""
            COPY (
              WITH overlay_keys AS (
                SELECT source_dataset, source_doc_id,
                       min(doc_key) AS relationship_key,
                       arg_max(best_overlap_stage, best_estimated_jaccard) AS best_overlap_stage,
                       max(best_estimated_jaccard) AS best_estimated_jaccard,
                       count(*)::BIGINT AS overlay_match_rows
                FROM overlay GROUP BY 1, 2
              )
              SELECT
                {sql_string(ACTION_SCHEMA)}::VARCHAR AS schema_version,
                base.stable_uid::VARCHAR AS stable_uid,
                'nanochat_base'::VARCHAR AS source_id,
                base.source_dataset::VARCHAR AS source_dataset,
                base.source_doc_id::VARCHAR AS source_doc_id,
                base.normalized_text_sha256::VARCHAR AS input_text_sha256,
                'drop'::VARCHAR AS action,
                'apertus_pretraining_overlap'::VARCHAR AS reason,
                overlay_keys.relationship_key::VARCHAR AS relationship_key,
                'validated_apertus_overlap_overlay_v1'::VARCHAR AS resolution_policy,
                overlay_keys.best_overlap_stage::VARCHAR AS best_overlap_stage,
                overlay_keys.best_estimated_jaccard::DOUBLE AS best_estimated_jaccard,
                overlay_keys.overlay_match_rows::BIGINT AS overlay_match_rows
              FROM base JOIN overlay_keys USING (source_dataset, source_doc_id)
              ORDER BY base.stable_uid
            ) TO {output_sql} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000)
            """
        )
        action_rows = int(
            connection.execute(
                f"SELECT count(*) FROM read_parquet({output_sql})"
            ).fetchone()[0]
        )
    finally:
        connection.close()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "completed_at": utc_now(),
        "sources_config": str(args.sources.resolve()),
        "sources_config_sha256": sha256_file(args.sources),
        "acquisition_receipt": str(args.acquisition_receipt.resolve()),
        "acquisition_receipt_sha256": sha256_file(args.acquisition_receipt),
        "normalization_manifest": str(args.normalization_manifest.resolve()),
        "normalization_manifest_sha256": sha256_file(args.normalization_manifest),
        "base_root": str(base_root.resolve()),
        "overlay": str(overlay),
        "overlay_sha256": sha256_file(overlay),
        "counts": {
            "overlay_rows": int(counts[0]),
            "overlay_natural_keys": int(counts[1]),
            "matched_natural_keys": int(counts[2]),
            "unmatched_natural_keys": int(counts[3]),
            "base_rows": int(counts[4]),
            "action_rows": action_rows,
        },
        "actions": parquet_file_receipt(args.output),
    }
    write_json_atomic(args.manifest, payload)
    return payload


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--normalization-manifest", type=Path, required=True)
    parser.add_argument("--acquisition-receipt", type=Path, required=True)
    parser.add_argument("--sources", type=Path, default=here / "configs" / "sources.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--temporary-directory", type=Path, required=True)
    parser.add_argument("--memory-limit", default="200GB")
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be >= 1")
    payload = build(args)
    print(json.dumps({"ok": True, "counts": payload["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
