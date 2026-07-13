#!/usr/bin/env python3
"""Close a verification-only duplicate scan after Agent 1 v3 anonymization.

The detector is allowed to *observe* any exact/near collisions introduced by
direct-identifier masking, but this tool never materializes a second deduped
corpus.  Before it reports zero collisions it proves all of the following:

* the anonymization manifest exactly receipts the supplied public corpus tree;
* each anonymized ``stable_uid``/``cleaned_text_sha256`` occurs once;
* each content-bound detector decision maps one-to-one to that inventory and
  binds the same anonymized text hash; and
* the generic detector used the frozen production recipe, with raw and
  content-bound decisions agreeing exactly.

Thus a zero result cannot be produced from a partial detector inventory or a
merely plausible corpus-root string.  The compact report contains no text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


WRAPPER_SCHEMA = "full_cpt_dedup_wrapper_manifest_v1"
ANONYMIZATION_SCHEMA = "agent1_full_corpus_v3_anonymization_manifest_v1"
REPORT_SCHEMA = "agent1_full_corpus_v3_postmask_duplicate_verification_v1"
CONTENT_BOUND_SCHEMA = "full_cpt_dedup_decisions_content_bound_v1"
PRODUCTION_RECIPE_ID = "greek_cpt_text_dedup_v1"
PRODUCTION_PARAMETERS = {
    "greek_diacritic_policy": "preserve",
    "minhash_threshold": 0.85,
    "num_perm": 128,
    "bands": 32,
    "rows_per_band": 4,
    "shingle_mode": "token",
    "shingle_size": 5,
    "max_bucket_size": 5000,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.stat().st_size < 1:
        raise FileNotFoundError(f"required non-empty file is missing: {resolved}")
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def read_object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size < 1:
        raise FileNotFoundError(f"required non-empty JSON file is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def safe_relative_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}: missing relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label}: path must be a safe relative path")
    return path


def validate_file_receipt(value: object, *, label: str, required_schema: str | None = None) -> Path:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}: expected receipt object")
    path = Path(str(value.get("path", ""))).resolve()
    if not path.is_file() or path.is_symlink() or path.stat().st_size < 1:
        raise FileNotFoundError(f"{label}: bound file is missing/unsafe: {path}")
    if path.stat().st_size != value.get("bytes") or sha256_file(path) != value.get("sha256"):
        raise ValueError(f"{label}: bytes/SHA-256 drift")
    if required_schema is not None and value.get("schema_version") != required_schema:
        raise ValueError(f"{label}: receipt schema drift")
    return path


def _verify_parquet_receipt(path: Path, receipt: Mapping[str, object], *, label: str) -> dict[str, Any]:
    import pyarrow.parquet as pq

    required = {"path", "sha256", "bytes", "rows", "row_groups"}
    if set(receipt) != required:
        raise ValueError(f"{label}: Parquet receipt key drift")
    if not path.is_file() or path.is_symlink() or path.stat().st_size < 1:
        raise FileNotFoundError(f"{label}: Parquet file is missing/unsafe: {path}")
    if path.stat().st_size != receipt.get("bytes") or sha256_file(path) != receipt.get("sha256"):
        raise ValueError(f"{label}: Parquet bytes/SHA-256 drift")
    metadata = pq.ParquetFile(path).metadata
    if metadata.num_rows != receipt.get("rows") or metadata.num_row_groups != receipt.get("row_groups"):
        raise ValueError(f"{label}: Parquet metadata drift")
    return {
        "relative_path": str(receipt["path"]),
        "bytes": int(receipt["bytes"]),
        "sha256": str(receipt["sha256"]),
        "rows": metadata.num_rows,
        "row_groups": metadata.num_row_groups,
    }


def validate_anonymized_inventory(
    *, anonymization_manifest: Path, source_corpus: Path
) -> tuple[dict[str, Any], list[Path], list[dict[str, Any]]]:
    """Verify the manifest's public-output receipt set exactly covers corpus."""

    manifest = read_object(anonymization_manifest)
    if manifest.get("schema_version") != ANONYMIZATION_SCHEMA or manifest.get("status") != "completed":
        raise ValueError("a completed v3 anonymization manifest is required")
    corpus = source_corpus.resolve()
    if not corpus.is_dir() or corpus.is_symlink():
        raise FileNotFoundError(f"anonymized corpus root is missing/unsafe: {corpus}")
    if Path(str(manifest.get("output", ""))).resolve() != corpus:
        raise ValueError("anonymization manifest output differs from supplied corpus root")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("anonymization manifest has no public output receipts")
    expected: dict[Path, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    for index, row in enumerate(files):
        if not isinstance(row, Mapping) or not isinstance(row.get("output"), Mapping):
            raise ValueError(f"anonymization manifest output receipt {index} is missing")
        receipt = row["output"]
        relative = safe_relative_path(receipt.get("path"), label=f"anonymization output receipt {index}")
        path = (corpus / relative).resolve()
        try:
            path.relative_to(corpus)
        except ValueError as exc:
            raise ValueError(f"anonymization output receipt {index} escapes corpus root") from exc
        if path in expected:
            raise ValueError(f"duplicate anonymization output receipt: {relative}")
        expected[path] = _verify_parquet_receipt(path, receipt, label=f"anonymization output receipt {index}")
        receipts.append(expected[path])
    actual = {path.resolve() for path in corpus.rglob("*.parquet") if not path.name.startswith(".")}
    if actual != set(expected):
        raise ValueError(
            "anonymization public-output inventory differs from its manifest receipts: "
            f"expected={len(expected)} actual={len(actual)}"
        )
    return manifest, sorted(expected), sorted(receipts, key=lambda item: str(item["relative_path"]))


def _sql_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sql_identifier(value: str) -> str:
    """Quote a Parquet column name before embedding it in closure SQL."""

    if not value or "\x00" in value:
        raise ValueError("unsafe empty/NUL Parquet column name")
    return '"' + value.replace('"', '""') + '"'


def _sql_path_list(paths: Iterable[Path]) -> str:
    values = ",".join(_sql_string(path.resolve()) for path in paths)
    if not values:
        raise ValueError("a non-empty Parquet inventory is required")
    return f"[{values}]"


def _validate_production_detector_manifest(manifest: Mapping[str, Any]) -> None:
    recipe = manifest.get("recipe")
    if not isinstance(recipe, Mapping) or recipe.get("id") != PRODUCTION_RECIPE_ID or recipe.get("mode") != "production":
        raise ValueError("post-mask detector did not use the frozen production recipe")
    if recipe.get("approved_production_parameters") != PRODUCTION_PARAMETERS:
        raise ValueError("post-mask detector recipe contract drift")
    if manifest.get("dedup_parameters") != PRODUCTION_PARAMETERS:
        raise ValueError("post-mask detector parameters differ from the frozen production recipe")
    identity = manifest.get("identity_contract")
    if not isinstance(identity, Mapping) or identity.get("input_text_sha256") != "verified canonical cleaned_text_sha256":
        raise ValueError("post-mask detector lacks the expected anonymized-text content-binding contract")


def validate_decision_inventory(
    *, corpus_files: list[Path], raw_path: Path, bound_path: Path
) -> dict[str, int]:
    """Use DuckDB to prove the detector inventories close exactly.

    The query is deliberately set-based: it avoids materialising the corpus or
    decisions into Python objects and can run against the full sharded corpus.
    """

    import duckdb
    import pyarrow.parquet as pq

    raw = pq.ParquetFile(raw_path)
    bound = pq.ParquetFile(bound_path)
    required = {"doc_key", "source_doc_id", "decision", "kept_doc_key"}
    missing = required - set(raw.schema_arrow.names)
    if missing:
        raise ValueError(f"post-mask raw decisions lack required columns: {sorted(missing)}")
    expected_bound = [*raw.schema_arrow.names, "stable_uid", "input_text_sha256"]
    if bound.schema_arrow.names != expected_bound:
        raise ValueError("post-mask content-bound decision schema does not close over raw decisions")
    corpus_columns = set(pq.ParquetFile(corpus_files[0]).schema_arrow.names)
    required_corpus = {"stable_uid", "cleaned_text_sha256"}
    if missing_corpus := required_corpus - corpus_columns:
        raise ValueError(f"anonymized corpus lacks identity/hash columns: {sorted(missing_corpus)}")

    raw_columns = raw.schema_arrow.names
    comparison = " OR ".join(
        f"r.{_sql_identifier(column)} IS DISTINCT FROM b.{_sql_identifier(column)}"
        for column in raw_columns
    )
    connection = duckdb.connect()
    try:
        corpus_paths = _sql_path_list(corpus_files)
        raw_sql = _sql_string(raw_path)
        bound_sql = _sql_string(bound_path)
        row = connection.execute(
            f"""
            WITH
              corpus AS (
                SELECT stable_uid, cleaned_text_sha256
                FROM read_parquet({corpus_paths})
              ),
              raw AS (SELECT * FROM read_parquet({raw_sql})),
              bound AS (SELECT * FROM read_parquet({bound_sql}))
            SELECT
              (SELECT count(*) FROM corpus) AS corpus_rows,
              (SELECT count(*) FROM raw) AS raw_rows,
              (SELECT count(*) FROM bound) AS bound_rows,
              (SELECT count(*) FROM (SELECT stable_uid FROM corpus GROUP BY stable_uid HAVING count(*) <> 1)) AS duplicate_corpus_uids,
              (SELECT count(*) FROM (SELECT stable_uid FROM bound GROUP BY stable_uid HAVING count(*) <> 1)) AS duplicate_bound_uids,
              (SELECT count(*) FROM corpus c LEFT JOIN bound b USING (stable_uid) WHERE b.stable_uid IS NULL) AS corpus_without_decision,
              (SELECT count(*) FROM bound b LEFT JOIN corpus c USING (stable_uid) WHERE c.stable_uid IS NULL) AS decision_outside_corpus,
              (SELECT count(*) FROM corpus c JOIN bound b USING (stable_uid) WHERE c.cleaned_text_sha256 IS DISTINCT FROM b.input_text_sha256) AS text_hash_drift,
              (SELECT count(*) FROM raw r LEFT JOIN bound b USING (doc_key) WHERE b.doc_key IS NULL) AS raw_without_bound,
              (SELECT count(*) FROM bound b LEFT JOIN raw r USING (doc_key) WHERE r.doc_key IS NULL) AS bound_without_raw,
              (SELECT count(*) FROM raw r JOIN bound b USING (doc_key) WHERE {comparison}) AS raw_bound_field_drift
            """
        ).fetchone()
    finally:
        connection.close()
    keys = (
        "corpus_rows",
        "raw_rows",
        "bound_rows",
        "duplicate_corpus_uids",
        "duplicate_bound_uids",
        "corpus_without_decision",
        "decision_outside_corpus",
        "text_hash_drift",
        "raw_without_bound",
        "bound_without_raw",
        "raw_bound_field_drift",
    )
    result = {key: int(value) for key, value in zip(keys, row, strict=True)}
    failures = (
        "duplicate_corpus_uids",
        "duplicate_bound_uids",
        "corpus_without_decision",
        "decision_outside_corpus",
        "text_hash_drift",
        "raw_without_bound",
        "bound_without_raw",
        "raw_bound_field_drift",
    )
    if result["corpus_rows"] < 1 or result["raw_rows"] != result["corpus_rows"] or result["bound_rows"] != result["corpus_rows"]:
        raise ValueError(f"post-mask detector inventory row count does not close: {result}")
    if any(result[name] for name in failures):
        raise ValueError(f"post-mask detector content-bound inventory closure failed: {result}")
    return result


def count_decisions(raw_path: Path) -> tuple[Counter[str], Counter[str]]:
    import pyarrow.parquet as pq

    raw = pq.ParquetFile(raw_path)
    decisions: Counter[str] = Counter()
    stages: Counter[str] = Counter()
    columns = ["decision", *(["decision_stage"] if "decision_stage" in raw.schema_arrow.names else [])]
    for batch in raw.iter_batches(columns=columns, batch_size=65_536, use_threads=False):
        for row in batch.to_pylist():
            decision = str(row.get("decision") or "")
            if decision not in {"keep", "drop"}:
                raise ValueError(f"post-mask detector emitted unsupported decision: {decision!r}")
            decisions[decision] += 1
            if decision == "drop":
                stages[str(row.get("decision_stage") or "unspecified")] += 1
    decisions["keep"] += 0
    decisions["drop"] += 0
    return decisions, stages


def write_no_replace(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_report(
    *, wrapper_manifest: Path, anonymization_manifest: Path, source_corpus: Path
) -> dict[str, Any]:
    manifest = read_object(wrapper_manifest)
    if manifest.get("schema_version") != WRAPPER_SCHEMA or manifest.get("status") != "completed":
        raise ValueError("a completed content-bound generic dedup wrapper manifest is required")
    _validate_production_detector_manifest(manifest)
    corpus = source_corpus.resolve()
    anonymization, corpus_files, corpus_receipts = validate_anonymized_inventory(
        anonymization_manifest=anonymization_manifest, source_corpus=corpus
    )
    if Path(str(manifest.get("input", ""))).resolve() != corpus:
        raise ValueError("verification dedup input differs from the exact anonymized corpus")
    output = manifest.get("dedup_output")
    if not isinstance(output, Mapping):
        raise ValueError("completed verification wrapper lacks dedup output")
    raw_path = validate_file_receipt(output.get("raw_decisions"), label="raw post-mask decisions")
    bound_path = validate_file_receipt(
        output.get("content_bound_decisions"),
        label="content-bound post-mask decisions",
        required_schema=CONTENT_BOUND_SCHEMA,
    )
    inventory = validate_decision_inventory(corpus_files=corpus_files, raw_path=raw_path, bound_path=bound_path)
    decisions, stages = count_decisions(raw_path)
    if sum(decisions.values()) != inventory["corpus_rows"]:
        raise ValueError("post-mask decision count differs from closed anonymized inventory")
    new_duplicates = int(decisions["drop"])
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "passed",
        "verification_only": True,
        "materialization_performed": False,
        "second_deduplication_applied": False,
        "source_corpus_root": str(corpus),
        "anonymization_manifest": binding(anonymization_manifest),
        "anonymized_corpus_inventory": {
            "manifest_output": str(Path(str(anonymization["output"])).resolve()),
            "shards": len(corpus_receipts),
            "rows": inventory["corpus_rows"],
            "receipt_inventory_sha256": hashlib.sha256(
                json.dumps(corpus_receipts, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        },
        "dedup_wrapper_manifest": binding(wrapper_manifest),
        "raw_decisions": binding(raw_path),
        "content_bound_decisions": binding(bound_path),
        "inventory_closure": inventory,
        "decision_counts": dict(sorted(decisions.items())),
        "new_duplicate_count": new_duplicates,
        "material_new_duplicate_count": new_duplicates,
        "duplicate_counts_by_detector_stage": dict(sorted(stages.items())),
        "requires_explicit_user_decision": new_duplicates > 0,
        "decision": "stop_for_user_postmask_dedup_decision" if new_duplicates else "no_new_duplicates_detected",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dedup-wrapper-manifest", type=Path, required=True)
    parser.add_argument("--anonymization-manifest", type=Path, required=True)
    parser.add_argument("--source-corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite post-mask duplicate report: {args.output}")
    payload = build_report(
        wrapper_manifest=args.dedup_wrapper_manifest.resolve(),
        anonymization_manifest=args.anonymization_manifest.resolve(),
        source_corpus=args.source_corpus.resolve(),
    )
    write_no_replace(args.output.resolve(), payload)
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output.resolve()),
                "material_new_duplicate_count": payload["material_new_duplicate_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
