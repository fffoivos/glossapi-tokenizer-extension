#!/usr/bin/env python3
"""Build the compact, receipt-bound Agent 1 v3 transformation waterfall.

The v3 lane deliberately changes representation order from the legacy corpus
pipeline.  This tool is its one compact accounting boundary:

``admitted pool -> dedup -> GreekMMLU -> direct-ID anonymization ->
prestructural freeze``.

It streams immutable Parquet inventories into a SQLite work database containing
only stable identifiers, hashes, source labels, and aggregate character/token
counts.  Text and protected span values are never written to the database or
to the JSON result.  The result is suitable for the later release/site handoff
and is deliberately *not* a corpus materializer.

The command fails closed on any receipt, identity, content-hash, partition, or
mass-accounting mismatch.  In particular, the protected anonymization ledger
is read without selecting its raw ``protected_spans_json`` column.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import stat
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


WATERFALL_SCHEMA = "agent1_full_corpus_v3_token_waterfall_v1"
ANONYMIZATION_AUDIT_SCHEMA = "agent1_full_corpus_v3_anonymization_audit_v1"
POOL_MANIFEST_SCHEMA = "agent1_full_corpus_v3_admitted_pool_manifest_v1"
DEDUP_LEDGER_MANIFEST_SCHEMA = "agent1_full_corpus_v3_dedup_ledger_manifest_v1"
DEDUP_MATERIALIZATION_MANIFEST_SCHEMA = "agent1_full_corpus_v3_dedup_materialization_manifest_v1"
ORDERED_DEDUP_SCHEMA = "agent1_full_corpus_v3_ordered_dedup_composition_v1"
DECONTAMINATION_MANIFEST_SCHEMA = "agent1_full_corpus_v3_decontamination_manifest_v1"
ANONYMIZATION_MANIFEST_SCHEMA = "agent1_full_corpus_v3_anonymization_manifest_v1"
LEDGER_CLOSURE_SCHEMA = "agent1_full_corpus_v3_protected_anonymization_ledger_closure_v1"
POSTMASK_REPORT_SCHEMA = "agent1_full_corpus_v3_postmask_duplicate_verification_v1"
PRESTRUCTURAL_SCHEMA = "agent1_full_corpus_v3_prestructural_manifest_v1"
ROSTER_SCHEMA = "agent1_full_corpus_v3_candidate_roster_v1"
ROUTE_POLICY_PRIORITY = "logical_source_then_observed_extraction"

ALLOWED_ROUTES = frozenset({"html_web", "pdf_ocr", "mixed", "structured"})
ALLOWED_MASK_TYPES = frozenset(
    {"email", "phone", "afm", "amka", "iban", "identity_or_passport", "ip"}
)
# Existing frozen reason codes include the source metadata spelling
# ``privateData``.  Permit ASCII case while still rejecting free-form text.
SAFE_REASON = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_:-]{0,127}$")
# ``acquisition_source_id`` is an internal roster key.  ``source_dataset`` is
# intentionally broader: the frozen Nanochat/candidate inventories preserve
# namespace-qualified repository identities such as ``HuggingFaceFW/finewiki``.
# Do not mistake a slash in that provenance identifier for unsafe corpus text.
SAFE_ACQUISITION_SOURCE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
SAFE_SOURCE_DATASET = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,255}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")

MASK_REASON = "approved_high_precision_direct_identifier_masking"
DROP_REASON = "diavgeia_privateData_true"
QUARANTINE_REASON = "diavgeia_pii_heavy_personnel_table"
FINAL_DEDUP_METHOD = "exact_then_within_source_then_cross_candidate_then_candidate_to_nanochat_precedence_v1"
IDENTITY_SELECTION_ORDER = (
    "exact_identity_then_within_source_then_cross_candidate_then_"
    "candidate_to_nanochat_before_representative_precedence"
)
ORDERED_DEDUP_PASSES = (
    "exact_content_work_representation",
    "within_source_near",
    "cross_candidate_near",
    "candidate_to_nanochat_near",
)


class WaterfallError(ValueError):
    """A receipt-bound accounting failure which never prints corpus text."""


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def require_regular_file(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_file() or resolved.stat().st_size < 1:
        raise FileNotFoundError(f"{label} must be a non-empty regular file: {path}")
    return resolved


def require_directory(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_dir():
        raise FileNotFoundError(f"{label} must be a non-symlink directory: {path}")
    return resolved


def read_object(path: Path, *, label: str) -> dict[str, Any]:
    resolved = require_regular_file(path, label=label)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WaterfallError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise WaterfallError(f"{label} must be a JSON object")
    return value


def binding(path: Path) -> dict[str, Any]:
    resolved = require_regular_file(path, label="bound artifact")
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def require_binding(value: object, expected_path: Path, *, label: str) -> None:
    if not isinstance(value, Mapping):
        raise WaterfallError(f"{label} must be a file binding")
    expected = binding(expected_path)
    if (
        value.get("path") != expected["path"]
        or value.get("bytes") != expected["bytes"]
        or value.get("sha256") != expected["sha256"]
    ):
        raise WaterfallError(f"{label} differs from its immutable input")


def safe_relative_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise WaterfallError(f"{label} must contain a non-empty relative path")
    result = Path(value)
    if result.is_absolute() or ".." in result.parts or result.as_posix() != value:
        raise WaterfallError(f"{label} contains an unsafe relative path")
    if result.suffix != ".parquet":
        raise WaterfallError(f"{label} must name a Parquet shard")
    return result


def parquet_receipt(path: Path, *, relative_to: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    metadata = pq.ParquetFile(path).metadata
    return {
        "path": path.resolve().relative_to(relative_to.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": int(metadata.num_rows),
        "row_groups": int(metadata.num_row_groups),
    }


def verify_parquet_receipt(path: Path, receipt: Mapping[str, Any], *, label: str) -> None:
    import pyarrow.parquet as pq

    expected_keys = {"path", "bytes", "sha256", "rows", "row_groups"}
    if set(receipt) != expected_keys:
        raise WaterfallError(f"{label} receipt keys drifted")
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} shard is missing or unsafe")
    if not isinstance(receipt.get("bytes"), int) or int(receipt["bytes"]) < 0:
        raise WaterfallError(f"{label} has an invalid byte receipt")
    if not isinstance(receipt.get("sha256"), str) or not SHA256.fullmatch(str(receipt["sha256"])):
        raise WaterfallError(f"{label} has an invalid SHA-256 receipt")
    if path.stat().st_size != int(receipt["bytes"]) or sha256_file(path) != receipt["sha256"]:
        raise WaterfallError(f"{label} bytes/SHA-256 receipt drift")
    metadata = pq.ParquetFile(path).metadata
    if int(metadata.num_rows) != receipt.get("rows") or int(metadata.num_row_groups) != receipt.get("row_groups"):
        raise WaterfallError(f"{label} Parquet metadata receipt drift")


def parquet_tree(root: Path, *, label: str) -> list[Path]:
    """Return an exact safe Parquet inventory and reject hidden side inputs."""

    resolved = require_directory(root, label=label)
    files: list[Path] = []
    for current, directories, names in os.walk(resolved, followlinks=False):
        current_path = Path(current)
        for directory in directories:
            if (current_path / directory).is_symlink():
                raise WaterfallError(f"{label} contains a symlink")
        for name in names:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise WaterfallError(f"{label} contains an unsafe file")
            if name.startswith("."):
                raise WaterfallError(f"{label} contains a hidden/partial file")
            if path.suffix != ".parquet":
                raise WaterfallError(f"{label} contains non-Parquet content")
            files.append(path.resolve())
    files.sort()
    if not files:
        raise WaterfallError(f"{label} has no Parquet shards")
    return files


def verify_manifest_tree(
    manifest: Mapping[str, Any],
    *,
    root: Path,
    receipt_key: str | None,
    label: str,
) -> list[Path]:
    """Verify that a manifest's nested receipts cover exactly one Parquet root."""

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise WaterfallError(f"{label} manifest lacks a non-empty file inventory")
    resolved_root = require_directory(root, label=f"{label} root")
    expected: set[Path] = set()
    for index, row in enumerate(files):
        receipt: object = row if receipt_key is None else row.get(receipt_key) if isinstance(row, Mapping) else None
        if not isinstance(receipt, Mapping):
            raise WaterfallError(f"{label} receipt {index} is absent")
        relative = safe_relative_path(receipt.get("path"), label=f"{label} receipt {index}")
        candidate = (resolved_root / relative).resolve()
        try:
            candidate.relative_to(resolved_root)
        except ValueError as exc:
            raise WaterfallError(f"{label} receipt escapes its root") from exc
        if candidate in expected:
            raise WaterfallError(f"{label} contains duplicate receipt paths")
        verify_parquet_receipt(candidate, receipt, label=f"{label} receipt {index}")
        expected.add(candidate)
    actual = set(parquet_tree(resolved_root, label=label))
    if actual != expected:
        raise WaterfallError(
            f"{label} receipt inventory differs from the on-disk Parquet inventory: "
            f"expected={len(expected)} actual={len(actual)}"
        )
    return sorted(expected)


def verify_absolute_parquet_receipt(receipt: object, path: Path, *, label: str) -> None:
    if not isinstance(receipt, Mapping):
        raise WaterfallError(f"{label} lacks a Parquet receipt")
    resolved = require_regular_file(path, label=label)
    declared = receipt.get("path")
    if not isinstance(declared, str) or Path(declared).resolve() != resolved:
        raise WaterfallError(f"{label} path differs from its manifest receipt")
    verify_parquet_receipt(resolved, receipt, label=label)


def require_manifest(manifest: Mapping[str, Any], *, schema: str, status: str, label: str) -> None:
    if manifest.get("schema_version") != schema or manifest.get("status") != status:
        raise WaterfallError(f"{label} is not a completed {schema} receipt")


def source_values(
    row: Mapping[str, Any],
    *,
    label: str,
    expected_routes: Mapping[str, tuple[str, str, str]] | None = None,
) -> tuple[str, str, str, str, str]:
    acquisition = row.get("acquisition_source_id")
    dataset = row.get("source_dataset")
    if not isinstance(acquisition, str) or not acquisition or not isinstance(dataset, str) or not dataset:
        raise WaterfallError(f"{label} lacks required source identity")
    if not SAFE_ACQUISITION_SOURCE.fullmatch(acquisition):
        raise WaterfallError(f"{label} acquisition_source_id is not a compact roster label")
    if not SAFE_SOURCE_DATASET.fullmatch(dataset):
        raise WaterfallError(f"{label} source_dataset is not a safe frozen dataset identifier")
    routes: list[str] = []
    for field in ("source_route", "review_route", "extraction_route"):
        value = row.get(field)
        if value is None:
            routes.append("")
        elif isinstance(value, str) and value:
            routes.append(value)
        else:
            raise WaterfallError(f"{label} has malformed {field}")
    if any(routes) and not all(routes):
        raise WaterfallError(f"{label} has incomplete logical/extraction route provenance")
    if routes and all(routes) and any(route not in ALLOWED_ROUTES for route in routes):
        raise WaterfallError(f"{label} has an unsupported logical/extraction route")
    if expected_routes is not None:
        expected = expected_routes.get(acquisition)
        if expected is None:
            raise WaterfallError(f"{label} acquisition_source_id is absent from the frozen candidate roster")
        if tuple(routes) != expected:
            raise WaterfallError(
                f"{label} logical/extraction route differs from the frozen candidate roster"
            )
    return acquisition, dataset, routes[0], routes[1], routes[2]


def _roster_route_map(
    roster: Mapping[str, Any],
    *,
    field: str,
    candidates: Sequence[str],
    fallback: Mapping[str, str] | None = None,
) -> dict[str, str]:
    value = roster.get(field)
    if value is None:
        if fallback is None:
            raise WaterfallError(f"frozen candidate roster lacks {field}")
        return dict(fallback)
    if not isinstance(value, Mapping) or set(value) != set(candidates):
        raise WaterfallError(f"frozen candidate roster {field} coverage drift")
    result: dict[str, str] = {}
    for source in candidates:
        route = value.get(source)
        if not isinstance(route, str) or route not in ALLOWED_ROUTES:
            raise WaterfallError(f"frozen candidate roster has unsupported {field} route")
        result[source] = route
    return result


def frozen_route_declarations(path: Path) -> dict[str, tuple[str, str, str]]:
    """Load the logical-first route contract used by every Stage-70 row.

    The Nanochat base is the sole documented null-route exception.  Candidate
    rows must carry all three frozen provenance values; this prevents a later
    Parquet transport or filename from silently changing the error model.
    """

    roster = read_object(path, label="frozen candidate roster")
    if roster.get("schema_version") != ROSTER_SCHEMA:
        raise WaterfallError("frozen candidate roster schema drift")
    if roster.get("base_source_id") != "nanochat_base":
        raise WaterfallError("frozen candidate roster base source drift")
    candidates = roster.get("candidate_source_ids")
    if (
        not isinstance(candidates, list)
        or not candidates
        or any(not isinstance(source, str) or not SAFE_ACQUISITION_SOURCE.fullmatch(source) for source in candidates)
        or len(candidates) != len(set(candidates))
    ):
        raise WaterfallError("frozen candidate roster has invalid candidate source IDs")
    review_routes = _roster_route_map(roster, field="review_routes", candidates=candidates)
    source_routes = _roster_route_map(
        roster, field="source_routes", candidates=candidates, fallback=review_routes
    )
    extraction_routes = _roster_route_map(
        roster, field="extraction_routes", candidates=candidates, fallback=review_routes
    )
    policy = roster.get("route_policy")
    if policy is not None and (
        not isinstance(policy, Mapping)
        or policy.get("priority") != ROUTE_POLICY_PRIORITY
    ):
        raise WaterfallError("frozen candidate roster logical-source route policy drift")
    declarations = {"nanochat_base": ("", "", "")}
    declarations.update(
        {
            source: (source_routes[source], review_routes[source], extraction_routes[source])
            for source in candidates
        }
    )
    return declarations


def nonempty_string(row: Mapping[str, Any], field: str, *, label: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise WaterfallError(f"{label} lacks a non-empty {field}")
    return value


def text_value(row: Mapping[str, Any], *, label: str) -> str:
    """Return a canonical text field while allowing a legitimate empty row."""

    value = row.get("text")
    if not isinstance(value, str):
        raise WaterfallError(f"{label} lacks a string text field")
    return value


def valid_hash(value: object, *, label: str, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise WaterfallError(f"{label} must be a lowercase SHA-256")
    return value


def safe_reason(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not SAFE_REASON.fullmatch(value):
        raise WaterfallError(f"{label} is not a compact reason code")
    return value


def parse_reason_list(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise WaterfallError(f"{label} must be a JSON reason list")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise WaterfallError(f"{label} is not valid JSON") from exc
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise WaterfallError(f"{label} must be a string-only JSON list")
    result = tuple(sorted(set(safe_reason(item, label=label) for item in parsed)))
    if list(parsed) != sorted(set(parsed)):
        raise WaterfallError(f"{label} must be sorted and unique")
    return result


def parse_pii_counts(value: object, *, label: str) -> dict[str, int]:
    if not isinstance(value, str):
        raise WaterfallError(f"{label} must be a JSON object")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise WaterfallError(f"{label} is not valid JSON") from exc
    if not isinstance(parsed, Mapping):
        raise WaterfallError(f"{label} must be a JSON object")
    result: dict[str, int] = {}
    for key, amount in parsed.items():
        if not isinstance(key, str) or key not in ALLOWED_MASK_TYPES or not isinstance(amount, int) or amount < 1:
            raise WaterfallError(f"{label} contains an unsupported PII count")
        result[key] = amount
    return result


class ExactTokenizer:
    """Pinned exact-tokenizer counter with no truncation or padding."""

    def __init__(self, path: Path) -> None:
        self.path = require_regular_file(path, label="tokenizer JSON")
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - exercised on Clariden
            raise RuntimeError("transformation waterfall requires tokenizers") from exc
        self._tokenizer = Tokenizer.from_file(str(self.path))
        self._tokenizer.no_padding()
        self._tokenizer.no_truncation()

    def counts(self, texts: Sequence[str], *, batch_docs: int) -> list[int]:
        result: list[int] = []
        for start in range(0, len(texts), batch_docs):
            chunk = list(texts[start : start + batch_docs])
            result.extend(len(encoded.ids) for encoded in self._tokenizer.encode_batch(chunk, add_special_tokens=False))
        return result


def parquet_batches(
    files: Iterable[Path], *, columns: Sequence[str], batch_rows: int, label: str
) -> Iterable[list[dict[str, Any]]]:
    import pyarrow.parquet as pq

    for path in files:
        parquet = pq.ParquetFile(path)
        missing = set(columns) - set(parquet.schema_arrow.names)
        if missing:
            raise WaterfallError(f"{label} shard lacks required columns {sorted(missing)}")
        for batch in parquet.iter_batches(columns=list(columns), batch_size=batch_rows, use_threads=False):
            yield batch.to_pylist()


def connect_database(path: Path) -> sqlite3.Connection:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to reuse mutable waterfall work database: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.executescript(
        """
        CREATE TABLE pool (
          stable_uid TEXT PRIMARY KEY,
          input_representation_id TEXT NOT NULL,
          acquisition_source_id TEXT NOT NULL,
          source_dataset TEXT NOT NULL,
          source_route TEXT NOT NULL,
          review_route TEXT NOT NULL,
          extraction_route TEXT NOT NULL,
          text_sha256 TEXT NOT NULL,
          characters INTEGER NOT NULL,
          tokens INTEGER NOT NULL
        );
        CREATE TABLE dedup (
          stable_uid TEXT PRIMARY KEY,
          input_representation_id TEXT NOT NULL,
          input_text_sha256 TEXT NOT NULL,
          action TEXT NOT NULL,
          representative_stable_uid TEXT NOT NULL,
          representative_input_representation_id TEXT NOT NULL,
          cluster_id TEXT NOT NULL,
          reason TEXT NOT NULL
        );
        CREATE TABLE materialized (
          stable_uid TEXT PRIMARY KEY,
          input_representation_id TEXT NOT NULL,
          text_sha256 TEXT NOT NULL,
          acquisition_source_id TEXT NOT NULL,
          source_dataset TEXT NOT NULL,
          source_route TEXT NOT NULL,
          review_route TEXT NOT NULL,
          extraction_route TEXT NOT NULL
        );
        CREATE TABLE decontam (
          stable_uid TEXT PRIMARY KEY,
          representation_id TEXT NOT NULL,
          input_text_sha256 TEXT NOT NULL,
          action TEXT NOT NULL,
          reason TEXT NOT NULL
        );
        CREATE TABLE decontam_observed (
          stable_uid TEXT PRIMARY KEY,
          action TEXT NOT NULL,
          representation_id TEXT NOT NULL,
          text_sha256 TEXT NOT NULL,
          acquisition_source_id TEXT NOT NULL,
          source_dataset TEXT NOT NULL,
          source_route TEXT NOT NULL,
          review_route TEXT NOT NULL,
          extraction_route TEXT NOT NULL
        );
        CREATE TABLE anonym (
          stable_uid TEXT PRIMARY KEY,
          acquisition_source_id TEXT NOT NULL,
          source_dataset TEXT NOT NULL,
          input_text_sha256 TEXT NOT NULL,
          output_text_sha256 TEXT,
          parent_representation_id TEXT NOT NULL,
          child_representation_id TEXT,
          action TEXT NOT NULL,
          reason_key TEXT NOT NULL,
          span_count INTEGER NOT NULL,
          pii_count INTEGER NOT NULL
        );
        CREATE TABLE anonym_observed (
          stable_uid TEXT PRIMARY KEY,
          action TEXT NOT NULL,
          parent_text_sha256 TEXT NOT NULL,
          output_text_sha256 TEXT,
          parent_representation_id TEXT NOT NULL,
          child_representation_id TEXT,
          characters INTEGER NOT NULL,
          tokens INTEGER NOT NULL,
          acquisition_source_id TEXT NOT NULL,
          source_dataset TEXT NOT NULL,
          source_route TEXT NOT NULL,
          review_route TEXT NOT NULL,
          extraction_route TEXT NOT NULL
        );
        CREATE TABLE pii_by_uid (
          stable_uid TEXT NOT NULL,
          pii_type TEXT NOT NULL,
          spans INTEGER NOT NULL,
          PRIMARY KEY (stable_uid, pii_type)
        );
        CREATE TABLE pii_types (
          acquisition_source_id TEXT NOT NULL,
          source_dataset TEXT NOT NULL,
          source_route TEXT NOT NULL,
          review_route TEXT NOT NULL,
          extraction_route TEXT NOT NULL,
          pii_type TEXT NOT NULL,
          spans INTEGER NOT NULL,
          PRIMARY KEY (acquisition_source_id, source_dataset, source_route, review_route, extraction_route, pii_type)
        );
        """
    )
    return connection


def query_scalar(connection: sqlite3.Connection, query: str, parameters: Sequence[object] = ()) -> int:
    value = connection.execute(query, tuple(parameters)).fetchone()[0]
    return int(value or 0)


def expect_zero(connection: sqlite3.Connection, query: str, *, label: str) -> None:
    observed = query_scalar(connection, query)
    if observed:
        raise WaterfallError(f"{label}: {observed}")


def load_pool(
    connection: sqlite3.Connection,
    *,
    files: Iterable[Path],
    tokenizer: ExactTokenizer,
    batch_rows: int,
    tokenizer_batch_docs: int,
    expected_routes: Mapping[str, tuple[str, str, str]],
) -> int:
    columns = [
        "stable_uid", "input_representation_id", "text", "cleaned_text_sha256", "acquisition_source_id", "source_dataset",
        "source_route", "review_route", "extraction_route",
    ]
    rows = 0
    for batch in parquet_batches(files, columns=columns, batch_rows=batch_rows, label="dedup pool"):
        texts = [text_value(row, label="dedup pool row") for row in batch]
        token_counts = tokenizer.counts(texts, batch_docs=tokenizer_batch_docs)
        records = []
        for row, text, tokens in zip(batch, texts, token_counts, strict=True):
            uid = nonempty_string(row, "stable_uid", label="dedup pool row")
            input_representation_id = nonempty_string(
                row, "input_representation_id", label="dedup pool row"
            )
            actual_hash = sha256_text(text)
            if valid_hash(row.get("cleaned_text_sha256"), label="dedup pool cleaned_text_sha256") != actual_hash:
                raise WaterfallError("dedup pool cleaned_text_sha256 drift")
            source = source_values(
                row,
                label="dedup pool row",
                expected_routes=expected_routes,
            )
            records.append((uid, input_representation_id, *source, actual_hash, len(text), int(tokens)))
        try:
            connection.executemany("INSERT INTO pool VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", records)
        except sqlite3.IntegrityError as exc:
            raise WaterfallError("dedup pool has duplicate stable_uid") from exc
        rows += len(records)
    connection.commit()
    if rows < 1:
        raise WaterfallError("dedup pool is empty")
    return rows


def load_dedup_ledger(connection: sqlite3.Connection, *, path: Path, batch_rows: int) -> int:
    required = [
        "stable_uid", "input_representation_id", "input_text_sha256", "action", "representative_stable_uid",
        "representative_input_representation_id", "cluster_id", "method", "raw_decision_stage", "reason",
    ]
    rows = 0
    methods: set[str] = set()
    for batch in parquet_batches([path], columns=required, batch_rows=batch_rows, label="dedup ledger"):
        records = []
        for row in batch:
            uid = nonempty_string(row, "stable_uid", label="dedup ledger row")
            action = nonempty_string(row, "action", label="dedup ledger row")
            if action not in {"keep", "drop"}:
                raise WaterfallError("dedup ledger contains an unsupported action")
            method = nonempty_string(row, "method", label="dedup ledger row")
            methods.add(method)
            records.append((
                uid,
                nonempty_string(row, "input_representation_id", label="dedup ledger row"),
                valid_hash(row.get("input_text_sha256"), label="dedup ledger input_text_sha256"),
                action,
                nonempty_string(row, "representative_stable_uid", label="dedup ledger row"),
                nonempty_string(
                    row,
                    "representative_input_representation_id",
                    label="dedup ledger row",
                ),
                nonempty_string(row, "cluster_id", label="dedup ledger row"),
                safe_reason(row.get("reason"), label="dedup ledger reason"),
            ))
        try:
            connection.executemany("INSERT INTO dedup VALUES (?, ?, ?, ?, ?, ?, ?, ?)", records)
        except sqlite3.IntegrityError as exc:
            raise WaterfallError("dedup ledger has duplicate stable_uid") from exc
        rows += len(records)
    if methods != {FINAL_DEDUP_METHOD}:
        raise WaterfallError("dedup ledger is not the ordered identity-first v3 representative ledger")
    if rows < 1:
        raise WaterfallError("dedup ledger is empty")
    connection.commit()
    expect_zero(connection, "SELECT count(*) FROM pool p LEFT JOIN dedup d USING(stable_uid) WHERE d.stable_uid IS NULL", label="pool rows missing dedup decisions")
    expect_zero(connection, "SELECT count(*) FROM dedup d LEFT JOIN pool p USING(stable_uid) WHERE p.stable_uid IS NULL", label="dedup decisions outside pool")
    expect_zero(
        connection,
        "SELECT count(*) FROM pool p JOIN dedup d USING(stable_uid) WHERE "
        "p.text_sha256 != d.input_text_sha256 OR p.input_representation_id != d.input_representation_id",
        label="dedup content/representation drift",
    )
    expect_zero(
        connection,
        "SELECT count(*) FROM dedup d LEFT JOIN pool p ON p.stable_uid = d.representative_stable_uid "
        "WHERE p.stable_uid IS NULL OR p.input_representation_id != d.representative_input_representation_id",
        label="dedup representative representation drift",
    )
    expect_zero(
        connection,
        "SELECT count(*) FROM ("
        "SELECT cluster_id FROM dedup GROUP BY cluster_id HAVING "
        "COUNT(DISTINCT representative_stable_uid) != 1 OR "
        "COUNT(DISTINCT representative_input_representation_id) != 1 OR "
        "SUM(CASE WHEN action = 'keep' THEN 1 ELSE 0 END) != 1 OR "
        "SUM(CASE WHEN action = 'keep' AND stable_uid = representative_stable_uid THEN 1 ELSE 0 END) != 1"
        ")",
        label="dedup cluster representative closure drift",
    )
    return rows


def scan_materialized(connection: sqlite3.Connection, *, files: Iterable[Path], batch_rows: int) -> int:
    columns = ["stable_uid", "input_representation_id", "text", "cleaned_text_sha256", "acquisition_source_id", "source_dataset", "source_route", "review_route", "extraction_route"]
    rows = 0
    for batch in parquet_batches(files, columns=columns, batch_rows=batch_rows, label="dedup materialization"):
        records = []
        for row in batch:
            uid = nonempty_string(row, "stable_uid", label="dedup materialization row")
            input_representation_id = nonempty_string(
                row, "input_representation_id", label="dedup materialization row"
            )
            text = text_value(row, label="dedup materialization row")
            actual_hash = sha256_text(text)
            if valid_hash(row.get("cleaned_text_sha256"), label="dedup materialization cleaned_text_sha256") != actual_hash:
                raise WaterfallError("dedup materialization cleaned_text_sha256 drift")
            source_values(row, label="dedup materialization row")
            records.append(
                (uid, input_representation_id, actual_hash, *source_values(row, label="dedup materialization row"))
            )
        try:
            connection.executemany("INSERT INTO materialized VALUES (?, ?, ?, ?, ?, ?, ?, ?)", records)
        except sqlite3.IntegrityError as exc:
            raise WaterfallError("dedup materialization has duplicate stable_uid") from exc
        rows += len(records)
    connection.commit()
    expect_zero(connection, "SELECT count(*) FROM materialized m LEFT JOIN dedup d USING(stable_uid) WHERE d.action IS NULL OR d.action != 'keep'", label="materialized row without a keep decision")
    expect_zero(connection, "SELECT count(*) FROM dedup d LEFT JOIN materialized m USING(stable_uid) WHERE d.action = 'keep' AND m.stable_uid IS NULL", label="dedup keep row missing from materialization")
    expect_zero(
        connection,
        "SELECT count(*) FROM materialized m JOIN dedup d USING(stable_uid) WHERE "
        "m.text_sha256 != d.input_text_sha256 OR m.input_representation_id != d.input_representation_id",
        label="materialized content/representation hash drift",
    )
    expect_zero(
        connection,
        "SELECT count(*) FROM materialized m JOIN pool p USING(stable_uid) WHERE "
        "m.input_representation_id != p.input_representation_id OR "
        "m.acquisition_source_id != p.acquisition_source_id OR m.source_dataset != p.source_dataset OR "
        "m.source_route != p.source_route OR m.review_route != p.review_route OR m.extraction_route != p.extraction_route",
        label="materialized source provenance drift",
    )
    return rows


def load_decontam_ledger(connection: sqlite3.Connection, *, files: Iterable[Path], batch_rows: int) -> int:
    columns = ["stable_uid", "representation_id", "input_text_sha256", "action", "reason"]
    rows = 0
    for batch in parquet_batches(files, columns=columns, batch_rows=batch_rows, label="GreekMMLU ledger"):
        records = []
        for row in batch:
            uid = nonempty_string(row, "stable_uid", label="GreekMMLU ledger row")
            action = nonempty_string(row, "action", label="GreekMMLU ledger row")
            if action not in {"keep", "drop", "quarantine"}:
                raise WaterfallError("GreekMMLU ledger contains an unsupported action")
            records.append((
                uid,
                nonempty_string(row, "representation_id", label="GreekMMLU ledger row"),
                valid_hash(row.get("input_text_sha256"), label="GreekMMLU input_text_sha256"),
                action,
                safe_reason(row.get("reason"), label="GreekMMLU reason"),
            ))
        try:
            connection.executemany("INSERT INTO decontam VALUES (?, ?, ?, ?, ?)", records)
        except sqlite3.IntegrityError as exc:
            raise WaterfallError("GreekMMLU ledger has duplicate stable_uid") from exc
        rows += len(records)
    connection.commit()
    expect_zero(connection, "SELECT count(*) FROM dedup d LEFT JOIN decontam g USING(stable_uid) WHERE d.action = 'keep' AND g.stable_uid IS NULL", label="dedup survivors missing GreekMMLU decisions")
    expect_zero(connection, "SELECT count(*) FROM decontam g LEFT JOIN dedup d USING(stable_uid) WHERE d.action IS NULL OR d.action != 'keep'", label="GreekMMLU decisions outside dedup survivors")
    expect_zero(
        connection,
        "SELECT count(*) FROM decontam g JOIN dedup d USING(stable_uid) WHERE "
        "g.input_text_sha256 != d.input_text_sha256 OR "
        "g.representation_id != d.input_representation_id",
        label="GreekMMLU content/representation drift",
    )
    return rows


def scan_decontam_partition(
    connection: sqlite3.Connection, *, files: Iterable[Path], action: str, batch_rows: int
) -> int:
    columns = ["stable_uid", "representation_id", "text", "cleaned_text_sha256", "acquisition_source_id", "source_dataset", "source_route", "review_route", "extraction_route"]
    rows = 0
    for batch in parquet_batches(files, columns=columns, batch_rows=batch_rows, label=f"GreekMMLU {action} partition"):
        records = []
        for row in batch:
            uid = nonempty_string(row, "stable_uid", label=f"GreekMMLU {action} row")
            representation_id = nonempty_string(
                row, "representation_id", label=f"GreekMMLU {action} row"
            )
            text = text_value(row, label=f"GreekMMLU {action} row")
            actual_hash = sha256_text(text)
            if valid_hash(row.get("cleaned_text_sha256"), label="GreekMMLU partition cleaned_text_sha256") != actual_hash:
                raise WaterfallError("GreekMMLU partition cleaned_text_sha256 drift")
            source = source_values(row, label=f"GreekMMLU {action} row")
            records.append((uid, action, representation_id, actual_hash, *source))
        try:
            connection.executemany("INSERT INTO decontam_observed VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", records)
        except sqlite3.IntegrityError as exc:
            raise WaterfallError("GreekMMLU partitions overlap on stable_uid") from exc
        rows += len(records)
    connection.commit()
    return rows


def close_decontam_partitions(connection: sqlite3.Connection) -> None:
    expect_zero(connection, "SELECT count(*) FROM decontam g LEFT JOIN decontam_observed o USING(stable_uid) WHERE o.stable_uid IS NULL", label="GreekMMLU ledger rows missing partition output")
    expect_zero(connection, "SELECT count(*) FROM decontam_observed o LEFT JOIN decontam g USING(stable_uid) WHERE g.stable_uid IS NULL", label="GreekMMLU partition rows outside ledger")
    expect_zero(connection, "SELECT count(*) FROM decontam g JOIN decontam_observed o USING(stable_uid) WHERE g.action != o.action", label="GreekMMLU partition action drift")
    expect_zero(
        connection,
        "SELECT count(*) FROM decontam g JOIN decontam_observed o USING(stable_uid) WHERE "
        "g.input_text_sha256 != o.text_sha256 OR g.representation_id != o.representation_id",
        label="GreekMMLU partition content/representation drift",
    )
    expect_zero(
        connection,
        "SELECT count(*) FROM decontam_observed o JOIN pool p USING(stable_uid) WHERE "
        "o.acquisition_source_id != p.acquisition_source_id OR o.source_dataset != p.source_dataset OR "
        "o.source_route != p.source_route OR o.review_route != p.review_route OR o.extraction_route != p.extraction_route",
        label="GreekMMLU partition source provenance drift",
    )


def load_anonym_ledger(
    connection: sqlite3.Connection, *, files: Iterable[Path], batch_rows: int
) -> tuple[int, Counter[str]]:
    columns = [
        "stable_uid", "acquisition_source_id", "source_dataset", "input_text_sha256", "output_text_sha256",
        "parent_representation_id", "child_representation_id", "action", "reasons_json",
        "pii_by_type_json", "span_count", "ledger_schema_version",
    ]
    rows = 0
    totals: Counter[str] = Counter()
    for batch in parquet_batches(files, columns=columns, batch_rows=batch_rows, label="protected anonymization ledger"):
        records = []
        type_records: list[tuple[str, str, int]] = []
        for row in batch:
            uid = nonempty_string(row, "stable_uid", label="protected anonymization ledger row")
            action = nonempty_string(row, "action", label="protected anonymization ledger row")
            if action not in {"keep", "drop", "quarantine"}:
                raise WaterfallError("protected anonymization ledger contains an unsupported action")
            source = (nonempty_string(row, "acquisition_source_id", label="protected anonymization ledger row"), nonempty_string(row, "source_dataset", label="protected anonymization ledger row"))
            input_hash = valid_hash(row.get("input_text_sha256"), label="protected anonymization input_text_sha256")
            output_hash = valid_hash(row.get("output_text_sha256"), label="protected anonymization output_text_sha256", nullable=True)
            parent_representation_id = nonempty_string(
                row, "parent_representation_id", label="protected anonymization ledger row"
            )
            child_value = row.get("child_representation_id")
            child_representation_id = (
                None
                if child_value is None
                else nonempty_string(row, "child_representation_id", label="protected anonymization ledger row")
            )
            reasons = parse_reason_list(row.get("reasons_json"), label="protected anonymization reasons")
            pii = parse_pii_counts(row.get("pii_by_type_json"), label="protected anonymization pii counts")
            span_count = row.get("span_count")
            if not isinstance(span_count, int) or span_count < 0 or sum(pii.values()) != span_count:
                raise WaterfallError("protected anonymization span counts do not close")
            if row.get("ledger_schema_version") != "agent1_full_corpus_v3_protected_anonymization_ledger_v1":
                raise WaterfallError("protected anonymization ledger schema drift")
            if action == "drop":
                if (
                    output_hash is not None
                    or child_representation_id is not None
                    or span_count != 0
                    or reasons != (DROP_REASON,)
                    or source[0] != "diavgeia"
                ):
                    raise WaterfallError("anonymization drop violates the direct-ID false-positive policy")
            else:
                if output_hash is None or child_representation_id is None:
                    raise WaterfallError("emitted anonymization row lacks output lineage")
                if span_count and MASK_REASON not in reasons:
                    raise WaterfallError("masked anonymization row lacks its approved reason")
                if not span_count and MASK_REASON in reasons:
                    raise WaterfallError("anonymization mask reason lacks an approved span")
                if action == "keep" and any(reason != MASK_REASON for reason in reasons):
                    raise WaterfallError("anonymization keep has an unsupported reason")
                if action == "quarantine":
                    if QUARANTINE_REASON not in reasons or source[0] != "diavgeia" or span_count < 3:
                        raise WaterfallError("anonymization quarantine violates the precision-gated policy")
                    if any(reason not in {QUARANTINE_REASON, MASK_REASON} for reason in reasons):
                        raise WaterfallError("anonymization quarantine has an unsupported reason")
            reason_key = "|".join(reasons) if reasons else "no_transform"
            records.append((
                uid,
                source[0],
                source[1],
                input_hash,
                output_hash,
                parent_representation_id,
                child_representation_id,
                action,
                reason_key,
                span_count,
                sum(pii.values()),
            ))
            totals[f"action:{action}"] += 1
            totals["spans"] += span_count
            type_records.extend((uid, pii_type, int(amount)) for pii_type, amount in pii.items())
        try:
            connection.executemany("INSERT INTO anonym VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", records)
        except sqlite3.IntegrityError as exc:
            raise WaterfallError("protected anonymization ledger has duplicate stable_uid") from exc
        try:
            connection.executemany("INSERT INTO pii_by_uid VALUES (?, ?, ?)", type_records)
        except sqlite3.IntegrityError as exc:
            raise WaterfallError("protected anonymization ledger repeats a PII type for one stable_uid") from exc
        rows += len(records)
    connection.commit()
    expect_zero(connection, "SELECT count(*) FROM decontam g LEFT JOIN anonym a USING(stable_uid) WHERE g.action = 'keep' AND a.stable_uid IS NULL", label="GreekMMLU keep rows missing anonymization decisions")
    expect_zero(connection, "SELECT count(*) FROM anonym a LEFT JOIN decontam g USING(stable_uid) WHERE g.action IS NULL OR g.action != 'keep'", label="anonymization decisions outside GreekMMLU survivors")
    expect_zero(connection, "SELECT count(*) FROM anonym a JOIN decontam g USING(stable_uid) WHERE a.input_text_sha256 != g.input_text_sha256", label="anonymization input content hash drift")
    expect_zero(connection, "SELECT count(*) FROM anonym a JOIN pool p USING(stable_uid) WHERE a.acquisition_source_id != p.acquisition_source_id OR a.source_dataset != p.source_dataset", label="anonymization source provenance drift")
    connection.execute(
        """
        INSERT INTO pii_types
        SELECT p.acquisition_source_id, p.source_dataset, p.source_route, p.review_route, p.extraction_route,
               u.pii_type, sum(u.spans)
        FROM pii_by_uid u JOIN pool p USING(stable_uid)
        GROUP BY p.acquisition_source_id, p.source_dataset, p.source_route, p.review_route, p.extraction_route, u.pii_type
        """
    )
    connection.commit()
    return rows, totals


def scan_anonym_emitted_partition(
    connection: sqlite3.Connection,
    *,
    files: Iterable[Path],
    action: str,
    tokenizer: ExactTokenizer,
    batch_rows: int,
    tokenizer_batch_docs: int,
) -> int:
    columns = [
        "stable_uid", "text", "cleaned_text_sha256", "anonymization_parent_text_sha256",
        "anonymization_output_text_sha256", "anonymization_action", "acquisition_source_id", "source_dataset",
        "source_route", "review_route", "extraction_route", "anonymization_parent_representation_id",
        "anonymization_child_representation_id",
    ]
    rows = 0
    for batch in parquet_batches(files, columns=columns, batch_rows=batch_rows, label=f"anonymization {action} partition"):
        texts = [text_value(row, label=f"anonymization {action} row") for row in batch]
        token_counts = tokenizer.counts(texts, batch_docs=tokenizer_batch_docs)
        records = []
        for row, text, tokens in zip(batch, texts, token_counts, strict=True):
            uid = nonempty_string(row, "stable_uid", label=f"anonymization {action} row")
            actual_hash = sha256_text(text)
            if valid_hash(row.get("cleaned_text_sha256"), label="anonymization output cleaned_text_sha256") != actual_hash:
                raise WaterfallError("anonymization output cleaned_text_sha256 drift")
            if valid_hash(row.get("anonymization_output_text_sha256"), label="anonymization output hash") != actual_hash:
                raise WaterfallError("anonymization output hash drift")
            parent_hash = valid_hash(
                row.get("anonymization_parent_text_sha256"), label="anonymization parent hash"
            )
            parent_representation_id = nonempty_string(
                row, "anonymization_parent_representation_id", label="anonymization output row"
            )
            child_representation_id = nonempty_string(
                row, "anonymization_child_representation_id", label="anonymization output row"
            )
            if nonempty_string(row, "anonymization_action", label="anonymization output row") != action:
                raise WaterfallError("anonymization output action differs from its partition")
            source = source_values(row, label=f"anonymization {action} row")
            records.append((
                uid,
                action,
                parent_hash,
                actual_hash,
                parent_representation_id,
                child_representation_id,
                len(text),
                int(tokens),
                *source,
            ))
        try:
            connection.executemany("INSERT INTO anonym_observed VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", records)
        except sqlite3.IntegrityError as exc:
            raise WaterfallError("anonymization emitted partitions overlap on stable_uid") from exc
        rows += len(records)
    connection.commit()
    return rows


def scan_anonym_dropped_partition(connection: sqlite3.Connection, *, files: Iterable[Path], batch_rows: int) -> int:
    import pyarrow.parquet as pq

    required = [
        "stable_uid",
        "anonymization_parent_text_sha256",
        "anonymization_parent_representation_id",
        "anonymization_action",
        "acquisition_source_id",
        "source_dataset",
    ]
    rows = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        if "text" in names:
            raise WaterfallError("private-data tombstone partition must not contain text")
        missing = set(required) - names
        if missing:
            raise WaterfallError(f"private-data tombstone lacks required columns {sorted(missing)}")
        for batch in parquet.iter_batches(columns=required, batch_size=batch_rows, use_threads=False):
            records = []
            for row in batch.to_pylist():
                uid = nonempty_string(row, "stable_uid", label="private-data tombstone row")
                if nonempty_string(row, "anonymization_action", label="private-data tombstone row") != "drop":
                    raise WaterfallError("private-data tombstone action drift")
                parent_hash = valid_hash(
                    row.get("anonymization_parent_text_sha256"),
                    label="private-data tombstone parent hash",
                )
                parent_representation_id = nonempty_string(
                    row,
                    "anonymization_parent_representation_id",
                    label="private-data tombstone row",
                )
                nonempty_string(row, "acquisition_source_id", label="private-data tombstone row")
                nonempty_string(row, "source_dataset", label="private-data tombstone row")
                # Tombstones intentionally omit the route fields.  They are
                # populated as empty sentinels and source-route closure is
                # checked only for emitted text representations below.
                records.append((
                    uid,
                    "drop",
                    parent_hash,
                    None,
                    parent_representation_id,
                    None,
                    0,
                    0,
                    str(row["acquisition_source_id"]),
                    str(row["source_dataset"]),
                    "",
                    "",
                    "",
                ))
            try:
                connection.executemany("INSERT INTO anonym_observed VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", records)
            except sqlite3.IntegrityError as exc:
                raise WaterfallError("anonymization partitions overlap on stable_uid") from exc
            rows += len(records)
    connection.commit()
    return rows


def close_anonym_partitions(connection: sqlite3.Connection) -> None:
    expect_zero(connection, "SELECT count(*) FROM anonym a LEFT JOIN anonym_observed o USING(stable_uid) WHERE o.stable_uid IS NULL", label="anonymization ledger rows missing partition output")
    expect_zero(connection, "SELECT count(*) FROM anonym_observed o LEFT JOIN anonym a USING(stable_uid) WHERE a.stable_uid IS NULL", label="anonymization partition rows outside ledger")
    expect_zero(connection, "SELECT count(*) FROM anonym a JOIN anonym_observed o USING(stable_uid) WHERE a.action != o.action", label="anonymization partition action drift")
    expect_zero(
        connection,
        "SELECT count(*) FROM anonym a JOIN anonym_observed o USING(stable_uid) "
        "WHERE a.input_text_sha256 != o.parent_text_sha256",
        label="anonymization parent content hash drift",
    )
    expect_zero(
        connection,
        "SELECT count(*) FROM anonym a JOIN anonym_observed o USING(stable_uid) WHERE "
        "a.parent_representation_id != o.parent_representation_id OR "
        "coalesce(a.child_representation_id, '') != coalesce(o.child_representation_id, '')",
        label="anonymization representation lineage drift",
    )
    expect_zero(connection, "SELECT count(*) FROM anonym a JOIN anonym_observed o USING(stable_uid) WHERE a.action != 'drop' AND a.output_text_sha256 != o.output_text_sha256", label="anonymization output content hash drift")
    expect_zero(connection, "SELECT count(*) FROM anonym a WHERE a.action != 'drop' AND a.span_count > 0 AND a.input_text_sha256 = a.output_text_sha256", label="masked output hash did not change")
    expect_zero(connection, "SELECT count(*) FROM anonym a WHERE a.action != 'drop' AND a.span_count = 0 AND a.input_text_sha256 != a.output_text_sha256", label="text changed without an approved mask span")
    expect_zero(
        connection,
        "SELECT count(*) FROM anonym a JOIN pool p USING(stable_uid) "
        "WHERE a.parent_representation_id != p.input_representation_id",
        label="anonymization parent representation drift from dedup survivor",
    )
    expect_zero(
        connection,
        "SELECT count(*) FROM anonym_observed o JOIN pool p USING(stable_uid) WHERE "
        "o.acquisition_source_id != p.acquisition_source_id OR o.source_dataset != p.source_dataset OR "
        "(o.action != 'drop' AND (o.source_route != p.source_route OR o.review_route != p.review_route OR o.extraction_route != p.extraction_route))",
        label="anonymization partition source provenance drift",
    )


def stage_totals(connection: sqlite3.Connection, *, stage: str) -> dict[str, int]:
    queries = {
        "dedup_input": "SELECT count(*), sum(characters), sum(tokens) FROM pool",
        "dedup_retained": "SELECT count(*), sum(p.characters), sum(p.tokens) FROM pool p JOIN dedup d USING(stable_uid) WHERE d.action = 'keep'",
        "greekmmlu_retained": "SELECT count(*), sum(p.characters), sum(p.tokens) FROM pool p JOIN dedup d USING(stable_uid) JOIN decontam g USING(stable_uid) WHERE d.action = 'keep' AND g.action = 'keep'",
        "anonymization_retained": "SELECT count(*), sum(o.characters), sum(o.tokens) FROM anonym_observed o WHERE o.action = 'keep'",
        "prestructural_frozen": "SELECT count(*), sum(o.characters), sum(o.tokens) FROM anonym_observed o WHERE o.action = 'keep'",
    }
    if stage not in queries:
        raise AssertionError(stage)
    documents, characters, tokens = connection.execute(queries[stage]).fetchone()
    return {"documents": int(documents or 0), "characters": int(characters or 0), "tokens": int(tokens or 0)}


SOURCE_FIELDS = "acquisition_source_id, source_dataset, source_route, review_route, extraction_route"


def source_groups(connection: sqlite3.Connection) -> list[tuple[str, str, str, str, str]]:
    return [tuple(str(value) for value in row) for row in connection.execute(f"SELECT {SOURCE_FIELDS} FROM pool GROUP BY {SOURCE_FIELDS} ORDER BY {SOURCE_FIELDS}")]


def source_stage_totals(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    stages = (
        ("dedup_input", "SELECT count(*), sum(characters), sum(tokens) FROM pool WHERE " + " AND ".join(f"{field} = ?" for field in SOURCE_FIELDS.split(", "))),
        ("dedup_retained", "SELECT count(*), sum(p.characters), sum(p.tokens) FROM pool p JOIN dedup d USING(stable_uid) WHERE d.action = 'keep' AND " + " AND ".join(f"p.{field} = ?" for field in SOURCE_FIELDS.split(", "))),
        ("greekmmlu_retained", "SELECT count(*), sum(p.characters), sum(p.tokens) FROM pool p JOIN dedup d USING(stable_uid) JOIN decontam g USING(stable_uid) WHERE d.action = 'keep' AND g.action = 'keep' AND " + " AND ".join(f"p.{field} = ?" for field in SOURCE_FIELDS.split(", "))),
        ("anonymization_retained", "SELECT count(*), sum(o.characters), sum(o.tokens) FROM anonym_observed o JOIN pool p USING(stable_uid) WHERE o.action = 'keep' AND " + " AND ".join(f"p.{field} = ?" for field in SOURCE_FIELDS.split(", "))),
        ("prestructural_frozen", "SELECT count(*), sum(o.characters), sum(o.tokens) FROM anonym_observed o JOIN pool p USING(stable_uid) WHERE o.action = 'keep' AND " + " AND ".join(f"p.{field} = ?" for field in SOURCE_FIELDS.split(", "))),
    )
    for source in source_groups(connection):
        stage_values: dict[str, dict[str, int]] = {}
        for name, query in stages:
            row = connection.execute(query, source).fetchone()
            stage_values[name] = {"documents": int(row[0] or 0), "characters": int(row[1] or 0), "tokens": int(row[2] or 0)}
        result.append({
            "acquisition_source_id": source[0],
            "source_dataset": source[1],
            # Empty strings are the explicit, non-inferred base route.  Keep
            # that fact visible as null instead of guessing from a file name.
            "source_route": source[2] or None,
            "review_route": source[3] or None,
            "extraction_route": source[4] or None,
            "stages": stage_values,
        })
    return result


def event_rows(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return source/reason-coded removals and allowed masking deltas only."""

    events: list[dict[str, Any]] = []
    definitions = (
        (
            "50-dedup",
            "drop",
            "d.reason",
            "FROM pool p JOIN dedup d USING(stable_uid) WHERE d.action = 'drop'",
            "p.characters",
            "p.tokens",
            "0",
            "0",
        ),
        (
            "60-greekmmlu-decontamination",
            "remove_or_quarantine",
            "g.reason",
            "FROM pool p JOIN decontam g USING(stable_uid) WHERE g.action != 'keep'",
            "p.characters",
            "p.tokens",
            "0",
            "0",
        ),
        (
            "65-anonymization-sanitization",
            "remove_or_quarantine",
            "a.reason_key",
            "FROM pool p JOIN anonym a USING(stable_uid) LEFT JOIN anonym_observed o USING(stable_uid) WHERE a.action != 'keep'",
            "p.characters",
            "p.tokens",
            "coalesce(o.characters, 0)",
            "coalesce(o.tokens, 0)",
        ),
        (
            "65-anonymization-sanitization",
            "approved_masking_transform",
            "a.reason_key",
            "FROM pool p JOIN anonym a USING(stable_uid) JOIN anonym_observed o USING(stable_uid) WHERE a.action != 'drop' AND a.span_count > 0",
            "p.characters",
            "p.tokens",
            "o.characters",
            "o.tokens",
        ),
    )
    group = "p.acquisition_source_id, p.source_dataset, p.source_route, p.review_route, p.extraction_route"
    for stage, action, reason, clause, before_char, before_tokens, after_char, after_tokens in definitions:
        query = f"""
          SELECT {group}, {reason} AS reason, count(*) AS documents,
                 sum({before_char}) AS characters_before, sum({after_char}) AS characters_after,
                 sum({before_tokens}) AS tokens_before, sum({after_tokens}) AS tokens_after
          {clause}
          GROUP BY {group}, {reason}
          ORDER BY {group}, {reason}
        """
        for row in connection.execute(query):
            events.append({
                "stage": stage,
                "action": action,
                "reason": str(row[5]),
                "acquisition_source_id": str(row[0]),
                "source_dataset": str(row[1]),
                "source_route": str(row[2]) or None,
                "review_route": str(row[3]) or None,
                "extraction_route": str(row[4]) or None,
                "documents": int(row[6] or 0),
                "characters_before": int(row[7] or 0),
                "characters_after": int(row[8] or 0),
                "characters_delta": int((row[8] or 0) - (row[7] or 0)),
                "tokens_before": int(row[9] or 0),
                "tokens_after": int(row[10] or 0),
                "tokens_delta": int((row[10] or 0) - (row[9] or 0)),
            })
    return events


def pii_counts_by_source(connection: sqlite3.Connection) -> dict[tuple[str, str, str, str, str], dict[str, int]]:
    result: dict[tuple[str, str, str, str, str], dict[str, int]] = defaultdict(dict)
    for row in connection.execute("SELECT acquisition_source_id, source_dataset, source_route, review_route, extraction_route, pii_type, spans FROM pii_types ORDER BY 1, 2, 3, 4, 5, 6"):
        result[(str(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4]))][str(row[5])] = int(row[6])
    return result


def anonymization_audit(connection: sqlite3.Connection, *, review_sample_limit: int) -> dict[str, Any]:
    checks = {
        "changed_without_approved_span": "SELECT count(*) FROM anonym WHERE action != 'drop' AND span_count = 0 AND input_text_sha256 != output_text_sha256",
        "approved_span_without_text_change": "SELECT count(*) FROM anonym WHERE action != 'drop' AND span_count > 0 AND input_text_sha256 = output_text_sha256",
        "drop_outside_private_data_policy": "SELECT count(*) FROM anonym WHERE action = 'drop' AND (reason_key != ? OR acquisition_source_id != 'diavgeia' OR span_count != 0 OR output_text_sha256 IS NOT NULL)",
        "quarantine_outside_precision_gate": "SELECT count(*) FROM anonym WHERE action = 'quarantine' AND (instr(reason_key, ?) = 0 OR acquisition_source_id != 'diavgeia' OR span_count < 3)",
        "decision_without_output_partition": "SELECT count(*) FROM anonym a LEFT JOIN anonym_observed o USING(stable_uid) WHERE o.stable_uid IS NULL",
    }
    parameters: dict[str, Sequence[object]] = {
        "drop_outside_private_data_policy": (DROP_REASON,),
        "quarantine_outside_precision_gate": (QUARANTINE_REASON,),
    }
    observed = {name: query_scalar(connection, query, parameters.get(name, ())) for name, query in checks.items()}
    if any(observed.values()):
        raise WaterfallError(f"anonymization false-positive audit failed: {observed}")

    pii = pii_counts_by_source(connection)
    query = f"""
      SELECT p.acquisition_source_id, p.source_dataset, p.source_route, p.review_route, p.extraction_route,
             count(*) AS input_documents,
             sum(p.characters) AS characters_before,
             sum(p.tokens) AS tokens_before,
             sum(CASE WHEN a.action = 'keep' THEN 1 ELSE 0 END) AS kept_documents,
             sum(CASE WHEN a.action = 'drop' THEN 1 ELSE 0 END) AS dropped_documents,
             sum(CASE WHEN a.action = 'quarantine' THEN 1 ELSE 0 END) AS quarantined_documents,
             sum(CASE WHEN a.action = 'keep' THEN o.characters ELSE 0 END) AS characters_after_kept,
             sum(CASE WHEN a.action = 'keep' THEN o.tokens ELSE 0 END) AS tokens_after_kept,
             sum(CASE WHEN a.action != 'drop' AND a.input_text_sha256 != a.output_text_sha256 THEN 1 ELSE 0 END) AS changed_emitted_documents,
             sum(a.span_count) AS approved_direct_identifier_spans
      FROM pool p
      JOIN dedup d USING(stable_uid)
      JOIN decontam g USING(stable_uid)
      JOIN anonym a USING(stable_uid)
      JOIN anonym_observed o USING(stable_uid)
      WHERE d.action = 'keep' AND g.action = 'keep'
      GROUP BY p.acquisition_source_id, p.source_dataset, p.source_route, p.review_route, p.extraction_route
      ORDER BY p.acquisition_source_id, p.source_dataset, p.source_route, p.review_route, p.extraction_route
    """
    # Use names rather than positional values: the output intentionally omits
    # document identifiers and protected span values, and this avoids an
    # accidental field shift silently changing its aggregate meaning.
    named_rows: list[dict[str, Any]] = []
    cursor = connection.execute(query)
    names = [description[0] for description in cursor.description]
    for raw in cursor.fetchall():
        item = dict(zip(names, raw, strict=True))
        key = (str(item["acquisition_source_id"]), str(item["source_dataset"]), str(item["source_route"]), str(item["review_route"]), str(item["extraction_route"]))
        before_char = int(item["characters_before"] or 0)
        after_char = int(item["characters_after_kept"] or 0)
        before_tokens = int(item["tokens_before"] or 0)
        after_tokens = int(item["tokens_after_kept"] or 0)
        named_rows.append({
            "acquisition_source_id": key[0],
            "source_dataset": key[1],
            "source_route": key[2] or None,
            "review_route": key[3] or None,
            "extraction_route": key[4] or None,
            "documents_before": int(item["input_documents"] or 0),
            "documents_kept": int(item["kept_documents"] or 0),
            "documents_dropped": int(item["dropped_documents"] or 0),
            "documents_quarantined": int(item["quarantined_documents"] or 0),
            "characters_before": before_char,
            "characters_after_kept": after_char,
            "characters_delta_kept": after_char - before_char,
            "tokens_before": before_tokens,
            "tokens_after_kept": after_tokens,
            "tokens_delta_kept": after_tokens - before_tokens,
            "changed_emitted_documents": int(item["changed_emitted_documents"] or 0),
            "approved_direct_identifier_spans": int(item["approved_direct_identifier_spans"] or 0),
            "approved_direct_identifier_spans_by_type": dict(sorted(pii.get(key, {}).items())),
        })
    review_candidates = query_scalar(
        connection,
        "SELECT count(*) FROM anonym WHERE action != 'keep' OR span_count > 0",
    )
    queue: list[dict[str, Any]] = []
    cursor = connection.execute(
        """
        SELECT a.stable_uid, a.action, a.reason_key, a.span_count,
               p.acquisition_source_id, p.source_dataset, p.source_route, p.review_route, p.extraction_route
        FROM anonym a JOIN pool p USING(stable_uid)
        WHERE a.action != 'keep' OR a.span_count > 0
        ORDER BY p.acquisition_source_id, p.source_dataset, a.action, a.reason_key, a.stable_uid
        LIMIT ?
        """,
        (review_sample_limit,),
    )
    for stable_uid, action, reason_key, span_count, acquisition, dataset, source_route, review_route, extraction_route in cursor:
        # This irreversible reference permits a protected reviewer to derive
        # the matching ledger row without putting a stable UID, source-doc ID,
        # raw identifier, or source text into the compact audit.
        reference = hashlib.sha256(
            ("agent1_v3_anonymization_review_reference_v1\x00" + str(stable_uid)).encode("utf-8")
        ).hexdigest()
        queue.append({
            "review_reference_sha256": reference,
            "action": str(action),
            "reason": str(reason_key),
            "approved_span_count": int(span_count),
            "acquisition_source_id": str(acquisition),
            "source_dataset": str(dataset),
            "source_route": str(source_route) or None,
            "review_route": str(review_route) or None,
            "extraction_route": str(extraction_route) or None,
        })
    return {
        "schema_version": ANONYMIZATION_AUDIT_SCHEMA,
        "status": "automatic_checks_passed_semantic_review_pending",
        "scope": {
            "direct_identifier_registry_only": True,
            "generic_person_name_ner": False,
            "street_address_masking": False,
            "html_cleaning": False,
            "ocr_cleaning": False,
            "structural_cleaning": False,
            "raw_text_or_pii_in_output": False,
        },
        "false_positive_audit": {
            "automatic_policy_lineage_checks": {
                "status": "passed",
                "checks": [{"name": name, "violations": count} for name, count in sorted(observed.items())],
            },
            "independent_semantic_review": {
                "status": "pending",
                "required_before_any_claim_of_semantic_false_positive_clearance": True,
                "eligible_rows": review_candidates,
                "reported_review_references": queue,
                "selection": "bounded deterministic source/action/reason/stable-reference ordering",
                "note": "The aggregate cannot judge whether a valid-looking identifier is contextually non-sensitive; review happens only in the protected ledger environment.",
            },
        },
        "before_after_by_source": named_rows,
    }


def stage_rows(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    definitions = (
        ("50-dedup-input-pre-greekmmlu", "admitted_pre_mmlu_pool", "dedup_input"),
        ("50-dedup-representatives", "identity_reconciled_content_work_representation_selection", "dedup_retained"),
        ("60-greekmmlu-decontamination", "high_confidence_drop_ambiguous_quarantine", "greekmmlu_retained"),
        ("65-anonymization-sanitization", "direct_identifier_masking_private_data_policy", "anonymization_retained"),
        ("70-prestructural-freeze", "non_publishable_no_structural_transform", "prestructural_frozen"),
    )
    previous: dict[str, int] | None = None
    result: list[dict[str, Any]] = []
    for order, (stage, reason, name) in enumerate(definitions):
        after = stage_totals(connection, stage=name)
        before = after if previous is None else previous
        result.append({
            "order": order,
            "stage": stage,
            "reason": reason,
            "documents_before": before["documents"],
            "documents_after": after["documents"],
            "documents_removed_or_quarantined": before["documents"] - after["documents"],
            "characters_before": before["characters"],
            "characters_after": after["characters"],
            "characters_delta": after["characters"] - before["characters"],
            "tokens_before": before["tokens"],
            "tokens_after": after["tokens"],
            "tokens_delta": after["tokens"] - before["tokens"],
        })
        previous = after
    return result


def inventory_closure(connection: sqlite3.Connection) -> dict[str, int]:
    return {
        "pool_rows": query_scalar(connection, "SELECT count(*) FROM pool"),
        "dedup_ledger_rows": query_scalar(connection, "SELECT count(*) FROM dedup"),
        "dedup_kept_rows": query_scalar(connection, "SELECT count(*) FROM dedup WHERE action = 'keep'"),
        "dedup_materialized_rows": query_scalar(connection, "SELECT count(*) FROM materialized"),
        "greekmmlu_ledger_rows": query_scalar(connection, "SELECT count(*) FROM decontam"),
        "greekmmlu_output_partition_rows": query_scalar(connection, "SELECT count(*) FROM decontam_observed"),
        "anonymization_ledger_rows": query_scalar(connection, "SELECT count(*) FROM anonym"),
        "anonymization_partition_rows": query_scalar(connection, "SELECT count(*) FROM anonym_observed"),
        "anonymization_kept_rows": query_scalar(connection, "SELECT count(*) FROM anonym WHERE action = 'keep'"),
    }


def require_counts(manifest: Mapping[str, Any], actual: Mapping[str, int], *, label: str, mapping: Mapping[str, str]) -> None:
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping):
        raise WaterfallError(f"{label} lacks aggregate counts")
    for manifest_key, actual_key in mapping.items():
        # Existing stage writers use Counter serialization, which omits a
        # genuinely zero action.  Treat only a missing *zero* as equivalent;
        # every positive or declared count must still close exactly.
        declared = counts.get(manifest_key, 0)
        if declared != actual[actual_key]:
            raise WaterfallError(f"{label} count drift for {manifest_key}")


def verify_anonymization_policy(manifest: Mapping[str, Any]) -> None:
    boundaries = manifest.get("transform_boundaries")
    if not isinstance(boundaries, Mapping):
        raise WaterfallError("anonymization manifest lacks transform boundaries")
    expected = {
        "generic_person_name_ner": False,
        "street_address_masking": False,
        "html_cleaning": False,
        "ocr_cleaning": False,
        "structural_cleaning": False,
        "stable_uid_preserved": True,
        "new_child_representation_ids": True,
    }
    if any(boundaries.get(key) is not value for key, value in expected.items()):
        raise WaterfallError("anonymization transform boundary drift")
    policy = manifest.get("policy")
    if not isinstance(policy, Mapping) or set(policy.get("mask_types", [])) != ALLOWED_MASK_TYPES:
        raise WaterfallError("anonymization direct-identifier registry drift")


def verify_ledger_closure(
    closure: Mapping[str, Any], *, closure_path: Path, anonymization_manifest: Path, protected_ledger_root: Path
) -> None:
    require_manifest(closure, schema=LEDGER_CLOSURE_SCHEMA, status="passed", label="protected anonymization ledger closure")
    require_binding(closure.get("anonymization_manifest"), anonymization_manifest, label="ledger closure anonymization manifest")
    protected = closure.get("protected_ledger")
    if not isinstance(protected, Mapping):
        raise WaterfallError("ledger closure lacks protected ledger metadata")
    if Path(str(protected.get("path", ""))).resolve() != protected_ledger_root.resolve():
        raise WaterfallError("ledger closure protected ledger root drift")
    if protected.get("contains_raw_span_values") is not True or protected.get("public_training_output") is not False:
        raise WaterfallError("ledger closure does not preserve private protected-ledger semantics")
    # Keep the binding call here so a caller cannot accidentally pass an
    # arbitrary readable JSON object in place of the immutable closure.
    require_regular_file(closure_path, label="protected anonymization ledger closure")


def verify_postmask_report(report: Mapping[str, Any], *, report_path: Path, anonymization_manifest: Path, anonymized_root: Path) -> None:
    require_manifest(report, schema=POSTMASK_REPORT_SCHEMA, status="passed", label="post-mask duplicate report")
    require_binding(report.get("anonymization_manifest"), anonymization_manifest, label="post-mask anonymization manifest")
    if Path(str(report.get("source_corpus_root", ""))).resolve() != anonymized_root.resolve():
        raise WaterfallError("post-mask report source corpus drift")
    if report.get("verification_only") is not True or report.get("materialization_performed") is not False or report.get("second_deduplication_applied") is not False:
        raise WaterfallError("post-mask report is not verification-only")
    if int(report.get("material_new_duplicate_count", -1)) != 0:
        raise WaterfallError("post-mask report has unresolved new duplicates")
    require_regular_file(report_path, label="post-mask duplicate report")


def verify_prestructural(
    manifest: Mapping[str, Any],
    *,
    dedup_manifest: Path,
    decontamination_manifest: Path,
    anonymization_manifest: Path,
    ledger_closure: Path,
    postmask_report: Path,
    anonymized_root: Path,
) -> None:
    require_manifest(manifest, schema=PRESTRUCTURAL_SCHEMA, status="prestructural_frozen", label="prestructural manifest")
    if manifest.get("publish_permitted") is not False or manifest.get("structural_state") != "awaiting_agent2_handoff":
        raise WaterfallError("prestructural manifest permits a structural/public action")
    if Path(str(manifest.get("corpus_root", ""))).resolve() != anonymized_root.resolve():
        raise WaterfallError("prestructural manifest corpus root drift")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        raise WaterfallError("prestructural manifest lacks immutable inputs")
    for name, path in (
        ("dedup_manifest", dedup_manifest),
        ("decontamination_manifest", decontamination_manifest),
        ("anonymization_manifest", anonymization_manifest),
        ("anonymization_ledger", ledger_closure),
        ("postmask_duplicate_report", postmask_report),
    ):
        require_binding(inputs.get(name), path, label=f"prestructural {name}")


def write_json_no_replace(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable waterfall: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite immutable waterfall: {path}") from exc
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class Arguments:
    dedup_pool: Path
    dedup_pool_manifest: Path
    dedup_ledger: Path
    dedup_ledger_manifest: Path
    dedup_materialized: Path
    dedup_materialization_manifest: Path
    decontamination_input: Path
    decontamination_output: Path
    decontamination_dropped: Path
    decontamination_quarantine: Path
    decontamination_ledger: Path
    decontamination_manifest: Path
    anonymization_input: Path
    anonymization_output: Path
    anonymization_dropped: Path
    anonymization_quarantine: Path
    protected_ledger_root: Path
    anonymization_manifest: Path
    anonymization_ledger_closure: Path
    postmask_duplicate_report: Path
    prestructural_manifest: Path
    candidate_roster: Path
    tokenizer_json: Path
    work_database: Path
    output: Path
    batch_rows: int
    tokenizer_batch_docs: int
    review_sample_limit: int


def build(args: Arguments) -> dict[str, Any]:
    if args.batch_rows < 1 or args.tokenizer_batch_docs < 1 or args.review_sample_limit < 1:
        raise WaterfallError("batch sizes must be positive")
    if args.decontamination_input.resolve() != args.dedup_materialized.resolve():
        raise WaterfallError("GreekMMLU input must be the exact dedup materialization root")
    if args.anonymization_input.resolve() != args.decontamination_output.resolve():
        raise WaterfallError("anonymization input must be the exact retained GreekMMLU root")
    route_declarations = frozen_route_declarations(args.candidate_roster)

    pool_manifest = read_object(args.dedup_pool_manifest, label="dedup pool manifest")
    require_manifest(pool_manifest, schema=POOL_MANIFEST_SCHEMA, status="passed", label="dedup pool manifest")
    pool_files = verify_manifest_tree(pool_manifest, root=args.dedup_pool, receipt_key=None, label="dedup pool")

    ledger_manifest = read_object(args.dedup_ledger_manifest, label="dedup ledger manifest")
    require_manifest(ledger_manifest, schema=DEDUP_LEDGER_MANIFEST_SCHEMA, status="passed", label="dedup ledger manifest")
    if Path(str(ledger_manifest.get("pool", ""))).resolve() != args.dedup_pool.resolve():
        raise WaterfallError("dedup ledger manifest pool root drift")
    identity = ledger_manifest.get("identity_reconciliation")
    ordered_dedup = ledger_manifest.get("ordered_dedup")
    if (
        not isinstance(identity, Mapping)
        or identity.get("selection_order") != IDENTITY_SELECTION_ORDER
        or identity.get("exact_identity_precedes_near_passes") is not True
    ):
        raise WaterfallError("dedup ledger manifest lacks the mandatory identity-first reconciliation")
    if (
        not isinstance(ordered_dedup, Mapping)
        or ordered_dedup.get("schema_version") != ORDERED_DEDUP_SCHEMA
        or ordered_dedup.get("pass_order") != list(ORDERED_DEDUP_PASSES)
        or ordered_dedup.get("exact_identity_precedes_near_passes") is not True
    ):
        raise WaterfallError("dedup ledger manifest lacks the mandatory ordered pass composition")
    verify_absolute_parquet_receipt(ledger_manifest.get("ledger"), args.dedup_ledger, label="dedup ledger")

    materialization_manifest = read_object(args.dedup_materialization_manifest, label="dedup materialization manifest")
    require_manifest(materialization_manifest, schema=DEDUP_MATERIALIZATION_MANIFEST_SCHEMA, status="passed", label="dedup materialization manifest")
    require_binding(materialization_manifest.get("ledger"), args.dedup_ledger, label="dedup materialization ledger")
    materialized_files = verify_manifest_tree(materialization_manifest, root=args.dedup_materialized, receipt_key=None, label="dedup materialization")

    decontamination_manifest = read_object(args.decontamination_manifest, label="decontamination manifest")
    require_manifest(decontamination_manifest, schema=DECONTAMINATION_MANIFEST_SCHEMA, status="passed", label="decontamination manifest")
    decontamination_input_files = verify_manifest_tree(decontamination_manifest, root=args.decontamination_input, receipt_key="input", label="decontamination input")
    if decontamination_input_files != materialized_files:
        raise WaterfallError("decontamination input receipt inventory differs from dedup materialization")
    decontamination_output_files = verify_manifest_tree(decontamination_manifest, root=args.decontamination_output, receipt_key="output", label="decontamination output")
    decontamination_dropped_files = verify_manifest_tree(decontamination_manifest, root=args.decontamination_dropped, receipt_key="dropped", label="decontamination dropped")
    decontamination_quarantine_files = verify_manifest_tree(decontamination_manifest, root=args.decontamination_quarantine, receipt_key="quarantine", label="decontamination quarantine")
    decontamination_ledger_files = verify_manifest_tree(decontamination_manifest, root=args.decontamination_ledger, receipt_key="ledger", label="decontamination ledger")

    anonymization_manifest = read_object(args.anonymization_manifest, label="anonymization manifest")
    require_manifest(anonymization_manifest, schema=ANONYMIZATION_MANIFEST_SCHEMA, status="completed", label="anonymization manifest")
    verify_anonymization_policy(anonymization_manifest)
    for field, path in (
        ("input", args.anonymization_input), ("output", args.anonymization_output),
        ("dropped", args.anonymization_dropped), ("quarantine", args.anonymization_quarantine),
    ):
        if Path(str(anonymization_manifest.get(field, ""))).resolve() != path.resolve():
            raise WaterfallError(f"anonymization manifest {field} root drift")
    protected = anonymization_manifest.get("protected_ledger")
    if not isinstance(protected, Mapping) or Path(str(protected.get("path", ""))).resolve() != args.protected_ledger_root.resolve():
        raise WaterfallError("anonymization manifest protected-ledger root drift")
    if protected.get("directory_mode") != "0700" or protected.get("file_mode") != "0600":
        raise WaterfallError("anonymization protected-ledger permission declaration drift")
    anonymization_input_files = verify_manifest_tree(anonymization_manifest, root=args.anonymization_input, receipt_key="input", label="anonymization input")
    if anonymization_input_files != decontamination_output_files:
        raise WaterfallError("anonymization input receipt inventory differs from decontamination output")
    anonymization_output_files = verify_manifest_tree(anonymization_manifest, root=args.anonymization_output, receipt_key="output", label="anonymization output")
    anonymization_dropped_files = verify_manifest_tree(anonymization_manifest, root=args.anonymization_dropped, receipt_key="dropped", label="anonymization dropped")
    anonymization_quarantine_files = verify_manifest_tree(anonymization_manifest, root=args.anonymization_quarantine, receipt_key="quarantine", label="anonymization quarantine")
    protected_ledger_files = verify_manifest_tree(anonymization_manifest, root=args.protected_ledger_root, receipt_key="protected_ledger", label="protected anonymization ledger")
    protected_mode = stat.S_IMODE(args.protected_ledger_root.lstat().st_mode)
    if protected_mode != 0o700:
        raise WaterfallError("protected anonymization ledger root must have mode 0700")
    for protected_file in protected_ledger_files:
        if stat.S_IMODE(protected_file.lstat().st_mode) != 0o600:
            raise WaterfallError("protected anonymization ledger shards must have mode 0600")

    closure = read_object(args.anonymization_ledger_closure, label="protected anonymization ledger closure")
    verify_ledger_closure(closure, closure_path=args.anonymization_ledger_closure, anonymization_manifest=args.anonymization_manifest, protected_ledger_root=args.protected_ledger_root)
    report = read_object(args.postmask_duplicate_report, label="post-mask duplicate report")
    verify_postmask_report(report, report_path=args.postmask_duplicate_report, anonymization_manifest=args.anonymization_manifest, anonymized_root=args.anonymization_output)
    prestructural = read_object(args.prestructural_manifest, label="prestructural manifest")
    verify_prestructural(
        prestructural,
        dedup_manifest=args.dedup_materialization_manifest,
        decontamination_manifest=args.decontamination_manifest,
        anonymization_manifest=args.anonymization_manifest,
        ledger_closure=args.anonymization_ledger_closure,
        postmask_report=args.postmask_duplicate_report,
        anonymized_root=args.anonymization_output,
    )

    tokenizer = ExactTokenizer(args.tokenizer_json)
    connection = connect_database(args.work_database)
    try:
        pool_rows = load_pool(
            connection,
            files=pool_files,
            tokenizer=tokenizer,
            batch_rows=args.batch_rows,
            tokenizer_batch_docs=args.tokenizer_batch_docs,
            expected_routes=route_declarations,
        )
        dedup_rows = load_dedup_ledger(connection, path=args.dedup_ledger, batch_rows=args.batch_rows)
        materialized_rows = scan_materialized(connection, files=materialized_files, batch_rows=args.batch_rows)
        decontam_rows = load_decontam_ledger(connection, files=decontamination_ledger_files, batch_rows=args.batch_rows)
        decontam_output_rows = scan_decontam_partition(connection, files=decontamination_output_files, action="keep", batch_rows=args.batch_rows)
        decontam_drop_rows = scan_decontam_partition(connection, files=decontamination_dropped_files, action="drop", batch_rows=args.batch_rows)
        decontam_quarantine_rows = scan_decontam_partition(connection, files=decontamination_quarantine_files, action="quarantine", batch_rows=args.batch_rows)
        close_decontam_partitions(connection)
        require_counts(
            decontamination_manifest,
            {"input": decontam_rows, "keep": decontam_output_rows, "drop": decontam_drop_rows, "quarantine": decontam_quarantine_rows},
            label="decontamination manifest",
            mapping={"input": "input", "keep": "keep", "drop": "drop", "quarantine": "quarantine"},
        )
        anonym_rows, anonym_totals = load_anonym_ledger(connection, files=protected_ledger_files, batch_rows=args.batch_rows)
        anonym_output_rows = scan_anonym_emitted_partition(connection, files=anonymization_output_files, action="keep", tokenizer=tokenizer, batch_rows=args.batch_rows, tokenizer_batch_docs=args.tokenizer_batch_docs)
        anonym_quarantine_rows = scan_anonym_emitted_partition(connection, files=anonymization_quarantine_files, action="quarantine", tokenizer=tokenizer, batch_rows=args.batch_rows, tokenizer_batch_docs=args.tokenizer_batch_docs)
        anonym_drop_rows = scan_anonym_dropped_partition(connection, files=anonymization_dropped_files, batch_rows=args.batch_rows)
        close_anonym_partitions(connection)
        require_counts(
            anonymization_manifest,
            {"input_rows": anonym_rows, "action:keep": anonym_output_rows, "action:drop": anonym_drop_rows, "action:quarantine": anonym_quarantine_rows, "protected_ledger_rows": anonym_rows},
            label="anonymization manifest",
            mapping={"input_rows": "input_rows", "action:keep": "action:keep", "action:drop": "action:drop", "action:quarantine": "action:quarantine", "protected_ledger_rows": "protected_ledger_rows"},
        )
        if anonym_rows != sum(anonym_totals.get(f"action:{action}", 0) for action in ("keep", "drop", "quarantine")):
            raise WaterfallError("anonymization action mass does not close")
        if pool_rows != dedup_rows or materialized_rows != query_scalar(connection, "SELECT count(*) FROM dedup WHERE action = 'keep'"):
            raise WaterfallError("dedup mass does not close")
        audit = anonymization_audit(connection, review_sample_limit=args.review_sample_limit)
        payload = {
            "schema_version": WATERFALL_SCHEMA,
            # Receipt/mass closure passed, but the compact artifact must not
            # be misread as a semantic false-positive clearance.
            "status": "passed_with_independent_semantic_review_pending",
            "completed_at": utc_now(),
            "tokenizer": binding(args.tokenizer_json),
            "bindings": {
                "dedup_pool_manifest": binding(args.dedup_pool_manifest),
                "dedup_ledger_manifest": binding(args.dedup_ledger_manifest),
                "dedup_materialization_manifest": binding(args.dedup_materialization_manifest),
                "decontamination_manifest": binding(args.decontamination_manifest),
                "anonymization_manifest": binding(args.anonymization_manifest),
                "anonymization_ledger_closure": binding(args.anonymization_ledger_closure),
                "postmask_duplicate_report": binding(args.postmask_duplicate_report),
                "prestructural_manifest": binding(args.prestructural_manifest),
                "candidate_roster": binding(args.candidate_roster),
            },
            "stage_order": ["50-dedup", "55-greekmmlu-freeze", "60-decontamination", "65-anonymization-sanitization", "70-prestructural-freeze"],
            "stages": stage_rows(connection),
            "source_stage_totals": source_stage_totals(connection),
            "removal_events": event_rows(connection),
            "anonymization_audit": audit,
            "inventory_closure": inventory_closure(connection),
            "invariants": {
                "dedup_precedes_greekmmlu": True,
                "greekmmlu_precedes_anonymization": True,
                "prestructural_freeze_has_no_structural_removal": True,
                "logical_source_route_preserved_without_filename_inference": True,
                "logical_source_and_extraction_routes_match_frozen_roster": True,
                "tokens_are_exact_pinned_tokenizer_counts": True,
                "masking_can_change_character_or_token_count": True,
                "raw_text_or_pii_in_output": False,
            },
        }
    finally:
        connection.close()
    return payload


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, help_text in (
        ("dedup-pool", "receipt-bound admitted pre-dedup Parquet root"),
        ("dedup-pool-manifest", "immutable admitted-pool manifest"),
        ("dedup-ledger", "identity-reconciled final dedup decision ledger"),
        ("dedup-ledger-manifest", "immutable identity-reconciled dedup manifest"),
        ("dedup-materialized", "receipt-bound dedup survivor Parquet root"),
        ("dedup-materialization-manifest", "immutable dedup materialization manifest"),
        ("decontamination-input", "must equal dedup materialized root"),
        ("decontamination-output", "GreekMMLU retained Parquet root"),
        ("decontamination-dropped", "GreekMMLU dropped Parquet root"),
        ("decontamination-quarantine", "GreekMMLU quarantine Parquet root"),
        ("decontamination-ledger", "GreekMMLU decision-ledger root"),
        ("decontamination-manifest", "immutable GreekMMLU manifest"),
        ("anonymization-input", "must equal retained GreekMMLU root"),
        ("anonymization-output", "public masked training-candidate root"),
        ("anonymization-dropped", "privateData tombstone root"),
        ("anonymization-quarantine", "precision-gated Diavgeia quarantine root"),
        ("protected-ledger-root", "0700/0600 protected anonymization ledger root"),
        ("anonymization-manifest", "immutable anonymization manifest"),
        ("anonymization-ledger-closure", "compact protected-ledger closure"),
        ("postmask-duplicate-report", "zero-result verification-only duplicate report"),
        ("prestructural-manifest", "immutable no-publication prestructural freeze"),
        ("candidate-roster", "frozen logical-first source/extraction route declaration"),
        ("tokenizer-json", "pinned exact tokenizer JSON"),
        ("work-database", "new SQLite work database; stores only hashes/counts"),
        ("output", "new immutable compact waterfall JSON"),
    ):
        parser.add_argument(f"--{name}", type=Path, required=True, help=help_text)
    parser.add_argument("--batch-rows", type=int, default=1024)
    parser.add_argument("--tokenizer-batch-docs", type=int, default=256)
    parser.add_argument("--review-sample-limit", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parsed = parser().parse_args(argv)
    args = Arguments(**vars(parsed))
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable waterfall: {args.output}")
    payload = build(args)
    write_json_no_replace(args.output.resolve(), payload)
    print(json.dumps({"ok": True, "output": str(args.output.resolve()), "sources": len(payload["source_stage_totals"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
