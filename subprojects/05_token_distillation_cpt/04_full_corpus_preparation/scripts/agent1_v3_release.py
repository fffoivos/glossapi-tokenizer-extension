#!/usr/bin/env python3
"""Fail-closed local release materialization for Agent 1's ordered v3 lane.

The legacy v2 materializer assumes that deduplication happens after cleaning
and GreekMMLU decontamination.  That assumption is deliberately invalid for
the post-Nanochat v3 lane.  This command instead proves the following before
it writes a local release:

* the complete deduplication decision ledger is bound to the admitted,
  *pre-GreekMMLU* representation and its exact text hash;
* GreekMMLU and anonymization ledgers close over those dedup survivors;
* every final row is a descendant of its immutable dedup representation using
  the generic ``representation_id`` / ``parent_representation_id`` contract;
* the final tree is the Stage 78 structural output (or an explicit no-op over
  the anonymized tree), never an older pre-transform corpus.
* every compact Stage 30/35/40 review, quality, admission, and execution
  receipt is present, and the exact Stage-70 waterfall has a completed,
  independent anonymization semantic false-positive clearance.

It creates a private local training release only.  It intentionally does not
construct a redistributable dataset or publish anything.  The accompanying
Agent 3 handoff is JSON-only and refuses protected PII ledgers, raw Parquet,
GreekMMLU answers, and model checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


RELEASE_SCHEMA = "agent1_full_corpus_v3_release_manifest_v1"
VALIDATION_SCHEMA = "agent1_full_corpus_v3_release_validation_v1"
HANDOFF_SCHEMA = "agent1_full_corpus_v3_dataset_review_site_handoff_v1"
WATERFALL_SCHEMA = "agent1_full_corpus_v3_token_waterfall_v1"
ANONYMIZATION_AUDIT_SCHEMA = "agent1_full_corpus_v3_anonymization_audit_v1"
SEMANTIC_CLEARANCE_SCHEMA = (
    "agent1_full_corpus_v3_anonymization_semantic_false_positive_clearance_v1"
)

QUALITY_SUMMARY_SCHEMA = "dataset_quality_summary_v2"
REVIEW_PACKET_SCHEMA = "agent1_v3_review_packet_manifest_v1"
REVIEW_REQUEST_SCHEMA = "agent1_v3_review_request_v1"
REVIEW_RESPONSE_SCHEMA = "agent1_v3_review_response_v1"
RESPONSE_EXECUTION_RECEIPT_SCHEMA = "agent1_v3_codex_review_response_execution_receipt_v1"
ADJUDICATION_EXECUTION_RECEIPT_SCHEMA = "agent1_v3_codex_review_adjudication_execution_receipt_v1"
STAGE35_CLOSURE_SCHEMA = "agent1_v3_quality_review_evidence_closure_v1"
REVIEW_SAMPLE_QUALITY_SUMMARY_SCHEMA = "agent1_v3_masked_review_sample_quality_summary_v1"
REVIEW_SAMPLE_QUALITY_HANDOFF_SCHEMA = "agent1_v3_masked_review_sample_quality_handoff_v1"
REVIEW_AGGREGATE_SCHEMA = "agent1_full_corpus_v3_source_review_aggregate_v1"
ADMISSION_CONFIRMATION_SCHEMA = "agent1_full_corpus_v3_source_admission_confirmation_v1"
LINEAGE_SUMMARY_SCHEMA = "full_cpt_lineage_summary_v1"
SOURCE_NOVELTY_SCHEMA = "full_cpt_source_novelty_v1"
LICENSE_ADJUDICATION_SCHEMA = "full_cpt_source_license_adjudication_v1"

DEDUP_MANIFEST_SCHEMA = "agent1_full_corpus_v3_dedup_ledger_manifest_v1"
DECONTAMINATION_MANIFEST_SCHEMA = "agent1_full_corpus_v3_decontamination_manifest_v1"
ANONYMIZATION_MANIFEST_SCHEMA = "agent1_full_corpus_v3_anonymization_manifest_v1"
RUN_CONTRACT_SCHEMA = "agent1_full_corpus_v3_run_contract_v1"
STAGE_RECEIPT_SCHEMA = "agent1_full_corpus_v3_stage_receipt_v1"

MANIFEST_RELATIVE_PATH = Path("provenance/agent1_v3_release_manifest.json")
VALIDATION_RELATIVE_PATH = Path("provenance/agent1_v3_release_validation.json")
HANDOFF_RELATIVE_PATH = Path("site_handoff/dataset_review_site_handoff.json")
COMPACT_ROOT_RELATIVE_PATH = Path("site_handoff/compact")
TRAINING_ROOT_RELATIVE_PATH = Path("training/data")

HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_SITE_INPUT_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
MAX_COMPACT_INPUT_BYTES = 64 * 1024 * 1024
MAX_LINEAGE_DEPTH = 16

ORDERED_STAGE_NAMES = (
    "10-normalize",
    "20-lineage",
    "30-review-packet",
    "35-quality-review-evidence",
    "40-admission",
    "50-dedup",
    "55-greekmmlu-freeze",
    "60-decontamination",
    "65-anonymization-sanitization",
    "70-prestructural-freeze",
    "75-structural-detection-audit",
    "78-structural-apply",
    "80-final-validation",
)

AUTOMATIC_COMPACT_FILES = {
    "source_inventory": "source_inventory.json",
    "dedup_summary": "dedup_summary.json",
}

# These are the exact compact artifacts already closed by Stage 30/35/40.
# A local release is not a place to choose a convenient subset: it is the
# first point at which a private training tree and an Agent-3 handoff can be
# written, so all review, quality, admission, and execution evidence must be
# present together.  The waterfall and semantic-clearance inputs use dedicated
# flags because their relationship is a separate safety gate.
REQUIRED_SITE_EVIDENCE_NAMES = (
    "candidate_roster",
    "review_packet",
    "review_requests",
    "review_responses",
    "response_execution_receipt",
    "adjudication_execution_receipt",
    "stage35_review_closure",
    "review_sample_quality_summary",
    "review_sample_quality_handoff",
    "quality_summary",
    "lineage_summary",
    "source_novelty",
    "license_adjudication",
    "review_aggregate",
    "admission_confirmation",
)
REQUIRED_HANDOFF_EVIDENCE_NAMES = (
    *REQUIRED_SITE_EVIDENCE_NAMES,
    "transformation_waterfall",
    "anonymization_semantic_clearance",
)
DIRECT_EVIDENCE_NAMES = frozenset(
    {"transformation_waterfall", "anonymization_semantic_clearance"}
)

FORBIDDEN_HANDOFF_KEY_FRAGMENTS = (
    "protectedspan",
    "rawvalue",
    "benchmarkanswer",
    "answer",
    "answerkey",
    "correctanswer",
    "modelcheckpoint",
    "checkpointpath",
)


class ReleaseValidationError(ValueError):
    """Raised for a release contract violation without leaking corpus text."""


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"required JSON artifact is missing: {path}") from exc
    if not isinstance(value, dict):
        raise ReleaseValidationError(f"{path}: expected a JSON object")
    return value


def require_sha256(value: object, *, label: str) -> str:
    text = str(value or "")
    if not HEX_SHA256.fullmatch(text):
        raise ReleaseValidationError(f"{label}: expected lowercase SHA-256")
    return text


def require_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReleaseValidationError(f"{label}: expected non-empty text")
    return value


def resolve_existing_file(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink() or not resolved.stat().st_size:
        raise FileNotFoundError(f"{label}: required regular non-empty file is missing: {resolved}")
    return resolved


def file_binding(path: Path, *, include_path: bool = True) -> dict[str, Any]:
    resolved = resolve_existing_file(path, label="binding")
    value: dict[str, Any] = {
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
    if include_path:
        value["path"] = str(resolved)
    return value


def compact_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Keep a verification binding without exposing a scratch-path topology."""

    return {
        "bytes": int(binding["bytes"]),
        "sha256": require_sha256(binding["sha256"], label="compact binding"),
    }


def verify_binding(binding: object, *, label: str) -> dict[str, Any]:
    if not isinstance(binding, Mapping):
        raise ReleaseValidationError(f"{label}: binding must be an object")
    path_value = binding.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ReleaseValidationError(f"{label}: binding path is missing")
    path = resolve_existing_file(Path(path_value), label=label)
    expected_bytes = binding.get("bytes")
    expected_hash = require_sha256(binding.get("sha256"), label=label)
    if not isinstance(expected_bytes, int) or expected_bytes < 1:
        raise ReleaseValidationError(f"{label}: binding byte count is invalid")
    if path.stat().st_size != expected_bytes or sha256_file(path) != expected_hash:
        raise ReleaseValidationError(f"{label}: bound artifact drifted")
    return {"path": str(path), "bytes": expected_bytes, "sha256": expected_hash}


def safe_relative_path(value: object, *, label: str, suffix: str | None = None) -> Path:
    if not isinstance(value, str) or not value:
        raise ReleaseValidationError(f"{label}: expected a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ReleaseValidationError(f"{label}: unsafe relative path")
    if suffix is not None and path.suffix != suffix:
        raise ReleaseValidationError(f"{label}: expected a {suffix} file")
    return path


def _tree_files(root: Path, *, require_only_parquet: bool) -> list[Path]:
    """Return an immutable tree's regular files, rejecting stealth inputs."""

    resolved = root.resolve()
    if not resolved.is_dir() or resolved.is_symlink():
        raise FileNotFoundError(f"Parquet root is missing or unsafe: {resolved}")
    files: list[Path] = []
    for current, directories, names in os.walk(resolved, followlinks=False):
        current_path = Path(current)
        for directory in directories:
            if (current_path / directory).is_symlink():
                raise ReleaseValidationError(f"unsafe symlink in input tree: {current_path / directory}")
        for name in names:
            path = current_path / name
            if path.is_symlink():
                raise ReleaseValidationError(f"unsafe symlink in input tree: {path}")
            if name.startswith("."):
                raise ReleaseValidationError(f"partial or hidden file in immutable input tree: {path}")
            if require_only_parquet and path.suffix != ".parquet":
                raise ReleaseValidationError(f"non-Parquet file in corpus tree: {path}")
            if not path.is_file():
                raise ReleaseValidationError(f"non-regular file in input tree: {path}")
            files.append(path)
    files.sort()
    if not files:
        raise ReleaseValidationError(f"no Parquet files below {resolved}")
    return files


def corpus_parquet_files(root: Path) -> list[Path]:
    return _tree_files(root, require_only_parquet=True)


def parquet_input_files(value: Path) -> list[Path]:
    resolved = value.resolve()
    if resolved.is_file():
        if resolved.is_symlink() or resolved.suffix != ".parquet":
            raise ReleaseValidationError(f"expected a regular Parquet input: {resolved}")
        return [resolved]
    return corpus_parquet_files(resolved)


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


def parquet_input_binding(value: Path) -> dict[str, Any]:
    """Bind either one ledger shard or a whole immutable ledger tree.

    A directory has no single file hash, so its deterministic digest is over
    the ordered individual Parquet receipts.  It remains a compact binding and
    never copies the protected ledger itself into a release or site handoff.
    """

    resolved = value.resolve()
    files = parquet_input_files(resolved)
    relative_root = resolved if resolved.is_dir() else resolved.parent
    receipts = [parquet_receipt(path, relative_to=relative_root) for path in files]
    return {
        "path": str(resolved),
        "bytes": sum(int(item["bytes"]) for item in receipts),
        "sha256": canonical_json_sha256({"files": receipts}),
        "files": len(receipts),
    }


def verify_parquet_receipt(path: Path, receipt: Mapping[str, Any], *, label: str) -> None:
    import pyarrow.parquet as pq

    expected_bytes = receipt.get("bytes")
    expected_rows = receipt.get("rows")
    expected_groups = receipt.get("row_groups")
    expected_hash = require_sha256(receipt.get("sha256"), label=label)
    if not all(isinstance(value, int) and value >= 0 for value in (expected_bytes, expected_rows, expected_groups)):
        raise ReleaseValidationError(f"{label}: invalid Parquet receipt numbers")
    if not path.is_file() or path.is_symlink():
        raise ReleaseValidationError(f"{label}: receipted Parquet file is missing or unsafe")
    metadata = pq.ParquetFile(path).metadata
    if (
        path.stat().st_size != expected_bytes
        or int(metadata.num_rows) != expected_rows
        or int(metadata.num_row_groups) != expected_groups
        or sha256_file(path) != expected_hash
    ):
        raise ReleaseValidationError(f"{label}: receipted Parquet file drifted")


def write_json_no_replace(path: Path, value: Mapping[str, Any]) -> None:
    """Publish a small artifact only at a previously absent path.

    The caller writes inside a job-unique attempt directory.  ``link`` gives a
    no-replace publication primitive on the same filesystem; it avoids the
    legacy ``os.replace`` behavior that can overwrite an earlier receipt.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite immutable artifact: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def copy_file_no_replace(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable release file: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target, source.open("rb") as origin:
            shutil.copyfileobj(origin, target, length=8 * 1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite immutable release file: {destination}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def check(name: str, observed: Any) -> dict[str, Any]:
    return {"name": name, "passed": True, "observed": observed}


class ExactTokenizer:
    def __init__(self, path: Path) -> None:
        self.path = resolve_existing_file(path, label="tokenizer")
        self.binding = file_binding(self.path)
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - exercised on Clariden runtime
            raise RuntimeError("agent1 v3 release validation requires the tokenizers package") from exc
        self._tokenizer = Tokenizer.from_file(str(self.path))
        self._tokenizer.no_padding()
        self._tokenizer.no_truncation()

    def counts(self, texts: Sequence[str]) -> list[int]:
        if not texts:
            return []
        return [len(value.ids) for value in self._tokenizer.encode_batch(list(texts), add_special_tokens=False)]


def require_columns(names: Iterable[str], *, required: set[str], label: str) -> None:
    missing = required - set(names)
    if missing:
        raise ReleaseValidationError(f"{label}: missing required columns {sorted(missing)}")


def string_field(row: Mapping[str, Any], field: str, *, label: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ReleaseValidationError(f"{label}: row has an empty {field}")
    return value


def source_fields(row: Mapping[str, Any], *, label: str) -> tuple[str, str, str, str, str]:
    source_id = str(row.get("source_id") or row.get("acquisition_source_id") or "")
    acquisition = str(row.get("acquisition_source_id") or source_id)
    dataset = str(row.get("source_dataset") or "")
    repo_id = str(row.get("source_repo_id") or "")
    revision = str(row.get("source_revision") or "")
    if not all((source_id, acquisition, dataset, repo_id, revision)):
        raise ReleaseValidationError(f"{label}: finalization requires complete source provenance fields")
    return source_id, acquisition, dataset, repo_id, revision


def actual_text_hash(row: Mapping[str, Any], *, label: str) -> tuple[str, str]:
    text = row.get("text")
    if not isinstance(text, str):
        raise ReleaseValidationError(f"{label}: row text is not a string")
    digest = sha256_text(text)
    return text, digest


def hash_field(row: Mapping[str, Any], field: str, actual: str, *, label: str) -> None:
    value = string_field(row, field, label=label)
    if value != actual:
        raise ReleaseValidationError(f"{label}: {field} drift")


def open_database(path: Path) -> sqlite3.Connection:
    if path.exists():
        raise FileExistsError(f"refusing to reuse v3 release validation database: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.executescript(
        """
        CREATE TABLE pool (
            stable_uid TEXT PRIMARY KEY,
            input_representation_id TEXT NOT NULL UNIQUE,
            input_text_sha256 TEXT NOT NULL,
            source_id TEXT NOT NULL,
            acquisition_source_id TEXT NOT NULL,
            source_dataset TEXT NOT NULL,
            source_repo_id TEXT NOT NULL,
            source_revision TEXT NOT NULL,
            tokens INTEGER NOT NULL
        );
        CREATE TABLE dedup (
            stable_uid TEXT PRIMARY KEY,
            input_representation_id TEXT NOT NULL,
            input_text_sha256 TEXT NOT NULL,
            action TEXT NOT NULL
        );
        CREATE TABLE decontam (
            stable_uid TEXT PRIMARY KEY,
            representation_id TEXT NOT NULL,
            input_text_sha256 TEXT NOT NULL,
            action TEXT NOT NULL
        );
        CREATE TABLE anonymization_ledger (
            stable_uid TEXT PRIMARY KEY,
            input_text_sha256 TEXT NOT NULL,
            parent_representation_id TEXT NOT NULL,
            child_representation_id TEXT,
            output_text_sha256 TEXT,
            action TEXT NOT NULL
        );
        CREATE TABLE anonymized (
            stable_uid TEXT PRIMARY KEY,
            representation_id TEXT NOT NULL UNIQUE,
            input_representation_id TEXT NOT NULL,
            text_sha256 TEXT NOT NULL,
            parent_representation_id TEXT NOT NULL,
            parent_text_sha256 TEXT NOT NULL,
            source_id TEXT NOT NULL,
            acquisition_source_id TEXT NOT NULL,
            source_dataset TEXT NOT NULL,
            source_repo_id TEXT NOT NULL,
            source_revision TEXT NOT NULL,
            tokens INTEGER NOT NULL
        );
        CREATE TABLE final_rows (
            stable_uid TEXT PRIMARY KEY,
            representation_id TEXT NOT NULL UNIQUE,
            input_representation_id TEXT NOT NULL,
            text_sha256 TEXT NOT NULL,
            parent_representation_id TEXT NOT NULL,
            parent_text_sha256 TEXT NOT NULL,
            source_id TEXT NOT NULL,
            acquisition_source_id TEXT NOT NULL,
            source_dataset TEXT NOT NULL,
            source_repo_id TEXT NOT NULL,
            source_revision TEXT NOT NULL,
            tokens INTEGER NOT NULL
        );
        CREATE TABLE nodes (
            representation_id TEXT PRIMARY KEY,
            stable_uid TEXT NOT NULL,
            text_sha256 TEXT NOT NULL,
            parent_representation_id TEXT,
            parent_text_sha256 TEXT,
            is_dedup_root INTEGER NOT NULL CHECK(is_dedup_root IN (0, 1))
        );
        CREATE TABLE structural (
            stable_uid TEXT PRIMARY KEY,
            input_representation_id TEXT NOT NULL,
            action TEXT NOT NULL
        );
        CREATE INDEX idx_dedup_action ON dedup(action);
        CREATE INDEX idx_decontam_action ON decontam(action);
        CREATE INDEX idx_anon_action ON anonymization_ledger(action);
        CREATE INDEX idx_nodes_parent ON nodes(parent_representation_id);
        """
    )
    return connection


def insert_or_verify_node(
    connection: sqlite3.Connection,
    *,
    representation_id: str,
    stable_uid: str,
    text_sha256: str,
    parent_representation_id: str | None,
    parent_text_sha256: str | None,
    is_dedup_root: bool,
    label: str,
) -> None:
    expected = (
        stable_uid,
        text_sha256,
        parent_representation_id,
        parent_text_sha256,
        int(is_dedup_root),
    )
    current = connection.execute(
        "SELECT stable_uid, text_sha256, parent_representation_id, parent_text_sha256, is_dedup_root "
        "FROM nodes WHERE representation_id = ?",
        (representation_id,),
    ).fetchone()
    if current is None:
        connection.execute(
            "INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?)",
            (representation_id, *expected),
        )
    elif tuple(current) != expected:
        raise ReleaseValidationError(f"{label}: representation_id collision or lineage drift")


def stream_rows(paths: Sequence[Path], *, batch_rows: int) -> Iterator[tuple[Path, list[dict[str, Any]], set[str]]]:
    import pyarrow.parquet as pq

    for path in paths:
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        for batch in parquet.iter_batches(batch_size=batch_rows, use_threads=False):
            yield path, batch.to_pylist(), names


def scalar(connection: sqlite3.Connection, query: str, parameters: Sequence[object] = ()) -> int:
    return int(connection.execute(query, parameters).fetchone()[0])


def manifest_status(path: Path, *, schema: str, statuses: set[str], label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = resolve_existing_file(path, label=label)
    payload = read_object(resolved)
    if payload.get("schema_version") != schema:
        raise ReleaseValidationError(f"{label}: unsupported schema version")
    if payload.get("status") not in statuses:
        raise ReleaseValidationError(f"{label}: unacceptable status")
    return payload, file_binding(resolved)


def validate_receipt_path(
    root: Path,
    receipt: Mapping[str, Any],
    *,
    label: str,
) -> None:
    relative = safe_relative_path(receipt.get("path"), label=label, suffix=".parquet")
    target = (root / relative).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise ReleaseValidationError(f"{label}: receipt escapes declared root") from exc
    verify_parquet_receipt(target, receipt, label=label)


def scan_pool(
    connection: sqlite3.Connection,
    *,
    root: Path,
    tokenizer: ExactTokenizer,
    batch_rows: int,
) -> dict[str, int]:
    required = {
        "stable_uid",
        "input_representation_id",
        "text",
        "cleaned_text_sha256",
        "source_dataset",
        "acquisition_source_id",
        "source_repo_id",
        "source_revision",
    }
    counts: Counter[str] = Counter()
    for path, rows, names in stream_rows(corpus_parquet_files(root), batch_rows=batch_rows):
        require_columns(names, required=required, label=str(path))
        texts = [actual_text_hash(row, label=str(path))[0] for row in rows]
        token_counts = tokenizer.counts(texts)
        for row, text, tokens in zip(rows, texts, token_counts, strict=True):
            label = f"pre-MMLU pool row in {path.name}"
            stable_uid = string_field(row, "stable_uid", label=label)
            input_representation_id = string_field(row, "input_representation_id", label=label)
            text_hash = sha256_text(text)
            hash_field(row, "cleaned_text_sha256", text_hash, label=label)
            source_id, acquisition, dataset, repo_id, revision = source_fields(row, label=label)
            try:
                connection.execute(
                    "INSERT INTO pool VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        stable_uid,
                        input_representation_id,
                        text_hash,
                        source_id,
                        acquisition,
                        dataset,
                        repo_id,
                        revision,
                        int(tokens),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ReleaseValidationError("pre-MMLU pool has duplicate stable_uid or representation identity") from exc
            insert_or_verify_node(
                connection,
                representation_id=input_representation_id,
                stable_uid=stable_uid,
                text_sha256=text_hash,
                parent_representation_id=None,
                parent_text_sha256=None,
                is_dedup_root=True,
                label=label,
            )
            counts["rows"] += 1
            counts["tokens"] += int(tokens)
        connection.commit()
    if not counts["rows"]:
        raise ReleaseValidationError("pre-MMLU dedup pool is empty")
    return dict(counts)


def scan_dedup_ledger(
    connection: sqlite3.Connection,
    *,
    ledger: Path,
    manifest: Mapping[str, Any],
    batch_rows: int,
) -> dict[str, int]:
    required = {"stable_uid", "input_representation_id", "input_text_sha256", "action"}
    counts: Counter[str] = Counter()
    for path, rows, names in stream_rows(parquet_input_files(ledger), batch_rows=batch_rows):
        require_columns(names, required=required, label=str(path))
        for row in rows:
            label = f"dedup ledger row in {path.name}"
            action = string_field(row, "action", label=label)
            if action not in {"keep", "drop"}:
                raise ReleaseValidationError(f"{label}: unsupported dedup action")
            try:
                connection.execute(
                    "INSERT INTO dedup VALUES (?, ?, ?, ?)",
                    (
                        string_field(row, "stable_uid", label=label),
                        string_field(row, "input_representation_id", label=label),
                        require_sha256(row.get("input_text_sha256"), label=label),
                        action,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ReleaseValidationError("dedup ledger has duplicate stable_uid") from exc
            counts["rows"] += 1
            counts[action] += 1
        connection.commit()
    missing_decisions = scalar(
        connection,
        "SELECT count(*) FROM pool p LEFT JOIN dedup d USING(stable_uid) WHERE d.stable_uid IS NULL",
    )
    extra_decisions = scalar(
        connection,
        "SELECT count(*) FROM dedup d LEFT JOIN pool p USING(stable_uid) WHERE p.stable_uid IS NULL",
    )
    drift = scalar(
        connection,
        "SELECT count(*) FROM dedup d JOIN pool p USING(stable_uid) "
        "WHERE d.input_representation_id != p.input_representation_id "
        "OR d.input_text_sha256 != p.input_text_sha256",
    )
    if missing_decisions or extra_decisions or drift:
        raise ReleaseValidationError(
            "dedup decisions are not a complete content-bound pre-GreekMMLU ledger: "
            f"missing={missing_decisions}, extra={extra_decisions}, drift={drift}"
        )
    ledger_receipt = manifest.get("ledger")
    if not isinstance(ledger_receipt, Mapping):
        raise ReleaseValidationError("dedup manifest lacks its authoritative ledger receipt")
    expected_hash = require_sha256(ledger_receipt.get("sha256"), label="dedup manifest ledger")
    if sha256_file(ledger.resolve()) != expected_hash:
        raise ReleaseValidationError("dedup ledger differs from its immutable manifest")
    declared = manifest.get("counts")
    if not isinstance(declared, Mapping):
        raise ReleaseValidationError("dedup manifest lacks counts")
    expected_counts = {
        "ledger_rows": counts["rows"],
        "kept_rows": counts["keep"],
        "dropped_rows": counts["drop"],
    }
    for key, expected in expected_counts.items():
        if int(declared.get(key, -1)) != expected:
            raise ReleaseValidationError(f"dedup manifest count drift for {key}")
    return dict(counts)


def scan_decontamination_ledger(
    connection: sqlite3.Connection,
    *,
    ledger_root: Path,
    manifest: Mapping[str, Any],
    batch_rows: int,
) -> dict[str, int]:
    required = {"stable_uid", "representation_id", "input_text_sha256", "action"}
    counts: Counter[str] = Counter()
    for path, rows, names in stream_rows(corpus_parquet_files(ledger_root), batch_rows=batch_rows):
        require_columns(names, required=required, label=str(path))
        for row in rows:
            label = f"GreekMMLU ledger row in {path.name}"
            action = string_field(row, "action", label=label)
            if action not in {"keep", "drop", "quarantine"}:
                raise ReleaseValidationError(f"{label}: unsupported GreekMMLU action")
            try:
                connection.execute(
                    "INSERT INTO decontam VALUES (?, ?, ?, ?)",
                    (
                        string_field(row, "stable_uid", label=label),
                        string_field(row, "representation_id", label=label),
                        require_sha256(row.get("input_text_sha256"), label=label),
                        action,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ReleaseValidationError("GreekMMLU ledger has duplicate stable_uid") from exc
            counts["rows"] += 1
            counts[action] += 1
        connection.commit()
    missing = scalar(
        connection,
        "SELECT count(*) FROM dedup d LEFT JOIN decontam g USING(stable_uid) "
        "WHERE d.action = 'keep' AND g.stable_uid IS NULL",
    )
    extra = scalar(
        connection,
        "SELECT count(*) FROM decontam g LEFT JOIN dedup d USING(stable_uid) "
        "WHERE d.stable_uid IS NULL OR d.action != 'keep'",
    )
    drift = scalar(
        connection,
        "SELECT count(*) FROM decontam g JOIN dedup d USING(stable_uid) "
        "WHERE g.representation_id != d.input_representation_id "
        "OR g.input_text_sha256 != d.input_text_sha256",
    )
    if missing or extra or drift:
        raise ReleaseValidationError(
            "GreekMMLU ledger does not close over pre-MMLU dedup survivors: "
            f"missing={missing}, extra={extra}, drift={drift}"
        )
    policy = manifest.get("policy")
    if not isinstance(policy, Mapping):
        raise ReleaseValidationError("GreekMMLU manifest lacks policy")
    if (
        policy.get("high_confidence_actions") != "drop"
        or policy.get("ambiguous_match_actions") != "quarantine"
        or policy.get("answer_only_action") != "audit_only"
    ):
        raise ReleaseValidationError("GreekMMLU manifest has an unsafe matching policy")
    declared = manifest.get("counts")
    if not isinstance(declared, Mapping):
        raise ReleaseValidationError("GreekMMLU manifest lacks counts")
    for key in ("input", "keep", "drop", "quarantine"):
        expected = counts["rows"] if key == "input" else counts[key]
        if int(declared.get(key, -1)) != expected:
            raise ReleaseValidationError(f"GreekMMLU manifest count drift for {key}")
    return dict(counts)


def validate_decontamination_ledger_receipts(
    manifest: Mapping[str, Any], *, ledger_root: Path
) -> None:
    """Bind the supplied Stage 60 ledger tree to the manifest's shard receipts."""

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ReleaseValidationError("GreekMMLU manifest lacks ledger file receipts")
    seen: set[str] = set()
    for row in files:
        if not isinstance(row, Mapping):
            raise ReleaseValidationError("GreekMMLU manifest file receipt is invalid")
        receipt = row.get("ledger")
        if not isinstance(receipt, Mapping):
            raise ReleaseValidationError("GreekMMLU manifest lacks a ledger receipt")
        relative = safe_relative_path(receipt.get("path"), label="GreekMMLU ledger", suffix=".parquet")
        if relative.as_posix() in seen:
            raise ReleaseValidationError("GreekMMLU manifest repeats a ledger receipt")
        seen.add(relative.as_posix())
        validate_receipt_path(ledger_root, receipt, label="GreekMMLU ledger")
    actual = {
        path.relative_to(ledger_root.resolve()).as_posix()
        for path in corpus_parquet_files(ledger_root)
    }
    if actual != seen:
        raise ReleaseValidationError("GreekMMLU ledger manifest inventory is not exact")


def scan_anonymization_ledger(
    connection: sqlite3.Connection,
    *,
    ledger_root: Path,
    manifest: Mapping[str, Any],
    batch_rows: int,
) -> dict[str, int]:
    required = {
        "stable_uid",
        "input_text_sha256",
        "parent_representation_id",
        "child_representation_id",
        "output_text_sha256",
        "action",
    }
    counts: Counter[str] = Counter()
    for path, rows, names in stream_rows(corpus_parquet_files(ledger_root), batch_rows=batch_rows):
        require_columns(names, required=required, label=str(path))
        for row in rows:
            label = f"anonymization ledger row in {path.name}"
            action = string_field(row, "action", label=label)
            if action not in {"keep", "drop", "quarantine"}:
                raise ReleaseValidationError(f"{label}: unsupported anonymization action")
            child = row.get("child_representation_id")
            output_hash = row.get("output_text_sha256")
            if action == "drop":
                if child is not None or output_hash is not None:
                    raise ReleaseValidationError(f"{label}: dropped row must not advertise a child representation")
            else:
                child = require_text(child, label=f"{label} child representation")
                output_hash = require_sha256(output_hash, label=label)
            try:
                connection.execute(
                    "INSERT INTO anonymization_ledger VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        string_field(row, "stable_uid", label=label),
                        require_sha256(row.get("input_text_sha256"), label=label),
                        string_field(row, "parent_representation_id", label=label),
                        child,
                        output_hash,
                        action,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ReleaseValidationError("anonymization protected ledger has duplicate stable_uid") from exc
            counts["rows"] += 1
            counts[action] += 1
        connection.commit()
    missing = scalar(
        connection,
        "SELECT count(*) FROM decontam g LEFT JOIN anonymization_ledger a USING(stable_uid) "
        "WHERE g.action = 'keep' AND a.stable_uid IS NULL",
    )
    extra = scalar(
        connection,
        "SELECT count(*) FROM anonymization_ledger a LEFT JOIN decontam g USING(stable_uid) "
        "WHERE g.stable_uid IS NULL OR g.action != 'keep'",
    )
    drift = scalar(
        connection,
        "SELECT count(*) FROM anonymization_ledger a JOIN decontam g USING(stable_uid) "
        "WHERE a.input_text_sha256 != g.input_text_sha256 "
        "OR a.parent_representation_id != g.representation_id",
    )
    if missing or extra or drift:
        raise ReleaseValidationError(
            "anonymization protected ledger does not close over GreekMMLU survivors: "
            f"missing={missing}, extra={extra}, drift={drift}"
        )
    protected = manifest.get("protected_ledger")
    if not isinstance(protected, Mapping) or protected.get("public_training_output") is not False:
        raise ReleaseValidationError("anonymization manifest does not protect its private ledger")
    path_value = protected.get("path")
    if not isinstance(path_value, str) or Path(path_value).resolve() != ledger_root.resolve():
        raise ReleaseValidationError("anonymization manifest protected-ledger root drift")
    declared = manifest.get("counts")
    if not isinstance(declared, Mapping):
        raise ReleaseValidationError("anonymization manifest lacks counts")
    if int(declared.get("input_rows", -1)) != counts["rows"]:
        raise ReleaseValidationError("anonymization manifest input count drift")
    for action in ("keep", "drop", "quarantine"):
        if int(declared.get(f"action:{action}", -1)) != counts[action]:
            raise ReleaseValidationError(f"anonymization manifest action count drift for {action}")
    return dict(counts)


def validate_anonymization_output_receipts(
    manifest: Mapping[str, Any],
    *,
    output_root: Path,
    ledger_root: Path,
) -> None:
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ReleaseValidationError("anonymization manifest lacks output file receipts")
    seen_output: set[str] = set()
    seen_ledger: set[str] = set()
    for row in files:
        if not isinstance(row, Mapping):
            raise ReleaseValidationError("anonymization manifest file receipt is invalid")
        for field, root, seen in (("output", output_root, seen_output), ("protected_ledger", ledger_root, seen_ledger)):
            receipt = row.get(field)
            if not isinstance(receipt, Mapping):
                raise ReleaseValidationError(f"anonymization manifest lacks {field} receipt")
            relative = safe_relative_path(receipt.get("path"), label=f"anonymization {field}", suffix=".parquet")
            if relative.as_posix() in seen:
                raise ReleaseValidationError(f"anonymization manifest repeats {field} receipt")
            seen.add(relative.as_posix())
            validate_receipt_path(root, receipt, label=f"anonymization {field}")
    actual_output = {path.relative_to(output_root.resolve()).as_posix() for path in corpus_parquet_files(output_root)}
    actual_ledger = {path.relative_to(ledger_root.resolve()).as_posix() for path in corpus_parquet_files(ledger_root)}
    if actual_output != seen_output or actual_ledger != seen_ledger:
        raise ReleaseValidationError("anonymization manifest inventory does not exactly cover output or protected ledger")


def generic_row(
    row: Mapping[str, Any],
    *,
    label: str,
) -> tuple[str, str, str, str, str, str, tuple[str, str, str, str, str]]:
    stable_uid = string_field(row, "stable_uid", label=label)
    input_representation_id = string_field(row, "input_representation_id", label=label)
    representation_id = string_field(row, "representation_id", label=label)
    parent_representation_id = string_field(row, "parent_representation_id", label=label)
    parent_text_sha256 = require_sha256(row.get("parent_text_sha256"), label=label)
    _, text_hash = actual_text_hash(row, label=label)
    hash_field(row, "text_sha256", text_hash, label=label)
    hash_field(row, "cleaned_text_sha256", text_hash, label=label)
    sources = source_fields(row, label=label)
    return (
        stable_uid,
        input_representation_id,
        representation_id,
        parent_representation_id,
        parent_text_sha256,
        text_hash,
        sources,
    )


def scan_anonymized_output(
    connection: sqlite3.Connection,
    *,
    root: Path,
    tokenizer: ExactTokenizer,
    batch_rows: int,
) -> dict[str, int]:
    required = {
        "stable_uid",
        "input_representation_id",
        "representation_id",
        "parent_representation_id",
        "parent_text_sha256",
        "text",
        "text_sha256",
        "cleaned_text_sha256",
        "anonymization_action",
        "source_dataset",
        "acquisition_source_id",
        "source_repo_id",
        "source_revision",
    }
    counts: Counter[str] = Counter()
    for path, rows, names in stream_rows(corpus_parquet_files(root), batch_rows=batch_rows):
        require_columns(names, required=required, label=str(path))
        texts = [actual_text_hash(row, label=str(path))[0] for row in rows]
        token_counts = tokenizer.counts(texts)
        for row, tokens in zip(rows, token_counts, strict=True):
            label = f"anonymized output row in {path.name}"
            if row.get("anonymization_action") != "keep":
                raise ReleaseValidationError(f"{label}: output root contains non-keep anonymization action")
            uid, input_id, representation, parent, parent_hash, text_hash, sources = generic_row(row, label=label)
            ledger = connection.execute(
                "SELECT parent_representation_id, child_representation_id, output_text_sha256, action "
                "FROM anonymization_ledger WHERE stable_uid = ?",
                (uid,),
            ).fetchone()
            if ledger is None or tuple(ledger) != (parent, representation, text_hash, "keep"):
                raise ReleaseValidationError(f"{label}: no matching keep action in protected anonymization ledger")
            source_id, acquisition, dataset, repo_id, revision = sources
            try:
                connection.execute(
                    "INSERT INTO anonymized VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        uid,
                        representation,
                        input_id,
                        text_hash,
                        parent,
                        parent_hash,
                        source_id,
                        acquisition,
                        dataset,
                        repo_id,
                        revision,
                        int(tokens),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ReleaseValidationError("anonymized output has duplicate stable_uid or representation identity") from exc
            insert_or_verify_node(
                connection,
                representation_id=representation,
                stable_uid=uid,
                text_sha256=text_hash,
                parent_representation_id=parent,
                parent_text_sha256=parent_hash,
                is_dedup_root=False,
                label=label,
            )
            counts["rows"] += 1
            counts["tokens"] += int(tokens)
        connection.commit()
    missing_output = scalar(
        connection,
        "SELECT count(*) FROM anonymization_ledger a LEFT JOIN anonymized o USING(stable_uid) "
        "WHERE a.action = 'keep' AND o.stable_uid IS NULL",
    )
    extra_output = scalar(
        connection,
        "SELECT count(*) FROM anonymized o LEFT JOIN anonymization_ledger a USING(stable_uid) "
        "WHERE a.stable_uid IS NULL OR a.action != 'keep'",
    )
    if missing_output or extra_output:
        raise ReleaseValidationError(
            "anonymized output does not close over protected ledger keeps: "
            f"missing={missing_output}, extra={extra_output}"
        )
    return dict(counts)


def scan_final_output(
    connection: sqlite3.Connection,
    *,
    root: Path,
    tokenizer: ExactTokenizer,
    batch_rows: int,
) -> dict[str, int]:
    required = {
        "stable_uid",
        "input_representation_id",
        "representation_id",
        "parent_representation_id",
        "parent_text_sha256",
        "text",
        "text_sha256",
        "cleaned_text_sha256",
        "anonymization_action",
        "source_dataset",
        "acquisition_source_id",
        "source_repo_id",
        "source_revision",
    }
    counts: Counter[str] = Counter()
    for path, rows, names in stream_rows(corpus_parquet_files(root), batch_rows=batch_rows):
        require_columns(names, required=required, label=str(path))
        texts = [actual_text_hash(row, label=str(path))[0] for row in rows]
        token_counts = tokenizer.counts(texts)
        for row, tokens in zip(rows, token_counts, strict=True):
            label = f"final structural output row in {path.name}"
            if row.get("anonymization_action") != "keep":
                raise ReleaseValidationError(f"{label}: final tree lost or changed anonymization keep state")
            uid, input_id, representation, parent, parent_hash, text_hash, sources = generic_row(row, label=label)
            source_id, acquisition, dataset, repo_id, revision = sources
            try:
                connection.execute(
                    "INSERT INTO final_rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        uid,
                        representation,
                        input_id,
                        text_hash,
                        parent,
                        parent_hash,
                        source_id,
                        acquisition,
                        dataset,
                        repo_id,
                        revision,
                        int(tokens),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ReleaseValidationError("final structural output has duplicate stable_uid or representation identity") from exc
            insert_or_verify_node(
                connection,
                representation_id=representation,
                stable_uid=uid,
                text_sha256=text_hash,
                parent_representation_id=parent,
                parent_text_sha256=parent_hash,
                is_dedup_root=False,
                label=label,
            )
            counts["rows"] += 1
            counts["tokens"] += int(tokens)
        connection.commit()
    if not counts["rows"]:
        raise ReleaseValidationError("final structural output is empty")
    missing_pre_mmlu = scalar(
        connection,
        "SELECT count(*) FROM final_rows f LEFT JOIN dedup d USING(stable_uid) "
        "LEFT JOIN pool p USING(stable_uid) "
        "WHERE d.stable_uid IS NULL OR d.action != 'keep' OR p.stable_uid IS NULL "
        "OR f.input_representation_id != d.input_representation_id "
        "OR f.input_representation_id != p.input_representation_id",
    )
    provenance_drift = scalar(
        connection,
        "SELECT count(*) FROM final_rows f JOIN pool p USING(stable_uid) "
        "WHERE f.source_id != p.source_id OR f.acquisition_source_id != p.acquisition_source_id "
        "OR f.source_dataset != p.source_dataset OR f.source_repo_id != p.source_repo_id "
        "OR f.source_revision != p.source_revision",
    )
    if missing_pre_mmlu or provenance_drift:
        raise ReleaseValidationError(
            "final structural output is not traceable to kept pre-MMLU dedup representations: "
            f"missing_pre_mmlu={missing_pre_mmlu}, provenance_drift={provenance_drift}"
        )
    return dict(counts)


def validate_generic_lineage(connection: sqlite3.Connection) -> dict[str, int]:
    invalid_roots = scalar(
        connection,
        "SELECT count(*) FROM nodes WHERE parent_representation_id IS NULL AND is_dedup_root != 1",
    )
    invalid_edges = scalar(
        connection,
        "SELECT count(*) FROM nodes n LEFT JOIN nodes p "
        "ON p.representation_id = n.parent_representation_id "
        "WHERE n.parent_representation_id IS NOT NULL AND (p.representation_id IS NULL "
        "OR p.stable_uid != n.stable_uid OR p.text_sha256 != n.parent_text_sha256)",
    )
    if invalid_roots or invalid_edges:
        raise ReleaseValidationError(
            "generic representation lineage has an unbound root or parent edge: "
            f"invalid_roots={invalid_roots}, invalid_edges={invalid_edges}"
        )
    connection.execute("DROP TABLE IF EXISTS temp.lineage_frontier")
    connection.execute(
        "CREATE TEMP TABLE lineage_frontier ("
        "final_uid TEXT NOT NULL, representation_id TEXT NOT NULL, depth INTEGER NOT NULL, "
        "PRIMARY KEY(final_uid, representation_id))"
    )
    connection.execute(
        "INSERT INTO lineage_frontier SELECT stable_uid, representation_id, 0 FROM final_rows"
    )
    max_depth = 0
    for depth in range(MAX_LINEAGE_DEPTH + 1):
        uid_drift = scalar(
            connection,
            "SELECT count(*) FROM lineage_frontier f JOIN nodes n USING(representation_id) "
            "WHERE f.depth = ? AND f.final_uid != n.stable_uid",
            (depth,),
        )
        if uid_drift:
            raise ReleaseValidationError("generic representation lineage changes stable document identity")
        terminal_drift = scalar(
            connection,
            "SELECT count(*) FROM lineage_frontier f JOIN nodes n USING(representation_id) "
            "JOIN final_rows r ON r.stable_uid = f.final_uid "
            "WHERE f.depth = ? AND n.parent_representation_id IS NULL "
            "AND n.representation_id != r.input_representation_id",
            (depth,),
        )
        if terminal_drift:
            raise ReleaseValidationError("final representation lineage does not terminate at its dedup input representation")
        candidates = scalar(
            connection,
            "SELECT count(*) FROM lineage_frontier f JOIN nodes n USING(representation_id) "
            "WHERE f.depth = ? AND n.parent_representation_id IS NOT NULL",
            (depth,),
        )
        if not candidates:
            max_depth = max(max_depth, depth)
            break
        if depth >= MAX_LINEAGE_DEPTH:
            raise ReleaseValidationError("generic representation lineage exceeds maximum safe depth")
        try:
            connection.execute(
                "INSERT INTO lineage_frontier "
                "SELECT f.final_uid, n.parent_representation_id, f.depth + 1 "
                "FROM lineage_frontier f JOIN nodes n USING(representation_id) "
                "WHERE f.depth = ? AND n.parent_representation_id IS NOT NULL",
                (depth,),
            )
        except sqlite3.IntegrityError as exc:
            raise ReleaseValidationError("generic representation lineage has a cycle") from exc
        max_depth = depth + 1
    else:  # pragma: no cover - defensive loop guard
        raise ReleaseValidationError("generic representation lineage did not converge")
    return {
        "nodes": scalar(connection, "SELECT count(*) FROM nodes"),
        "dedup_roots": scalar(connection, "SELECT count(*) FROM nodes WHERE is_dedup_root = 1"),
        "final_rows": scalar(connection, "SELECT count(*) FROM final_rows"),
        "max_depth": max_depth,
    }


def _manifest_path_value(payload: Mapping[str, Any], *names: str) -> Path:
    for name in names:
        value = payload.get(name)
        if isinstance(value, str) and value:
            return Path(value).resolve()
    raise ReleaseValidationError(f"structural manifest lacks one of {names}")


def truthy_gate(payload: Mapping[str, Any], field: str) -> bool:
    if payload.get(field) is True:
        return True
    gates = payload.get("safety_gates")
    return isinstance(gates, Mapping) and gates.get(field) is True


def scan_structural_ledger(
    connection: sqlite3.Connection,
    *,
    ledger: Path,
    batch_rows: int,
) -> dict[str, int]:
    required = {"stable_uid", "input_representation_id", "action"}
    counts: Counter[str] = Counter()
    for path, rows, names in stream_rows(parquet_input_files(ledger), batch_rows=batch_rows):
        require_columns(names, required=required, label=str(path))
        for row in rows:
            label = f"structural action ledger row in {path.name}"
            action = string_field(row, "action", label=label)
            if action not in {"keep", "drop", "quarantine"}:
                raise ReleaseValidationError(f"{label}: unsupported structural action")
            try:
                connection.execute(
                    "INSERT INTO structural VALUES (?, ?, ?)",
                    (
                        string_field(row, "stable_uid", label=label),
                        string_field(row, "input_representation_id", label=label),
                        action,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ReleaseValidationError("structural action ledger has duplicate stable_uid") from exc
            counts["rows"] += 1
            counts[action] += 1
        connection.commit()
    missing = scalar(
        connection,
        "SELECT count(*) FROM anonymized a LEFT JOIN structural s USING(stable_uid) WHERE s.stable_uid IS NULL",
    )
    extra = scalar(
        connection,
        "SELECT count(*) FROM structural s LEFT JOIN anonymized a USING(stable_uid) WHERE a.stable_uid IS NULL",
    )
    drift = scalar(
        connection,
        "SELECT count(*) FROM structural s JOIN anonymized a USING(stable_uid) "
        "WHERE s.input_representation_id != a.input_representation_id",
    )
    final_missing = scalar(
        connection,
        "SELECT count(*) FROM structural s LEFT JOIN final_rows f USING(stable_uid) "
        "WHERE s.action = 'keep' AND f.stable_uid IS NULL",
    )
    final_extra = scalar(
        connection,
        "SELECT count(*) FROM final_rows f LEFT JOIN structural s USING(stable_uid) "
        "WHERE s.stable_uid IS NULL OR s.action != 'keep'",
    )
    if missing or extra or drift or final_missing or final_extra:
        raise ReleaseValidationError(
            "structural action ledger does not close over anonymized representations: "
            f"missing={missing}, extra={extra}, drift={drift}, "
            f"final_missing={final_missing}, final_extra={final_extra}"
        )
    return dict(counts)


def validate_structural_manifest(
    connection: sqlite3.Connection,
    *,
    path: Path,
    anonymized_root: Path,
    final_root: Path,
    structural_ledger: Path | None,
    batch_rows: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int] | None]:
    resolved = resolve_existing_file(path, label="structural manifest")
    payload = read_object(resolved)
    schema = str(payload.get("schema_version") or "")
    if not schema.startswith("agent1_full_corpus_v3_structural_"):
        raise ReleaseValidationError("structural manifest is not an Agent 1 v3 structural application receipt")
    if payload.get("status") not in {"passed", "completed"}:
        raise ReleaseValidationError("structural manifest is not completed")
    mode = payload.get("mode")
    if mode not in {"no_op", "applied"}:
        raise ReleaseValidationError("structural manifest mode must be no_op or applied")
    input_root = _manifest_path_value(payload, "input_root", "input", "corpus_input")
    output_root = _manifest_path_value(payload, "output_root", "output", "corpus_output")
    if input_root != anonymized_root.resolve() or output_root != final_root.resolve():
        raise ReleaseValidationError("structural manifest input/output roots do not bind the supplied v3 representations")
    if mode == "no_op":
        if not isinstance(payload.get("no_op_reason"), str) or not payload["no_op_reason"]:
            raise ReleaseValidationError("no-op structural manifest lacks an explicit reason")
        if final_root.resolve() != anonymized_root.resolve():
            raise ReleaseValidationError("no-op structural application must retain the exact anonymized root")
        if structural_ledger is not None:
            raise ReleaseValidationError("a no-op structural application must not smuggle a structural ledger")
        return payload, file_binding(resolved), None

    required_gates = (
        "ready_for_application",
        "python_rust_probability_parity_passed",
        "python_rust_decoded_span_parity_passed",
        "source_balanced_safety_metrics_passed",
        "false_deletion_audit_passed",
    )
    missing = [field for field in required_gates if not truthy_gate(payload, field)]
    if missing:
        raise ReleaseValidationError(f"structural application lacks required safety gates: {missing}")
    model_handoff = payload.get("model_handoff")
    if not isinstance(model_handoff, Mapping):
        raise ReleaseValidationError("structural application lacks immutable model-handoff binding")
    verify_binding(model_handoff, label="structural model handoff")
    if structural_ledger is None:
        raise ReleaseValidationError("applied structural output requires --structural-ledger for closure validation")
    return payload, file_binding(resolved), scan_structural_ledger(
        connection, ledger=structural_ledger, batch_rows=batch_rows
    )


def contract_digest(contract: Mapping[str, Any]) -> str:
    value = dict(contract)
    value.pop("created_at", None)
    value.pop("contract_sha256", None)
    return canonical_json_sha256(value)


def validate_run_contract(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = resolve_existing_file(path, label="run contract")
    payload = read_object(resolved)
    if payload.get("schema_version") != RUN_CONTRACT_SCHEMA:
        raise ReleaseValidationError("run contract has an unsupported schema")
    if payload.get("prestructural_only") is True:
        raise ReleaseValidationError("a prestructural-only run cannot create a Stage 80 release")
    if payload.get("contract_sha256") != contract_digest(payload):
        raise ReleaseValidationError("run contract digest drift")
    graph = payload.get("stage_graph")
    if not isinstance(graph, list):
        raise ReleaseValidationError("run contract lacks stage graph")
    required = ("50-dedup", "55-greekmmlu-freeze", "60-decontamination", "65-anonymization-sanitization", "78-structural-apply", "80-final-validation")
    try:
        positions = [graph.index(stage) for stage in required]
    except ValueError as exc:
        raise ReleaseValidationError("run contract omits required ordered v3 stages") from exc
    if positions != sorted(positions):
        raise ReleaseValidationError("run contract violates required dedup → decontam → anonymize → structural order")
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ReleaseValidationError("run contract lacks frozen input bindings")
    for name, binding in inputs.items():
        verify_binding(binding, label=f"run contract input {name}")
    return payload, file_binding(resolved)


def validate_stage_receipt(path: Path, *, contract: Mapping[str, Any]) -> dict[str, Any]:
    resolved = resolve_existing_file(path, label="stage receipt")
    payload = read_object(resolved)
    expected = {
        "schema_version": STAGE_RECEIPT_SCHEMA,
        "status": "passed",
        "run_id": contract.get("run_id"),
        "code_commit": contract.get("code_commit"),
        "run_contract_sha256": contract.get("contract_sha256"),
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ReleaseValidationError(f"stage receipt {resolved}: {field} drift")
    stage = payload.get("stage")
    if stage not in ORDERED_STAGE_NAMES:
        raise ReleaseValidationError("stage receipt has an unknown stage")
    outputs = payload.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ReleaseValidationError("stage receipt lacks output bindings")
    for index, binding in enumerate(outputs):
        verify_binding(binding, label=f"stage receipt output {stage}:{index}")
    return {"stage": stage, **file_binding(resolved)}


def parse_site_inputs(values: Sequence[Sequence[str]]) -> list[tuple[str, Path]]:
    seen: set[str] = set(AUTOMATIC_COMPACT_FILES) | set(DIRECT_EVIDENCE_NAMES)
    result: list[tuple[str, Path]] = []
    for pair in values:
        if len(pair) != 2:
            raise ReleaseValidationError("--site-input needs NAME PATH")
        name, raw_path = pair
        if not SAFE_SITE_INPUT_NAME.fullmatch(name) or name in seen:
            raise ReleaseValidationError("site-input names must be unique, safe, and not reserved")
        path = resolve_existing_file(Path(raw_path), label=f"site input {name}")
        if path.suffix not in {".json", ".jsonl"} or path.stat().st_size > MAX_COMPACT_INPUT_BYTES:
            raise ReleaseValidationError("site input must be a compact JSON or JSONL artifact")
        validate_compact_json(path)
        seen.add(name)
        result.append((name, path))
    return result


def _key_is_forbidden(key: object) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", str(key).casefold())
    return any(fragment in normalized for fragment in FORBIDDEN_HANDOFF_KEY_FRAGMENTS)


def validate_compact_value(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if _key_is_forbidden(key):
                raise ReleaseValidationError("compact site input contains a protected or benchmark-answer field")
            validate_compact_value(nested)
    elif isinstance(value, list):
        for nested in value:
            validate_compact_value(nested)


def validate_compact_json(path: Path) -> None:
    if path.suffix == ".json":
        validate_compact_value(json.loads(path.read_text(encoding="utf-8")))
        return
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    validate_compact_value(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ReleaseValidationError(f"compact JSONL site input is invalid at line {number}") from exc


def _status_is_pending_or_omitted(value: object) -> bool:
    """Return whether a status string advertises an incomplete handoff.

    We deliberately inspect terminal *artifact* statuses rather than arbitrary
    nested prose.  In particular, the immutable transformation waterfall
    truthfully records that its automatic semantic audit was initially
    pending; a separate, hash-bound clearance artifact closes that one known
    historical state before release materialization.
    """

    if not isinstance(value, str):
        return False
    normalized = re.sub(r"[^a-z0-9]", "", value.casefold())
    return "pending" in normalized or "notincluded" in normalized


def require_terminal_status(
    payload: Mapping[str, Any],
    *,
    label: str,
    accepted: set[str],
) -> str:
    status = payload.get("status")
    if _status_is_pending_or_omitted(status):
        raise ReleaseValidationError(f"{label}: pending/not_included evidence is not releasable")
    if status not in accepted:
        raise ReleaseValidationError(f"{label}: unsupported terminal status")
    return str(status)


def _read_jsonl_objects(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReleaseValidationError(f"{label}: invalid JSONL row {number}") from exc
            if not isinstance(value, dict):
                raise ReleaseValidationError(f"{label}: JSONL row {number} is not an object")
            validate_compact_value(value)
            rows.append(value)
    if not rows:
        raise ReleaseValidationError(f"{label}: compact JSONL evidence is empty")
    return rows


def _assert_binding_matches_file(binding: object, path: Path, *, label: str) -> dict[str, Any]:
    """Verify a supplied receipt binds exactly the compact file at ``path``."""

    if not isinstance(binding, Mapping):
        raise ReleaseValidationError(f"{label}: expected an immutable file binding")
    actual = file_binding(path)
    if binding.get("bytes") != actual["bytes"] or binding.get("sha256") != actual["sha256"]:
        raise ReleaseValidationError(f"{label}: bytes/SHA-256 binding drift")
    return actual


def _require_json_schema(
    path: Path,
    *,
    label: str,
    schema: str,
    statuses: set[str] | None = None,
) -> dict[str, Any]:
    payload = read_object(path)
    validate_compact_value(payload)
    if payload.get("schema_version") != schema:
        raise ReleaseValidationError(f"{label}: unsupported schema")
    if statuses is not None:
        require_terminal_status(payload, label=label, accepted=statuses)
    elif _status_is_pending_or_omitted(payload.get("status")):
        raise ReleaseValidationError(f"{label}: pending/not_included evidence is not releasable")
    return payload


def _validate_response_rows(path: Path, *, name: str, schema: str) -> None:
    rows = _read_jsonl_objects(path, label=name)
    for index, row in enumerate(rows, start=1):
        if row.get("schema_version") != schema:
            raise ReleaseValidationError(f"{name}: row {index} has an unsupported schema")
        if _status_is_pending_or_omitted(row.get("status")):
            raise ReleaseValidationError(f"{name}: row {index} is pending/not_included")


def _validate_review_aggregate(
    payload: Mapping[str, Any],
    *,
    evidence_paths: Mapping[str, Path],
) -> None:
    require_terminal_status(
        payload,
        label="review aggregate",
        accepted={"passed_review_evidence_no_admission_decision"},
    )
    closure = payload.get("review_closure")
    if not isinstance(closure, Mapping) or closure.get("status") != "complete" or closure.get("pending_count") != 0:
        raise ReleaseValidationError("review aggregate: review/adjudication closure is incomplete")
    aggregate_inputs = payload.get("inputs")
    expected_names = set(REQUIRED_SITE_EVIDENCE_NAMES) - {"review_aggregate", "admission_confirmation"}
    if not isinstance(aggregate_inputs, Mapping) or set(aggregate_inputs) != expected_names:
        raise ReleaseValidationError("review aggregate: immutable input inventory drift")
    for name in sorted(expected_names):
        _assert_binding_matches_file(
            aggregate_inputs.get(name), evidence_paths[name], label=f"review aggregate input {name}"
        )


def _validate_admission_confirmation(
    payload: Mapping[str, Any],
    *,
    aggregate_path: Path,
) -> None:
    require_terminal_status(payload, label="admission confirmation", accepted={"approved"})
    if payload.get("confirmation_mode") != "explicit_hash_confirmed_user_confirmation":
        raise ReleaseValidationError("admission confirmation lacks explicit user hash confirmation")
    packet_binding = payload.get("packet")
    if not isinstance(packet_binding, Mapping):
        raise ReleaseValidationError("admission confirmation lacks its confirmed admission packet")
    packet_path_raw = packet_binding.get("path")
    if not isinstance(packet_path_raw, str) or not packet_path_raw:
        raise ReleaseValidationError("admission confirmation packet path is missing")
    packet_path = resolve_existing_file(Path(packet_path_raw), label="admission confirmation packet")
    _assert_binding_matches_file(packet_binding, packet_path, label="admission confirmation packet")
    packet = read_object(packet_path)
    # A packet is necessarily created in a pending state before the user
    # confirms its exact bytes.  It is not a releasable artifact on its own;
    # only the surrounding approved confirmation may carry it forward.
    if (
        packet.get("schema_version") != "agent1_full_corpus_v3_source_admission_packet_v2"
        or packet.get("status") != "pending_user_confirmation"
    ):
        raise ReleaseValidationError("admission confirmation does not bind the expected historical packet")
    if payload.get("user_confirmed_packet_sha256") != sha256_file(packet_path):
        raise ReleaseValidationError("admission confirmation user hash does not bind exact packet bytes")
    packet_inputs = packet.get("inputs")
    if not isinstance(packet_inputs, Mapping):
        raise ReleaseValidationError("confirmed admission packet lacks immutable inputs")
    _assert_binding_matches_file(
        packet_inputs.get("review_aggregate"), aggregate_path, label="confirmed admission packet review aggregate"
    )
    if packet.get("review_aggregate_sha256") != sha256_file(aggregate_path):
        raise ReleaseValidationError("confirmed admission packet aggregate hash drift")


def _validate_transformation_waterfall(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = read_object(path)
    validate_compact_value(payload)
    if payload.get("schema_version") != WATERFALL_SCHEMA:
        raise ReleaseValidationError("transformation waterfall: unsupported schema")
    # The Stage-70 producer intentionally records its automatic semantic audit
    # as pending.  Do not mistake the automatic checks for human clearance;
    # the companion artifact below is mandatory and binds this exact file.
    if payload.get("status") not in {
        "passed_with_independent_semantic_review_pending",
        "passed_with_independent_semantic_review_cleared",
    }:
        raise ReleaseValidationError("transformation waterfall is not a completed v3 receipt")
    audit = payload.get("anonymization_audit")
    if not isinstance(audit, Mapping) or audit.get("schema_version") != ANONYMIZATION_AUDIT_SCHEMA:
        raise ReleaseValidationError("transformation waterfall lacks the anonymization false-positive audit")
    automatic = audit.get("false_positive_audit")
    if not isinstance(automatic, Mapping):
        raise ReleaseValidationError("transformation waterfall lacks false-positive audit closure")
    policy_checks = automatic.get("automatic_policy_lineage_checks")
    semantic = automatic.get("independent_semantic_review")
    if (
        not isinstance(policy_checks, Mapping)
        or policy_checks.get("status") != "passed"
        or not isinstance(semantic, Mapping)
        or semantic.get("required_before_any_claim_of_semantic_false_positive_clearance") is not True
    ):
        raise ReleaseValidationError("transformation waterfall anonymization audit is incomplete")
    invariants = payload.get("invariants")
    for name in (
        "dedup_precedes_greekmmlu",
        "greekmmlu_precedes_anonymization",
        "tokens_are_exact_pinned_tokenizer_counts",
        "raw_text_or_pii_in_output",
    ):
        expected = False if name == "raw_text_or_pii_in_output" else True
        if not isinstance(invariants, Mapping) or invariants.get(name) is not expected:
            raise ReleaseValidationError("transformation waterfall invariant drift")
    return payload, file_binding(path)


def _clearance_digest(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("clearance_sha256", None)
    return canonical_json_sha256(value)


def _validate_anonymization_semantic_clearance(
    path: Path,
    *,
    waterfall_path: Path,
    waterfall_payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _require_json_schema(
        path,
        label="independent anonymization semantic false-positive clearance",
        schema=SEMANTIC_CLEARANCE_SCHEMA,
        statuses={"passed"},
    )
    if payload.get("clearance_sha256") != _clearance_digest(payload):
        raise ReleaseValidationError("anonymization semantic clearance digest drift")
    require_text(
        payload.get("completed_at"),
        label="anonymization semantic clearance completion timestamp",
    )
    _assert_binding_matches_file(
        payload.get("transformation_waterfall"),
        waterfall_path,
        label="anonymization semantic clearance waterfall",
    )
    audit = waterfall_payload["anonymization_audit"]
    if payload.get("anonymization_audit_sha256") != canonical_json_sha256(audit):
        raise ReleaseValidationError("anonymization semantic clearance audit binding drift")
    independence = payload.get("independence")
    if not isinstance(independence, Mapping) or any(
        independence.get(name) is not True
        for name in (
            "reviewer_is_independent",
            "independent_of_automatic_policy_lineage_checks",
            "protected_review_environment",
        )
    ) or independence.get("raw_text_or_pii_in_clearance") is not False:
        raise ReleaseValidationError("anonymization semantic clearance lacks independent protected-review evidence")
    review = payload.get("independent_semantic_review")
    if not isinstance(review, Mapping) or review.get("status") != "cleared":
        raise ReleaseValidationError("anonymization semantic clearance remains pending/not_included")
    for name in ("eligible_rows", "reviewed_rows", "unresolved_rows", "false_positive_findings"):
        value = review.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ReleaseValidationError(f"anonymization semantic clearance {name} is invalid")
    if review["unresolved_rows"] != 0 or review["false_positive_findings"] != 0:
        raise ReleaseValidationError("anonymization semantic clearance has unresolved false-positive findings")
    waterfall_semantic = audit["false_positive_audit"]["independent_semantic_review"]
    eligible = waterfall_semantic.get("eligible_rows")
    if isinstance(eligible, bool) or not isinstance(eligible, int) or eligible < 0:
        raise ReleaseValidationError("transformation waterfall semantic-review denominator is invalid")
    if review["eligible_rows"] != eligible or review["reviewed_rows"] > eligible:
        raise ReleaseValidationError("anonymization semantic clearance review denominator drift")
    if eligible and review["reviewed_rows"] == 0:
        raise ReleaseValidationError("anonymization semantic clearance cannot clear a non-empty review scope without review")
    return payload, file_binding(path)


def validate_required_release_evidence(
    *,
    site_inputs: Sequence[tuple[str, Path]],
    transformation_waterfall: Path,
    anonymization_semantic_clearance: Path,
) -> tuple[list[tuple[str, Path]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Validate the entire compact Phase-10 evidence closure before any write.

    The result is the exact set of compact inputs that must be copied into the
    Agent-3 handoff, their source receipts, and a small semantic-clearance
    summary suitable for the local release contract.  This deliberately does
    not accept an "optional" or "not included" category.
    """

    by_name = dict(site_inputs)
    if len(by_name) != len(site_inputs):  # parse_site_inputs already prevents this.
        raise ReleaseValidationError("duplicate compact evidence input")
    if set(by_name) & DIRECT_EVIDENCE_NAMES:
        raise ReleaseValidationError("waterfall and semantic clearance must use their dedicated arguments")
    missing = sorted(set(REQUIRED_SITE_EVIDENCE_NAMES) - set(by_name))
    if missing:
        raise ReleaseValidationError(
            "mandatory compact handoff evidence is missing: " + ", ".join(missing)
        )
    unexpected = sorted(set(by_name) - set(REQUIRED_SITE_EVIDENCE_NAMES))
    if unexpected:
        raise ReleaseValidationError(
            "unrecognized optional evidence is forbidden in the final handoff: " + ", ".join(unexpected)
        )

    payloads: dict[str, dict[str, Any]] = {}
    payloads["candidate_roster"] = _require_json_schema(
        by_name["candidate_roster"],
        label="candidate roster",
        schema="agent1_full_corpus_v3_candidate_roster_v1",
    )
    payloads["review_packet"] = _require_json_schema(
        by_name["review_packet"],
        label="review packet",
        schema=REVIEW_PACKET_SCHEMA,
        statuses={"materialized_no_model_invocation"},
    )
    _validate_response_rows(by_name["review_requests"], name="review requests", schema=REVIEW_REQUEST_SCHEMA)
    _validate_response_rows(by_name["review_responses"], name="review responses", schema=REVIEW_RESPONSE_SCHEMA)
    payloads["response_execution_receipt"] = _require_json_schema(
        by_name["response_execution_receipt"],
        label="review response execution receipt",
        schema=RESPONSE_EXECUTION_RECEIPT_SCHEMA,
        statuses={"complete"},
    )
    payloads["adjudication_execution_receipt"] = _require_json_schema(
        by_name["adjudication_execution_receipt"],
        label="review adjudication execution receipt",
        schema=ADJUDICATION_EXECUTION_RECEIPT_SCHEMA,
        statuses={"complete"},
    )
    final_adjudication = payloads["adjudication_execution_receipt"].get("final_adjudication_manifest")
    if not isinstance(final_adjudication, Mapping) or final_adjudication.get("status") != "complete" or final_adjudication.get("pending_count") != 0:
        raise ReleaseValidationError("review adjudication execution receipt remains pending")
    payloads["stage35_review_closure"] = _require_json_schema(
        by_name["stage35_review_closure"],
        label="Stage 35 review/quality execution closure",
        schema=STAGE35_CLOSURE_SCHEMA,
        statuses={"passed"},
    )
    payloads["review_sample_quality_summary"] = _require_json_schema(
        by_name["review_sample_quality_summary"],
        label="masked review-sample quality summary",
        schema=REVIEW_SAMPLE_QUALITY_SUMMARY_SCHEMA,
        statuses={"passed"},
    )
    payloads["review_sample_quality_handoff"] = _require_json_schema(
        by_name["review_sample_quality_handoff"],
        label="masked review-sample quality handoff",
        schema=REVIEW_SAMPLE_QUALITY_HANDOFF_SCHEMA,
        statuses={"passed"},
    )
    payloads["quality_summary"] = _require_json_schema(
        by_name["quality_summary"],
        label="full GlossAPI quality summary",
        schema=QUALITY_SUMMARY_SCHEMA,
        statuses={"passed"},
    )
    if payloads["quality_summary"].get("scan_mode") != "full_scan":
        raise ReleaseValidationError("full GlossAPI quality summary is not a mandatory full scan")
    payloads["lineage_summary"] = _require_json_schema(
        by_name["lineage_summary"], label="lineage summary", schema=LINEAGE_SUMMARY_SCHEMA
    )
    payloads["source_novelty"] = _require_json_schema(
        by_name["source_novelty"], label="source novelty summary", schema=SOURCE_NOVELTY_SCHEMA
    )
    payloads["license_adjudication"] = _require_json_schema(
        by_name["license_adjudication"],
        label="source license adjudication",
        schema=LICENSE_ADJUDICATION_SCHEMA,
        statuses={"technical_audit_complete"},
    )
    payloads["review_aggregate"] = _require_json_schema(
        by_name["review_aggregate"],
        label="source review aggregate",
        schema=REVIEW_AGGREGATE_SCHEMA,
    )
    _validate_review_aggregate(payloads["review_aggregate"], evidence_paths=by_name)
    payloads["admission_confirmation"] = _require_json_schema(
        by_name["admission_confirmation"],
        label="source admission confirmation",
        schema=ADMISSION_CONFIRMATION_SCHEMA,
    )
    _validate_admission_confirmation(
        payloads["admission_confirmation"], aggregate_path=by_name["review_aggregate"]
    )

    waterfall_path = resolve_existing_file(transformation_waterfall, label="transformation waterfall")
    if waterfall_path.suffix != ".json" or waterfall_path.stat().st_size > MAX_COMPACT_INPUT_BYTES:
        raise ReleaseValidationError("transformation waterfall must be a compact JSON artifact")
    validate_compact_json(waterfall_path)
    waterfall_payload, _ = _validate_transformation_waterfall(waterfall_path)
    clearance_path = resolve_existing_file(
        anonymization_semantic_clearance, label="anonymization semantic false-positive clearance"
    )
    if clearance_path.suffix != ".json" or clearance_path.stat().st_size > MAX_COMPACT_INPUT_BYTES:
        raise ReleaseValidationError("anonymization semantic clearance must be a compact JSON artifact")
    validate_compact_json(clearance_path)
    clearance_payload, _ = _validate_anonymization_semantic_clearance(
        clearance_path, waterfall_path=waterfall_path, waterfall_payload=waterfall_payload
    )

    all_inputs = [
        *site_inputs,
        ("transformation_waterfall", waterfall_path),
        ("anonymization_semantic_clearance", clearance_path),
    ]
    bindings = {name: file_binding(path) for name, path in all_inputs}
    return all_inputs, bindings, {
        "status": "passed",
        "required_categories": list(REQUIRED_HANDOFF_EVIDENCE_NAMES),
        "anonymization_semantic_false_positive_clearance": {
            "status": "cleared",
            "sha256": bindings["anonymization_semantic_clearance"]["sha256"],
            "waterfall_sha256": bindings["transformation_waterfall"]["sha256"],
        },
    }


def _contains_not_included_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if re.sub(r"[^a-z0-9]", "", str(key).casefold()) == "notincluded":
                return True
            if _contains_not_included_key(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_not_included_key(item) for item in value)
    return False


def _validate_compact_binding(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleaseValidationError(f"{label}: compact binding must be an object")
    byte_count = value.get("bytes")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 1:
        raise ReleaseValidationError(f"{label}: compact binding bytes are invalid")
    return {"bytes": byte_count, "sha256": require_sha256(value.get("sha256"), label=label)}


def validate_required_evidence_summary(value: object, *, label: str) -> dict[str, Any]:
    """Validate the path-free evidence closure persisted in a release/handoff."""

    if not isinstance(value, Mapping) or value.get("status") != "passed":
        raise ReleaseValidationError(f"{label}: required evidence closure is not passed")
    if _contains_not_included_key(value):
        raise ReleaseValidationError(f"{label}: required evidence closure contains not_included content")
    categories = value.get("required_categories")
    if categories != list(REQUIRED_HANDOFF_EVIDENCE_NAMES):
        raise ReleaseValidationError(f"{label}: required evidence category inventory drift")
    source_bindings = value.get("source_bindings")
    if not isinstance(source_bindings, Mapping) or set(source_bindings) != set(REQUIRED_HANDOFF_EVIDENCE_NAMES):
        raise ReleaseValidationError(f"{label}: required evidence source bindings drift")
    normalized_bindings = {
        name: _validate_compact_binding(source_bindings[name], label=f"{label} source binding {name}")
        for name in REQUIRED_HANDOFF_EVIDENCE_NAMES
    }
    clearance = value.get("anonymization_semantic_false_positive_clearance")
    if not isinstance(clearance, Mapping) or clearance.get("status") != "cleared":
        raise ReleaseValidationError(f"{label}: anonymization semantic false-positive clearance is incomplete")
    if clearance.get("sha256") != normalized_bindings["anonymization_semantic_clearance"]["sha256"]:
        raise ReleaseValidationError(f"{label}: semantic-clearance hash does not match its evidence binding")
    if clearance.get("waterfall_sha256") != normalized_bindings["transformation_waterfall"]["sha256"]:
        raise ReleaseValidationError(f"{label}: semantic-clearance waterfall binding drift")
    return {
        "status": "passed",
        "required_categories": list(REQUIRED_HANDOFF_EVIDENCE_NAMES),
        "source_bindings": normalized_bindings,
        "anonymization_semantic_false_positive_clearance": dict(clearance),
    }


def source_inventory(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        "SELECT source_id, acquisition_source_id, source_dataset, source_repo_id, source_revision, "
        "count(*) AS documents, sum(tokens) AS tokens "
        "FROM final_rows GROUP BY 1, 2, 3, 4, 5 ORDER BY 1, 2, 3, 4, 5"
    ).fetchall()
    return [
        {
            "source_id": str(row[0]),
            "acquisition_source_id": str(row[1]),
            "source_dataset": str(row[2]),
            "source_repo_id": str(row[3]),
            "source_revision": str(row[4]),
            "documents": int(row[5]),
            "tokens": int(row[6]),
        }
        for row in rows
    ]


def _stage_totals(connection: sqlite3.Connection, query: str) -> dict[str, int]:
    row = connection.execute(query).fetchone()
    return {"documents": int(row[0] or 0), "tokens": int(row[1] or 0)}


def build_waterfall(connection: sqlite3.Connection, *, tokenizer: ExactTokenizer) -> dict[str, Any]:
    stages = [
        (
            "50-dedup-input-pre-greekmmlu",
            "admitted_pre_mmlu_pool",
            _stage_totals(connection, "SELECT count(*), sum(tokens) FROM pool"),
        ),
        (
            "50-dedup-representatives",
            "content_bound_representative_selection",
            _stage_totals(
                connection,
                "SELECT count(*), sum(p.tokens) FROM pool p JOIN dedup d USING(stable_uid) WHERE d.action = 'keep'",
            ),
        ),
        (
            "60-greekmmlu-decontamination",
            "high_confidence_drop_ambiguous_quarantine",
            _stage_totals(
                connection,
                "SELECT count(*), sum(p.tokens) FROM pool p JOIN decontam g USING(stable_uid) WHERE g.action = 'keep'",
            ),
        ),
        (
            "65-anonymization-sanitization",
            "direct_identifier_masking_and_private_data_policy",
            _stage_totals(connection, "SELECT count(*), sum(tokens) FROM anonymized"),
        ),
        (
            "78-structural-apply",
            "approved_structural_application_or_no_op",
            _stage_totals(connection, "SELECT count(*), sum(tokens) FROM final_rows"),
        ),
    ]
    previous = stages[0][2]
    stage_rows: list[dict[str, Any]] = []
    for order, (stage, reason, totals) in enumerate(stages):
        before = totals if order == 0 else previous
        stage_rows.append(
            {
                "order": order,
                "stage": stage,
                "reason": reason,
                "documents_before": int(before["documents"]),
                "documents_after": int(totals["documents"]),
                "documents_removed_or_quarantined": int(before["documents"] - totals["documents"]),
                "tokens_before": int(before["tokens"]),
                "tokens_after": int(totals["tokens"]),
                "tokens_delta": int(totals["tokens"] - before["tokens"]),
            }
        )
        previous = totals

    rows = connection.execute(
        "SELECT source_id, acquisition_source_id, source_dataset, source_repo_id, source_revision FROM pool "
        "GROUP BY 1, 2, 3, 4, 5 ORDER BY 1, 2, 3, 4, 5"
    ).fetchall()
    by_source: list[dict[str, Any]] = []
    for source_id, acquisition, dataset, repo_id, revision in rows:
        params = (source_id, acquisition, dataset, repo_id, revision)
        where = (
            "source_id = ? AND acquisition_source_id = ? AND source_dataset = ? "
            "AND source_repo_id = ? AND source_revision = ?"
        )
        pool_where = " AND ".join(f"p.{part.strip()}" for part in where.split(" AND "))
        fixed: dict[str, dict[str, int]] = {}
        queries = {
            "dedup_input_pre_greekmmlu": f"SELECT count(*), sum(tokens) FROM pool WHERE {where}",
            "dedup_representatives": f"SELECT count(*), sum(p.tokens) FROM pool p JOIN dedup d USING(stable_uid) WHERE d.action = 'keep' AND {pool_where}",
            "greekmmlu_retained": f"SELECT count(*), sum(p.tokens) FROM pool p JOIN decontam g USING(stable_uid) WHERE g.action = 'keep' AND {pool_where}",
            "anonymization_retained": f"SELECT count(*), sum(tokens) FROM anonymized WHERE {where}",
            "structural_final": f"SELECT count(*), sum(tokens) FROM final_rows WHERE {where}",
        }
        for name, query in queries.items():
            fixed[name] = _stage_totals_with_parameters(connection, query, params)
        by_source.append(
            {
                "source_id": str(source_id),
                "acquisition_source_id": str(acquisition),
                "source_dataset": str(dataset),
                "source_repo_id": str(repo_id),
                "source_revision": str(revision),
                "stages": fixed,
            }
        )
    return {
        "schema_version": WATERFALL_SCHEMA,
        "tokenizer": compact_binding(tokenizer.binding),
        "stages": stage_rows,
        "source_stage_totals": by_source,
        "invariants": {
            "dedup_input_is_pre_greekmmlu": True,
            "final_representations_are_post_anonymization_and_structural": True,
            "tokens_are_exact_tokenizer_counts": True,
            "note": "Masking can increase token count; token deltas are not assumed monotonic.",
        },
    }


def _stage_totals_with_parameters(
    connection: sqlite3.Connection, query: str, parameters: Sequence[object]
) -> dict[str, int]:
    row = connection.execute(query, parameters).fetchone()
    return {"documents": int(row[0] or 0), "tokens": int(row[1] or 0)}


def dedup_summary(connection: sqlite3.Connection) -> dict[str, Any]:
    return {
        "dedup_input_rows": scalar(connection, "SELECT count(*) FROM pool"),
        "dedup_kept_rows": scalar(connection, "SELECT count(*) FROM dedup WHERE action = 'keep'"),
        "dedup_dropped_rows": scalar(connection, "SELECT count(*) FROM dedup WHERE action = 'drop'"),
        "greekmmlu_kept_rows": scalar(connection, "SELECT count(*) FROM decontam WHERE action = 'keep'"),
        "greekmmlu_dropped_rows": scalar(connection, "SELECT count(*) FROM decontam WHERE action = 'drop'"),
        "greekmmlu_quarantined_rows": scalar(connection, "SELECT count(*) FROM decontam WHERE action = 'quarantine'"),
        "anonymization_kept_rows": scalar(connection, "SELECT count(*) FROM anonymization_ledger WHERE action = 'keep'"),
        "anonymization_dropped_rows": scalar(connection, "SELECT count(*) FROM anonymization_ledger WHERE action = 'drop'"),
        "anonymization_quarantined_rows": scalar(connection, "SELECT count(*) FROM anonymization_ledger WHERE action = 'quarantine'"),
        "final_rows": scalar(connection, "SELECT count(*) FROM final_rows"),
    }


def copy_compact_inputs(
    *,
    release_root: Path,
    automatic: Mapping[str, Mapping[str, Any]],
    explicit: Sequence[tuple[str, Path]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    compact_root = release_root / COMPACT_ROOT_RELATIVE_PATH
    receipts: dict[str, Any] = {}
    source_bindings: dict[str, Any] = {}
    for name, value in automatic.items():
        relative = COMPACT_ROOT_RELATIVE_PATH / AUTOMATIC_COMPACT_FILES[name]
        destination = release_root / relative
        write_json_no_replace(destination, value)
        receipts[name] = {
            "path": relative.as_posix(),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }
    for name, source in explicit:
        relative = COMPACT_ROOT_RELATIVE_PATH / f"{name}{source.suffix}"
        destination = release_root / relative
        copy_file_no_replace(source, destination)
        receipts[name] = {
            "path": relative.as_posix(),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }
        source_bindings[name] = compact_binding(file_binding(source))
    if not compact_root.is_dir():  # pragma: no cover - write helpers guarantee this
        raise RuntimeError("compact handoff root was not materialized")
    return receipts, source_bindings


def materialize_release(args: argparse.Namespace) -> dict[str, Any]:
    if args.release_root.exists():
        raise FileExistsError(f"release root must be a new job-unique path: {args.release_root}")
    input_paths = (
        args.dedup_pool,
        args.dedup_ledger,
        args.dedup_manifest,
        args.decontamination_ledger,
        args.decontamination_manifest,
        args.anonymized_root,
        args.anonymization_manifest,
        args.anonymization_protected_ledger,
        args.final_corpus_root,
        args.structural_manifest,
        args.tokenizer_json,
        args.transformation_waterfall,
        args.anonymization_semantic_clearance,
    )
    if args.structural_ledger is not None:
        input_paths = (*input_paths, args.structural_ledger)
    release_root = args.release_root.resolve()
    for raw in input_paths:
        resolved = raw.resolve()
        if resolved == release_root or release_root in resolved.parents or resolved in release_root.parents:
            raise ReleaseValidationError("release root must be disjoint from every immutable input")
    work_database = args.work_database.resolve()
    if work_database == release_root or release_root in work_database.parents or work_database in release_root.parents:
        raise ReleaseValidationError("work database must be disjoint from the immutable release root")
    site_inputs = parse_site_inputs(args.site_input)
    explicit_site_inputs, evidence_bindings, evidence_summary = validate_required_release_evidence(
        site_inputs=site_inputs,
        transformation_waterfall=args.transformation_waterfall,
        anonymization_semantic_clearance=args.anonymization_semantic_clearance,
    )
    tokenizer = ExactTokenizer(args.tokenizer_json)
    dedup_payload, dedup_binding = manifest_status(
        args.dedup_manifest,
        schema=DEDUP_MANIFEST_SCHEMA,
        statuses={"passed"},
        label="dedup manifest",
    )
    decontam_payload, decontam_binding = manifest_status(
        args.decontamination_manifest,
        schema=DECONTAMINATION_MANIFEST_SCHEMA,
        statuses={"passed"},
        label="GreekMMLU manifest",
    )
    anonymization_payload, anonymization_binding = manifest_status(
        args.anonymization_manifest,
        schema=ANONYMIZATION_MANIFEST_SCHEMA,
        statuses={"completed"},
        label="anonymization manifest",
    )
    manifest_output = anonymization_payload.get("output")
    if not isinstance(manifest_output, str) or Path(manifest_output).resolve() != args.anonymized_root.resolve():
        raise ReleaseValidationError("anonymization manifest output root drift")

    run_contract_payload: dict[str, Any] | None = None
    run_contract_binding: dict[str, Any] | None = None
    stage_receipts: dict[str, dict[str, Any]] = {}
    if args.run_contract is not None:
        run_contract_payload, run_contract_binding = validate_run_contract(args.run_contract)
        for stage, receipt_path in args.stage_receipt:
            receipt = validate_stage_receipt(receipt_path, contract=run_contract_payload)
            if receipt["stage"] != stage:
                raise ReleaseValidationError("--stage-receipt STAGE does not match receipt contents")
            if stage in stage_receipts:
                raise ReleaseValidationError("duplicate stage receipt supplied")
            stage_receipts[stage] = receipt
        expected_receipts = {"50-dedup", "60-decontamination", "65-anonymization-sanitization", "78-structural-apply"}
        if set(stage_receipts) != expected_receipts:
            raise ReleaseValidationError(
                "a run contract requires receipts for dedup, decontamination, anonymization, and structural apply"
            )
    elif args.stage_receipt:
        raise ReleaseValidationError("--stage-receipt requires --run-contract")

    connection = open_database(work_database)
    try:
        pool_counts = scan_pool(connection, root=args.dedup_pool, tokenizer=tokenizer, batch_rows=args.batch_rows)
        dedup_counts = scan_dedup_ledger(
            connection,
            ledger=args.dedup_ledger,
            manifest=dedup_payload,
            batch_rows=args.batch_rows,
        )
        validate_decontamination_ledger_receipts(
            decontam_payload,
            ledger_root=args.decontamination_ledger,
        )
        decontam_counts = scan_decontamination_ledger(
            connection,
            ledger_root=args.decontamination_ledger,
            manifest=decontam_payload,
            batch_rows=args.batch_rows,
        )
        anonymization_counts = scan_anonymization_ledger(
            connection,
            ledger_root=args.anonymization_protected_ledger,
            manifest=anonymization_payload,
            batch_rows=args.batch_rows,
        )
        validate_anonymization_output_receipts(
            anonymization_payload,
            output_root=args.anonymized_root,
            ledger_root=args.anonymization_protected_ledger,
        )
        anonymized_counts = scan_anonymized_output(
            connection,
            root=args.anonymized_root,
            tokenizer=tokenizer,
            batch_rows=args.batch_rows,
        )
        final_counts = scan_final_output(
            connection,
            root=args.final_corpus_root,
            tokenizer=tokenizer,
            batch_rows=args.batch_rows,
        )
        structural_payload, structural_binding, structural_counts = validate_structural_manifest(
            connection,
            path=args.structural_manifest,
            anonymized_root=args.anonymized_root,
            final_root=args.final_corpus_root,
            structural_ledger=args.structural_ledger,
            batch_rows=args.batch_rows,
        )
        lineage = validate_generic_lineage(connection)
        waterfall = build_waterfall(connection, tokenizer=tokenizer)
        inventory = source_inventory(connection)
        dedup_summary_payload = dedup_summary(connection)
    finally:
        connection.close()

    validation_checks = [
        check("dedup_ledger_is_complete_pre_greekmmlu", dedup_counts),
        check("greekmmlu_ledger_closes_over_dedup_survivors", decontam_counts),
        check("anonymization_ledger_closes_over_greekmmlu_survivors", anonymization_counts),
        check("anonymized_output_closes_over_keep_actions", anonymized_counts),
        check("generic_representation_lineage", lineage),
        check("final_structural_output", final_counts),
        check("structural_action_closure", structural_counts or {"mode": "no_op"}),
    ]
    input_bindings: dict[str, Any] = {
        "dedup_manifest": dedup_binding,
        "decontamination_manifest": decontam_binding,
        "anonymization_manifest": anonymization_binding,
        "structural_manifest": structural_binding,
        "tokenizer": tokenizer.binding,
        "dedup_ledger": file_binding(args.dedup_ledger),
    }
    for name, binding in evidence_bindings.items():
        input_bindings[f"evidence_{name}"] = binding
    if run_contract_binding is not None:
        input_bindings["run_contract"] = run_contract_binding
    if args.structural_ledger is not None:
        input_bindings["structural_ledger"] = parquet_input_binding(args.structural_ledger)

    release_contract = {
        "schema_version": RELEASE_SCHEMA,
        "release_kind": "local_private_training_release",
        "publish_permitted": False,
        "ordered_transform_contract": {
            "dedup_input": "admitted_pre_greekmmlu_representation",
            "dedup_before_greekmmlu": True,
            "greekmmlu_before_anonymization": True,
            "anonymization_before_structural": True,
            "final_representation": "post_anonymization_post_structural",
            "structural_mode": structural_payload["mode"],
        },
        "upstream_bindings": input_bindings,
        "required_evidence": {
            **evidence_summary,
            "source_bindings": {
                name: compact_binding(binding) for name, binding in evidence_bindings.items()
            },
        },
        "source_inventory": inventory,
        "waterfall": waterfall,
        "representation_lineage": lineage,
        "counts": {
            "pre_greekmmlu_dedup_input_rows": pool_counts["rows"],
            "dedup_representative_rows": dedup_counts["keep"],
            "greekmmlu_survivor_rows": decontam_counts["keep"],
            "anonymized_training_rows": anonymized_counts["rows"],
            "final_training_rows": final_counts["rows"],
        },
    }
    release_contract_sha256 = canonical_json_sha256(release_contract)

    # Validation occurs before any release byte is published.  Output copying
    # only begins after every input ledger, manifest, and lineage edge passed.
    release_root.mkdir(parents=True, exist_ok=False)
    training_root = release_root / TRAINING_ROOT_RELATIVE_PATH
    copied_files: list[dict[str, Any]] = []
    for source in corpus_parquet_files(args.final_corpus_root):
        relative = source.relative_to(args.final_corpus_root.resolve())
        destination = training_root / relative
        copy_file_no_replace(source, destination)
        receipt = parquet_receipt(destination, relative_to=release_root)
        source_hash = sha256_file(source)
        if receipt["sha256"] != source_hash or receipt["bytes"] != source.stat().st_size:
            raise RuntimeError("immutable release copy checksum mismatch")
        copied_files.append(receipt)
    copied_files.sort(key=lambda item: str(item["path"]))

    automatic_compact = {
        "source_inventory": {
            "schema_version": "agent1_full_corpus_v3_source_inventory_summary_v1",
            "sources": inventory,
        },
        "dedup_summary": {
            "schema_version": "agent1_full_corpus_v3_dedup_summary_v1",
            **dedup_summary_payload,
        },
    }
    compact_receipts, compact_source_bindings = copy_compact_inputs(
        release_root=release_root,
        automatic=automatic_compact,
        explicit=explicit_site_inputs,
    )
    missing_compact_evidence = sorted(
        set(REQUIRED_HANDOFF_EVIDENCE_NAMES) - set(compact_receipts)
    )
    if missing_compact_evidence:  # pragma: no cover - guarded before any copy
        raise RuntimeError("required compact evidence was not materialized: " + ", ".join(missing_compact_evidence))

    validation_payload = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "passed",
        "completed_at": utc_now(),
        "release_contract_sha256": release_contract_sha256,
        "checks": validation_checks,
        "failed_checks": [],
        "final_data": {
            "root": TRAINING_ROOT_RELATIVE_PATH.as_posix(),
            "files": copied_files,
            "rows": final_counts["rows"],
            "tokens": final_counts["tokens"],
        },
    }
    validation_path = release_root / VALIDATION_RELATIVE_PATH
    write_json_no_replace(validation_path, validation_payload)
    validation_receipt = {
        "path": VALIDATION_RELATIVE_PATH.as_posix(),
        "bytes": validation_path.stat().st_size,
        "sha256": sha256_file(validation_path),
    }

    handoff_payload = {
        "schema_version": HANDOFF_SCHEMA,
        "status": "passed",
        "created_at": utc_now(),
        "release_contract_sha256": release_contract_sha256,
        "release_manifest_relative_path": MANIFEST_RELATIVE_PATH.as_posix(),
        "release_is_public_dataset": False,
        "content_exclusions": [
            "raw_parquet",
            "canonical_shards",
            "protected_pii_audit_ledgers",
            "greekmmlu_answers",
            "model_checkpoints",
        ],
        "ordered_transform_contract": release_contract["ordered_transform_contract"],
        "source_inventory": inventory,
        "dedup_and_transformation_summary": dedup_summary_payload,
        "waterfall": {
            "path": compact_receipts["transformation_waterfall"]["path"],
            "sha256": compact_receipts["transformation_waterfall"]["sha256"],
        },
        "validation": validation_receipt,
        "upstream_receipts": {
            name: compact_binding(binding)
            for name, binding in input_bindings.items()
            if name not in {"structural_ledger"}
        },
        "compact_files": compact_receipts,
        "required_evidence": {
            "status": "passed",
            "required_categories": list(REQUIRED_HANDOFF_EVIDENCE_NAMES),
            "compact_files": {
                name: compact_receipts[name] for name in REQUIRED_HANDOFF_EVIDENCE_NAMES
            },
            "source_bindings": {
                name: compact_source_bindings[name] for name in REQUIRED_HANDOFF_EVIDENCE_NAMES
            },
            "anonymization_semantic_false_positive_clearance": evidence_summary[
                "anonymization_semantic_false_positive_clearance"
            ],
        },
    }
    handoff_path = release_root / HANDOFF_RELATIVE_PATH
    write_json_no_replace(handoff_path, handoff_payload)
    handoff_receipt = {
        "path": HANDOFF_RELATIVE_PATH.as_posix(),
        "bytes": handoff_path.stat().st_size,
        "sha256": sha256_file(handoff_path),
    }

    manifest = {
        "schema_version": RELEASE_SCHEMA,
        "status": "passed",
        "completed_at": utc_now(),
        "release_contract_sha256": release_contract_sha256,
        **release_contract,
        "layout": {
            "training_root": TRAINING_ROOT_RELATIVE_PATH.as_posix(),
            "redistribution_root": None,
            "public_dataset_materialized": False,
        },
        "final_data": {
            "root": TRAINING_ROOT_RELATIVE_PATH.as_posix(),
            "files": copied_files,
            "rows": final_counts["rows"],
            "tokens": final_counts["tokens"],
        },
        "validation": validation_receipt,
        "site_handoff": handoff_receipt,
        "compact_handoff_files": compact_receipts,
        "stage_receipts": stage_receipts,
    }
    manifest_path = release_root / MANIFEST_RELATIVE_PATH
    # This is intentionally the final stage artifact written by this command.
    write_json_no_replace(manifest_path, manifest)
    validate_existing_release(release_root)
    return {
        "release_root": str(release_root),
        "manifest": str(manifest_path),
        "handoff": str(handoff_path),
        "rows": final_counts["rows"],
        "tokens": final_counts["tokens"],
        "release_contract_sha256": release_contract_sha256,
    }


def _relative_inventory(root: Path) -> list[dict[str, Any]]:
    return [parquet_receipt(path, relative_to=root) for path in corpus_parquet_files(root)]


def validate_existing_release(release_root: Path) -> dict[str, Any]:
    root = release_root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(f"release root is missing or unsafe: {root}")
    manifest_path = root / MANIFEST_RELATIVE_PATH
    manifest = read_object(manifest_path)
    if manifest.get("schema_version") != RELEASE_SCHEMA or manifest.get("status") != "passed":
        raise ReleaseValidationError("release manifest is unsupported or incomplete")
    if manifest.get("publish_permitted") is not False:
        raise ReleaseValidationError("v3 local release must never silently permit publication")
    contract = {
        key: manifest[key]
        for key in (
            "schema_version",
            "release_kind",
            "publish_permitted",
            "ordered_transform_contract",
            "upstream_bindings",
            "required_evidence",
            "source_inventory",
            "waterfall",
            "representation_lineage",
            "counts",
        )
    }
    if canonical_json_sha256(contract) != manifest.get("release_contract_sha256"):
        raise ReleaseValidationError("release manifest contract digest drift")
    release_evidence = validate_required_evidence_summary(
        manifest.get("required_evidence"), label="release manifest"
    )
    data = manifest.get("final_data")
    if not isinstance(data, Mapping) or data.get("root") != TRAINING_ROOT_RELATIVE_PATH.as_posix():
        raise ReleaseValidationError("release manifest has unsafe training-data root")
    expected_files = data.get("files")
    if not isinstance(expected_files, list) or not expected_files:
        raise ReleaseValidationError("release manifest lacks final-data inventory")
    training_root = root / TRAINING_ROOT_RELATIVE_PATH
    expected_paths: set[str] = set()
    for item in expected_files:
        if not isinstance(item, Mapping):
            raise ReleaseValidationError("release inventory item is invalid")
        relative = safe_relative_path(item.get("path"), label="release data inventory", suffix=".parquet")
        if not relative.as_posix().startswith(TRAINING_ROOT_RELATIVE_PATH.as_posix() + "/"):
            raise ReleaseValidationError("release inventory references data outside training root")
        if relative.as_posix() in expected_paths:
            raise ReleaseValidationError("release inventory repeats a data path")
        expected_paths.add(relative.as_posix())
        validate_receipt_path(root, item, label="release data inventory")
    actual_paths = {
        path.relative_to(root).as_posix() for path in corpus_parquet_files(training_root)
    }
    if actual_paths != expected_paths:
        raise ReleaseValidationError("release training-data inventory is not exact")
    validation = manifest.get("validation")
    handoff = manifest.get("site_handoff")
    if not isinstance(validation, Mapping) or not isinstance(handoff, Mapping):
        raise ReleaseValidationError("release manifest lacks validation or Agent 3 handoff receipt")
    validation_path = root / safe_relative_path(validation.get("path"), label="validation receipt", suffix=".json")
    handoff_path = root / safe_relative_path(handoff.get("path"), label="Agent 3 handoff", suffix=".json")
    for target, receipt, label in ((validation_path, validation, "validation receipt"), (handoff_path, handoff, "Agent 3 handoff")):
        if target.is_symlink() or not target.is_file() or target.stat().st_size != receipt.get("bytes"):
            raise ReleaseValidationError(f"{label} receipt is missing or changed")
        if sha256_file(target) != require_sha256(receipt.get("sha256"), label=label):
            raise ReleaseValidationError(f"{label} receipt hash drift")
    validation_payload = read_object(validation_path)
    if (
        validation_payload.get("schema_version") != VALIDATION_SCHEMA
        or validation_payload.get("status") != "passed"
        or validation_payload.get("release_contract_sha256") != manifest.get("release_contract_sha256")
        or validation_payload.get("failed_checks") != []
    ):
        raise ReleaseValidationError("validation receipt does not pass the release contract")
    handoff_payload = read_object(handoff_path)
    if (
        handoff_payload.get("schema_version") != HANDOFF_SCHEMA
        or handoff_payload.get("status") != "passed"
        or handoff_payload.get("release_contract_sha256") != manifest.get("release_contract_sha256")
        or handoff_payload.get("release_is_public_dataset") is not False
    ):
        raise ReleaseValidationError("Agent 3 handoff does not bind the local release")
    validate_compact_value(handoff_payload)
    handoff_evidence = validate_required_evidence_summary(
        handoff_payload.get("required_evidence"), label="Agent 3 handoff"
    )
    handoff_compact_evidence = handoff_payload["required_evidence"].get("compact_files")
    if (
        not isinstance(handoff_compact_evidence, Mapping)
        or set(handoff_compact_evidence) != set(REQUIRED_HANDOFF_EVIDENCE_NAMES)
    ):
        raise ReleaseValidationError("Agent 3 handoff required compact evidence inventory drift")
    if handoff_evidence != release_evidence:
        raise ReleaseValidationError("Agent 3 handoff required evidence differs from the local release")
    compact_files = handoff_payload.get("compact_files")
    if not isinstance(compact_files, Mapping):
        raise ReleaseValidationError("Agent 3 handoff lacks compact file inventory")
    for name, receipt in compact_files.items():
        if not isinstance(receipt, Mapping):
            raise ReleaseValidationError("Agent 3 compact file receipt is invalid")
        relative = safe_relative_path(receipt.get("path"), label=f"compact file {name}")
        if not relative.as_posix().startswith(COMPACT_ROOT_RELATIVE_PATH.as_posix() + "/"):
            raise ReleaseValidationError("Agent 3 handoff points outside compact handoff root")
        target = root / relative
        if target.suffix not in {".json", ".jsonl"} or target.is_symlink() or not target.is_file():
            raise ReleaseValidationError("Agent 3 compact inventory contains unsafe content")
        if target.stat().st_size != receipt.get("bytes") or sha256_file(target) != require_sha256(receipt.get("sha256"), label=f"compact file {name}"):
            raise ReleaseValidationError("Agent 3 compact file receipt drift")
        validate_compact_json(target)
    for name in REQUIRED_HANDOFF_EVIDENCE_NAMES:
        receipt = handoff_compact_evidence[name]
        if receipt != compact_files.get(name):
            raise ReleaseValidationError("Agent 3 handoff required evidence receipt differs from compact inventory")
        compact = _validate_compact_binding(receipt, label=f"Agent 3 compact evidence {name}")
        if compact != release_evidence["source_bindings"][name]:
            raise ReleaseValidationError("Agent 3 compact evidence differs from source binding")
    return {
        "release_root": str(root),
        "manifest": str(manifest_path),
        "rows": int(data.get("rows", 0)),
        "files": len(expected_paths),
        "release_contract_sha256": str(manifest.get("release_contract_sha256")),
    }


def cmd_validate(args: argparse.Namespace) -> None:
    result = validate_existing_release(args.release_root)
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite validation output: {args.output}")
        write_json_no_replace(
            args.output,
            {
                "schema_version": VALIDATION_SCHEMA,
                "status": "passed",
                "validated_at": utc_now(),
                **result,
            },
        )
    print(json.dumps({"ok": True, **result}, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    materialize = sub.add_parser("materialize", help="validate ordered v3 lineage and write a private local release")
    materialize.add_argument("--dedup-pool", type=Path, required=True, help="complete admitted pre-GreekMMLU Parquet tree")
    materialize.add_argument("--dedup-ledger", type=Path, required=True, help="Stage 50 v3 content-bound decision ledger")
    materialize.add_argument("--dedup-manifest", type=Path, required=True)
    materialize.add_argument("--decontamination-ledger", type=Path, required=True, help="Stage 60 ledger tree")
    materialize.add_argument("--decontamination-manifest", type=Path, required=True)
    materialize.add_argument("--anonymized-root", type=Path, required=True, help="Stage 65 keep-only masked output tree")
    materialize.add_argument("--anonymization-manifest", type=Path, required=True)
    materialize.add_argument("--anonymization-protected-ledger", type=Path, required=True)
    materialize.add_argument("--final-corpus-root", type=Path, required=True, help="Stage 78 output tree")
    materialize.add_argument("--structural-manifest", type=Path, required=True, help="Stage 78 application/no-op manifest")
    materialize.add_argument("--structural-ledger", type=Path, help="required for mode=applied structural manifest")
    materialize.add_argument("--tokenizer-json", type=Path, required=True, help="frozen exact tokenizer.json")
    materialize.add_argument(
        "--transformation-waterfall",
        type=Path,
        required=True,
        help="completed Stage-70 compact waterfall; semantic clearance is checked separately",
    )
    materialize.add_argument(
        "--anonymization-semantic-clearance",
        type=Path,
        required=True,
        help="completed independent semantic false-positive clearance bound to --transformation-waterfall",
    )
    materialize.add_argument("--work-database", type=Path, required=True, help="new job-unique SQLite validation index")
    materialize.add_argument("--release-root", type=Path, required=True, help="new immutable local release directory")
    materialize.add_argument("--batch-rows", type=int, default=2048)
    materialize.add_argument("--run-contract", type=Path, help="optional frozen v3 contract; requires core stage receipts")
    materialize.add_argument(
        "--stage-receipt",
        nargs=2,
        action="append",
        default=[],
        metavar=("STAGE", "PATH"),
        help="receipt bindings when --run-contract is supplied",
    )
    materialize.add_argument(
        "--site-input",
        nargs=2,
        action="append",
        default=[],
        metavar=("NAME", "PATH"),
        help=(
            "required compact review/quality/admission/execution evidence copied into the Agent 3 handoff; "
            "all named categories are required at materialization time"
        ),
    )
    materialize.set_defaults(func=lambda args: print(json.dumps({"ok": True, **materialize_release(args)}, sort_keys=True)))
    validate = sub.add_parser("validate", help="recheck an existing immutable local v3 release")
    validate.add_argument("--release-root", type=Path, required=True)
    validate.add_argument("--output", type=Path, help="optional new immutable revalidation receipt")
    validate.set_defaults(func=cmd_validate)
    return parser


def main() -> int:
    args = parser().parse_args()
    if getattr(args, "batch_rows", 1) < 1:
        raise ValueError("--batch-rows must be positive")
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
