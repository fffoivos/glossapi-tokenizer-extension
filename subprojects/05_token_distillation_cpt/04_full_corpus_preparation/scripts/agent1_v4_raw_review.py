#!/usr/bin/env python3
"""Receipt-bound raw-document review packet primitives for Agent 1 v4.

This module intentionally precedes canonicalization and GlossAPI.  It samples
the source-selected raw text representation directly from acquired Parquet,
materializes exactly one immutable file per selected logical document, and
binds every request to source receipt, path, bytes, and content hash.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from full_corpus_io import (
    SourceArtifact,
    artifact_relative_path,
    artifacts_from_receipt,
    first_nonempty,
    iter_parquet_rows,
    sha256_file,
)
from source_lineage import canonical_json, load_json


POLICY_SCHEMA = "agent1_v4_raw_review_policy_v1"
REQUEST_SCHEMA = "agent1_v4_raw_review_request_v1"
PACKET_SCHEMA = "agent1_v4_raw_review_packet_manifest_v1"
BLOCKED_SCHEMA = "agent1_v4_raw_review_blocked_receipt_v1"
HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SAFE_SOURCE_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
ALLOWED_ROUTES = frozenset({"html_web", "pdf_ocr", "mixed", "structured"})


class PacketBlockedError(RuntimeError):
    """Raised after a blocking receipt has been published."""


@dataclass(frozen=True)
class RawCandidate:
    source_id: str
    repo_id: str
    source_dataset: str
    source_doc_id: str
    source_revision: str
    source_route: str
    extraction_route: str
    artifact_path: str
    row_index: int
    representation_suffix: str
    text_field: str
    text: str
    identity: str
    logical_key: str
    rank: bytes


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: object) -> str:
    return sha256_text(canonical_json(value))


def file_binding(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink() or resolved.stat().st_size < 1:
        raise FileNotFoundError(f"required regular non-empty file is missing: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def packet_file_binding(root: Path, path: Path) -> dict[str, object]:
    """Bind a packet-internal file without making an atomic rename observable.

    A packet is built in a sibling staging directory and atomically renamed
    into its final receipt-bound location.  Its own manifest therefore carries
    a safe relative path, while external input bindings retain absolute paths.
    """

    binding = file_binding(path)
    binding["path"] = path.resolve().relative_to(root.resolve()).as_posix()
    return binding


def _write_no_replace(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.link(temporary, path)
        os.chmod(path, mode)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_no_replace(path: Path, value: Mapping[str, object]) -> None:
    _write_no_replace(
        path,
        (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )


def write_jsonl_no_replace(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    payload = b"".join(
        (canonical_json(dict(row)) + "\n").encode("utf-8") for row in rows
    )
    if not payload:
        raise ValueError("refusing to publish an empty request file")
    _write_no_replace(path, payload)


def read_json_object(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or not HEX_SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_commit(value: object) -> str:
    if not isinstance(value, str) or not GIT_COMMIT_RE.fullmatch(value):
        raise ValueError("code_commit must be a 40-character lowercase Git commit")
    return value


def load_policy(path: Path) -> dict[str, Any]:
    policy = read_json_object(path)
    if policy.get("schema_version") != POLICY_SCHEMA:
        raise ValueError(f"{path}: unsupported policy schema")
    source_ids = policy.get("source_ids")
    excluded = policy.get("excluded_source_ids")
    if not isinstance(source_ids, list) or len(source_ids) != 18:
        raise ValueError(f"{path}: policy must contain exactly 18 source_ids")
    if not isinstance(excluded, list) or len(excluded) != 8:
        raise ValueError(f"{path}: policy must contain exactly 8 excluded_source_ids")
    if any(not isinstance(item, str) or not SAFE_SOURCE_RE.fullmatch(item) for item in source_ids):
        raise ValueError(f"{path}: source_ids contain an unsafe identifier")
    if len(set(source_ids)) != len(source_ids) or set(source_ids) & set(excluded):
        raise ValueError(f"{path}: source scope has duplicates or overlap with exclusions")
    if policy.get("documents_per_source") != 20:
        raise ValueError(f"{path}: documents_per_source must be 20")
    if policy.get("review_transport") != "exact_raw_user_approved":
        raise ValueError(f"{path}: exact raw review approval is required")
    if policy.get("model") != "gpt-5.6-terra":
        raise ValueError(f"{path}: only gpt-5.6-terra is valid for this lane")
    if policy.get("max_attempts_per_document") != 3:
        raise ValueError(f"{path}: max attempts must remain 3")
    return policy


def load_roster(path: Path, source_ids: Sequence[str]) -> dict[str, dict[str, str]]:
    roster = read_json_object(path)
    candidate_ids = roster.get("candidate_source_ids")
    if not isinstance(candidate_ids, list):
        raise ValueError(f"{path}: missing candidate_source_ids")
    source_routes = roster.get("source_routes")
    extraction_routes = roster.get("extraction_routes")
    if not isinstance(source_routes, Mapping) or not isinstance(extraction_routes, Mapping):
        raise ValueError(f"{path}: missing source/extraction routes")
    result: dict[str, dict[str, str]] = {}
    for source_id in source_ids:
        if source_id not in candidate_ids:
            raise ValueError(f"{path}: selected source not in candidate roster: {source_id}")
        source_route = str(source_routes.get(source_id, ""))
        extraction_route = str(extraction_routes.get(source_id, ""))
        if source_route not in ALLOWED_ROUTES or extraction_route not in ALLOWED_ROUTES:
            raise ValueError(f"{path}: invalid route declaration for {source_id}")
        result[source_id] = {
            "source_route": source_route,
            "extraction_route": extraction_route,
        }
    return result


def _source_dataset(source: SourceArtifact, row: Mapping[str, Any]) -> str:
    column = str(source.config.get("source_column", "source_dataset"))
    value = row.get(column)
    return str(value) if value not in (None, "") else source.repo_id


def _source_doc_id(
    source: SourceArtifact,
    row: Mapping[str, Any],
    artifact_path: str,
    row_index: int,
    representation_suffix: str,
) -> str:
    values = [
        str(row[column])
        for column in source.config.get("id_columns", [])
        if row.get(column) not in (None, "")
    ]
    if values:
        base = "|".join(values)
    else:
        base = "synthetic:" + sha256_json(
            {
                "namespace": "agent1_v4_source_doc_id_v1",
                "source_id": source.source_id,
                "revision": source.revision,
                "artifact_path": artifact_path,
                "row_index": row_index,
            }
        )
    return base if representation_suffix == "0" else f"{base}#{representation_suffix}"


def _identity(candidate: Mapping[str, object]) -> str:
    return canonical_json({"namespace": "agent1_v4_raw_review_identity_v1", **candidate})


def _rank(seed: bytes, identity: str) -> bytes:
    return hmac.new(seed, identity.encode("utf-8"), hashlib.sha256).digest()


def _candidate_key(source_dataset: str, source_doc_id: str) -> str:
    return canonical_json(
        {
            "namespace": "agent1_v4_raw_review_logical_document_v1",
            "source_dataset": source_dataset,
            "source_doc_id": source_doc_id,
        }
    )


def _iter_raw_candidates(
    source: SourceArtifact,
    routes: Mapping[str, str],
    seed: bytes,
) -> Iterable[RawCandidate]:
    text_columns = list(source.config.get("text_columns", []))
    if not text_columns:
        raise ValueError(f"{source.source_id}: no configured provisional text column")
    for artifact_path in sorted(source.files):
        relative = artifact_relative_path(source, artifact_path)
        for row_index, row in iter_parquet_rows(artifact_path):
            # Stage 05 must sample the selected raw source representation.  It
            # never falls through to alternate cleaned text columns.
            selected = first_nonempty(row, text_columns)
            if selected is None:
                continue
            text_field, text = selected
            if not text.strip():
                continue
            source_dataset = _source_dataset(source, row)
            source_doc_id = _source_doc_id(source, row, relative, row_index, "0")
            identity = _identity(
                {
                    "source_id": source.source_id,
                    "repo_id": source.repo_id,
                    "source_revision": source.revision,
                    "artifact_path": relative,
                    "row_index": row_index,
                    "representation_suffix": "0",
                    "source_dataset": source_dataset,
                    "source_doc_id": source_doc_id,
                    "text_field": text_field,
                }
            )
            yield RawCandidate(
                source_id=source.source_id,
                repo_id=source.repo_id,
                source_dataset=source_dataset,
                source_doc_id=source_doc_id,
                source_revision=source.revision,
                source_route=str(routes["source_route"]),
                extraction_route=str(routes["extraction_route"]),
                artifact_path=relative,
                row_index=row_index,
                representation_suffix="0",
                text_field=text_field,
                text=text,
                identity=identity,
                logical_key=_candidate_key(source_dataset, source_doc_id),
                rank=_rank(seed, identity),
            )


def select_source_documents(
    source: SourceArtifact,
    routes: Mapping[str, str],
    seed: bytes,
    documents_per_source: int,
) -> tuple[list[RawCandidate], dict[str, int]]:
    """Choose deterministic top-k unique logical documents without full retention."""

    chosen: dict[str, RawCandidate] = {}
    eligible_units = 0
    for candidate in _iter_raw_candidates(source, routes, seed):
        eligible_units += 1
        existing = chosen.get(candidate.logical_key)
        if existing is not None and (candidate.rank, candidate.identity) >= (
            existing.rank,
            existing.identity,
        ):
            continue
        if existing is not None:
            del chosen[candidate.logical_key]
        if len(chosen) < documents_per_source:
            chosen[candidate.logical_key] = candidate
            continue
        worst = max(chosen.values(), key=lambda item: (item.rank, item.identity))
        if (candidate.rank, candidate.identity) < (worst.rank, worst.identity):
            del chosen[worst.logical_key]
            chosen[candidate.logical_key] = candidate

    selected = sorted(chosen.values(), key=lambda item: (item.rank, item.identity))
    return selected, {
        "eligible_document_units": eligible_units,
        "eligible_unique_documents_at_selection_cutoff": len(selected),
    }


def _safe_relative_document_path(source_id: str, sample_id: str) -> str:
    if not SAFE_SOURCE_RE.fullmatch(source_id):
        raise ValueError(f"unsafe source ID for output path: {source_id!r}")
    _require_sha256("sample_id", sample_id)
    return f"documents/{source_id}/{sample_id}.txt"


def _request_for_candidate(
    candidate: RawCandidate,
    *,
    sample_id: str,
    document_path: str,
    document_sha256: str,
    document_bytes: int,
    document_line_count: int,
    model: str,
    prompt_sha256: str,
    response_schema_sha256: str,
    code_commit: str,
) -> dict[str, object]:
    base: dict[str, object] = {
        "schema_version": REQUEST_SCHEMA,
        "sample_id": sample_id,
        "source_id": candidate.source_id,
        "source_dataset": candidate.source_dataset,
        "source_doc_id": candidate.source_doc_id,
        "source_revision": candidate.source_revision,
        "source_route": candidate.source_route,
        "extraction_route": candidate.extraction_route,
        "document_path": document_path,
        "document_sha256": document_sha256,
        "document_bytes": document_bytes,
        "document_line_count": document_line_count,
        "origin_locator": {
            "repo_id": candidate.repo_id,
            "artifact_path": candidate.artifact_path,
            "row_index": candidate.row_index,
            "representation_suffix": candidate.representation_suffix,
            "text_field": candidate.text_field,
        },
        "model": model,
        "prompt_sha256": prompt_sha256,
        "response_schema_sha256": response_schema_sha256,
        "code_commit": code_commit,
    }
    request_id = sha256_json({"namespace": "agent1_v4_raw_review_request_v1", **base})
    return {"request_id": request_id, **base}


def validate_request(request: Mapping[str, object]) -> None:
    required = {
        "schema_version",
        "request_id",
        "sample_id",
        "source_id",
        "source_dataset",
        "source_doc_id",
        "source_revision",
        "source_route",
        "extraction_route",
        "document_path",
        "document_sha256",
        "document_bytes",
        "document_line_count",
        "origin_locator",
        "model",
        "prompt_sha256",
        "response_schema_sha256",
        "code_commit",
    }
    if set(request) != required:
        raise ValueError(f"request keys drift: {sorted(set(request) ^ required)}")
    if request.get("schema_version") != REQUEST_SCHEMA:
        raise ValueError("unsupported review request schema")
    for key in ("request_id", "sample_id", "document_sha256", "prompt_sha256", "response_schema_sha256"):
        _require_sha256(key, request.get(key))
    _require_commit(request.get("code_commit"))
    if request.get("model") != "gpt-5.6-terra":
        raise ValueError("raw review request must target gpt-5.6-terra")
    document_path = request.get("document_path")
    if not isinstance(document_path, str) or not re.fullmatch(
        r"documents/[A-Za-z0-9_.-]+/[a-f0-9]{64}\.txt", document_path
    ):
        raise ValueError("request document_path is unsafe")
    if request.get("source_route") not in ALLOWED_ROUTES or request.get("extraction_route") not in ALLOWED_ROUTES:
        raise ValueError("request route is invalid")
    if not isinstance(request.get("document_bytes"), int) or int(request["document_bytes"]) < 1:
        raise ValueError("request document_bytes must be positive")
    if not isinstance(request.get("document_line_count"), int) or int(request["document_line_count"]) < 1:
        raise ValueError("request document_line_count must be positive")
    locator = request.get("origin_locator")
    expected_locator = {"repo_id", "artifact_path", "row_index", "representation_suffix", "text_field"}
    if not isinstance(locator, Mapping) or set(locator) != expected_locator:
        raise ValueError("request origin_locator is invalid")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: request must be an object")
            rows.append(value)
    return rows


def validate_packet(packet_root: Path, manifest_path: Path | None = None) -> dict[str, object]:
    root = packet_root.resolve()
    manifest_path = manifest_path or root / "packet_manifest.json"
    manifest = read_json_object(manifest_path)
    if manifest.get("schema_version") != PACKET_SCHEMA or manifest.get("status") != "passed":
        raise ValueError(f"{manifest_path}: not a passed v4 raw review packet")
    requests_binding = manifest.get("requests")
    if not isinstance(requests_binding, Mapping):
        raise ValueError(f"{manifest_path}: missing requests binding")
    request_path = root / "requests.jsonl"
    actual_requests = file_binding(request_path)
    if (
        requests_binding.get("path") != "requests.jsonl"
        or actual_requests["bytes"] != requests_binding.get("bytes")
        or actual_requests["sha256"] != requests_binding.get("sha256")
    ):
        raise ValueError(f"{manifest_path}: request binding drift")
    requests = _read_jsonl(request_path)
    source_counts: dict[str, int] = {}
    seen_requests: set[str] = set()
    seen_documents: set[str] = set()
    for request in requests:
        validate_request(request)
        request_id = str(request["request_id"])
        if request_id in seen_requests:
            raise ValueError("duplicate request_id")
        seen_requests.add(request_id)
        relative = Path(str(request["document_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("document path escapes packet root")
        document = root / relative
        if document.is_symlink() or not document.is_file():
            raise ValueError(f"missing/symlinked document: {relative}")
        if str(relative) in seen_documents:
            raise ValueError("duplicate materialized document path")
        seen_documents.add(str(relative))
        if document.stat().st_size != request["document_bytes"] or sha256_file(document) != request["document_sha256"]:
            raise ValueError(f"document binding drift: {relative}")
        source_id = str(request["source_id"])
        source_counts[source_id] = source_counts.get(source_id, 0) + 1
    expected = manifest.get("source_counts")
    if source_counts != expected:
        raise ValueError("packet source counts drift")
    if any(count != 20 for count in source_counts.values()) or len(source_counts) != 18:
        raise ValueError("packet must close exactly 18 sources with 20 documents each")
    if len(requests) != 360:
        raise ValueError("packet must contain exactly 360 requests")
    return manifest


def materialize_raw_review_packet(
    *,
    sources_path: Path,
    acquisition_receipt: Path,
    roster_path: Path,
    policy_path: Path,
    seed_hex: str,
    prompt_path: Path,
    response_schema_path: Path,
    code_commit: str,
    output: Path,
) -> dict[str, object]:
    """Materialize the complete 18x20 review packet or a blocked receipt."""

    policy = load_policy(policy_path)
    _require_commit(code_commit)
    if not HEX_SHA256_RE.fullmatch(seed_hex):
        raise ValueError("seed must be a 32-byte lowercase hex value")
    seed = bytes.fromhex(seed_hex)
    if len(seed) != 32:
        raise ValueError("seed must encode exactly 32 bytes")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"review packet output already exists: {output}")
    sources = artifacts_from_receipt(
        sources_path, acquisition_receipt, set(policy["source_ids"])
    )
    if [source.source_id for source in sources] != sorted(policy["source_ids"]):
        raise ValueError("receipt/source-policy closure drift")
    routes = load_roster(roster_path, policy["source_ids"])
    selected_by_source: dict[str, list[RawCandidate]] = {}
    eligibility: dict[str, dict[str, int]] = {}
    issues: list[dict[str, object]] = []
    for source in sources:
        selected, statistics = select_source_documents(
            source, routes[source.source_id], seed, int(policy["documents_per_source"])
        )
        selected_by_source[source.source_id] = selected
        eligibility[source.source_id] = statistics
        if len(selected) != int(policy["documents_per_source"]):
            issues.append(
                {
                    "source_id": source.source_id,
                    "reason": "fewer_than_20_unique_nonempty_raw_documents",
                    "eligible_document_units": statistics["eligible_document_units"],
                    "eligible_unique_documents_at_selection_cutoff": statistics[
                        "eligible_unique_documents_at_selection_cutoff"
                    ],
                }
            )

    common = {
        "policy": file_binding(policy_path),
        "sources": file_binding(sources_path),
        "acquisition_receipt": file_binding(acquisition_receipt),
        "roster": file_binding(roster_path),
        "prompt": file_binding(prompt_path),
        "response_schema": file_binding(response_schema_path),
        "code_commit": code_commit,
        "sampling_seed_sha256": sha256_text(seed_hex),
        "review_transport": policy["review_transport"],
        "model": policy["model"],
    }
    if issues:
        output.mkdir(mode=0o700)
        blocked = {
            "schema_version": BLOCKED_SCHEMA,
            "status": "blocked",
            **common,
            "eligibility": eligibility,
            "issues": issues,
        }
        write_json_no_replace(output / "blocking_issues.json", blocked)
        raise PacketBlockedError(
            f"review packet blocked; inspect {output / 'blocking_issues.json'}"
        )

    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        os.chmod(staging, 0o700)
        requests: list[dict[str, object]] = []
        documents: list[dict[str, object]] = []
        for source_id in policy["source_ids"]:
            for candidate in selected_by_source[source_id]:
                sample_id = sha256_json(
                    {
                        "namespace": "agent1_v4_raw_review_sample_v1",
                        "identity": candidate.identity,
                    }
                )
                relative_path = _safe_relative_document_path(source_id, sample_id)
                document_path = staging / relative_path
                payload = candidate.text.encode("utf-8")
                _write_no_replace(document_path, payload)
                digest = sha256_text(candidate.text)
                request = _request_for_candidate(
                    candidate,
                    sample_id=sample_id,
                    document_path=relative_path,
                    document_sha256=digest,
                    document_bytes=len(payload),
                    document_line_count=candidate.text.count("\n") + 1,
                    model=str(policy["model"]),
                    prompt_sha256=str(file_binding(prompt_path)["sha256"]),
                    response_schema_sha256=str(file_binding(response_schema_path)["sha256"]),
                    code_commit=code_commit,
                )
                validate_request(request)
                requests.append(request)
                documents.append(
                    {
                        "path": relative_path,
                        "bytes": len(payload),
                        "sha256": digest,
                        "request_id": request["request_id"],
                    }
                )
        requests.sort(key=lambda row: (str(row["source_id"]), str(row["sample_id"])))
        write_jsonl_no_replace(staging / "requests.jsonl", requests)
        source_counts = {
            source_id: sum(1 for request in requests if request["source_id"] == source_id)
            for source_id in policy["source_ids"]
        }
        manifest: dict[str, object] = {
            "schema_version": PACKET_SCHEMA,
            "status": "passed",
            **common,
            "documents_per_source": policy["documents_per_source"],
            "logical_review_count": len(requests),
            "source_counts": source_counts,
            "eligibility": eligibility,
            "requests": packet_file_binding(staging, staging / "requests.jsonl"),
            "documents": documents,
        }
        write_json_no_replace(staging / "packet_manifest.json", manifest)
        validate_packet(staging)
        os.rename(staging, output)
        validate_packet(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
