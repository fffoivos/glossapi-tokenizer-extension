#!/usr/bin/env python3
"""Close Agent 1 v3's compact local-review evidence boundary.

Stage 30 deliberately stops before invoking a model.  The authenticated
MacBook runs :mod:`run_agent1_v3_codex_reviews` over that compact, masked
packet.  This tool is the only supported bridge back into the CPU lane:

* ``package-external`` creates a strict, no-raw-corpus bundle containing only
  responses and the runner's immutable calibration/response/adjudication
  execution receipts;
* ``import-external`` copies that bundle into a Stage-35 attempt after hashing
  every file (the original external directory remains an explicit input);
* ``validate-closure`` proves exact request/response/adjudication closure,
  including the frozen model, prompt, schema and code commit; and
* ``materialize-masked-sample`` emits the exact primary review sample in a
  small v3-only adapter format for the Rust-backed diagnostic.

None of these commands evaluates source admission or reads canonical Parquet.
The adapter contains only the position-preserving, direct-identifier-masked
review copies already materialized by Stage 30.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SCRIPT_DIR))

import agent1_v3_review as review  # noqa: E402


EXTERNAL_BUNDLE_SCHEMA = "agent1_v3_external_review_evidence_bundle_v1"
EXTERNAL_IMPORT_SCHEMA = "agent1_v3_external_review_evidence_import_v1"
CLOSURE_SCHEMA = "agent1_v3_quality_review_evidence_closure_v1"
MASKED_SAMPLE_SCHEMA = "agent1_v3_masked_review_sample_v1"
MASKED_SAMPLE_RECEIPT_SCHEMA = "agent1_v3_masked_review_sample_receipt_v1"

RESPONSE_RECEIPT_SCHEMA = "agent1_v3_codex_review_response_execution_receipt_v1"
ADJUDICATION_RECEIPT_SCHEMA = "agent1_v3_codex_review_adjudication_execution_receipt_v1"
CALIBRATION_RECEIPT_SCHEMA = "agent1_v3_codex_review_calibration_receipt_v1"
PACKET_MANIFEST_SCHEMA = "agent1_v3_review_packet_manifest_v1"
REQUIRED_MODEL = "gpt-5.6-luna"

BUNDLE_MANIFEST_NAME = "external_review_evidence_manifest.json"
RESPONSES_NAME = "responses.jsonl"
RESPONSE_RECEIPT_NAME = "response_execution_receipt.json"
ADJUDICATION_RECEIPT_NAME = "adjudication_execution_receipt.json"
CALIBRATION_RECEIPT_NAME = "calibration_receipt.json"
BUNDLE_FILE_NAMES = (
    RESPONSES_NAME,
    RESPONSE_RECEIPT_NAME,
    ADJUDICATION_RECEIPT_NAME,
    CALIBRATION_RECEIPT_NAME,
)
BUNDLE_ALL_NAMES = frozenset((BUNDLE_MANIFEST_NAME, *BUNDLE_FILE_NAMES))
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CODE_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def require_sha256(label: str, value: Any) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def read_json(path: Path, *, label: str | None = None) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label or path}: JSON must be UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label or path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label or path}: expected a JSON object")
    return value


def regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file() or path.stat().st_size < 1:
        raise FileNotFoundError(f"{label} must be a non-empty regular file: {path}")
    return path.resolve()


def regular_directory(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise FileNotFoundError(f"{label} must be a non-symlink directory: {path}")
    return path.resolve()


def file_binding(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = regular_file(path, label="bound input")
    result: dict[str, Any] = {
        "path": str(resolved.relative_to(relative_to.resolve())) if relative_to else str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
    return result


def binding_matches(
    actual: Any,
    expected: Mapping[str, Any],
    *,
    label: str,
    require_path: bool = False,
) -> None:
    if not isinstance(actual, Mapping):
        raise ValueError(f"{label}: expected a file binding")
    if actual.get("bytes") != expected.get("bytes") or actual.get("sha256") != expected.get("sha256"):
        raise ValueError(f"{label}: byte/hash binding drift")
    require_sha256(f"{label}.sha256", actual.get("sha256"))
    if require_path and actual.get("path") != expected.get("path"):
        raise ValueError(f"{label}: path binding drift")


def write_bytes_no_replace(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
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
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_no_replace(path: Path, value: Mapping[str, Any]) -> None:
    write_bytes_no_replace(
        path,
        (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8"),
    )


def write_jsonl_no_replace(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = b"".join((canonical_json(dict(row)) + "\n").encode("utf-8") for row in rows)
    if not payload:
        raise ValueError("masked review sample cannot be empty")
    write_bytes_no_replace(path, payload)


def read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    regular_file(path, label=label)
    result: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{label}:{number}: invalid JSONL") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{label}:{number}: row must be an object")
            result.append(value)
    if not result:
        raise ValueError(f"{label}: JSONL is empty")
    return result


def receipt_digest(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("receipt_sha256", None)
    return sha256_json(payload)


def manifest_digest(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("manifest_sha256", None)
    return sha256_json(payload)


def validate_self_hash(value: Mapping[str, Any], *, field: str, label: str) -> None:
    expected = value.get(field)
    actual = receipt_digest(value) if field == "receipt_sha256" else manifest_digest(value)
    if expected != actual:
        raise ValueError(f"{label}: {field} drift")


def _require_exact_file_names(root: Path) -> None:
    entries = {entry.name for entry in root.iterdir()}
    if entries != BUNDLE_ALL_NAMES:
        raise ValueError(
            "external review evidence must contain exactly "
            f"{sorted(BUNDLE_ALL_NAMES)}, found {sorted(entries)}"
        )
    for name in entries:
        regular_file(root / name, label=f"external evidence {name}")


def bundle_paths(root: Path) -> dict[str, Path]:
    root = regular_directory(root, label="external review evidence directory")
    _require_exact_file_names(root)
    return {
        "root": root,
        "manifest": root / BUNDLE_MANIFEST_NAME,
        "responses": root / RESPONSES_NAME,
        "response_receipt": root / RESPONSE_RECEIPT_NAME,
        "adjudication_receipt": root / ADJUDICATION_RECEIPT_NAME,
        "calibration_receipt": root / CALIBRATION_RECEIPT_NAME,
    }


def _forbid_raw_corpus_fields(value: Any, *, label: str) -> None:
    """Reject a bundle that smuggles review/corpus text through an unknown field."""

    forbidden = {
        "review_copy",
        "comparison_bundle",
        "canonical_root",
        "canonical_text",
        "raw_text",
        "normalized_text",
        "source_doc_id",
        "raw_corpus",
    }
    if isinstance(value, Mapping):
        overlap = forbidden & {str(key) for key in value}
        if overlap:
            raise ValueError(f"{label}: external evidence contains forbidden raw-corpus fields {sorted(overlap)}")
        for key, child in value.items():
            _forbid_raw_corpus_fields(child, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _forbid_raw_corpus_fields(child, label=f"{label}[{index}]")


def validate_bundle(root: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    paths = bundle_paths(root)
    manifest = read_json(paths["manifest"], label="external evidence manifest")
    if manifest.get("schema_version") != EXTERNAL_BUNDLE_SCHEMA or manifest.get("status") != "complete":
        raise ValueError("external evidence manifest has unsupported schema/status")
    validate_self_hash(manifest, field="manifest_sha256", label="external evidence manifest")
    if manifest.get("raw_corpus_included") is not False:
        raise ValueError("external evidence manifest must explicitly exclude raw corpus")
    if manifest.get("bundle_contents") != "responses_and_execution_receipts_only":
        raise ValueError("external evidence manifest bundle content declaration drift")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != {
        "responses",
        "response_receipt",
        "adjudication_receipt",
        "calibration_receipt",
    }:
        raise ValueError("external evidence manifest file inventory drift")
    named_paths = {
        "responses": paths["responses"],
        "response_receipt": paths["response_receipt"],
        "adjudication_receipt": paths["adjudication_receipt"],
        "calibration_receipt": paths["calibration_receipt"],
    }
    for name, path in named_paths.items():
        expected = file_binding(path, relative_to=paths["root"])
        binding_matches(files[name], expected, label=f"external evidence {name}", require_path=True)
    review_requests = manifest.get("review_requests")
    if not isinstance(review_requests, Mapping):
        raise ValueError("external evidence manifest lacks review request binding")
    if review_requests.get("path") is not None:
        raise ValueError("external evidence request binding must not disclose/copy the request file path")
    if not isinstance(review_requests.get("bytes"), int) or review_requests["bytes"] < 1:
        raise ValueError("external evidence request byte binding is invalid")
    require_sha256("external evidence request hash", review_requests.get("sha256"))
    if not CODE_COMMIT_RE.fullmatch(str(manifest.get("code_commit", ""))):
        raise ValueError("external evidence manifest code_commit is invalid")
    if manifest.get("model") != REQUIRED_MODEL:
        raise ValueError("external evidence manifest model is not the frozen review model")
    _forbid_raw_corpus_fields(manifest, label="external evidence manifest")
    for name in ("response_receipt", "adjudication_receipt", "calibration_receipt"):
        _forbid_raw_corpus_fields(read_json(paths[name], label=name), label=name)
    for index, row in enumerate(read_jsonl(paths["responses"], label="external responses"), start=1):
        _forbid_raw_corpus_fields(row, label=f"external responses[{index}]")
    return paths, manifest


def _copy_regular_no_replace(source: Path, destination: Path) -> None:
    source = regular_file(source, label="external evidence source")
    before = file_binding(source)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"immutable import destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        after = file_binding(source)
        if after != before:
            raise ValueError(f"external evidence changed while importing: {source}")
        copied = file_binding(temporary)
        if copied["bytes"] != before["bytes"] or copied["sha256"] != before["sha256"]:
            raise ValueError(f"external evidence copy hash drift: {source}")
        os.link(temporary, destination)
        os.chmod(destination, stat.S_IRUSR | stat.S_IWUSR)
    finally:
        temporary.unlink(missing_ok=True)


def import_external_bundle(
    *, external_dir: Path, destination: Path, receipt_path: Path
) -> dict[str, Any]:
    paths, manifest = validate_bundle(external_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"immutable imported evidence directory already exists: {destination}")
    destination.mkdir(parents=True, mode=0o700)
    try:
        for name in (BUNDLE_MANIFEST_NAME, *BUNDLE_FILE_NAMES):
            _copy_regular_no_replace(paths["root"] / name, destination / name)
    except BaseException:
        # Do not remove an incomplete attempt automatically: immutable-stage
        # failure inspection is preferable to hiding an import problem.
        raise
    imported_paths, imported_manifest = validate_bundle(destination)
    if imported_manifest != manifest:
        raise ValueError("imported external evidence manifest byte/content drift")
    source_manifest = file_binding(paths["manifest"])
    imported_files = {
        name: file_binding(imported_paths[name])
        for name in (
            "manifest",
            "responses",
            "response_receipt",
            "adjudication_receipt",
            "calibration_receipt",
        )
    }
    payload: dict[str, Any] = {
        "schema_version": EXTERNAL_IMPORT_SCHEMA,
        "status": "passed",
        "created_at": utc_now(),
        "external_bundle": source_manifest,
        "external_bundle_manifest_sha256": str(manifest["manifest_sha256"]),
        "imported_directory": str(imported_paths["root"]),
        "imported_files": imported_files,
        "raw_corpus_imported": False,
        "import_scope": "responses_and_execution_receipts_only",
    }
    payload["receipt_sha256"] = receipt_digest(payload)
    write_json_no_replace(receipt_path, payload)
    return payload


def validate_policy(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    policy_document = read_json(path, label="review policy")
    if policy_document.get("schema_version") != review.POLICY_SCHEMA:
        raise ValueError("review policy has unsupported schema")
    policy = policy_document.get("review")
    if not isinstance(policy, Mapping):
        raise ValueError("review policy lacks review object")
    if policy.get("required_model") != REQUIRED_MODEL:
        raise ValueError("review policy model drift")
    if policy.get("model_environment_variable") != "CODEX_REVIEW_MODEL":
        raise ValueError("review policy model environment-variable drift")
    if policy.get("no_model_fallback") is not True:
        raise ValueError("review policy must forbid model fallback")
    if policy.get("reasoning_effort") not in {"low", "medium", "high"}:
        raise ValueError("review policy reasoning effort is invalid")
    return dict(policy), file_binding(path)


def validate_response_schema(path: Path) -> dict[str, Any]:
    value = read_json(path, label="review response schema")
    properties = value.get("properties")
    if (
        value.get("type") != "object"
        or value.get("additionalProperties") is not False
        or not isinstance(properties, Mapping)
        or not isinstance(properties.get("schema_version"), Mapping)
        or properties["schema_version"].get("const") != review.RESPONSE_SCHEMA
    ):
        raise ValueError("review response schema is not the strict v3 schema")
    return file_binding(path)


def validate_initial_requests(
    path: Path,
    *,
    model: str,
    prompt_sha256: str,
    schema_sha256: str,
    code_commit: str,
) -> list[dict[str, Any]]:
    rows = read_jsonl(path, label="Stage 30 review requests")
    review_ids: set[str] = set()
    request_hashes: set[str] = set()
    result: list[dict[str, Any]] = []
    for number, row in enumerate(rows, start=1):
        errors = review._validate_request_binding(row)
        if errors:
            raise ValueError(f"Stage 30 review requests:{number}: {'; '.join(errors)}")
        if row.get("reviewer_slot") not in {"primary", "secondary"}:
            raise ValueError("Stage 30 review requests may contain only primary/secondary slots")
        if row.get("model") != model:
            raise ValueError("Stage 30 request model drift")
        if row.get("prompt_sha256") != prompt_sha256:
            raise ValueError("Stage 30 request prompt hash drift")
        if row.get("response_schema_sha256") != schema_sha256:
            raise ValueError("Stage 30 request response schema hash drift")
        if row.get("code_commit") != code_commit:
            raise ValueError("Stage 30 request code commit drift")
        review_id = str(row["review_id"])
        request_hash = str(row["request_sha256"])
        if review_id in review_ids or request_hash in request_hashes:
            raise ValueError("Stage 30 request inventory repeats an immutable review identity")
        review_ids.add(review_id)
        request_hashes.add(request_hash)
        result.append(dict(row))
    if not any(row["reviewer_slot"] == "primary" for row in result):
        raise ValueError("Stage 30 review request inventory lacks primary reviews")
    primary_samples = [str(row["sample_id"]) for row in result if row["reviewer_slot"] == "primary"]
    if len(primary_samples) != len(set(primary_samples)):
        raise ValueError("Stage 30 primary request inventory repeats a selected sample")
    return result


def validate_packet_manifest(
    path: Path,
    *,
    requests_path: Path,
    requests: Sequence[Mapping[str, Any]],
    policy_binding: Mapping[str, Any],
    prompt_binding: Mapping[str, Any],
    schema_binding: Mapping[str, Any],
    model: str,
    code_commit: str,
) -> dict[str, Any]:
    packet = read_json(path, label="Stage 30 review packet manifest")
    if packet.get("schema_version") != PACKET_MANIFEST_SCHEMA:
        raise ValueError("Stage 30 review packet manifest schema drift")
    if packet.get("status") != "materialized_no_model_invocation":
        raise ValueError("Stage 30 must remain model-free")
    validate_self_hash(packet, field="manifest_sha256", label="Stage 30 review packet manifest")
    expected_requests = file_binding(requests_path)
    binding_matches(packet.get("requests"), expected_requests, label="packet request file")
    inputs = packet.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("Stage 30 review packet manifest lacks input bindings")
    binding_matches(inputs.get("review_policy"), policy_binding, label="packet policy")
    binding_matches(inputs.get("prompt"), prompt_binding, label="packet prompt")
    binding_matches(inputs.get("response_schema"), schema_binding, label="packet response schema")
    execution = packet.get("review_execution")
    if not isinstance(execution, Mapping):
        raise ValueError("Stage 30 review packet manifest lacks review execution binding")
    expected_execution = {
        "model_environment_variable": "CODEX_REVIEW_MODEL",
        "model": model,
        "no_model_fallback": True,
        "model_invocation": "not_run",
        "code_commit": code_commit,
        "prompt_sha256": prompt_binding["sha256"],
        "response_schema_sha256": schema_binding["sha256"],
    }
    for key, expected in expected_execution.items():
        if execution.get(key) != expected:
            raise ValueError(f"Stage 30 review execution binding drift: {key}")
    inventory = packet.get("request_inventory")
    expected_inventory = [
        {
            "review_id": str(row["review_id"]),
            "request_sha256": str(row["request_sha256"]),
            "sample_id": str(row["sample_id"]),
            "reviewer_slot": str(row["reviewer_slot"]),
        }
        for row in requests
    ]
    if inventory != expected_inventory:
        raise ValueError("Stage 30 review packet request inventory drift")
    request_counts = packet.get("request_counts")
    expected_counts = Counter(str(row["reviewer_slot"]) for row in requests)
    if not isinstance(request_counts, Mapping) or request_counts.get("total") != len(requests):
        raise ValueError("Stage 30 review packet request count drift")
    for slot in ("primary", "secondary"):
        if request_counts.get(slot, 0) != expected_counts.get(slot, 0):
            raise ValueError(f"Stage 30 review packet {slot} count drift")
    privacy = packet.get("privacy")
    if (
        not isinstance(privacy, Mapping)
        or privacy.get("raw_canonical_text_in_manifest") is not False
        or privacy.get("raw_source_document_identifier_in_manifest") is not False
        or privacy.get("review_copy_masking") != "high_precision_direct_identifiers_position_preserving"
    ):
        raise ValueError("Stage 30 packet privacy/masking declaration drift")
    attestations = packet.get("review_copy_attestations")
    primary = {str(row["sample_id"]): row for row in requests if row["reviewer_slot"] == "primary"}
    if not isinstance(attestations, list) or len(attestations) != len(primary):
        raise ValueError("Stage 30 review-copy attestation coverage drift")
    attested: set[str] = set()
    for row in attestations:
        if not isinstance(row, Mapping):
            raise ValueError("Stage 30 review-copy attestation must be an object")
        sample_id = str(row.get("stable_uid", ""))
        request = primary.get(sample_id)
        if request is None or sample_id in attested:
            raise ValueError("Stage 30 review-copy attestation has unknown/duplicate sample")
        if (
            row.get("original_text_sha256") != request.get("original_text_sha256")
            or row.get("review_copy_sha256") != request.get("review_copy_sha256")
            or row.get("positions_preserved") is not True
        ):
            raise ValueError("Stage 30 review-copy attestation identity/position drift")
        attested.add(sample_id)
    if attested != set(primary):
        raise ValueError("Stage 30 review-copy attestation misses a primary sample")
    return packet


def _validate_runner_receipt_hash(value: Mapping[str, Any], *, label: str, schema: str) -> None:
    if value.get("schema_version") != schema or value.get("status") != "complete":
        raise ValueError(f"{label}: unsupported schema/status")
    validate_self_hash(value, field="receipt_sha256", label=label)


def _slot_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["reviewer_slot"]) for row in rows).items()))


def _calibration_request_identity(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """Validate a receipt-safe calibration request summary.

    The local runner intentionally excludes ``review_copy`` and comparison
    text from this summary.  Stage 35 verifies the immutable request identity
    and the route coverage without importing that text back from the Mac.
    """

    expected = {
        "review_id",
        "request_sha256",
        "reviewer_slot",
        "sample_id",
        "source_id",
        "source_dataset",
        "source_revision",
        "source_route",
        "sampling_stratum",
        "original_text_sha256",
        "review_copy_sha256",
        "prompt_sha256",
        "response_schema_sha256",
        "model",
        "code_commit",
        "attempt",
    }
    if set(value) != expected:
        raise ValueError(f"{label}: calibration request-summary fields drift")
    for field in (
        "review_id",
        "request_sha256",
        "sample_id",
        "original_text_sha256",
        "review_copy_sha256",
        "prompt_sha256",
        "response_schema_sha256",
    ):
        require_sha256(f"{label}.{field}", value.get(field))
    for field in ("source_id", "source_dataset", "source_revision", "model", "code_commit"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise ValueError(f"{label}.{field} must be non-empty text")
    if value.get("reviewer_slot") not in {"primary", "secondary"}:
        raise ValueError(f"{label}.reviewer_slot drift")
    if value.get("source_route") not in review.ALLOWED_ROUTES:
        raise ValueError(f"{label}.source_route drift")
    if value.get("sampling_stratum") not in review.STRATA:
        raise ValueError(f"{label}.sampling_stratum drift")
    if not isinstance(value.get("attempt"), int) or isinstance(value["attempt"], bool) or value["attempt"] < 1:
        raise ValueError(f"{label}.attempt drift")
    return dict(value)


def validate_calibration_receipt(
    *,
    calibration_receipt: Mapping[str, Any],
    requests_path: Path,
    policy_binding: Mapping[str, Any],
    prompt_binding: Mapping[str, Any],
    schema_binding: Mapping[str, Any],
    model: str,
    code_commit: str,
    expected_routes: set[str],
) -> dict[str, Any]:
    """Require a passed, route-spanning calibration before full review closure."""

    if (
        calibration_receipt.get("schema_version") != CALIBRATION_RECEIPT_SCHEMA
        or calibration_receipt.get("status") != "passed"
    ):
        raise ValueError("Codex calibration receipt must be a passed v3 calibration receipt")
    validate_self_hash(calibration_receipt, field="receipt_sha256", label="Codex calibration receipt")
    if calibration_receipt.get("input_scope") != "compact_v3_request_jsonl_only_no_corpus_files":
        raise ValueError("Codex calibration receipt input scope drift")
    if calibration_receipt.get("corpus_files_read") is not False:
        raise ValueError("Codex calibration receipt claims corpus files were read")
    if calibration_receipt.get("prompt_schema_frozen_for_full_review") is not True:
        raise ValueError("Codex calibration did not freeze prompt/schema for full review")
    if calibration_receipt.get("prompt_or_schema_tuning_performed") is not False:
        raise ValueError("Codex calibration cannot tune prompt/schema in the frozen run")
    if calibration_receipt.get("admission_decision_performed") is not False:
        raise ValueError("Codex calibration cannot decide source admission")
    if calibration_receipt.get("primary_secondary_sessions_separated") is not True:
        raise ValueError("Codex calibration lost primary/secondary isolation")
    inputs = calibration_receipt.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("Codex calibration receipt lacks input bindings")
    binding_matches(inputs.get("requests"), file_binding(requests_path), label="Codex calibration requests")
    binding_matches(inputs.get("policy"), policy_binding, label="Codex calibration policy")
    binding_matches(inputs.get("prompt"), prompt_binding, label="Codex calibration prompt")
    binding_matches(inputs.get("response_schema"), schema_binding, label="Codex calibration response schema")
    calibration_model = calibration_receipt.get("model")
    expected_model = {
        "environment_variable": "CODEX_REVIEW_MODEL",
        "required_model": model,
        "accepted_model": model,
        "no_fallback": True,
    }
    if calibration_model != expected_model:
        raise ValueError("Codex calibration model/fallback binding drift")
    assessment = calibration_receipt.get("assessment")
    if not isinstance(assessment, Mapping) or assessment.get("status") != "passed":
        raise ValueError("Codex calibration assessment is not passed")
    if assessment.get("failure_count") != 0:
        raise ValueError("Codex calibration reports failed consistency cases")
    criteria = assessment.get("criteria")
    expected_criteria = {
        "all_primary_secondary_response_identities_valid": True,
        "all_represented_logical_routes_sampled": True,
        "low_confidence_or_uncertain_is_a_failure": True,
        "material_score_issue_or_recommendation_disagreement_is_a_failure": True,
        "prompt_or_schema_tuning_performed": False,
        "admission_decision_performed": False,
    }
    if criteria != expected_criteria:
        raise ValueError("Codex calibration criteria drift")
    route_rows = assessment.get("logical_route_coverage")
    if not isinstance(route_rows, list) or not route_rows:
        raise ValueError("Codex calibration lacks logical-route coverage")
    observed_routes: set[str] = set()
    for row in route_rows:
        if not isinstance(row, Mapping):
            raise ValueError("Codex calibration route row is malformed")
        route = row.get("logical_source_route")
        if route not in review.ALLOWED_ROUTES or route in observed_routes:
            raise ValueError("Codex calibration route coverage is malformed")
        observed_routes.add(str(route))
        if not isinstance(row.get("pair_count"), int) or row["pair_count"] < 1:
            raise ValueError("Codex calibration route has no paired review")
        sources = row.get("source_ids")
        if not isinstance(sources, list) or not sources or len(sources) != len(set(sources)):
            raise ValueError("Codex calibration route source balance is malformed")
        if row.get("distinct_source_count") != len(sources):
            raise ValueError("Codex calibration route source-count drift")
    if observed_routes != expected_routes:
        raise ValueError(
            f"Codex calibration route coverage drift: expected {sorted(expected_routes)}, observed {sorted(observed_routes)}"
        )
    selection = calibration_receipt.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("Codex calibration receipt lacks selection")
    if selection.get("algorithm") != "route_spanning_source_round_robin_request_hash_v1":
        raise ValueError("Codex calibration selection algorithm drift")
    pairs = selection.get("selected_pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Codex calibration selection has no pairs")
    if selection.get("selected_pair_inventory_sha256") != sha256_json(pairs):
        raise ValueError("Codex calibration selected pair inventory hash drift")
    selected_routes: set[str] = set()
    selected_samples: set[str] = set()
    for index, pair in enumerate(pairs):
        if not isinstance(pair, Mapping) or set(pair) != {"primary_request", "secondary_request"}:
            raise ValueError("Codex calibration pair shape drift")
        primary = _calibration_request_identity(pair["primary_request"], label=f"calibration[{index}].primary")
        secondary = _calibration_request_identity(pair["secondary_request"], label=f"calibration[{index}].secondary")
        if primary["reviewer_slot"] != "primary" or secondary["reviewer_slot"] != "secondary":
            raise ValueError("Codex calibration pair reviewer-slot drift")
        for field in (
            "sample_id",
            "source_id",
            "source_dataset",
            "source_revision",
            "source_route",
            "sampling_stratum",
            "original_text_sha256",
            "review_copy_sha256",
            "prompt_sha256",
            "response_schema_sha256",
            "model",
            "code_commit",
            "attempt",
        ):
            if primary[field] != secondary[field]:
                raise ValueError(f"Codex calibration pair identity drift: {field}")
        if primary["model"] != model or primary["code_commit"] != code_commit:
            raise ValueError("Codex calibration pair model/code drift")
        selected_routes.add(str(primary["source_route"]))
        if primary["sample_id"] in selected_samples:
            raise ValueError("Codex calibration selection repeats a primary sample")
        selected_samples.add(str(primary["sample_id"]))
    if selected_routes != expected_routes:
        raise ValueError("Codex calibration selected pairs do not span every logical route")
    return dict(calibration_receipt)


def validate_runner_outputs(
    *,
    bundle_paths_value: Mapping[str, Path],
    requests_path: Path,
    requests: Sequence[Mapping[str, Any]],
    policy_binding: Mapping[str, Any],
    prompt_binding: Mapping[str, Any],
    schema_binding: Mapping[str, Any],
    model: str,
    code_commit: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    responses_path = bundle_paths_value["responses"]
    response_receipt_path = bundle_paths_value["response_receipt"]
    adjudication_receipt_path = bundle_paths_value["adjudication_receipt"]
    calibration_receipt_path = bundle_paths_value["calibration_receipt"]
    responses = read_jsonl(responses_path, label="Codex review responses")
    response_receipt = read_json(response_receipt_path, label="Codex response execution receipt")
    adjudication_receipt = read_json(adjudication_receipt_path, label="Codex adjudication receipt")
    calibration_receipt = read_json(calibration_receipt_path, label="Codex calibration receipt")
    _validate_runner_receipt_hash(
        response_receipt, label="Codex response execution receipt", schema=RESPONSE_RECEIPT_SCHEMA
    )
    _validate_runner_receipt_hash(
        adjudication_receipt, label="Codex adjudication receipt", schema=ADJUDICATION_RECEIPT_SCHEMA
    )
    if response_receipt.get("input_scope") != "compact_v3_request_jsonl_only_no_corpus_files":
        raise ValueError("Codex response receipt did not declare compact-v3-only input scope")
    if response_receipt.get("corpus_files_read") is not False:
        raise ValueError("Codex response receipt claims corpus files were read")
    receipt_inputs = response_receipt.get("inputs")
    if not isinstance(receipt_inputs, Mapping):
        raise ValueError("Codex response receipt lacks input bindings")
    binding_matches(receipt_inputs.get("requests"), file_binding(requests_path), label="Codex receipt requests")
    binding_matches(receipt_inputs.get("policy"), policy_binding, label="Codex receipt policy")
    binding_matches(receipt_inputs.get("prompt"), prompt_binding, label="Codex receipt prompt")
    binding_matches(receipt_inputs.get("response_schema"), schema_binding, label="Codex receipt response schema")
    normalized_schema_hash = receipt_inputs.get("response_schema", {}).get("normalized_execution_schema_sha256")
    require_sha256("Codex normalized execution schema hash", normalized_schema_hash)
    model_receipt = response_receipt.get("model")
    if not isinstance(model_receipt, Mapping) or model_receipt != {
        "environment_variable": "CODEX_REVIEW_MODEL",
        "required_model": model,
        "accepted_model": model,
        "no_fallback": True,
    }:
        raise ValueError("Codex response receipt model identity/fallback binding drift")
    preflight = response_receipt.get("preflight")
    if not isinstance(preflight, Mapping) or preflight.get("synthetic_fixture") is not True:
        raise ValueError("Codex response receipt lacks the required synthetic preflight")
    if preflight.get("accepted_model") != model:
        raise ValueError("Codex preflight model drift")
    for name in ("request_sha256", "review_id", "response_sha256", "execution_schema_sha256"):
        require_sha256(f"Codex preflight {name}", preflight.get(name))
    settings = response_receipt.get("execution_settings")
    if not isinstance(settings, Mapping) or settings.get("model") != model:
        raise ValueError("Codex response receipt execution settings drift")
    if settings.get("sandbox") != "read-only" or settings.get("ephemeral") is not True:
        raise ValueError("Codex response receipt lost isolated read-only execution settings")
    if settings.get("schema_normalizer") != "agent1_v3_openai_schema_compat_v1":
        raise ValueError("Codex response receipt schema-normalizer binding drift")
    require_sha256("Codex execution settings hash", settings.get("settings_sha256"))
    response_binding = file_binding(responses_path)
    binding_matches(response_receipt.get("responses"), response_binding, label="Codex response receipt output")
    if response_receipt.get("adjudication_receipt") is None:
        raise ValueError("Codex response receipt lacks adjudication receipt binding")
    binding_matches(
        response_receipt["adjudication_receipt"],
        file_binding(adjudication_receipt_path),
        label="Codex response receipt adjudication binding",
    )
    calibration = validate_calibration_receipt(
        calibration_receipt=calibration_receipt,
        requests_path=requests_path,
        policy_binding=policy_binding,
        prompt_binding=prompt_binding,
        schema_binding=schema_binding,
        model=model,
        code_commit=code_commit,
        expected_routes={str(row["source_route"]) for row in requests if row["reviewer_slot"] == "primary"},
    )
    calibration_binding = file_binding(calibration_receipt_path)
    binding_matches(
        response_receipt.get("passed_calibration_receipt"),
        calibration_binding,
        label="Codex response receipt calibration binding",
    )
    binding_matches(
        adjudication_receipt.get("passed_calibration_receipt"),
        calibration_binding,
        label="Codex adjudication receipt calibration binding",
    )
    initial_slot_counts = _slot_counts(requests)
    request_counts = response_receipt.get("requests")
    if not isinstance(request_counts, Mapping) or request_counts.get("initial_rows") != len(requests):
        raise ValueError("Codex response receipt initial request count drift")
    if request_counts.get("initial_slot_counts") != initial_slot_counts:
        raise ValueError("Codex response receipt initial slot counts drift")

    seen_response_ids: set[str] = set()
    initial_by_id = {str(row["review_id"]): row for row in requests}
    for number, response in enumerate(responses, start=1):
        review_id = response.get("review_id")
        if not isinstance(review_id, str) or review_id in seen_response_ids:
            raise ValueError(f"Codex response {number}: duplicate/malformed review_id")
        seen_response_ids.add(review_id)
        if response.get("reviewer_slot") != "adjudicator":
            request = initial_by_id.get(review_id)
            if request is None:
                raise ValueError(f"Codex response {number}: unrequested initial response")
            review.assert_valid_review_response(response, request)
        else:
            errors = review.validate_review_response(response)
            if errors:
                raise ValueError(f"Codex adjudicator response {number}: {'; '.join(errors)}")
        if (
            response.get("model") != model
            or response.get("prompt_sha256") != prompt_binding["sha256"]
            or response.get("response_schema_sha256") != schema_binding["sha256"]
            or response.get("code_commit") != code_commit
        ):
            raise ValueError(f"Codex response {number}: model/prompt/schema/code binding drift")
    final_manifest = review.build_adjudication_manifest(requests, responses)
    review.assert_adjudication_closed(final_manifest)
    expected_slot_counts = _slot_counts(responses)
    response_rows = response_receipt.get("responses")
    if not isinstance(response_rows, Mapping) or response_rows.get("rows") != len(responses):
        raise ValueError("Codex response receipt response row count drift")
    if response_rows.get("slot_counts") != expected_slot_counts:
        raise ValueError("Codex response receipt response slot count drift")
    expected_adjudication_rows = expected_slot_counts.get("adjudicator", 0)
    if request_counts.get("adjudication_rows") != expected_adjudication_rows:
        raise ValueError("Codex response receipt adjudication request count drift")

    if adjudication_receipt.get("model") != model:
        raise ValueError("Codex adjudication receipt model drift")
    if adjudication_receipt.get("initial_request_rows") != len(requests):
        raise ValueError("Codex adjudication receipt initial request count drift")
    if adjudication_receipt.get("adjudication_request_rows") != expected_adjudication_rows:
        raise ValueError("Codex adjudication receipt adjudication request count drift")
    if adjudication_receipt.get("response_rows") != len(responses):
        raise ValueError("Codex adjudication receipt response count drift")
    if adjudication_receipt.get("response_slot_counts") != expected_slot_counts:
        raise ValueError("Codex adjudication receipt slot count drift")
    binding_matches(adjudication_receipt.get("responses"), response_binding, label="Codex adjudication responses")
    if adjudication_receipt.get("final_adjudication_manifest") != final_manifest:
        raise ValueError("Codex adjudication receipt final manifest drift")
    if adjudication_receipt.get("no_adjudication_noop") != (expected_adjudication_rows == 0):
        raise ValueError("Codex adjudication receipt no-op declaration drift")
    pending_before = adjudication_receipt.get("pending_before_execution")
    if not isinstance(pending_before, Mapping):
        raise ValueError("Codex adjudication receipt lacks pre-execution pending manifest")
    if pending_before.get("pending_count") is None or pending_before.get("case_count") is None:
        raise ValueError("Codex adjudication receipt pending counters are incomplete")
    return responses, response_receipt, adjudication_receipt, calibration, final_manifest


def validate_closure(
    *,
    run_id: str,
    requests_path: Path,
    packet_manifest_path: Path,
    external_dir: Path,
    policy_path: Path,
    prompt_path: Path,
    schema_path: Path,
    code_commit: str,
    output: Path | None = None,
) -> dict[str, Any]:
    if not CODE_COMMIT_RE.fullmatch(code_commit):
        raise ValueError("code commit must be an exact 40-character lowercase git SHA")
    policy, policy_binding = validate_policy(policy_path)
    prompt_binding = file_binding(prompt_path)
    if not prompt_path.read_text(encoding="utf-8").strip():
        raise ValueError("review prompt is empty")
    schema_binding = validate_response_schema(schema_path)
    requests = validate_initial_requests(
        requests_path,
        model=str(policy["required_model"]),
        prompt_sha256=str(prompt_binding["sha256"]),
        schema_sha256=str(schema_binding["sha256"]),
        code_commit=code_commit,
    )
    packet = validate_packet_manifest(
        packet_manifest_path,
        requests_path=requests_path,
        requests=requests,
        policy_binding=policy_binding,
        prompt_binding=prompt_binding,
        schema_binding=schema_binding,
        model=str(policy["required_model"]),
        code_commit=code_commit,
    )
    paths, external_manifest = validate_bundle(external_dir)
    if external_manifest.get("run_id") != run_id:
        raise ValueError("external review evidence run_id drift")
    if external_manifest.get("code_commit") != code_commit:
        raise ValueError("external review evidence code commit drift")
    requests_binding = file_binding(requests_path)
    binding_matches(external_manifest.get("review_requests"), requests_binding, label="external review requests")
    if external_manifest.get("model") != policy["required_model"]:
        raise ValueError("external review evidence model drift")
    responses, response_receipt, adjudication_receipt, calibration_receipt, final_manifest = validate_runner_outputs(
        bundle_paths_value=paths,
        requests_path=requests_path,
        requests=requests,
        policy_binding=policy_binding,
        prompt_binding=prompt_binding,
        schema_binding=schema_binding,
        model=str(policy["required_model"]),
        code_commit=code_commit,
    )
    primary = [row for row in requests if row["reviewer_slot"] == "primary"]
    payload: dict[str, Any] = {
        "schema_version": CLOSURE_SCHEMA,
        "status": "passed",
        "created_at": utc_now(),
        "run_id": run_id,
        "code_commit": code_commit,
        "inputs": {
            "review_requests": requests_binding,
            "review_packet_manifest": file_binding(packet_manifest_path),
            "review_policy": policy_binding,
            "review_prompt": prompt_binding,
            "review_response_schema": schema_binding,
            "external_evidence_manifest": file_binding(paths["manifest"]),
            "external_responses": file_binding(paths["responses"]),
            "external_response_receipt": file_binding(paths["response_receipt"]),
            "external_adjudication_receipt": file_binding(paths["adjudication_receipt"]),
            "external_calibration_receipt": file_binding(paths["calibration_receipt"]),
        },
        "packet": {
            "manifest_sha256": str(packet["manifest_sha256"]),
            "model_invocation": "not_run",
            "primary_request_rows": len(primary),
            "secondary_request_rows": len(requests) - len(primary),
            "primary_sample_inventory_sha256": sha256_json(sorted(str(row["sample_id"]) for row in primary)),
        },
        "review_execution": {
            "model_environment_variable": "CODEX_REVIEW_MODEL",
            "required_model": str(policy["required_model"]),
            "accepted_model": str(response_receipt["model"]["accepted_model"]),
            "no_model_fallback": True,
            "prompt_sha256": str(prompt_binding["sha256"]),
            "response_schema_sha256": str(schema_binding["sha256"]),
            "normalized_execution_schema_sha256": str(
                response_receipt["inputs"]["response_schema"]["normalized_execution_schema_sha256"]
            ),
            "preflight_execution_schema_sha256": str(response_receipt["preflight"]["execution_schema_sha256"]),
            "calibration_receipt_sha256": str(calibration_receipt["receipt_sha256"]),
            "calibration_prompt_schema_frozen_for_full_review": True,
            "code_commit": code_commit,
        },
        "response_closure": {
            "response_rows": len(responses),
            "response_slot_counts": _slot_counts(responses),
            "response_execution_receipt_sha256": str(response_receipt["receipt_sha256"]),
            "adjudication_execution_receipt_sha256": str(adjudication_receipt["receipt_sha256"]),
            "final_adjudication_manifest": final_manifest,
            "pending_adjudication_count": 0,
        },
        "privacy": {
            "external_bundle_contains_raw_corpus": False,
            "external_bundle_contents": "responses_and_execution_receipts_only",
            "stage30_packet_model_invocation": "not_run",
        },
        "admission_decision": "not_evaluated_in_stage35",
    }
    payload["closure_sha256"] = sha256_json(payload)
    if output is not None:
        write_json_no_replace(output, payload)
    return payload


def materialize_masked_sample(
    *,
    requests_path: Path,
    packet_manifest_path: Path,
    closure_path: Path,
    output: Path,
    receipt: Path,
) -> dict[str, Any]:
    closure = read_json(closure_path, label="Stage 35 evidence closure")
    if closure.get("schema_version") != CLOSURE_SCHEMA or closure.get("status") != "passed":
        raise ValueError("Stage 35 evidence closure schema/status drift")
    closure_without_hash = dict(closure)
    observed_hash = closure_without_hash.pop("closure_sha256", None)
    if observed_hash != sha256_json(closure_without_hash):
        raise ValueError("Stage 35 evidence closure hash drift")
    closure_inputs = closure.get("inputs")
    if not isinstance(closure_inputs, Mapping):
        raise ValueError("Stage 35 evidence closure lacks inputs")
    binding_matches(closure_inputs.get("review_requests"), file_binding(requests_path), label="masked sample requests")
    binding_matches(
        closure_inputs.get("review_packet_manifest"),
        file_binding(packet_manifest_path),
        label="masked sample packet manifest",
    )
    requests = read_jsonl(requests_path, label="Stage 30 review requests")
    packet = read_json(packet_manifest_path, label="Stage 30 review packet manifest")
    if packet.get("schema_version") != PACKET_MANIFEST_SCHEMA:
        raise ValueError("Stage 30 packet schema drift while materializing masked sample")
    attestations = packet.get("review_copy_attestations")
    if not isinstance(attestations, list):
        raise ValueError("Stage 30 packet lacks review-copy attestations")
    attestation_by_sample = {str(row.get("stable_uid")): row for row in attestations if isinstance(row, Mapping)}
    if len(attestation_by_sample) != len(attestations):
        raise ValueError("Stage 30 review-copy attestations repeat/malformed sample identity")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for number, request in enumerate(requests, start=1):
        if request.get("reviewer_slot") != "primary":
            continue
        errors = review._validate_request_binding(request)
        if errors:
            raise ValueError(f"Stage 30 primary request {number}: {'; '.join(errors)}")
        sample_id = str(request["sample_id"])
        if sample_id in seen:
            raise ValueError("Stage 30 primary request sample repeats")
        review_copy = request.get("review_copy")
        if not isinstance(review_copy, str) or hashlib.sha256(review_copy.encode("utf-8")).hexdigest() != request.get("review_copy_sha256"):
            raise ValueError(f"Stage 30 primary request {sample_id}: masked review-copy hash drift")
        attestation = attestation_by_sample.get(sample_id)
        if (
            not isinstance(attestation, Mapping)
            or attestation.get("positions_preserved") is not True
            or attestation.get("original_text_sha256") != request.get("original_text_sha256")
            or attestation.get("review_copy_sha256") != request.get("review_copy_sha256")
        ):
            raise ValueError(f"Stage 30 primary request {sample_id}: masking attestation drift")
        rows.append(
            {
                "schema_version": MASKED_SAMPLE_SCHEMA,
                "sample_id": sample_id,
                "source_id": str(request["source_id"]),
                "source_dataset": str(request["source_dataset"]),
                "source_revision": str(request["source_revision"]),
                "source_route": str(request["source_route"]),
                "sampling_stratum": str(request["sampling_stratum"]),
                "original_text_sha256": str(request["original_text_sha256"]),
                "review_copy_sha256": str(request["review_copy_sha256"]),
                "review_request_sha256": str(request["request_sha256"]),
                "text_variant": "high_precision_identifier_masked_review_sample",
                "review_copy": review_copy,
            }
        )
        seen.add(sample_id)
    if not rows:
        raise ValueError("cannot materialize an empty primary review sample")
    expected_inventory = closure.get("packet", {}).get("primary_sample_inventory_sha256")
    actual_inventory = sha256_json(sorted(seen))
    if expected_inventory != actual_inventory:
        raise ValueError("masked sample primary inventory does not close against Stage 35 evidence closure")
    write_jsonl_no_replace(output, rows)
    output_binding = file_binding(output)
    inventory = [
        {
            "sample_id": row["sample_id"],
            "source_id": row["source_id"],
            "sampling_stratum": row["sampling_stratum"],
            "original_text_sha256": row["original_text_sha256"],
            "review_copy_sha256": row["review_copy_sha256"],
            "review_request_sha256": row["review_request_sha256"],
        }
        for row in rows
    ]
    payload: dict[str, Any] = {
        "schema_version": MASKED_SAMPLE_RECEIPT_SCHEMA,
        "status": "passed",
        "created_at": utc_now(),
        "text_variant": "high_precision_identifier_masked_review_sample",
        "raw_corpus_included": False,
        "inputs": {
            "review_requests": file_binding(requests_path),
            "review_packet_manifest": file_binding(packet_manifest_path),
            "quality_review_evidence_closure": file_binding(closure_path),
        },
        "output": {**output_binding, "rows": len(rows)},
        "primary_sample_count": len(rows),
        "primary_sample_inventory": inventory,
        "primary_sample_inventory_sha256": sha256_json(inventory),
        "selection_source": "exact_stage30_primary_review_requests",
        "admission_decision": "not_evaluated_in_stage35",
    }
    payload["receipt_sha256"] = receipt_digest(payload)
    write_json_no_replace(receipt, payload)
    return payload


def package_external(
    *,
    run_id: str,
    requests_path: Path,
    responses_path: Path,
    response_receipt_path: Path,
    adjudication_receipt_path: Path,
    calibration_receipt_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Create the transferable no-raw-corpus package on the local reviewer host."""

    requests_binding = file_binding(requests_path)
    response_receipt = read_json(response_receipt_path, label="response execution receipt")
    adjudication_receipt = read_json(adjudication_receipt_path, label="adjudication execution receipt")
    calibration_receipt = read_json(calibration_receipt_path, label="calibration receipt")
    _validate_runner_receipt_hash(response_receipt, label="response execution receipt", schema=RESPONSE_RECEIPT_SCHEMA)
    _validate_runner_receipt_hash(adjudication_receipt, label="adjudication execution receipt", schema=ADJUDICATION_RECEIPT_SCHEMA)
    if (
        calibration_receipt.get("schema_version") != CALIBRATION_RECEIPT_SCHEMA
        or calibration_receipt.get("status") != "passed"
    ):
        raise ValueError("cannot package a non-passed calibration receipt")
    validate_self_hash(calibration_receipt, field="receipt_sha256", label="calibration receipt")
    model = response_receipt.get("model", {}).get("accepted_model")
    if model != REQUIRED_MODEL:
        raise ValueError("cannot package review evidence for a non-frozen model")
    rows = read_jsonl(responses_path, label="responses")
    code_commits = {str(row.get("code_commit", "")) for row in rows}
    if len(code_commits) != 1 or not CODE_COMMIT_RE.fullmatch(next(iter(code_commits))):
        raise ValueError("responses do not bind one exact code commit")
    binding_matches(response_receipt.get("inputs", {}).get("requests"), requests_binding, label="package response receipt requests")
    binding_matches(response_receipt.get("responses"), file_binding(responses_path), label="package response receipt output")
    binding_matches(adjudication_receipt.get("responses"), file_binding(responses_path), label="package adjudication responses")
    calibration_binding = file_binding(calibration_receipt_path)
    binding_matches(
        response_receipt.get("passed_calibration_receipt"),
        calibration_binding,
        label="package response receipt calibration",
    )
    binding_matches(
        adjudication_receipt.get("passed_calibration_receipt"),
        calibration_binding,
        label="package adjudication receipt calibration",
    )
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"external evidence output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, mode=0o700)
    source_paths = {
        RESPONSES_NAME: responses_path,
        RESPONSE_RECEIPT_NAME: response_receipt_path,
        ADJUDICATION_RECEIPT_NAME: adjudication_receipt_path,
        CALIBRATION_RECEIPT_NAME: calibration_receipt_path,
    }
    for target_name, source in source_paths.items():
        _copy_regular_no_replace(source, output_dir / target_name)
    files = {
        "responses": file_binding(output_dir / RESPONSES_NAME, relative_to=output_dir),
        "response_receipt": file_binding(output_dir / RESPONSE_RECEIPT_NAME, relative_to=output_dir),
        "adjudication_receipt": file_binding(output_dir / ADJUDICATION_RECEIPT_NAME, relative_to=output_dir),
        "calibration_receipt": file_binding(output_dir / CALIBRATION_RECEIPT_NAME, relative_to=output_dir),
    }
    payload: dict[str, Any] = {
        "schema_version": EXTERNAL_BUNDLE_SCHEMA,
        "status": "complete",
        "created_at": utc_now(),
        "run_id": run_id,
        "code_commit": next(iter(code_commits)),
        "model": REQUIRED_MODEL,
        "raw_corpus_included": False,
        "bundle_contents": "responses_and_execution_receipts_only",
        "review_requests": {"bytes": requests_binding["bytes"], "sha256": requests_binding["sha256"]},
        "files": files,
    }
    payload["manifest_sha256"] = manifest_digest(payload)
    write_json_no_replace(output_dir / BUNDLE_MANIFEST_NAME, payload)
    validate_bundle(output_dir)
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)

    package = sub.add_parser("package-external", help="package local Codex outputs without review/corpus text")
    package.add_argument("--run-id", required=True)
    package.add_argument("--requests", type=Path, required=True)
    package.add_argument("--responses", type=Path, required=True)
    package.add_argument("--response-receipt", type=Path, required=True)
    package.add_argument("--adjudication-receipt", type=Path, required=True)
    package.add_argument("--calibration-receipt", type=Path, required=True)
    package.add_argument("--output-dir", type=Path, required=True)
    package.set_defaults(command_func=_cmd_package)

    imported = sub.add_parser("import-external", help="copy a strict external evidence bundle into Stage 35")
    imported.add_argument("--external-evidence-dir", type=Path, required=True)
    imported.add_argument("--destination", type=Path, required=True)
    imported.add_argument("--receipt", type=Path, required=True)
    imported.set_defaults(command_func=_cmd_import)

    closure = sub.add_parser("validate-closure", help="verify exact request/response/adjudication closure")
    closure.add_argument("--run-id", required=True)
    closure.add_argument("--requests", type=Path, required=True)
    closure.add_argument("--packet-manifest", type=Path, required=True)
    closure.add_argument("--external-evidence-dir", type=Path, required=True)
    closure.add_argument("--policy", type=Path, required=True)
    closure.add_argument("--prompt", type=Path, required=True)
    closure.add_argument("--response-schema", type=Path, required=True)
    closure.add_argument("--code-commit", required=True)
    closure.add_argument("--output", type=Path, required=True)
    closure.set_defaults(command_func=_cmd_closure)

    sample = sub.add_parser("materialize-masked-sample", help="emit the exact v3 masked primary review sample")
    sample.add_argument("--requests", type=Path, required=True)
    sample.add_argument("--packet-manifest", type=Path, required=True)
    sample.add_argument("--closure", type=Path, required=True)
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--receipt", type=Path, required=True)
    sample.set_defaults(command_func=_cmd_sample)
    return result


def _cmd_package(args: argparse.Namespace) -> int:
    payload = package_external(
        run_id=args.run_id,
        requests_path=args.requests,
        responses_path=args.responses,
        response_receipt_path=args.response_receipt,
        adjudication_receipt_path=args.adjudication_receipt,
        calibration_receipt_path=args.calibration_receipt,
        output_dir=args.output_dir,
    )
    print(json.dumps({"ok": True, "manifest_sha256": payload["manifest_sha256"], "output": str(args.output_dir)}))
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    payload = import_external_bundle(
        external_dir=args.external_evidence_dir, destination=args.destination, receipt_path=args.receipt
    )
    print(json.dumps({"ok": True, "receipt_sha256": payload["receipt_sha256"], "output": str(args.receipt)}))
    return 0


def _cmd_closure(args: argparse.Namespace) -> int:
    payload = validate_closure(
        run_id=args.run_id,
        requests_path=args.requests,
        packet_manifest_path=args.packet_manifest,
        external_dir=args.external_evidence_dir,
        policy_path=args.policy,
        prompt_path=args.prompt,
        schema_path=args.response_schema,
        code_commit=args.code_commit,
        output=args.output,
    )
    print(json.dumps({"ok": True, "closure_sha256": payload["closure_sha256"], "output": str(args.output)}))
    return 0


def _cmd_sample(args: argparse.Namespace) -> int:
    payload = materialize_masked_sample(
        requests_path=args.requests,
        packet_manifest_path=args.packet_manifest,
        closure_path=args.closure,
        output=args.output,
        receipt=args.receipt,
    )
    print(json.dumps({"ok": True, "receipt_sha256": payload["receipt_sha256"], "output": str(args.output)}))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.command_func(args))


if __name__ == "__main__":
    raise SystemExit(main())
