#!/usr/bin/env python3
"""Materialize deterministic, privacy-preserving Agent 1 v3 review packets.

The full GlossAPI scan is evidence, not a source of review text.  This program
uses that evidence to choose 60 random / 20 risk / 20 cluster documents per
ordinary source (and the frozen 200-document policy for named large sources),
then streams canonical Parquet only to recover the selected documents.  The
result contains position-preserving, high-confidence-masked review copies and
strict request identities, but deliberately does not invoke Codex.

The implementation is a narrow bridge around :mod:`agent1_v3_review`: that
module remains the source of truth for route validation, selection, masking,
secondary selection, model validation, and request hashing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SCRIPT_DIR))

import agent1_v3_review as review  # noqa: E402


PACKET_MANIFEST_SCHEMA = "agent1_v3_review_packet_manifest_v1"
PACKET_VERSION = "agent1_v3_review_packet_materializer_v1"
REQUESTS_SCHEMA = review.REQUEST_SCHEMA
CODE_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_COMPARISON_BUNDLE_SIZE = 4
QUALITY_DOCUMENT_SCHEMA = "dataset_quality_document_v1"
QUALITY_DOCUMENT_ID_NAMESPACE = "dataset-quality-document-v1"
EVIDENCE_KEY_CANONICAL_STABLE_UID = "canonical_stable_uid"
EVIDENCE_KEY_QUALITY_DOCUMENT_ID = "dataset_quality_document_id"
EXPECTED_MODEL_ENVIRONMENT_VARIABLE = "CODEX_REVIEW_MODEL"
EXPECTED_REASONING_EFFORT = "low"
EXPECTED_REVIEW_COPY_POLICY = (
    "mask_high_confidence_direct_identifiers_preserve_position_and_original_hash"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def quality_document_id(stable_uid: str) -> str:
    """Return the opaque identifier emitted by the quality full scan.

    The quality profiler deliberately publishes ``document_id`` rather than a
    corpus stable UID.  It is still a deterministic one-to-one selection key
    for a canonical stable UID, and is resolved only while materializing the
    compact review packet.
    """

    _require_sha256("canonical stable_uid", stable_uid)
    return sha256_text(f"{QUALITY_DOCUMENT_ID_NAMESPACE}\0{stable_uid}")


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or not HEX_SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_nonempty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _binding(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise FileNotFoundError(f"required non-empty input is missing: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _tree_binding(root: Path, files: Sequence[tuple[Path, Path]]) -> dict[str, object]:
    inventory = [
        {
            "path": relative.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path, relative in files
    ]
    return {
        "root": str(root.resolve()),
        "files": inventory,
        "inventory_sha256": sha256_json(inventory),
    }


def _atomic_publish(path: Path, data: bytes, *, mode: int) -> None:
    """Atomically publish a new immutable file without replacing an existing one."""

    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        # link() has no replacement semantics, unlike os.replace().  It keeps
        # the final review artifact immutable even if two operators race.
        os.link(temporary, path)
        os.chmod(path, mode)
    except BaseException:
        raise
    finally:
        temporary.unlink(missing_ok=True)


def _write_jsonl_no_replace(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    payload = b"".join(
        (canonical_json(dict(row)) + "\n").encode("utf-8") for row in rows
    )
    if not payload:
        raise ValueError("review request packet must contain at least one request")
    _atomic_publish(path, payload, mode=0o600)


def _write_json_no_replace(path: Path, value: Mapping[str, object]) -> None:
    _atomic_publish(
        path,
        (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
        mode=0o600,
    )


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc


def _discover_parquet(root: Path) -> list[tuple[Path, Path]]:
    root = root.resolve()
    if root.is_file():
        if root.suffix.lower() != ".parquet":
            raise ValueError(f"Parquet input must end in .parquet: {root}")
        return [(root, Path(root.name))]
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = sorted(path for path in root.rglob("*.parquet") if path.is_file())
    if not files:
        raise FileNotFoundError(f"no Parquet files beneath {root}")
    return [(path, path.relative_to(root)) for path in files]


def _discover_evidence(path: Path) -> list[tuple[Path, Path]]:
    path = path.resolve()
    if path.is_file():
        return [(path, Path(path.name))]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(
        file
        for file in path.rglob("*")
        if file.is_file() and file.suffix.lower() in {".json", ".jsonl", ".parquet"}
    )
    if not files:
        raise FileNotFoundError(f"no JSON/JSONL/Parquet full-scan evidence beneath {path}")
    return [(file, file.relative_to(path)) for file in files]


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: full-scan evidence row must be an object")
            yield row


def _iter_evidence_file(path: Path) -> Iterator[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from _iter_jsonl(path)
        return
    if suffix == ".json":
        value = _read_json(path)
        if isinstance(value, Mapping):
            value = value.get("rows")
        if not isinstance(value, list):
            raise ValueError(f"{path}: JSON evidence must be a list or an object containing rows")
        for row_number, row in enumerate(value, 1):
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{row_number}: full-scan evidence row must be an object")
            yield row
        return
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - runner image dependency
            raise RuntimeError("Parquet full-scan evidence requires pyarrow") from exc
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=8192, use_threads=False):
            for row in batch.to_pylist():
                if not isinstance(row, dict):
                    raise ValueError(f"{path}: full-scan evidence row must be an object")
                yield row
        return
    raise ValueError(f"unsupported evidence suffix: {path}")


def _load_full_scan_evidence(path: Path) -> tuple[list[dict[str, Any]], dict[str, object], str]:
    files = _discover_evidence(path)
    rows: list[dict[str, Any]] = []
    for file, _ in files:
        rows.extend(_iter_evidence_file(file))
    if not rows:
        raise ValueError("full-scan evidence contains no rows")
    root = path.resolve() if path.is_dir() else path.resolve().parent
    binding = _tree_binding(root, files)
    return rows, binding, str(binding["inventory_sha256"])


def _load_roster(path: Path) -> tuple[dict[str, Any], dict[str, object]]:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: candidate roster root must be an object")
    review.validate_candidate_roster_routes(value)
    return value, _binding(path)


def _load_review_policy(path: Path) -> tuple[dict[str, Any], dict[str, object]]:
    value = _read_json(path)
    if not isinstance(value, Mapping) or value.get("schema_version") != "agent1_full_corpus_v3_policy_v1":
        raise ValueError(f"{path}: unsupported Agent 1 v3 policy")
    policy = value.get("review")
    if not isinstance(policy, Mapping):
        raise ValueError(f"{path}: review policy must be an object")
    required_model = _require_nonempty_string("policy.review.required_model", policy.get("required_model"))
    review.validate_review_model(required_model)
    seed = _require_nonempty_string("policy.review.seed", policy.get("seed"))
    model_environment_variable = _require_nonempty_string(
        "policy.review.model_environment_variable", policy.get("model_environment_variable")
    )
    if model_environment_variable != EXPECTED_MODEL_ENVIRONMENT_VARIABLE:
        raise ValueError(
            f"{path}: review model environment variable must remain "
            f"{EXPECTED_MODEL_ENVIRONMENT_VARIABLE}"
        )
    if policy.get("reasoning_effort") != EXPECTED_REASONING_EFFORT:
        raise ValueError(f"{path}: review reasoning effort must remain {EXPECTED_REASONING_EFFORT!r}")
    if policy.get("review_copy") != EXPECTED_REVIEW_COPY_POLICY:
        raise ValueError(f"{path}: review-copy policy drift")
    if policy.get("no_model_fallback") is not True:
        raise ValueError(f"{path}: review policy must prohibit model fallback")
    if int(policy.get("minimum_documents_per_eligible_source", -1)) != review.MINIMUM_ELIGIBLE_DOCUMENTS:
        raise ValueError(f"{path}: minimum review denominator must be exactly 100")
    if dict(policy.get("sample_strata") or {}) != dict(review.DEFAULT_QUOTAS):
        raise ValueError(f"{path}: review sample strata must remain 60/20/20")
    if int(policy.get("large_or_heterogeneous_total", -1)) != sum(review.LARGE_QUOTAS.values()):
        raise ValueError(f"{path}: large/heterogeneous review target must remain 200")
    fraction = policy.get("double_review_fraction")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or float(fraction) != 0.1:
        raise ValueError(f"{path}: deterministic second-review fraction must remain 0.1")
    large = policy.get("large_or_heterogeneous_source_ids", [])
    if not isinstance(large, list) or len(large) != len(set(large)) or not all(
        isinstance(value, str) and value for value in large
    ):
        raise ValueError(f"{path}: large_or_heterogeneous_source_ids must be unique non-empty strings")
    return dict(policy), _binding(path)


def _validate_response_schema(path: Path) -> dict[str, object]:
    value = _read_json(path)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: review response schema must be an object")
    properties = value.get("properties")
    if (
        value.get("additionalProperties") is not False
        or not isinstance(properties, Mapping)
        or properties.get("schema_version", {}).get("const") != review.RESPONSE_SCHEMA
    ):
        raise ValueError(f"{path}: expected strict {review.RESPONSE_SCHEMA} response schema")
    return _binding(path)


def _selection_evidence_key(row: Mapping[str, Any]) -> tuple[str, str]:
    """Return the full-scan key accepted for deterministic sample selection.

    Native v3 evidence contains a canonical stable UID.  The existing quality
    profiler's public full-scan schema instead contains an opaque
    ``dataset_quality_document_v1.document_id``.  The latter is accepted only
    under its exact schema and later verified against canonical text before it
    can become a review request sample ID.  This avoids treating arbitrary
    document hashes as reversible corpus identities.
    """

    for field in ("stable_uid", "sample_id"):
        value = row.get(field)
        if value is not None:
            return (
                _require_sha256(f"full-scan {field}", value),
                EVIDENCE_KEY_CANONICAL_STABLE_UID,
            )
    document_id = row.get("document_id")
    if document_id is None:
        raise ValueError("full-scan row evidence requires stable_uid or dataset-quality document_id")
    if row.get("schema_version") != QUALITY_DOCUMENT_SCHEMA:
        raise ValueError(
            "opaque full-scan document_id is accepted only for "
            f"{QUALITY_DOCUMENT_SCHEMA} evidence"
        )
    return (
        _require_sha256("full-scan document_id", document_id),
        EVIDENCE_KEY_QUALITY_DOCUMENT_ID,
    )


def _selection_metric_index(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index complete full-scan rows by their immutable selection evidence key."""

    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key, kind = _selection_evidence_key(row)
        if key in result:
            raise ValueError(f"full-scan evidence repeats selection key {key}")
        enriched = dict(row)
        enriched["_agent1_v3_selection_evidence_key"] = key
        enriched["_agent1_v3_selection_evidence_key_kind"] = kind
        result[key] = enriched
    return result


@dataclass(frozen=True)
class MaterializedSample:
    selection: Mapping[str, Any]
    metric: Mapping[str, Any]
    original_text_sha256: str
    review_copy: str
    redaction_report: Mapping[str, Any]

    @property
    def stable_uid(self) -> str:
        return str(self.selection["stable_uid"])

    def comparison_item(self) -> dict[str, object]:
        return {
            "sample_id": self.stable_uid,
            "sampling_stratum": str(self.selection["sampling_stratum"]),
            "review_cluster_id": str(self.selection["review_cluster_id"]),
            "review_cluster_size": int(self.selection["review_cluster_size"]),
            "risk_score": float(self.selection["risk_score"]),
            "original_text_sha256": self.original_text_sha256,
            "review_copy_sha256": str(self.redaction_report["review_copy_sha256"]),
            "review_copy": self.review_copy,
        }

    def attestation(self) -> dict[str, object]:
        return {
            "stable_uid": self.stable_uid,
            "source_id": str(self.selection["source_id"]),
            "source_dataset": str(self.selection["source_dataset"]),
            "source_revision": str(self.selection["source_revision"]),
            "sampling_stratum": str(self.selection["sampling_stratum"]),
            "original_text_sha256": self.original_text_sha256,
            "review_copy_sha256": str(self.redaction_report["review_copy_sha256"]),
            "original_characters": int(self.redaction_report["original_characters"]),
            "review_copy_characters": int(self.redaction_report["review_copy_characters"]),
            "positions_preserved": bool(
                int(self.redaction_report["original_characters"])
                == int(self.redaction_report["review_copy_characters"])
            ),
            "redaction_counts": dict(self.redaction_report["redaction_counts"]),
            "redaction_spans": list(self.redaction_report["redaction_spans"]),
        }


def _bound_selection(
    selection: Mapping[str, Any],
    *,
    canonical_stable_uid: str,
    evidence_key: str,
    evidence_key_kind: str,
) -> dict[str, Any]:
    """Bind a selected full-scan key to the canonical request identity."""

    if evidence_key_kind not in {
        EVIDENCE_KEY_CANONICAL_STABLE_UID,
        EVIDENCE_KEY_QUALITY_DOCUMENT_ID,
    }:
        raise AssertionError(f"unsupported selection evidence key kind {evidence_key_kind!r}")
    if str(selection.get("stable_uid")) != evidence_key:
        raise AssertionError("selection evidence key drift")
    if evidence_key_kind == EVIDENCE_KEY_CANONICAL_STABLE_UID and evidence_key != canonical_stable_uid:
        raise ValueError("canonical stable_uid does not match native full-scan selection key")
    if (
        evidence_key_kind == EVIDENCE_KEY_QUALITY_DOCUMENT_ID
        and quality_document_id(canonical_stable_uid) != evidence_key
    ):
        raise ValueError("quality document_id does not bind to canonical stable_uid")
    result = {
        **dict(selection),
        "stable_uid": canonical_stable_uid,
        "full_scan_selection_evidence_key": evidence_key,
        "full_scan_selection_evidence_key_kind": evidence_key_kind,
    }
    if evidence_key_kind == EVIDENCE_KEY_QUALITY_DOCUMENT_ID:
        result["full_scan_document_id"] = evidence_key
    return result


def _expected_metric_binding(
    metric: Mapping[str, Any],
    *,
    input_path: Path,
    input_relative: Path,
    row_index: int,
    file_sha256_cache: dict[Path, str],
) -> None:
    """Verify optional full-scan shard/index evidence for a selected row."""

    metric_key = str(
        metric.get("_agent1_v3_selection_evidence_key")
        or metric.get("stable_uid")
        or metric.get("document_id")
        or "<unknown>"
    )
    expected_path = metric.get("input_shard_path")
    if expected_path is not None and str(expected_path) != input_relative.as_posix():
        raise ValueError(
            f"full-scan shard binding drift for {metric_key}: "
            f"{expected_path!r} != {input_relative.as_posix()!r}"
        )
    expected_index = metric.get("input_row_index")
    if expected_index is not None:
        if isinstance(expected_index, bool) or not isinstance(expected_index, int) or expected_index != row_index:
            raise ValueError(
                f"full-scan row-index binding drift for {metric_key}: "
                f"{expected_index!r} != {row_index}"
            )
    expected_sha = metric.get("input_shard_sha256")
    if expected_sha is not None:
        expected = _require_sha256("full-scan input_shard_sha256", expected_sha)
        actual = file_sha256_cache.setdefault(input_path, sha256_file(input_path))
        if expected != actual:
            raise ValueError(f"full-scan shard checksum drift for {metric_key}")


def _validate_canonical_row(
    row: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
    metric: Mapping[str, Any],
    input_path: Path,
) -> tuple[str, str]:
    stable_uid = row.get("stable_uid")
    if stable_uid != selection.get("stable_uid"):
        raise ValueError(f"{input_path}: selected stable_uid drift")
    text = row.get("text")
    if not isinstance(text, str):
        raise ValueError(f"{input_path}: {stable_uid}: canonical text must be a string")
    actual_hash = sha256_text(text)
    claimed = row.get("normalized_text_sha256")
    if claimed is not None and str(claimed) != actual_hash:
        raise ValueError(f"{input_path}: {stable_uid}: canonical normalized_text_sha256 drift")
    metric_hash = metric.get("normalized_text_sha256")
    if metric_hash is not None and str(metric_hash) != actual_hash:
        raise ValueError(f"{input_path}: {stable_uid}: full-scan text hash drift")
    for field in ("source_id", "source_dataset", "source_revision"):
        expected = selection.get(field)
        actual = row.get(field)
        if expected is not None and actual is not None and str(expected) != str(actual):
            raise ValueError(f"{input_path}: {stable_uid}: {field} differs from selected full-scan identity")
    profile_variant = metric.get("profile_text_variant")
    if profile_variant is not None and profile_variant != "canonical":
        raise ValueError(
            f"{stable_uid}: full-scan evidence must be canonical, not {profile_variant!r} review-copy data"
        )
    return str(stable_uid), text


def _materialize_selected_samples(
    *,
    canonical_root: Path,
    selected_documents: Sequence[Mapping[str, Any]],
    metrics_by_selection_key: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, MaterializedSample], dict[str, object], list[dict[str, Any]]]:
    """Stream canonical shards and recover only the immutable selected texts.

    Selection itself happens from full-scan evidence.  For native evidence the
    selected key is already the canonical stable UID.  For the quality
    profiler's opaque document ID, derive the documented identifier from each
    canonical UID while streaming and require an exact match before any text
    can be included in a request.
    """

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - runner image dependency
        raise RuntimeError("canonical review-packet materialization requires pyarrow") from exc

    files = _discover_parquet(canonical_root)
    selected_by_key = {str(row["stable_uid"]): dict(row) for row in selected_documents}
    if len(selected_by_key) != len(selected_documents):
        raise ValueError("selection manifest repeats a stable_uid")
    recovered: dict[str, MaterializedSample] = {}
    resolved_by_key: dict[str, dict[str, Any]] = {}
    file_sha256_cache: dict[Path, str] = {}
    required = {"stable_uid", "text", "source_id", "source_dataset", "source_revision", "normalized_text_sha256"}
    for input_path, relative in files:
        parquet = pq.ParquetFile(input_path)
        missing = required - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{input_path}: canonical input missing required columns {sorted(missing)}")
        row_index = 0
        for batch in parquet.iter_batches(columns=sorted(required), batch_size=8192, use_threads=False):
            for row in batch.to_pylist():
                uid = row.get("stable_uid")
                if not isinstance(uid, str) or not HEX_SHA256_RE.fullmatch(uid):
                    raise ValueError(f"{input_path}: canonical stable_uid must be a lowercase SHA-256 digest")
                native_selection = selected_by_key.get(uid)
                opaque_key = quality_document_id(uid)
                opaque_selection = selected_by_key.get(opaque_key)
                matched: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
                if native_selection is not None:
                    metric = metrics_by_selection_key[uid]
                    if metric.get("_agent1_v3_selection_evidence_key_kind") == EVIDENCE_KEY_CANONICAL_STABLE_UID:
                        matched.append((uid, native_selection, metric))
                if opaque_selection is not None:
                    metric = metrics_by_selection_key[opaque_key]
                    if metric.get("_agent1_v3_selection_evidence_key_kind") == EVIDENCE_KEY_QUALITY_DOCUMENT_ID:
                        matched.append((opaque_key, opaque_selection, metric))
                if len(matched) > 1:
                    raise ValueError(f"{input_path}: canonical stable_uid maps to multiple selected evidence keys")
                if matched:
                    evidence_key, selection, metric = matched[0]
                    if uid in recovered:
                        raise ValueError(f"canonical tree repeats selected stable_uid {uid}")
                    evidence_kind = str(metric["_agent1_v3_selection_evidence_key_kind"])
                    bound_selection = _bound_selection(
                        selection,
                        canonical_stable_uid=uid,
                        evidence_key=evidence_key,
                        evidence_key_kind=evidence_kind,
                    )
                    _expected_metric_binding(
                        metric,
                        input_path=input_path,
                        input_relative=relative,
                        row_index=row_index,
                        file_sha256_cache=file_sha256_cache,
                    )
                    _, text = _validate_canonical_row(
                        row,
                        selection=bound_selection,
                        metric=metric,
                        input_path=input_path,
                    )
                    review_copy, redaction = review.redact_review_copy(text)
                    if redaction["original_text_sha256"] != sha256_text(text):
                        raise AssertionError(f"{uid}: redaction input hash drift")
                    if len(review_copy) != len(text):
                        raise AssertionError(f"{uid}: review copy changed character positions")
                    recovered[uid] = MaterializedSample(
                        selection=bound_selection,
                        metric=metric,
                        original_text_sha256=sha256_text(text),
                        review_copy=review_copy,
                        redaction_report=redaction,
                    )
                    resolved_by_key[evidence_key] = bound_selection
                row_index += 1
    missing = sorted(set(selected_by_key) - set(resolved_by_key))
    if missing:
        raise ValueError(f"canonical input did not contain selected full-scan rows: {missing[:20]}")
    return (
        recovered,
        _tree_binding(canonical_root.resolve(), files),
        [resolved_by_key[str(selection["stable_uid"])] for selection in selected_documents],
    )


def _selection_key_kind_label(kinds: Iterable[str]) -> str:
    unique = sorted(set(kinds))
    if not unique:
        raise AssertionError("selection evidence key kind unexpectedly absent")
    return unique[0] if len(unique) == 1 else "mixed"


def _bind_selection_manifest_to_canonical(
    selection: Mapping[str, Any],
    *,
    canonical_selected_documents: Sequence[Mapping[str, Any]],
    metrics_by_selection_key: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Replace opaque selected keys with canonical request IDs without erasing evidence.

    The pre-bridge selection receipt is retained by hash.  Per-source selected
    inventory gets a canonical UID hash for request closure plus the original
    opaque-key hash for independent verification against full-scan evidence.
    """

    if len(canonical_selected_documents) != int(selection["selected_document_count"]):
        raise AssertionError("canonical selection count differs from evidence selection")
    result: dict[str, Any] = {
        **dict(selection),
        "sources": [dict(source) for source in selection["sources"]],
        "selected_documents": [dict(document) for document in canonical_selected_documents],
    }
    evidence_manifest_sha256 = _require_sha256(
        "evidence selection manifest_sha256", result.pop("manifest_sha256", None)
    )
    canonical_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    kinds_by_source: dict[str, list[str]] = defaultdict(list)
    for document in result["selected_documents"]:
        source_id = str(document["source_id"])
        evidence_key = str(document["full_scan_selection_evidence_key"])
        metric = metrics_by_selection_key.get(evidence_key)
        if metric is None:
            raise AssertionError("selected document is missing full-scan evidence")
        kind = str(document["full_scan_selection_evidence_key_kind"])
        if kind != metric.get("_agent1_v3_selection_evidence_key_kind"):
            raise AssertionError("selected document evidence kind drift")
        canonical_by_source[source_id].append(document)
        kinds_by_source[source_id].append(kind)

    candidate_source_ids = {str(source["source_id"]) for source in result["sources"]}
    all_kinds_by_source: dict[str, list[str]] = defaultdict(list)
    for metric in metrics_by_selection_key.values():
        source_id = next(
            (
                value
                for field in ("source_id", "source_dataset")
                if isinstance((value := metric.get(field)), str) and value in candidate_source_ids
            ),
            None,
        )
        if source_id is None:
            raise AssertionError("normalized metric is not bound to a selected candidate source")
        all_kinds_by_source[source_id].append(
            str(metric["_agent1_v3_selection_evidence_key_kind"])
        )

    for source in result["sources"]:
        source_id = str(source["source_id"])
        documents = canonical_by_source[source_id]
        if len(documents) != int(source["review_denominator"]["selected_unique_documents"]):
            raise AssertionError(f"{source_id}: canonical selection does not close against denominator")
        source["selected_evidence_inventory_sha256"] = str(source["selected_inventory_sha256"])
        source["selected_evidence_inventory_key_kind"] = _selection_key_kind_label(
            kinds_by_source[source_id]
        )
        source["eligible_inventory_key_kind"] = _selection_key_kind_label(
            all_kinds_by_source[source_id]
        )
        source["selected_inventory_key_kind"] = EVIDENCE_KEY_CANONICAL_STABLE_UID
        source["selected_inventory_sha256"] = review.sha256_json(
            [
                {
                    "stable_uid": str(document["stable_uid"]),
                    "sampling_stratum": str(document["sampling_stratum"]),
                }
                for document in documents
            ]
        )

    result["selection_evidence_manifest_sha256"] = evidence_manifest_sha256
    result["selection_evidence_key_counts"] = dict(
        sorted(
            Counter(
                str(document["full_scan_selection_evidence_key_kind"])
                for document in result["selected_documents"]
            ).items()
        )
    )
    result["manifest_sha256"] = review.sha256_json(result)
    return result


def _bundle_order(sample: MaterializedSample) -> tuple[object, ...]:
    return (
        -int(sample.selection["review_cluster_size"]),
        int(sample.selection["selection_rank"]),
        sample.stable_uid,
    )


def _comparison_bundles(
    samples: Mapping[str, MaterializedSample], *, size: int
) -> tuple[dict[str, list[dict[str, object]]], dict[str, dict[str, int]]]:
    """Build source-local diversity comparison bundles from cluster exemplars."""

    by_source: dict[str, list[MaterializedSample]] = defaultdict(list)
    for sample in samples.values():
        by_source[str(sample.selection["source_id"])].append(sample)
    bundles: dict[str, list[dict[str, object]]] = {}
    coverage: dict[str, dict[str, int]] = {}
    for source_id, source_samples in sorted(by_source.items()):
        cluster_first = sorted(
            (sample for sample in source_samples if sample.selection["sampling_stratum"] == "cluster"),
            key=_bundle_order,
        )
        all_ordered = sorted(source_samples, key=_bundle_order)
        actual_sizes: list[int] = []
        for target in all_ordered:
            selected: list[MaterializedSample] = []
            seen: set[str] = {target.stable_uid}
            for candidate in [*cluster_first, *all_ordered]:
                if candidate.stable_uid in seen:
                    continue
                selected.append(candidate)
                seen.add(candidate.stable_uid)
                if len(selected) == size:
                    break
            bundles[target.stable_uid] = [candidate.comparison_item() for candidate in selected]
            actual_sizes.append(len(selected))
        required_minimum = min(2, max(0, len(source_samples) - 1))
        if any(value < required_minimum for value in actual_sizes):
            raise AssertionError(f"{source_id}: comparison bundle coverage unexpectedly fell below available peers")
        coverage[source_id] = {
            "selected_documents": len(source_samples),
            "cluster_representatives_available": len(cluster_first),
            "required_minimum_comparisons": required_minimum,
            "minimum_bundle_documents": min(actual_sizes, default=0),
            "maximum_bundle_documents": max(actual_sizes, default=0),
        }
    return bundles, coverage


def _request_sort_key(request: Mapping[str, Any], selection: Mapping[str, Mapping[str, Any]]) -> tuple[object, ...]:
    sample = selection[str(request["sample_id"])]
    slot = {"primary": 0, "secondary": 1}.get(str(request["reviewer_slot"]), 2)
    return (
        str(request["source_id"]),
        review.STRATA.index(str(request["sampling_stratum"])),
        slot,
        int(sample["selection_rank"]),
        str(request["sample_id"]),
    )


def _make_requests(
    *,
    selection_manifest: Mapping[str, Any],
    samples: Mapping[str, MaterializedSample],
    comparison_bundles: Mapping[str, Sequence[Mapping[str, object]]],
    prompt_sha256: str,
    response_schema_sha256: str,
    model: str,
    code_commit: str,
    secondary_seed: str,
    secondary_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = list(selection_manifest["selected_documents"])
    selection_by_uid = {str(row["stable_uid"]): dict(row) for row in selected}
    primary: list[dict[str, Any]] = []
    for selection in selected:
        sample = samples[str(selection["stable_uid"])]
        primary.append(
            review.make_review_request(
                selection,
                reviewer_slot="primary",
                original_text_sha256=sample.original_text_sha256,
                review_copy_sha256=str(sample.redaction_report["review_copy_sha256"]),
                prompt_sha256=prompt_sha256,
                response_schema_sha256=response_schema_sha256,
                model=model,
                code_commit=code_commit,
                attempt=1,
                review_copy=sample.review_copy,
                comparison_bundle=comparison_bundles[sample.stable_uid],
            )
        )
    secondary_selections = review.select_secondary_samples(
        selected, seed=secondary_seed, fraction=secondary_fraction
    )
    secondary: list[dict[str, Any]] = []
    for selection in secondary_selections:
        sample = samples[str(selection["stable_uid"])]
        secondary.append(
            review.make_review_request(
                selection,
                reviewer_slot="secondary",
                original_text_sha256=sample.original_text_sha256,
                review_copy_sha256=str(sample.redaction_report["review_copy_sha256"]),
                prompt_sha256=prompt_sha256,
                response_schema_sha256=response_schema_sha256,
                model=model,
                code_commit=code_commit,
                attempt=1,
                review_copy=sample.review_copy,
                comparison_bundle=comparison_bundles[sample.stable_uid],
            )
        )
    for request in [*primary, *secondary]:
        errors = review._validate_request_binding(request)
        if errors:
            raise AssertionError(f"request binding validation failed: {errors}")
    return (
        sorted(primary, key=lambda request: _request_sort_key(request, selection_by_uid)),
        sorted(secondary, key=lambda request: _request_sort_key(request, selection_by_uid)),
    )


def _source_coverage(
    selection_manifest: Mapping[str, Any],
    primary: Sequence[Mapping[str, Any]],
    secondary: Sequence[Mapping[str, Any]],
    bundle_coverage: Mapping[str, Mapping[str, int]],
) -> list[dict[str, object]]:
    counts: dict[tuple[str, str, str], int] = Counter(
        (str(row["source_id"]), str(row["reviewer_slot"]), str(row["sampling_stratum"]))
        for row in [*primary, *secondary]
    )
    result: list[dict[str, object]] = []
    for source in selection_manifest["sources"]:
        source_id = str(source["source_id"])
        denominator = dict(source["review_denominator"])
        actual = {stratum: int(source["actual_strata"][stratum]) for stratum in review.STRATA}
        primary_actual = {stratum: counts[(source_id, "primary", stratum)] for stratum in review.STRATA}
        secondary_actual = {stratum: counts[(source_id, "secondary", stratum)] for stratum in review.STRATA}
        if actual != primary_actual:
            raise AssertionError(f"{source_id}: primary request count differs from frozen selection")
        selected_total = int(denominator["selected_unique_documents"])
        if sum(primary_actual.values()) != selected_total:
            raise AssertionError(f"{source_id}: primary review denominator does not close")
        if denominator["minimum_requirement_status"] == "met" and selected_total < review.MINIMUM_ELIGIBLE_DOCUMENTS:
            raise AssertionError(f"{source_id}: false claim that the 100-document requirement was met")
        if denominator["minimum_requirement_status"] == "unattainable_exhaustive" and not denominator["selection_is_exhaustive"]:
            raise AssertionError(f"{source_id}: sub-100 source must be exhaustive")
        result.append(
            {
                "source_id": source_id,
                "source_dataset": str(source["source_dataset"]),
                "source_revision": str(source["source_revision"]),
                # ``source_route`` remains the only route sent to the compact
                # external review request.  Keep the complete frozen triplet
                # in the receipt so an audit can establish that logical source
                # provenance, not Parquet transport, set the primary error
                # model while visible secondary extraction defects remained in
                # scope for the reviewer prompt.
                "source_route": str(source["source_route"]),
                "review_route": str(source["review_route"]),
                "extraction_route": str(source["extraction_route"]),
                "review_denominator": denominator,
                "requested_strata": dict(source["requested_strata"]),
                "primary_requests_by_stratum": primary_actual,
                "secondary_requests_by_stratum": secondary_actual,
                "comparison_bundle_coverage": dict(bundle_coverage[source_id]),
                "eligible_inventory_sha256": str(source["eligible_inventory_sha256"]),
                "eligible_inventory_key_kind": str(source["eligible_inventory_key_kind"]),
                "selected_evidence_inventory_sha256": str(
                    source["selected_evidence_inventory_sha256"]
                ),
                "selected_evidence_inventory_key_kind": str(
                    source["selected_evidence_inventory_key_kind"]
                ),
                "selected_inventory_sha256": str(source["selected_inventory_sha256"]),
                "selected_inventory_key_kind": str(source["selected_inventory_key_kind"]),
            }
        )
    return result


def _redaction_summary(samples: Mapping[str, MaterializedSample]) -> tuple[dict[str, int], list[dict[str, object]]]:
    totals: Counter[str] = Counter()
    attestations: list[dict[str, object]] = []
    for sample in sorted(samples.values(), key=lambda item: item.stable_uid):
        report = sample.attestation()
        if report["positions_preserved"] is not True:
            raise AssertionError(f"{sample.stable_uid}: review-copy positions were not preserved")
        totals.update({str(key): int(value) for key, value in dict(report["redaction_counts"]).items()})
        attestations.append(report)
    return dict(sorted(totals.items())), attestations


def _manifest_sha256(payload: Mapping[str, object]) -> str:
    copy = dict(payload)
    copy.pop("manifest_sha256", None)
    return sha256_json(copy)


def build_packet(
    *,
    full_scan_rows: Sequence[Mapping[str, Any]],
    full_scan_binding: Mapping[str, object],
    canonical_root: Path,
    roster: Mapping[str, Any],
    roster_binding: Mapping[str, object],
    review_policy: Mapping[str, Any],
    policy_binding: Mapping[str, object],
    seed: str,
    model: str,
    prompt_binding: Mapping[str, object],
    response_schema_binding: Mapping[str, object],
    code_commit: str,
    comparison_bundle_size: int,
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    """Build all deterministic packet contents without writing an artifact."""

    if seed != review_policy["seed"]:
        raise ValueError("selection seed must exactly match the frozen review policy")
    required_model = str(review_policy["required_model"])
    if model != required_model:
        raise ValueError("review model must exactly match the frozen review policy; no fallback")
    review.validate_review_model(model)
    if not CODE_COMMIT_RE.fullmatch(code_commit):
        raise ValueError("--code-commit must be an exact 40-character lowercase git commit")
    if comparison_bundle_size < 2:
        raise ValueError("comparison bundle size must be >= 2")

    metrics_by_selection_key = _selection_metric_index(full_scan_rows)
    route_report = review.validate_candidate_roster_routes(roster)
    large_sources = list(review_policy["large_or_heterogeneous_source_ids"])
    selection = review.build_sample_manifest(
        metrics_by_selection_key.values(),
        roster,
        seed=seed,
        large_or_heterogeneous_sources=large_sources,
        require_all_candidate_sources=True,
        full_scan_metrics_sha256=str(full_scan_binding["inventory_sha256"]),
    )
    if selection["route_validation"] != route_report:
        raise AssertionError("selection route validation drift")
    samples, canonical_binding, canonical_selected_documents = _materialize_selected_samples(
        canonical_root=canonical_root,
        selected_documents=selection["selected_documents"],
        metrics_by_selection_key=metrics_by_selection_key,
    )
    selection = _bind_selection_manifest_to_canonical(
        selection,
        canonical_selected_documents=canonical_selected_documents,
        metrics_by_selection_key=metrics_by_selection_key,
    )
    bundles, bundle_coverage = _comparison_bundles(samples, size=comparison_bundle_size)
    secondary_seed = sha256_text(f"{seed}\0agent1-v3-secondary-review-v1")
    primary, secondary = _make_requests(
        selection_manifest=selection,
        samples=samples,
        comparison_bundles=bundles,
        prompt_sha256=str(prompt_binding["sha256"]),
        response_schema_sha256=str(response_schema_binding["sha256"]),
        model=model,
        code_commit=code_commit,
        secondary_seed=secondary_seed,
        secondary_fraction=float(review_policy["double_review_fraction"]),
    )
    redaction_totals, attestations = _redaction_summary(samples)
    coverage = _source_coverage(selection, primary, secondary, bundle_coverage)
    requests = [*primary, *secondary]
    request_inventory = [
        {
            "review_id": str(request["review_id"]),
            "request_sha256": str(request["request_sha256"]),
            "sample_id": str(request["sample_id"]),
            "reviewer_slot": str(request["reviewer_slot"]),
        }
        for request in requests
    ]
    manifest: dict[str, object] = {
        "schema_version": PACKET_MANIFEST_SCHEMA,
        "status": "materialized_no_model_invocation",
        "implementation_version": PACKET_VERSION,
        "inputs": {
            "full_scan_evidence": dict(full_scan_binding),
            "canonical": canonical_binding,
            "candidate_roster": dict(roster_binding),
            "review_policy": dict(policy_binding),
            "prompt": dict(prompt_binding),
            "response_schema": dict(response_schema_binding),
        },
        "selection": selection,
        "review_execution": {
            "model_environment_variable": str(review_policy["model_environment_variable"]),
            "model": model,
            "reasoning_effort": str(review_policy["reasoning_effort"]),
            "no_model_fallback": True,
            "model_invocation": "not_run",
            "code_commit": code_commit,
            "selection_seed": seed,
            "secondary_selection_seed": secondary_seed,
            "secondary_fraction": float(review_policy["double_review_fraction"]),
            "prompt_sha256": str(prompt_binding["sha256"]),
            "response_schema_sha256": str(response_schema_binding["sha256"]),
        },
        "comparison_bundles": {
            "policy": "source_local_cluster_representatives_then_deterministic_fill_excluding_target",
            "configured_size": comparison_bundle_size,
            "source_coverage": bundle_coverage,
        },
        "source_review_coverage": coverage,
        "request_counts": {
            "primary": len(primary),
            "secondary": len(secondary),
            "total": len(requests),
            "primary_by_stratum": dict(
                sorted(Counter(str(row["sampling_stratum"]) for row in primary).items())
            ),
            "secondary_by_stratum": dict(
                sorted(Counter(str(row["sampling_stratum"]) for row in secondary).items())
            ),
        },
        "request_inventory": request_inventory,
        "review_copy_attestations": attestations,
        "review_copy_redaction_totals": redaction_totals,
        "privacy": {
            "raw_canonical_text_in_manifest": False,
            "raw_source_document_identifier_in_manifest": False,
            "review_copy_masking": "high_precision_direct_identifiers_position_preserving",
        },
    }
    manifest["manifest_sha256"] = _manifest_sha256(manifest)
    return requests, manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-scan-evidence", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--response-schema", type=Path, required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--comparison-bundle-size", type=int, default=DEFAULT_COMPARISON_BUNDLE_SIZE)
    parser.add_argument("--output", type=Path, required=True, help="Strict review-request JSONL")
    parser.add_argument("--manifest", type=Path, required=True, help="Immutable packet manifest JSON")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.resolve() == args.manifest.resolve():
        raise ValueError("--output and --manifest must be different paths")
    if args.output.exists() or args.manifest.exists():
        existing = args.output if args.output.exists() else args.manifest
        raise FileExistsError(f"immutable packet output already exists: {existing}")
    for path in (args.roster, args.policy, args.prompt, args.response_schema):
        if not path.is_file():
            raise FileNotFoundError(path)
    rows, full_scan_binding, _ = _load_full_scan_evidence(args.full_scan_evidence)
    roster, roster_binding = _load_roster(args.roster)
    policy, policy_binding = _load_review_policy(args.policy)
    prompt_binding = _binding(args.prompt)
    response_schema_binding = _validate_response_schema(args.response_schema)
    requests, manifest = build_packet(
        full_scan_rows=rows,
        full_scan_binding=full_scan_binding,
        canonical_root=args.canonical_root,
        roster=roster,
        roster_binding=roster_binding,
        review_policy=policy,
        policy_binding=policy_binding,
        seed=args.seed,
        model=args.model,
        prompt_binding=prompt_binding,
        response_schema_binding=response_schema_binding,
        code_commit=args.code_commit,
        comparison_bundle_size=args.comparison_bundle_size,
    )
    _write_jsonl_no_replace(args.output.resolve(), requests)
    request_binding = _binding(args.output.resolve())
    manifest = {
        **manifest,
        "requests": {
            "path": str(args.output.resolve()),
            **request_binding,
        },
    }
    manifest["manifest_sha256"] = _manifest_sha256(manifest)
    _write_json_no_replace(args.manifest.resolve(), manifest)
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output.resolve()),
                "manifest": str(args.manifest.resolve()),
                "primary_requests": manifest["request_counts"]["primary"],
                "secondary_requests": manifest["request_counts"]["secondary"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
