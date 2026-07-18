#!/usr/bin/env python3
"""Build and freeze the prediction-blind bibliography sealed test set.

This module is the remote-worker half of the fresh 150-document evaluation
lane.  Corpus text and labels stay below a sealed Clariden run root.  The only
intended public artifact is an exclusion manifest containing opaque IDs and
content hashes, never text or labels.

The workflow is deliberately split into immutable stages:

``select-candidates`` -> dual quality review -> ``finalize-selection`` ->
``prepare-annotation`` -> two independent Sol passes -> ``adjudication-packet``
-> third Sol pass -> ``merge-labels`` -> ``freeze``.

Every writer uses O_EXCL and every downstream stage binds its inputs by SHA256.
Running a stage twice is therefore either an exact resume or a hard failure.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import heapq
import importlib
import importlib.metadata
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .contract import canonical_json_sha256, sha256_file


SELECTION_CANDIDATE_SCHEMA = "bibliography-sealed-candidate-v1"
SELECTION_RECEIPT_SCHEMA = "bibliography-sealed-selection-receipt-v1"
PRIVATE_DOCUMENT_SCHEMA = "bibliography-sealed-document-v1"
PUBLIC_EXCLUSION_SCHEMA = "bibliography-sealed-public-exclusions-v1"
QUALITY_PACKET_SCHEMA = "bibliography-sealed-quality-packet-v1"
QUALITY_RESPONSE_SCHEMA = "bibliography-sealed-quality-response-v1"
QUALITY_CONSENSUS_SCHEMA = "bibliography-sealed-quality-consensus-v1"
CHUNK_SCHEMA = "bibliography-sealed-role-chunk-v1"
LINE_KEY_SCHEMA = "bibliography-sealed-line-key-v1"
ROLE_RESPONSE_SCHEMA = "bibliography-sealed-role-response-v1"
PASS_SCHEMA = "bibliography-sealed-role-pass-v1"
MERGED_SCHEMA = "bibliography-sealed-merged-labels-v1"
FREEZE_SCHEMA = "bibliography-sealed-freeze-v1"
RUN_CONTRACT_SCHEMA = "bibliography-sealed-sol-run-contract-v1"
RUN_RECORD_SCHEMA = "bibliography-sealed-sol-batch-record-v1"

SOURCES = ("greek_phd", "kallipos", "openarchives")
DEFAULT_QUOTAS = {source: 50 for source in SOURCES}
SOURCE_SPECS = {
    "greek_phd": {"receipt_source": "nanochat_base", "source_dataset": "greek_phd"},
    "kallipos": {
        "receipt_source": "kallipos_sections",
        "source_dataset": "glossAPI/Apothetirio_Kallipos",
    },
    "openarchives": {
        "receipt_source": "nanochat_base",
        "source_dataset": "openarchives.gr",
    },
}
ROLE_NAMES = (
    "ENTRY",
    "CONTINUATION",
    "FILLER",
    "BIB_HEADER",
    "BIB_SUBHEADER",
    "NON_BIB_HEADER",
    "OTHER",
    "UNKNOWN",
)
BIB_ROLES = frozenset(
    {"ENTRY", "CONTINUATION", "FILLER", "BIB_HEADER", "BIB_SUBHEADER"}
)
QUALITY_DECISIONS = frozenset({"KEEP", "UNUSABLE"})
MODEL = "gpt-5.6-sol"
REASONING_EFFORT = "high"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_WORD = re.compile(r"[0-9A-Za-zΑ-Ωα-ωΆ-ώϊϋΐΰἀ-῾]+", re.UNICODE)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write_json_new(path: Path, value: Any, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _write_jsonl_new(
    path: Path, rows: Iterable[Mapping[str, Any]], *, mode: int = 0o600
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for row in rows:
                handle.write(_canonical_bytes(row))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected an object")
            rows.append(value)
    return rows


def _directory_inventory_sha256(root: Path) -> str:
    rows = [
        {
            "relative_path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]
    return canonical_json_sha256(rows)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    return value


def _same_or_write(path: Path, value: Any) -> None:
    if path.exists():
        if _json(path) != value:
            raise ValueError(f"immutable artifact differs: {path}")
        return
    _write_json_new(path, value)


def _parse_quotas(values: Sequence[str]) -> dict[str, int]:
    if not values:
        return dict(DEFAULT_QUOTAS)
    result: dict[str, int] = {}
    for raw in values:
        source, separator, count_text = raw.partition("=")
        if not separator or source not in SOURCES or source in result:
            raise ValueError(f"invalid/duplicate quota {raw!r}")
        count = int(count_text)
        if count <= 0:
            raise ValueError("quotas must be positive")
        result[source] = count
    if set(result) != set(SOURCES):
        raise ValueError("quotas must cover greek_phd, kallipos, and openarchives")
    return result


def _identity_hash(source: str, kind: str, value: str) -> str:
    return hashlib.sha256(f"{source}\0{kind}\0{value}".encode("utf-8")).hexdigest()


def _exclusion_identity_hash(source: str, value: str) -> str:
    """Canonical source-scoped identity shared by all historical field aliases."""

    return hashlib.sha256(f"bibliography-exclusion-id-v1\0{source}\0{value}".encode()).hexdigest()


def _document_id(source: str, stable_uid: str, text_hash: str) -> str:
    return hashlib.sha256(
        f"bibliography-sealed-v1\0{source}\0{stable_uid}\0{text_hash}".encode("utf-8")
    ).hexdigest()


def _line_id(document_id: str, abs_idx: int, text: str) -> str:
    return hashlib.sha256(
        f"bibliography-sealed-line-v1\0{document_id}\0{abs_idx}\0{text}".encode("utf-8")
    ).hexdigest()


def _alias(secret: bytes, namespace: str, value: str) -> str:
    if not 16 <= len(secret) <= 64:
        raise ValueError("alias secret must contain 16..64 bytes")
    digest = hashlib.blake2b(
        f"{namespace}\0{value}".encode("utf-8"),
        key=secret,
        digest_size=16,
        person=b"bib-sealed-v1",
    ).hexdigest()
    return f"{namespace}_{digest}"


def _similarity(left: frozenset[int], right: frozenset[int]) -> float:
    denominator = min(len(left), len(right))
    return len(left & right) / denominator if denominator else 0.0


def bottom_k_word_shingles(
    text: str, *, k: int = 256, ngram: int = 5, maximum_tokens: int = 12_000
) -> frozenset[int]:
    """Return the historical holdout-compatible bounded word-shingle sketch."""

    tokens = _WORD.findall(text.casefold())
    if len(tokens) > maximum_tokens:
        edge = maximum_tokens // 3
        middle = maximum_tokens - 2 * edge
        start = max(edge, (len(tokens) - middle) // 2)
        tokens = tokens[:edge] + tokens[start : start + middle] + tokens[-edge:]
    if len(tokens) < ngram:
        payload = " ".join(tokens).encode("utf-8")
        return frozenset(
            [int.from_bytes(hashlib.blake2b(payload, digest_size=8, person=b"holdout1").digest(), "big")]
            if payload
            else []
        )
    heap: list[int] = []
    present: set[int] = set()
    for index in range(len(tokens) - ngram + 1):
        gram = " ".join(tokens[index : index + ngram]).encode("utf-8")
        value = int.from_bytes(
            hashlib.blake2b(gram, digest_size=8, person=b"holdout1").digest(), "big"
        )
        if value in present:
            continue
        if len(heap) < k:
            heapq.heappush(heap, -value)
            present.add(value)
        elif value < -heap[0]:
            removed = -heapq.heapreplace(heap, -value)
            present.remove(removed)
            present.add(value)
    return frozenset(present)


class GlobalSketchIndex:
    """Sparse candidate index with no source filter.

    The old holdout index intentionally compared only within a source.  The
    sealed test instead treats a copied work under another source as leakage.
    """

    def __init__(self) -> None:
        self.signatures: dict[str, frozenset[int]] = {}
        self.inverted: dict[int, set[str]] = collections.defaultdict(set)

    def add(self, identity: str, signature: frozenset[int]) -> None:
        if identity in self.signatures:
            raise ValueError(f"duplicate sketch identity {identity}")
        self.signatures[identity] = signature
        for value in signature:
            self.inverted[value].add(identity)

    def closest(self, signature: frozenset[int]) -> tuple[str | None, float]:
        overlap: collections.Counter[str] = collections.Counter()
        for value in signature:
            overlap.update(self.inverted.get(value, ()))
        best_id: str | None = None
        best_score = 0.0
        for identity, _ in overlap.most_common():
            score = _similarity(signature, self.signatures[identity])
            if score > best_score or (
                math.isclose(score, best_score) and best_id is not None and identity < best_id
            ):
                best_id, best_score = identity, score
        return best_id, best_score


def _dedup_candidate(
    *,
    text: str,
    normalized_hash: str,
    exclusions: "Exclusions",
    accepted_index: GlobalSketchIndex,
    accepted_exact_hashes: set[str],
    threshold: float,
) -> tuple[str | None, dict[str, Any], frozenset[int], str]:
    """Apply exact and global near-copy gates without consulting a source label."""

    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if text_hash in exclusions.materialized_hashes or normalized_hash in exclusions.normalized_hashes:
        return "excluded_exact_prior_text", {}, frozenset(), normalized_hash or text_hash
    exact_key = normalized_hash or text_hash
    if exact_key in accepted_exact_hashes:
        return "excluded_exact_selected_text", {}, frozenset(), exact_key
    signature = bottom_k_word_shingles(text)
    old_id, old_score = exclusions.sketches.closest(signature)
    if old_score >= threshold:
        return "excluded_global_prior_near_duplicate", {}, signature, exact_key
    selected_id, selected_score = accepted_index.closest(signature)
    if selected_score >= threshold:
        return "excluded_global_selected_near_duplicate", {}, signature, exact_key
    return (
        None,
        {
            "method": "global_bottom_k_word_5gram_v1",
            "threshold": threshold,
            "prior_closest_id": old_id,
            "prior_similarity": old_score,
            "selected_closest_id": selected_id,
            "selected_similarity": selected_score,
        },
        signature,
        exact_key,
    )


@dataclass
class Exclusions:
    identities: set[str]
    normalized_hashes: set[str]
    materialized_hashes: set[str]
    sketches: GlobalSketchIndex
    input_hashes: list[dict[str, Any]]


@dataclass(frozen=True)
class CandidateRow:
    source: str
    canonical_file: str
    row_group: int
    row_offset: int
    metadata: Mapping[str, Any]
    rank: int


def _row_identity_hashes(source: str, row: Mapping[str, Any]) -> Iterator[str]:
    for field in ("doc_id", "source_doc_id", "work_id", "work_key", "stable_uid"):
        value = str(row.get(field) or "")
        if value:
            # STRUCT-2K calls the same source identity ``doc_id`` that fresh
            # canonical rows may expose as ``source_doc_id``, ``work_id`` or
            # ``work_key``.  Field names must not prevent the exclusion join.
            yield _exclusion_identity_hash(source, value)


def load_exclusions(
    *,
    historical_manifest: Path,
    historical_root: Path,
    previous_documents: Path,
    expected_historical: int = 2000,
    expected_previous: int = 500,
) -> Exclusions:
    from .source_matched_holdout import load_historical_manifest, load_historical_texts

    historical_rows, _ = load_historical_manifest(historical_manifest)
    if len(historical_rows) != expected_historical:
        raise ValueError("historical exclusion count differs from the frozen expectation")
    previous_rows = _read_jsonl(previous_documents)
    if len(previous_rows) != expected_previous:
        raise ValueError(
            f"previous holdout has {len(previous_rows)} rows, expected {expected_previous}"
        )
    identities: set[str] = set()
    normalized_hashes: set[str] = set()
    materialized_hashes: set[str] = set()
    index = GlobalSketchIndex()

    historical_texts = load_historical_texts(historical_rows, historical_root)
    for manifest_row, text_row in zip(historical_rows, historical_texts, strict=True):
        source = str(manifest_row["source"])
        identities.update(_row_identity_hashes(source, manifest_row))
        text = str(text_row["text"])
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        materialized_hashes.add(text_hash)
        index.add(f"struct2k:{source}:{manifest_row['doc_id']}", bottom_k_word_shingles(text))

    for row in previous_rows:
        source = str(row.get("source") or "")
        if source not in SOURCES:
            raise ValueError(f"previous holdout has unsupported source {source!r}")
        identities.update(_row_identity_hashes(source, row))
        for field, target in (
            ("normalized_text_sha256", normalized_hashes),
            ("materialized_text_sha256", materialized_hashes),
        ):
            value = str(row.get(field) or "")
            if value:
                if not _HEX64.fullmatch(value):
                    raise ValueError(f"previous holdout has invalid {field}")
                target.add(value)
        lines = row.get("lines")
        if not isinstance(lines, list) or not lines:
            raise ValueError("previous holdout row has no line inventory")
        text = "\n".join(str(line.get("text", "")) for line in lines)
        if not text.strip():
            raise ValueError("previous holdout contains empty text")
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        materialized_hashes.add(text_hash)
        index.add(f"prior500:{row.get('document_id')}", bottom_k_word_shingles(text))

    return Exclusions(
        identities=identities,
        normalized_hashes=normalized_hashes,
        materialized_hashes=materialized_hashes,
        sketches=index,
        input_hashes=[
            {
                "kind": "struct2k_manifest",
                "rows": len(historical_rows),
                "sha256": sha256_file(historical_manifest),
            },
            {
                "kind": "struct2k_text_inventory",
                "rows": len(historical_rows),
                "sha256": getattr(load_historical_texts, "inventory_sha256"),
            },
            {
                "kind": "previous_holdout_documents",
                "rows": len(previous_rows),
                "sha256": sha256_file(previous_documents),
            },
        ],
    )


def _representation_key(row: Mapping[str, Any]) -> tuple[str, str]:
    """Select one canonical representation per work without looking at text."""

    return (str(row.get("representation_generation") or ""), str(row.get("stable_uid") or ""))


def _scan_ranked_works(
    paths: Sequence[Path],
    *,
    source: str,
    source_dataset: str,
    exclusions: Exclusions,
    limit: int,
    seed: str,
) -> tuple[list[CandidateRow], dict[str, int]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - remote dependency
        raise RuntimeError("pyarrow is required on the Clariden worker") from error

    columns = [
        "source_dataset", "source_doc_id", "source_family_id", "acquisition_source_id",
        "source_repo_id", "source_revision", "source_artifact_path", "source_row_id",
        "source_text_field", "original_text_sha256", "normalized_text_sha256", "stable_uid",
        "work_key", "work_id", "representation_generation", "source_metadata_json",
        "cleaning_profile", "structural_policy",
    ]
    selected: dict[str, CandidateRow] = {}
    heap: list[tuple[int, str]] = []
    counts: collections.Counter[str] = collections.Counter()
    for path in sorted(paths):
        parquet = pq.ParquetFile(path)
        for row_group in range(parquet.num_row_groups):
            table = parquet.read_row_group(row_group, columns=columns)
            for row_offset, metadata in enumerate(table.to_pylist()):
                if metadata.get("source_dataset") != source_dataset:
                    continue
                counts["rows_in_route"] += 1
                stable_uid = str(metadata.get("stable_uid") or "")
                work_key = str(metadata.get("work_key") or metadata.get("work_id") or "")
                if not stable_uid or not work_key:
                    counts["invalid_identity"] += 1
                    continue
                identity_hashes = set(_row_identity_hashes(source, metadata))
                if identity_hashes & exclusions.identities:
                    counts["excluded_identity"] += 1
                    continue
                normalized_hash = str(metadata.get("normalized_text_sha256") or "")
                if not _HEX64.fullmatch(normalized_hash):
                    counts["invalid_normalized_text_hash"] += 1
                    continue
                if normalized_hash and normalized_hash in exclusions.normalized_hashes:
                    counts["excluded_exact_normalized_hash"] += 1
                    continue
                rank = int.from_bytes(
                    hashlib.sha256(f"{seed}\0{source}\0{work_key}".encode()).digest(), "big"
                )
                candidate = CandidateRow(source, str(path), row_group, row_offset, metadata, rank)
                current = selected.get(work_key)
                if current is not None:
                    counts["alternate_representation"] += 1
                    if _representation_key(metadata) < _representation_key(current.metadata):
                        selected[work_key] = candidate
                    continue
                if len(selected) < limit:
                    selected[work_key] = candidate
                    heapq.heappush(heap, (-rank, work_key))
                    continue
                if rank >= -heap[0][0]:
                    counts["ranked_out"] += 1
                    continue
                _, removed = heapq.heapreplace(heap, (-rank, work_key))
                del selected[removed]
                selected[work_key] = candidate
    rows = sorted(selected.values(), key=lambda item: (item.rank, str(item.metadata["stable_uid"])))
    counts["ranked_unique_works"] = len(rows)
    return rows, dict(sorted(counts.items()))


def _line_rows(document_id: str, text: str) -> tuple[list[dict[str, Any]], int]:
    physical = text.splitlines()
    lines = [
        {"line_id": _line_id(document_id, abs_idx, value), "abs_idx": abs_idx, "text": value}
        for abs_idx, value in enumerate(physical)
        if value.strip()
    ]
    if not lines:
        raise ValueError("candidate has no nonblank lines")
    return lines, len(physical)


def _quality_sample(lines: Sequence[Mapping[str, Any]], limit: int = 240) -> list[dict[str, Any]]:
    if len(lines) <= limit:
        return [dict(row) for row in lines]
    # Deterministic head/middle/tail coverage; no detector predictions are used.
    third = limit // 3
    middle = max(0, (len(lines) - third) // 2)
    positions = list(range(third))
    positions += list(range(middle, min(middle + third, len(lines))))
    positions += list(range(max(0, len(lines) - (limit - 2 * third)), len(lines)))
    return [dict(lines[position]) for position in sorted(set(positions))]


def _candidate_document(candidate: CandidateRow, text: str, rust_module: Any) -> dict[str, Any]:
    from .bibliography_validation_quality import analyze_text, candidate_reasons, _score_rust

    metadata = dict(candidate.metadata)
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    document_id = _document_id(candidate.source, str(metadata["stable_uid"]), text_hash)
    lines, n_physical = _line_rows(document_id, text)
    quality = analyze_text([str(row["text"]) for row in lines])
    rust = _score_rust(text, rust_module)
    reasons = candidate_reasons(quality, float(rust["greek_badness_score"]))
    return {
        "schema_version": SELECTION_CANDIDATE_SCHEMA,
        "document_id": document_id,
        "source": candidate.source,
        "source_dataset": metadata["source_dataset"],
        "source_doc_id": metadata["source_doc_id"],
        "work_id": metadata["work_id"],
        "work_key": metadata["work_key"],
        "stable_uid": metadata["stable_uid"],
        "source_repo_id": metadata["source_repo_id"],
        "source_revision": metadata["source_revision"],
        "source_artifact_path": metadata["source_artifact_path"],
        "source_row_id": metadata["source_row_id"],
        "representation_generation": metadata["representation_generation"],
        "original_text_sha256": metadata["original_text_sha256"],
        "normalized_text_sha256": metadata["normalized_text_sha256"],
        "materialized_text_sha256": text_hash,
        "selection_rank": f"{candidate.rank:064x}",
        "n_physical_lines": n_physical,
        "n_present_lines": len(lines),
        "text_characters": len(text),
        "quality": {
            "text": vars(quality),
            "rust": rust,
            "automatic_reasons": reasons,
            "flagged_for_dual_sol": bool(reasons),
        },
        "lines": lines,
    }


def select_candidates(args: argparse.Namespace) -> dict[str, Any]:
    from .source_matched_holdout import _load_candidate_texts, _resolve_source_shards

    quotas = _parse_quotas(args.quota)
    if args.oversample < 2:
        raise ValueError("oversample must be at least 2")
    if not 0.5 <= args.near_duplicate_threshold <= 1.0:
        raise ValueError("near-duplicate threshold must be in [0.5, 1]")
    root = Path(args.normalization_root).resolve()
    normalization_manifest = root / "normalization_manifest.json"
    if not normalization_manifest.is_file():
        raise ValueError("normalization_manifest.json is absent")
    exclusions = load_exclusions(
        historical_manifest=Path(args.historical_manifest).resolve(),
        historical_root=Path(args.historical_root).resolve(),
        previous_documents=Path(args.previous_documents).resolve(),
        expected_historical=args.expected_historical,
        expected_previous=args.expected_previous,
    )
    rust_module = importlib.import_module("glossapi_rs_noise")
    rust_root = Path(str(rust_module.__file__)).resolve().parent
    rust_version = importlib.metadata.version("glossapi-rs-noise")
    if rust_version != "0.1.0" or not hasattr(rust_module, "score_markdown_file_detailed"):
        raise ValueError("unexpected GlossAPI Rust scorer contract")
    ranked: dict[str, list[CandidateRow]] = {}
    source_audits: dict[str, Any] = {}
    texts_by_source: dict[str, dict[str, str]] = {}
    for source in SOURCES:
        shards, receipts = _resolve_source_shards(root, source)
        rows, counts = _scan_ranked_works(
            shards,
            source=source,
            source_dataset=str(SOURCE_SPECS[source]["source_dataset"]),
            exclusions=exclusions,
            limit=quotas[source] * args.oversample * 2,
            seed=args.seed,
        )
        ranked[source] = rows
        texts_by_source[source] = _load_candidate_texts(rows)
        source_audits[source] = {
            "scan_counts": counts,
            "shard_count": len(shards),
            "receipt_inventory_sha256": canonical_json_sha256(
                [sha256_file(path) for path in sorted(receipts)]
            ),
        }

    # Compete globally so a cross-source copy has one deterministic winner.
    queue = sorted(
        (candidate for rows in ranked.values() for candidate in rows),
        key=lambda row: (row.rank, row.source, str(row.metadata["stable_uid"])),
    )
    accepted_index = GlobalSketchIndex()
    exact_hashes: set[str] = set()
    counts: collections.Counter[str] = collections.Counter()
    source_counts: collections.Counter[str] = collections.Counter()
    candidates: list[dict[str, Any]] = []
    pool_target = {source: quotas[source] * args.oversample for source in SOURCES}
    for candidate in queue:
        source = candidate.source
        if source_counts[source] >= pool_target[source]:
            continue
        text = texts_by_source[source][str(candidate.metadata["stable_uid"])]
        normalized_hash = str(candidate.metadata.get("normalized_text_sha256") or "")
        reason, audit, signature, exact_key = _dedup_candidate(
            text=text,
            normalized_hash=normalized_hash,
            exclusions=exclusions,
            accepted_index=accepted_index,
            accepted_exact_hashes=exact_hashes,
            threshold=args.near_duplicate_threshold,
        )
        if reason is not None:
            counts[reason] += 1
            continue
        document = _candidate_document(candidate, text, rust_module)
        document["near_duplicate_audit"] = audit
        candidates.append(document)
        source_counts[source] += 1
        exact_hashes.add(exact_key)
        accepted_index.add(str(document["document_id"]), signature)
    if dict(source_counts) != pool_target:
        raise ValueError(f"candidate pool shortfall: got {dict(source_counts)}, need {pool_target}")
    candidates.sort(key=lambda row: (row["source"], row["selection_rank"], row["document_id"]))

    candidates_out = Path(args.candidates_out).resolve()
    quality_packet_out = Path(args.quality_packet_out).resolve()
    receipt_out = Path(args.receipt_out).resolve()
    _write_jsonl_new(candidates_out, candidates)
    quality_rows = []
    for row in candidates:
        if not row["quality"]["flagged_for_dual_sol"]:
            continue
        quality_rows.append(
            {
                "schema_version": QUALITY_PACKET_SCHEMA,
                "document_alias": "q_" + row["document_id"],
                "source": row["source"],
                "n_physical_lines": row["n_physical_lines"],
                "n_present_lines": row["n_present_lines"],
                "text_characters": row["text_characters"],
                "automatic_reasons": row["quality"]["automatic_reasons"],
                "text_quality": row["quality"]["text"],
                "rust_metrics": row["quality"]["rust"],
                "sample_lines": _quality_sample(row["lines"]),
            }
        )
    _write_jsonl_new(quality_packet_out, quality_rows)
    receipt = {
        "schema_version": SELECTION_RECEIPT_SCHEMA,
        "status": "candidate_pool_ready_for_prediction_blind_quality_gate",
        "seed": args.seed,
        "quotas": quotas,
        "oversample": args.oversample,
        "pool_counts": dict(sorted(source_counts.items())),
        "flagged_quality_documents": len(quality_rows),
        "global_deduplication": {
            "threshold": args.near_duplicate_threshold,
            "same_source_only": False,
            "counts": dict(sorted(counts.items())),
        },
        "one_representation_per_work": True,
        "representation_choice": "minimum (representation_generation, stable_uid), text blind",
        "exclusions": exclusions.input_hashes,
        "normalization_manifest_sha256": sha256_file(normalization_manifest),
        "quality_scorer": {
            "distribution": "glossapi-rs-noise",
            "version": rust_version,
            "package_inventory_sha256": _directory_inventory_sha256(rust_root),
        },
        "source_audits": source_audits,
        "outputs": {
            "candidates_sha256": sha256_file(candidates_out),
            "quality_packet_sha256": sha256_file(quality_packet_out),
        },
        "selection_accessed_model_predictions": False,
        "code_sha256": sha256_file(__file__),
    }
    _write_json_new(receipt_out, receipt)
    return receipt


def validate_quality_response(
    packet: Sequence[Mapping[str, Any]], payload: Mapping[str, Any], reviewer_id: str
) -> dict[str, Any]:
    if payload.get("schema_version") != QUALITY_RESPONSE_SCHEMA:
        raise ValueError("unsupported quality response schema")
    if payload.get("reviewer") != reviewer_id:
        raise ValueError("quality reviewer identity differs from the run contract")
    expected = {str(row["document_alias"]) for row in packet}
    documents = payload.get("documents")
    if not isinstance(documents, list) or len(documents) != len(expected):
        raise ValueError("quality response omits or invents documents")
    normalized = []
    seen: set[str] = set()
    for row in documents:
        if not isinstance(row, dict) or set(row) != {
            "document_alias", "decision", "confidence", "reasons"
        }:
            raise ValueError("quality response document has unexpected fields")
        alias = str(row.get("document_alias") or "")
        confidence = row.get("confidence")
        reasons = row.get("reasons")
        if alias not in expected or alias in seen:
            raise ValueError("quality response repeats or invents a document")
        if row.get("decision") not in QUALITY_DECISIONS:
            raise ValueError("quality decision must be KEEP or UNUSABLE")
        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0 <= confidence <= 1
            or not isinstance(reasons, list)
            or not all(isinstance(value, str) and value.strip() for value in reasons)
        ):
            raise ValueError("quality confidence/reasons are invalid")
        seen.add(alias)
        normalized.append(dict(row))
    if seen != expected:
        raise ValueError("quality response inventory differs from the packet")
    normalized.sort(key=lambda row: row["document_alias"])
    return {
        "schema_version": QUALITY_RESPONSE_SCHEMA,
        "reviewer": reviewer_id,
        "documents": normalized,
    }


def merge_quality(args: argparse.Namespace) -> dict[str, Any]:
    packet_path = Path(args.packet).resolve()
    packet = _read_jsonl(packet_path)
    if any(row.get("schema_version") != QUALITY_PACKET_SCHEMA for row in packet):
        raise ValueError("unsupported quality packet")
    response_paths = [Path(value).resolve() for value in args.response]
    reviewer_ids = list(args.reviewer_id)
    if len(response_paths) not in (2, 3) or len(response_paths) != len(reviewer_ids):
        raise ValueError("supply two responses, or three when adjudicating, with matching IDs")
    if len(set(reviewer_ids)) != len(reviewer_ids):
        raise ValueError("quality passes must use distinct reviewer identities")
    responses = [
        validate_quality_response(packet, _json(response_paths[index]), reviewer_ids[index])
        for index in range(2)
    ]
    first_by_alias = [
        {str(row["document_alias"]): row for row in response["documents"]}
        for response in responses
    ]
    disagreement_aliases = {
        alias
        for alias in first_by_alias[0]
        if first_by_alias[0][alias]["decision"] != first_by_alias[1][alias]["decision"]
    }
    if disagreement_aliases:
        if len(response_paths) != 3:
            raise ValueError("quality disagreements require a de-novo third response")
        disagreement_packet = [
            row for row in packet if str(row["document_alias"]) in disagreement_aliases
        ]
        responses.append(
            validate_quality_response(
                disagreement_packet, _json(response_paths[2]), reviewer_ids[2]
            )
        )
    elif len(response_paths) == 3:
        raise ValueError("third quality response supplied although A and B fully agree")
    by_response = [
        {str(row["document_alias"]): row for row in response["documents"]}
        for response in responses
    ]
    rows = []
    disagreement_count = 0
    for packet_row in sorted(packet, key=lambda row: str(row["document_alias"])):
        alias = str(packet_row["document_alias"])
        votes = [str(response[alias]["decision"]) for response in by_response[:2]]
        if alias in disagreement_aliases:
            votes.append(str(by_response[2][alias]["decision"]))
        counts = collections.Counter(votes)
        decision, frequency = counts.most_common(1)[0]
        if votes[0] != votes[1]:
            disagreement_count += 1
            if len(votes) != 3 or frequency < 2:
                decision = "UNRESOLVED"
        rows.append(
            {
                "document_alias": alias,
                "decision": decision,
                "votes": dict(sorted(counts.items())),
                "reviewers": reviewer_ids,
            }
        )
    output = {
        "schema_version": QUALITY_CONSENSUS_SCHEMA,
        "status": "passed" if all(row["decision"] != "UNRESOLVED" for row in rows) else "blocked",
        "packet_sha256": sha256_file(packet_path),
        "response_sha256": [sha256_file(path) for path in response_paths],
        "reviewers": reviewer_ids,
        "document_count": len(rows),
        "first_pass_disagreements": disagreement_count,
        "documents": rows,
    }
    if output["status"] != "passed":
        raise ValueError("quality disagreements require a de-novo third response")
    _write_json_new(Path(args.output).resolve(), output)
    return output


def build_quality_adjudication_packet(args: argparse.Namespace) -> dict[str, Any]:
    packet_path = Path(args.packet).resolve()
    packet = _read_jsonl(packet_path)
    response_a_path = Path(args.response_a).resolve()
    response_b_path = Path(args.response_b).resolve()
    first = validate_quality_response(packet, _json(response_a_path), args.reviewer_a)
    second = validate_quality_response(packet, _json(response_b_path), args.reviewer_b)
    first_by_alias = {str(row["document_alias"]): row for row in first["documents"]}
    second_by_alias = {str(row["document_alias"]): row for row in second["documents"]}
    disagreement_aliases = {
        alias
        for alias in first_by_alias
        if first_by_alias[alias]["decision"] != second_by_alias[alias]["decision"]
    }
    # The output is a direct subset of the original label-blind packet.  No A/B
    # decisions, confidence values, or rationales are copied into it.
    rows = [row for row in packet if str(row["document_alias"]) in disagreement_aliases]
    output_path = Path(args.output).resolve()
    _write_jsonl_new(output_path, rows)
    receipt = {
        "schema_version": "bibliography-sealed-quality-adjudication-packet-v1",
        "status": "ready_for_de_novo_third_sol" if rows else "no_adjudication_needed",
        "document_count": len(rows),
        "blinding": "direct source-packet subset; contains no earlier decisions",
        "inputs": {
            "packet_sha256": sha256_file(packet_path),
            "response_a_sha256": sha256_file(response_a_path),
            "response_b_sha256": sha256_file(response_b_path),
        },
        "output_sha256": sha256_file(output_path),
    }
    _write_json_new(Path(args.receipt_out).resolve(), receipt)
    return receipt


def _private_document(row: Mapping[str, Any]) -> dict[str, Any]:
    keep = {
        "document_id", "source", "source_dataset", "source_doc_id", "work_id", "work_key",
        "stable_uid", "source_repo_id", "source_revision", "source_artifact_path",
        "source_row_id", "representation_generation", "original_text_sha256",
        "normalized_text_sha256", "materialized_text_sha256", "selection_rank",
        "n_physical_lines", "n_present_lines", "text_characters", "quality",
        "near_duplicate_audit", "lines",
    }
    result = {key: row[key] for key in keep}
    result["schema_version"] = PRIVATE_DOCUMENT_SCHEMA
    return result


def _public_exclusion(row: Mapping[str, Any]) -> dict[str, str]:
    source = str(row["source"])
    return {
        "document_id": str(row["document_id"]),
        "source_identity_sha256": _identity_hash(source, "source", source),
        "source_doc_identity_sha256": _identity_hash(
            source, "source_doc_id", str(row["source_doc_id"])
        ),
        "work_identity_sha256": _identity_hash(source, "work_key", str(row["work_key"])),
        "stable_identity_sha256": _identity_hash(
            source, "stable_uid", str(row["stable_uid"])
        ),
        "normalized_text_sha256": str(row["normalized_text_sha256"]),
        "materialized_text_sha256": str(row["materialized_text_sha256"]),
    }


def finalize_selection(args: argparse.Namespace) -> dict[str, Any]:
    candidates_path = Path(args.candidates).resolve()
    candidate_receipt_path = Path(args.candidate_receipt).resolve()
    quality_path = Path(args.quality_consensus).resolve()
    candidates = _read_jsonl(candidates_path)
    if not candidates or any(
        row.get("schema_version") != SELECTION_CANDIDATE_SCHEMA for row in candidates
    ):
        raise ValueError("unsupported/empty candidate pool")
    receipt = _json(candidate_receipt_path)
    if (
        receipt.get("schema_version") != SELECTION_RECEIPT_SCHEMA
        or receipt.get("outputs", {}).get("candidates_sha256") != sha256_file(candidates_path)
    ):
        raise ValueError("candidate receipt is unsupported or unbound")
    quality = _json(quality_path)
    if quality.get("schema_version") != QUALITY_CONSENSUS_SCHEMA or quality.get("status") != "passed":
        raise ValueError("quality consensus has not passed")
    if quality.get("packet_sha256") != receipt.get("outputs", {}).get("quality_packet_sha256"):
        raise ValueError("quality consensus is not bound to this candidate pool")
    quality_by_alias = {
        str(row["document_alias"]): str(row["decision"])
        for row in quality.get("documents", [])
    }
    flagged_aliases = {
        "q_" + str(row["document_id"])
        for row in candidates
        if row["quality"]["flagged_for_dual_sol"]
    }
    if set(quality_by_alias) != flagged_aliases:
        raise ValueError("quality consensus does not exactly cover flagged candidates")
    quotas = {key: int(value) for key, value in receipt["quotas"].items()}
    selected: list[dict[str, Any]] = []
    source_counts: collections.Counter[str] = collections.Counter()
    quality_excluded: collections.Counter[str] = collections.Counter()
    ordered = sorted(
        candidates, key=lambda row: (row["source"], row["selection_rank"], row["document_id"])
    )
    for row in ordered:
        source = str(row["source"])
        if source_counts[source] >= quotas[source]:
            continue
        alias = "q_" + str(row["document_id"])
        if row["quality"]["flagged_for_dual_sol"]:
            if quality_by_alias[alias] == "UNUSABLE":
                quality_excluded[source] += 1
                continue
            if quality_by_alias[alias] != "KEEP":
                raise ValueError("unresolved quality decision")
        selected.append(_private_document(row))
        source_counts[source] += 1
    if dict(source_counts) != quotas:
        raise ValueError(f"quality gate caused a quota shortfall: {dict(source_counts)}")
    if len({(str(row["source"]), str(row["work_key"])) for row in selected}) != len(selected):
        raise ValueError("selected set has more than one representation of a work")
    if len({str(row["normalized_text_sha256"]) for row in selected}) != len(selected):
        raise ValueError("selected set has an exact normalized-text duplicate")

    documents_out = Path(args.documents_out).resolve()
    public_out = Path(args.public_exclusions_out).resolve()
    receipt_out = Path(args.receipt_out).resolve()
    selected.sort(key=lambda row: (row["source"], row["document_id"]))
    _write_jsonl_new(documents_out, selected)
    public_rows = sorted(
        (_public_exclusion(row) for row in selected), key=lambda row: row["document_id"]
    )
    public_manifest = {
        "schema_version": PUBLIC_EXCLUSION_SCHEMA,
        "purpose": "future leakage exclusion only; contains opaque IDs and hashes, no text or labels",
        "document_count": len(public_rows),
        "source_counts": dict(sorted(source_counts.items())),
        "documents": public_rows,
    }
    _write_json_new(public_out, public_manifest, mode=0o644)
    final_receipt = {
        "schema_version": SELECTION_RECEIPT_SCHEMA,
        "status": "passed_150_document_quality_gated_selection",
        "document_count": len(selected),
        "source_counts": dict(sorted(source_counts.items())),
        "quality_excluded": dict(sorted(quality_excluded.items())),
        "selection_accessed_model_predictions": False,
        "inputs": {
            "candidate_receipt_sha256": sha256_file(candidate_receipt_path),
            "quality_consensus_sha256": sha256_file(quality_path),
        },
        "sealed_outputs": {"documents_sha256": sha256_file(documents_out)},
        "public_outputs": {"exclusions_sha256": sha256_file(public_out)},
        "code_sha256": sha256_file(__file__),
    }
    _write_json_new(receipt_out, final_receipt)
    return final_receipt


def _ranges_from(
    lines: Sequence[Mapping[str, Any]],
    *,
    start: int,
    max_lines: int,
    max_chars: int,
    overlap: int,
) -> list[tuple[int, int]]:
    if not 0 <= start < len(lines):
        return []
    ranges: list[tuple[int, int]] = []
    position = start
    while position < len(lines):
        end = position
        characters = 0
        while end < len(lines) and end - position < max_lines:
            extra = len(str(lines[end]["text"])) + (1 if end > position else 0)
            if end == position and extra > max_chars:
                raise ValueError(f"line {lines[end]['line_id']} exceeds the chunk character cap")
            if end > position and characters + extra > max_chars:
                break
            characters += extra
            end += 1
        if end <= position:
            raise AssertionError("chunk builder made no progress")
        ranges.append((position, end))
        if end == len(lines):
            break
        next_position = end - overlap
        if next_position <= position:
            next_position = end
        position = next_position
    return ranges


def chunk_ranges(
    lines: Sequence[Mapping[str, Any]],
    *,
    pass_id: str,
    max_lines: int = 400,
    max_chars: int = 80_000,
    overlap: int = 15,
) -> list[tuple[int, int]]:
    """Return covering ranges with staggered boundaries for the second pass."""

    if not lines:
        raise ValueError("cannot chunk an empty document")
    if max_lines <= 2 * overlap or max_chars <= 0 or overlap < 0:
        raise ValueError("invalid chunk limits")
    ordinary = _ranges_from(
        lines, start=0, max_lines=max_lines, max_chars=max_chars, overlap=overlap
    )
    if pass_id == "pass-a" or len(ordinary) == 1:
        return ordinary
    if pass_id != "pass-b":
        raise ValueError("pass-id must be pass-a or pass-b")
    offset = max(overlap + 1, ordinary[0][1] // 2)
    prefix_end = min(len(lines), offset + overlap)
    staggered = [(0, prefix_end)] + _ranges_from(
        lines, start=offset, max_lines=max_lines, max_chars=max_chars, overlap=overlap
    )
    # The order, not the line direction, is reversed.  This changes presentation
    # order without destroying the textual context needed to recognize entries.
    unique = list(dict.fromkeys(staggered))
    return list(reversed(unique))


def _ownership(ranges: Sequence[tuple[int, int]], line_count: int) -> list[tuple[int, int]]:
    owners: list[int] = []
    for position in range(line_count):
        candidates = []
        for chunk_index, (start, end) in enumerate(ranges):
            if start <= position < end:
                edge_distance = min(position - start, end - 1 - position)
                candidates.append((-edge_distance, chunk_index))
        if not candidates:
            raise ValueError(f"chunk ranges omit line position {position}")
        owners.append(min(candidates)[1])
    owned_ranges: list[tuple[int, int]] = []
    for chunk_index, (start, end) in enumerate(ranges):
        offsets = [position - start for position, owner in enumerate(owners) if owner == chunk_index]
        if not offsets:
            owned_ranges.append((-1, -1))
            continue
        if offsets != list(range(min(offsets), max(offsets) + 1)):
            raise AssertionError("chunk ownership unexpectedly became non-contiguous")
        owned_ranges.append((min(offsets), max(offsets) + 1))
    return owned_ranges


def _validate_documents(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    if not rows or any(row.get("schema_version") != PRIVATE_DOCUMENT_SCHEMA for row in rows):
        raise ValueError("unsupported/empty sealed documents")
    document_ids: set[str] = set()
    line_ids: set[str] = set()
    for row in rows:
        document_id = str(row.get("document_id") or "")
        if not _HEX64.fullmatch(document_id) or document_id in document_ids:
            raise ValueError("invalid/duplicate document ID")
        document_ids.add(document_id)
        lines = row.get("lines")
        if not isinstance(lines, list) or not lines:
            raise ValueError(f"{document_id}: missing line inventory")
        if int(row.get("n_present_lines", len(lines))) != len(lines):
            raise ValueError(f"{document_id}: present-line count differs")
        stable_uid = str(row.get("stable_uid") or "")
        text_hash = str(row.get("materialized_text_sha256") or "")
        source = str(row.get("source") or "")
        if stable_uid and _HEX64.fullmatch(text_hash):
            if _document_id(source, stable_uid, text_hash) != document_id:
                raise ValueError(f"{document_id}: document identity does not bind its text hash")
        for line in lines:
            line_id = str(line.get("line_id") or "")
            if not _HEX64.fullmatch(line_id) or line_id in line_ids:
                raise ValueError("invalid/duplicate line ID")
            if _line_id(document_id, int(line.get("abs_idx", -1)), str(line.get("text", ""))) != line_id:
                raise ValueError(f"{document_id}: line identity does not bind coordinate/text")
            line_ids.add(line_id)
    return rows


def create_alias_secret(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        secret = os.urandom(32)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(secret)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return {
        "status": "created",
        "path": str(path),
        "bytes": 32,
        "sha256": sha256_file(path),
    }


def _packet_for_pass(
    documents: Sequence[Mapping[str, Any]],
    *,
    pass_id: str,
    secret: bytes,
    max_lines: int,
    max_chars: int,
    overlap: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chunks: list[dict[str, Any]] = []
    keys: list[dict[str, Any]] = []
    for document in documents:
        document_id = str(document["document_id"])
        document_alias = _alias(secret, "doc", document_id)
        lines = list(document["lines"])
        aliases = [_alias(secret, "ln", str(line["line_id"])) for line in lines]
        if pass_id == "pass-a":
            for line, line_alias in zip(lines, aliases, strict=True):
                keys.append(
                    {
                        "schema_version": LINE_KEY_SCHEMA,
                        "document_alias": document_alias,
                        "document_id": document_id,
                        "source": document["source"],
                        "line_alias": line_alias,
                        "line_id": line["line_id"],
                        "abs_idx": line["abs_idx"],
                    }
                )
        ranges = chunk_ranges(
            lines,
            pass_id=pass_id,
            max_lines=max_lines,
            max_chars=max_chars,
            overlap=overlap,
        )
        ownership = _ownership(ranges, len(lines))
        document_chunks = []
        for presentation_index, ((start, end), (owned_start, owned_end)) in enumerate(
            zip(ranges, ownership, strict=True)
        ):
            if owned_start < 0:
                continue
            content_contract = {
                "pass_id": pass_id,
                "document_alias": document_alias,
                "start": start,
                "end": end,
                "line_aliases": aliases[start:end],
            }
            chunk_id = "ch_" + canonical_json_sha256(content_contract)[:32]
            chunk_lines = []
            denominator = max(int(document["n_physical_lines"]) - 1, 1)
            for offset, line in enumerate(lines[start:end]):
                chunk_lines.append(
                    {
                        "offset": offset,
                        "line_alias": aliases[start + offset],
                        "abs_idx": int(line["abs_idx"]),
                        "document_position_percent": round(100 * int(line["abs_idx"]) / denominator, 4),
                        "text": str(line["text"]),
                    }
                )
            document_chunks.append(
                {
                    "schema_version": CHUNK_SCHEMA,
                    "kind": "full_document_role_pass",
                    "pass_id": pass_id,
                    "chunk_id": chunk_id,
                    "document_alias": document_alias,
                    "source": document["source"],
                    "presentation_index": presentation_index,
                    "start_present_position": start,
                    "end_present_position_exclusive": end,
                    "owned_start_offset": owned_start,
                    "owned_end_offset_exclusive": owned_end,
                    "target_offsets": [],
                    "n_physical_lines": document["n_physical_lines"],
                    "n_present_lines": len(lines),
                    "lines": chunk_lines,
                }
            )
        chunks.extend(document_chunks)
    chunks.sort(
        key=lambda row: (
            row["source"], row["document_alias"], row["presentation_index"], row["chunk_id"]
        )
    )
    keys.sort(key=lambda row: (row["source"], row["document_alias"], row["abs_idx"]))
    return chunks, keys


def prepare_annotation(args: argparse.Namespace) -> dict[str, Any]:
    documents_path = Path(args.documents).resolve()
    selection_receipt_path = Path(args.selection_receipt).resolve()
    secret_path = Path(args.alias_secret).resolve()
    documents = _validate_documents(documents_path)
    selection_receipt = _json(selection_receipt_path)
    if selection_receipt.get("sealed_outputs", {}).get("documents_sha256") != sha256_file(
        documents_path
    ):
        raise ValueError("selection receipt does not bind sealed documents")
    secret = secret_path.read_bytes()
    if len(secret) != 32 or (secret_path.stat().st_mode & 0o077):
        raise ValueError("alias secret must be a 32-byte mode-0600 file")
    pass_a, keys = _packet_for_pass(
        documents,
        pass_id="pass-a",
        secret=secret,
        max_lines=args.max_lines,
        max_chars=args.max_chars,
        overlap=args.overlap,
    )
    pass_b, _ = _packet_for_pass(
        documents,
        pass_id="pass-b",
        secret=secret,
        max_lines=args.max_lines,
        max_chars=args.max_chars,
        overlap=args.overlap,
    )
    pass_a_out = Path(args.pass_a_out).resolve()
    pass_b_out = Path(args.pass_b_out).resolve()
    key_out = Path(args.line_key_out).resolve()
    receipt_out = Path(args.receipt_out).resolve()
    _write_jsonl_new(pass_a_out, pass_a)
    _write_jsonl_new(pass_b_out, pass_b)
    _write_jsonl_new(key_out, keys)
    receipt = {
        "schema_version": "bibliography-sealed-annotation-packets-v1",
        "status": "ready_for_two_independent_sol_passes",
        "document_count": len(documents),
        "line_count": len(keys),
        "source_document_counts": dict(sorted(collections.Counter(row["source"] for row in documents).items())),
        "chunk_policy": {
            "max_lines": args.max_lines,
            "max_characters": args.max_chars,
            "overlap_lines": args.overlap,
            "pass_a": "ordinary forward boundaries",
            "pass_b": "half-chunk staggered boundaries, reversed chunk presentation",
        },
        "chunks": {"pass-a": len(pass_a), "pass-b": len(pass_b)},
        "inputs": {
            "documents_sha256": sha256_file(documents_path),
            "selection_receipt_sha256": sha256_file(selection_receipt_path),
            "alias_secret_sha256": sha256_file(secret_path),
        },
        "outputs": {
            "pass_a_packet_sha256": sha256_file(pass_a_out),
            "pass_b_packet_sha256": sha256_file(pass_b_out),
            "line_key_sha256": sha256_file(key_out),
        },
        "packet_contains_predictions": False,
        "code_sha256": sha256_file(__file__),
    }
    _write_json_new(receipt_out, receipt)
    return receipt


def _load_chunks(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    if not rows or any(row.get("schema_version") != CHUNK_SCHEMA for row in rows):
        raise ValueError("unsupported/empty role packet")
    ids = [str(row.get("chunk_id") or "") for row in rows]
    if "" in ids or len(ids) != len(set(ids)):
        raise ValueError("role packet has empty/duplicate chunk IDs")
    for row in rows:
        lines = row.get("lines")
        if not isinstance(lines, list) or not lines:
            raise ValueError("role chunk has no lines")
        if [line.get("offset") for line in lines] != list(range(len(lines))):
            raise ValueError("role chunk offsets are not contiguous")
    return rows


def _expand_runs(chunk: Mapping[str, Any], result: Mapping[str, Any]) -> list[dict[str, Any]]:
    if set(result) != {"chunk_id", "runs", "notes"}:
        raise ValueError("role response chunk has unexpected fields")
    if result.get("chunk_id") != chunk.get("chunk_id"):
        raise ValueError("role response chunk ID differs")
    runs = result.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("role response has no RLE runs")
    expanded: list[dict[str, Any]] = []
    next_offset = 0
    for run in runs:
        if not isinstance(run, dict) or set(run) != {
            "start_offset", "end_offset", "role", "confidence"
        }:
            raise ValueError("role run has unexpected fields")
        start = run.get("start_offset")
        end = run.get("end_offset")
        confidence = run.get("confidence")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or start != next_offset
            or end < start
            or end >= len(chunk["lines"])
        ):
            raise ValueError("role RLE runs must exactly and contiguously cover the chunk")
        if run.get("role") not in ROLE_NAMES:
            raise ValueError("role RLE contains an unknown role")
        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("role RLE confidence is invalid")
        for offset in range(start, end + 1):
            expanded.append(
                {
                    "offset": offset,
                    "line_alias": chunk["lines"][offset]["line_alias"],
                    "role": run["role"],
                    "confidence": float(confidence),
                }
            )
        next_offset = end + 1
    if next_offset != len(chunk["lines"]):
        raise ValueError("role RLE response omits trailing lines")
    if not isinstance(result.get("notes"), str):
        raise ValueError("role response notes must be a string")
    return expanded


def validate_role_response(
    chunks: Sequence[Mapping[str, Any]], payload: Mapping[str, Any], reviewer_id: str
) -> dict[str, Any]:
    if payload.get("schema_version") != ROLE_RESPONSE_SCHEMA:
        raise ValueError("unsupported role response schema")
    if payload.get("reviewer") != reviewer_id:
        raise ValueError("role reviewer identity differs from the run contract")
    expected = {str(row["chunk_id"]): row for row in chunks}
    results = payload.get("chunks")
    if not isinstance(results, list) or len(results) != len(expected):
        raise ValueError("role response omits or invents chunks")
    seen: set[str] = set()
    normalized = []
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("role response chunk is not an object")
        chunk_id = str(result.get("chunk_id") or "")
        if chunk_id not in expected or chunk_id in seen:
            raise ValueError("role response repeats or invents a chunk")
        _expand_runs(expected[chunk_id], result)
        seen.add(chunk_id)
        normalized.append(dict(result))
    normalized.sort(key=lambda row: row["chunk_id"])
    return {
        "schema_version": ROLE_RESPONSE_SCHEMA,
        "reviewer": reviewer_id,
        "chunks": normalized,
    }


def _batches(chunks: Sequence[Mapping[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    if batch_size not in (1, 2):
        raise ValueError("batch-size must be 1 or 2")
    return [
        [dict(row) for row in chunks[start : start + batch_size]]
        for start in range(0, len(chunks), batch_size)
    ]


def _run_contract(run_dir: Path) -> dict[str, Any]:
    value = _json(run_dir / "run.contract.json")
    if value.get("schema_version") != RUN_CONTRACT_SCHEMA:
        raise ValueError("unsupported Sol run contract")
    return value


def prepare_run(args: argparse.Namespace) -> dict[str, Any]:
    packet_path = Path(args.packet).resolve()
    chunks = _load_chunks(packet_path)
    if {str(row["pass_id"]) for row in chunks} != {args.pass_id}:
        raise ValueError("packet pass ID differs from requested pass")
    if args.model != MODEL or args.reasoning_effort != REASONING_EFFORT:
        raise ValueError(f"sealed reviews are pinned to {MODEL}/high")
    prompt_path = Path(args.prompt).resolve()
    schema_path = Path(args.output_schema).resolve()
    batches = _batches(chunks, args.batch_size)
    contract = {
        "schema_version": RUN_CONTRACT_SCHEMA,
        "kind": "role",
        "pass_id": args.pass_id,
        "reviewer_id": args.reviewer_id,
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "sandbox": "read-only",
        "ephemeral": True,
        "packet_sha256": sha256_file(packet_path),
        "prompt_sha256": sha256_file(prompt_path),
        "output_schema_sha256": sha256_file(schema_path),
        "code_sha256": sha256_file(__file__),
        "batch_size": args.batch_size,
        "batch_count": len(batches),
        "batch_ids": [
            canonical_json_sha256(
                {
                    "pass_id": args.pass_id,
                    "reviewer_id": args.reviewer_id,
                    "chunk_sha256": [canonical_json_sha256(row) for row in batch],
                }
            )
            for batch in batches
        ],
    }
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "responses").mkdir(exist_ok=True)
    _same_or_write(run_dir / "run.contract.json", contract)
    return contract


def prepare_quality_run(args: argparse.Namespace) -> dict[str, Any]:
    packet_path = Path(args.packet).resolve()
    documents = _read_jsonl(packet_path)
    if any(row.get("schema_version") != QUALITY_PACKET_SCHEMA for row in documents):
        raise ValueError("unsupported quality packet")
    aliases = [str(row.get("document_alias") or "") for row in documents]
    if "" in aliases or len(aliases) != len(set(aliases)):
        raise ValueError("quality packet has empty/duplicate aliases")
    if args.model != MODEL or args.reasoning_effort != REASONING_EFFORT:
        raise ValueError(f"sealed reviews are pinned to {MODEL}/high")
    prompt_path = Path(args.prompt).resolve()
    schema_path = Path(args.output_schema).resolve()
    batches = [
        documents[start : start + args.batch_size]
        for start in range(0, len(documents), args.batch_size)
    ]
    contract = {
        "schema_version": RUN_CONTRACT_SCHEMA,
        "kind": "quality",
        "pass_id": args.pass_id,
        "reviewer_id": args.reviewer_id,
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "sandbox": "read-only",
        "ephemeral": True,
        "packet_sha256": sha256_file(packet_path),
        "prompt_sha256": sha256_file(prompt_path),
        "output_schema_sha256": sha256_file(schema_path),
        "code_sha256": sha256_file(__file__),
        "batch_size": args.batch_size,
        "batch_count": len(batches),
        "batch_ids": [
            canonical_json_sha256(
                {
                    "pass_id": args.pass_id,
                    "reviewer_id": args.reviewer_id,
                    "document_sha256": [canonical_json_sha256(row) for row in batch],
                }
            )
            for batch in batches
        ],
    }
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "responses").mkdir(exist_ok=True)
    _same_or_write(run_dir / "run.contract.json", contract)
    return contract


def _bound_quality_batch(
    packet_path: Path, run_dir: Path, batch_index: int
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    contract = _run_contract(run_dir)
    if contract.get("kind") != "quality":
        raise ValueError("run contract is not a quality run")
    if sha256_file(packet_path) != contract["packet_sha256"]:
        raise ValueError("quality packet differs from the immutable run contract")
    documents = _read_jsonl(packet_path)
    batches = [
        documents[start : start + int(contract["batch_size"])]
        for start in range(0, len(documents), int(contract["batch_size"]))
    ]
    if not 0 <= batch_index < len(batches):
        raise ValueError("batch-index is out of range")
    return contract, batches[batch_index], str(contract["batch_ids"][batch_index])


def export_quality_batch(args: argparse.Namespace) -> dict[str, Any]:
    packet = Path(args.packet).resolve()
    run_dir = Path(args.run_dir).resolve()
    contract, documents, batch_id = _bound_quality_batch(packet, run_dir, args.batch_index)
    record_path = run_dir / "responses" / f"{batch_id}.json"
    if record_path.exists():
        record = _json(record_path)
        return {"status": "complete", "batch_id": batch_id, "review_sha256": record["review_sha256"]}
    return {
        "status": "pending",
        "batch_id": batch_id,
        "pass_id": contract["pass_id"],
        "reviewer_id": contract["reviewer_id"],
        "independence": "No predictions, labels, or other reviewer output are supplied.",
        "documents": documents,
    }


def ingest_quality_batch(args: argparse.Namespace) -> dict[str, Any]:
    packet = Path(args.packet).resolve()
    run_dir = Path(args.run_dir).resolve()
    contract, documents, batch_id = _bound_quality_batch(packet, run_dir, args.batch_index)
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise ValueError("Codex response is not an object")
    review = validate_quality_response(documents, payload, str(contract["reviewer_id"]))
    record = {
        "schema_version": RUN_RECORD_SCHEMA,
        "batch_id": batch_id,
        "contract_sha256": sha256_file(run_dir / "run.contract.json"),
        "review": review,
        "review_sha256": canonical_json_sha256(review),
    }
    _same_or_write(run_dir / "responses" / f"{batch_id}.json", record)
    return {"status": "accepted", "batch_id": batch_id, "review_sha256": record["review_sha256"]}


def finalize_quality_pass(args: argparse.Namespace) -> dict[str, Any]:
    packet_path = Path(args.packet).resolve()
    run_dir = Path(args.run_dir).resolve()
    contract = _run_contract(run_dir)
    if contract.get("kind") != "quality":
        raise ValueError("run contract is not a quality run")
    if sha256_file(packet_path) != contract.get("packet_sha256"):
        raise ValueError("quality packet differs from the immutable run contract")
    documents = _read_jsonl(packet_path)
    batches = [
        documents[start : start + int(contract["batch_size"])]
        for start in range(0, len(documents), int(contract["batch_size"]))
    ]
    rows = []
    record_hashes = []
    for index, batch in enumerate(batches):
        batch_id = str(contract["batch_ids"][index])
        record_path = run_dir / "responses" / f"{batch_id}.json"
        if not record_path.is_file():
            raise ValueError(f"quality pass is incomplete; missing batch {index}")
        record = _json(record_path)
        if (
            record.get("batch_id") != batch_id
            or record.get("contract_sha256") != sha256_file(run_dir / "run.contract.json")
            or record.get("review_sha256") != canonical_json_sha256(record.get("review"))
        ):
            raise ValueError("quality response record is not bound to this run")
        response = validate_quality_response(
            batch, record["review"], str(contract["reviewer_id"])
        )
        rows.extend(response["documents"])
        record_hashes.append(sha256_file(record_path))
    rows.sort(key=lambda row: row["document_alias"])
    output = {
        "schema_version": QUALITY_RESPONSE_SCHEMA,
        "reviewer": contract["reviewer_id"],
        "documents": rows,
    }
    output_path = Path(args.output).resolve()
    _same_or_write(output_path, output)
    receipt = {
        "schema_version": "bibliography-sealed-quality-pass-receipt-v1",
        "status": "passed",
        "reviewer": contract["reviewer_id"],
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "document_count": len(rows),
        "packet_sha256": sha256_file(packet_path),
        "run_contract_sha256": sha256_file(run_dir / "run.contract.json"),
        "record_inventory_sha256": canonical_json_sha256(record_hashes),
        "response_sha256": sha256_file(output_path),
    }
    return receipt


def _bound_batch(
    packet_path: Path, run_dir: Path, batch_index: int
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    contract = _run_contract(run_dir)
    if contract.get("kind") != "role":
        raise ValueError("run contract is not a role run")
    if sha256_file(packet_path) != contract["packet_sha256"]:
        raise ValueError("role packet differs from the immutable run contract")
    chunks = _load_chunks(packet_path)
    batches = _batches(chunks, int(contract["batch_size"]))
    if not 0 <= batch_index < len(batches):
        raise ValueError("batch-index is out of range")
    return contract, batches[batch_index], str(contract["batch_ids"][batch_index])


def export_batch(args: argparse.Namespace) -> dict[str, Any]:
    packet = Path(args.packet).resolve()
    run_dir = Path(args.run_dir).resolve()
    contract, chunks, batch_id = _bound_batch(packet, run_dir, args.batch_index)
    record_path = run_dir / "responses" / f"{batch_id}.json"
    if record_path.exists():
        record = _json(record_path)
        return {"status": "complete", "batch_id": batch_id, "review_sha256": record["review_sha256"]}
    return {
        "status": "pending",
        "batch_id": batch_id,
        "pass_id": contract["pass_id"],
        "reviewer_id": contract["reviewer_id"],
        "independence": "No labels, predictions, or other reviewer output are supplied.",
        "chunks": chunks,
    }


def ingest_batch(args: argparse.Namespace) -> dict[str, Any]:
    packet = Path(args.packet).resolve()
    run_dir = Path(args.run_dir).resolve()
    contract, chunks, batch_id = _bound_batch(packet, run_dir, args.batch_index)
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise ValueError("Codex response is not an object")
    review = validate_role_response(chunks, payload, str(contract["reviewer_id"]))
    record = {
        "schema_version": RUN_RECORD_SCHEMA,
        "batch_id": batch_id,
        "contract_sha256": sha256_file(run_dir / "run.contract.json"),
        "review": review,
        "review_sha256": canonical_json_sha256(review),
    }
    path = run_dir / "responses" / f"{batch_id}.json"
    _same_or_write(path, record)
    return {"status": "accepted", "batch_id": batch_id, "review_sha256": record["review_sha256"]}


def finalize_pass(args: argparse.Namespace) -> dict[str, Any]:
    packet_path = Path(args.packet).resolve()
    run_dir = Path(args.run_dir).resolve()
    contract = _run_contract(run_dir)
    if contract.get("kind") != "role":
        raise ValueError("run contract is not a role run")
    chunks = _load_chunks(packet_path)
    batches = _batches(chunks, int(contract["batch_size"]))
    results: dict[str, dict[str, Any]] = {}
    record_hashes: list[str] = []
    for index, batch in enumerate(batches):
        batch_id = str(contract["batch_ids"][index])
        path = run_dir / "responses" / f"{batch_id}.json"
        if not path.is_file():
            raise ValueError(f"Sol pass is incomplete; missing batch {index}")
        record = _json(path)
        if (
            record.get("batch_id") != batch_id
            or record.get("contract_sha256") != sha256_file(run_dir / "run.contract.json")
            or record.get("review_sha256") != canonical_json_sha256(record.get("review"))
        ):
            raise ValueError("Sol response record has the wrong batch ID")
        review = validate_role_response(batch, record["review"], str(contract["reviewer_id"]))
        record_hashes.append(sha256_file(path))
        for result in review["chunks"]:
            results[str(result["chunk_id"])] = result
    predictions: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    owners: dict[str, dict[str, Any]] = {}
    source_by_alias: dict[str, str] = {}
    document_by_alias: dict[str, str] = {}
    target_mode = any(row.get("kind") == "disagreement_adjudication" for row in chunks)
    for chunk in chunks:
        expanded = _expand_runs(chunk, results[str(chunk["chunk_id"])])
        source = str(chunk["source"])
        document_alias = str(chunk["document_alias"])
        by_offset = {row["offset"]: row for row in expanded}
        for row in expanded:
            alias = str(row["line_alias"])
            source_by_alias[alias] = source
            document_by_alias[alias] = document_alias
            predictions[alias].append(row)
        owned_offsets = (
            list(chunk["target_offsets"])
            if target_mode
            else list(
                range(
                    int(chunk["owned_start_offset"]),
                    int(chunk["owned_end_offset_exclusive"]),
                )
            )
        )
        for offset in owned_offsets:
            row = by_offset[int(offset)]
            alias = str(row["line_alias"])
            if alias in owners:
                raise ValueError("a line has multiple owning chunks")
            owners[alias] = row
    if not target_mode and set(owners) != set(predictions):
        raise ValueError("pass ownership does not provide complete line coverage")
    overlap_aliases = [alias for alias, values in predictions.items() if len(values) > 1]
    overlap_agree = sum(
        len({str(row["role"]) for row in predictions[alias]}) == 1 for alias in overlap_aliases
    )
    line_rows = [
        {
            "line_alias": alias,
            "document_alias": document_by_alias[alias],
            "source": source_by_alias[alias],
            "role": row["role"],
            "confidence": row["confidence"],
        }
        for alias, row in sorted(owners.items())
    ]
    output = {
        "schema_version": PASS_SCHEMA,
        "pass_id": contract["pass_id"],
        "reviewer": contract["reviewer_id"],
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "packet_sha256": sha256_file(packet_path),
        "run_contract_sha256": sha256_file(run_dir / "run.contract.json"),
        "record_inventory_sha256": canonical_json_sha256(record_hashes),
        "chunk_count": len(chunks),
        "line_count": len(line_rows),
        "overlap_line_count": len(overlap_aliases),
        "overlap_exact_role_agreement": overlap_agree / max(len(overlap_aliases), 1),
        "lines": line_rows,
    }
    _same_or_write(Path(args.output).resolve(), output)
    return output


def _pass_lines(path: Path, expected_pass: str | None = None) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    value = _json(path)
    if value.get("schema_version") != PASS_SCHEMA:
        raise ValueError(f"{path}: unsupported role pass")
    if expected_pass is not None and value.get("pass_id") != expected_pass:
        raise ValueError(f"{path}: expected {expected_pass}")
    rows = value.get("lines")
    if not isinstance(rows, list):
        raise ValueError("role pass has no line inventory")
    by_alias = {str(row.get("line_alias") or ""): dict(row) for row in rows}
    if "" in by_alias or len(by_alias) != len(rows):
        raise ValueError("role pass has empty/duplicate line aliases")
    return value, by_alias


def build_adjudication_packet(args: argparse.Namespace) -> dict[str, Any]:
    documents_path = Path(args.documents).resolve()
    key_path = Path(args.line_key).resolve()
    pass_a_path = Path(args.pass_a).resolve()
    pass_b_path = Path(args.pass_b).resolve()
    documents = _validate_documents(documents_path)
    keys = _read_jsonl(key_path)
    if not keys or any(row.get("schema_version") != LINE_KEY_SCHEMA for row in keys):
        raise ValueError("unsupported/empty line key")
    pass_a, a_lines = _pass_lines(pass_a_path, "pass-a")
    pass_b, b_lines = _pass_lines(pass_b_path, "pass-b")
    expected_aliases = {str(row["line_alias"]) for row in keys}
    if set(a_lines) != expected_aliases or set(b_lines) != expected_aliases:
        raise ValueError("independent passes do not completely cover the line key")
    targets = {
        alias
        for alias in expected_aliases
        if a_lines[alias]["role"] != b_lines[alias]["role"]
        or "UNKNOWN" in {a_lines[alias]["role"], b_lines[alias]["role"]}
    }
    key_by_alias = {str(row["line_alias"]): row for row in keys}
    aliases_by_line_id = {str(row["line_id"]): str(row["line_alias"]) for row in keys}
    targets_by_document: dict[str, list[int]] = collections.defaultdict(list)
    document_aliases: dict[str, str] = {}
    for alias in sorted(targets):
        key = key_by_alias[alias]
        targets_by_document[str(key["document_id"])].append(int(key["abs_idx"]))
        document_aliases[str(key["document_id"])] = str(key["document_alias"])
    document_by_id = {str(row["document_id"]): row for row in documents}
    chunks: list[dict[str, Any]] = []
    context_radius = args.context_radius
    for document_id, target_abs_indices in sorted(targets_by_document.items()):
        document = document_by_id[document_id]
        lines = list(document["lines"])
        position_by_abs = {int(row["abs_idx"]): position for position, row in enumerate(lines)}
        target_positions = sorted(position_by_abs[value] for value in target_abs_indices)
        groups: list[list[int]] = []
        for position in target_positions:
            if not groups:
                groups.append([position])
                continue
            proposed = groups[-1] + [position]
            start = max(0, proposed[0] - context_radius)
            end = min(len(lines), proposed[-1] + context_radius + 1)
            characters = sum(len(str(row["text"])) + 1 for row in lines[start:end])
            if end - start <= args.max_lines and characters <= args.max_chars:
                groups[-1] = proposed
            else:
                groups.append([position])
        for group_index, group in enumerate(groups):
            start = max(0, group[0] - context_radius)
            end = min(len(lines), group[-1] + context_radius + 1)
            # Shrink context symmetrically if a pathological local window hits a cap.
            while end - start > args.max_lines or sum(
                len(str(row["text"])) + 1 for row in lines[start:end]
            ) > args.max_chars:
                if start < group[0]:
                    start += 1
                if end > group[-1] + 1:
                    end -= 1
                if start == group[0] and end == group[-1] + 1:
                    break
            characters = sum(len(str(row["text"])) + 1 for row in lines[start:end])
            if end - start > args.max_lines or characters > args.max_chars:
                raise ValueError("adjudication target group cannot fit within the chunk caps")
            document_alias = document_aliases[document_id]
            target_offsets = [position - start for position in group]
            line_rows = []
            denominator = max(int(document["n_physical_lines"]) - 1, 1)
            for offset, line in enumerate(lines[start:end]):
                alias = aliases_by_line_id[str(line["line_id"])]
                line_rows.append(
                    {
                        "offset": offset,
                        "line_alias": alias,
                        "abs_idx": int(line["abs_idx"]),
                        "document_position_percent": round(100 * int(line["abs_idx"]) / denominator, 4),
                        "text": str(line["text"]),
                    }
                )
            contract = {
                "pass_id": "adjudication",
                "document_alias": document_alias,
                "group_index": group_index,
                "line_aliases": [row["line_alias"] for row in line_rows],
                "target_offsets": target_offsets,
            }
            chunks.append(
                {
                    "schema_version": CHUNK_SCHEMA,
                    "kind": "disagreement_adjudication",
                    "pass_id": "adjudication",
                    "chunk_id": "ch_" + canonical_json_sha256(contract)[:32],
                    "document_alias": document_alias,
                    "source": document["source"],
                    "presentation_index": group_index,
                    "start_present_position": start,
                    "end_present_position_exclusive": end,
                    "owned_start_offset": -1,
                    "owned_end_offset_exclusive": -1,
                    "target_offsets": target_offsets,
                    "n_physical_lines": document["n_physical_lines"],
                    "n_present_lines": len(lines),
                    "lines": line_rows,
                }
            )
    chunks.sort(key=lambda row: (row["source"], row["document_alias"], row["presentation_index"]))
    packet_out = Path(args.packet_out).resolve()
    receipt_out = Path(args.receipt_out).resolve()
    _write_jsonl_new(packet_out, chunks)
    receipt = {
        "schema_version": "bibliography-sealed-adjudication-packet-v1",
        "status": "ready_for_de_novo_third_sol" if targets else "no_adjudication_needed",
        "target_line_count": len(targets),
        "chunk_count": len(chunks),
        "context_radius": context_radius,
        "blinding": "packet contains text/context and target coordinates, but no pass-A/pass-B labels",
        "inputs": {
            "documents_sha256": sha256_file(documents_path),
            "line_key_sha256": sha256_file(key_path),
            "pass_a_sha256": sha256_file(pass_a_path),
            "pass_b_sha256": sha256_file(pass_b_path),
        },
        "output_sha256": sha256_file(packet_out),
    }
    _write_json_new(receipt_out, receipt)
    return receipt


def _binary(role: str) -> bool | None:
    if role == "UNKNOWN":
        return None
    return role in BIB_ROLES


def merge_labels(args: argparse.Namespace) -> dict[str, Any]:
    key_path = Path(args.line_key).resolve()
    pass_a_path = Path(args.pass_a).resolve()
    pass_b_path = Path(args.pass_b).resolve()
    keys = _read_jsonl(key_path)
    key_by_alias = {str(row.get("line_alias") or ""): row for row in keys}
    if "" in key_by_alias or len(key_by_alias) != len(keys):
        raise ValueError("line key has empty/duplicate aliases")
    pass_a, a_lines = _pass_lines(pass_a_path, "pass-a")
    pass_b, b_lines = _pass_lines(pass_b_path, "pass-b")
    expected = set(key_by_alias)
    if set(a_lines) != expected or set(b_lines) != expected:
        raise ValueError("both independent passes must cover every sealed line exactly once")
    if pass_a["reviewer"] == pass_b["reviewer"]:
        raise ValueError("pass A and B must use independent reviewer identities")
    disagreements = {
        alias
        for alias in expected
        if a_lines[alias]["role"] != b_lines[alias]["role"]
        or "UNKNOWN" in {a_lines[alias]["role"], b_lines[alias]["role"]}
    }
    c_lines: dict[str, dict[str, Any]] = {}
    adjudication_sha: str | None = None
    if disagreements:
        if not args.adjudication:
            raise ValueError("pass disagreements require the de-novo third Sol pass")
        adjudication_path = Path(args.adjudication).resolve()
        adjudication, c_lines = _pass_lines(adjudication_path, "adjudication")
        adjudication_sha = sha256_file(adjudication_path)
        if set(c_lines) != disagreements:
            raise ValueError("third pass must cover exactly all disagreements/UNKNOWN lines")
        if adjudication["reviewer"] in {pass_a["reviewer"], pass_b["reviewer"]}:
            raise ValueError("third pass must have a distinct reviewer identity")
    elif args.adjudication:
        raise ValueError("an adjudication pass was supplied although A and B fully agree")

    binary_totals: collections.Counter[str] = collections.Counter()
    binary_by_source: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    unresolved = 0
    rows = []
    for alias in sorted(expected):
        key = key_by_alias[alias]
        source = str(key["source"])
        first, second = str(a_lines[alias]["role"]), str(b_lines[alias]["role"])
        binary_agree = _binary(first) is not None and _binary(first) == _binary(second)
        binary_totals["agree" if binary_agree else "disagree"] += 1
        binary_by_source[source]["agree" if binary_agree else "disagree"] += 1
        votes = [first, second]
        if alias in c_lines:
            votes.append(str(c_lines[alias]["role"]))
        vote_counts = collections.Counter(votes)
        winner, frequency = vote_counts.most_common(1)[0]
        if first == second and first != "UNKNOWN":
            role = first
        elif len(votes) == 3 and frequency >= 2 and winner != "UNKNOWN":
            role = winner
        else:
            role = "UNKNOWN"
            unresolved += 1
        rows.append(
            {
                "schema_version": MERGED_SCHEMA,
                "document_id": key["document_id"],
                "line_id": key["line_id"],
                "line_alias": alias,
                "abs_idx": key["abs_idx"],
                "source": source,
                "role": role,
                "binary_label": None if role == "UNKNOWN" else ("BIB" if role in BIB_ROLES else "NON_BIB"),
                "votes": votes,
                "consensus_count": 0 if role == "UNKNOWN" else vote_counts[role],
                "label_origin": "dual_sol" if len(votes) == 2 else "dual_sol_plus_de_novo_third",
            }
        )
    total = len(rows)
    overall_agreement = binary_totals["agree"] / max(total, 1)
    per_source = {
        source: values["agree"] / max(sum(values.values()), 1)
        for source, values in sorted(binary_by_source.items())
    }
    unresolved_fraction = unresolved / max(total, 1)
    gates = {
        "complete_coverage": len(rows) == len(keys),
        "binary_agreement_overall_gte_0_98": overall_agreement >= 0.98,
        "binary_agreement_each_source_gte_0_95": all(value >= 0.95 for value in per_source.values()),
        "unresolved_fraction_lte_0_005": unresolved_fraction <= 0.005,
    }
    receipt = {
        "schema_version": "bibliography-sealed-consensus-receipt-v1",
        "status": "passed" if all(gates.values()) else "blocked",
        "label_semantics": "dual-Sol LLM-silver; not human gold",
        "line_count": total,
        "a_b_binary_agreement_overall": overall_agreement,
        "a_b_binary_agreement_by_source": per_source,
        "unresolved_count": unresolved,
        "unresolved_fraction": unresolved_fraction,
        "gates": gates,
        "inputs": {
            "line_key_sha256": sha256_file(key_path),
            "pass_a_sha256": sha256_file(pass_a_path),
            "pass_b_sha256": sha256_file(pass_b_path),
            "adjudication_sha256": adjudication_sha,
        },
    }
    output_path = Path(args.output).resolve()
    receipt_path = Path(args.receipt_out).resolve()
    _write_jsonl_new(output_path, rows)
    receipt["labels_sha256"] = sha256_file(output_path)
    _write_json_new(receipt_path, receipt)
    if receipt["status"] != "passed":
        raise ValueError(f"sealed label gates failed; receipt preserved at {receipt_path}")
    return receipt


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    documents_path = Path(args.documents).resolve()
    public_path = Path(args.public_exclusions).resolve()
    labels_path = Path(args.labels).resolve()
    consensus_path = Path(args.consensus_receipt).resolve()
    documents = _validate_documents(documents_path)
    public = _json(public_path)
    labels = _read_jsonl(labels_path)
    consensus = _json(consensus_path)
    if public.get("schema_version") != PUBLIC_EXCLUSION_SCHEMA:
        raise ValueError("unsupported public exclusions")
    if consensus.get("status") != "passed" or consensus.get("labels_sha256") != sha256_file(labels_path):
        raise ValueError("consensus receipt is blocked or unbound")
    counts = collections.Counter(str(row["source"]) for row in documents)
    if dict(counts) != DEFAULT_QUOTAS or len(documents) != 150:
        raise ValueError(f"sealed test must contain exactly 50/source, got {dict(counts)}")
    expected_lines = {(row["document_id"], line["line_id"]) for row in documents for line in row["lines"]}
    label_lines = {(row.get("document_id"), row.get("line_id")) for row in labels}
    if expected_lines != label_lines or len(label_lines) != len(labels):
        raise ValueError("labels do not exactly cover the sealed documents")
    if any(row.get("role") == "UNKNOWN" for row in labels) and consensus["unresolved_fraction"] > 0.005:
        raise ValueError("unresolved labels exceed the frozen gate")
    public_entries = public.get("documents")
    if not isinstance(public_entries, list) or {
        str(row["document_id"]) for row in public_entries
    } != {str(row["document_id"]) for row in documents}:
        raise ValueError("public exclusion IDs differ from the sealed documents")
    public_fields = {
        "document_id", "source_identity_sha256", "source_doc_identity_sha256",
        "work_identity_sha256", "stable_identity_sha256", "normalized_text_sha256",
        "materialized_text_sha256",
    }
    if any(
        set(row) != public_fields
        or any(not _HEX64.fullmatch(str(value)) for value in row.values())
        for row in public_entries
    ):
        raise ValueError("public exclusion rows may contain only the fixed opaque hash fields")
    frozen = {
        "schema_version": FREEZE_SCHEMA,
        "status": "frozen_prediction_blind_test_set",
        "label_semantics": "dual-Sol LLM-silver; not human gold",
        "document_count": len(documents),
        "line_count": len(labels),
        "source_document_counts": dict(sorted(counts.items())),
        "natural_zero_bibliography_subset": "report after model evaluation; no artificial zero-BIB cohort",
        "sealed_hashes": {
            "documents_sha256": sha256_file(documents_path),
            "labels_sha256": sha256_file(labels_path),
            "consensus_receipt_sha256": sha256_file(consensus_path),
        },
        "public_hashes": {"exclusions_sha256": sha256_file(public_path)},
        "gates": consensus["gates"],
        "code_sha256": sha256_file(__file__),
    }
    output_path = Path(args.output).resolve()
    _write_json_new(output_path, frozen, mode=0o440 if args.lock_inputs else 0o600)
    if args.lock_inputs:
        for path in (documents_path, labels_path, consensus_path):
            path.chmod(0o440)
        public_path.chmod(0o444)
    return frozen


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    select = commands.add_parser(
        "select-candidates", help="build a globally de-duplicated, quality-scored oversample pool"
    )
    select.add_argument("--normalization-root", required=True)
    select.add_argument("--historical-manifest", required=True)
    select.add_argument("--historical-root", required=True)
    select.add_argument("--previous-documents", required=True)
    select.add_argument("--expected-historical", type=int, default=2000)
    select.add_argument("--expected-previous", type=int, default=500)
    select.add_argument("--quota", action="append", default=[])
    select.add_argument("--oversample", type=int, default=4)
    select.add_argument("--seed", default="bibliography-sealed-150-v1-20260718")
    select.add_argument("--near-duplicate-threshold", type=float, default=0.80)
    select.add_argument("--candidates-out", required=True)
    select.add_argument("--quality-packet-out", required=True)
    select.add_argument("--receipt-out", required=True)

    quality = commands.add_parser(
        "merge-quality", help="merge two (or a third adjudicating) quality responses"
    )
    quality.add_argument("--packet", required=True)
    quality.add_argument("--response", action="append", required=True)
    quality.add_argument("--reviewer-id", action="append", required=True)
    quality.add_argument("--output", required=True)

    quality_adjudicate = commands.add_parser(
        "quality-adjudication-packet",
        help="make a label-blind direct subset for A/B quality disagreements",
    )
    quality_adjudicate.add_argument("--packet", required=True)
    quality_adjudicate.add_argument("--response-a", required=True)
    quality_adjudicate.add_argument("--reviewer-a", required=True)
    quality_adjudicate.add_argument("--response-b", required=True)
    quality_adjudicate.add_argument("--reviewer-b", required=True)
    quality_adjudicate.add_argument("--output", required=True)
    quality_adjudicate.add_argument("--receipt-out", required=True)

    finalize = commands.add_parser(
        "finalize-selection", help="apply quality consensus and admit exactly 50 documents/source"
    )
    finalize.add_argument("--candidates", required=True)
    finalize.add_argument("--candidate-receipt", required=True)
    finalize.add_argument("--quality-consensus", required=True)
    finalize.add_argument("--documents-out", required=True)
    finalize.add_argument("--public-exclusions-out", required=True)
    finalize.add_argument("--receipt-out", required=True)

    secret = commands.add_parser("create-alias-secret", help="create the sealed line-alias key")
    secret.add_argument("--output", required=True)

    prepare = commands.add_parser(
        "prepare-annotation", help="make independent full-document pass A/B packets"
    )
    prepare.add_argument("--documents", required=True)
    prepare.add_argument("--selection-receipt", required=True)
    prepare.add_argument("--alias-secret", required=True)
    prepare.add_argument("--max-lines", type=int, default=400)
    prepare.add_argument("--max-chars", type=int, default=80_000)
    prepare.add_argument("--overlap", type=int, default=15)
    prepare.add_argument("--pass-a-out", required=True)
    prepare.add_argument("--pass-b-out", required=True)
    prepare.add_argument("--line-key-out", required=True)
    prepare.add_argument("--receipt-out", required=True)

    run = commands.add_parser("prepare-run", help="bind a packet to an immutable Sol run")
    run.add_argument("--packet", required=True)
    run.add_argument("--pass-id", choices=("pass-a", "pass-b", "adjudication"), required=True)
    run.add_argument("--reviewer-id", required=True)
    run.add_argument("--model", default=MODEL)
    run.add_argument("--reasoning-effort", default=REASONING_EFFORT)
    run.add_argument("--prompt", required=True)
    run.add_argument("--output-schema", required=True)
    run.add_argument("--batch-size", type=int, choices=(1, 2), default=2)
    run.add_argument("--run-dir", required=True)

    quality_run = commands.add_parser(
        "prepare-quality-run", help="bind flagged documents to an immutable quality Sol run"
    )
    quality_run.add_argument("--packet", required=True)
    quality_run.add_argument("--pass-id", choices=("quality-a", "quality-b", "quality-c"), required=True)
    quality_run.add_argument("--reviewer-id", required=True)
    quality_run.add_argument("--model", default=MODEL)
    quality_run.add_argument("--reasoning-effort", default=REASONING_EFFORT)
    quality_run.add_argument("--prompt", required=True)
    quality_run.add_argument("--output-schema", required=True)
    quality_run.add_argument("--batch-size", type=int, choices=(1, 2), default=2)
    quality_run.add_argument("--run-dir", required=True)

    export = commands.add_parser("export-batch", help="stream one bounded pending batch as JSON")
    export.add_argument("--packet", required=True)
    export.add_argument("--run-dir", required=True)
    export.add_argument("--batch-index", type=int, required=True)

    ingest = commands.add_parser("ingest-batch", help="validate and atomically accept stdin JSON")
    ingest.add_argument("--packet", required=True)
    ingest.add_argument("--run-dir", required=True)
    ingest.add_argument("--batch-index", type=int, required=True)

    quality_export = commands.add_parser(
        "export-quality-batch", help="stream one bounded quality batch as JSON"
    )
    quality_export.add_argument("--packet", required=True)
    quality_export.add_argument("--run-dir", required=True)
    quality_export.add_argument("--batch-index", type=int, required=True)

    quality_ingest = commands.add_parser(
        "ingest-quality-batch", help="validate and atomically accept quality stdin JSON"
    )
    quality_ingest.add_argument("--packet", required=True)
    quality_ingest.add_argument("--run-dir", required=True)
    quality_ingest.add_argument("--batch-index", type=int, required=True)

    pass_parser = commands.add_parser("finalize-pass", help="prove coverage and aggregate a Sol pass")
    pass_parser.add_argument("--packet", required=True)
    pass_parser.add_argument("--run-dir", required=True)
    pass_parser.add_argument("--output", required=True)

    quality_pass = commands.add_parser(
        "finalize-quality-pass", help="prove coverage and aggregate a quality review pass"
    )
    quality_pass.add_argument("--packet", required=True)
    quality_pass.add_argument("--run-dir", required=True)
    quality_pass.add_argument("--output", required=True)

    adjudicate = commands.add_parser(
        "adjudication-packet", help="make a label-blind context packet for A/B disagreements"
    )
    adjudicate.add_argument("--documents", required=True)
    adjudicate.add_argument("--line-key", required=True)
    adjudicate.add_argument("--pass-a", required=True)
    adjudicate.add_argument("--pass-b", required=True)
    adjudicate.add_argument("--context-radius", type=int, default=30)
    adjudicate.add_argument("--max-lines", type=int, default=400)
    adjudicate.add_argument("--max-chars", type=int, default=80_000)
    adjudicate.add_argument("--packet-out", required=True)
    adjudicate.add_argument("--receipt-out", required=True)

    merge = commands.add_parser("merge-labels", help="accept exact 2/3 role consensus and run gates")
    merge.add_argument("--line-key", required=True)
    merge.add_argument("--pass-a", required=True)
    merge.add_argument("--pass-b", required=True)
    merge.add_argument("--adjudication")
    merge.add_argument("--output", required=True)
    merge.add_argument("--receipt-out", required=True)

    frozen = commands.add_parser("freeze", help="verify all gates and write the terminal seal receipt")
    frozen.add_argument("--documents", required=True)
    frozen.add_argument("--public-exclusions", required=True)
    frozen.add_argument("--labels", required=True)
    frozen.add_argument("--consensus-receipt", required=True)
    frozen.add_argument("--output", required=True)
    frozen.add_argument("--lock-inputs", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    handlers = {
        "select-candidates": select_candidates,
        "merge-quality": merge_quality,
        "quality-adjudication-packet": build_quality_adjudication_packet,
        "finalize-selection": finalize_selection,
        "create-alias-secret": create_alias_secret,
        "prepare-annotation": prepare_annotation,
        "prepare-run": prepare_run,
        "prepare-quality-run": prepare_quality_run,
        "export-batch": export_batch,
        "ingest-batch": ingest_batch,
        "export-quality-batch": export_quality_batch,
        "ingest-quality-batch": ingest_quality_batch,
        "finalize-pass": finalize_pass,
        "finalize-quality-pass": finalize_quality_pass,
        "adjudication-packet": build_adjudication_packet,
        "merge-labels": merge_labels,
        "freeze": freeze,
    }
    result = handlers[args.command](args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
