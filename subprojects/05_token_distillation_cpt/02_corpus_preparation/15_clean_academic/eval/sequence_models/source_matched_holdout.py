#!/usr/bin/env python3
"""Build, score, and audit a source-matched unseen-work ToC/BIB holdout.

The holdout uses the same canonical source representations as STRUCT-2K:

* Nanochat ``source_dataset=greek_phd``;
* grouped ``kallipos_sections`` works; and
* Nanochat ``source_dataset=openarchives.gr``.

All 2,000 historical STRUCT-2K identities are excluded before text is read.
Selected candidates then pass a bounded bottom-k word-shingle near-duplicate
check against the historical observed text and against one another.  This
module never changes corpus text and never tunes or fits a model.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import heapq
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

from .codex56_audit import (
    KEY_SCHEMA,
    MANIFEST_SCHEMA,
    REQUEST_SCHEMA,
    canonical_json_sha256,
    validate_responses,
)
from .contract import sha256_file
from .deterministic_structure import (
    BibRole,
    TocRole,
    analyze_bib_line,
    analyze_toc_line,
)
from .feature_crf import LinearChainCRF
from .features import FeatureEncoder, TAGS, bioes_to_classes

HOLDOUT_SCHEMA = "academic-structure-source-matched-holdout-v1"
HOLDOUT_MANIFEST_SCHEMA = "academic-structure-source-matched-holdout-manifest-v1"
PREDICTION_SCHEMA = "academic-structure-source-matched-predictions-v1"
PREDICTION_RECEIPT_SCHEMA = "academic-structure-source-matched-prediction-receipt-v1"
REVIEW_SUMMARY_SCHEMA = "academic-structure-source-matched-review-summary-v1"

DEFAULT_QUOTAS = {"greek_phd": 150, "kallipos": 150, "openarchives": 200}
SOURCE_SPECS = {
    "greek_phd": {
        "receipt_source": "nanochat_base",
        "source_dataset": "greek_phd",
        "source_family_id": "greek_phd",
    },
    "kallipos": {
        "receipt_source": "kallipos_sections",
        "source_dataset": "glossAPI/Apothetirio_Kallipos",
        "source_family_id": "apothetirio_kallipos",
    },
    "openarchives": {
        "receipt_source": "nanochat_base",
        "source_dataset": "openarchives.gr",
        "source_family_id": "openarchives",
    },
}
REVIEW_STRATA = (
    "toc_high_risk",
    "bib_high_risk",
    "model_disagreement",
    "hard_negative",
)
_NUMBERED_LINE = re.compile(r"^L\d+:\s?")
_WORD = re.compile(r"[0-9A-Za-zΑ-Ωα-ωΆ-ώϊϋΐΰἀ-῾]+", re.UNICODE)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write_json_new(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _write_jsonl_new(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for row in rows:
                handle.write(_canonical_bytes(row))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected object")
            rows.append(row)
    return rows


def _file_inventory_sha256(paths: Sequence[Path]) -> str:
    inventory = [
        {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(paths)
    ]
    return canonical_json_sha256(inventory)


def _parse_quotas(values: Sequence[str]) -> dict[str, int]:
    quotas = dict(DEFAULT_QUOTAS) if not values else {}
    for value in values:
        source, separator, raw_count = value.partition("=")
        if not separator or source not in SOURCE_SPECS:
            raise ValueError(f"invalid quota {value!r}")
        count = int(raw_count)
        if count <= 0 or source in quotas:
            raise ValueError(f"invalid/duplicate quota {value!r}")
        quotas[source] = count
    if set(quotas) != set(SOURCE_SPECS):
        raise ValueError("quotas must cover greek_phd, kallipos, and openarchives")
    return quotas


def load_historical_manifest(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    rows = _read_jsonl(path)
    if len(rows) != 2000:
        raise ValueError(f"historical STRUCT-2K manifest has {len(rows)} rows, expected 2000")
    expected = {"greek_phd": 667, "kallipos": 666, "openarchives": 667}
    counts = collections.Counter(str(row.get("source", "")) for row in rows)
    if dict(counts) != expected:
        raise ValueError(f"historical source counts differ: {dict(counts)!r}")
    identities: dict[str, set[str]] = {source: set() for source in SOURCE_SPECS}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        source = str(row.get("source", ""))
        document_id = str(row.get("doc_id", ""))
        identity = (source, document_id)
        if not document_id or identity in seen:
            raise ValueError("historical manifest has empty/duplicate identity")
        seen.add(identity)
        identities[source].add(document_id)
    return rows, identities


def _strip_numbered_text(value: str) -> str:
    return "\n".join(_NUMBERED_LINE.sub("", line) for line in value.splitlines())


def load_historical_texts(
    historical_rows: Sequence[Mapping[str, Any]], historical_root: str | Path
) -> list[dict[str, str]]:
    root = Path(historical_root)
    texts: list[dict[str, str]] = []
    batch_paths: list[Path] = []
    for row in historical_rows:
        index = int(row["i"])
        batch_path = root / f"batch_{index:05d}.json"
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
        if not isinstance(batch, list) or len(batch) != 1 or not isinstance(batch[0], dict):
            raise ValueError(f"{batch_path}: expected one document")
        item = batch[0]
        if item.get("doc_id") != row.get("doc_id") or item.get("source") != row.get("source"):
            raise ValueError(f"{batch_path}: identity differs from historical manifest")
        text = _strip_numbered_text(str(item.get("text_numbered", "")))
        if not text.strip():
            raise ValueError(f"{batch_path}: empty observed text")
        texts.append({"source": str(row["source"]), "doc_id": str(row["doc_id"]), "text": text})
        batch_paths.append(batch_path)
    # Bind every historical text artifact without exposing the inventory in the output row set.
    load_historical_texts.inventory_sha256 = _file_inventory_sha256(batch_paths)  # type: ignore[attr-defined]
    return texts


def bottom_k_word_shingles(
    text: str,
    *,
    k: int = 256,
    ngram: int = 5,
    maximum_tokens: int = 12000,
) -> frozenset[int]:
    """Return a deterministic bounded sketch of document word shingles."""

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


def _similarity(left: frozenset[int], right: frozenset[int]) -> float:
    denominator = min(len(left), len(right))
    return len(left.intersection(right)) / denominator if denominator else 0.0


class SketchIndex:
    def __init__(self) -> None:
        self.signatures: dict[str, frozenset[int]] = {}
        self.sources: dict[str, str] = {}
        self.inverted: dict[int, set[str]] = collections.defaultdict(set)

    def add(self, identity: str, source: str, signature: frozenset[int]) -> None:
        if identity in self.signatures:
            raise ValueError(f"duplicate sketch identity {identity!r}")
        self.signatures[identity] = signature
        self.sources[identity] = source
        for value in signature:
            self.inverted[value].add(identity)

    def closest(self, source: str, signature: frozenset[int]) -> tuple[str | None, float]:
        candidates: collections.Counter[str] = collections.Counter()
        for value in signature:
            candidates.update(self.inverted.get(value, ()))
        best_id: str | None = None
        best_score = 0.0
        for identity, _overlap in candidates.most_common():
            if self.sources[identity] != source:
                continue
            score = _similarity(signature, self.signatures[identity])
            if score > best_score:
                best_id, best_score = identity, score
        return best_id, best_score


def _resolve_source_shards(
    normalization_root: Path, source: str
) -> tuple[list[Path], list[Path]]:
    spec = SOURCE_SPECS[source]
    receipt_dir = (
        normalization_root
        / "canonical"
        / ".receipts"
        / "files"
        / str(spec["receipt_source"])
    )
    receipts: list[Path] = []
    shards: list[Path] = []
    for receipt_path in sorted(receipt_dir.glob("*.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        exact = receipt.get("exact_source_dataset_counts", {})
        if int(exact.get(spec["source_dataset"], 0)) <= 0:
            continue
        for shard in receipt.get("shards", []):
            output = shard.get("output", {})
            path = Path(str(output.get("path", ""))).resolve()
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"source shard is absent/linked: {path}")
            shards.append(path)
        receipts.append(receipt_path)
    if not receipts or not shards:
        raise ValueError(f"no canonical shards resolved for {source}")
    return shards, receipts


@dataclass(frozen=True)
class CandidateRow:
    source: str
    canonical_file: str
    row_group: int
    row_offset: int
    metadata: Mapping[str, Any]
    rank: int


def _candidate_metadata(
    paths: Sequence[Path],
    *,
    source: str,
    source_dataset: str,
    historical_ids: set[str],
    limit: int,
    seed: str,
) -> tuple[list[CandidateRow], dict[str, int]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - worker dependency
        raise RuntimeError("pyarrow is required on the remote worker") from error

    columns = [
        "source_dataset",
        "source_doc_id",
        "source_family_id",
        "acquisition_source_id",
        "source_repo_id",
        "source_revision",
        "source_artifact_path",
        "source_row_id",
        "source_text_field",
        "original_text_sha256",
        "normalized_text_sha256",
        "stable_uid",
        "work_key",
        "work_id",
        "representation_generation",
        "source_metadata_json",
        "cleaning_profile",
        "structural_policy",
    ]
    heap: list[tuple[int, int, CandidateRow]] = []
    sequence = 0
    counts: collections.Counter[str] = collections.Counter()
    for path in paths:
        parquet = pq.ParquetFile(path)
        for row_group in range(parquet.num_row_groups):
            table = parquet.read_row_group(row_group, columns=columns)
            for row_offset, row in enumerate(table.to_pylist()):
                if row.get("source_dataset") != source_dataset:
                    continue
                counts["rows_in_route"] += 1
                source_doc_id = str(row.get("source_doc_id") or "")
                work_id = str(row.get("work_id") or "")
                if not source_doc_id or not work_id or not row.get("stable_uid"):
                    counts["invalid_identity"] += 1
                    continue
                if source_doc_id in historical_ids or work_id in historical_ids:
                    counts["historical_identity_excluded"] += 1
                    continue
                rank = int.from_bytes(
                    hashlib.sha256(
                        f"{seed}\0{source}\0{work_id}\0{row['stable_uid']}".encode()
                    ).digest(),
                    "big",
                )
                item = CandidateRow(source, str(path), row_group, row_offset, row, rank)
                sequence += 1
                entry = (-rank, sequence, item)
                if len(heap) < limit:
                    heapq.heappush(heap, entry)
                elif rank < -heap[0][0]:
                    heapq.heapreplace(heap, entry)
    rows = [entry[2] for entry in heap]
    rows.sort(key=lambda item: (item.rank, str(item.metadata["stable_uid"])))
    counts["ranked_candidates"] = len(rows)
    return rows, dict(sorted(counts.items()))


def _load_candidate_texts(rows: Sequence[CandidateRow]) -> dict[str, str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - worker dependency
        raise RuntimeError("pyarrow is required on the remote worker") from error

    grouped: dict[tuple[str, int], list[CandidateRow]] = collections.defaultdict(list)
    for row in rows:
        grouped[(row.canonical_file, row.row_group)].append(row)
    result: dict[str, str] = {}
    for (path, row_group), selected in sorted(grouped.items()):
        table = pq.ParquetFile(path).read_row_group(row_group, columns=["stable_uid", "text"])
        materialized = table.to_pylist()
        for row in selected:
            actual = materialized[row.row_offset]
            if actual.get("stable_uid") != row.metadata.get("stable_uid"):
                raise ValueError("canonical row identity changed between metadata and text reads")
            text = actual.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"candidate {actual.get('stable_uid')}: empty text")
            result[str(actual["stable_uid"])] = text
    if len(result) != len(rows):
        raise ValueError("candidate text materialization is incomplete")
    return result


def _document_row(candidate: CandidateRow, text: str, near_duplicate: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(candidate.metadata)
    source = candidate.source
    stable_uid = str(metadata["stable_uid"])
    document_id = hashlib.sha256(f"source-matched-holdout\0{source}\0{stable_uid}".encode()).hexdigest()
    physical = text.splitlines()
    lines = []
    for abs_idx, value in enumerate(physical):
        if not value.strip():
            continue
        line_id = hashlib.sha256(
            f"{document_id}\0{abs_idx}\0{value}".encode("utf-8")
        ).hexdigest()
        lines.append({"line_id": line_id, "abs_idx": abs_idx, "text": value})
    if not lines:
        raise ValueError(f"candidate {stable_uid}: no nonblank lines")
    return {
        "schema_version": HOLDOUT_SCHEMA,
        "document_id": document_id,
        "source": source,
        "source_family_id": SOURCE_SPECS[source]["source_family_id"],
        "source_dataset": metadata["source_dataset"],
        "source_doc_id": metadata["source_doc_id"],
        "work_id": metadata["work_id"],
        "work_key": metadata["work_key"],
        "stable_uid": stable_uid,
        "source_repo_id": metadata["source_repo_id"],
        "source_revision": metadata["source_revision"],
        "source_artifact_path": metadata["source_artifact_path"],
        "source_row_id": metadata["source_row_id"],
        "source_text_field": metadata["source_text_field"],
        "representation_generation": metadata["representation_generation"],
        "cleaning_profile": metadata["cleaning_profile"],
        "structural_policy": metadata["structural_policy"],
        "original_text_sha256": metadata["original_text_sha256"],
        "normalized_text_sha256": metadata["normalized_text_sha256"],
        "materialized_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "n_physical_lines": len(physical),
        "n_present_lines": len(lines),
        "text_characters": len(text),
        "near_duplicate_audit": dict(near_duplicate),
        "source_metadata_json": metadata.get("source_metadata_json"),
        "lines": lines,
    }


def build_holdout(
    *,
    normalization_root: str | Path,
    historical_manifest: str | Path,
    historical_root: str | Path,
    quotas: Mapping[str, int],
    oversample: int,
    seed: str,
    near_duplicate_threshold: float,
    documents_out: str | Path,
    manifest_out: str | Path,
) -> dict[str, Any]:
    if oversample < 2:
        raise ValueError("oversample must be at least 2")
    if not 0.5 <= near_duplicate_threshold <= 1:
        raise ValueError("near-duplicate threshold must be in [0.5, 1]")
    normalization_root = Path(normalization_root).resolve()
    normalization_manifest = normalization_root / "normalization_manifest.json"
    if not normalization_manifest.is_file():
        raise ValueError("normalization manifest is absent")
    history, historical_ids = load_historical_manifest(historical_manifest)
    historical_texts = load_historical_texts(history, historical_root)
    historical_index = SketchIndex()
    for row in historical_texts:
        historical_index.add(
            str(row["doc_id"]),
            str(row["source"]),
            bottom_k_word_shingles(str(row["text"])),
        )
    # The index contains only bounded sketches; release the complete observed
    # documents before scanning canonical candidates on a memory-limited worker.
    del historical_texts

    source_inputs: dict[str, Any] = {}
    source_counts: dict[str, Any] = {}
    selected_documents: list[dict[str, Any]] = []
    accepted_index = SketchIndex()
    selected_by_source: collections.Counter[str] = collections.Counter()
    for source in SOURCE_SPECS:
        shards, receipts = _resolve_source_shards(normalization_root, source)
        source_inputs[source] = {
            "source_dataset": SOURCE_SPECS[source]["source_dataset"],
            "shards": [
                {"path": str(path), "bytes": path.stat().st_size}
                for path in sorted(shards)
            ],
            "receipt_inventory_sha256": _file_inventory_sha256(receipts),
        }
        candidates, counts = _candidate_metadata(
            shards,
            source=source,
            source_dataset=str(SOURCE_SPECS[source]["source_dataset"]),
            historical_ids=historical_ids[source],
            limit=int(quotas[source]) * oversample,
            seed=seed,
        )
        texts = _load_candidate_texts(candidates)
        local_counts: collections.Counter[str] = collections.Counter(counts)
        for candidate in candidates:
            stable_uid = str(candidate.metadata["stable_uid"])
            text = texts[stable_uid]
            signature = bottom_k_word_shingles(text)
            historical_match, historical_score = historical_index.closest(source, signature)
            if historical_score >= near_duplicate_threshold:
                local_counts["historical_near_duplicate_excluded"] += 1
                continue
            selected_match, selected_score = accepted_index.closest(source, signature)
            if selected_score >= near_duplicate_threshold:
                local_counts["selected_near_duplicate_excluded"] += 1
                continue
            audit = {
                "method": "bottom_k_word_5gram_v1",
                "sketch_size": 256,
                "threshold": near_duplicate_threshold,
                "historical_closest_doc_id": historical_match,
                "historical_similarity": historical_score,
                "selected_closest_document_id": selected_match,
                "selected_similarity": selected_score,
            }
            document = _document_row(candidate, text, audit)
            selected_documents.append(document)
            selected_by_source[source] += 1
            accepted_index.add(document["document_id"], source, signature)
            if selected_by_source[source] == quotas[source]:
                break
        if selected_by_source[source] != quotas[source]:
            raise ValueError(
                f"{source}: selected {selected_by_source[source]} of {quotas[source]} requested"
            )
        local_counts["selected"] = selected_by_source[source]
        source_counts[source] = dict(sorted(local_counts.items()))

    selected_documents.sort(key=lambda row: (row["source"], row["document_id"]))
    if len({row["document_id"] for row in selected_documents}) != sum(quotas.values()):
        raise ValueError("selected document identities are incomplete or duplicated")
    _write_jsonl_new(documents_out, selected_documents)
    manifest = {
        "schema_version": HOLDOUT_MANIFEST_SCHEMA,
        "status": "passed_unseen_work_selection",
        "seed": seed,
        "selection_policy": (
            "source-matched deterministic rank after all 2,000 historical identity exclusions; "
            "bottom-k word-shingle near-duplicate exclusion before quota admission"
        ),
        "source_counts": dict(sorted(selected_by_source.items())),
        "document_count": len(selected_documents),
        "work_count": len({(row["source"], row["work_key"]) for row in selected_documents}),
        "quotas": dict(sorted(quotas.items())),
        "oversample": oversample,
        "near_duplicate": {
            "method": "bottom_k_word_5gram_v1",
            "sketch_size": 256,
            "maximum_tokens": 12000,
            "threshold": near_duplicate_threshold,
            "same_source_only": True,
        },
        "historical": {
            "manifest": str(Path(historical_manifest).resolve()),
            "manifest_sha256": sha256_file(historical_manifest),
            "document_count": len(history),
            "source_counts": {source: len(values) for source, values in historical_ids.items()},
            "batch_inventory_sha256": getattr(load_historical_texts, "inventory_sha256"),
        },
        "normalization": {
            "root": str(normalization_root),
            "manifest": str(normalization_manifest),
            "manifest_sha256": sha256_file(normalization_manifest),
        },
        "source_inputs": source_inputs,
        "selection_counts": source_counts,
        "outputs": {
            "documents": str(Path(documents_out).resolve()),
            "documents_sha256": sha256_file(documents_out),
        },
        "execution": {
            "code_path": str(Path(__file__).resolve()),
            "code_sha256": sha256_file(__file__),
            "corpus_mutation_performed": False,
            "model_accessed": False,
            "labels_accessed_for_selection": False,
        },
    }
    _write_json_new(manifest_out, manifest)
    return manifest


def _read_holdout(path: str | Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    if not rows or any(row.get("schema_version") != HOLDOUT_SCHEMA for row in rows):
        raise ValueError("unsupported/empty holdout document set")
    identities = [str(row.get("document_id", "")) for row in rows]
    if "" in identities or len(identities) != len(set(identities)):
        raise ValueError("holdout document IDs are empty/duplicated")
    return rows


def _document_adapter(row: Mapping[str, Any]) -> SimpleNamespace:
    lines = [SimpleNamespace(**line) for line in row["lines"]]
    return SimpleNamespace(
        document_id=row["document_id"],
        work_id=row["work_id"],
        source=row["source"],
        n_physical_lines=row["n_physical_lines"],
        lines=lines,
    )


def _spans(lines: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    position = 0
    while position < len(lines):
        label = str(lines[position]["prediction"])
        if label == "O":
            position += 1
            continue
        end = position
        while end + 1 < len(lines) and lines[end + 1]["prediction"] == label:
            end += 1
        spans.append(
            {
                "label": label,
                "start_abs_idx": lines[position]["abs_idx"],
                "end_abs_idx": lines[end]["abs_idx"],
                "start_position": position,
                "end_position": end,
                "present_line_count": end - position + 1,
            }
        )
        position = end + 1
    return spans


_WORKER_MODEL: LinearChainCRF | None = None
_WORKER_ENCODER: FeatureEncoder | None = None
_WORKER_DELETION_BIAS: float | None = None


def _initialize_prediction_worker(model_path: str, deletion_bias: float) -> None:
    global _WORKER_MODEL, _WORKER_ENCODER, _WORKER_DELETION_BIAS
    model, metadata = LinearChainCRF.load(model_path)
    if metadata.get("architecture_id") != "c2-char-ngram-feature-bioes-crf":
        raise ValueError("prediction worker received a non-C2 model")
    if float(metadata.get("deletion_bias", -1)) != deletion_bias:
        raise ValueError("prediction worker deletion bias differs")
    char_hash = metadata.get("feature_encoder", {}).get("char_hash", {})
    _WORKER_MODEL = model
    _WORKER_ENCODER = FeatureEncoder(
        char_hash_dim=int(char_hash.get("dimension", 0)),
        char_ngram_min=int(char_hash.get("minimum_n", 2)),
        char_ngram_max=int(char_hash.get("maximum_n", 5)),
    )
    _WORKER_DELETION_BIAS = deletion_bias


def _predict_document_row(row: Mapping[str, Any]) -> dict[str, Any]:
    if _WORKER_MODEL is None or _WORKER_ENCODER is None or _WORKER_DELETION_BIAS is None:
        raise RuntimeError("prediction worker is not initialized")
    document = _document_adapter(row)
    features = _WORKER_ENCODER.encode_document(document)
    tags = _WORKER_MODEL.viterbi(features, deletion_bias=_WORKER_DELETION_BIAS)
    classes = bioes_to_classes([TAGS[int(tag)] for tag in tags])
    line_rows = [
        {"line_id": line.line_id, "abs_idx": line.abs_idx, "prediction": label}
        for line, label in zip(document.lines, classes)
    ]
    return {
        "schema_version": PREDICTION_SCHEMA,
        "model_id": "c2-char-ngram-feature-bioes-crf",
        "document_id": document.document_id,
        "work_id": document.work_id,
        "source": document.source,
        "lines": line_rows,
        "spans": _spans(line_rows),
    }


def predict_holdout(
    *,
    documents: str | Path,
    selection_manifest: str | Path,
    model_path: str | Path,
    expected_model_sha256: str,
    expected_deletion_bias: float,
    workers: int,
    predictions_out: str | Path,
    receipt_out: str | Path,
) -> dict[str, Any]:
    manifest = json.loads(Path(selection_manifest).read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != HOLDOUT_MANIFEST_SCHEMA
        or manifest.get("outputs", {}).get("documents_sha256") != sha256_file(documents)
    ):
        raise ValueError("holdout manifest is unsupported or unbound")
    actual_model_sha = sha256_file(model_path)
    if actual_model_sha != expected_model_sha256:
        raise ValueError("C2 model hash differs from the frozen expected hash")
    model, metadata = LinearChainCRF.load(model_path)
    if metadata.get("architecture_id") != "c2-char-ngram-feature-bioes-crf":
        raise ValueError("frozen model is not C2")
    deletion_bias = float(metadata.get("deletion_bias", -1))
    if deletion_bias != expected_deletion_bias:
        raise ValueError("C2 deletion bias differs from the frozen operating point")
    if not 1 <= workers <= 16:
        raise ValueError("prediction workers must be between 1 and 16")
    document_rows = _read_holdout(documents)
    if workers == 1:
        _initialize_prediction_worker(str(Path(model_path).resolve()), deletion_bias)
        output_rows = [_predict_document_row(row) for row in document_rows]
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_prediction_worker,
            initargs=(str(Path(model_path).resolve()), deletion_bias),
        ) as executor:
            output_rows = list(executor.map(_predict_document_row, document_rows, chunksize=1))
    source_counts: collections.Counter[str] = collections.Counter()
    label_counts: collections.Counter[str] = collections.Counter()
    for row in output_rows:
        source_counts[str(row["source"])] += 1
        label_counts.update(str(line["prediction"]) for line in row["lines"])
    _write_jsonl_new(predictions_out, output_rows)
    receipt = {
        "schema_version": PREDICTION_RECEIPT_SCHEMA,
        "status": "passed_frozen_c2_holdout_inference",
        "production_eligible": False,
        "model": {
            "path": str(Path(model_path).resolve()),
            "sha256": actual_model_sha,
            "architecture_id": metadata["architecture_id"],
            "deletion_bias": deletion_bias,
            "training_config_sha256": metadata.get("config_sha256"),
        },
        "inputs": {
            "documents_sha256": sha256_file(documents),
            "selection_manifest_sha256": sha256_file(selection_manifest),
        },
        "counts": {
            "documents": len(output_rows),
            "sources": dict(sorted(source_counts.items())),
            "line_predictions": dict(sorted(label_counts.items())),
            "spans": sum(len(row["spans"]) for row in output_rows),
        },
        "outputs": {
            "predictions": str(Path(predictions_out).resolve()),
            "predictions_sha256": sha256_file(predictions_out),
        },
        "execution": {
            "code_sha256": sha256_file(__file__),
            "model_fitting_performed": False,
            "threshold_tuning_performed": False,
            "corpus_mutation_performed": False,
            "workers": workers,
        },
    }
    _write_json_new(receipt_out, receipt)
    return receipt


@dataclass(frozen=True)
class ReviewCandidate:
    stratum: str
    document_id: str
    line_position: int
    risk: tuple[int, int, int, str]


def _source_balanced(
    candidates: Sequence[ReviewCandidate],
    documents: Mapping[str, Mapping[str, Any]],
    used: set[tuple[str, int]],
    limit: int,
) -> list[ReviewCandidate]:
    queues: dict[str, list[ReviewCandidate]] = collections.defaultdict(list)
    for item in candidates:
        queues[str(documents[item.document_id]["source"])].append(item)
    cursors = {source: 0 for source in queues}
    selected: list[ReviewCandidate] = []
    while len(selected) < limit:
        progressed = False
        for source in sorted(queues):
            queue = queues[source]
            cursor = cursors[source]
            while cursor < len(queue):
                item = queue[cursor]
                cursor += 1
                identity = (item.document_id, item.line_position)
                if identity in used:
                    continue
                used.add(identity)
                selected.append(item)
                progressed = True
                break
            cursors[source] = cursor
            if len(selected) == limit:
                break
        if not progressed:
            break
    return selected


def _context(
    document: Mapping[str, Any], position: int, *, radius: int
) -> list[Mapping[str, Any]]:
    lines = document["lines"]
    return lines[max(0, position - radius) : min(len(lines), position + radius + 1)]


def build_review_packet(
    *,
    documents: str | Path,
    selection_manifest: str | Path,
    predictions: str | Path,
    prediction_receipt: str | Path,
    per_stratum: int,
    context_radius: int,
    seed: str,
    requests_out: str | Path,
    key_out: str | Path,
    manifest_out: str | Path,
) -> dict[str, Any]:
    document_rows = _read_holdout(documents)
    by_id = {str(row["document_id"]): row for row in document_rows}
    prediction_rows = _read_jsonl(predictions)
    prediction_by_id = {str(row.get("document_id", "")): row for row in prediction_rows}
    if set(by_id) != set(prediction_by_id):
        raise ValueError("holdout and prediction inventories differ")
    receipt = json.loads(Path(prediction_receipt).read_text(encoding="utf-8"))
    if (
        receipt.get("schema_version") != PREDICTION_RECEIPT_SCHEMA
        or receipt.get("outputs", {}).get("predictions_sha256") != sha256_file(predictions)
        or receipt.get("inputs", {}).get("documents_sha256") != sha256_file(documents)
    ):
        raise ValueError("prediction receipt is unsupported or unbound")

    pools: dict[str, list[ReviewCandidate]] = {name: [] for name in REVIEW_STRATA}
    for document_id, document in by_id.items():
        prediction = prediction_by_id[document_id]
        labels = [str(line["prediction"]) for line in prediction["lines"]]
        if len(labels) != len(document["lines"]):
            raise ValueError(f"{document_id}: line inventory mismatch")
        span_position: dict[int, Mapping[str, Any]] = {}
        for span in prediction.get("spans", []):
            middle = (int(span["start_position"]) + int(span["end_position"])) // 2
            span_position[middle] = span
        for position, (line, label) in enumerate(zip(document["lines"], labels)):
            toc = analyze_toc_line(str(line["text"]), int(line["abs_idx"]))
            bib = analyze_bib_line(str(line["text"]), int(line["abs_idx"]))
            stable = hashlib.sha256(
                f"{seed}\0{document_id}\0{line['line_id']}".encode()
            ).hexdigest()
            span = span_position.get(position)
            if span and label == "TOC":
                unusual = int(int(span["start_abs_idx"]) > 500 or int(span["present_line_count"]) <= 2)
                pools["toc_high_risk"].append(
                    ReviewCandidate("toc_high_risk", document_id, position, (unusual, int(toc.hard_negative), -int(span["present_line_count"]), stable))
                )
            if span and label == "BIB":
                unusual = int(int(span["start_abs_idx"]) < int(document["n_physical_lines"]) // 3 or int(span["present_line_count"]) <= 2)
                pools["bib_high_risk"].append(
                    ReviewCandidate("bib_high_risk", document_id, position, (unusual, int(bib.hard_negative), -int(span["present_line_count"]), stable))
                )
            evidence_label = None
            if toc.role in {TocRole.HEADING, TocRole.STRONG_ENTRY} and toc.score >= 2:
                evidence_label = "TOC"
            if bib.role in {BibRole.HEADING, BibRole.STRONG_ENTRY_START} and bib.score >= 2:
                evidence_label = "BIB" if evidence_label is None else "CONFLICT"
            if evidence_label and evidence_label != label:
                pools["model_disagreement"].append(
                    ReviewCandidate(
                        "model_disagreement",
                        document_id,
                        position,
                        (int(evidence_label != "CONFLICT"), int(max(toc.score, bib.score) * 100), int(label != "O"), stable),
                    )
                )
            if label == "O" and (toc.hard_negative or bib.hard_negative):
                pools["hard_negative"].append(
                    ReviewCandidate(
                        "hard_negative",
                        document_id,
                        position,
                        (int(toc.hard_negative and bib.hard_negative), int(max(toc.score, bib.score) * 100), 0, stable),
                    )
                )
    for rows in pools.values():
        rows.sort(key=lambda item: item.risk, reverse=True)

    used: set[tuple[str, int]] = set()
    requests: list[dict[str, Any]] = []
    keys: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    source_counts: dict[str, dict[str, int]] = {}
    for stratum in REVIEW_STRATA:
        chosen = _source_balanced(pools[stratum], by_id, used, per_stratum)
        if len(chosen) != per_stratum:
            raise ValueError(f"{stratum}: selected {len(chosen)} of {per_stratum}")
        counts[stratum] = len(chosen)
        source_counts[stratum] = dict(
            sorted(collections.Counter(by_id[item.document_id]["source"] for item in chosen).items())
        )
        for item in chosen:
            document = by_id[item.document_id]
            prediction = prediction_by_id[item.document_id]
            target = document["lines"][item.line_position]
            context = _context(document, item.line_position, radius=context_radius)
            identity = {
                "seed": seed,
                "stratum": item.stratum,
                "document_id": item.document_id,
                "line_id": target["line_id"],
                "prompt_version": "source-matched-holdout-review-v1",
            }
            request_id = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
            request = {
                "schema_version": REQUEST_SCHEMA,
                "request_id": request_id,
                "prompt_version": "source-matched-holdout-review-v1",
                "source": document["source"],
                "opaque_document_id": hashlib.sha256(
                    f"{seed}\0{item.document_id}".encode()
                ).hexdigest(),
                "target_abs_idx": target["abs_idx"],
                "context_start_abs_idx": context[0]["abs_idx"],
                "context_end_abs_idx": context[-1]["abs_idx"],
                "context_coverage": "full_document_local_present_line_window",
                "crosses_unrepresented_interval": False,
                "lines": [
                    {"abs_idx": line["abs_idx"], "line_id": line["line_id"], "text": line["text"]}
                    for line in context
                ],
            }
            request_sha = canonical_json_sha256(request)
            request["request_sha256"] = request_sha
            labels = prediction["lines"]
            candidate_label = str(labels[item.line_position]["prediction"])
            containing_spans = [
                span
                for span in prediction.get("spans", [])
                if int(span["start_position"]) <= item.line_position <= int(span["end_position"])
            ]
            requests.append(request)
            keys.append(
                {
                    "schema_version": KEY_SCHEMA,
                    "request_id": request_id,
                    "request_sha256": request_sha,
                    "stratum": item.stratum,
                    "document_id": item.document_id,
                    "work_id": document["work_id"],
                    "source": document["source"],
                    "source_doc_id": document["source_doc_id"],
                    "line_id": target["line_id"],
                    "abs_idx": target["abs_idx"],
                    "candidate_prediction": candidate_label,
                    "candidate_spans": containing_spans,
                    "gold_label": "UNKNOWN",
                    "selection_risk": list(item.risk[:3]),
                }
            )
    requests.sort(key=lambda row: row["request_id"])
    keys.sort(key=lambda row: row["request_id"])
    _write_jsonl_new(requests_out, requests)
    _write_jsonl_new(key_out, keys)
    os.chmod(key_out, 0o600)
    packet_manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "seed": seed,
        "prompt_version": "source-matched-holdout-review-v1",
        "allowed_split": "source_matched_unseen_work_holdout",
        "per_stratum_requested": per_stratum,
        "context_radius_present_lines": context_radius,
        "counts": counts,
        "source_counts": source_counts,
        "shortfalls": {},
        "request_count": len(requests),
        "request_set_sha256": canonical_json_sha256(requests),
        "key_set_sha256": canonical_json_sha256(keys),
        "blinding_level": "prompt_blinded_not_access_isolated",
        "blinding": "review requests omit selection stratum and C2 predictions",
        "selection_design": (
            "source-balanced risk sample of C2 ToC/BIB spans, deterministic evidence disagreements, "
            "and retained hard negatives on a frozen unseen-work holdout"
        ),
        "inputs": {
            "documents_sha256": sha256_file(documents),
            "selection_manifest_sha256": sha256_file(selection_manifest),
            "predictions_sha256": sha256_file(predictions),
            "prediction_receipt_sha256": sha256_file(prediction_receipt),
        },
    }
    _write_json_new(manifest_out, packet_manifest)
    return packet_manifest


def summarize_review(
    *,
    requests: str | Path,
    key: str | Path,
    responses: str | Path,
    expected_model: str,
    output: str | Path,
) -> dict[str, Any]:
    request_rows = _read_jsonl(requests)
    key_rows = _read_jsonl(key)
    response_rows = _read_jsonl(responses)
    validation = validate_responses(request_rows, response_rows, expected_model=expected_model)
    keys = {str(row["request_id"]): row for row in key_rows}
    responses_by_id = {str(row["request_id"]): row for row in response_rows}
    if set(keys) != set(responses_by_id):
        raise ValueError("review key/response inventories differ")
    global_counts: collections.Counter[str] = collections.Counter()
    slices: dict[tuple[str, str], collections.Counter[str]] = collections.defaultdict(collections.Counter)
    examples: list[dict[str, Any]] = []
    for request_id in sorted(keys):
        audit_key = keys[request_id]
        response = responses_by_id[request_id]
        model_label = str(audit_key["candidate_prediction"])
        review_label = "O" if response["label"] in {"OTHER", "UNKNOWN"} else str(response["label"])
        agreement = model_label == review_label
        bucket = "agreement" if agreement else "disagreement"
        global_counts[bucket] += 1
        slices[(str(audit_key["source"]), str(audit_key["stratum"]))][bucket] += 1
        if not agreement:
            examples.append(
                {
                    "request_id": request_id,
                    "source": audit_key["source"],
                    "stratum": audit_key["stratum"],
                    "document_id": audit_key["document_id"],
                    "abs_idx": audit_key["abs_idx"],
                    "c2_prediction": model_label,
                    "codex_review": response["label"],
                    "confidence": response["confidence"],
                }
            )
    summary = {
        "schema_version": REVIEW_SUMMARY_SCHEMA,
        "status": "passed_codex_review_summary",
        "evidence_semantics": "source-balanced risk audit; not a corpus prevalence estimate",
        "reviewer_model": expected_model,
        "counts": dict(sorted(global_counts.items())),
        "agreement_rate": global_counts["agreement"] / len(keys),
        "by_source_stratum": {
            f"{source}/{stratum}": dict(sorted(values.items()))
            for (source, stratum), values in sorted(slices.items())
        },
        "disagreements": examples,
        "validation": validation,
        "inputs": {
            "requests_sha256": sha256_file(requests),
            "key_sha256": sha256_file(key),
            "responses_sha256": sha256_file(responses),
        },
    }
    _write_json_new(output, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--normalization-root", required=True)
    build.add_argument("--historical-manifest", required=True)
    build.add_argument("--historical-root", required=True)
    build.add_argument("--quota", action="append", default=[])
    build.add_argument("--oversample", type=int, default=4)
    build.add_argument("--seed", default="source-matched-holdout-v1-20260713")
    build.add_argument("--near-duplicate-threshold", type=float, default=0.80)
    build.add_argument("--documents-out", required=True)
    build.add_argument("--manifest-out", required=True)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--documents", required=True)
    predict.add_argument("--selection-manifest", required=True)
    predict.add_argument("--model", required=True)
    predict.add_argument("--expected-model-sha256", required=True)
    predict.add_argument("--expected-deletion-bias", type=float, required=True)
    predict.add_argument("--workers", type=int, default=8)
    predict.add_argument("--predictions-out", required=True)
    predict.add_argument("--receipt-out", required=True)

    packet = subparsers.add_parser("build-review")
    packet.add_argument("--documents", required=True)
    packet.add_argument("--selection-manifest", required=True)
    packet.add_argument("--predictions", required=True)
    packet.add_argument("--prediction-receipt", required=True)
    packet.add_argument("--per-stratum", type=int, default=50)
    packet.add_argument("--context-radius", type=int, default=20)
    packet.add_argument("--seed", default="source-matched-holdout-review-v1-20260713")
    packet.add_argument("--requests-out", required=True)
    packet.add_argument("--key-out", required=True)
    packet.add_argument("--manifest-out", required=True)

    summary = subparsers.add_parser("summarize")
    summary.add_argument("--requests", required=True)
    summary.add_argument("--key", required=True)
    summary.add_argument("--responses", required=True)
    summary.add_argument("--expected-model", required=True)
    summary.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    if args.command == "build":
        result = build_holdout(
            normalization_root=args.normalization_root,
            historical_manifest=args.historical_manifest,
            historical_root=args.historical_root,
            quotas=_parse_quotas(args.quota),
            oversample=args.oversample,
            seed=args.seed,
            near_duplicate_threshold=args.near_duplicate_threshold,
            documents_out=args.documents_out,
            manifest_out=args.manifest_out,
        )
    elif args.command == "predict":
        result = predict_holdout(
            documents=args.documents,
            selection_manifest=args.selection_manifest,
            model_path=args.model,
            expected_model_sha256=args.expected_model_sha256,
            expected_deletion_bias=args.expected_deletion_bias,
            workers=args.workers,
            predictions_out=args.predictions_out,
            receipt_out=args.receipt_out,
        )
    elif args.command == "build-review":
        result = build_review_packet(
            documents=args.documents,
            selection_manifest=args.selection_manifest,
            predictions=args.predictions,
            prediction_receipt=args.prediction_receipt,
            per_stratum=args.per_stratum,
            context_radius=args.context_radius,
            seed=args.seed,
            requests_out=args.requests_out,
            key_out=args.key_out,
            manifest_out=args.manifest_out,
        )
    else:
        result = summarize_review(
            requests=args.requests,
            key=args.key,
            responses=args.responses,
            expected_model=args.expected_model,
            output=args.output,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
