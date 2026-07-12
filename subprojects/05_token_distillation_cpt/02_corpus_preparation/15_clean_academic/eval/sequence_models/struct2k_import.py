#!/usr/bin/env python3
"""Audit and transcode the recovered joint STRUCT-2K LLM-silver handoff.

This importer is intentionally separate from model fitting.  It verifies the
offline handoff inventory, replays the historical batch+annotation -> line-label
join, applies only the explicitly locked coordinate-typo corrections, and then
physically excludes the historical test partition.  The emitted train/validation
corpus is the only artifact that classifier processes may open.

The legacy ``*_gold`` filename is historical.  No row in this workflow is human
gold, and this module never creates a new semantic annotation.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence

from .contract import (
    build_split_manifest,
    canonical_json_sha256,
    parse_gold_rows,
    read_gold,
    sha256_file,
    validate_silver,
)
from .silver_reconstruct import ExactTokenizer, TOKENIZER_REVISION, TOKENIZER_SHA256

LINE_RE = re.compile(r"^L(\d+):\s?(.*)$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SOURCE_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
LEGACY_LABELS = {0: "O", 1: "BIB", 2: "TOC"}
SECTION_LABELS = {"bibliography": 1, "table_of_contents": 2}
SNAPSHOT_SCHEMA = "struct2k-handoff-audit-receipt-v1"


class Struct2KImportError(ValueError):
    """Raised when recovered evidence cannot support the joint comparison."""


@dataclass(frozen=True)
class Identity:
    document_id: str
    work_id: str
    source: str


@dataclass(frozen=True)
class LegacyDocumentAudit:
    row_index: int
    upstream_document_id: str
    document_id: str
    work_id: str
    representation_id: str
    source: str
    historical_split: str
    mode: str
    n_physical_lines: int
    n_present_lines: int
    observed_text_sha256: str
    corrected_label_delta_lines: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Struct2KImportError(message)


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            _require(
                isinstance(value, dict), f"{path}: row {row_number} is not an object"
            )
            yield value


def _hash(*parts: object) -> str:
    return hashlib.sha256("\0".join(str(part) for part in parts).encode()).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _atomic_bytes(path: str | Path, payload: bytes) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing immutable output overwrite: {output}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: str | Path, value: Any) -> None:
    _atomic_bytes(path, _canonical_bytes(value))


def _safe_inventory_path(raw: str) -> str:
    _require(raw.startswith("./"), f"inventory path lacks ./ prefix: {raw!r}")
    normalized = raw[2:]
    candidate = PurePosixPath(normalized)
    _require(
        normalized
        and not candidate.is_absolute()
        and ".." not in candidate.parts
        and str(candidate) == normalized,
        f"unsafe inventory path: {raw!r}",
    )
    return normalized


def parse_inventory(path: str | Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.rstrip("\n")
            parts = line.split("  ", 1)
            _require(len(parts) == 2, f"inventory line {line_number}: malformed")
            digest, raw_path = parts
            _require(
                bool(SHA256_RE.fullmatch(digest)),
                f"inventory line {line_number}: bad SHA",
            )
            relative = _safe_inventory_path(raw_path)
            _require(
                relative not in entries,
                f"inventory line {line_number}: duplicate {relative}",
            )
            entries[relative] = digest
    _require(bool(entries), "handoff inventory is empty")
    return entries


def verify_handoff_inventory(
    handoff_root: str | Path,
    lock: Mapping[str, Any],
    *,
    verify_all_files: bool,
) -> dict[str, str]:
    root = Path(handoff_root)
    _require(
        root.is_dir() and not root.is_symlink(), "handoff root must be a real directory"
    )
    inventory_lock = lock.get("inventory")
    _require(isinstance(inventory_lock, Mapping), "lock lacks inventory contract")
    inventory_path = root / str(inventory_lock.get("filename"))
    _require(
        inventory_path.is_file() and not inventory_path.is_symlink(),
        "inventory is not regular",
    )
    _require(
        sha256_file(inventory_path) == inventory_lock.get("sha256"),
        "INVENTORY.sha256 bytes differ from the pinned lock",
    )
    entries = parse_inventory(inventory_path)
    _require(
        len(entries) == inventory_lock.get("entry_count"),
        "handoff inventory entry count differs from the lock",
    )
    required = inventory_lock.get("required_files")
    _require(isinstance(required, Mapping), "lock lacks required-file hashes")
    for relative, digest in required.items():
        _require(
            entries.get(relative) == digest,
            f"required inventory entry drift: {relative}",
        )
    selected = entries if verify_all_files else {key: entries[key] for key in required}
    for relative, expected in selected.items():
        path = root / relative
        _require(
            path.is_file() and not path.is_symlink(),
            f"listed handoff file is not regular: {relative}",
        )
        _require(
            sha256_file(path) == expected, f"handoff file hash mismatch: {relative}"
        )
    return entries


def _present_lines(numbered_text: str) -> list[list[Any]]:
    result: list[list[Any]] = []
    for raw in numbered_text.split("\n"):
        match = LINE_RE.match(raw)
        if match and match.group(2).strip() and "elided" not in match.group(2):
            result.append([int(match.group(1)), match.group(2)])
    return result


def _expected_historical_split(document_id: str) -> str:
    value = int(
        hashlib.md5(document_id.encode(), usedforsecurity=False).hexdigest(), 16
    )
    return "test" if value % 10 < 3 else "train"


def _correction_key(
    annotation_file: str, document_id: str, kind: str, coordinates: Sequence[int]
) -> tuple[str, str, str, tuple[int, int]]:
    _require(len(coordinates) == 2, "coordinate pair must contain exactly two values")
    return (
        annotation_file,
        document_id,
        kind,
        (int(coordinates[0]), int(coordinates[1])),
    )


def _correction_map(
    lock: Mapping[str, Any],
) -> dict[tuple[str, str, str, tuple[int, int]], tuple[int, int]]:
    result: dict[tuple[str, str, str, tuple[int, int]], tuple[int, int]] = {}
    for row_number, row in enumerate(lock.get("coordinate_corrections", []), 1):
        _require(
            isinstance(row, Mapping),
            f"coordinate correction {row_number} is not an object",
        )
        key = _correction_key(
            str(row.get("annotation_file")),
            str(row.get("document_id")),
            str(row.get("kind")),
            row.get("original", []),
        )
        corrected = tuple(int(value) for value in row.get("corrected", []))
        _require(
            len(corrected) == 2, f"coordinate correction {row_number} is malformed"
        )
        _require(key not in result, f"duplicate coordinate correction {key!r}")
        _require(
            isinstance(row.get("evidence"), str) and row["evidence"],
            "correction lacks evidence",
        )
        result[key] = (corrected[0], corrected[1])
    return result


def _sections_with_locked_corrections(
    annotation: Mapping[str, Any],
    annotation_relative: str,
    corrections: Mapping[tuple[str, str, str, tuple[int, int]], tuple[int, int]],
    consumed: set[tuple[str, str, str, tuple[int, int]]],
) -> list[tuple[str, int, int]]:
    document_id = str(annotation.get("doc_id"))
    n_lines = annotation.get("n_lines")
    _require(
        isinstance(n_lines, int) and n_lines > 0,
        f"{annotation_relative}: invalid n_lines",
    )
    sections: list[tuple[str, int, int]] = []
    raw_sections = annotation.get("sections")
    _require(
        isinstance(raw_sections, list), f"{annotation_relative}: sections is not a list"
    )
    for position, raw in enumerate(raw_sections):
        _require(
            isinstance(raw, Mapping),
            f"{annotation_relative}: section {position} is not an object",
        )
        kind = raw.get("kind")
        start, end = raw.get("start_line"), raw.get("end_line")
        _require(
            kind in SECTION_LABELS,
            f"{annotation_relative}: unsupported section kind {kind!r}",
        )
        _require(
            isinstance(start, int) and isinstance(end, int),
            f"{annotation_relative}: non-integer span",
        )
        key = _correction_key(annotation_relative, document_id, str(kind), (start, end))
        if key in corrections:
            start, end = corrections[key]
            consumed.add(key)
        _require(
            0 <= start <= end < n_lines,
            f"{annotation_relative}: invalid uncorrected section {kind}[{start},{end}]",
        )
        sections.append((str(kind), start, end))
    for left_index, left in enumerate(sections):
        for right in sections[left_index + 1 :]:
            _require(
                left[2] < right[1] or right[2] < left[1],
                f"{annotation_relative}: corrected sections overlap: {left!r}/{right!r}",
            )
    return sections


def _labels_for_lines(
    lines: Sequence[Sequence[Any]], sections: Sequence[tuple[str, int, int]]
) -> list[int]:
    labels: dict[int, int] = {}
    for kind, start, end in sections:
        for coordinate in range(start, end + 1):
            labels[coordinate] = SECTION_LABELS[kind]
    return [labels.get(int(line[0]), 0) for line in lines]


def _legacy_sections(annotation: Mapping[str, Any]) -> list[tuple[str, int, int]]:
    return [
        (str(row["kind"]), int(row["start_line"]), int(row["end_line"]))
        for row in annotation["sections"]
    ]


def _identity(
    source: str, upstream_id: str, mode: str, text_sha256: str
) -> tuple[str, str, str]:
    document_id = _hash(
        "struct2k-llm-silver-document-v1", source, upstream_id, text_sha256
    )
    work_id = _hash("struct2k-llm-silver-exact-work-v1", text_sha256)
    representation_id = _hash(
        "struct2k-llm-silver-representation-v1", source, upstream_id, mode, text_sha256
    )
    return document_id, work_id, representation_id


def audit_handoff(
    handoff_root: str | Path,
    lock_path: str | Path,
    *,
    verify_all_files: bool = True,
) -> tuple[dict[str, Any], list[LegacyDocumentAudit], dict[str, str]]:
    """Replay all legacy joins and return a receipt-ready audit summary."""
    root = Path(handoff_root)
    lock = _load_json(lock_path)
    _require(
        lock.get("schema_version") == "struct2k-handoff-lock-v1", "unsupported lock"
    )
    source = lock.get("source", {})
    _require(
        bool(SOURCE_COMMIT_RE.fullmatch(str(source.get("commit", "")))),
        "bad source commit",
    )
    inventory = verify_handoff_inventory(root, lock, verify_all_files=verify_all_files)
    manifest_rows = list(_iter_jsonl(root / "STRUCT_2K" / "manifest.jsonl"))
    corrections = _correction_map(lock)
    consumed: set[tuple[str, str, str, tuple[int, int]]] = set()
    audits: list[LegacyDocumentAudit] = []
    seen_upstream: set[str] = set()
    source_counts: collections.Counter[str] = collections.Counter()
    split_counts: collections.Counter[str] = collections.Counter()
    label_counts: collections.Counter[int] = collections.Counter()
    corrected_label_counts: collections.Counter[int] = collections.Counter()
    mode_counts: collections.Counter[str] = collections.Counter()
    engine_counts: collections.Counter[str] = collections.Counter()
    effort_counts: collections.Counter[str] = collections.Counter()
    total_delta = 0
    for index, legacy in enumerate(_iter_jsonl(root / "STRUCT_2K_gold.jsonl")):
        _require(
            index < len(manifest_rows), "gold contains more rows than the manifest"
        )
        annotation_relative = f"STRUCT_2K/ann_{index:05d}.json"
        batch_relative = f"STRUCT_2K/batch_{index:05d}.json"
        _require(
            annotation_relative in inventory, f"inventory lacks {annotation_relative}"
        )
        _require(batch_relative in inventory, f"inventory lacks {batch_relative}")
        annotation = _load_json(root / annotation_relative)
        batch_container = _load_json(root / batch_relative)
        _require(
            isinstance(batch_container, list)
            and len(batch_container) == 1
            and isinstance(batch_container[0], Mapping),
            f"{batch_relative}: expected one batch object",
        )
        batch = batch_container[0]
        manifest = manifest_rows[index]
        upstream_id = legacy.get("doc_id")
        _require(
            isinstance(upstream_id, str) and upstream_id,
            f"gold row {index}: missing doc_id",
        )
        _require(
            upstream_id not in seen_upstream,
            f"duplicate legacy document_id {upstream_id}",
        )
        seen_upstream.add(upstream_id)
        source_name = legacy.get("source")
        split = legacy.get("split")
        mode = legacy.get("mode")
        n_lines = legacy.get("n_lines")
        _require(
            source_name in {"greek_phd", "openarchives", "kallipos"},
            "unexpected source",
        )
        _require(split in {"train", "test"}, f"{upstream_id}: invalid historical split")
        _require(
            split == _expected_historical_split(upstream_id),
            f"{upstream_id}: split drift",
        )
        _require(isinstance(mode, str) and mode, f"{upstream_id}: missing mode")
        _require(
            isinstance(n_lines, int) and n_lines > 0, f"{upstream_id}: invalid n_lines"
        )
        for context, value in (
            ("annotation", annotation),
            ("batch", batch),
            ("manifest", manifest),
        ):
            _require(
                value.get("doc_id") == upstream_id,
                f"{upstream_id}: {context} identity drift",
            )
            _require(
                value.get("source") == source_name,
                f"{upstream_id}: {context} source drift",
            )
            _require(
                value.get("split") == split, f"{upstream_id}: {context} split drift"
            )
            _require(value.get("mode") == mode, f"{upstream_id}: {context} mode drift")
            _require(
                value.get("n_lines") == n_lines,
                f"{upstream_id}: {context} n_lines drift",
            )
        _require(manifest.get("i") == index, f"{upstream_id}: manifest row index drift")
        engine = annotation.get("_engine")
        _require(
            isinstance(engine, Mapping), f"{upstream_id}: annotation engine missing"
        )
        engine_counts[str(engine.get("model"))] += 1
        effort_counts[str(engine.get("effort"))] += 1
        numbered = batch.get("text_numbered")
        _require(
            isinstance(numbered, str), f"{upstream_id}: batch numbered text missing"
        )
        present = _present_lines(numbered)
        raw_lines = legacy.get("lines")
        _require(
            isinstance(raw_lines, list) and raw_lines, f"{upstream_id}: lines missing"
        )
        _require(
            all(
                isinstance(row, list)
                and len(row) == 3
                and isinstance(row[0], int)
                and isinstance(row[1], str)
                and isinstance(row[2], int)
                and row[2] in LEGACY_LABELS
                for row in raw_lines
            ),
            f"{upstream_id}: malformed legacy line triple",
        )
        _require(
            [[row[0], row[1]] for row in raw_lines] == present,
            f"{upstream_id}: gold text does not replay from its batch",
        )
        coordinates = [int(row[0]) for row in raw_lines]
        _require(
            coordinates == sorted(set(coordinates)),
            f"{upstream_id}: coordinates are not unique/increasing",
        )
        _require(
            coordinates[-1] < n_lines, f"{upstream_id}: last coordinate exceeds n_lines"
        )
        legacy_labels = [int(row[2]) for row in raw_lines]
        _require(
            legacy_labels == _labels_for_lines(raw_lines, _legacy_sections(annotation)),
            f"{upstream_id}: legacy labels do not replay from uncorrected annotations",
        )
        corrected_sections = _sections_with_locked_corrections(
            annotation, annotation_relative, corrections, consumed
        )
        corrected_labels = _labels_for_lines(raw_lines, corrected_sections)
        delta = sum(
            left != right for left, right in zip(legacy_labels, corrected_labels)
        )
        text_sha = canonical_json_sha256([[row[0], row[1]] for row in raw_lines])
        document_id, work_id, representation_id = _identity(
            str(source_name), upstream_id, str(mode), text_sha
        )
        audits.append(
            LegacyDocumentAudit(
                row_index=index,
                upstream_document_id=upstream_id,
                document_id=document_id,
                work_id=work_id,
                representation_id=representation_id,
                source=str(source_name),
                historical_split=str(split),
                mode=str(mode),
                n_physical_lines=n_lines,
                n_present_lines=len(raw_lines),
                observed_text_sha256=text_sha,
                corrected_label_delta_lines=delta,
            )
        )
        source_counts[str(source_name)] += 1
        split_counts[str(split)] += 1
        mode_counts[str(mode)] += 1
        label_counts.update(legacy_labels)
        corrected_label_counts.update(corrected_labels)
        total_delta += delta
    _require(len(audits) == len(manifest_rows), "gold/manifest document count mismatch")
    _require(
        consumed == set(corrections),
        "one or more locked coordinate corrections were not consumed",
    )
    expected = lock.get("legacy_contract", {})
    _require(
        len(audits) == expected.get("document_count"), "legacy document count drift"
    )
    _require(
        sum(row.n_present_lines for row in audits)
        == expected.get("present_line_count"),
        "line count drift",
    )
    _require(
        dict(sorted(source_counts.items())) == expected.get("source_counts"),
        "source counts drift",
    )
    _require(
        dict(sorted(split_counts.items())) == expected.get("historical_split_counts"),
        "split counts drift",
    )
    _require(
        {str(key): value for key, value in sorted(label_counts.items())}
        == expected.get("label_counts"),
        "legacy label counts drift",
    )
    _require(
        engine_counts == {expected.get("annotation_engine"): len(audits)},
        "engine drift",
    )
    _require(
        effort_counts == {expected.get("annotation_effort"): len(audits)},
        "effort drift",
    )
    exact_hash_splits: dict[str, set[str]] = collections.defaultdict(set)
    for row in audits:
        exact_hash_splits[row.observed_text_sha256].add(row.historical_split)
    summary = {
        "schema_version": SNAPSHOT_SCHEMA,
        "status": "passed_inventory_and_legacy_replay_with_locked_coordinate_corrections",
        "operation": "audit_and_transcode_existing_joint_LLM_silver",
        "annotation_status": "LLM_silver",
        "annotation_engine": expected.get("annotation_engine"),
        "human_gold": False,
        "new_semantic_annotations_created": False,
        "coordinate_corrections_applied": len(corrections),
        "coordinate_correction_lock_sha256": sha256_file(lock_path),
        "corrected_documents": sum(
            row.corrected_label_delta_lines > 0 for row in audits
        ),
        "corrected_present_lines": total_delta,
        "legacy_label_counts": {
            str(key): value for key, value in sorted(label_counts.items())
        },
        "corrected_label_counts": {
            str(key): value for key, value in sorted(corrected_label_counts.items())
        },
        "document_count": len(audits),
        "present_line_count": sum(row.n_present_lines for row in audits),
        "source_counts": dict(sorted(source_counts.items())),
        "historical_split_counts": dict(sorted(split_counts.items())),
        "whole_document_count": mode_counts.get("whole", 0),
        "windowed_document_count": len(audits) - mode_counts.get("whole", 0),
        "exact_text_groups": len(exact_hash_splits),
        "historical_cross_split_exact_text_groups": sum(
            len(value) > 1 for value in exact_hash_splits.values()
        ),
        "handoff": {
            "inventory_filename": lock["inventory"]["filename"],
            "inventory_sha256": lock["inventory"]["sha256"],
            "inventory_entry_count": len(inventory),
            "all_inventory_files_rehashed": verify_all_files,
            "source_state_sha256": inventory["SOURCE_STATE.txt"],
            "legacy_silver_sha256": inventory["STRUCT_2K_gold.jsonl"],
            "legacy_manifest_sha256": inventory["STRUCT_2K/manifest.jsonl"],
            "source_commit": lock["source"]["commit"],
        },
        "historical_partition": {
            "algorithm": expected.get("historical_split_algorithm"),
            "eligible_for_new_split": "train_only",
            "test_documents_available_to_model_processes": 0,
        },
        "research_fit_eligible": True,
        "research_evidence_scope": "LLM_silver_comparison_only",
        "production_eligible": False,
    }
    return summary, audits, inventory


def _token_counts(tokenizer: ExactTokenizer, texts: Sequence[str]) -> list[int]:
    result: list[int] = []
    for start in range(0, len(texts), 512):
        result.extend(tokenizer.counts(texts[start : start + 512]))
    _require(len(result) == len(texts), "tokenizer returned a different row count")
    _require(
        all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in result
        ),
        "tokenizer returned an invalid token count",
    )
    return result


def _write_materialized_silver(
    output: Path,
    *,
    handoff_root: Path,
    lock: Mapping[str, Any],
    inventory: Mapping[str, str],
    audits: Sequence[LegacyDocumentAudit],
    assignments: Mapping[str, str],
    tokenizer: ExactTokenizer,
    lock_sha256: str,
) -> str:
    audit_by_upstream = {row.upstream_document_id: row for row in audits}
    corrections = _correction_map(lock)
    consumed: set[tuple[str, str, str, tuple[int, int]]] = set()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial", dir=output.parent
    )
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for index, legacy in enumerate(
                _iter_jsonl(handoff_root / "STRUCT_2K_gold.jsonl")
            ):
                upstream_id = str(legacy["doc_id"])
                audit = audit_by_upstream[upstream_id]
                if audit.historical_split != "train":
                    continue
                annotation_relative = f"STRUCT_2K/ann_{index:05d}.json"
                annotation = _load_json(handoff_root / annotation_relative)
                sections = _sections_with_locked_corrections(
                    annotation, annotation_relative, corrections, consumed
                )
                labels = _labels_for_lines(legacy["lines"], sections)
                texts = [str(row[1]) for row in legacy["lines"]]
                token_counts = _token_counts(tokenizer, texts)
                lines = [
                    {
                        "line_id": _hash(
                            "struct2k-llm-silver-line-v1",
                            audit.document_id,
                            raw[0],
                            raw[1],
                        ),
                        "abs_idx": int(raw[0]),
                        "text": str(raw[1]),
                        "label": LEGACY_LABELS[label],
                        "token_count": int(token_count),
                        "is_running_prose": None,
                    }
                    for raw, label, token_count in zip(
                        legacy["lines"], labels, token_counts
                    )
                ]
                row = {
                    "schema_version": "academic-structure-gold-v1",
                    "document_id": audit.document_id,
                    "work_id": audit.work_id,
                    "representation_id": audit.representation_id,
                    "source": audit.source,
                    "split": assignments[audit.document_id],
                    "coverage": "full_document"
                    if audit.mode == "whole"
                    else "annotated_windows",
                    "n_physical_lines": audit.n_physical_lines,
                    "n_present_lines": audit.n_present_lines,
                    "annotation": {
                        "status": "LLM_silver",
                        "engine": "gpt-5.5 medium STRUCT-2K annotation workflow",
                        "task_scope": "bibliography_toc_windows",
                        "annotator_ids": ["LLM:gpt-5.5"],
                        "adjudicator_id": None,
                        "coordinate_correction_lock_sha256": lock_sha256,
                        "coordinate_correction_applied": audit.corrected_label_delta_lines
                        > 0,
                    },
                    "tokenizer": {
                        "id": "ModernGreek-148k",
                        "revision": TOKENIZER_REVISION,
                    },
                    "upstream_document_id": upstream_id,
                    "historical_split": audit.historical_split,
                    "historical_mode": audit.mode,
                    "observed_text_sha256": audit.observed_text_sha256,
                    "source_annotation_sha256": inventory[annotation_relative],
                    "source_batch_sha256": inventory[
                        f"STRUCT_2K/batch_{index:05d}.json"
                    ],
                    "lines": lines,
                }
                parse_gold_rows([row])
                payload = (
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                ).encode("utf-8")
                handle.write(payload)
                digest.update(payload)
            handle.flush()
            os.fsync(handle.fileno())
        eligible_upstream = {
            row.upstream_document_id
            for row in audits
            if row.historical_split == "train"
        }
        expected_consumed = {key for key in corrections if key[1] in eligible_upstream}
        _require(
            consumed == expected_consumed,
            "materialization did not consume every eligible coordinate correction",
        )
        os.link(temporary, output)
        os.unlink(temporary)
        return digest.hexdigest()
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def materialize(
    *,
    handoff_root: str | Path,
    lock_path: str | Path,
    tokenizer_path: str | Path,
    config_path: str | Path,
    silver_path: str | Path,
    split_manifest_path: str | Path,
    snapshot_receipt_path: str | Path,
    receipt_path: str | Path,
) -> dict[str, Any]:
    outputs = list(
        map(
            Path,
            (silver_path, split_manifest_path, snapshot_receipt_path, receipt_path),
        )
    )
    _require(
        len({path.resolve() for path in outputs}) == len(outputs),
        "output paths collide",
    )
    _require(
        all(not path.exists() and not path.is_symlink() for path in outputs),
        "output exists",
    )
    parent = outputs[0].resolve().parent
    _require(
        all(path.resolve().parent == parent for path in outputs),
        "outputs require one root",
    )
    lock = _load_json(lock_path)
    config = _load_json(config_path)
    _require(
        config.get("schema_version") == "academic-structure-sequence-eval-v1",
        "bad config",
    )
    _require(
        config.get("active_classes") == ["BIB", "TOC"],
        "joint config must activate BIB+TOC",
    )
    split_policy = config.get("split", {})
    _require(
        float(split_policy.get("test_fraction", -1)) == 0.0,
        "joint import cannot emit test",
    )
    snapshot, audits, inventory = audit_handoff(
        handoff_root, lock_path, verify_all_files=True
    )
    eligible = [row for row in audits if row.historical_split == "train"]
    identities = [
        Identity(row.document_id, row.work_id, row.source) for row in eligible
    ]
    split_manifest = build_split_manifest(identities, split_policy)
    assignments = split_manifest["assignments"]
    _require(
        set(assignments.values()) == {"train", "validation"},
        "joint split lacks train/validation",
    )
    _require(
        "test" not in assignments.values(),
        "historical test leaked into materialized split",
    )
    tokenizer = ExactTokenizer(Path(tokenizer_path))
    silver_sha = _write_materialized_silver(
        Path(silver_path),
        handoff_root=Path(handoff_root),
        lock=lock,
        inventory=inventory,
        audits=audits,
        assignments=assignments,
        tokenizer=tokenizer,
        lock_sha256=sha256_file(lock_path),
    )
    _atomic_json(split_manifest_path, split_manifest)
    documents = list(_iter_jsonl(silver_path))
    parsed = [parse_gold_rows([row])[0] for row in documents]
    contract = validate_silver(
        parsed, config["silver_contract"], split_manifest=split_manifest
    )
    snapshot.update(
        {
            "materialized_document_count": len(parsed),
            "materialized_split_counts": dict(
                sorted(collections.Counter(doc.split for doc in parsed).items())
            ),
            "materialized_inventory_sha256": contract["inventory_sha256"],
            "materialized_silver_sha256": silver_sha,
            "materialized_split_manifest_sha256": sha256_file(split_manifest_path),
            "config_sha256": sha256_file(config_path),
            "tokenizer": {
                "id": "ModernGreek-148k",
                "revision": TOKENIZER_REVISION,
                "sha256": sha256_file(tokenizer_path),
            },
        }
    )
    _atomic_json(snapshot_receipt_path, snapshot)
    source_receipt = {
        **contract,
        "silver_sha256": silver_sha,
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "config_sha256": sha256_file(config_path),
        "sequence_fit_eligible": True,
        "sequence_evidence_scope": "LLM_silver_comparison_only",
        "production_eligible": False,
        "materialized_artifacts": {
            "silver_filename": Path(silver_path).name,
            "split_manifest_filename": Path(split_manifest_path).name,
        },
        "source_unit_snapshot": {
            "receipt_path": Path(snapshot_receipt_path).name,
            "receipt_sha256": sha256_file(snapshot_receipt_path),
            "snapshot_schema_version": SNAPSHOT_SCHEMA,
            "snapshot_equivalence_status": (
                "verified_handoff_inventory_and_legacy_replay_with_locked_coordinate_corrections"
            ),
            "research_fit_eligible": True,
            "research_evidence_scope": "LLM_silver_comparison_only",
            "production_eligible": False,
        },
        "historical_partition_exclusion": {
            "status": "passed_before_materialization",
            "source_document_count": len(audits),
            "eligible_historical_train_documents": len(eligible),
            "historical_test_documents_excluded": sum(
                row.historical_split == "test" for row in audits
            ),
            "historical_test_rows_emitted": 0,
            "historical_test_predictions_permitted": False,
            "excluded_document_inventory_sha256": canonical_json_sha256(
                sorted(
                    (row.upstream_document_id, row.source)
                    for row in audits
                    if row.historical_split == "test"
                )
            ),
            "semantics": (
                "historical test was reused during old feature/model work and is sealed; "
                "new validation is derived only from the historical train partition"
            ),
        },
    }
    _atomic_json(receipt_path, source_receipt)
    return source_receipt


def verify_materialized_source(
    *,
    root: str | Path,
    lock_path: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    source_root = Path(root).resolve()
    _require(
        source_root.is_dir() and not source_root.is_symlink(), "source root is not real"
    )
    receipt_path = source_root / "struct2k.LLM_silver.receipt.json"
    _require(
        receipt_path.is_file() and not receipt_path.is_symlink(),
        "source receipt missing",
    )
    receipt = _load_json(receipt_path)
    lock = _load_json(lock_path)
    config = _load_json(config_path)
    _require(
        receipt.get("schema_version")
        == "academic-structure-silver-contract-receipt-v1",
        "bad receipt schema",
    )
    _require(
        receipt.get("status") == "pass", "materialized silver contract did not pass"
    )
    _require(
        receipt.get("production_eligible") is False,
        "LLM silver became production eligible",
    )
    _require(receipt.get("sequence_fit_eligible") is True, "source is not fit eligible")
    _require(
        receipt.get("config_sha256") == sha256_file(config_path), "config binding drift"
    )
    artifacts = receipt.get("materialized_artifacts")
    _require(
        isinstance(artifacts, Mapping), "receipt lacks materialized artifact names"
    )
    names = (
        artifacts.get("silver_filename"),
        artifacts.get("split_manifest_filename"),
    )
    _require(
        names == ("struct2k.LLM_silver.jsonl", "struct2k.LLM_silver.split.json"),
        "unexpected artifact names",
    )
    silver_path = source_root / str(names[0])
    split_path = source_root / str(names[1])
    for path in (silver_path, split_path):
        _require(
            path.is_file() and not path.is_symlink(),
            f"materialized artifact missing: {path.name}",
        )
    _require(
        receipt.get("silver_sha256") == sha256_file(silver_path), "silver hash drift"
    )
    _require(
        receipt.get("split_manifest_sha256") == sha256_file(split_path),
        "split hash drift",
    )
    snapshot_contract = receipt.get("source_unit_snapshot")
    _require(isinstance(snapshot_contract, Mapping), "source snapshot contract missing")
    snapshot_name = snapshot_contract.get("receipt_path")
    _require(
        snapshot_name == "struct2k.handoff.audit.receipt.json",
        "unexpected source snapshot receipt path",
    )
    snapshot_path = source_root / str(snapshot_name)
    _require(
        snapshot_path.is_file() and not snapshot_path.is_symlink(),
        "snapshot receipt missing",
    )
    _require(
        snapshot_contract.get("receipt_sha256") == sha256_file(snapshot_path),
        "snapshot receipt hash drift",
    )
    snapshot = _load_json(snapshot_path)
    legacy = lock.get("legacy_contract", {})
    _require(
        snapshot.get("schema_version") == SNAPSHOT_SCHEMA
        and snapshot.get("status")
        == "passed_inventory_and_legacy_replay_with_locked_coordinate_corrections"
        and snapshot.get("operation") == "audit_and_transcode_existing_joint_LLM_silver"
        and snapshot.get("annotation_status") == "LLM_silver"
        and snapshot.get("annotation_engine") == legacy.get("annotation_engine")
        and snapshot.get("human_gold") is False
        and snapshot.get("new_semantic_annotations_created") is False
        and snapshot.get("coordinate_corrections_applied")
        == len(lock.get("coordinate_corrections", []))
        and snapshot.get("document_count") == legacy.get("document_count")
        and snapshot.get("present_line_count") == legacy.get("present_line_count")
        and snapshot.get("source_counts") == legacy.get("source_counts")
        and snapshot.get("historical_split_counts")
        == legacy.get("historical_split_counts")
        and snapshot.get("research_fit_eligible") is True
        and snapshot.get("research_evidence_scope") == "LLM_silver_comparison_only"
        and snapshot.get("production_eligible") is False,
        "snapshot replay semantics drift",
    )
    _require(
        snapshot.get("coordinate_correction_lock_sha256") == sha256_file(lock_path),
        "coordinate correction lock drift",
    )
    handoff = snapshot.get("handoff", {})
    _require(
        handoff.get("inventory_sha256") == lock["inventory"]["sha256"],
        "handoff inventory drift",
    )
    _require(
        handoff.get("source_commit") == lock["source"]["commit"], "source commit drift"
    )
    _require(
        handoff.get("legacy_silver_sha256")
        == lock["inventory"]["required_files"]["STRUCT_2K_gold.jsonl"],
        "legacy silver binding drift",
    )
    documents = read_gold(silver_path)
    _require(
        all(
            document.task_scope == "bibliography_toc_windows" for document in documents
        ),
        "non-joint row",
    )
    _require(
        all(document.annotation_status == "LLM_silver" for document in documents),
        "non-silver row",
    )
    _require(
        all(document.split in {"train", "validation"} for document in documents),
        "test row leaked",
    )
    expected_eligible = int(
        config.get("historical_partition_usage", {}).get(
            "historical_train_document_count", len(documents)
        )
    )
    expected_test = int(
        config.get("historical_partition_usage", {}).get(
            "historical_test_document_count", -1
        )
    )
    _require(
        len(documents) == expected_eligible, "eligible historical-train count drift"
    )
    manifest = _load_json(split_path)
    recomputed_manifest = build_split_manifest(documents, config["split"])
    _require(manifest == recomputed_manifest, "materialized split does not recompute")
    _require(
        split_path.read_bytes() == _canonical_bytes(recomputed_manifest),
        "split JSON is not canonical",
    )
    recomputed_contract = validate_silver(
        documents, config["silver_contract"], split_manifest=manifest
    )
    for key, value in recomputed_contract.items():
        _require(receipt.get(key) == value, f"source contract field drift: {key}")
    _require(
        snapshot.get("materialized_document_count") == len(documents)
        and snapshot.get("materialized_split_counts")
        == dict(sorted(collections.Counter(doc.split for doc in documents).items()))
        and snapshot.get("materialized_inventory_sha256")
        == recomputed_contract["inventory_sha256"]
        and snapshot.get("materialized_silver_sha256") == sha256_file(silver_path)
        and snapshot.get("materialized_split_manifest_sha256")
        == sha256_file(split_path)
        and snapshot.get("config_sha256") == sha256_file(config_path)
        and snapshot.get("tokenizer")
        == {
            "id": "ModernGreek-148k",
            "revision": TOKENIZER_REVISION,
            "sha256": TOKENIZER_SHA256,
        },
        "snapshot materialization binding drift",
    )
    exclusion = receipt.get("historical_partition_exclusion")
    _require(
        isinstance(exclusion, Mapping)
        and exclusion.get("status") == "passed_before_materialization"
        and exclusion.get("source_document_count") == expected_eligible + expected_test
        and exclusion.get("eligible_historical_train_documents") == expected_eligible
        and exclusion.get("historical_test_documents_excluded") == expected_test
        and exclusion.get("historical_test_rows_emitted") == 0
        and exclusion.get("historical_test_predictions_permitted") is False,
        "historical partition exclusion receipt drift",
    )
    expected_files = {
        receipt_path.resolve(),
        silver_path.resolve(),
        split_path.resolve(),
        snapshot_path.resolve(),
    }
    discovered: set[Path] = set()
    for path in source_root.rglob("*"):
        _require(not path.is_symlink(), f"materialized source contains symlink: {path}")
        if path.is_file():
            discovered.add(path.resolve())
    _require(
        discovered == expected_files, "materialized source contains unreceipted files"
    )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--handoff-root", required=True)
    audit.add_argument("--lock", required=True)
    audit.add_argument("--receipt", required=True)
    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--handoff-root", required=True)
    materialize_parser.add_argument("--lock", required=True)
    materialize_parser.add_argument("--tokenizer-json", required=True)
    materialize_parser.add_argument("--config", required=True)
    materialize_parser.add_argument("--silver", required=True)
    materialize_parser.add_argument("--split-manifest", required=True)
    materialize_parser.add_argument("--snapshot-receipt", required=True)
    materialize_parser.add_argument("--receipt", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--root", required=True)
    verify.add_argument("--lock", required=True)
    verify.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    if args.command == "audit":
        snapshot, _audits, _inventory = audit_handoff(
            args.handoff_root, args.lock, verify_all_files=True
        )
        _atomic_json(args.receipt, snapshot)
        return 0
    if args.command == "verify":
        verify_materialized_source(
            root=args.root, lock_path=args.lock, config_path=args.config
        )
        return 0
    materialize(
        handoff_root=args.handoff_root,
        lock_path=args.lock,
        tokenizer_path=args.tokenizer_json,
        config_path=args.config,
        silver_path=args.silver,
        split_manifest_path=args.split_manifest,
        snapshot_receipt_path=args.snapshot_receipt,
        receipt_path=args.receipt,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
