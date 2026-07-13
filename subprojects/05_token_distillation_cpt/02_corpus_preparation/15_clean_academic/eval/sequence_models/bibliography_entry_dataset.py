#!/usr/bin/env python3
"""Materialize bibliography-entry line features with a conservative header mask.

The source ``BIB`` label describes a region.  This module derives an entry-line
training view without modifying that source label:

* exact bibliography headings/subheadings inside silver BIB regions are MASK;
* all other BIB lines are ENTRY positives;
* O and TOC are negatives; and
* UNKNOWN is MASK.

The 35 count features are stored losslessly.  Binary and log transforms belong
to model arms and are not baked into this shared table.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .bibliography_feature_explorer import FEATURE_SPECS
from .bibliography_v2 import extract_bibliography_features
from .deterministic_structure import BibRole, analyze_bib_line


SCHEMA_VERSION = "bibliography-entry-feature-table-v1"
FEATURE_NAMES = tuple(spec.key for spec in FEATURE_SPECS)
LABEL_TO_ID = {"O": 0, "BIB": 1, "TOC": 2, "UNKNOWN": 3}
TARGET_MASK = -1
TARGET_NEGATIVE = 0
TARGET_ENTRY = 1
HEADER_NONE = 0
HEADER_EXACT = 1
SUBHEADER_EXACT = 2
MAX_PHYSICAL_GAP = 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_rank(*values: str) -> int:
    payload = "\0".join(values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def _iter_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"row {row_number} is not an object")
            yield row


def _line_block_indices(lines: Sequence[Mapping[str, Any]]) -> np.ndarray:
    result = np.full(len(lines), -1, dtype=np.int32)
    active = -1
    previous_abs: int | None = None
    previous_bib = False
    for offset, line in enumerate(lines):
        abs_idx, label = line.get("abs_idx"), line.get("label")
        if not isinstance(abs_idx, int) or abs_idx < 0:
            raise ValueError(f"invalid abs_idx at line {offset}")
        if previous_abs is not None and abs_idx <= previous_abs:
            raise ValueError("line coordinates are not strictly increasing")
        is_bib = label == "BIB"
        if is_bib and (
            not previous_bib
            or previous_abs is None
            or abs_idx - previous_abs > MAX_PHYSICAL_GAP
        ):
            active += 1
        if is_bib:
            result[offset] = active
        previous_abs, previous_bib = abs_idx, is_bib
    return result


def materialize_document(row: Mapping[str, Any]) -> dict[str, Any]:
    document_id = str(row.get("document_id", ""))
    work_id = str(row.get("work_id", ""))
    if not document_id or not work_id:
        raise ValueError("document_id and work_id are required")
    raw_lines = row.get("lines")
    if not isinstance(raw_lines, list) or not raw_lines:
        raise ValueError(f"{document_id}: non-empty lines are required")

    n_lines = len(raw_lines)
    counts = np.zeros((n_lines, len(FEATURE_NAMES)), dtype=np.uint32)
    targets = np.empty(n_lines, dtype=np.int8)
    original_labels = np.empty(n_lines, dtype=np.uint8)
    header_kinds = np.zeros(n_lines, dtype=np.uint8)
    abs_indices = np.empty(n_lines, dtype=np.uint32)
    token_counts = np.empty(n_lines, dtype=np.uint32)
    char_lengths = np.empty(n_lines, dtype=np.uint32)
    block_indices = _line_block_indices(raw_lines)

    for offset, line in enumerate(raw_lines):
        text, label, abs_idx = line.get("text"), line.get("label"), line.get("abs_idx")
        if not isinstance(text, str) or label not in LABEL_TO_ID:
            raise ValueError(f"{document_id}: malformed line {offset}")
        feature_values = extract_bibliography_features(text).as_dict()
        row_counts = [int(feature_values[name]) for name in FEATURE_NAMES]
        if any(value < 0 or value > np.iinfo(np.uint32).max for value in row_counts):
            raise ValueError(f"{document_id}: feature count overflow at line {offset}")
        counts[offset] = row_counts
        original_labels[offset] = LABEL_TO_ID[str(label)]
        abs_indices[offset] = int(abs_idx)
        token_count = line.get("token_count", feature_values["token_count"])
        if not isinstance(token_count, int) or token_count < 0:
            raise ValueError(f"{document_id}: invalid token_count at line {offset}")
        token_counts[offset] = token_count
        char_lengths[offset] = len(text)

        evidence = analyze_bib_line(text, int(abs_idx))
        if evidence.role == BibRole.HEADING:
            header_kinds[offset] = HEADER_EXACT
        elif evidence.role == BibRole.SUBHEADING:
            header_kinds[offset] = SUBHEADER_EXACT

        if label == "UNKNOWN":
            targets[offset] = TARGET_MASK
        elif label != "BIB":
            targets[offset] = TARGET_NEGATIVE
        else:
            if evidence.role == BibRole.HEADING:
                targets[offset] = TARGET_MASK
            elif evidence.role == BibRole.SUBHEADING:
                targets[offset] = TARGET_MASK
            else:
                targets[offset] = TARGET_ENTRY

    return {
        "document_id": document_id,
        "work_id": work_id,
        "source": str(row.get("source", "")),
        "split": str(row.get("split", "")),
        "coverage": str(row.get("coverage", "")),
        "n_physical_lines": int(row.get("n_physical_lines", 0)),
        "counts": counts,
        "targets": targets,
        "original_labels": original_labels,
        "header_kinds": header_kinds,
        "abs_indices": abs_indices,
        "token_counts": token_counts,
        "char_lengths": char_lengths,
        "block_indices": block_indices,
    }


def assign_grouped_folds(
    documents: Sequence[Mapping[str, Any]], *, n_folds: int, seed: str
) -> tuple[list[int], dict[str, int]]:
    """Greedily balance work groups within source/block-count strata."""

    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    groups: dict[str, dict[str, Any]] = {}
    for doc_index, document in enumerate(documents):
        work_id = str(document["work_id"])
        group = groups.setdefault(
            work_id,
            {
                "document_indices": [],
                "sources": set(),
                "line_count": 0,
                "block_count": 0,
            },
        )
        group["document_indices"].append(doc_index)
        group["sources"].add(str(document["source"]))
        group["line_count"] += int(document["line_count"])
        group["block_count"] += int(document["block_count"])

    strata: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = collections.defaultdict(list)
    for work_id, group in groups.items():
        source = next(iter(group["sources"])) if len(group["sources"]) == 1 else "mixed"
        block_bin = "0" if group["block_count"] == 0 else "1" if group["block_count"] == 1 else "2+"
        strata[(source, block_bin)].append((work_id, group))

    fold_lines = [0] * n_folds
    fold_groups = [0] * n_folds
    assignment: dict[str, int] = {}
    for stratum in sorted(strata):
        local_lines = [0] * n_folds
        local_groups = [0] * n_folds
        ordered = sorted(
            strata[stratum],
            key=lambda item: (
                -int(item[1]["line_count"]),
                _stable_rank(seed, stratum[0], stratum[1], item[0]),
            ),
        )
        for work_id, group in ordered:
            fold = min(
                range(n_folds),
                key=lambda index: (
                    local_groups[index],
                    local_lines[index],
                    fold_groups[index],
                    fold_lines[index],
                    index,
                ),
            )
            assignment[work_id] = fold
            local_groups[fold] += 1
            local_lines[fold] += int(group["line_count"])
            fold_groups[fold] += 1
            fold_lines[fold] += int(group["line_count"])

    document_folds = [assignment[str(document["work_id"])] for document in documents]
    return document_folds, assignment


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _save_array(path: Path, value: np.ndarray) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not input_path.is_file() or input_path.is_symlink():
        raise ValueError(f"input must be a regular file: {input_path}")
    input_sha256 = sha256_file(input_path)
    if args.expected_input_sha256 and input_sha256 != args.expected_input_sha256:
        raise ValueError("input SHA-256 does not match the pinned value")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"immutable output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    rows = [row for row in _iter_rows(input_path) if row.get("split") == args.split]
    if not rows:
        raise ValueError(f"no {args.split!r} documents found")
    workers = max(1, int(args.workers))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        materialized = list(executor.map(materialize_document, rows, chunksize=1))

    document_rows: list[dict[str, Any]] = []
    cursor = 0
    for result in materialized:
        length = len(result["targets"])
        block_count = int(result["block_indices"].max()) + 1 if length and result["block_indices"].max() >= 0 else 0
        document_rows.append(
            {
                "document_id": result["document_id"],
                "work_id": result["work_id"],
                "source": result["source"],
                "split": result["split"],
                "coverage": result["coverage"],
                "n_physical_lines": result["n_physical_lines"],
                "line_start": cursor,
                "line_end": cursor + length,
                "line_count": length,
                "block_count": block_count,
            }
        )
        cursor += length
    document_folds, work_folds = assign_grouped_folds(
        document_rows, n_folds=int(args.n_folds), seed=str(args.fold_seed)
    )
    for row, fold in zip(document_rows, document_folds, strict=True):
        row["fold"] = fold

    arrays = {
        "counts": np.concatenate([row["counts"] for row in materialized]),
        "targets": np.concatenate([row["targets"] for row in materialized]),
        "original_labels": np.concatenate([row["original_labels"] for row in materialized]),
        "header_kinds": np.concatenate([row["header_kinds"] for row in materialized]),
        "abs_indices": np.concatenate([row["abs_indices"] for row in materialized]),
        "token_counts": np.concatenate([row["token_counts"] for row in materialized]),
        "char_lengths": np.concatenate([row["char_lengths"] for row in materialized]),
        "block_indices": np.concatenate([row["block_indices"] for row in materialized]),
        "document_indices": np.concatenate(
            [np.full(row["line_count"], index, dtype=np.uint32) for index, row in enumerate(document_rows)]
        ),
        "folds": np.concatenate(
            [np.full(row["line_count"], row["fold"], dtype=np.uint8) for row in document_rows]
        ),
    }
    for name, value in arrays.items():
        _save_array(output_dir / f"{name}.npy", value)
    _write_jsonl(output_dir / "documents.jsonl", document_rows)
    _write_json(
        output_dir / "folds.json",
        {
            "schema_version": "bibliography-entry-grouped-folds-v1",
            "seed": str(args.fold_seed),
            "n_folds": int(args.n_folds),
            "group_key": "work_id",
            "stratification": ["source", "bib_block_count=0|1|2+"],
            "work_assignments": dict(sorted(work_folds.items())),
        },
    )
    target_counts = collections.Counter(int(value) for value in arrays["targets"])
    header_counts = collections.Counter(int(value) for value in arrays["header_kinds"])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_exact_header_mask_only",
        "split": str(args.split),
        "input": {"path": str(input_path), "sha256": input_sha256},
        "code_commit": str(args.code_commit),
        "slurm_job_id": str(args.slurm_job_id),
        "document_count": len(document_rows),
        "work_count": len(work_folds),
        "line_count": cursor,
        "feature_names": list(FEATURE_NAMES),
        "feature_count": len(FEATURE_NAMES),
        "feature_dtype": "uint32",
        "target_encoding": {"MASK": TARGET_MASK, "NEGATIVE": TARGET_NEGATIVE, "ENTRY": TARGET_ENTRY},
        "target_counts": {str(key): value for key, value in sorted(target_counts.items())},
        "header_encoding": {"NONE": HEADER_NONE, "EXACT_HEADING": HEADER_EXACT, "EXACT_SUBHEADING": SUBHEADER_EXACT},
        "header_candidate_counts_all_labels": {str(key): value for key, value in sorted(header_counts.items())},
        "mask_policy": "exact deterministic heading/subheading inside silver BIB only",
        "n_folds": int(args.n_folds),
        "fold_seed": str(args.fold_seed),
    }
    _write_json(output_dir / "manifest.json", manifest)
    outputs = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            outputs[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    receipt = {**manifest, "outputs": outputs}
    _write_json(output_dir / "receipt.json", receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train", choices=("train", "validation"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold-seed", default="bibliography-entry-folds-v1")
    parser.add_argument("--expected-input-sha256")
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
