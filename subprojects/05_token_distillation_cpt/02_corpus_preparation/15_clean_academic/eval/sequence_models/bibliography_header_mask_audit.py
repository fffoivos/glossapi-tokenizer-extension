#!/usr/bin/env python3
"""Build a contextual audit packet for bibliography header masking.

The deterministic rules nominate possible headings; they never decide the
entry-training mask.  The output preserves exact silver labels and enough
context for an independent reviewer to distinguish entries, headings,
subheadings, other structure, and uncertain cases.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .bibliography_block_audit import audit_document
from .bibliography_feature_explorer import FEATURE_SPECS
from .bibliography_v2 import extract_bibliography_feature_review
from .deterministic_structure import BibRole, analyze_bib_line


SCHEMA_VERSION = "bibliography-header-mask-audit-v1"
STRATA = (
    "exact_heading",
    "exact_subheading",
    "block_start_probe",
    "internal_sparse_probe",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"JSONL row {row_number} is not an object")
            yield row


def _candidate_id(document_id: str, block_index: int, abs_idx: int) -> str:
    value = f"{document_id}\0{block_index}\0{abs_idx}".encode()
    return hashlib.sha256(value).hexdigest()


def _line_fields(raw: Mapping[str, Any], document_id: str) -> tuple[int, str, str]:
    abs_idx, text, label = raw.get("abs_idx"), raw.get("text"), raw.get("label")
    if not isinstance(abs_idx, int) or not isinstance(text, str):
        raise ValueError(f"{document_id}: malformed materialized line")
    if label not in {"O", "BIB", "TOC"}:
        raise ValueError(f"{document_id}: invalid label {label!r} at {abs_idx}")
    return abs_idx, text, str(label)


def _review_features(text: str) -> tuple[int, dict[str, int]]:
    review = extract_bibliography_feature_review(text)
    values = review.features.as_dict()
    nonzero = {
        spec.key: int(values[spec.key])
        for spec in FEATURE_SPECS
        if int(values[spec.key]) > 0
    }
    return len(nonzero), nonzero


def _context(
    lines: Sequence[tuple[int, str, str]], position: int, radius: int
) -> list[dict[str, Any]]:
    start, end = max(0, position - radius), min(len(lines), position + radius + 1)
    return [
        {
            "abs_idx": abs_idx,
            "label": label,
            "relative": offset - position,
            "text": text,
        }
        for offset, (abs_idx, text, label) in enumerate(lines[start:end], start)
    ]


def audit_document_headers(
    row: Mapping[str, Any], *, context_radius: int = 3, max_physical_gap: int = 64
) -> dict[str, Any]:
    audit = audit_document(row, max_physical_gap=max_physical_gap)
    document_id = str(audit["document_id"])
    raw_lines = row.get("lines")
    if not isinstance(raw_lines, list):
        raise ValueError(f"{document_id}: missing materialized lines")
    lines = [_line_fields(raw, document_id) for raw in raw_lines]
    positions = {abs_idx: position for position, (abs_idx, _, _) in enumerate(lines)}
    candidates: list[dict[str, Any]] = []
    bib_lines = 0

    for block in audit["blocks"]:
        start_idx, end_idx = int(block["start_abs_idx"]), int(block["end_abs_idx"])
        members = [
            (position, abs_idx, text)
            for position, (abs_idx, text, label) in enumerate(lines)
            if label == "BIB" and start_idx <= abs_idx <= end_idx
        ]
        bib_lines += len(members)
        for member_offset, (position, abs_idx, text) in enumerate(members):
            evidence = analyze_bib_line(text, abs_idx)
            feature_points, nonzero = _review_features(text)
            stratum: str | None = None
            reasons: list[str] = []
            if evidence.role == BibRole.HEADING:
                stratum = "exact_heading"
                reasons.append("deterministic_exact_heading")
            elif evidence.role == BibRole.SUBHEADING:
                stratum = "exact_subheading"
                reasons.append("deterministic_exact_subheading")
            elif member_offset < 2:
                stratum = "block_start_probe"
                reasons.append(f"silver_block_member_offset_{member_offset}")
            elif (
                member_offset < len(members) - 2
                and len(text.strip()) <= 160
                and evidence.token_count <= 12
                and feature_points <= 2
            ):
                stratum = "internal_sparse_probe"
                reasons.extend(
                    [
                        "internal_short_line",
                        "at_most_12_tokens",
                        "at_most_2_deterministic_features",
                    ]
                )
            if stratum is None:
                continue
            candidates.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "candidate_id": _candidate_id(
                        document_id, int(block["block_index"]), abs_idx
                    ),
                    "document_id": document_id,
                    "work_id": str(audit["work_id"]),
                    "source": str(audit["source"]),
                    "split": str(audit["split"]),
                    "coverage": str(audit["coverage"]),
                    "block_index": int(block["block_index"]),
                    "block_start_abs_idx": start_idx,
                    "block_end_abs_idx": end_idx,
                    "block_present_line_count": int(block["present_line_count"]),
                    "block_member_offset": member_offset,
                    "stratum": stratum,
                    "nomination_reasons": reasons,
                    "abs_idx": abs_idx,
                    "text": text,
                    "char_length": len(text),
                    "token_count": int(evidence.token_count),
                    "deterministic_role": evidence.role.value,
                    "deterministic_reason_codes": list(evidence.reason_codes),
                    "feature_points": feature_points,
                    "nonzero_features": nonzero,
                    "silver_label": "BIB",
                    "context": _context(lines, positions[abs_idx], context_radius),
                }
            )

    return {
        "document_id": document_id,
        "source": str(audit["source"]),
        "split": str(audit["split"]),
        "bib_block_count": int(audit["bib_block_count"]),
        "bib_line_count": bib_lines,
        "candidates": candidates,
    }


def _hash_order(candidate: Mapping[str, Any], seed: str) -> str:
    return hashlib.sha256(
        f"{seed}\0{candidate['candidate_id']}".encode()
    ).hexdigest()


def sample_candidates(
    candidates: Sequence[Mapping[str, Any]], *, per_stratum: int, seed: str
) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    for stratum in STRATA:
        pool = [row for row in candidates if row["stratum"] == stratum]
        chosen: list[Mapping[str, Any]] = []
        for source in sorted({str(row["source"]) for row in pool}):
            source_pool = sorted(
                (row for row in pool if row["source"] == source),
                key=lambda row: _hash_order(row, seed),
            )
            quota = per_stratum // 3 + int(
                source in sorted({str(row["source"]) for row in pool})[
                    : per_stratum % 3
                ]
            )
            chosen.extend(source_pool[:quota])
        if len(chosen) < per_stratum:
            selected_ids = {row["candidate_id"] for row in chosen}
            remainder = sorted(
                (row for row in pool if row["candidate_id"] not in selected_ids),
                key=lambda row: _hash_order(row, seed),
            )
            chosen.extend(remainder[: per_stratum - len(chosen)])
        selected.extend(chosen[:per_stratum])
    return sorted(selected, key=lambda row: (STRATA.index(row["stratum"]), row["source"], row["candidate_id"]))


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path, output_dir = Path(args.input).resolve(), Path(args.output_dir).resolve()
    if not input_path.is_file() or input_path.is_symlink():
        raise ValueError(f"input must be a regular file: {input_path}")
    if output_dir.exists():
        raise ValueError(f"output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    documents = [row for row in _iter_jsonl(input_path) if row.get("split") == args.split]
    worker = max(1, int(args.workers))
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker) as executor:
        results = list(
            executor.map(
                _audit_worker,
                (
                    (row, int(args.context_radius), int(args.max_physical_gap))
                    for row in documents
                ),
                chunksize=4,
            )
        )
    candidates = sorted(
        (candidate for result in results for candidate in result["candidates"]),
        key=lambda row: (row["document_id"], row["block_index"], row["abs_idx"]),
    )
    sample = sample_candidates(
        candidates, per_stratum=int(args.per_stratum), seed=str(args.seed)
    )
    candidate_path = output_dir / "candidates.jsonl"
    with candidate_path.open("x", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate, ensure_ascii=False, sort_keys=True) + "\n")

    sample_packet = {
        "schema_version": SCHEMA_VERSION,
        "task": "contextually distinguish bibliography entries from headers and other structural lines",
        "allowed_labels": [
            "ENTRY",
            "BIB_HEADER",
            "BIB_SUBHEADER",
            "OTHER_STRUCTURE",
            "UNCERTAIN",
        ],
        "decision_warning": "nomination is not a mask decision; silver labels remain unchanged",
        "cases": sample,
    }
    sample_path = output_dir / "audit_sample.json"
    _write_json(sample_path, sample_packet)

    counts = collections.Counter(row["stratum"] for row in candidates)
    sample_counts = collections.Counter(row["stratum"] for row in sample)
    source_counts = collections.Counter(row["source"] for row in candidates)
    sample_source_counts = collections.Counter(row["source"] for row in sample)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "split": str(args.split),
        "document_count": len(results),
        "bib_block_count": sum(result["bib_block_count"] for result in results),
        "bib_line_count": sum(result["bib_line_count"] for result in results),
        "candidate_count": len(candidates),
        "candidate_counts_by_stratum": dict(sorted(counts.items())),
        "candidate_counts_by_source": dict(sorted(source_counts.items())),
        "sample_count": len(sample),
        "sample_counts_by_stratum": dict(sorted(sample_counts.items())),
        "sample_counts_by_source": dict(sorted(sample_source_counts.items())),
        "sampling_seed": str(args.seed),
        "context_radius": int(args.context_radius),
        "max_physical_gap": int(args.max_physical_gap),
        "input": {"path": str(input_path), "sha256": _sha256_file(input_path)},
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    receipt = {
        "schema_version": "bibliography-header-mask-audit-receipt-v1",
        "status": "passed_candidate_generation_only_no_mask_decision",
        "code_commit": str(args.code_commit),
        "slurm_job_id": str(args.slurm_job_id),
        **summary,
        "outputs": {
            path.name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
            for path in (candidate_path, sample_path, summary_path)
        },
    }
    _write_json(output_dir / "receipt.json", receipt)
    return receipt


def _audit_worker(args: tuple[Mapping[str, Any], int, int]) -> dict[str, Any]:
    row, context_radius, max_physical_gap = args
    return audit_document_headers(
        row, context_radius=context_radius, max_physical_gap=max_physical_gap
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train", choices=("train", "validation"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--context-radius", type=int, default=3)
    parser.add_argument("--max-physical-gap", type=int, default=64)
    parser.add_argument("--per-stratum", type=int, default=30)
    parser.add_argument("--seed", default="bib-header-mask-audit-v1")
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
