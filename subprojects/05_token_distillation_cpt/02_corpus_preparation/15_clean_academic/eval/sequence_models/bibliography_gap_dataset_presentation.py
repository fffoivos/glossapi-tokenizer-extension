#!/usr/bin/env python3
"""Build a source-balanced presentation packet for the selected gap dataset.

The packet contains a deterministic sample of the exact training rows selected
for the winning ``threshold_ladder`` / 1,000-negative-boundary configuration.
It restores source text only for presentation: model endpoint lines are marked
as excluded anchors and the strictly interior gap is marked as the training
span.  The command is create-only and never opens validation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .bibliography_gap_candidate_screen import CandidateTable
from .bibliography_gap_connect_table import gap_length_bucket
from .bibliography_gap_sampling import select_training_rows
from .contract import sha256_file


SCHEMA_VERSION = "bibliography-gap-training-presentation-v1"
DEFAULT_REGIME = "threshold_ladder"
DEFAULT_NEGATIVE_LIMIT = 1000
DEFAULT_SEED = 20260717
DEFAULT_PER_SOURCE_LABEL = 30


def _stable_rank(seed: int, *values: object) -> str:
    payload = ":".join((str(seed), *(str(value) for value in values)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _iter_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"source row {row_number} is not an object")
            yield row


def select_presentation_rows(
    metadata: Sequence[Mapping[str, Any]],
    targets: np.ndarray,
    training_rows: np.ndarray,
    *,
    per_source_label: int,
    seed: int,
) -> np.ndarray:
    """Select a deterministic source/label sample spread over folds and lengths."""

    if per_source_label < 1:
        raise ValueError("per_source_label must be positive")
    if len(metadata) != len(targets):
        raise ValueError("metadata and targets are not aligned")
    grouped: defaultdict[tuple[str, int], defaultdict[tuple[int, str], list[int]]]
    grouped = defaultdict(lambda: defaultdict(list))
    for raw_index in training_rows:
        index = int(raw_index)
        row = metadata[index]
        grouped[(str(row["source"]), int(targets[index]))][
            (int(row["fold"]), gap_length_bucket(int(row["model_line_count"])))
        ].append(index)

    selected: list[int] = []
    for source_label, strata in sorted(grouped.items()):
        queues = {
            key: deque(sorted(
                indices,
                key=lambda index: _stable_rank(
                    seed,
                    source_label,
                    key,
                    metadata[index]["variant_id"],
                ),
            ))
            for key, indices in strata.items()
        }
        stratum_order = sorted(
            queues,
            key=lambda key: _stable_rank(seed, source_label, key),
        )
        local: list[int] = []
        while len(local) < per_source_label and any(queues[key] for key in stratum_order):
            for key in stratum_order:
                if queues[key] and len(local) < per_source_label:
                    local.append(queues[key].popleft())
        selected.extend(local)
    return np.asarray(selected, dtype=np.int64)


def _display_indices(
    left: int,
    right: int,
    *,
    line_count: int,
    context: int,
    maximum_gap: int,
) -> list[int | None]:
    if not 0 <= left < right < line_count:
        raise ValueError("invalid candidate endpoints")
    before = list(range(max(0, left - context), left + 1))
    interior = list(range(left + 1, right))
    if len(interior) > maximum_gap:
        half = maximum_gap // 2
        interior_display: list[int | None] = [
            *interior[:half],
            None,
            *interior[-(maximum_gap - half):],
        ]
    else:
        interior_display = interior
    after = list(range(right, min(line_count, right + context + 1)))
    return [*before, *interior_display, *after]


def _line_role(index: int, left: int, right: int) -> str:
    if index == left:
        return "left_anchor"
    if index == right:
        return "right_anchor"
    if left < index < right:
        return "training_span"
    return "context"


def _case(
    index: int,
    row: Mapping[str, Any],
    target: int,
    source_row: Mapping[str, Any],
    *,
    context: int,
    maximum_gap: int,
) -> dict[str, Any]:
    raw_lines = source_row.get("lines")
    if not isinstance(raw_lines, list) or not raw_lines:
        raise ValueError(f"{row['document_id']}: source lines are missing")
    left, right = int(row["left_local_index"]), int(row["right_local_index"])
    if not 0 <= left < right < len(raw_lines):
        raise ValueError(f"{row['document_id']}: candidate endpoints are out of range")
    if int(raw_lines[left]["abs_idx"]) != int(row["left_abs_idx"]):
        raise ValueError(f"{row['document_id']}: left endpoint provenance mismatch")
    if int(raw_lines[right]["abs_idx"]) != int(row["right_abs_idx"]):
        raise ValueError(f"{row['document_id']}: right endpoint provenance mismatch")

    display = []
    for local_index in _display_indices(
        left,
        right,
        line_count=len(raw_lines),
        context=context,
        maximum_gap=maximum_gap,
    ):
        if local_index is None:
            display.append({
                "role": "omission",
                "omitted_line_count": max(0, right - left - 1 - maximum_gap),
            })
            continue
        line = raw_lines[local_index]
        display.append({
            "local_index": local_index,
            "abs_idx": int(line["abs_idx"]),
            "text": str(line["text"]),
            "silver_label": str(line["label"]),
            "role": _line_role(local_index, left, right),
        })

    full_gap = raw_lines[left + 1:right]
    labels = Counter(str(line["label"]) for line in full_gap)
    return {
        "row_index": index,
        "variant_id": str(row["variant_id"]),
        "boundary_group_id": str(row["boundary_group_id"]),
        "document_id": str(row["document_id"]),
        "work_id": str(row["work_id"]),
        "source": str(row["source"]),
        "fold": int(row["fold"]),
        "target_connect": int(target),
        "training_label": "BIB / CONNECT" if target else "NOT BIB / BREAK",
        "regime": str(row["regime"]),
        "generation_thresholds": [float(value) for value in row["generation_thresholds"]],
        "gap_line_count": int(row["model_line_count"]),
        "gap_length_bucket": gap_length_bucket(int(row["model_line_count"])),
        "gap_silver_labels": dict(sorted(labels.items())),
        "entry_mean": float(row["entry_mean"]),
        "entry_max": float(row["entry_max"]),
        "lines": display,
    }


def _summary(
    metadata: Sequence[Mapping[str, Any]], targets: np.ndarray, rows: np.ndarray
) -> dict[str, Any]:
    sources = Counter()
    folds = Counter()
    lengths = Counter()
    regimes = Counter()
    works = set()
    documents = set()
    for raw_index in rows:
        index = int(raw_index)
        row = metadata[index]
        label = "connect" if int(targets[index]) else "break"
        sources[f"{row['source']}:{label}"] += 1
        folds[f"fold_{int(row['fold'])}:{label}"] += 1
        lengths[f"{gap_length_bucket(int(row['model_line_count']))}:{label}"] += 1
        regimes[f"{row['regime']}:{label}"] += 1
        works.add(str(row["work_id"]))
        documents.add(str(row["document_id"]))
    return {
        "training_example_count": len(rows),
        "connect_count": int(np.count_nonzero(targets[rows] == 1)),
        "break_count": int(np.count_nonzero(targets[rows] == 0)),
        "work_count": len(works),
        "document_count": len(documents),
        "by_source_label": dict(sorted(sources.items())),
        "by_fold_label": dict(sorted(folds.items())),
        "by_gap_length_label": dict(sorted(lengths.items())),
        "by_regime_label": dict(sorted(regimes.items())),
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    table = CandidateTable(Path(args.table_dir))
    training_rows = select_training_rows(
        table.metadata,
        table.targets,
        regime=args.regime,
        negative_group_limit=args.negative_limit,
        seed=args.seed,
    )
    sample_rows = select_presentation_rows(
        table.metadata,
        table.targets,
        training_rows,
        per_source_label=args.per_source_label,
        seed=args.seed,
    )
    selected_documents = {
        str(table.metadata[int(index)]["document_id"]) for index in sample_rows
    }
    source_rows = {
        str(row.get("document_id")): row
        for row in _iter_jsonl(Path(args.source_jsonl))
        if str(row.get("document_id")) in selected_documents
    }
    missing = selected_documents - set(source_rows)
    if missing:
        raise ValueError(f"source text missing for {len(missing)} selected documents")

    cases = [
        _case(
            int(index),
            table.metadata[int(index)],
            int(table.targets[int(index)]),
            source_rows[str(table.metadata[int(index)]["document_id"])],
            context=args.context_lines,
            maximum_gap=args.maximum_display_gap_lines,
        )
        for index in sample_rows
    ]
    packet = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_selected_training_dataset_presentation",
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "validation_opened": False,
        "label_tier": "LLM_silver_region",
        "selection": {
            "regime": args.regime,
            "negative_boundary_limit": args.negative_limit,
            "seed": args.seed,
            "positive_per_negative": 2,
            "maximum_positive_per_gold_block": 4,
            "presentation_per_source_label": args.per_source_label,
            "context_lines": args.context_lines,
            "maximum_display_gap_lines": args.maximum_display_gap_lines,
        },
        "training_summary": _summary(table.metadata, table.targets, training_rows),
        "presentation_summary": _summary(table.metadata, table.targets, sample_rows),
        "cases": cases,
        "inputs": {
            "table_manifest_sha256": sha256_file(Path(args.table_dir) / "manifest.json"),
            "source_jsonl_sha256": sha256_file(Path(args.source_jsonl)),
        },
    }
    output = Path(args.output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    packet_path = output / "packet.json"
    packet_path.write_text(
        json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt = {
        **{key: packet[key] for key in (
            "schema_version", "status", "code_commit", "slurm_job_id",
            "validation_opened", "label_tier", "selection", "training_summary",
            "presentation_summary", "inputs",
        )},
        "outputs": {
            "packet.json": {
                "bytes": packet_path.stat().st_size,
                "sha256": sha256_file(packet_path),
            }
        },
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--regime", default=DEFAULT_REGIME)
    parser.add_argument("--negative-limit", type=int, default=DEFAULT_NEGATIVE_LIMIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--per-source-label", type=int, default=DEFAULT_PER_SOURCE_LABEL)
    parser.add_argument("--context-lines", type=int, default=5)
    parser.add_argument("--maximum-display-gap-lines", type=int, default=40)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
