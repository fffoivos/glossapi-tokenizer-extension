#!/usr/bin/env python3
"""Derive a block table with a train-OOF continuation-head probability."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contract import sha256_file


SCHEMA_VERSION = "bibliography-continuation-block-table-ablation-v1"
CONTINUATION_COLUMN = 1


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"expected object in {path}")
                yield value


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _save(path: Path, value: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)


def _candidate_coordinates(
    *, source_path: Path, row_indices: np.ndarray, expected_line_count: int,
    expected_split: str,
) -> list[tuple[str, int]]:
    if len(row_indices) and not np.all(row_indices[1:] > row_indices[:-1]):
        raise ValueError("candidate row indices are not strictly increasing")
    result: list[tuple[str, int] | None] = [None] * len(row_indices)
    cursor = candidate_cursor = 0
    with source_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            document = json.loads(raw)
            if document["split"] != expected_split:
                continue
            lines = document["lines"]
            end = cursor + len(lines)
            while candidate_cursor < len(row_indices):
                index = int(row_indices[candidate_cursor])
                if index >= end:
                    break
                if index < cursor:
                    raise ValueError("candidate/source ordering mismatch")
                line = lines[index - cursor]
                result[candidate_cursor] = (
                    str(document["document_id"]), int(line["abs_idx"])
                )
                candidate_cursor += 1
            cursor = end
    if cursor != expected_line_count or candidate_cursor != len(row_indices):
        raise ValueError("candidate/source coverage mismatch")
    if any(value is None for value in result):
        raise ValueError("candidate coordinates are incomplete")
    return [value for value in result if value is not None]


def derive(
    *, base_block_table: Path, connector_table: Path, source_path: Path,
    continuation_probability_path: Path, continuation_report_path: Path,
    output_dir: Path, code_commit: str, slurm_job_id: str,
    blend_alpha: float = 1.0, connector_rule: str = "max",
) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    if not 0 < blend_alpha <= 1:
        raise ValueError("blend_alpha must be in (0,1]")
    if connector_rule not in {"preserve", "max"}:
        raise ValueError("connector_rule must be preserve or max")
    base_manifest = json.loads((base_block_table / "manifest.json").read_text(encoding="utf-8"))
    connector_manifest = json.loads((connector_table / "manifest.json").read_text(encoding="utf-8"))
    continuation_report = json.loads(continuation_report_path.read_text(encoding="utf-8"))
    if (
        base_manifest.get("split") != "train"
        or base_manifest.get("validation_opened") is not False
        or connector_manifest.get("split") != "train"
        or continuation_report.get("validation_opened") is not False
    ):
        raise ValueError("the continuation block ablation accepts train-only inputs")
    if sha256_file(source_path) != connector_manifest["inputs"]["source_sha256"]:
        raise ValueError("source does not match the connector table")

    candidate_rows = np.load(connector_table / "row_indices.npy", mmap_mode="r", allow_pickle=False)
    continuation = np.load(continuation_probability_path, mmap_mode="r", allow_pickle=False)
    if continuation.shape != candidate_rows.shape:
        raise ValueError("continuation probability is not aligned with connector candidates")
    coordinates = _candidate_coordinates(
        source_path=source_path, row_indices=candidate_rows,
        expected_line_count=int(connector_manifest["line_count"]), expected_split="train",
    )
    by_coordinate: dict[tuple[str, int], float] = {}
    for coordinate, probability in zip(coordinates, continuation, strict=True):
        if coordinate in by_coordinate:
            raise ValueError(f"duplicate candidate coordinate: {coordinate}")
        if np.isfinite(probability):
            if not 0 <= float(probability) <= 1:
                raise ValueError("continuation probability is outside [0,1]")
            by_coordinate[coordinate] = float(probability)

    arrays = {
        name: np.load(base_block_table / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in (
            "role_probability", "connector_probability", "abs_indices", "char_lengths",
            "gold_roles", "trusted",
        )
    }
    roles = np.asarray(arrays["role_probability"], dtype=np.float32).copy()
    connector = np.asarray(arrays["connector_probability"], dtype=np.float32).copy()
    documents = list(_iter_jsonl(base_block_table / "documents.jsonl"))
    overridden = missing = 0
    deltas = []
    for document in documents:
        parent = str(document["parent_document_id"])
        start, end = int(document["line_start"]), int(document["line_end"])
        for index in range(start, end):
            coordinate = (parent, int(arrays["abs_indices"][index]))
            probability = by_coordinate.get(coordinate)
            if probability is None:
                missing += 1
                continue
            old = float(roles[index, CONTINUATION_COLUMN])
            blended = (1.0 - blend_alpha) * old + blend_alpha * probability
            roles[index, CONTINUATION_COLUMN] = blended
            if connector_rule == "max":
                connector[index] = max(float(connector[index]), blended)
            deltas.append(blended - old)
            overridden += 1
    if not overridden or not np.isfinite(roles).all() or not np.isfinite(connector).all():
        raise ValueError("continuation override did not produce a valid block table")

    output_dir.mkdir(parents=True)
    _save(output_dir / "role_probability.npy", roles)
    _save(output_dir / "connector_probability.npy", connector)
    for name in ("abs_indices", "char_lengths", "gold_roles", "trusted"):
        _save(output_dir / f"{name}.npy", np.asarray(arrays[name]))
    shutil.copyfile(base_block_table / "documents.jsonl", output_dir / "documents.jsonl")
    manifest = {
        **base_manifest,
        "schema_version": SCHEMA_VERSION,
        "status": "passed_train_only_continuation_block_table_ablation",
        "code_commit": code_commit,
        "slurm_job_id": slurm_job_id,
        "validation_opened": False,
        "continuation_override": {
            "column": CONTINUATION_COLUMN,
            "blend_alpha": blend_alpha,
            "eligible_probability_count": len(by_coordinate),
            "overridden_block_line_count": overridden,
            "block_line_without_eligible_continuation_probability": missing,
            "mean_probability_delta": float(np.mean(deltas)),
            "median_probability_delta": float(np.median(deltas)),
            "connector_probability_rule": connector_rule,
            "other_role_probabilities_changed": False,
        },
        "inputs": {
            **base_manifest.get("inputs", {}),
            "base_block_manifest_sha256": sha256_file(base_block_table / "manifest.json"),
            "connector_table_manifest_sha256": sha256_file(connector_table / "manifest.json"),
            "continuation_probability_sha256": sha256_file(continuation_probability_path),
            "continuation_report_sha256": sha256_file(continuation_report_path),
            "source_sha256": sha256_file(source_path),
        },
    }
    _write_json_new(output_dir / "manifest.json", manifest)
    _write_json_new(output_dir / "receipt.json", {
        **manifest,
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(output_dir.iterdir()) if path.is_file()
        },
    })
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-block-table", type=Path, required=True)
    parser.add_argument("--connector-table", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--continuation-probability", type=Path, required=True)
    parser.add_argument("--continuation-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blend-alpha", type=float, default=1.0)
    parser.add_argument("--connector-rule", choices=("preserve", "max"), default="max")
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", default=os.environ.get("SLURM_JOB_ID", ""))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = derive(
        base_block_table=args.base_block_table.resolve(),
        connector_table=args.connector_table.resolve(),
        source_path=args.source_jsonl.resolve(),
        continuation_probability_path=args.continuation_probability.resolve(),
        continuation_report_path=args.continuation_report.resolve(),
        output_dir=args.output_dir.resolve(),
        code_commit=args.code_commit,
        slurm_job_id=args.slurm_job_id,
        blend_alpha=args.blend_alpha,
        connector_rule=args.connector_rule,
    )
    print(json.dumps(result["continuation_override"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
