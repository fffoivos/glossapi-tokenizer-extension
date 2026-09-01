#!/usr/bin/env python3
"""Apply equations (1)-(2) to aligned checkpoint predictions.jsonl files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from checkpoint_instability import (
    analyze_example_trajectories,
    mean_total_variation,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_receipt(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def load_ids(path: Path) -> set[str]:
    ids = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not ids:
        raise ValueError(f"{path}: ID filter is empty")
    return ids


def load_predictions(path: Path, *, id_field: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            if id_field not in row:
                raise ValueError(f"{path}:{line_number}: missing {id_field!r}")
            example_id = str(row[id_field])
            if example_id in rows:
                raise ValueError(f"{path}:{line_number}: duplicate example ID {example_id!r}")
            rows[example_id] = row
    if not rows:
        raise ValueError(f"{path}: no prediction rows")
    return rows


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=/path/to/predictions.jsonl")
    label, raw_path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("checkpoint label must not be empty")
    path = Path(raw_path)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"prediction file does not exist: {path}")
    return label, path


def analyze_prediction_files(
    checkpoints: list[tuple[str, Path]],
    *,
    selected_ids: set[str] | None = None,
    id_field: str = "example_id",
    score_field: str = "correct",
    output_field: str = "pred_index",
) -> dict[str, Any]:
    if len(checkpoints) < 2:
        raise ValueError(f"at least two checkpoints are required, got {len(checkpoints)}")
    labels = [label for label, _ in checkpoints]
    if len(labels) != len(set(labels)):
        raise ValueError(f"checkpoint labels must be unique: {labels}")

    loaded: list[tuple[str, Path, dict[str, dict[str, Any]]]] = []
    for label, path in checkpoints:
        loaded.append((label, path, load_predictions(path, id_field=id_field)))

    expected_ids = selected_ids if selected_ids is not None else set(loaded[0][2])
    if not expected_ids:
        raise ValueError("selected example set is empty")
    for label, path, rows in loaded:
        missing = expected_ids - set(rows)
        if missing:
            raise ValueError(
                f"{label} ({path}): missing {len(missing)} selected IDs; "
                f"first={sorted(missing)[:5]}"
            )
        if selected_ids is None and set(rows) != expected_ids:
            unexpected = set(rows) - expected_ids
            raise ValueError(
                f"{label} ({path}): checkpoint example-set drift; "
                f"unexpected={sorted(unexpected)[:5]}"
            )

    ordered_ids = sorted(expected_ids)
    scores_by_example: dict[str, list[float]] = {example_id: [] for example_id in ordered_ids}
    outputs_by_example: dict[str, list[Any]] = {example_id: [] for example_id in ordered_ids}
    checkpoint_scores: list[float] = []
    transition_rows: list[dict[str, Any]] = []

    for label, path, rows in loaded:
        total = 0.0
        for example_id in ordered_ids:
            row = rows[example_id]
            if score_field not in row:
                raise ValueError(f"{label} ({path}): {example_id} missing {score_field!r}")
            if output_field not in row:
                raise ValueError(f"{label} ({path}): {example_id} missing {output_field!r}")
            raw_score = row[score_field]
            if isinstance(raw_score, bool):
                score = float(raw_score)
            elif isinstance(raw_score, (int, float)):
                score = float(raw_score)
            else:
                raise TypeError(
                    f"{label} ({path}): {example_id} field {score_field!r} "
                    f"must be numeric or boolean"
                )
            scores_by_example[example_id].append(score)
            outputs_by_example[example_id].append(row[output_field])
            total += score
        checkpoint_scores.append(total / len(ordered_ids))

    metrics = analyze_example_trajectories(scores_by_example, outputs_by_example)
    for index in range(len(loaded) - 1):
        changed_scores = 0
        changed_outputs = 0
        absolute_score_change = 0.0
        for example_id in ordered_ids:
            previous_score = scores_by_example[example_id][index]
            current_score = scores_by_example[example_id][index + 1]
            absolute_score_change += abs(current_score - previous_score)
            changed_scores += previous_score != current_score
            changed_outputs += (
                outputs_by_example[example_id][index]
                != outputs_by_example[example_id][index + 1]
            )
        transition_rows.append(
            {
                "from": loaded[index][0],
                "to": loaded[index + 1][0],
                "mean_absolute_score_change": absolute_score_change / len(ordered_ids),
                "score_change_rate": changed_scores / len(ordered_ids),
                "output_change_rate": changed_outputs / len(ordered_ids),
            }
        )

    return {
        "schema_version": "checkpoint_instability_arxiv_2510_04848_v1",
        "paper": {
            "title": "Instability in Downstream Task Performance During LLM Pretraining",
            "arxiv_id": "2510.04848v1",
            "url": "https://arxiv.org/abs/2510.04848",
        },
        "mapping": {
            "id_field": id_field,
            "reference_score_field": score_field,
            "output_field": output_field,
            "output_similarity": "exact_match",
            "aggregation": "unweighted_mean_over_examples",
        },
        "checkpoint_labels": labels,
        "example_count": len(ordered_ids),
        "checkpoint_count": len(checkpoints),
        "checkpoint_mean_scores": [
            {"checkpoint": label, "mean_score": score}
            for label, score in zip(labels, checkpoint_scores, strict=True)
        ],
        "expression_1": {
            "name": "mean_total_variation",
            "mean_example_mtv": metrics["mean_example_mtv"],
            "aggregate_score_trajectory_mtv": mean_total_variation(checkpoint_scores),
        },
        "expression_2": {
            "name": "instability_score",
            "mean_example_is": metrics["mean_example_is"],
        },
        "adjacent_transitions": transition_rows,
        "per_example": metrics["per_example"],
        "sources": [
            {"checkpoint": label, "predictions": file_receipt(path)}
            for label, path, _ in loaded
        ],
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_checkpoint,
        required=True,
        metavar="LABEL=PATH",
        help="ordered checkpoint prediction file; repeat at least twice",
    )
    parser.add_argument("--ids", type=Path, help="optional newline-delimited example-ID subset")
    parser.add_argument("--id-field", default="example_id")
    parser.add_argument("--score-field", default="correct")
    parser.add_argument("--output-field", default="pred_index")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selected_ids = load_ids(args.ids) if args.ids else None
    payload = analyze_prediction_files(
        args.checkpoint,
        selected_ids=selected_ids,
        id_field=args.id_field,
        score_field=args.score_field,
        output_field=args.output_field,
    )
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    if args.ids:
        payload["id_filter"] = file_receipt(args.ids)
    write_json_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output.resolve()),
                "example_count": payload["example_count"],
                "checkpoint_count": payload["checkpoint_count"],
                "mean_example_mtv": payload["expression_1"]["mean_example_mtv"],
                "mean_example_is": payload["expression_2"]["mean_example_is"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
