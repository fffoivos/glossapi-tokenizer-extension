#!/usr/bin/env python3
"""Aggregate exact frozen GreekMMLU shards and emit nested scored views."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from contract_utils import (
    executing_code_bundle,
    file_binding,
    read_json,
    require,
    require_file_binding,
    write_json_atomic,
)
from score_frozen_greekmmlu_shard import (
    clean_rows,
    panel_rows,
    read_jsonl,
)


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    require(not path.exists(), f"immutable JSONL output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def choice_nll(row: dict[str, Any]) -> float:
    scores = [float(item["avg_logprob"]) for item in row["choice_scores"]]
    answer = int(row["answer_index"])
    require(0 <= answer < len(scores), "answer index drift")
    maximum = max(scores)
    return maximum + math.log(sum(math.exp(value - maximum) for value in scores)) - scores[answer]


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    require(rows, "cannot summarize empty GreekMMLU view")
    correct = sum(int(bool(row["correct"])) for row in rows)
    nlls = [choice_nll(row) for row in rows]
    answer_bits = sum(
        -float(row["choice_scores"][int(row["answer_index"])]["sum_logprob"])
        / math.log(2.0)
        for row in rows
    )
    answer_bytes = sum(int(row["correct_answer_utf8_bytes"]) for row in rows)
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "choice_nll": sum(nlls) / len(nlls),
        "choice_nll_sum": sum(nlls),
        "correct_answer_bpb": answer_bits / answer_bytes,
        "correct_answer_neg_log2_sum": answer_bits,
        "correct_answer_utf8_bytes": answer_bytes,
    }


def stratified_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    level: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        subject[str(row.get("subject") or "__missing__")].append(row)
        value = row.get("metadata", {}).get("level")
        level[str(value or "__missing__")].append(row)
    return {
        "overall": metrics(rows),
        "by_subject": {key: metrics(value) for key, value in sorted(subject.items())},
        "by_educational_level": {
            key: metrics(value) for key, value in sorted(level.items())
        },
    }


def sentinel_ids(manifest_path: Path) -> tuple[dict[int, set[str]], dict[str, Any]]:
    manifest = read_json(manifest_path)
    require(
        manifest.get("schema_version") == "apertus_greekmmlu_sentinel_manifest_v1"
        and manifest.get("status") == "frozen"
        and manifest.get("sizes") == [4096, 8192]
        and manifest.get("strictly_nested") is True
        and manifest.get("selection_authorized") is False,
        "sentinel manifest drift",
    )
    result: dict[int, set[str]] = {}
    for size in (4096, 8192):
        path = require_file_binding(manifest["panels"][str(size)])
        ids = {str(row["example_id"]) for row in read_jsonl(path)}
        require(len(ids) == size, f"sentinel {size} count drift")
        result[size] = ids
    require(result[4096] < result[8192], "sentinel nesting drift")
    return result, manifest


def aggregate_shards(
    shards_root: Path,
    *,
    expected_shards: int,
    panel: Path,
    clean_examples: Path,
    model: Path,
    model_label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected_panel = file_binding(panel)
    expected_clean = file_binding(clean_examples)
    rows: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for index in range(expected_shards):
        root = shards_root / f"shard_{index:03d}"
        receipt_path = root / "receipt.json"
        receipt = read_json(receipt_path)
        predictions_path = root / "predictions.jsonl"
        require(
            receipt.get("schema_version") == "apertus_frozen_greekmmlu_shard_v1"
            and receipt.get("status") == "completed"
            and int(receipt.get("shard_index", -1)) == index
            and int(receipt.get("shard_count", -1)) == expected_shards
            and Path(str(receipt.get("model", ""))).resolve() == model.resolve()
            and receipt.get("model_label") == model_label
            and receipt.get("panel") == expected_panel
            and receipt.get("clean_examples") == expected_clean
            and receipt.get("predictions") == file_binding(predictions_path)
            and receipt.get("scoring", {}).get("candidate_batch_size") == 1
            and receipt.get("scoring", {}).get("dtype") == "float32",
            f"GreekMMLU shard receipt drift: {index}",
        )
        shard = read_jsonl(predictions_path)
        require(len(shard) == int(receipt["rows"]), f"GreekMMLU shard row drift: {index}")
        rows.extend(shard)
        receipts.append(file_binding(receipt_path))
    expected_rows = panel_rows(panel, clean_examples)
    expected_ids = {str(row["example_id"]) for row in expected_rows}
    observed_ids = [str(row["example_id"]) for row in rows]
    require(
        len(observed_ids) == len(set(observed_ids)) == len(expected_ids)
        and set(observed_ids) == expected_ids,
        "aggregated GreekMMLU id set drift",
    )
    order = {str(row["example_id"]): int(row["row_index"]) for row in clean_rows(clean_examples)}
    rows.sort(key=lambda row: order[str(row["example_id"])])
    return rows, receipts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full_clean", "sentinel_pair"), required=True)
    parser.add_argument("--clean-examples", type=Path, required=True)
    parser.add_argument("--sentinel-manifest", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=16)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--scale", choices=("8b", "1p5b"), required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--checkpoint-export", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    require(args.expected_shards == 16, "authoritative shard count must remain 16")
    require(not args.output_dir.exists(), f"immutable aggregate output exists: {args.output_dir}")
    rows, shard_receipts = aggregate_shards(
        args.shards_root,
        expected_shards=args.expected_shards,
        panel=args.panel,
        clean_examples=args.clean_examples,
        model=args.model,
        model_label=args.model_label,
    )
    ids_by_size, manifest = sentinel_ids(args.sentinel_manifest)
    if args.mode == "full_clean":
        require(len(rows) == 16_159, "full clean GreekMMLU count drift")
    else:
        require(
            len(rows) == 8192
            and {str(row["example_id"]) for row in rows} == ids_by_size[8192],
            "sentinel-pair scoring panel drift",
        )

    args.output_dir.mkdir(parents=True)
    scored_path = args.output_dir / "predictions.jsonl"
    write_jsonl_atomic(scored_path, rows)
    views: dict[str, Any] = {}
    for size in (4096, 8192):
        view_rows = [row for row in rows if str(row["example_id"]) in ids_by_size[size]]
        require(len(view_rows) == size, f"scored sentinel view is incomplete: {size}")
        view_path = args.output_dir / "views" / f"sentinel_{size}.predictions.jsonl"
        write_jsonl_atomic(view_path, view_rows)
        views[f"sentinel_{size}"] = {
            "predictions": file_binding(view_path),
            "metrics": stratified_metrics(view_rows),
        }
    if args.mode == "full_clean":
        full_path = args.output_dir / "views/full_clean.predictions.jsonl"
        write_jsonl_atomic(full_path, rows)
        views["full_clean"] = {
            "predictions": file_binding(full_path),
            "metrics": stratified_metrics(rows),
        }

    summary_path = args.output_dir / "summary.json"
    write_json_atomic(
        summary_path,
        {
            "schema_version": "apertus_frozen_greekmmlu_summary_v1",
            "status": "completed",
            "scale": args.scale,
            "iteration": args.iteration,
            "mode": args.mode,
            "model_label": args.model_label,
            "scored_rows": len(rows),
            "views": views,
        },
    )
    receipt_path = args.output_dir / "receipt.json"
    write_json_atomic(
        receipt_path,
        {
            "schema_version": "apertus_frozen_greekmmlu_evaluation_v1",
            "status": "completed",
            "executing_code_bundle": executing_code_bundle(),
            "scale": args.scale,
            "iteration": args.iteration,
            "mode": args.mode,
            "model": str(args.model.resolve()),
            "model_label": args.model_label,
            "checkpoint_export": file_binding(args.checkpoint_export),
            "clean_examples": file_binding(args.clean_examples),
            "sentinel_manifest": file_binding(args.sentinel_manifest),
            "sentinel_source_examples": manifest["source_examples"],
            "scoring": {
                "dtype": "float32",
                "max_input_tokens": 3072,
                "candidate_batch_size": 1,
                "example_batch_size": 16,
                "shards": 16,
            },
            "shard_receipts": shard_receipts,
            "predictions": file_binding(scored_path),
            "summary": file_binding(summary_path),
            "views": {name: value["predictions"] for name, value in views.items()},
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
