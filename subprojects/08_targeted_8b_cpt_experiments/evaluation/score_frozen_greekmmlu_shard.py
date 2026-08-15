#!/usr/bin/env python3
"""Score one deterministic shard of the frozen benchmark-clean GreekMMLU panel."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from contract_utils import (
    executing_code_bundle,
    file_binding,
    read_json,
    require,
    write_json_atomic,
)

REPO_ID = "dascim/GreekMMLU"
REVISION = "6a03aa06b68beb932fb75edff3a34e50b3674649"
CONFIG = "All"
SPLIT = "test"
FULL_COUNT = 16_632
CLEAN_COUNT = 16_159


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            require(isinstance(value, dict), f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def clean_rows(path: Path) -> list[dict[str, Any]]:
    value = read_json(path)
    require(
        value.get("schema_version") == "apertus_greekmmlu_clean_examples_v1"
        and value.get("status") == "frozen"
        and value.get("dataset")
        == {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "config": CONFIG,
            "split": SPLIT,
        }
        and int(value.get("full_count", -1)) == FULL_COUNT
        and int(value.get("clean_count", -1)) == CLEAN_COUNT,
        "frozen clean GreekMMLU identity drift",
    )
    rows = value.get("examples")
    require(isinstance(rows, list) and len(rows) == CLEAN_COUNT, "clean example rows drift")
    return rows


def panel_rows(panel: Path, clean_examples: Path) -> list[dict[str, Any]]:
    clean = clean_rows(clean_examples)
    clean_by_id = {str(row["example_id"]): row for row in clean}
    require(len(clean_by_id) == CLEAN_COUNT, "duplicate clean example ids")
    if panel.resolve() == clean_examples.resolve():
        rows = clean
    else:
        rows = read_jsonl(panel)
    ids = [str(row.get("example_id", "")) for row in rows]
    require(ids and len(ids) == len(set(ids)), "panel ids are empty or duplicated")
    require(set(ids) <= set(clean_by_id), "panel is not a subset of clean GreekMMLU")
    normalized: list[dict[str, Any]] = []
    for row in rows:
        example_id = str(row["example_id"])
        authority = clean_by_id[example_id]
        require(
            int(row["row_index"]) == int(authority["row_index"])
            and str(row["subject"]) == str(authority["subject"])
            and row.get("educational_level") == authority.get("educational_level"),
            f"panel metadata drift: {example_id}",
        )
        normalized.append(authority)
    return sorted(normalized, key=lambda row: int(row["row_index"]))


def shard_rows(
    rows: list[dict[str, Any]], *, shard_index: int, shard_count: int
) -> list[dict[str, Any]]:
    require(shard_count > 0 and 0 <= shard_index < shard_count, "invalid shard geometry")
    return [row for ordinal, row in enumerate(rows) if ordinal % shard_count == shard_index]


def import_native_runner(path: Path) -> ModuleType:
    require(path.is_file(), f"native Greek scorer missing: {path}")
    spec = importlib.util.spec_from_file_location("h2g_frozen_native_greek_runner", path)
    require(spec is not None and spec.loader is not None, "cannot load native Greek scorer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in ("ChoiceScorer", "examples_from_row"):
        require(hasattr(module, name), f"native scorer API missing: {name}")
    return module


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-examples", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--native-runner", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-batch-size", type=int, default=1)
    parser.add_argument("--example-batch-size", type=int, default=16)
    args = parser.parse_args()

    require(args.candidate_batch_size == 1, "authoritative candidate batch must remain 1")
    require(args.example_batch_size == 16, "authoritative example batch must remain 16")
    require(not args.output_dir.exists(), f"immutable shard output exists: {args.output_dir}")
    selected = shard_rows(
        panel_rows(args.panel, args.clean_examples),
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    require(selected, "empty GreekMMLU shard")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("datasets is required in the frozen evaluation environment") from exc

    dataset = load_dataset(REPO_ID, CONFIG, revision=REVISION, split=SPLIT)
    require(len(dataset) == FULL_COUNT, "pinned GreekMMLU row count drift")
    runner = import_native_runner(args.native_runner)
    examples = []
    for authority in selected:
        row_index = int(authority["row_index"])
        example = runner.examples_from_row(
            {"id": "greekmmlu"}, dict(dataset[row_index]), row_index
        )
        require(example is not None, f"scorer rejected frozen row {row_index}")
        require(
            str(example.example_id) == str(authority["example_id"])
            and str(example.subject) == str(authority["subject"]),
            f"pinned dataset reconciliation drift: {authority['example_id']}",
        )
        examples.append(example)

    scorer = runner.ChoiceScorer(
        str(args.model.resolve()),
        dtype="float32",
        max_input_tokens=3072,
        trust_remote_code=True,
    )
    predictions: list[dict[str, Any]] = []
    for start in range(0, len(examples), args.example_batch_size):
        chunk = examples[start : start + args.example_batch_size]
        scored = scorer.score_examples(chunk, candidate_batch_size=1)
        for example, result in zip(chunk, scored):
            predictions.append(
                {
                    "model": args.model_label,
                    "benchmark": "greekmmlu",
                    "example_id": str(example.example_id),
                    "subject": example.subject,
                    "answer_index": int(example.answer_index),
                    "pred_index": int(result["pred_index"]),
                    "correct": bool(result["correct"]),
                    "choice_scores": result["choice_scores"],
                    "num_choices": len(example.choices),
                    "correct_answer_utf8_bytes": len(
                        example.choices[example.answer_index].encode("utf-8")
                    ),
                    "metadata": example.metadata or {},
                    "frozen_row_index": int(selected[len(predictions)]["row_index"]),
                }
            )
        print(
            f"shard {args.shard_index}/{args.shard_count}: "
            f"{min(start + len(chunk), len(examples))}/{len(examples)}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    write_jsonl_atomic(predictions_path, predictions)
    write_json_atomic(
        args.output_dir / "receipt.json",
        {
            "schema_version": "apertus_frozen_greekmmlu_shard_v1",
            "status": "completed",
            "executing_code_bundle": executing_code_bundle(),
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "rows": len(predictions),
            "model": str(args.model.resolve()),
            "model_label": args.model_label,
            "clean_examples": file_binding(args.clean_examples),
            "panel": file_binding(args.panel),
            "native_runner": file_binding(args.native_runner),
            "scoring": {
                "dtype": "float32",
                "max_input_tokens": 3072,
                "candidate_batch_size": 1,
                "example_batch_size": 16,
                "choice_score": "mean_continuation_log_probability",
            },
            "predictions": file_binding(predictions_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
