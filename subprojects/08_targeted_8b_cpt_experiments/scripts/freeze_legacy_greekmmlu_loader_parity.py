#!/usr/bin/env python3
"""Prove a frozen GreekMMLU snapshot preserves revision-aware and legacy examples."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from contract_utils import executing_code_bundle, file_binding, read_json, require, write_json_atomic


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot load evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def canonical_sha256(value: Any) -> str:
    import hashlib

    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def query_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(rows) == 16_632, "legacy snapshot question-count drift")
    require([int(row["row_index"]) for row in rows] == list(range(16_632)), "legacy snapshot row order drift")
    return rows


def example_tuple(example: Any, prompt: str) -> tuple[Any, ...]:
    return (
        str(example.example_id), str(example.question), tuple(map(str, example.choices)),
        int(example.answer_index), str(example.subject or ""), prompt,
    )


def reconstructed_raw(row: dict[str, Any]) -> dict[str, Any]:
    raw = {
        "id": str(row["example_id"]),
        "question": str(row["question"]),
        "choices": list(row["choices"]),
        "answer": int(row["answer_index"]),
        "subject": str(row.get("subject") or ""),
    }
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for key in ("category", "sub_category", "level", "subject"):
            if key in metadata and metadata[key] is not None:
                raw[key] = metadata[key]
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-receipt", type=Path, required=True)
    parser.add_argument("--current-evaluator", type=Path, required=True)
    parser.add_argument("--current-registry", type=Path, required=True)
    parser.add_argument("--legacy-evaluator", type=Path, required=True)
    parser.add_argument("--legacy-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable loader-parity receipt exists: {args.output}")
    receipt = read_json(args.query_receipt)
    require(receipt.get("schema_version") == "apertus_frozen_greekmmlu_queries_receipt_v1" and receipt.get("status") == "passed", "GreekMMLU query receipt drift")
    queries_binding = receipt.get("queries")
    require(isinstance(queries_binding, dict), "GreekMMLU snapshot binding missing")
    queries_path = Path(str(queries_binding.get("path", "")))
    require(queries_path.is_file() and queries_binding == {**file_binding(queries_path), "rows": 16_632, "unique_example_ids": 16_632}, "GreekMMLU snapshot binding drift")
    frozen = query_rows(queries_path)
    current = load_module("h2g_current_greekmmlu", args.current_evaluator)
    legacy = load_module("h2g_legacy_greekmmlu", args.legacy_evaluator)
    current_registry = current.load_registry(args.current_registry)
    current_specs = [row for row in current.selected_benchmarks(current_registry, "all") if row["id"] == "greekmmlu"]
    require(len(current_specs) == 1, "current GreekMMLU registry drift")
    current_spec = current_specs[0]
    require(current_spec.get("revision") == "6a03aa06b68beb932fb75edff3a34e50b3674649", "current GreekMMLU revision drift")
    dataset, split = current._load_dataset(current_spec)
    require(split == "test" and len(dataset) == 16_632, "revision-aware GreekMMLU load drift")
    legacy_registry = legacy.load_registry(args.legacy_registry)
    legacy_specs = [row for row in legacy.selected_benchmarks(legacy_registry, "all") if row["id"] == "greekmmlu"]
    require(len(legacy_specs) == 1, "legacy GreekMMLU registry drift")
    legacy_spec = legacy_specs[0]
    for index, (source_row, query) in enumerate(zip(dataset, frozen, strict=True)):
        raw = dict(source_row)
        require(canonical_sha256(raw) == query["raw_row_sha256"], f"pinned raw-row hash drift at {index}")
        current_example = current.examples_from_row(current_spec, raw, index)
        require(current_example is not None, f"current evaluator omitted GreekMMLU row {index}")
        legacy_example = legacy.examples_from_row(legacy_spec, reconstructed_raw(query), index)
        require(legacy_example is not None, f"legacy adapter omitted GreekMMLU row {index}")
        expected = (
            str(query["example_id"]), str(query["question"]), tuple(map(str, query["choices"])),
            int(query["answer_index"]), str(query.get("subject") or ""), str(query["surfaces"]["eval_prompt"]),
        )
        require(example_tuple(current_example, current.build_prompt(current_example)) == expected, f"revision-aware example/prompt drift at {index}")
        require(example_tuple(legacy_example, legacy.build_prompt(legacy_example)) == expected, f"legacy snapshot adapter example/prompt drift at {index}")
    payload = {
        "schema_version": "apertus_legacy_public_greekmmlu_loader_parity_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset": {"repo_id": "dascim/GreekMMLU", "revision": current_spec["revision"], "config": "All", "split": "test", "questions": 16_632},
        "query_receipt": file_binding(args.query_receipt),
        "snapshot": file_binding(queries_path),
        "checks": {
            "all_raw_row_hashes_match_pinned_revision": True,
            "all_example_fields_match": True,
            "all_prompts_match": True,
            "legacy_change_is_loader_only": True,
        },
        "sources": {
            "current_evaluator": file_binding(args.current_evaluator),
            "current_registry": file_binding(args.current_registry),
            "legacy_evaluator": file_binding(args.legacy_evaluator),
            "legacy_registry": file_binding(args.legacy_registry),
        },
        "executing_code_bundle": executing_code_bundle(),
    }
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
