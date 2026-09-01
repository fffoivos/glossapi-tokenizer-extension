#!/usr/bin/env python3
"""Deterministically rebuild the historical GreekMMLU query JSONL schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def canonical_sha256(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.output_jsonl, args.summary_json):
        if output.exists():
            raise FileExistsError(f"immutable output exists: {output}")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    expected = contract["dataset"]
    generated_utc = contract["builder_arguments"]["generated_utc"]
    sys.path.insert(0, str(args.eval_dir))
    import run_native_greek_mcq_eval as native_mcq  # type: ignore

    registry = native_mcq.load_registry(args.registry)
    specs = [row for row in native_mcq.selected_benchmarks(registry, "all") if row["id"] == "greekmmlu"]
    if len(specs) != 1:
        raise ValueError(f"expected one GreekMMLU spec, found {len(specs)}")
    spec = specs[0]
    if {
        "repo_id": spec["source"], "revision": spec.get("revision"),
        "config": spec.get("config"), "split": spec.get("split"),
    } != {key: expected[key] for key in ("repo_id", "revision", "config", "split")}:
        raise ValueError("GreekMMLU registry binding drift")
    dataset, split = native_mcq._load_dataset(spec)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with args.output_jsonl.open("x", encoding="utf-8") as output:
        for row_index, source_row in enumerate(dataset):
            raw_row = dict(source_row)
            example = native_mcq.examples_from_row(spec, raw_row, row_index)
            if example is None:
                continue
            answer_text = example.choices[example.answer_index]
            choices_text = "\n".join(str(choice) for choice in example.choices)
            query = {
                "schema": "greek-mcq-decontam-query-v1",
                "generated_utc": generated_utc,
                "benchmark": "greekmmlu",
                "dataset_repo_id": spec["source"],
                "dataset_revision": spec.get("revision"),
                "dataset_config": spec.get("config"),
                "split": split,
                "example_id": example.example_id,
                "row_index": row_index,
                "subject": example.subject,
                "question": example.question,
                "choices": example.choices,
                "answer_index": example.answer_index,
                "answer_text": answer_text,
                "metadata": example.metadata or {},
                "raw_row_sha256": canonical_sha256(raw_row),
                "surfaces": {
                    "question": example.question,
                    "question_all_choices": f"{example.question}\n{choices_text}",
                    "question_correct_answer": f"{example.question}\n{answer_text}",
                    "eval_prompt": native_mcq.build_prompt(example),
                },
            }
            output.write(json.dumps(query, ensure_ascii=False, sort_keys=True) + "\n")
            rows += 1
    if rows != expected["expected_questions"]:
        raise ValueError(f"GreekMMLU query count drift: {rows}")
    summary = {
        "schema": "greek-mcq-decontam-query-summary-v1",
        "generated_utc": generated_utc,
        "registry": str(args.registry.resolve()),
        "registry_sha256": hashlib.sha256(args.registry.read_bytes()).hexdigest(),
        "eval_dir": str(args.eval_dir.resolve()),
        "benchmarks": {
            "greekmmlu": {
                "source": spec["source"], "config": spec.get("config"),
                "revision": spec.get("revision"), "split_requested": spec.get("split"),
                "split_loaded": split, "items": rows, "surfaces": rows * 4,
            }
        },
        "total_items": rows,
        "total_surfaces": rows * 4,
        "output_jsonl": str(args.output_jsonl.resolve()),
        "output_jsonl_sha256": hashlib.sha256(args.output_jsonl.read_bytes()).hexdigest(),
        "deterministic_generated_utc": True,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.summary_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
