#!/usr/bin/env python3
"""Evaluate Greek Belebele and DemosQA with the frozen native MCQ scorer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
from pathlib import Path


LABELS = {"Α": 0, "Β": 1, "Γ": 2, "Δ": 3}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_native_greek_mcq", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def demos_choices(value: str) -> list[str]:
    matches = list(re.finditer(r"(?:^|\n\s*\n)([ΑΒΓΔ])\.\s*", value))
    if [match.group(1) for match in matches] != ["Α", "Β", "Γ", "Δ"]:
        raise ValueError("DemosQA answer serialization drift")
    choices = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(value)
        choices.append(value[match.end() : end].strip().strip('"').strip())
    return choices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--native-runner", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from datasets import load_dataset

    native = load_module(args.native_runner.resolve())
    contract = json.loads(args.contract.read_text())
    if contract.get("schema_version") != "apertus_mini_endpoint_benchmark_contract_v1":
        raise ValueError("endpoint benchmark contract drift")
    examples = []
    counts = {}
    for benchmark, spec in contract["benchmarks"].items():
        dataset = load_dataset(
            spec["repo_id"], spec.get("config"), split=spec["split"], revision=spec["revision"]
        )
        if len(dataset) != int(spec["expected_rows"]):
            raise ValueError(f"{benchmark} row-count drift: {len(dataset)}")
        counts[benchmark] = len(dataset)
        for index, row in enumerate(dataset):
            if benchmark == "greek_belebele":
                question = f"Κείμενο:\n{row['flores_passage']}\n\nΕρώτηση:\n{row['question']}"
                choices = [str(row[f"mc_answer{choice}"]) for choice in range(1, 5)]
                answer = int(row["correct_answer_num"]) - 1
                example_id = f"ell_Grek:{row['question_number']}"
            elif benchmark == "demosqa":
                question = str(row["question"])
                choices = demos_choices(str(row["answers"]))
                answer = LABELS[str(row["best_answer_index"]).strip()]
                example_id = str(row["id"])
            else:
                raise ValueError(benchmark)
            examples.append(
                native.MCQExample(
                    benchmark=benchmark,
                    example_id=example_id,
                    question=question,
                    choices=choices,
                    answer_index=answer,
                    subject=None,
                    metadata=None,
                )
            )
    scorer = native.ChoiceScorer(
        args.model, dtype="bfloat16", max_input_tokens=3072, trust_remote_code=True
    )
    rows = []
    for start in range(0, len(examples), 16):
        batch = examples[start : start + 16]
        scored = scorer.score_examples(batch, candidate_batch_size=args.candidate_batch_size)
        for example, result in zip(batch, scored, strict=True):
            rows.append(
                {
                    "benchmark": example.benchmark,
                    "example_id": example.example_id,
                    "answer_index": example.answer_index,
                    "correct": result["correct"],
                    "pred_index": result["pred_index"],
                    "choice_scores": result["choice_scores"],
                    "correct_answer_utf8_bytes": len(
                        example.choices[example.answer_index].encode("utf-8")
                    ),
                }
            )
    summaries = []
    for benchmark in contract["benchmarks"]:
        group = [row for row in rows if row["benchmark"] == benchmark]
        correct = sum(int(row["correct"]) for row in group)
        nll = 0.0; neglog2 = 0.0; byte_count = 0
        for row in group:
            scores = [float(value["avg_logprob"]) for value in row["choice_scores"]]
            maximum = max(scores)
            normalizer = maximum + math.log(sum(math.exp(value - maximum) for value in scores))
            answer = int(row["answer_index"])
            nll += normalizer - scores[answer]
            neglog2 += -float(row["choice_scores"][answer]["sum_logprob"]) / math.log(2)
            byte_count += int(row["correct_answer_utf8_bytes"])
        summaries.append(
            {
                "benchmark": benchmark,
                "n": len(group),
                "accuracy": correct / len(group),
                "choice_nll": nll / len(group),
                "correct_answer_bpb": neglog2 / byte_count,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    predictions = args.output_dir / "predictions.jsonl"
    with predictions.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = args.output_dir / "summary.csv"
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    receipt = {
        "schema_version": "apertus_mini_greek_endpoint_benchmarks_v1",
        "status": "completed",
        "model_label": args.model_label,
        "model_path": str(Path(args.model).resolve()),
        "contract": {"path": str(args.contract.resolve()), "sha256": sha256_file(args.contract)},
        "native_runner": {"path": str(args.native_runner.resolve()), "sha256": sha256_file(args.native_runner)},
        "metrics": summaries,
        "artifacts": {
            "predictions": {"path": str(predictions.resolve()), "sha256": sha256_file(predictions)},
            "summary": {"path": str(summary.resolve()), "sha256": sha256_file(summary)},
        },
    }
    receipt_path = args.output_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "model": args.model_label, "rows": len(rows), "benchmarks": counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
