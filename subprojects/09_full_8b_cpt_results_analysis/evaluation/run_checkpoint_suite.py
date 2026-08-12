#!/usr/bin/env python3
"""Score one Apertus checkpoint on a frozen native-Greek benchmark file."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--native-runner", type=Path, required=True)
    parser.add_argument("--model", required=True, help="LABEL=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], required=True)
    parser.add_argument("--candidate-batch-size", type=int, default=64)
    parser.add_argument("--example-batch-size", type=int, default=64)
    parser.add_argument("--max-examples-per-benchmark", type=int, default=0)
    return parser.parse_args()


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


def parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--model must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not label or not path.is_dir():
        raise ValueError(f"invalid model binding: {value}")
    return label, path


def verify_model(model: Path, expected: dict[str, Any]) -> dict[str, Any]:
    config_path = model / "config.json"
    tokenizer_path = model / "tokenizer.json"
    config = json.loads(config_path.read_text())
    tokenizer_sha256 = sha256_file(tokenizer_path)
    allowed_tokenizers = set(expected["tokenizer_json_sha256_allowed"])
    checks = {
        "vocab_size": int(config["vocab_size"]) == int(expected["vocab_size"]),
        "rope_theta": float(config["rope_theta"]) == float(expected["rope_theta"]),
        "max_position_embeddings": int(config["max_position_embeddings"]) == int(expected["max_position_embeddings"]),
        "tie_word_embeddings": bool(config["tie_word_embeddings"]) is bool(expected["tie_word_embeddings"]),
        "tokenizer_json_sha256": tokenizer_sha256 in allowed_tokenizers,
    }
    if not all(checks.values()):
        raise ValueError(f"model contract failed: {checks}")
    return {"checks": checks, "config_sha256": sha256_file(config_path), "tokenizer_sha256": tokenizer_sha256}


def binary_stats(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    tp = sum(row["pred_index"] == 1 and row["answer_index"] == 1 for row in rows)
    fp = sum(row["pred_index"] == 1 and row["answer_index"] == 0 for row in rows)
    fn = sum(row["pred_index"] == 0 and row["answer_index"] == 1 for row in rows)
    tn = sum(row["pred_index"] == 0 and row["answer_index"] == 0 for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1_pos = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    precision_neg = tn / (tn + fn) if tn + fn else 0.0
    recall_neg = tn / (tn + fp) if tn + fp else 0.0
    f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg) if precision_neg + recall_neg else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "binary_macro_f1": (f1_pos + f1_neg) / 2,
        "balanced_accuracy": ((tp / (tp + fn) if tp + fn else 0.0) + (tn / (tn + fp) if tn + fp else 0.0)) / 2,
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["benchmark"], "__all__")].append(row)
        if row.get("subject"):
            groups[(row["benchmark"], str(row["subject"]))].append(row)
    output = []
    for (benchmark, subject), group in sorted(groups.items()):
        nll_sum = 0.0
        neglog2_sum = 0.0
        byte_sum = 0
        for row in group:
            scores = [float(value["avg_logprob"]) for value in row["choice_scores"]]
            maximum = max(scores)
            logz = maximum + math.log(sum(math.exp(value - maximum) for value in scores))
            nll_sum += logz - scores[int(row["answer_index"])]
            answer_score = row["choice_scores"][int(row["answer_index"])]
            neglog2_sum += -float(answer_score["sum_logprob"]) / math.log(2.0)
            byte_sum += int(row["correct_answer_utf8_bytes"])
        item: dict[str, Any] = {
            "benchmark": benchmark,
            "subject": subject,
            "n": len(group),
            "correct": sum(bool(row["correct"]) for row in group),
            "accuracy": sum(bool(row["correct"]) for row in group) / len(group),
            "choice_nll": nll_sum / len(group),
            "correct_answer_bpb": neglog2_sum / byte_sum,
        }
        if all(len(row["choice_scores"]) == 2 for row in group):
            item.update(binary_stats(group))
        output.append(item)

    nli = [row for row in rows if row["benchmark"] == "oyxoy_nli"]
    if nli:
        by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in nli:
            by_group[str(row["group_id"])].append(row)
        exact = sum(
            all(row["pred_index"] == row["answer_index"] for row in group)
            for group in by_group.values()
        )
        output.append(
            {
                "benchmark": "oyxoy_nli_exact_set",
                "subject": "__all__",
                "n": len(by_group),
                "correct": exact,
                "accuracy": exact / len(by_group),
                "choice_nll": None,
                "correct_answer_bpb": None,
            }
        )
    return output


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    label, model_path = parse_model(args.model)
    contract = json.loads(args.contract.read_text())
    manifest = json.loads(args.manifest.read_text())
    examples_path = Path(manifest["examples"]["path"])
    if sha256_file(examples_path) != manifest["examples"]["sha256"]:
        raise ValueError("frozen example hash drift")
    model_verification = verify_model(model_path, contract["model_contract"])
    native = load_module(args.native_runner.resolve())

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with examples_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            group = grouped[row["benchmark"]]
            if args.max_examples_per_benchmark <= 0 or len(group) < args.max_examples_per_benchmark:
                group.append(row)

    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    scorer = native.ChoiceScorer(
        str(model_path),
        dtype=args.dtype,
        max_input_tokens=int(contract["scoring"]["max_input_tokens"]),
        trust_remote_code=True,
    )
    predictions: list[dict[str, Any]] = []
    timings: dict[str, float] = {}
    for benchmark, raw_examples in grouped.items():
        benchmark_start = time.monotonic()
        examples = [
            native.MCQExample(
                benchmark=row["benchmark"],
                example_id=row["example_id"],
                question=row["question"],
                choices=row["choices"],
                answer_index=int(row["answer_index"]),
                subject=row.get("subject"),
                metadata=row.get("metadata") or {},
            )
            for row in raw_examples
        ]
        for start in range(0, len(examples), args.example_batch_size):
            chunk = examples[start : start + args.example_batch_size]
            scored = scorer.score_examples(chunk, candidate_batch_size=args.candidate_batch_size)
            for raw, example, result in zip(raw_examples[start : start + len(chunk)], chunk, scored, strict=True):
                predictions.append(
                    {
                        "model": label,
                        "benchmark": benchmark,
                        "example_id": example.example_id,
                        "subject": example.subject,
                        "group_id": raw.get("group_id"),
                        "answer_index": example.answer_index,
                        "pred_index": int(result["pred_index"]),
                        "correct": bool(result["correct"]),
                        "choice_scores": result["choice_scores"],
                        "correct_answer_utf8_bytes": len(example.choices[example.answer_index].encode("utf-8")),
                        "metadata": example.metadata or {},
                    }
                )
            done = start + len(chunk)
            if done == len(examples) or done % 1000 == 0:
                print(f"[{label}:{benchmark}] {done}/{len(examples)}", flush=True)
        timings[benchmark] = time.monotonic() - benchmark_start

    prediction_path = args.output_dir / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    metrics = summarize(predictions)
    metrics_path = args.output_dir / "metrics.csv"
    fields = sorted({key for row in metrics for key in row})
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)
    receipt = {
        "schema_version": "apertus_full8_native_greek_checkpoint_eval_v1",
        "status": "completed",
        "model": {"label": label, "path": str(model_path.resolve()), **model_verification},
        "dtype": args.dtype,
        "candidate_batch_size": args.candidate_batch_size,
        "example_batch_size": args.example_batch_size,
        "max_examples_per_benchmark": args.max_examples_per_benchmark,
        "contract": {"path": str(args.contract.resolve()), "sha256": sha256_file(args.contract)},
        "manifest": {"path": str(args.manifest.resolve()), "sha256": sha256_file(args.manifest)},
        "native_runner": {"path": str(args.native_runner.resolve()), "sha256": sha256_file(args.native_runner)},
        "counts": {key: len(value) for key, value in grouped.items()},
        "timings_seconds": timings,
        "wall_seconds": time.monotonic() - started,
        "metrics": metrics,
        "artifacts": {
            "predictions": {"path": str(prediction_path.resolve()), "sha256": sha256_file(prediction_path)},
            "metrics": {"path": str(metrics_path.resolve()), "sha256": sha256_file(metrics_path)},
        },
    }
    receipt_path = args.output_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "model": label, "wall_seconds": receipt["wall_seconds"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
