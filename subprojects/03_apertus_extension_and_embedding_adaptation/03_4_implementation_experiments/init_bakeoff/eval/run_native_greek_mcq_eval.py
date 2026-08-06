#!/usr/bin/env python3
"""Evaluate native Greek MCQ datasets with causal-LM log-likelihood scoring."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REGISTRY = Path(__file__).with_name("native_greek_benchmark_registry.json")
LABELS = ["Α", "Β", "Γ", "Δ", "Ε", "ΣΤ", "Ζ", "Η"]
HEADLINE_MCQ_BENCHMARKS = ("greekmmlu", "ilsp_medical_mcqa", "ilsp_mcqa_asep")
DIAGNOSTIC_MCQ_BENCHMARKS = ("plutus_qa",)


@dataclass
class MCQExample:
    benchmark: str
    example_id: str
    question: str
    choices: list[str]
    answer_index: int
    subject: str | None = None
    metadata: dict[str, Any] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--benchmarks", default="all", help="Comma-separated benchmark ids, or all native MCQ.")
    parser.add_argument("--model", required=True, help="LABEL=HF_MODEL_OR_PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=0, help="0 means full split.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--max-input-tokens", type=int, default=3072)
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=16,
        help="Number of answer candidates scored per forward pass.",
    )
    parser.add_argument(
        "--example-batch-size",
        type=int,
        default=16,
        help="Number of MCQ examples prepared before scoring. Keep modest for long-context datasets.",
    )
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_model(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise SystemExit("--model must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise SystemExit("--model must be LABEL=PATH")
    return label, path


def load_registry(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def selected_benchmarks(registry: dict[str, Any], raw: str) -> list[dict[str, Any]]:
    mcq = [b for b in registry["benchmarks"] if b.get("runner") == "run_native_greek_mcq_eval.py"]
    if raw == "all":
        return mcq
    wanted = {item.strip() for item in raw.split(",") if item.strip()}
    selected = [b for b in mcq if b["id"] in wanted]
    missing = wanted - {b["id"] for b in selected}
    if missing:
        raise SystemExit(f"Unknown or non-MCQ benchmark ids: {', '.join(sorted(missing))}")
    return selected


def _ensure_deps() -> None:
    missing = []
    for module in ["datasets", "torch", "transformers"]:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise SystemExit(f"Missing dependencies: {', '.join(missing)}")


def _load_dataset(spec: dict[str, Any]):
    from datasets import get_dataset_config_names, load_dataset

    dataset_id = spec["source"]
    revision = spec.get("revision")
    config = spec.get("config")
    if not config:
        try:
            configs = get_dataset_config_names(dataset_id, revision=revision)
        except Exception:
            configs = []
        if len(configs) == 1 and configs[0] != "default":
            config = configs[0]

    split = spec.get("split")
    candidates: list[str | None]
    if split == "test_or_train":
        candidates = ["test", "validation", "train", None]
    elif split == "test_or_default":
        candidates = ["test", "validation", None, "train"]
    elif split in (None, "default"):
        candidates = [None, "test", "validation", "train"]
    else:
        candidates = [split, None]

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            kwargs: dict[str, Any] = {}
            if revision:
                kwargs["revision"] = revision
            if candidate:
                kwargs["split"] = candidate
            ds = load_dataset(dataset_id, config, **kwargs)
            if isinstance(ds, dict):
                split_name = "test" if "test" in ds else next(iter(ds))
                selected = ds[split_name]
                return selected, split_name, str(getattr(selected, "_fingerprint", ""))
            return (
                ds,
                candidate or str(getattr(ds, "split", None) or "default"),
                str(getattr(ds, "_fingerprint", "")),
            )
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not load {dataset_id}: {last_error!r}")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return list(value)


def _correct_from_scores(scores: Iterable[Any]) -> int:
    values = list(scores)
    for index, score in enumerate(values):
        try:
            if float(score) == 1.0:
                return index
        except Exception:
            continue
    raise ValueError(f"No positive multiple-choice score in {values!r}")


def examples_from_row(spec: dict[str, Any], row: dict[str, Any], row_index: int) -> MCQExample | None:
    bid = spec["id"]
    if bid == "greekmmlu":
        choices = [str(item) for item in _as_list(row["choices"])]
        answer = int(row["answer"])
        question = str(row["question"])
        subject = str(row.get("subject") or row.get("sub_category") or row.get("category") or "")
        example_id = str(row.get("id") or f"{bid}:{row_index}")
    elif bid == "ilsp_medical_mcqa":
        choices = [str(item) for item in _as_list(row["multiple_choice_targets"])]
        answer = _correct_from_scores(row["multiple_choice_scores"])
        question = str(row["inputs"])
        subject = str(row.get("subject") or "")
        example_id = str(row.get("idx") or f"{bid}:{row_index}")
    elif bid == "ilsp_mcqa_asep":
        choices = [str(item) for item in _as_list(row["choices"])]
        answer = int(row["answer"])
        question = str(row["question"])
        subject = str(row.get("subject") or "")
        example_id = str(row.get("id") or f"{bid}:{row_index}")
    elif bid == "plutus_qa":
        choices = [str(item) for item in _as_list(row["choices"])]
        answer = int(row["gold"])
        context = str(row.get("text") or "").strip()
        query = str(row.get("query") or "").strip()
        question = f"Κείμενο:\n{context}\n\nΕρώτηση:\n{query}" if context else query
        subject = "finance"
        example_id = str(row.get("id") or f"{bid}:{row_index}")
    else:
        return None

    if answer < 0 or answer >= len(choices):
        raise ValueError(f"Bad answer index {answer} for {bid}:{row_index} with {len(choices)} choices")
    return MCQExample(
        benchmark=bid,
        example_id=example_id,
        question=question,
        choices=choices,
        answer_index=answer,
        subject=subject or None,
        metadata={key: row.get(key) for key in ("category", "sub_category", "level", "subject") if key in row},
    )


def build_prompt(example: MCQExample) -> str:
    lines = [
        "Απάντησε επιλέγοντας τη σωστή επιλογή.",
        "",
        "Ερώτηση:",
        example.question.strip(),
        "",
        "Επιλογές:",
    ]
    for index, choice in enumerate(example.choices):
        label = LABELS[index] if index < len(LABELS) else str(index + 1)
        lines.append(f"{label}. {choice}")
    lines.extend(["", "Σωστή απάντηση:"])
    return "\n".join(lines)


class ChoiceScorer:
    def __init__(self, model_path: str, *, dtype: str, max_input_tokens: int, trust_remote_code: bool) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        self.max_input_tokens = max_input_tokens
        self.device = next(self.model.parameters()).device

    def _candidate(self, example: MCQExample, prompt: str, choice: str) -> dict[str, Any]:
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        choice_ids = self.tokenizer(" " + choice.strip(), add_special_tokens=False)["input_ids"]
        if not prompt_ids or not choice_ids:
            return {
                "input_ids": [],
                "choice_start": 0,
                "choice_len": 0,
                "example": example,
                "score": {"sum_logprob": -math.inf, "avg_logprob": -math.inf, "num_tokens": 0},
            }

        max_prompt = max(1, self.max_input_tokens - len(choice_ids))
        prompt_ids = prompt_ids[-max_prompt:]
        return {
            "input_ids": prompt_ids + choice_ids,
            "choice_start": len(prompt_ids) - 1,
            "choice_len": len(choice_ids),
            "example": example,
        }

    def _score_candidates(self, candidates: list[dict[str, Any]]) -> list[dict[str, float]]:
        import torch

        scores: list[dict[str, float] | None] = [None] * len(candidates)
        live = [(index, candidate) for index, candidate in enumerate(candidates) if candidate.get("input_ids")]
        for index, candidate in enumerate(candidates):
            if not candidate.get("input_ids"):
                scores[index] = candidate["score"]
        if not live:
            return [score or {"sum_logprob": -math.inf, "avg_logprob": -math.inf, "num_tokens": 0} for score in scores]

        max_len = max(len(candidate["input_ids"]) for _, candidate in live)
        pad_id = self.tokenizer.pad_token_id
        input_ids = []
        attention_mask = []
        for _, candidate in live:
            ids = candidate["input_ids"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)

        tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        mask_tensor = torch.tensor(attention_mask, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            logits = self.model(input_ids=tensor, attention_mask=mask_tensor).logits
            log_probs = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
        labels = tensor[:, 1:]
        gathered = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

        for batch_row, (original_index, candidate) in enumerate(live):
            start = int(candidate["choice_start"])
            length = int(candidate["choice_len"])
            choice_log_probs = gathered[batch_row, start : start + length]
            total = float(choice_log_probs.sum().item())
            count = int(choice_log_probs.numel())
            scores[original_index] = {
                "sum_logprob": total,
                "avg_logprob": total / count if count else -math.inf,
                "num_tokens": count,
            }
        return [score or {"sum_logprob": -math.inf, "avg_logprob": -math.inf, "num_tokens": 0} for score in scores]

    def score_examples(self, examples: list[MCQExample], *, candidate_batch_size: int) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        example_ranges: list[tuple[int, int]] = []
        for example in examples:
            start = len(candidates)
            prompt = build_prompt(example)
            for choice in example.choices:
                candidates.append(self._candidate(example, prompt, choice))
            example_ranges.append((start, len(candidates)))

        all_scores: list[dict[str, float]] = []
        for start in range(0, len(candidates), candidate_batch_size):
            chunk = candidates[start : start + candidate_batch_size]
            all_scores.extend(self._score_candidates(chunk))

        outputs = []
        for example, (start, end) in zip(examples, example_ranges):
            choice_scores = all_scores[start:end]
            pred = max(range(len(choice_scores)), key=lambda idx: choice_scores[idx]["avg_logprob"])
            outputs.append(
                {
                    "pred_index": pred,
                    "correct": pred == example.answer_index,
                    "choice_scores": choice_scores,
                }
            )
        return outputs


def _sample_examples(examples: list[MCQExample], sample_size: int, random_state: int) -> list[MCQExample]:
    if sample_size <= 0 or sample_size >= len(examples):
        return examples
    import random

    rng = random.Random(random_state)
    indices = sorted(rng.sample(range(len(examples)), sample_size))
    return [examples[index] for index in indices]


def _summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    summary = []
    by_key: dict[tuple[str, str | None], list[dict[str, Any]]] = {}
    for row in rows:
        by_key.setdefault((row["benchmark"], None), []).append(row)
        subject = row.get("subject")
        if subject is not None:
            by_key.setdefault((row["benchmark"], str(subject)), []).append(row)
    for (benchmark, subject), group in sorted(by_key.items(), key=lambda item: (item[0][0], item[0][1] or "")):
        if subject is None and len(group) != sum(1 for row in rows if row["benchmark"] == benchmark):
            continue
        n = len(group)
        correct = sum(1 for row in group if row["correct"])
        choice_nlls = []
        correct_answer_neg_log2 = 0.0
        correct_answer_utf8_bytes = 0
        for row in group:
            normalized_scores = [float(score["avg_logprob"]) for score in row["choice_scores"]]
            maximum = max(normalized_scores)
            log_normalizer = maximum + math.log(
                sum(math.exp(score - maximum) for score in normalized_scores)
            )
            choice_nlls.append(log_normalizer - normalized_scores[int(row["answer_index"])])
            correct_score = row["choice_scores"][int(row["answer_index"])]
            correct_answer_neg_log2 += -float(correct_score["sum_logprob"]) / math.log(2.0)
            correct_answer_utf8_bytes += int(row["correct_answer_utf8_bytes"])
        summary.append(
            {
                "benchmark": benchmark,
                "subject": subject or "__all__",
                "n": n,
                "accuracy": correct / n if n else float("nan"),
                "correct": correct,
                "choice_nll": sum(choice_nlls) / n if n else float("nan"),
                "choice_nll_sum": sum(choice_nlls),
                "correct_answer_bpb": (
                    correct_answer_neg_log2 / correct_answer_utf8_bytes
                    if correct_answer_utf8_bytes
                    else float("nan")
                ),
                "correct_answer_neg_log2_sum": correct_answer_neg_log2,
                "correct_answer_utf8_bytes": correct_answer_utf8_bytes,
            }
        )
    all_rows = [row for row in summary if row["subject"] == "__all__"]
    headline_order = {benchmark: index for index, benchmark in enumerate(HEADLINE_MCQ_BENCHMARKS)}
    diagnostic_order = {benchmark: index for index, benchmark in enumerate(DIAGNOSTIC_MCQ_BENCHMARKS)}
    headline = sorted(
        [row for row in all_rows if row["benchmark"] in HEADLINE_MCQ_BENCHMARKS],
        key=lambda row: headline_order[row["benchmark"]],
    )
    diagnostics = sorted(
        [row for row in all_rows if row["benchmark"] not in HEADLINE_MCQ_BENCHMARKS],
        key=lambda row: (diagnostic_order.get(row["benchmark"], len(diagnostic_order)), row["benchmark"]),
    )

    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        total_n = sum(int(row["n"]) for row in items)
        total_correct = sum(int(row["correct"]) for row in items)
        accuracies = [float(row["accuracy"]) for row in items]
        total_choice_nll = sum(float(row["choice_nll_sum"]) for row in items)
        total_correct_answer_neg_log2 = sum(
            float(row["correct_answer_neg_log2_sum"]) for row in items
        )
        total_correct_answer_bytes = sum(
            int(row["correct_answer_utf8_bytes"]) for row in items
        )
        return {
            "n_tasks": len(items),
            "total_n": total_n,
            "total_correct": total_correct,
            "macro_accuracy": sum(accuracies) / len(accuracies) if accuracies else None,
            "micro_accuracy": total_correct / total_n if total_n else None,
            "macro_choice_nll": (
                sum(float(row["choice_nll"]) for row in items) / len(items)
                if items
                else None
            ),
            "micro_choice_nll": total_choice_nll / total_n if total_n else None,
            "macro_correct_answer_bpb": (
                sum(float(row["correct_answer_bpb"]) for row in items) / len(items)
                if items
                else None
            ),
            "micro_correct_answer_bpb": (
                total_correct_answer_neg_log2 / total_correct_answer_bytes
                if total_correct_answer_bytes
                else None
            ),
        }

    aggregate_report = {
        "schema": "native-greek-mcq-aggregate-v1",
        "headline_policy": {
            "headline_benchmarks": list(HEADLINE_MCQ_BENCHMARKS),
            "diagnostic_benchmarks": list(DIAGNOSTIC_MCQ_BENCHMARKS),
            "note": "Headline excludes Plutus QA; Plutus is diagnostic/domain-specific.",
        },
        "headline": aggregate(headline),
        "headline_with_diagnostics": aggregate(headline + diagnostics),
        "diagnostics": aggregate(diagnostics),
    }
    return summary, headline, diagnostics, aggregate_report


def main() -> None:
    args = parse_args()
    _ensure_deps()
    model_label, model_path = parse_model(args.model)
    registry = load_registry(args.registry)
    specs = selected_benchmarks(registry, args.benchmarks)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "schema": "native-greek-mcq-run-v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_label": model_label,
        "model_path": model_path,
        "benchmarks": [spec["id"] for spec in specs],
        "sample_size": args.sample_size,
        "random_state": args.random_state,
        "dtype": args.dtype,
        "max_input_tokens": args.max_input_tokens,
        "candidate_batch_size": args.candidate_batch_size,
        "example_batch_size": args.example_batch_size,
        "registry": str(args.registry.resolve()),
        "benchmark_specs": [
            {
                "id": spec["id"],
                "source": spec["source"],
                "revision": spec.get("revision"),
                "config": spec.get("config"),
                "split": spec.get("split"),
            }
            for spec in specs
        ],
        "dataset_bindings": [],
        "metrics": [
            "official_zero_shot_accuracy_from_average_answer_token_logprob",
            "multiple_choice_cross_entropy_from_normalized_average_answer_token_logprob",
            "correct_answer_continuation_bits_per_utf8_byte_from_sum_logprob",
        ],
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")

    scorer = ChoiceScorer(
        model_path,
        dtype=args.dtype,
        max_input_tokens=args.max_input_tokens,
        trust_remote_code=args.trust_remote_code,
    )

    rows = []
    dataset_bindings = []
    for spec in specs:
        dataset, split, fingerprint = _load_dataset(spec)
        dataset_bindings.append(
            {
                "id": spec["id"],
                "source": spec["source"],
                "revision": spec.get("revision"),
                "config": spec.get("config"),
                "resolved_split": split,
                "fingerprint": fingerprint,
                "rows_before_sampling": len(dataset),
            }
        )
        examples = []
        for row_index, row in enumerate(dataset):
            example = examples_from_row(spec, row, row_index)
            if example is not None:
                examples.append(example)
        examples = _sample_examples(examples, args.sample_size, args.random_state)
        print(f"[{spec['id']}] split={split} examples={len(examples)}", flush=True)
        for start in range(0, len(examples), args.example_batch_size):
            chunk = examples[start : start + args.example_batch_size]
            scored_chunk = scorer.score_examples(chunk, candidate_batch_size=args.candidate_batch_size)
            for example, scored in zip(chunk, scored_chunk):
                rows.append(
                    {
                        "model": model_label,
                        "benchmark": example.benchmark,
                        "example_id": example.example_id,
                        "subject": example.subject,
                        "answer_index": example.answer_index,
                        "pred_index": scored["pred_index"],
                        "correct": scored["correct"],
                        "choice_scores": scored["choice_scores"],
                        "num_choices": len(example.choices),
                        "correct_answer_utf8_bytes": len(
                            example.choices[example.answer_index].encode("utf-8")
                        ),
                        "metadata": example.metadata or {},
                    }
                )
            done = min(start + len(chunk), len(examples))
            if done % 100 == 0 or done == len(examples):
                print(f"[{spec['id']}] {done}/{len(examples)}", flush=True)

    metadata["dataset_bindings"] = dataset_bindings
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )

    predictions_path = args.output_dir / f"{model_label}_native_mcq_predictions.jsonl"
    with predictions_path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary, headline, diagnostics, aggregate_report = _summarize(rows)
    summary_path = args.output_dir / f"{model_label}_native_mcq_summary.csv"
    with summary_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "benchmark",
                "subject",
                "n",
                "accuracy",
                "correct",
                "choice_nll",
                "choice_nll_sum",
                "correct_answer_bpb",
                "correct_answer_neg_log2_sum",
                "correct_answer_utf8_bytes",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)
    (args.output_dir / f"{model_label}_native_mcq_headline.json").write_text(
        json.dumps(headline, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / f"{model_label}_native_mcq_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / f"{model_label}_native_mcq_aggregate.json").write_text(
        json.dumps(aggregate_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )

    print("\nHeadline:")
    for row in headline:
        print(
            f"{row['benchmark']}: n={row['n']} acc={row['accuracy']:.4f} "
            f"choice_nll={row['choice_nll']:.4f} correct_bpb={row['correct_answer_bpb']:.4f}",
            flush=True,
        )
    if diagnostics:
        print("\nDiagnostics:")
        for row in diagnostics:
            print(f"{row['benchmark']}: n={row['n']} acc={row['accuracy']:.4f}", flush=True)


if __name__ == "__main__":
    main()
