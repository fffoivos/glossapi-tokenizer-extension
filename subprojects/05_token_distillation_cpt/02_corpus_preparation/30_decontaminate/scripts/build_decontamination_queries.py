#!/usr/bin/env python3
"""Build canonical benchmark query surfaces for the 5B decontamination scan."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


# parents: [0]=scripts [1]=30_decontaminate [2]=02_corpus_preparation
# [3]=05_token_distillation_cpt [4]=subprojects [5]=<repo root>.
# (Was parents[4] when this lived at 05/decontamination/scripts/; the 2026-06-09
# fold added the 02_corpus_preparation/30_decontaminate level, so it is now [5].)
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_REGISTRY = (
    DEFAULT_REPO_ROOT
    / "subprojects/03_apertus_extension_and_embedding_adaptation/"
    "03_4_implementation_experiments/init_bakeoff/eval/native_greek_benchmark_registry.json"
)
DEFAULT_EVAL_DIR = DEFAULT_REGISTRY.parent
MCQ_BENCHMARKS = ("greekmmlu", "ilsp_medical_mcqa", "ilsp_mcqa_asep", "plutus_qa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--benchmarks", default=",".join(MCQ_BENCHMARKS))
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_eval_helpers(eval_dir: Path):
    sys.path.insert(0, str(eval_dir))
    import run_native_greek_mcq_eval as native_mcq  # type: ignore

    return native_mcq


def main() -> None:
    args = parse_args()
    native_mcq = load_eval_helpers(args.eval_dir)
    registry = native_mcq.load_registry(args.registry)
    wanted = {item.strip() for item in args.benchmarks.split(",") if item.strip()}
    specs = [spec for spec in native_mcq.selected_benchmarks(registry, "all") if spec["id"] in wanted]
    missing = wanted - {spec["id"] for spec in specs}
    if missing:
        raise SystemExit(f"missing benchmark specs: {', '.join(sorted(missing))}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    counts: dict[str, dict[str, Any]] = {}
    n_items = 0
    n_surfaces = 0
    generated_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with args.output_jsonl.open("w", encoding="utf-8") as out:
        for spec in specs:
            dataset, split = native_mcq._load_dataset(spec)
            benchmark = spec["id"]
            counts[benchmark] = {
                "source": spec["source"],
                "config": spec.get("config"),
                "split_requested": spec.get("split"),
                "split_loaded": split,
                "items": 0,
                "surfaces": 0,
            }
            for row_index, row in enumerate(dataset):
                raw_row = dict(row)
                example = native_mcq.examples_from_row(spec, raw_row, row_index)
                if example is None:
                    continue
                answer_text = example.choices[example.answer_index]
                choices_text = "\n".join(str(choice) for choice in example.choices)
                prompt = native_mcq.build_prompt(example)
                surfaces = {
                    "question": example.question,
                    "question_all_choices": f"{example.question}\n{choices_text}",
                    "question_correct_answer": f"{example.question}\n{answer_text}",
                    "eval_prompt": prompt,
                }
                item = {
                    "schema": "greek-mcq-decontam-query-v1",
                    "generated_utc": generated_utc,
                    "benchmark": benchmark,
                    "example_id": example.example_id,
                    "row_index": row_index,
                    "subject": example.subject,
                    "question": example.question,
                    "choices": example.choices,
                    "answer_index": example.answer_index,
                    "answer_text": answer_text,
                    "metadata": example.metadata or {},
                    "raw_row_sha256": sha256_json(raw_row),
                    "surfaces": surfaces,
                }
                out.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
                counts[benchmark]["items"] += 1
                counts[benchmark]["surfaces"] += len(surfaces)
                n_items += 1
                n_surfaces += len(surfaces)

    summary = {
        "schema": "greek-mcq-decontam-query-summary-v1",
        "generated_utc": generated_utc,
        "registry": str(args.registry),
        "eval_dir": str(args.eval_dir),
        "benchmarks": counts,
        "total_items": n_items,
        "total_surfaces": n_surfaces,
        "output_jsonl": str(args.output_jsonl),
    }
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
