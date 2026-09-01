#!/usr/bin/env python3
"""Build canonical native-Greek benchmark queries for contamination scans.

The registry path preserves the historical MCQ workflow.  The frozen-examples
path is the production interface for multi-benchmark audits: it consumes the
exact examples used by evaluation and removes evaluator-authored scaffolding
from OYXOY before constructing the matching surfaces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    parser.add_argument(
        "--frozen-examples-jsonl",
        type=Path,
        help="Use the exact frozen evaluation rows instead of loading the legacy registry.",
    )
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def load_eval_helpers(eval_dir: Path):
    sys.path.insert(0, str(eval_dir))
    import run_native_greek_mcq_eval as native_mcq  # type: ignore

    return native_mcq


def _between(text: str, start: str, end: str | None = None) -> str:
    if start not in text:
        raise ValueError(f"missing expected prompt marker: {start!r}")
    value = text.split(start, 1)[1]
    if end is not None:
        if end not in value:
            raise ValueError(f"missing expected prompt marker: {end!r}")
        value = value.split(end, 1)[0]
    return value.strip()


def query_from_frozen_row(row: dict[str, Any]) -> dict[str, Any]:
    """Map an evaluation row to human-authored contamination surfaces.

    OYXOY questions contain Greek evaluator instructions that never appeared in
    the upstream lexical/NLI resource.  Matching those instructions would be a
    false-negative-prone test of our own prompt template.  Instead we match the
    source premise/hypothesis, usage examples and definitions.  Every strong
    rule still requires two nearby surfaces.
    """
    benchmark = str(row["benchmark"])
    question = str(row["question"])
    choices = [str(value) for value in row["choices"]]
    answer_index = int(row["answer_index"])
    example_id = str(row["example_id"])
    group_id = row.get("group_id")
    metadata = dict(row.get("metadata") or {})
    query_kind = "mcq_question_correct_answer"
    evaluation_unit_id = example_id
    discount_example_ids = [example_id]

    if benchmark == "oyxoy_nli":
        premise = _between(question, "Πρόταση αναφοράς:\n", "\n\nΥπόθεση:\n")
        hypothesis = _between(question, "\n\nΥπόθεση:\n", "\n\nΙσχύει η σχέση")
        question, choices, answer_index = premise, [hypothesis], 0
        query_kind = "nli_premise_hypothesis"
        evaluation_unit_id = str(group_id)
        discount_example_ids = [
            f"{group_id}:Unknown",
            f"{group_id}:Entailment",
            f"{group_id}:Contradiction",
        ]
    elif benchmark == "oyxoy_wsd_definition":
        usage = _between(question, "\nΧρήση: ", "\nΠοιος ορισμός")
        question = usage
        query_kind = "lexical_usage_correct_definition"
    elif benchmark == "oyxoy_wic":
        usage1 = _between(question, "\nΧρήση 1: ", "\nΧρήση 2: ")
        usage2 = _between(question, "\nΧρήση 2: ", "\nΧρησιμοποιείται η λέξη")
        question, choices, answer_index = usage1, [usage2], 0
        query_kind = "lexical_usage_pair"
    elif benchmark == "oyxoy_metaphor":
        usage = _between(question, "\nΧρήση: ", "\nΕίναι μεταφορική")
        definition = str(metadata.get("definition") or "").strip()
        if not definition:
            raise ValueError(f"missing OYXOY definition for {example_id}")
        question, choices, answer_index = usage, [definition], 0
        query_kind = "lexical_usage_definition"

    answer_text = choices[answer_index]
    return {
        "schema": "greek-benchmark-decontam-query-v2",
        "benchmark": benchmark,
        "example_id": example_id,
        "evaluation_unit_id": evaluation_unit_id,
        "discount_example_ids": discount_example_ids,
        "source_group_id": None if group_id is None else str(group_id),
        "query_kind": query_kind,
        "subject": row.get("subject"),
        "question": question,
        "choices": choices,
        "answer_index": answer_index,
        "answer_text": answer_text,
        "metadata": metadata,
        "raw_row_sha256": sha256_json(row),
        "surfaces": {
            "question": question,
            "question_all_choices": f"{question}\n" + "\n".join(choices),
            "question_correct_answer": f"{question}\n{answer_text}",
        },
    }


def build_from_frozen(args: argparse.Namespace) -> dict[str, Any]:
    wanted = {item.strip() for item in args.benchmarks.split(",") if item.strip()}
    select_all = not wanted or wanted == {"all"}
    rows: list[dict[str, Any]] = []
    with args.frozen_examples_jsonl.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{args.frozen_examples_jsonl}:{line_no}: {exc}") from exc
            if select_all or row.get("benchmark") in wanted:
                rows.append(row)
    present = {str(row["benchmark"]) for row in rows}
    if not select_all and (wanted - present):
        raise SystemExit(f"missing frozen benchmark rows: {', '.join(sorted(wanted - present))}")

    # OYXOY NLI has three evaluator decisions for one source pair.  Emit one
    # source-overlap query and map it back to all three scored decisions.
    queries: list[dict[str, Any]] = []
    seen_units: set[tuple[str, str]] = set()
    counts: dict[str, dict[str, Any]] = {}
    for row in rows:
        query = query_from_frozen_row(row)
        key = (query["benchmark"], query["evaluation_unit_id"])
        if key in seen_units:
            continue
        seen_units.add(key)
        queries.append(query)
        block = counts.setdefault(query["benchmark"], {"items": 0, "scored_examples": 0, "query_kinds": {}})
        block["items"] += 1
        block["scored_examples"] += len(query["discount_example_ids"])
        kind = query["query_kind"]
        block["query_kinds"][kind] = block["query_kinds"].get(kind, 0) + 1

    atomic_write_text(
        args.output_jsonl,
        "".join(json.dumps(query, ensure_ascii=False, sort_keys=True) + "\n" for query in queries),
    )
    return {
        "schema": "greek-benchmark-decontam-query-summary-v2",
        "input_mode": "frozen_examples",
        "frozen_examples_jsonl": str(args.frozen_examples_jsonl),
        "frozen_examples_sha256": hashlib.sha256(args.frozen_examples_jsonl.read_bytes()).hexdigest(),
        "benchmarks": counts,
        "total_items": len(queries),
        "total_scored_examples": sum(len(query["discount_example_ids"]) for query in queries),
        "output_jsonl": str(args.output_jsonl),
        "output_sha256": hashlib.sha256(args.output_jsonl.read_bytes()).hexdigest(),
    }


def main() -> None:
    args = parse_args()
    if args.frozen_examples_jsonl is not None:
        summary = build_from_frozen(args)
        atomic_write_text(
            args.summary_json,
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return
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
                "revision": spec.get("revision"),
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
        "registry_sha256": hashlib.sha256(args.registry.read_bytes()).hexdigest(),
        "eval_dir": str(args.eval_dir),
        "benchmarks": counts,
        "total_items": n_items,
        "total_surfaces": n_surfaces,
        "output_jsonl": str(args.output_jsonl),
        "output_jsonl_sha256": hashlib.sha256(args.output_jsonl.read_bytes()).hexdigest(),
    }
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
