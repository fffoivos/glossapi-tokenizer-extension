#!/usr/bin/env python3
"""Materialize deterministic, checkpoint-independent native-Greek eval examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


DEMO_LABELS = {"Α": 0, "Β": 1, "Γ": 2, "Δ": 3}
OYXOY_LABELS = ("Unknown", "Entailment", "Contradiction")
OYXOY_GREEK_LABELS = {
    "Unknown": "ουδέτερη σχέση",
    "Entailment": "συνεπαγωγή",
    "Contradiction": "αντίφαση",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--oyxoy-root", type=Path, required=True)
    parser.add_argument("--gpcr-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def demos_choices(value: str) -> list[str]:
    matches = list(re.finditer(r"(?:^|\n\s*\n)([ΑΒΓΔ])\.\s*", value))
    if [match.group(1) for match in matches] != ["Α", "Β", "Γ", "Δ"]:
        raise ValueError("DemosQA answer serialization drift")
    choices = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(value)
        choices.append(value[match.end() : end].strip().strip('"').strip())
    return choices


def onehot_index(values: Iterable[Any]) -> int:
    indices = [index for index, value in enumerate(values) if float(value) == 1.0]
    if len(indices) != 1:
        raise ValueError(f"expected one positive score, got {indices}")
    return indices[0]


def record(
    benchmark: str,
    example_id: str,
    question: str,
    choices: list[str],
    answer: int,
    *,
    subject: str | None = None,
    group_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not question.strip() or len(choices) < 2 or not 0 <= answer < len(choices):
        raise ValueError(f"invalid example {benchmark}:{example_id}")
    return {
        "benchmark": benchmark,
        "example_id": str(example_id),
        "question": question.strip(),
        "choices": [str(choice).strip() for choice in choices],
        "answer_index": int(answer),
        "subject": subject,
        "group_id": group_id,
        "metadata": metadata or {},
    }


def load_hf_examples(contract: dict[str, Any], gpcr_parquet: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from datasets import load_dataset

    specs = {row["id"]: row for row in contract["benchmarks"]}
    rows: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    for benchmark in ("demosqa", "medical_mcqa", "asep_mcqa"):
        spec = specs[benchmark]
        ds = load_dataset(
            spec["source"],
            split=spec["split"],
            revision=spec["revision"],
        )
        if len(ds) != int(spec["expected_rows"]):
            raise ValueError(f"{benchmark} row-count drift: {len(ds)}")
        bindings.append(
            {
                "benchmark": benchmark,
                "source": spec["source"],
                "revision": spec["revision"],
                "split": spec["split"],
                "rows": len(ds),
                "fingerprint": str(ds._fingerprint),
            }
        )
        for index, item in enumerate(ds):
            if benchmark == "demosqa":
                rows.append(
                    record(
                        benchmark,
                        str(item["id"]),
                        str(item["question"]),
                        demos_choices(str(item["answers"])),
                        DEMO_LABELS[str(item["best_answer_index"]).strip()],
                    )
                )
            elif benchmark == "medical_mcqa":
                rows.append(
                    record(
                        benchmark,
                        str(item.get("idx", index)),
                        str(item["inputs"]),
                        [str(value) for value in item["multiple_choice_targets"]],
                        onehot_index(item["multiple_choice_scores"]),
                        subject=str(item.get("subject", "")) or None,
                    )
                )
            else:
                rows.append(
                    record(
                        benchmark,
                        str(item.get("id", index)),
                        str(item["question"]),
                        [str(value) for value in item["choices"]],
                        int(item["answer"]),
                        subject=str(item.get("subject", "")) or None,
                    )
                )

    gpcr = load_dataset("parquet", data_files=str(gpcr_parquet), split="train")
    spec = specs["gpcr"]
    if len(gpcr) != int(spec["expected_rows"]):
        raise ValueError(f"gpcr row-count drift: {len(gpcr)}")
    bindings.append(
        {
            "benchmark": "gpcr",
            "source": spec["source"],
            "revision": spec["revision"],
            "split": spec["split"],
            "rows": len(gpcr),
            "parquet": str(gpcr_parquet.resolve()),
            "parquet_sha256": sha256_file(gpcr_parquet),
        }
    )
    for index, item in enumerate(gpcr):
        rows.append(
            record(
                "gpcr",
                str(item.get("id", index)),
                str(item["prompt"]),
                [str(item["solution0"]), str(item["solution1"])],
                int(item["label"]),
                subject=str(item.get("domain", "")) or None,
                metadata={"culturally_specific": int(item.get("culturally specific", 0))},
            )
        )
    return rows, bindings


def load_oyxoy_examples(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    nli_gold = root / "src/nli/gold.json"
    nli_fracas = root / "src/nli/FraCaS.json"
    wordsense = root / "src/wordsense/dataset.json"
    for path in (nli_gold, nli_fracas, wordsense):
        if not path.is_file():
            raise FileNotFoundError(path)

    rows: list[dict[str, Any]] = []
    nli_count = 0
    for source, path in (("gold", nli_gold), ("fracas", nli_fracas)):
        samples = json.loads(path.read_text())["samples"]
        for index, item in enumerate(samples):
            group_id = f"{source}:{index}"
            labels = set(item["labels"])
            for label in OYXOY_LABELS:
                question = (
                    f"Πρόταση αναφοράς:\n{item['premise']}\n\n"
                    f"Υπόθεση:\n{item['hypothesis']}\n\n"
                    f"Ισχύει η σχέση «{OYXOY_GREEK_LABELS[label]}»;"
                )
                rows.append(
                    record(
                        "oyxoy_nli",
                        f"{group_id}:{label}",
                        question,
                        ["Όχι", "Ναι"],
                        int(label in labels),
                        subject=label,
                        group_id=group_id,
                        metadata={"source": source, "tags": item.get("tags", [])},
                    )
                )
            nli_count += 1

    entries = json.loads(wordsense.read_text())["entries"]
    wsd_count = 0
    wic_count = 0
    metaphor_count = 0
    for entry_index, entry in enumerate(entries):
        lemma = str(entry["lemma"])
        senses = entry["senses"]
        definitions = [str(sense["definition"]) for sense in senses]
        flat_examples: list[tuple[int, str]] = []
        for sense_index, sense in enumerate(senses):
            for example_index, example in enumerate(sense["examples"]):
                flat_examples.append((sense_index, str(example)))
                # Six upstream entries have only one sense. They cannot test
                # sense selection and would add guaranteed-correct examples.
                if len(definitions) >= 2:
                    rows.append(
                        record(
                            "oyxoy_wsd_definition",
                            f"{entry_index}:{sense_index}:{example_index}",
                            f"Λέξη: {lemma}\nΧρήση: {example}\nΠοιος ορισμός αντιστοιχεί στη σημασία της λέξης;",
                            definitions,
                            sense_index,
                            subject="definition_selection",
                            group_id=str(entry_index),
                            metadata={"lemma": lemma},
                        )
                    )
                    wsd_count += 1

        for left in range(len(flat_examples)):
            left_sense, left_example = flat_examples[left]
            for right in range(left + 1, len(flat_examples)):
                right_sense, right_example = flat_examples[right]
                rows.append(
                    record(
                        "oyxoy_wic",
                        f"{entry_index}:{left}:{right}",
                        (
                            f"Λέξη: {lemma}\nΧρήση 1: {left_example}\nΧρήση 2: {right_example}\n"
                            "Χρησιμοποιείται η λέξη με την ίδια σημασία στις δύο φράσεις;"
                        ),
                        ["Όχι", "Ναι"],
                        int(left_sense == right_sense),
                        subject="same_sense",
                        group_id=str(entry_index),
                        metadata={"lemma": lemma},
                    )
                )
                wic_count += 1

        if any("μτφ" in definition for definition in definitions):
            for flat_index, (sense_index, example) in enumerate(flat_examples):
                rows.append(
                    record(
                        "oyxoy_metaphor",
                        f"{entry_index}:{flat_index}",
                        f"Λέξη: {lemma}\nΧρήση: {example}\nΕίναι μεταφορική η χρήση της λέξης;",
                        ["Όχι", "Ναι"],
                        int("μτφ" in definitions[sense_index]),
                        subject="metaphor",
                        group_id=str(entry_index),
                        metadata={"lemma": lemma, "definition": definitions[sense_index]},
                    )
                )
                metaphor_count += 1

    measured = {
        "nli_samples": nli_count,
        "nli_binary_decisions": nli_count * len(OYXOY_LABELS),
        "entries": len(entries),
        "senses": sum(len(entry["senses"]) for entry in entries),
        "wsd_examples": wsd_count,
        "wic_unordered_pairs": wic_count,
        "metaphor_examples": metaphor_count,
    }
    expected = {
        "nli_samples": 1762,
        "nli_binary_decisions": 5286,
        "entries": 2326,
        "senses": 6895,
        "wsd_examples": 14398,
        "wic_unordered_pairs": 58831,
        "metaphor_examples": 3015,
    }
    if measured != expected:
        raise ValueError(f"OYXOY source drift: measured={measured} expected={expected}")
    return rows, {
        "benchmark": "oyxoy",
        "root": str(root.resolve()),
        "files": {str(path.relative_to(root)): sha256_file(path) for path in (nli_gold, nli_fracas, wordsense)},
        "counts": measured,
    }


def main() -> int:
    args = parse_args()
    contract = json.loads(args.contract.read_text())
    if contract.get("schema_version") != "apertus_full8_native_greek_3cp_contract_v1":
        raise ValueError("contract schema drift")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    hf_rows, hf_bindings = load_hf_examples(contract, args.gpcr_parquet)
    oyxoy_rows, oyxoy_binding = load_oyxoy_examples(args.oyxoy_root)
    rows = hf_rows + oyxoy_rows
    examples_path = args.output_dir / "examples.jsonl"
    with examples_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["benchmark"]] = counts.get(row["benchmark"], 0) + 1
    manifest = {
        "schema_version": "apertus_full8_native_greek_frozen_examples_v1",
        "status": "complete_except_protipa_access",
        "contract": {"path": str(args.contract.resolve()), "sha256": sha256_file(args.contract)},
        "examples": {"path": str(examples_path.resolve()), "sha256": sha256_file(examples_path), "rows": len(rows)},
        "counts": counts,
        "bindings": hf_bindings + [oyxoy_binding],
        "blocked": {
            "protipa_text_only": "current Hugging Face token has not been approved for the manual dataset gate"
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "rows": len(rows), "counts": counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
