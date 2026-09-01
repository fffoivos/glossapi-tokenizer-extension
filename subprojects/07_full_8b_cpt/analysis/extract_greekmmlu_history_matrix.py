#!/usr/bin/env python3
"""Extract the frozen clean GreekMMLU correctness history into a compact NPZ.

This is deliberately a data-local CSCS analysis: it validates every frozen
prediction payload against the existing drift receipt, reads only the clean
example IDs, and emits a compact matrix suitable for historical simulations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked(path: Path, binding: dict) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(binding["bytes"]):
        raise ValueError(f"byte-size drift: {path}")
    if sha256_file(path) != str(binding["sha256"]):
        raise ValueError(f"sha256 drift: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drift-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    receipt = json.loads(args.drift_receipt.read_text(encoding="utf-8"))
    clean_binding = receipt["clean_subset"]["ids"]
    clean_path = Path(clean_binding["path"])
    checked(clean_path, clean_binding)
    example_ids = [line.strip() for line in clean_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(example_ids) != int(receipt["clean_subset"]["n"]):
        raise ValueError("clean ID count drift")
    index = {example_id: i for i, example_id in enumerate(example_ids)}

    bindings = sorted(receipt["bindings"], key=lambda row: int(row["iteration"]))
    states = np.zeros((len(bindings), len(example_ids)), dtype=np.bool_)
    choices = np.zeros((len(bindings), len(example_ids)), dtype=np.int8)
    choice_nll = np.zeros((len(bindings), len(example_ids)), dtype=np.float64)
    correct_answer_bits = np.zeros((len(bindings), len(example_ids)), dtype=np.float64)
    subjects = np.full(len(example_ids), "Unknown", dtype="U128")
    levels = np.full(len(example_ids), "Unknown", dtype="U64")
    answer_indices = np.full(len(example_ids), -1, dtype=np.int8)
    num_choices = np.full(len(example_ids), -1, dtype=np.int8)
    correct_answer_bytes = np.full(len(example_ids), -1, dtype=np.int16)
    seen_subject = np.zeros(len(example_ids), dtype=np.bool_)

    for t, binding in enumerate(bindings):
        prediction = Path(binding["predictions"]["path"])
        checked(prediction, binding["predictions"])
        seen = np.zeros(len(example_ids), dtype=np.bool_)
        with prediction.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                i = index.get(str(row["example_id"]))
                if i is None:
                    continue
                if seen[i]:
                    raise ValueError(f"duplicate clean example: {row['example_id']}")
                seen[i] = True
                states[t, i] = bool(row["correct"])
                choices[t, i] = int(row["pred_index"])
                scores = [float(value["avg_logprob"]) for value in row["choice_scores"]]
                maximum = max(scores)
                normalizer = maximum + math.log(sum(math.exp(value - maximum) for value in scores))
                answer = int(row["answer_index"])
                choice_nll[t, i] = normalizer - scores[answer]
                correct_answer_bits[t, i] = -float(row["choice_scores"][answer]["sum_logprob"]) / math.log(2)
                subject = str(row.get("subject") or row.get("metadata", {}).get("subject") or "Unknown")
                level = str(row.get("metadata", {}).get("level") or "Unknown")
                choices_count = int(row["num_choices"])
                answer_bytes = int(row["correct_answer_utf8_bytes"])
                if not seen_subject[i]:
                    subjects[i] = subject
                    levels[i] = level
                    answer_indices[i] = answer
                    num_choices[i] = choices_count
                    correct_answer_bytes[i] = answer_bytes
                    seen_subject[i] = True
                elif (
                    subjects[i] != subject
                    or levels[i] != level
                    or int(answer_indices[i]) != answer
                    or int(num_choices[i]) != choices_count
                    or int(correct_answer_bytes[i]) != answer_bytes
                ):
                    raise ValueError(f"frozen example metadata drift: {row['example_id']}")
        if not bool(seen.all()):
            raise ValueError(f"missing clean examples at iteration {binding['iteration']}: {int((~seen).sum())}")

    expected_iterations = np.array([int(row["iteration"]) for row in bindings], dtype=np.int32)
    observed_correct = states.sum(axis=1)
    receipt_correct = np.array([int(row["correct"]) for row in receipt["checkpoint_table"]], dtype=np.int32)
    if not np.array_equal(observed_correct, receipt_correct):
        raise ValueError("checkpoint correctness totals do not reproduce the frozen receipt")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output) + ".partial.npz")
    np.savez_compressed(
        temporary,
        states=states,
        choices=choices,
        choice_nll=choice_nll,
        correct_answer_bits=correct_answer_bits,
        correct_answer_bytes=correct_answer_bytes,
        iterations=expected_iterations,
        example_ids=np.asarray(example_ids, dtype="U64"),
        subjects=subjects,
        levels=levels,
        answer_indices=answer_indices,
        num_choices=num_choices,
        source_receipt_sha256=np.asarray([sha256_file(args.drift_receipt)], dtype="U64"),
    )
    temporary.replace(args.output)
    print(json.dumps({
        "ok": True,
        "output": str(args.output.resolve()),
        "checkpoints": int(states.shape[0]),
        "questions": int(states.shape[1]),
        "accuracy_start": float(states[0].mean()),
        "accuracy_end": float(states[-1].mean()),
        "sha256": sha256_file(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
