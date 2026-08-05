#!/usr/bin/env python3
"""Freeze full and decontaminated GreekMMLU metrics for an HF baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(rows: list[dict]) -> dict:
    correct = 0
    choice_nll = 0.0
    answer_bits = 0.0
    answer_bytes = 0
    for row in rows:
        correct += int(row["correct"])
        scores = [float(value["avg_logprob"]) for value in row["choice_scores"]]
        maximum = max(scores)
        normalizer = maximum + math.log(sum(math.exp(value - maximum) for value in scores))
        answer = int(row["answer_index"])
        choice_nll += normalizer - scores[answer]
        answer_bits += -float(row["choice_scores"][answer]["sum_logprob"]) / math.log(2)
        answer_bytes += int(row["correct_answer_utf8_bytes"])
    return {
        "n": len(rows),
        "accuracy": correct / len(rows),
        "choice_nll": choice_nll / len(rows),
        "correct_answer_bpb": answer_bits / answer_bytes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--clean-subset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predictions = args.evaluation_root / f"{args.model_label}_native_mcq_predictions.jsonl"
    rows = [json.loads(line) for line in predictions.open() if line.strip()]
    if len(rows) != 16_632:
        raise ValueError("GreekMMLU full prediction count drift")
    clean = json.loads(args.clean_subset_manifest.read_text())
    ids_path = Path(clean["clean_example_ids"]["path"])
    if sha(ids_path) != clean["clean_example_ids"]["sha256"]:
        raise ValueError("clean subset identity drift")
    clean_ids = {line.strip() for line in ids_path.open() if line.strip()}
    clean_rows = [row for row in rows if str(row["example_id"]) in clean_ids]
    if len(clean_rows) != len(clean_ids):
        raise ValueError("clean GreekMMLU subset is incomplete")
    artifacts = {}
    for path in sorted(item for item in args.evaluation_root.rglob("*") if item.is_file()):
        artifacts[str(path.relative_to(args.evaluation_root))] = {
            "bytes": path.stat().st_size,
            "sha256": sha(path),
        }
    payload = {
        "schema_version": "apertus_full_8b_initial_greekmmlu_v1",
        "status": "completed",
        "model": str(args.model.resolve()),
        "metrics": {"full": metrics(rows), "decontaminated": metrics(clean_rows)},
        "clean_subset_manifest_sha256": sha(args.clean_subset_manifest),
        "artifacts": artifacts,
    }
    temporary = Path(str(args.output) + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"ok": True, "clean_n": len(clean_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
