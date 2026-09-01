#!/usr/bin/env python3
"""Bind one frozen native-GreekMMLU result to its exact checkpoint export."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_prediction_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = 0
    choice_nll = 0.0
    correct_neg_log2 = 0.0
    correct_bytes = 0
    for row in rows:
        correct += int(bool(row["correct"]))
        scores = [float(item["avg_logprob"]) for item in row["choice_scores"]]
        maximum = max(scores)
        normalizer = maximum + math.log(sum(math.exp(value - maximum) for value in scores))
        answer = int(row["answer_index"])
        choice_nll += normalizer - scores[answer]
        correct_neg_log2 += -float(row["choice_scores"][answer]["sum_logprob"]) / math.log(2)
        correct_bytes += int(row["correct_answer_utf8_bytes"])
    n = len(rows)
    return {
        "n": n,
        "accuracy": correct / n if n else None,
        "choice_nll": choice_nll / n if n else None,
        "correct_answer_bpb": correct_neg_log2 / correct_bytes if correct_bytes else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-receipt", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--evaluation-namespace", required=True)
    parser.add_argument("--clean-subset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    export = read_json(args.export_receipt)
    if (
        export.get("schema_version") != "native_greekmmlu_exact_checkpoint_export_v1"
        or export.get("status") != "completed"
        or export.get("ready_for_frozen_native_greekmmlu") is not True
    ):
        raise ValueError("checkpoint export is not ready for native GreekMMLU")
    evaluation_root = args.evaluation_root.resolve()
    stem = evaluation_root / f"{args.model_label}_native_mcq"
    artifacts = {
        "aggregate": Path(str(stem) + "_aggregate.json"),
        "headline": Path(str(stem) + "_headline.json"),
        "predictions": Path(str(stem) + "_predictions.jsonl"),
        "summary": Path(str(stem) + "_summary.csv"),
        "run_metadata": evaluation_root / "run_metadata.json",
    }
    for path in artifacts.values():
        if not path.is_file():
            raise ValueError(f"missing GreekMMLU evaluation artifact: {path}")
    metadata = read_json(artifacts["run_metadata"])
    if (
        Path(metadata["model_path"]).resolve() != Path(export["hf_export"]["path"]).resolve()
        or metadata.get("benchmarks") != ["greekmmlu"]
        or metadata.get("sample_size") != 0
        or metadata.get("dtype") != "float32"
        or metadata.get("random_state") != 42
        or metadata.get("max_input_tokens") != 3072
        or metadata.get("candidate_batch_size") != 16
        or metadata.get("example_batch_size") != 16
    ):
        raise ValueError("GreekMMLU run metadata is not bound to the exact HF export")
    dataset_bindings = metadata.get("dataset_bindings")
    if (
        not isinstance(dataset_bindings, list)
        or len(dataset_bindings) != 1
        or dataset_bindings[0].get("id") != "greekmmlu"
        or dataset_bindings[0].get("source") != "dascim/GreekMMLU"
        or dataset_bindings[0].get("revision")
        != "6a03aa06b68beb932fb75edff3a34e50b3674649"
        or dataset_bindings[0].get("resolved_split") != "test"
        or int(dataset_bindings[0].get("rows_before_sampling", -1)) != 16632
        or not dataset_bindings[0].get("fingerprint")
    ):
        raise ValueError("GreekMMLU dataset revision/fingerprint binding drift")
    headline_rows = read_json(artifacts["headline"])
    if not isinstance(headline_rows, list) or len(headline_rows) != 1:
        raise ValueError("unexpected GreekMMLU headline structure")
    headline = headline_rows[0]
    if (
        headline.get("benchmark") != "greekmmlu"
        or headline.get("subject") != "__all__"
        or int(headline.get("n", 0)) != 16632
        or any(metric not in headline for metric in ("accuracy", "choice_nll", "correct_answer_bpb"))
    ):
        raise ValueError("incomplete native GreekMMLU full-split metrics")
    prediction_rows = sum(1 for line in artifacts["predictions"].open() if line.strip())
    if prediction_rows != 16632:
        raise ValueError("native GreekMMLU prediction row count drift")
    clean_manifest = read_json(args.clean_subset_manifest)
    if (
        clean_manifest.get("schema_version") != "apertus_mini_greekmmlu_clean_subset_v1"
        or clean_manifest.get("status") != "frozen"
        or clean_manifest.get("dataset_revision")
        != "6a03aa06b68beb932fb75edff3a34e50b3674649"
        or int(clean_manifest.get("full_count", -1)) != 16632
    ):
        raise ValueError("GreekMMLU clean-subset contract drift")
    clean_ids_path = Path(clean_manifest["clean_example_ids"]["path"])
    if sha256_file(clean_ids_path) != clean_manifest["clean_example_ids"]["sha256"]:
        raise ValueError("GreekMMLU clean example-ID payload drift")
    clean_ids = {line.strip() for line in clean_ids_path.open() if line.strip()}
    if len(clean_ids) != int(clean_manifest["clean_count"]):
        raise ValueError("GreekMMLU clean example-ID count drift")
    predictions = [json.loads(line) for line in artifacts["predictions"].open() if line.strip()]
    if len({str(row["example_id"]) for row in predictions}) != 16632:
        raise ValueError("GreekMMLU prediction example IDs are not unique")
    clean_rows = [row for row in predictions if str(row["example_id"]) in clean_ids]
    if len(clean_rows) != len(clean_ids):
        raise ValueError("GreekMMLU clean subset is not a subset of evaluated examples")
    clean_metrics = summarize_prediction_rows(clean_rows)
    payload = {
        "schema_version": "exact_checkpoint_native_greekmmlu_receipt_v1",
        "status": "completed",
        "benchmark_origin": "natively_authored_greek",
        "evaluation_namespace": args.evaluation_namespace,
        "evaluator": {
            "dtype": "float32",
            "random_state": 42,
            "max_input_tokens": 3072,
            "candidate_batch_size": 16,
            "example_batch_size": 16,
        },
        "dataset": dataset_bindings[0],
        "checkpoint": {
            "iteration": export["source"]["iteration"],
            "source_tree_manifest_sha256": export["source"]["source_tree_manifest_sha256"],
            "hf_tree_manifest_sha256": export["hf_export"]["tree_manifest_sha256"],
            "export_receipt_path": str(args.export_receipt.resolve()),
            "export_receipt_sha256": sha256_file(args.export_receipt),
        },
        "metrics": {
            "n": 16632,
            "accuracy": headline["accuracy"],
            "choice_nll": headline["choice_nll"],
            "correct_answer_bpb": headline["correct_answer_bpb"],
            "decontaminated": clean_metrics,
        },
        "clean_subset_manifest": {
            "path": str(args.clean_subset_manifest.resolve()),
            "sha256": sha256_file(args.clean_subset_manifest),
        },
        "artifacts": {
            label: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for label, path in artifacts.items()
        },
        "same_frozen_evaluator_contract_required_for_all_arms_and_checkpoints": True,
    }
    temporary = Path(str(args.output) + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"ok": True, "iteration": payload["checkpoint"]["iteration"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
