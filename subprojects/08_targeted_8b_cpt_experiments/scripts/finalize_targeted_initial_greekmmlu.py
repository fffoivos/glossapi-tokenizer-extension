#!/usr/bin/env python3
"""Bind an iteration-zero GreekMMLU evaluation to its exact model and dataset.

The inherited score finalizer intentionally reports only aggregate scores.  A
targeted launch anchor needs more: the exact public benchmark revision, a
content fingerprint, the evaluated corrected RoPE geometry, and the frozen
bundle which ran the raw scorer.  This performs no model inference; it checks
the completed prediction artifact against the pinned dataset and publishes a
new immutable receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

from contract_utils import file_binding, read_json, require, sha256_file, write_json_atomic


DATASET = "dascim/GreekMMLU"
REVISION = "6a03aa06b68beb932fb75edff3a34e50b3674649"
CONFIG = "All"
SPLIT = "test"
ROWS = 16_632


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def content_fingerprint(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_json(row))
        digest.update(b"\n")
    return digest.hexdigest()


def load_evaluator_examples(code_root: Path, rows: list[dict[str, Any]]) -> list[Any]:
    evaluator = (
        code_root
        / "subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/run_native_greek_mcq_eval.py"
    )
    require(evaluator.is_file(), f"evaluation runner missing from evaluated code bundle: {evaluator}")
    spec = importlib.util.spec_from_file_location("targeted_native_greek_mcq", evaluator)
    require(spec is not None and spec.loader is not None, "could not load evaluated GreekMMLU runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    benchmark = {"id": "greekmmlu"}
    examples = [module.examples_from_row(benchmark, row, index) for index, row in enumerate(rows)]
    require(all(example is not None for example in examples), "GreekMMLU evaluator skipped a source row")
    return examples


def verify_bundle(root: Path, receipt: Path) -> dict[str, Any]:
    verifier = root / "subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py"
    require(verifier.is_file(), f"evaluated code verifier missing: {verifier}")
    subprocess.run(
        ["/usr/bin/python3.11", str(verifier), "--root", str(root), "--receipt", str(receipt), "--kind", "scientific"],
        check=True,
    )
    value = read_json(receipt)
    require(value.get("kind") == "scientific" and value.get("tree_sha256"), "evaluated bundle receipt drift")
    return value


def load_pinned_rows() -> tuple[list[dict[str, Any]], str | None]:
    from datasets import load_dataset

    dataset = load_dataset(DATASET, CONFIG, revision=REVISION, split=SPLIT)
    rows = [dict(row) for row in dataset]
    return rows, getattr(dataset, "_fingerprint", None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-code-root", type=Path, required=True)
    parser.add_argument("--evaluation-code-receipt", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--legacy-receipt", type=Path, required=True)
    parser.add_argument("--clean-subset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    model = args.model.resolve()
    evaluation_root = args.evaluation_root.resolve()
    code_root = args.evaluation_code_root.resolve()
    require(not output.exists(), f"immutable targeted GreekMMLU receipt exists: {output}")
    require(model.is_dir() and evaluation_root.is_dir(), "model or evaluation root missing")
    bundle = verify_bundle(code_root, args.evaluation_code_receipt.resolve())

    run_metadata_path = evaluation_root / "run_metadata.json"
    predictions_path = evaluation_root / "full8b_mixed_iter0_native_mcq_predictions.jsonl"
    require(run_metadata_path.is_file() and predictions_path.is_file(), "GreekMMLU raw artifacts missing")
    metadata = read_json(run_metadata_path)
    require(metadata.get("schema") == "native-greek-mcq-run-v1", "GreekMMLU run metadata schema drift")
    require(Path(metadata.get("model_path", "")).resolve() == model, "GreekMMLU evaluated model drift")
    specs = metadata.get("benchmark_specs", [])
    require(len(specs) == 1 and specs[0].get("id") == "greekmmlu", "GreekMMLU benchmark identity drift")
    benchmark_spec = specs[0]
    require(
        benchmark_spec.get("source") == DATASET
        and benchmark_spec.get("revision") == REVISION
        and benchmark_spec.get("config") == CONFIG
        and benchmark_spec.get("split") == SPLIT,
        "GreekMMLU benchmark metadata drift",
    )
    predictions = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(predictions) == ROWS and all(row.get("benchmark") == "greekmmlu" for row in predictions), "GreekMMLU prediction count/type drift")
    source_rows, internal_fingerprint = load_pinned_rows()
    require(len(source_rows) == ROWS, "pinned GreekMMLU row count drift")
    expected_examples = load_evaluator_examples(code_root, source_rows)
    for index, (prediction, example) in enumerate(zip(predictions, expected_examples)):
        require(
            str(prediction.get("example_id")) == str(example.example_id)
            and int(prediction.get("answer_index", -1)) == int(example.answer_index)
            and prediction.get("subject") == example.subject
            and prediction.get("metadata", {}) == (example.metadata or {})
            and int(prediction.get("correct_answer_utf8_bytes", -1))
            == len(example.choices[example.answer_index].encode("utf-8")),
            f"GreekMMLU prediction/dataset binding drift at row {index}",
        )
    legacy = read_json(args.legacy_receipt)
    require(
        legacy.get("schema_version") == "apertus_full_8b_initial_greekmmlu_v1"
        and str(legacy.get("status", "")).lower() in {"completed", "passed"}
        and Path(legacy.get("model", "")).resolve() == model,
        "legacy GreekMMLU receipt drift",
    )
    for split_name in ("full", "decontaminated"):
        values = legacy.get("metrics", {}).get(split_name, {})
        require(
            int(values.get("n", 0)) > 0
            and all(math.isfinite(float(values.get(key, math.nan))) for key in ("accuracy", "choice_nll", "correct_answer_bpb")),
            f"non-finite GreekMMLU {split_name} metrics",
        )
    clean = read_json(args.clean_subset_manifest)
    clean_ids = clean.get("clean_example_ids", {})
    clean_ids_path = Path(clean_ids.get("path", ""))
    require(
        clean_ids_path.is_file()
        and clean_ids.get("sha256") == sha256_file(clean_ids_path),
        "GreekMMLU clean subset identity drift",
    )
    clean_count = sum(1 for line in clean_ids_path.read_text(encoding="utf-8").splitlines() if line.strip())
    clean_binding = file_binding(args.clean_subset_manifest)
    require(
        legacy.get("clean_subset_manifest_sha256") == clean_binding["sha256"]
        and int(legacy["metrics"]["decontaminated"]["n"]) == clean_count,
        "legacy/decontaminated GreekMMLU subset binding drift",
    )
    config_path = model / "config.json"
    config = read_json(config_path)
    require(
        config.get("rope_theta") == 500_000
        and config.get("max_position_embeddings") == 4_096
        and config.get("tie_word_embeddings") is False
        and config.get("vocab_size") == 148_992,
        "evaluated initial model geometry drift",
    )
    payload = {
        "schema_version": "apertus_full_8b_initial_greekmmlu_v1",
        "status": "completed",
        "model": str(model),
        "model_config": {
            "rope_theta": float(config["rope_theta"]),
            "max_position_embeddings": int(config["max_position_embeddings"]),
            "tie_word_embeddings": config["tie_word_embeddings"],
            "vocab_size": int(config["vocab_size"]),
        },
        "model_config_file": file_binding(config_path),
        "dataset": {
            "source": DATASET,
            "revision": REVISION,
            "config": CONFIG,
            "resolved_split": SPLIT,
            "rows_before_sampling": ROWS,
            "fingerprint": content_fingerprint(source_rows),
            "datasets_internal_fingerprint": internal_fingerprint,
            "binding_contract": "canonical_json_sha256_over_pinned_split_rows_plus_per_prediction_id_answer_subject_metadata_checks",
        },
        "metrics": legacy["metrics"],
        "prediction_artifact": file_binding(predictions_path),
        "evaluation_run_metadata": file_binding(run_metadata_path),
        "legacy_aggregate_receipt": file_binding(args.legacy_receipt),
        "clean_subset_manifest": clean_binding,
        "clean_example_ids": file_binding(clean_ids_path),
        "evaluated_scientific_bundle": {
            "root": str(code_root),
            "receipt": file_binding(args.evaluation_code_receipt),
            "tree_sha256": bundle["tree_sha256"],
        },
    }
    write_json_atomic(output, payload)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
