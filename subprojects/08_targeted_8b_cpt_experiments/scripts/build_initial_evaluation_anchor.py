#!/usr/bin/env python3
"""Freeze experiment-specific source-validation and conversion anchors."""

from __future__ import annotations

import argparse
import datetime as dt
import math
import shutil
from pathlib import Path
from typing import Any

from contract_utils import copy_file_atomic, file_binding, read_json, require, sha256_file, write_json_atomic


PASSING = {"completed", "frozen", "passed"}


def passing(path: Path, schema: str) -> dict[str, Any]:
    value = read_json(path)
    require(value.get("schema_version") == schema, f"{path}: schema drift")
    require(str(value.get("status", "")).lower() in PASSING, f"{path}: non-passing status")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("A", "B"), required=True)
    parser.add_argument("--checkpoint-iteration", type=int, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--per-document-root", type=Path, required=True)
    parser.add_argument("--greekmmlu-receipt", type=Path, required=True)
    parser.add_argument("--checkpoint-receipt", type=Path, required=True)
    parser.add_argument("--model-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require(not args.output_dir.exists(), f"immutable anchor output exists: {args.output_dir}")
    expected_iteration = 0 if args.experiment == "A" else 9536
    require(args.checkpoint_iteration == expected_iteration, "checkpoint iteration/experiment drift")
    validation = passing(args.validation_manifest, "apertus_full_8b_validation_manifest_v1")
    expected_panels = {row["name"]: row for row in validation.get("panels", [])}
    require(len(expected_panels) == 13, "validation panel count drift")

    receipts = sorted(args.per_document_root.glob("*.receipt.json"))
    require(len(receipts) == 13, "per-document receipt count drift")
    require({path.name.removesuffix(".receipt.json") for path in receipts} == set(expected_panels), "per-document panel identity drift")
    panels = []
    model_root: Path | None = None
    for path in receipts:
        panel = path.name.removesuffix(".receipt.json")
        value = passing(path, "apertus_per_document_validation_v1")
        aggregate = value.get("aggregate", {})
        observed_model = Path(value.get("model", "")).resolve()
        if model_root is None:
            model_root = observed_model
        require(observed_model == model_root and model_root.is_dir(), f"per-document model drift: {panel}")
        expected_input = expected_panels[panel]["raw_jsonl"]
        observed_input = value.get("input", {})
        require(
            Path(observed_input.get("path", "")).resolve() == Path(expected_input["path"]).resolve()
            and observed_input.get("sha256") == expected_input["sha256"],
            f"per-document input drift: {panel}",
        )
        bpb = float(aggregate.get("bpb", math.nan))
        loss = float(aggregate.get("mean_nll", math.nan))
        require(math.isfinite(bpb) and math.isfinite(loss), f"non-finite per-document metric: {panel}")
        require(int(aggregate.get("documents", 0)) > 0, f"empty per-document panel: {panel}")
        panels.append(
            {
                "name": panel,
                "documents": int(aggregate["documents"]),
                "bpb": bpb,
                "lm_loss": loss,
                "receipt": file_binding(path),
            }
        )
    assert model_root is not None
    model_config_path = model_root / "config.json"
    require(model_config_path.is_file(), "initial model config missing")
    model_config = read_json(model_config_path)

    greek = read_json(args.greekmmlu_receipt)
    require(
        greek.get("schema_version")
        in {"apertus_full_8b_initial_greekmmlu_v1", "exact_checkpoint_native_greekmmlu_receipt_v1"}
        and str(greek.get("status", "")).lower() in PASSING,
        "GreekMMLU receipt drift",
    )
    if args.experiment == "A":
        require(Path(greek.get("model", "")).resolve() == model_root, "A GreekMMLU model drift")
        evidence = passing(args.model_evidence, "apertus_full_8b_corrected_initial_hf_v1")
        require(Path(evidence.get("model_root", "")).resolve() == model_root, "A model-evidence root drift")
        require(
            evidence.get("zero_tensor_and_logit_drift") is True
            and evidence.get("model_and_support_files_hardlinked_to_zero_drift_source") is True
            and evidence.get("tokenizer_semantically_identical_to_roundtrip") is True,
            "A initial HF zero-drift evidence failed",
        )
        geometry = {
            "rope_theta": model_config.get("rope_theta"),
            "max_position_embeddings": model_config.get("max_position_embeddings"),
            "tie_word_embeddings": model_config.get("tie_word_embeddings"),
            "vocab_size": model_config.get("vocab_size"),
        }
        exact_mapping = True
        tokenizer_semantic = True
    else:
        evidence = read_json(args.model_evidence)
        require(evidence.get("schema_version") == "native_greekmmlu_exact_checkpoint_export_v1", "B model-evidence schema drift")
        export_receipt = greek.get("checkpoint", {})
        require(
            int(export_receipt.get("iteration", -1)) == 9536
            and Path(export_receipt.get("export_receipt_path", "")).resolve() == args.model_evidence.resolve()
            and export_receipt.get("export_receipt_sha256") == sha256_file(args.model_evidence),
            "B GreekMMLU/export binding drift",
        )
        hf = evidence.get("hf_export", {})
        require(Path(hf.get("path", "")).resolve() == model_root, "B exported model root drift")
        mapping = evidence.get("exact_weight_mapping", {})
        require(
            mapping.get("all_hf_tensors_accounted_for") is True
            and mapping.get("all_mapped_parameter_tensors_bit_exact") is True
            and mapping.get("all_source_parameters_covered") is True,
            "B exact weight mapping failed",
        )
        source = evidence.get("source", {})
        checkpoint = read_json(args.checkpoint_receipt)
        require(
            source.get("source_tree_manifest_sha256") == checkpoint.get("source_tree_manifest_sha256")
            and int(source.get("iteration", -1)) == 9536,
            "B checkpoint/export source drift",
        )
        geometry = hf.get("geometry", {})
        exact_mapping = True
        tokenizer_semantic = hf.get("tokenizer_semantically_identical_to_frozen_overlay") is True

    require(
        float(geometry.get("rope_theta", -1)) == 500000.0
        and int(geometry.get("max_position_embeddings", -1)) == 4096
        and geometry.get("tie_word_embeddings") is False
        and int(geometry.get("vocab_size", -1)) == 148992
        and tokenizer_semantic,
        "model geometry/tokenizer drift",
    )
    initial_validation = {
        "schema_version": "targeted_8b_initial_validation_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment": args.experiment,
        "checkpoint_iteration": expected_iteration,
        "model": str(model_root),
        "validation_manifest": file_binding(args.validation_manifest),
        "panels": sorted(panels, key=lambda row: row["name"]),
    }
    conversion = {
        "schema_version": "targeted_8b_conversion_greekmmlu_smoke_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment": args.experiment,
        "checkpoint_iteration": expected_iteration,
        "model": str(model_root),
        "model_evidence": file_binding(args.model_evidence),
        "model_config": file_binding(model_config_path),
        "checkpoint_receipt": file_binding(args.checkpoint_receipt),
        "greekmmlu_receipt": file_binding(args.greekmmlu_receipt),
        "geometry": {
            "rope_theta": float(geometry["rope_theta"]),
            "max_position_embeddings": int(geometry["max_position_embeddings"]),
            "tie_word_embeddings": False,
            "vocab_size": 148992,
        },
        "tokenizer_semantically_identical_to_frozen_overlay": tokenizer_semantic,
        "exact_weight_mapping_passed": exact_mapping,
    }
    # The whole anchor directory is a single immutable transaction. A failed
    # copy or receipt write is removed so retry never encounters a poisoned,
    # apparently complete output directory.
    created_root = False
    try:
        args.output_dir.mkdir(parents=True)
        created_root = True
        per_document_output = args.output_dir / "per_document"
        per_document_output.mkdir()
        for path in receipts:
            copy_file_atomic(path, per_document_output / path.name)
        copy_file_atomic(args.greekmmlu_receipt, args.output_dir / "greekmmlu_receipt.json")
        write_json_atomic(args.output_dir / "initial_validation_receipt.json", initial_validation)
        write_json_atomic(args.output_dir / "conversion_greekmmlu_smoke_receipt.json", conversion)
    except BaseException:
        if created_root:
            shutil.rmtree(args.output_dir, ignore_errors=True)
        raise
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
