#!/usr/bin/env python3
"""Prepare and finalize one scale-aware checkpoint export for evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from checkpoint_export_contract import validate_geometry
from contract_utils import (
    file_binding,
    read_json,
    require,
    require_file_binding,
    write_json_atomic,
)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def tree_inventory(root: Path) -> list[dict[str, Any]]:
    root = root.resolve()
    rows = []
    for path in sorted(
        candidate for candidate in root.rglob("*") if candidate.is_file()
    ):
        binding = file_binding(path)
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": binding["bytes"],
                "sha256": binding["sha256"],
            }
        )
    require(bool(rows), f"empty artifact tree: {root}")
    return rows


def prepare(args: argparse.Namespace) -> None:
    source_root = args.source_checkpoint_root.resolve()
    checkpoint_name = "release" if args.iteration == 0 else f"iter_{args.iteration:07d}"
    tracker_value = "release" if args.iteration == 0 else str(args.iteration)
    source_iteration = source_root / checkpoint_name
    require(
        source_iteration.is_dir() and not source_iteration.is_symlink(),
        "exact source iteration missing",
    )
    output_root = args.output_root.resolve()
    require(
        not output_root.exists(), f"immutable checkpoint export exists: {output_root}"
    )
    output_root.mkdir(parents=True)
    view = output_root / "source_view"
    view.mkdir()
    (view / checkpoint_name).symlink_to(source_iteration, target_is_directory=True)
    (view / "latest_checkpointed_iteration.txt").write_text(
        f"{tracker_value}\n", encoding="utf-8"
    )
    files = tree_inventory(source_iteration)
    write_json_atomic(
        output_root / "source_checkpoint_receipt.json",
        {
            "schema_version": "apertus_exact_checkpoint_view_v2",
            "status": "frozen",
            "scale": args.scale,
            "iteration": args.iteration,
            "source_checkpoint_root": str(source_root),
            "source_iteration": str(source_iteration),
            "source_files": files,
            "source_tree_manifest_sha256": canonical_sha256(files),
        },
    )


def parse_runtime_parity(log: str) -> dict[str, Any]:
    agreement = re.search(r"Converted model agrees on ([0-9.]+)% of predictions", log)
    closeness = re.search(r"Converted logits are close on ([0-9.]+)% of values", log)
    dtype = re.search(r"Converted semantic parity dtype: ([A-Za-z0-9_.-]+)", log)
    require(
        agreement is not None and closeness is not None,
        "converter omitted logit-equivalence diagnostics",
    )
    require(
        dtype is not None and dtype.group(1) == "float32",
        "converter parity was not measured in float32",
    )
    prediction_agreement = float(agreement.group(1))
    logits_close = float(closeness.group(1))
    patterns = {
        "mean_kl_divergence": r"Converted mean KL divergence: ([-+0-9.eE]+)",
        "max_kl_divergence": r"Converted max KL divergence: ([-+0-9.eE]+)",
        "mean_total_variation": r"Converted mean total variation: ([-+0-9.eE]+)",
        "max_total_variation": r"Converted max total variation: ([-+0-9.eE]+)",
        "mean_top_token_logprob_abs_difference": r"Converted mean top-token log-prob absolute difference: ([-+0-9.eE]+)",
        "p99_top_token_logprob_abs_difference": r"Converted p99 top-token log-prob absolute difference: ([-+0-9.eE]+)",
        "p999_top_token_logprob_abs_difference": r"Converted p99\.9 top-token log-prob absolute difference: ([-+0-9.eE]+)",
        "max_top_token_logprob_abs_difference": r"Converted max top-token log-prob absolute difference: ([-+0-9.eE]+)",
    }
    metrics = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, log)
        if match is not None:
            metrics[key] = float(match.group(1))
    limits = {
        "mean_kl_divergence": 1.0e-4,
        "max_kl_divergence": 1.0e-1,
        "mean_total_variation": 5.0e-3,
        "max_total_variation": 2.0e-1,
        "mean_top_token_logprob_abs_difference": 1.0e-2,
        "p99_top_token_logprob_abs_difference": 7.5e-2,
        "p999_top_token_logprob_abs_difference": 2.0e-1,
    }
    probability_pass = (
        set(metrics) == set(limits) | {"max_top_token_logprob_abs_difference"}
        and all(math.isfinite(value) and value >= 0 for value in metrics.values())
        and all(metrics[key] <= limit for key, limit in limits.items())
    )
    raw_pass = logits_close >= 95.0
    passed = prediction_agreement >= 99.5 and (raw_pass or probability_pass)
    require(passed, "converted checkpoint failed runtime semantic parity")
    return {
        "semantic_parity_dtype": "float32",
        "prediction_agreement_percent": prediction_agreement,
        "logits_close_percent": logits_close,
        "raw_logit_threshold_passed": raw_pass,
        "semantic_probability_metrics": metrics,
        "semantic_probability_limits": limits,
        "semantic_probability_thresholds_passed": probability_pass,
        "runtime_semantic_parity_passed": True,
        "parity_acceptance_path": "canonical_raw_logits"
        if raw_pass
        else "probability_space",
    }


def finalize(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    source_path = output_root / "source_checkpoint_receipt.json"
    source = read_json(source_path)
    require(
        source.get("schema_version") == "apertus_exact_checkpoint_view_v2",
        "source checkpoint receipt drift",
    )
    require(source.get("scale") == args.scale, "source checkpoint scale drift")
    hf_root = output_root / "hf"
    config_path = hf_root / "config.json"
    config = read_json(config_path)
    model_contract = read_json(args.model_contract)
    geometry = validate_geometry(
        config, model_contract, scale=args.scale, true_vocab_size=args.true_vocab_size
    )

    frozen_tokenizer = read_json(args.tokenizer_json)
    exported_tokenizer = read_json(hf_root / "tokenizer.json")
    require(
        exported_tokenizer == frozen_tokenizer,
        "exported tokenizer differs from frozen overlay",
    )
    require(
        file_binding(args.tokenizer_json)["sha256"] == args.tokenizer_sha256,
        "frozen tokenizer SHA-256 drift",
    )

    mapping_path = output_root / "exact_weight_mapping_receipt.json"
    mapping = read_json(mapping_path)
    require(
        mapping.get("schema_version") == "apertus_exact_checkpoint_weight_mapping_v2"
        and mapping.get("status") == "passed"
        and mapping.get("scale") == args.scale
        and mapping.get("model_contract") == file_binding(args.model_contract)
        and mapping.get("all_source_parameters_covered") is True
        and mapping.get("all_hf_tensors_accounted_for") is True
        and mapping.get("all_mapped_parameter_tensors_bit_exact") is True,
        "exact checkpoint mapping receipt drift",
    )
    require_file_binding(mapping["intermediate_checkpoint"])
    require_file_binding(mapping["hf_config"], expected_path=config_path)
    for binding in mapping["hf_weight_files"]:
        require_file_binding(binding)

    parity = parse_runtime_parity(
        (output_root / "hf_conversion.log").read_text(
            encoding="utf-8", errors="replace"
        )
    )
    hf_files = tree_inventory(hf_root)
    timing_path = output_root / "conversion_timing.json"
    timing = read_json(timing_path)
    require(
        all(
            int(timing.get(key, -1)) >= 0
            for key in (
                "torch_dist_to_torch_seconds",
                "torch_to_hf_and_logit_test_seconds",
                "exact_weight_mapping_verification_seconds",
                "total_conversion_seconds",
            )
        ),
        "conversion timing receipt drift",
    )
    result = {
        "schema_version": "apertus_hard_h_to_g_checkpoint_export_v1",
        "status": "completed",
        "scale": args.scale,
        "iteration": int(source["iteration"]),
        "source_checkpoint": file_binding(source_path),
        "model_contract": file_binding(args.model_contract),
        "geometry": geometry,
        "conversion": parity
        | {
            "pipeline": [
                "SwissAI Megatron scripts/conversion/torchdist_2_torch.py",
                "SwissAI Megatron tools/checkpoint/convert.py --loader core --saver swissai_hf",
            ],
            "timing": timing,
        },
        "exact_weight_mapping": file_binding(mapping_path),
        "hf_export": {
            "path": str(hf_root.resolve()),
            "files": hf_files,
            "tree_manifest_sha256": canonical_sha256(hf_files),
            "tokenizer_json_sha256": args.tokenizer_sha256,
            "tokenizer_semantically_identical_to_frozen_overlay": True,
        },
        "ready_for_frozen_evaluators": True,
    }
    write_json_atomic(output_root / "checkpoint_export_receipt.json", result)
    print(
        json.dumps({"ok": True, "scale": args.scale, "iteration": result["iteration"]})
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "finalize"))
    parser.add_argument("--source-checkpoint-root", type=Path)
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--scale", choices=("8b", "1p5b"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-contract", type=Path)
    parser.add_argument("--true-vocab-size", type=int)
    parser.add_argument("--tokenizer-json", type=Path)
    parser.add_argument("--tokenizer-sha256")
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.source_checkpoint_root is None or args.iteration is None:
            parser.error("prepare requires --source-checkpoint-root and --iteration")
    else:
        if (
            args.model_contract is None
            or args.true_vocab_size is None
            or args.tokenizer_json is None
            or not args.tokenizer_sha256
        ):
            parser.error(
                "finalize requires model contract, true vocabulary and tokenizer identity"
            )
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.mode == "prepare":
        prepare(parsed)
    else:
        finalize(parsed)
