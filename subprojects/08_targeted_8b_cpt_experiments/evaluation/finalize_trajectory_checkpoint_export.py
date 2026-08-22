#!/usr/bin/env python3
"""Finalize an exact-weight HF checkpoint for trajectory-only evaluation.

This receipt is deliberately distinct from the frozen-evaluator receipt: it
records runtime parity as measured and permits scoring only through the
trajectory evaluator.  Exact weight mapping, geometry and tokenizer identity
remain fail-closed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from checkpoint_export_contract import validate_geometry  # noqa: E402
from contract_utils import file_binding, read_json, require, require_file_binding, write_json_atomic  # noqa: E402
from checkpoint_export_receipt import canonical_sha256, tree_inventory  # noqa: E402


def measure_parity(log: str) -> dict[str, object]:
    def number_optional(pattern: str) -> float | None:
        match = re.search(pattern, log)
        return None if match is None else float(match.group(1))

    dtype = re.search(r"Converted semantic parity dtype: ([A-Za-z0-9_.-]+)", log)
    require(dtype is not None and dtype.group(1) == "float32", "converter parity dtype drift")
    prediction_agreement = number_optional(r"Converted model agrees on ([0-9.]+)% of predictions")
    require(prediction_agreement is not None, "converter omitted prediction-agreement diagnostics")
    logits_close = number_optional(r"Converted logits are close on ([0-9.]+)% of values")
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
    metrics = {key: number_optional(pattern) for key, pattern in patterns.items()}
    require(
        all(value is None or (math.isfinite(value) and value >= 0) for value in metrics.values()),
        "non-finite parity metric",
    )
    limits = {
        "mean_kl_divergence": 1.0e-4,
        "max_kl_divergence": 1.0e-1,
        "mean_total_variation": 5.0e-3,
        "max_total_variation": 2.0e-1,
        "mean_top_token_logprob_abs_difference": 1.0e-2,
        "p99_top_token_logprob_abs_difference": 7.5e-2,
        "p999_top_token_logprob_abs_difference": 2.0e-1,
    }
    missing = [key for key, value in metrics.items() if value is None]
    if logits_close is None:
        missing.insert(0, "logits_close_percent")
    diagnostics_complete = not missing
    probability_pass = diagnostics_complete and all(
        metrics[key] is not None and metrics[key] <= limit for key, limit in limits.items()
    )
    raw_pass = logits_close is not None and logits_close >= 95.0
    prediction_pass = prediction_agreement >= 99.5
    return {
        "semantic_parity_dtype": "float32",
        "prediction_agreement_percent": prediction_agreement,
        "prediction_agreement_threshold_percent": 99.5,
        "prediction_agreement_threshold_passed": prediction_pass,
        "logits_close_percent": logits_close,
        "raw_logit_threshold_passed": raw_pass,
        "semantic_probability_metrics": metrics,
        "semantic_probability_limits": limits,
        "semantic_probability_thresholds_passed": probability_pass,
        "diagnostics_complete": diagnostics_complete,
        "missing_diagnostics": missing,
        "runtime_semantic_parity_passed": prediction_pass and (raw_pass or probability_pass),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("8b", "1p5b"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-contract", type=Path, required=True)
    parser.add_argument("--true-vocab-size", type=int, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--converter-overlay-receipt", type=Path, required=True)
    args = parser.parse_args()

    root = args.output_root.resolve()
    source_path = root / "source_checkpoint_receipt.json"
    source = read_json(source_path)
    require(source.get("schema_version") == "apertus_exact_checkpoint_view_v2", "source receipt drift")
    require(source.get("scale") == args.scale, "source scale drift")
    hf_root = root / "hf"
    config_path = hf_root / "config.json"
    geometry = validate_geometry(
        read_json(config_path), read_json(args.model_contract),
        scale=args.scale, true_vocab_size=args.true_vocab_size,
    )
    require(read_json(hf_root / "tokenizer.json") == read_json(args.tokenizer_json), "tokenizer semantic drift")
    require(file_binding(args.tokenizer_json)["sha256"] == args.tokenizer_sha256, "tokenizer digest drift")

    mapping_path = root / "exact_weight_mapping_receipt.json"
    mapping = read_json(mapping_path)
    require(
        mapping.get("schema_version") == "apertus_exact_checkpoint_weight_mapping_v2"
        and mapping.get("status") == "passed"
        and mapping.get("scale") == args.scale
        and mapping.get("model_contract") == file_binding(args.model_contract)
        and mapping.get("all_source_parameters_covered") is True
        and mapping.get("all_hf_tensors_accounted_for") is True
        and mapping.get("all_mapped_parameter_tensors_bit_exact") is True,
        "exact weight mapping drift",
    )
    require_file_binding(mapping["intermediate_checkpoint"])
    require_file_binding(mapping["hf_config"], expected_path=config_path)
    for row in mapping["hf_weight_files"]:
        require_file_binding(row)

    overlay = read_json(args.converter_overlay_receipt)
    require(overlay.get("status") == "completed", "converter overlay receipt drift")
    timing = read_json(root / "conversion_timing.json")
    require(int(timing.get("total_conversion_seconds", -1)) >= 0, "conversion timing drift")
    parity = measure_parity((root / "hf_conversion.log").read_text(encoding="utf-8", errors="replace"))
    files = tree_inventory(hf_root)
    result = {
        "schema_version": "apertus_h2g_trajectory_checkpoint_export_v1",
        "status": "completed",
        "scale": args.scale,
        "iteration": int(source["iteration"]),
        "source_checkpoint": file_binding(source_path),
        "model_contract": file_binding(args.model_contract),
        "geometry": geometry,
        "conversion": parity | {
            "timing": timing,
            "converter_overlay": file_binding(args.converter_overlay_receipt),
        },
        "exact_weight_mapping": file_binding(mapping_path),
        "exact_weight_mapping_passed": True,
        "hf_export": {
            "path": str(hf_root.resolve()),
            "files": files,
            "tree_manifest_sha256": canonical_sha256(files),
            "tokenizer_json_sha256": args.tokenizer_sha256,
            "tokenizer_semantically_identical_to_frozen_overlay": True,
        },
        "ready_for_frozen_evaluators": False,
        "ready_for_trajectory_evaluator": True,
        "scope": "matched_cross_scale_greekmmlu_trajectory_only",
    }
    write_json_atomic(root / "checkpoint_export_receipt.json", result)
    print(json.dumps({"ok": True, "scale": args.scale, "iteration": result["iteration"], "runtime_semantic_parity_passed": parity["runtime_semantic_parity_passed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
