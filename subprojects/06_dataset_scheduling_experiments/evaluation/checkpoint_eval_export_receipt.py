#!/usr/bin/env python3
"""Prepare and receipt an exact Megatron-checkpoint export for evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_receipt(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "relative_path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not rows:
        raise ValueError(f"empty artifact tree: {root}")
    return rows


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = Path(str(path) + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def prepare(args: argparse.Namespace) -> None:
    source_root = args.source_checkpoint_root.resolve()
    checkpoint_name = "release" if args.iteration == 0 else f"iter_{args.iteration:07d}"
    tracker_value = "release" if args.iteration == 0 else str(args.iteration)
    source_iteration = source_root / checkpoint_name
    if not source_iteration.is_dir():
        raise ValueError(f"missing exact source iteration: {source_iteration}")
    view = args.output_root.resolve() / "source_view"
    if view.exists():
        raise ValueError(f"refusing to replace checkpoint view: {view}")
    view.mkdir(parents=True)
    (view / source_iteration.name).symlink_to(source_iteration, target_is_directory=True)
    (view / "latest_checkpointed_iteration.txt").write_text(f"{tracker_value}\n")
    receipt = {
        "schema_version": "megatron_exact_checkpoint_view_v1",
        "source_checkpoint_root": str(source_root),
        "iteration": args.iteration,
        "source_iteration": str(source_iteration),
        "source_files": tree_receipt(source_iteration),
    }
    receipt["source_tree_manifest_sha256"] = canonical_sha256(receipt["source_files"])
    write_json_atomic(args.output_root / "source_checkpoint_receipt.json", receipt)


def finalize(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    source = json.loads((output_root / "source_checkpoint_receipt.json").read_text())
    hf_root = (output_root / "hf").resolve()
    config = json.loads((hf_root / "config.json").read_text())
    expected = {
        "vocab_size": 148992,
        "hidden_size": 1024,
        "intermediate_size": 6144,
        "num_hidden_layers": 20,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "tie_word_embeddings": True,
        "rope_theta": 500000.0,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"converted HF geometry drift ({key}): {config.get(key)!r}")
    if config.get("rope_scaling") is not None:
        raise ValueError("converted HF checkpoint unexpectedly enables RoPE scaling")
    frozen_tokenizer_path = args.tokenizer_json.resolve()
    frozen_tokenizer_sha = sha256_file(frozen_tokenizer_path)
    if frozen_tokenizer_sha != args.tokenizer_json_sha256:
        raise ValueError("frozen tokenizer payload hash drift")
    exported_tokenizer_path = hf_root / "tokenizer.json"
    tokenizer_sha = sha256_file(exported_tokenizer_path)
    frozen_tokenizer = json.loads(frozen_tokenizer_path.read_text(encoding="utf-8"))
    exported_tokenizer = json.loads(exported_tokenizer_path.read_text(encoding="utf-8"))
    if exported_tokenizer != frozen_tokenizer:
        raise ValueError("converted HF tokenizer is semantically different from the frozen overlay")
    tokenizer_semantic_sha = canonical_sha256(frozen_tokenizer)
    if not (
        (hf_root / "model.safetensors").is_file()
        or (hf_root / "model.safetensors.index.json").is_file()
    ):
        raise ValueError("converted HF safetensors weights are absent")
    mapping_path = output_root / "exact_weight_mapping_receipt.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapped_intermediate = Path(
        mapping.get("intermediate_checkpoint", {}).get("path", "")
    ).resolve()
    mapped_hf = Path(mapping.get("hf_model", {}).get("path", "")).resolve()
    mapped_config = Path(mapping.get("hf_config", {}).get("path", "")).resolve()
    if (
        mapping.get("schema_version")
        != "apertus_mini_exact_checkpoint_weight_mapping_v1"
        or mapping.get("status") != "passed"
        or mapping.get("all_source_parameters_covered") is not True
        or mapping.get("all_hf_tensors_accounted_for") is not True
        or mapping.get("all_mapped_parameter_tensors_bit_exact") is not True
        or int(mapping.get("source_parameter_tensors_expected", -1)) != 202
        or int(mapping.get("source_parameter_tensors_checked_exact", -1))
        not in {202, 203}
        or mapped_hf != (hf_root / "model.safetensors").resolve()
        or mapped_config != (hf_root / "config.json").resolve()
        or (output_root / "intermediate").resolve() not in mapped_intermediate.parents
        or sha256_file(mapped_intermediate)
        != mapping["intermediate_checkpoint"]["sha256"]
        or sha256_file(mapped_hf) != mapping["hf_model"]["sha256"]
        or sha256_file(mapped_config) != mapping["hf_config"]["sha256"]
    ):
        raise ValueError("exact checkpoint weight-mapping receipt drift")
    conversion_log = (output_root / "hf_conversion.log").read_text(
        encoding="utf-8", errors="replace"
    )
    agreement_match = re.search(
        r"Converted model agrees on ([0-9.]+)% of predictions", conversion_log
    )
    closeness_match = re.search(
        r"Converted logits are close on ([0-9.]+)% of values", conversion_log
    )
    if agreement_match is None or closeness_match is None:
        raise ValueError("canonical converter did not emit both logit-equivalence receipts")
    prediction_agreement_percent = float(agreement_match.group(1))
    logits_close_percent = float(closeness_match.group(1))
    parity_dtype_match = re.search(
        r"Converted semantic parity dtype: ([A-Za-z0-9_.-]+)", conversion_log
    )
    if parity_dtype_match is None or parity_dtype_match.group(1) != "float32":
        raise ValueError("canonical converter did not prove float32 semantic parity")
    semantic_parity_dtype = parity_dtype_match.group(1)
    semantic_patterns = {
        "mean_kl_divergence": r"Converted mean KL divergence: ([-+0-9.eE]+)",
        "max_kl_divergence": r"Converted max KL divergence: ([-+0-9.eE]+)",
        "mean_total_variation": r"Converted mean total variation: ([-+0-9.eE]+)",
        "max_total_variation": r"Converted max total variation: ([-+0-9.eE]+)",
        "mean_top_token_logprob_abs_difference": (
            r"Converted mean top-token log-prob absolute difference: ([-+0-9.eE]+)"
        ),
        "p99_top_token_logprob_abs_difference": (
            r"Converted p99 top-token log-prob absolute difference: ([-+0-9.eE]+)"
        ),
        "p999_top_token_logprob_abs_difference": (
            r"Converted p99\.9 top-token log-prob absolute difference: ([-+0-9.eE]+)"
        ),
        "max_top_token_logprob_abs_difference": (
            r"Converted max top-token log-prob absolute difference: ([-+0-9.eE]+)"
        ),
    }
    semantic_metrics = {}
    for name, pattern in semantic_patterns.items():
        match = re.search(pattern, conversion_log)
        if match is not None:
            semantic_metrics[name] = float(match.group(1))
    semantic_limits = {
        "mean_kl_divergence": 1.0e-4,
        "max_kl_divergence": 1.0e-1,
        "mean_total_variation": 5.0e-3,
        "max_total_variation": 2.0e-1,
        "mean_top_token_logprob_abs_difference": 1.0e-2,
        "p99_top_token_logprob_abs_difference": 7.5e-2,
        "p999_top_token_logprob_abs_difference": 2.0e-1,
    }
    raw_logit_threshold_passed = logits_close_percent >= 95.0
    semantic_probability_thresholds_passed = (
        set(semantic_metrics) == set(semantic_limits) | {
            "max_top_token_logprob_abs_difference"
        }
        and all(math.isfinite(value) and value >= 0.0 for value in semantic_metrics.values())
        and all(
            semantic_metrics[name] <= limit
            for name, limit in semantic_limits.items()
        )
    )
    runtime_semantic_parity_passed = prediction_agreement_percent >= 99.5 and (
        raw_logit_threshold_passed or semantic_probability_thresholds_passed
    )
    hf_files = tree_receipt(hf_root)
    timing = json.loads((output_root / "conversion_timing.json").read_text())
    receipt = {
        "schema_version": "native_greekmmlu_exact_checkpoint_export_v1",
        "status": "completed",
        "source": source,
        "conversion": {
            "intermediate": "SwissAI Megatron scripts/conversion/torchdist_2_torch.py",
            "hf": "SwissAI Megatron tools/checkpoint/convert.py --loader core --saver swissai_hf",
            "test_logits_passed": runtime_semantic_parity_passed,
            "exact_weight_mapping_passed": True,
            "semantic_parity_dtype": semantic_parity_dtype,
            "parity_gate_policy": "exact_parameter_mapping_with_runtime_diagnostics_v3",
            "minimum_prediction_agreement_percent": 99.5,
            "prediction_agreement_percent": prediction_agreement_percent,
            "logits_close_percent": logits_close_percent,
            "raw_logit_threshold_passed": raw_logit_threshold_passed,
            "semantic_probability_metrics": semantic_metrics or None,
            "semantic_probability_limits": semantic_limits,
            "max_top_token_logprob_abs_difference_is_diagnostic_only": True,
            "semantic_probability_thresholds_passed": (
                semantic_probability_thresholds_passed
            ),
            "runtime_semantic_parity_passed": runtime_semantic_parity_passed,
            "runtime_numerical_sensitivity_detected": (
                not runtime_semantic_parity_passed
            ),
            "parity_acceptance_path": (
                "canonical_raw_logits"
                if runtime_semantic_parity_passed and raw_logit_threshold_passed
                else (
                    "probability_space"
                    if runtime_semantic_parity_passed
                    else "bit_exact_parameter_mapping"
                )
            ),
            "timing": timing,
        },
        "exact_weight_mapping": {
            "receipt_path": str(mapping_path.resolve()),
            "receipt_sha256": sha256_file(mapping_path),
            "source_parameter_tensors_checked_exact": mapping[
                "source_parameter_tensors_checked_exact"
            ],
            "xielu_constant_tensors_checked": mapping[
                "xielu_constant_tensors_checked"
            ],
            "all_source_parameters_covered": True,
            "all_hf_tensors_accounted_for": True,
            "all_mapped_parameter_tensors_bit_exact": True,
        },
        "hf_export": {
            "path": str(hf_root),
            "files": hf_files,
            "tree_manifest_sha256": canonical_sha256(hf_files),
            "tokenizer_json_sha256": tokenizer_sha,
            "frozen_tokenizer_json_sha256": frozen_tokenizer_sha,
            "tokenizer_semantic_manifest_sha256": tokenizer_semantic_sha,
            "tokenizer_semantically_identical_to_frozen_overlay": True,
            "geometry": expected | {"rope_scaling": None},
        },
        "ready_for_frozen_native_greekmmlu": True,
    }
    write_json_atomic(output_root / "checkpoint_eval_export_receipt.json", receipt)
    print(json.dumps({"ok": True, "output": str(output_root), "iteration": source["iteration"]}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "finalize"))
    parser.add_argument("--source-checkpoint-root", type=Path)
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tokenizer-json-sha256")
    parser.add_argument("--tokenizer-json", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare" and (args.source_checkpoint_root is None or args.iteration is None):
        parser.error("prepare requires --source-checkpoint-root and --iteration")
    if args.mode == "finalize" and (
        not args.tokenizer_json_sha256 or args.tokenizer_json is None
    ):
        parser.error("finalize requires --tokenizer-json and --tokenizer-json-sha256")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.mode == "prepare":
        prepare(parsed)
    else:
        finalize(parsed)
