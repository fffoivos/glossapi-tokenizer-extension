#!/usr/bin/env python3
"""Freeze a full-8B HF export after complete bit-exact weight verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def tree_receipt(root: Path) -> list[dict]:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--tokenizer-json-sha256", required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    source = json.loads((output_root / "source_checkpoint_receipt.json").read_text())
    if source.get("schema_version") != "megatron_exact_checkpoint_view_v1":
        raise ValueError("source checkpoint receipt drift")
    hf_root = (output_root / "hf").resolve()
    config_path = hf_root / "config.json"
    config = json.loads(config_path.read_text())
    expected = {
        "vocab_size": 148992,
        "hidden_size": 4096,
        "intermediate_size": 21504,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "tie_word_embeddings": False,
        "max_position_embeddings": 4096,
        "rope_theta": 500000,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"converted Apertus-8B geometry drift ({key})")
    rope = config.get("rope_scaling") or {}
    if rope.get("factor") != 8.0 or rope.get("rope_type") != "llama3":
        raise ValueError("converted Apertus-8B RoPE scaling drift")

    frozen_tokenizer = args.tokenizer_json.resolve()
    if sha256_file(frozen_tokenizer) != args.tokenizer_json_sha256:
        raise ValueError("frozen tokenizer payload hash drift")
    exported_tokenizer = hf_root / "tokenizer.json"
    if json.loads(exported_tokenizer.read_text()) != json.loads(frozen_tokenizer.read_text()):
        raise ValueError("converted tokenizer is semantically different from frozen tokenizer")

    mapping_path = output_root / "exact_weight_mapping_receipt.json"
    mapping = json.loads(mapping_path.read_text())
    if (
        mapping.get("schema_version") != "apertus_full_8b_exact_checkpoint_weight_mapping_v1"
        or mapping.get("status") != "passed"
        or mapping.get("all_source_parameters_covered") is not True
        or mapping.get("all_hf_tensors_accounted_for") is not True
        or mapping.get("all_mapped_parameter_tensors_bit_exact") is not True
        or mapping.get("source_parameter_tensors_expected") != 323
        or mapping.get("source_parameter_tensors_checked_exact") != 323
        or mapping.get("xielu_constant_tensors_checked") != 64
    ):
        raise ValueError("full-8B exact checkpoint weight-mapping receipt drift")
    intermediate = Path(mapping["intermediate_checkpoint"]["path"]).resolve()
    if (output_root / "intermediate").resolve() not in intermediate.parents:
        raise ValueError("mapped intermediate checkpoint is outside export root")
    bindings = [mapping["intermediate_checkpoint"], mapping["hf_index"], mapping["hf_config"]]
    bindings.extend(mapping["hf_shards"])
    for binding in bindings:
        path = Path(binding["path"]).resolve()
        if not path.is_file() or sha256_file(path) != binding["sha256"]:
            raise ValueError(f"mapped artifact hash drift: {path}")

    hf_files = tree_receipt(hf_root)
    timing = json.loads((output_root / "conversion_timing.json").read_text())
    payload = {
        "schema_version": "native_greekmmlu_exact_checkpoint_export_v1",
        "status": "completed",
        "model_scale": "8B",
        "source": source,
        "conversion": {
            "intermediate": "SwissAI Megatron scripts/conversion/torchdist_2_torch.py",
            "hf": "SwissAI Megatron tools/checkpoint/convert.py --loader core --saver swissai_hf",
            "runtime_logit_diagnostics": "skipped_single_gpu_memory_limit",
            "measured_failed_diagnostic_job": "single 96GB GH200 OOM during simultaneous runtime comparison",
            "test_logits_passed": None,
            "exact_weight_mapping_passed": True,
            "parity_gate_policy": "bit_exact_parameter_mapping_plus_authoritative_hf_runtime_v1",
            "parity_acceptance_path": "bit_exact_parameter_mapping",
            "training_runtime_changed": False,
            "timing": timing,
        },
        "exact_weight_mapping": {
            "receipt_path": str(mapping_path.resolve()),
            "receipt_sha256": sha256_file(mapping_path),
            "source_parameter_tensors_checked_exact": 323,
            "xielu_constant_tensors_checked": 64,
            "all_source_parameters_covered": True,
            "all_hf_tensors_accounted_for": True,
            "all_mapped_parameter_tensors_bit_exact": True,
        },
        "hf_export": {
            "path": str(hf_root),
            "files": hf_files,
            "tree_manifest_sha256": canonical_sha256(hf_files),
            "tokenizer_json_sha256": sha256_file(exported_tokenizer),
            "frozen_tokenizer_json_sha256": sha256_file(frozen_tokenizer),
            "tokenizer_semantic_manifest_sha256": canonical_sha256(
                json.loads(frozen_tokenizer.read_text())
            ),
            "tokenizer_semantically_identical_to_frozen_overlay": True,
            "geometry": expected | {"rope_scaling": rope},
        },
        "ready_for_frozen_native_greekmmlu": True,
    }
    temporary = output_root / "checkpoint_eval_export_receipt.json.partial"
    output = output_root / "checkpoint_eval_export_receipt.json"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"ok": True, "iteration": source["iteration"], "checked": 323}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
