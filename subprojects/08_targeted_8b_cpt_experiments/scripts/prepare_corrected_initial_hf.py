#!/usr/bin/env python3
"""Create an immutable corrected-geometry HF view of the verified TD init.

The Token Distillation weights and tokenizer are already verified.  The source
HF export, however, retained its long-context export geometry.  Apertus CPT
uses 4,096-token sequences and RoPE theta 500,000.  This utility creates a
new model directory without copying or rewriting any tensor or tokenizer
bytes: every unchanged file is hard-linked to the verified round-trip export;
only ``config.json`` is written with the two approved geometry corrections.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from contract_utils import file_binding, read_json, require, write_json_atomic


TOKENIZER_SHA256 = "bbb08e71929b519c5c2362338b0fc6a0e99955cb8fdbf0729ae1311117e6561b"
SOURCE_GEOMETRY = {"rope_theta": 12_000_000, "max_position_embeddings": 65_536}
CORRECTED_GEOMETRY = {"rope_theta": 500_000, "max_position_embeddings": 4_096}
REQUIRED_MODEL_FILES = {
    "config.json",
    "generation_config.json",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
}


def canonical_json_digest(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require_roundtrip_zero_drift(value: dict[str, Any]) -> None:
    zero_fields = (
        "standard_max_abs_diff",
        "r17_max_abs_diff",
        "xielu_max_abs_diff",
        "qk_norm_max_abs_diff",
    )
    require(all(float(value.get(field, float("nan"))) == 0.0 for field in zero_fields), "round-trip tensor drift")
    require(value.get("orig_only") == [] and value.get("trip_only") == [] and value.get("shape_mismatches") == [], "round-trip tensor identity drift")
    require(value.get("r17_changed_over_tol_count") == 0 and value.get("standard_changed_over_tol_count") == 0, "round-trip tolerance drift")
    logits = value.get("logits", {})
    require(float(logits.get("logit_max_abs_diff", float("nan"))) == 0.0, "round-trip logit max drift")
    require(float(logits.get("logit_mean_abs_diff_max", float("nan"))) == 0.0, "round-trip logit mean drift")
    prompts = logits.get("per_prompt", [])
    require(bool(prompts) and all(row.get("top_id_match") is True for row in prompts), "round-trip top-id drift")


def require_td_initialization(value: dict[str, Any]) -> None:
    require(value.get("schema_version") == "production_polytonic_td_init_verification_v1", "TD verification schema drift")
    require(value.get("status") == "passed", "TD verification non-passing")
    require(
        value.get("existing_input_rows_exact") is True
        and value.get("existing_output_rows_exact") is True
        and value.get("non_embedding_tensors_exact") is True
        and value.get("new_rows_finite") is True
        and value.get("new_rows_nonzero") is True,
        "TD initialization evidence drift",
    )
    require(value.get("target_layer") == 11, "TD target-layer drift")
    require(value.get("tokenizer_json_sha256") == TOKENIZER_SHA256, "TD tokenizer hash drift")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-root", type=Path, required=True)
    parser.add_argument("--roundtrip-verification", type=Path, required=True)
    parser.add_argument("--td-initialization-verification", type=Path, required=True)
    parser.add_argument("--frozen-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source_model_root.resolve()
    target = args.output_root.resolve()
    frozen_tokenizer = args.frozen_tokenizer_dir.resolve()
    require(source.is_dir(), f"source model root missing: {source}")
    require(frozen_tokenizer.is_dir(), f"frozen tokenizer directory missing: {frozen_tokenizer}")
    require(not target.exists(), f"immutable corrected model output exists: {target}")
    require(target.parent.is_dir(), f"output parent missing: {target.parent}")
    source_files = {path.name for path in source.iterdir() if path.is_file()}
    require(source_files == REQUIRED_MODEL_FILES, f"source model file set drift: {sorted(source_files ^ REQUIRED_MODEL_FILES)}")

    source_config_path = source / "config.json"
    source_config = read_json(source_config_path)
    require(
        {key: source_config.get(key) for key in SOURCE_GEOMETRY} == SOURCE_GEOMETRY,
        "source HF geometry is not the expected long-context export",
    )
    require(source_config.get("tie_word_embeddings") is False and source_config.get("vocab_size") == 148_992, "source model invariant drift")

    roundtrip = read_json(args.roundtrip_verification)
    td_verification = read_json(args.td_initialization_verification)
    require_roundtrip_zero_drift(roundtrip)
    require_td_initialization(td_verification)
    source_tokenizer = source / "tokenizer.json"
    frozen_tokenizer_json = frozen_tokenizer / "tokenizer.json"
    require(frozen_tokenizer_json.is_file(), "frozen tokenizer.json missing")
    frozen_hash = hashlib.sha256(frozen_tokenizer_json.read_bytes()).hexdigest()
    require(frozen_hash == TOKENIZER_SHA256, "frozen tokenizer byte hash drift")
    source_tokenizer_semantic_digest = canonical_json_digest(source_tokenizer)
    frozen_tokenizer_semantic_digest = canonical_json_digest(frozen_tokenizer_json)
    require(source_tokenizer_semantic_digest == frozen_tokenizer_semantic_digest, "round-trip/frozen tokenizer semantic drift")

    corrected_config = dict(source_config)
    corrected_config.update(CORRECTED_GEOMETRY)
    changed = sorted(key for key in corrected_config if corrected_config.get(key) != source_config.get(key))
    require(changed == sorted(CORRECTED_GEOMETRY), f"configuration correction scope drift: {changed}")
    require(
        corrected_config.get("tie_word_embeddings") is False and corrected_config.get("vocab_size") == 148_992,
        "corrected model invariant drift",
    )

    created = False
    try:
        target.mkdir(parents=False)
        created = True
        hardlinked: list[dict[str, Any]] = []
        for filename in sorted(REQUIRED_MODEL_FILES - {"config.json"}):
            source_path = source / filename
            target_path = target / filename
            os.link(source_path, target_path)
            source_stat = source_path.stat()
            target_stat = target_path.stat()
            require(
                source_stat.st_dev == target_stat.st_dev and source_stat.st_ino == target_stat.st_ino,
                f"hardlink verification failed: {filename}",
            )
            hardlinked.append(
                {
                    "name": filename,
                    "bytes": source_stat.st_size,
                    "device": source_stat.st_dev,
                    "inode": source_stat.st_ino,
                }
            )
        write_json_atomic(target / "config.json", corrected_config)
        target_config = read_json(target / "config.json")
        require(target_config == corrected_config, "corrected config write drift")
        require(
            sorted(key for key in target_config if target_config.get(key) != source_config.get(key)) == sorted(CORRECTED_GEOMETRY),
            "corrected config changed keys drift",
        )
        receipt = {
            "schema_version": "apertus_full_8b_corrected_initial_hf_v1",
            "status": "passed",
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "model_root": str(target),
            "source_model_root": str(source),
            "source_model_config": file_binding(source_config_path),
            "corrected_model_config": file_binding(target / "config.json"),
            "source_geometry": SOURCE_GEOMETRY,
            "corrected_geometry": CORRECTED_GEOMETRY,
            "corrected_config_keys": sorted(CORRECTED_GEOMETRY),
            "model_config": {
                "rope_theta": corrected_config["rope_theta"],
                "max_position_embeddings": corrected_config["max_position_embeddings"],
                "tie_word_embeddings": corrected_config["tie_word_embeddings"],
                "vocab_size": corrected_config["vocab_size"],
            },
            "roundtrip_verification": file_binding(args.roundtrip_verification),
            "td_initialization_verification": file_binding(args.td_initialization_verification),
            "frozen_tokenizer": file_binding(frozen_tokenizer_json),
            "frozen_tokenizer_sha256": frozen_hash,
            "source_tokenizer_semantic_sha256": source_tokenizer_semantic_digest,
            "frozen_tokenizer_semantic_sha256": frozen_tokenizer_semantic_digest,
            "tokenizer_semantically_identical_to_roundtrip": True,
            "zero_tensor_and_logit_drift": True,
            "model_and_support_files_hardlinked_to_zero_drift_source": True,
            "hardlinked_files": hardlinked,
        }
        write_json_atomic(target.parent / "corrected_initial_hf_receipt.json", receipt)
    except BaseException:
        if created:
            shutil.rmtree(target, ignore_errors=True)
        raise
    print(target.parent / "corrected_initial_hf_receipt.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
