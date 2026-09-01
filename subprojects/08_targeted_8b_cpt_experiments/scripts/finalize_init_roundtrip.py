#!/usr/bin/env python3
"""Finalize an exact HF/Megatron/HF initialization round-trip receipt."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from contract_utils import (
    executing_code_bundle,
    file_binding,
    require,
    require_receipt,
    require_relative_inventory,
    sha256_file,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("8b", "1p5b"), required=True)
    parser.add_argument("--reference-hf-root", type=Path, required=True)
    parser.add_argument("--geometry-receipt", type=Path, required=True)
    parser.add_argument("--roundtrip-hf-root", type=Path, required=True)
    parser.add_argument("--megatron-root", type=Path, required=True)
    parser.add_argument("--verification-json", type=Path, required=True)
    parser.add_argument("--megatron-commit", required=True)
    parser.add_argument("--target-tensor-parallel", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def tree_inventory(root: Path) -> list[dict]:
    require(root.is_dir(), f"artifact root missing: {root}")
    rows = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        rows.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    require(bool(rows), f"artifact root is empty: {root}")
    return rows


def main() -> int:
    args = parse_args()
    require(not args.output.exists(), f"immutable roundtrip receipt exists: {args.output}")
    require(args.megatron_commit == "c92402e39ef3c8e69ea378a59e79059dc14541f4", "Megatron revision drift")
    expected_tp = 2 if args.scale == "8b" else 1
    require(args.target_tensor_parallel == expected_tp, "scale/TP drift")
    geometry = require_receipt(
        args.geometry_receipt,
        schemas={"apertus_training_geometry_hf_view_v1"},
    )
    require(Path(str(geometry.get("output_root", ""))).resolve() == args.reference_hf_root.resolve(), "roundtrip geometry root drift")
    require_relative_inventory(root=args.reference_hf_root, rows=geometry.get("output_files"))
    verification = json.loads(args.verification_json.read_text(encoding="utf-8"))
    for key in ("standard_max_abs_diff", "r17_max_abs_diff", "xielu_max_abs_diff", "qk_norm_max_abs_diff"):
        require(float(verification.get(key, float("nan"))) == 0.0, f"roundtrip tensor drift: {key}")
    for key in ("orig_only", "trip_only", "standard_changed_over_tol", "r17_changed_over_tol", "shape_mismatches"):
        require(verification.get(key) == [], f"roundtrip inventory drift: {key}")
    require(int(verification.get("standard_changed_over_tol_count", 0)) == 0 and int(verification.get("r17_changed_over_tol_count", 0)) == 0, "roundtrip tolerance drift")
    logits = verification.get("logits")
    require(isinstance(logits, dict), "roundtrip logits evidence missing")
    require(float(logits.get("logit_max_abs_diff", float("nan"))) == 0.0 and float(logits.get("logit_mean_abs_diff_max", float("nan"))) == 0.0, "roundtrip logit drift")
    prompts = logits.get("per_prompt", [])
    require(isinstance(prompts, list) and bool(prompts), "roundtrip per-prompt logit evidence missing")
    require(all(isinstance(row, dict) and row.get("top_id_match") is True for row in prompts), "roundtrip top-id drift")
    reference_config = json.loads((args.reference_hf_root / "config.json").read_text(encoding="utf-8"))
    roundtrip_config = json.loads((args.roundtrip_hf_root / "config.json").read_text(encoding="utf-8"))
    for config in (reference_config, roundtrip_config):
        require(float(config.get("rope_theta")) == 500_000.0 and int(config.get("max_position_embeddings")) == 4_096, "training geometry drift")
        require(config.get("tie_word_embeddings") is False and int(config.get("vocab_size")) == 148_480, "model/tokenizer invariant drift")
    megatron_files = tree_inventory(args.megatron_root)
    roundtrip_files = tree_inventory(args.roundtrip_hf_root)
    receipt = {
        "schema_version": "apertus_targeted_init_roundtrip_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scale": args.scale,
        "megatron_commit": args.megatron_commit,
        "target_tensor_parallel": args.target_tensor_parallel,
        "target_pipeline_parallel": 1,
        "tensor_and_logit_roundtrip_exact": True,
        "training_geometry": {"rope_theta": 500_000.0, "max_position_embeddings": 4_096},
        "reference_config": file_binding(args.reference_hf_root / "config.json"),
        "reference_tokenizer": file_binding(args.reference_hf_root / "tokenizer.json"),
        "geometry_receipt": file_binding(args.geometry_receipt),
        "verification": file_binding(args.verification_json),
        "megatron_root": str(args.megatron_root.resolve()),
        "megatron_files": megatron_files,
        "roundtrip_hf_root": str(args.roundtrip_hf_root.resolve()),
        "roundtrip_files": roundtrip_files,
        "executing_code_bundle": executing_code_bundle(),
    }
    write_json_atomic(args.output, receipt)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
