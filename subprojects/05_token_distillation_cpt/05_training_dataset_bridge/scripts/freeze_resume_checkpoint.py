#!/usr/bin/env python3
"""Freeze and validate one completed segment checkpoint before a relaunch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bridge_common import (
    bound_code_sha,
    file_tree_receipt,
    read_json,
    sha256_file,
    utc_now,
    validate_file_tree_receipt,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--expected-iteration", type=int, required=True)
    parser.add_argument("--probe-plan", type=Path, required=True)
    parser.add_argument("--training-assets-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.expected_iteration <= 0:
        raise ValueError("a resume receipt requires a positive completed iteration")
    plan = read_json(args.probe_plan)
    assets = read_json(args.training_assets_receipt)
    if (
        plan.get("schema_version") != "full_cpt_25b_probe_plan_v2"
        or int(plan.get("iterations", -1)) < args.expected_iteration
    ):
        raise ValueError("resume checkpoint is incompatible with the probe plan")
    if (
        assets.get("schema_version") != "full_cpt_training_assets_receipt_v1"
        or assets.get("status") != "completed"
        or plan.get("training_assets_receipt", {}).get("sha256")
        != sha256_file(args.training_assets_receipt.resolve())
    ):
        raise ValueError("resume checkpoint uses different frozen training assets")
    bridge_path = Path(str(assets["bridge_manifest"])).resolve()
    bridge = read_json(bridge_path)
    input_receipt_path = Path(str(bridge["input_receipt"]["path"])).resolve()
    input_receipt = read_json(input_receipt_path)
    implementation_sha = bound_code_sha(input_receipt, Path(__file__))
    bound_code_sha(input_receipt, Path(__file__).with_name("bridge_common.py"))

    checkpoint_dir = args.checkpoint_dir.resolve()
    if checkpoint_dir != Path(str(plan["output_dir"])).resolve() / "checkpoints":
        raise ValueError("resume checkpoint is outside the frozen probe output")
    marker = checkpoint_dir / "latest_checkpointed_iteration.txt"
    if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != str(
        args.expected_iteration
    ):
        raise ValueError(
            "checkpoint marker does not equal the completed segment boundary"
        )
    iteration_dir = checkpoint_dir / f"iter_{args.expected_iteration:07d}"
    if not iteration_dir.is_dir() or not any(
        path.is_file() for path in iteration_dir.rglob("*")
    ):
        raise ValueError("expected iteration checkpoint payload is absent")
    tree = file_tree_receipt(checkpoint_dir)
    payload = {
        "schema_version": "full_cpt_segment_checkpoint_receipt_v1",
        "status": "completed",
        "completed_at": utc_now(),
        "expected_iteration": args.expected_iteration,
        "probe_plan": {
            "path": str(args.probe_plan.resolve()),
            "sha256": sha256_file(args.probe_plan.resolve()),
        },
        "training_assets_receipt": {
            "path": str(args.training_assets_receipt.resolve()),
            "sha256": sha256_file(args.training_assets_receipt.resolve()),
        },
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": implementation_sha,
        "checkpoint_tree": tree,
        "marker": {
            "path": str(marker),
            "sha256": sha256_file(marker),
            "value": str(args.expected_iteration),
        },
        "iteration_directory": str(iteration_dir),
        "invariants": {
            "exact_segment_boundary": True,
            "checkpoint_inventory_complete_and_hash_bound": True,
            "probe_plan_and_training_assets_bound": True,
        },
    }
    if args.output.exists():
        existing = read_json(args.output)
        if (
            existing.get("schema_version") != "full_cpt_segment_checkpoint_receipt_v1"
            or existing.get("expected_iteration") != args.expected_iteration
            or existing.get("probe_plan", {}).get("sha256")
            != payload["probe_plan"]["sha256"]
            or existing.get("training_assets_receipt", {}).get("sha256")
            != payload["training_assets_receipt"]["sha256"]
            or existing.get("implementation_sha256") != implementation_sha
            or existing.get("checkpoint_tree", {}).get("root") != str(checkpoint_dir)
            or existing.get("checkpoint_tree", {}).get("tree_sha256")
            != tree["tree_sha256"]
        ):
            raise ValueError(
                "existing segment checkpoint receipt has different bindings"
            )
        validate_file_tree_receipt(existing["checkpoint_tree"])
        print(json.dumps({"ok": True, "resumed": True, "output": str(args.output)}))
        return 0
    write_json_atomic(args.output.resolve(), payload)
    print(
        json.dumps({"ok": True, "output": str(args.output.resolve())}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
