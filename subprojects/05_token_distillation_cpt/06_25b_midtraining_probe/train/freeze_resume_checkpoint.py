#!/usr/bin/env python3
"""Freeze a completed global-iteration checkpoint before the next segment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROBE_ROOT = HERE.parent
SHARED = PROBE_ROOT.parent / "05_training_dataset_bridge" / "scripts"
sys.path.insert(0, str(SHARED))

from bridge_common import (  # noqa: E402
    file_tree_receipt,
    read_json,
    sha256_file,
    utc_now,
    validate_file_tree_receipt,
    write_json_atomic,
)


RESUME_BOUNDARIES = {1785, 3570}
FINAL_ITERATION = 5960


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--training-assets-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="freeze synthetic smoke boundaries 1 or 2 instead of production boundaries",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    allowed = {1, 2} if args.smoke else RESUME_BOUNDARIES | {FINAL_ITERATION}
    if args.iteration not in allowed:
        raise ValueError("iteration is not a frozen segment boundary")
    assets_path = args.training_assets_receipt.resolve()
    assets = read_json(assets_path)
    if (
        assets.get("schema_version") != "greek_cpt_training_assets_receipt_v1"
        or assets.get("status") != "frozen"
    ):
        raise ValueError("training assets receipt schema drift")
    checkpoint_dir = args.checkpoint_dir.resolve()
    marker = checkpoint_dir / "latest_checkpointed_iteration.txt"
    if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != str(args.iteration):
        raise ValueError("checkpoint marker is not the requested global iteration")
    iteration_dir = checkpoint_dir / f"iter_{args.iteration:07d}"
    if not iteration_dir.is_dir() or not (iteration_dir / ".metadata").is_file():
        raise ValueError("complete torch_dist iteration checkpoint is absent")
    tree = file_tree_receipt(checkpoint_dir)
    payload = {
        "schema_version": "greek_cpt_resume_checkpoint_receipt_v1",
        "status": "frozen",
        "completed_at": utc_now(),
        "iteration": args.iteration,
        "smoke": args.smoke,
        "terminal": args.iteration == (2 if args.smoke else FINAL_ITERATION),
        "training_assets_receipt": {
            "path": str(assets_path),
            "sha256": sha256_file(assets_path),
        },
        "checkpoint_tree": tree,
        "marker": {"path": str(marker), "sha256": sha256_file(marker), "value": str(args.iteration)},
        "iteration_directory": str(iteration_dir),
    }
    if args.output.exists():
        existing = read_json(args.output.resolve())
        if (
            existing.get("iteration") != args.iteration
            or bool(existing.get("smoke", False)) != args.smoke
            or existing.get("terminal") != payload["terminal"]
            or existing.get("training_assets_receipt")
            != payload["training_assets_receipt"]
            or existing.get("checkpoint_tree", {}).get("tree_sha256") != tree["tree_sha256"]
        ):
            raise ValueError("existing checkpoint receipt has different bindings")
        validate_file_tree_receipt(existing["checkpoint_tree"])
        print(json.dumps({"ok": True, "resumed": True, "output": str(args.output.resolve())}, sort_keys=True))
        return 0
    write_json_atomic(args.output.resolve(), payload)
    print(json.dumps({"ok": True, "output": str(args.output.resolve()), "iteration": args.iteration}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
