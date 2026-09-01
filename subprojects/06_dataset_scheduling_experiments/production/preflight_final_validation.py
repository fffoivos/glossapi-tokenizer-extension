#!/usr/bin/env python3
"""Build a five-arm final-validation preflight from the passed segment-1 receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from campaign_contract import (
    ARMS,
    TOTAL_ITERATIONS,
    atomic_write_json,
    read_json,
    sha256_file,
    verify_code_bundle_receipt,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-receipt", type=Path, required=True)
    parser.add_argument("--runtime-scientific-bundle", type=Path)
    parser.add_argument("--runtime-scientific-bundle-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime_override = args.runtime_scientific_bundle is not None
    if runtime_override != (args.runtime_scientific_bundle_receipt is not None):
        raise ValueError("runtime scientific bundle and receipt must be supplied together")
    if runtime_override:
        verify_code_bundle_receipt(
            args.runtime_scientific_bundle_receipt,
            args.runtime_scientific_bundle,
            "scientific",
        )
    campaign = read_json(args.campaign_manifest)
    receipt = read_json(args.checkpoint_receipt)
    rows = {row["arm_id"]: row for row in receipt.get("arms", [])}
    if (
        receipt.get("schema_version") != "apertus_mini_segment_checkpoint_receipt_v1"
        or receipt.get("status") != "passed"
        or int(receipt.get("segment_id", -1)) != 1
        or int(receipt.get("iteration", -1)) != TOTAL_ITERATIONS
        or receipt.get("campaign_manifest_sha256") != sha256_file(args.campaign_manifest)
        or tuple(sorted(rows)) != tuple(sorted(ARMS))
    ):
        raise ValueError("final checkpoint receipt drift")
    load_roots = {arm: rows[arm]["checkpoint_root"] for arm in ARMS}
    for arm, root_text in load_roots.items():
        if not (Path(root_text) / f"iter_{TOTAL_ITERATIONS:07d}" / ".metadata").is_file():
            raise ValueError(f"incomplete final checkpoint for {arm}")
    payload = {
        "schema_version": "apertus_mini_segment_preflight_v1",
        "status": "passed",
        "purpose": "full_endpoint_validation",
        "segment_id": 1,
        "start_iteration": TOTAL_ITERATIONS,
        "end_iteration": TOTAL_ITERATIONS,
        "campaign_manifest": str(args.campaign_manifest.resolve()),
        "campaign_manifest_sha256": sha256_file(args.campaign_manifest),
        "load_roots": load_roots,
        "eval_iters": 5,
        "scheduled_tokens_per_panel": 5 * 512 * 4096,
        "runtime_scientific_bundle": (
            str(args.runtime_scientific_bundle.resolve()) if runtime_override else None
        ),
        "runtime_scientific_bundle_receipt": (
            str(args.runtime_scientific_bundle_receipt.resolve())
            if runtime_override
            else None
        ),
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
