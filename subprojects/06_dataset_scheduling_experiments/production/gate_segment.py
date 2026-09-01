#!/usr/bin/env python3
"""Gate a segment on five checkpoints and all required evaluation receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from campaign_contract import (
    ARMS,
    AUTHORITATIVE_EVALUATION_DTYPE,
    atomic_write_json,
    evaluation_namespace,
    read_json,
    scoped_evaluation_root,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--segment-id", type=int, choices=(0, 1), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint-receipt", type=Path, required=True)
    parser.add_argument("--source-validation-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    campaign = read_json(args.campaign_manifest)
    checkpoint = read_json(args.checkpoint_receipt)
    end = int(campaign["segments"][args.segment_id]["end_iteration"])
    if (
        checkpoint.get("status") != "passed"
        or int(checkpoint.get("iteration", -1)) != end
        or checkpoint.get("campaign_manifest_sha256") != sha256_file(args.campaign_manifest)
        or tuple(row["arm_id"] for row in checkpoint.get("arms", [])) != ARMS
    ):
        raise ValueError("segment checkpoint receipt drift")
    namespace = evaluation_namespace()
    watch_summary_path = (
        scoped_evaluation_root(args.run_root, "evaluation_watch")
        / f"segment_{args.segment_id}"
        / "summary.json"
    )
    watch = read_json(watch_summary_path)
    start = int(campaign["segments"][args.segment_id]["start_iteration"])
    expected_iterations = [
        int(value)
        for value in campaign["evaluation"]["checkpoint_iterations"]
        if (start < int(value) <= end) or (args.segment_id == 0 and int(value) == 0)
    ]
    if (
        watch.get("status") != "completed"
        or watch.get("evaluation_namespace") != namespace
        or watch.get("authoritative_evaluation_dtype")
        != AUTHORITATIVE_EVALUATION_DTYPE
        or watch.get("iterations") != expected_iterations
        or int(watch.get("bindings", -1)) != len(expected_iterations) * len(ARMS)
    ):
        raise ValueError("segment evaluation watcher is incomplete")
    source_validation = read_json(args.source_validation_receipt)
    expected_source_bindings = (
        len(expected_iterations)
        * len(ARMS)
        * len(campaign["evaluation"]["validation_panels"])
    )
    if (
        source_validation.get("schema_version")
        != "apertus_mini_segment_source_validation_gate_v1"
        or source_validation.get("status") != "passed"
        or int(source_validation.get("segment_id", -1)) != args.segment_id
        or source_validation.get("iterations") != expected_iterations
        or source_validation.get("arms") != list(ARMS)
        or source_validation.get("panels")
        != campaign["evaluation"]["validation_panels"]
        or int(source_validation.get("binding_count", -1))
        != expected_source_bindings
        or source_validation.get("all_metrics_finite") is not True
        or source_validation.get("complete_panels_per_arm_iteration") is not True
        or source_validation.get("campaign_manifest", {}).get("sha256")
        != sha256_file(args.campaign_manifest)
    ):
        raise ValueError("segment source-conditioned validation is incomplete")
    payload = {
        "schema_version": "apertus_mini_segment_gate_v1",
        "status": "passed",
        "segment_id": args.segment_id,
        "iteration": end,
        "checkpoint_receipt": str(args.checkpoint_receipt.resolve()),
        "checkpoint_receipt_sha256": sha256_file(args.checkpoint_receipt),
        "evaluation_summary": str(watch_summary_path.resolve()),
        "evaluation_bindings": watch["bindings"],
        "evaluation_namespace": namespace,
        "authoritative_evaluation_dtype": AUTHORITATIVE_EVALUATION_DTYPE,
        "source_validation_receipt": str(args.source_validation_receipt.resolve()),
        "source_validation_receipt_sha256": sha256_file(
            args.source_validation_receipt
        ),
        "source_validation_bindings": expected_source_bindings,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
