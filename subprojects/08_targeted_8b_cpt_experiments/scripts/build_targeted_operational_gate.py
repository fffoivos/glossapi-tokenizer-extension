#!/usr/bin/env python3
"""Bind targeted science gates to the proven resource-aware operations bundle."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from contract_utils import file_binding, read_json, require, write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("A", "B"), required=True)
    parser.add_argument("--scientific-root", type=Path, required=True)
    parser.add_argument("--scientific-receipt", type=Path, required=True)
    parser.add_argument("--operational-root", type=Path, required=True)
    parser.add_argument("--operational-receipt", type=Path, required=True)
    parser.add_argument("--launch-gate", type=Path, required=True)
    parser.add_argument("--selected-profile", type=Path, required=True)
    parser.add_argument("--launch-environment", type=Path, required=True)
    parser.add_argument("--nested-sbatch-proof", type=Path, required=True)
    parser.add_argument("--prequeue-schedule", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable operational gate exists: {args.output}")
    sys.path.insert(0, str(args.scientific_root / "subprojects/07_full_8b_cpt/scripts"))
    from contract import verify_code_bundle_receipt  # pylint: disable=import-error,import-outside-toplevel

    scientific = verify_code_bundle_receipt(args.scientific_receipt, args.scientific_root, "scientific")
    operational = verify_code_bundle_receipt(args.operational_receipt, args.operational_root, "efficiency")
    launch = read_json(args.launch_gate)
    selected = read_json(args.selected_profile)
    environment = read_json(args.launch_environment)
    nested = read_json(args.nested_sbatch_proof)
    require(
        launch.get("schema_version") == "apertus_full_8b_launch_gate_v1"
        and launch.get("status") == "passed"
        and launch.get("experiment") == args.experiment
        and launch.get("executing_code_bundle")
        == {"root": str(args.scientific_root.resolve()), "tree_sha256": scientific["tree_sha256"]},
        "targeted scientific launch gate drift",
    )
    require(
        selected.get("schema_version") == "apertus_full_8b_selected_execution_profile_v1"
        and selected.get("status") == "frozen"
        and selected.get("selection") == launch.get("selected_profile"),
        "selected-profile/launch-gate drift",
    )
    selection = selected["selection"]
    require(
        selection.get("profile_id") == "dp32_16node"
        and int(selection.get("nodes", -1)) == 16
        and int(selection.get("data_parallel", -1)) == 32,
        "operational DP32 geometry drift",
    )
    require(
        environment.get("schema_version") == "apertus_full_8b_launch_environment_v1"
        and environment.get("status") == "passed"
        and int(environment.get("nodes", -1)) == 16
        and "--switches=1" in environment.get("test_only", {}).get("command", []),
        "launch-environment drift",
    )
    require(
        nested.get("schema_version") == "apertus_full_8b_nested_sbatch_proof_v1"
        and nested.get("status") == "passed"
        and Path(nested.get("code_root", "")).resolve() == args.scientific_root.resolve()
        and Path(nested.get("code_bundle_receipt", "")).resolve() == args.scientific_receipt.resolve()
        and nested.get("nested_submit_flag") == "--uenv-passthrough=ignore",
        "nested-submit proof drift",
    )
    required_ops = (
        "clariden/supervise_campaign_resource_aware.sbatch",
        "clariden/run_prequeued_train_holder.sbatch",
        "scripts/supervise_campaign_resource_aware.py",
        "scripts/prequeue_next_segment.py",
        "scripts/audit_submitted_job_resources.py",
    )
    require(all((args.operational_root / relative).is_file() for relative in required_ops), "operational bundle lacks required scripts")
    prequeue = None
    if args.experiment == "A":
        require(args.prequeue_schedule is not None, "A prequeue schedule is required")
        value = read_json(args.prequeue_schedule)
        require(
            value.get("schema_version") == "apertus_full_8b_prequeue_schedule_v1"
            and value.get("status") == "approved"
            and value.get("experiment") == "A"
            and len(value.get("segment_boundaries", [])) == 3
            and len(value.get("targets", [])) == 1
            and value.get("invariants", {}).get("at_most_one_delayed_successor") is True,
            "A prequeue schedule drift",
        )
        prequeue = file_binding(args.prequeue_schedule)
    else:
        require(args.prequeue_schedule is None, "B must not prequeue a successor allocation")
    payload = {
        "schema_version": "apertus_full_8b_operational_launch_gate_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment": args.experiment,
        "scientific_root": str(args.scientific_root.resolve()),
        "operational_root": str(args.operational_root.resolve()),
        "scientific_bundle": {"receipt": file_binding(args.scientific_receipt), "tree_sha256": scientific["tree_sha256"]},
        "operational_bundle": {"receipt": file_binding(args.operational_receipt), "tree_sha256": operational["tree_sha256"]},
        "launch_gate": file_binding(args.launch_gate),
        "selected_profile": file_binding(args.selected_profile),
        "launch_environment": file_binding(args.launch_environment),
        "nested_sbatch_proof": file_binding(args.nested_sbatch_proof),
        "prequeue_schedule": prequeue,
        "allocation_policy": {
            "normal_nodes": 16,
            "single_leaf": True,
            "wall_seconds": 43200,
            "dp64_allowed": False,
            "debug_supervisor": True,
            "maximum_pending_successors": 1 if args.experiment == "A" else 0,
        },
    }
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
