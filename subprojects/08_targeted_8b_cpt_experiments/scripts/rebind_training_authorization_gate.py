#!/usr/bin/env python3
"""Rebind a passed training gate to a compatibility-authorized code bundle."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
from pathlib import Path

from contract_utils import (
    executing_code_bundle,
    file_binding,
    read_json,
    require,
    write_json_atomic,
)
from freeze_hard_h_to_g_contract import artifacts_for_stage, validate_artifact_manifest
from producer_bundle_compatibility import load_authority


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-gate", type=Path, required=True)
    parser.add_argument("--new-experiment", type=Path, required=True)
    parser.add_argument("--producer-compatibility", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = read_json(args.source_gate.resolve())
    require(
        source.get("schema_version") == "apertus_hard_h_to_g_frozen_contract_v2"
        and source.get("status") == "launch_ready"
        and source.get("mode") == "launch"
        and source.get("blockers") == []
        and source.get("all_gates_receipt_backed") is True,
        "source authorization gate is not launch-ready",
    )
    stage = str(source.get("gate_stage"))
    scale = source.get("scale")
    require(stage in {"pre_main", "pre_extension", "pre_second_extension"}, "gate stage drift")
    require(scale in {"8b", "1p5b"} if stage == "pre_main" else scale is None, "gate scale drift")

    old_experiment = source.get("experiment")
    require(isinstance(old_experiment, dict), "source experiment binding missing")
    old_experiment_path = Path(str(old_experiment.get("path", "")))
    require(
        old_experiment_path.is_file()
        and old_experiment == file_binding(old_experiment_path),
        "source experiment binding drift",
    )
    new_experiment = file_binding(args.new_experiment.resolve())
    require(
        (old_experiment["bytes"], old_experiment["sha256"])
        == (new_experiment["bytes"], new_experiment["sha256"]),
        "experiment bytes changed during gate rebind",
    )

    current = executing_code_bundle()
    _, accepted = load_authority(args.producer_compatibility.resolve(), current)
    manifest_binding = source.get("artifact_manifest")
    require(isinstance(manifest_binding, dict), "source artifact manifest missing")
    manifest_path = Path(str(manifest_binding.get("path", "")))
    require(
        manifest_path.is_file() and manifest_binding == file_binding(manifest_path),
        "source artifact-manifest binding drift",
    )
    _, blockers = validate_artifact_manifest(
        manifest_path,
        artifacts_for_stage(stage, scale if stage == "pre_main" else None),
        accepted_producers=accepted,
        expected_scale=scale if stage == "pre_main" else None,
    )
    require(not blockers, f"artifact manifest does not pass under new authority: {blockers}")

    output = copy.deepcopy(source)
    output["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    output["executing_code_bundle"] = current
    output["experiment"] = new_experiment
    output["producer_bundle_compatibility"] = file_binding(
        args.producer_compatibility.resolve()
    )
    output["rebind_provenance"] = {
        "source_gate": file_binding(args.source_gate.resolve()),
        "experiment_bytes_identical": True,
        "artifact_manifest_revalidated": True,
        "all_artifact_producers_compatibility_authorized": True,
    }
    write_json_atomic(args.output.resolve(), output)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
