#!/usr/bin/env python3
"""Rebind a partial preauthorization manifest to a compatible code bundle."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
from pathlib import Path

from contract_utils import executing_code_bundle, file_binding, read_json, require, write_json_atomic
from freeze_hard_h_to_g_contract import artifacts_for_stage, validate_artifact_manifest
from producer_bundle_compatibility import load_authority


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--producer-compatibility", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = read_json(args.source_manifest.resolve())
    require(
        source.get("schema_version") == "apertus_hard_h_to_g_artifact_manifest_v1"
        and source.get("status") == "frozen"
        and source.get("partial") is True,
        "source preauthorization manifest drift",
    )
    stage = str(source.get("gate_stage"))
    scale = source.get("scale")
    require(stage in {"pre_main", "pre_extension", "pre_second_extension"}, "manifest stage drift")
    require(scale in {"8b", "1p5b"} if stage == "pre_main" else scale is None, "manifest scale drift")

    current = executing_code_bundle()
    compatibility = args.producer_compatibility.resolve()
    _, accepted = load_authority(compatibility, current)
    owner_roles = {"owner_production_authorization", "owner_extension_authorization"}
    required = [
        role
        for role in artifacts_for_stage(stage, scale if stage == "pre_main" else None)
        if role not in owner_roles
    ]
    _, blockers = validate_artifact_manifest(
        args.source_manifest.resolve(),
        required,
        accepted_producers=accepted,
        expected_scale=scale if stage == "pre_main" else None,
    )
    require(not blockers, f"source artifacts fail under new authority: {blockers}")

    output = copy.deepcopy(source)
    output["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    output["executing_code_bundle"] = current
    output["producer_bundle_compatibility"] = file_binding(compatibility)
    output["rebind_provenance"] = {
        "source_manifest": file_binding(args.source_manifest.resolve()),
        "artifact_manifest_revalidated": True,
        "all_artifact_producers_compatibility_authorized": True,
    }
    write_json_atomic(args.output.resolve(), output)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
