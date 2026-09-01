#!/usr/bin/env python3
"""Freeze a hash-bound artifact manifest for one hard-H-to-G gate stage."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from contract_utils import executing_code_bundle, file_binding, read_json, require, write_json_atomic
from freeze_hard_h_to_g_contract import (
    ALL_ARTIFACTS,
    ARTIFACTS_BY_STAGE,
    PRE_MAIN_ARTIFACTS_BY_SCALE,
    artifacts_for_stage,
    validate_artifact_manifest,
)
from producer_bundle_compatibility import load_authority, require_accepted_producer


def artifact_assignment(value: str) -> tuple[str, Path]:
    role, separator, raw_path = value.partition("=")
    require(separator == "=" and role and raw_path, f"invalid --artifact assignment: {value}")
    require(role in ALL_ARTIFACTS, f"unknown artifact role: {role}")
    return role, Path(raw_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-stage", choices=tuple(ARTIFACTS_BY_STAGE), required=True)
    parser.add_argument("--scale", choices=tuple(PRE_MAIN_ARTIFACTS_BY_SCALE))
    parser.add_argument("--producer-compatibility", type=Path, required=True)
    parser.add_argument("--artifact", action="append", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable artifact manifest exists: {args.output}")
    current = executing_code_bundle()
    _, accepted = load_authority(args.producer_compatibility, current)
    assignments = [artifact_assignment(value) for value in args.artifact]
    roles = [role for role, _ in assignments]
    require(len(roles) == len(set(roles)), "duplicate artifact role")
    required = artifacts_for_stage(args.gate_stage, args.scale)
    require(set(roles) <= set(required), "artifact role is outside the selected gate stage")
    if not args.allow_partial:
        require(set(roles) == set(required), "artifact manifest does not contain the exact gate-stage role set")
    artifacts = {}
    for role, path in assignments:
        require(path.is_file(), f"artifact receipt missing: {path}")
        receipt = read_json(path)
        if role != "evaluation_code_bundle":
            require_accepted_producer(receipt, accepted, role)
        artifacts[role] = file_binding(path)
    payload = {
        "schema_version": "apertus_hard_h_to_g_artifact_manifest_v1",
        "status": "frozen",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gate_stage": args.gate_stage,
        "scale": args.scale,
        "partial": args.allow_partial,
        "executing_code_bundle": current,
        "producer_bundle_compatibility": file_binding(args.producer_compatibility),
        "artifacts": artifacts,
    }
    temporary = args.output.with_name(f".{args.output.name}.partial")
    require(not temporary.exists(), f"partial artifact manifest exists: {temporary}")
    try:
        write_json_atomic(temporary, payload)
        _, blockers = validate_artifact_manifest(
            temporary,
            roles if args.allow_partial else required,
            accepted_producers=accepted,
            expected_scale=args.scale,
        )
        require(not blockers, f"artifact manifest failed validation: {blockers}")
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
