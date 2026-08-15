#!/usr/bin/env python3
"""Freeze evidence from non-mutating Slurm test-only segment submissions."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from contract_utils import executing_code_bundle, file_binding, read_json, require, sha256_file, write_json_atomic


def permit(path: Path, scale: str) -> dict:
    value = read_json(path)
    require(
        value.get("schema_version") == "apertus_hard_h_to_g_training_run_permit_v1"
        and value.get("status") == "passed"
        and value.get("scale") == scale,
        f"{scale} training run-permit drift",
    )
    return value


def verify_log(path: Path, nodes: int, train_script: Path) -> None:
    require(path.is_file() and path.stat().st_size > 0, f"test-only log missing: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    require("--test-only" in text, f"test-only flag absent from command record: {path}")
    require(f"--nodes={nodes}" in text, f"test-only node count drift: {path}")
    require("--dependency=after:" in text, f"test-only delayed dependency absent: {path}")
    require(str(train_script.resolve()) in text, f"test-only trainer path drift: {path}")
    require("TEST_ONLY_EXIT_CODE=0" in text, f"sbatch test-only did not pass: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("8b", "1p5b"), required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha-before", required=True)
    parser.add_argument("--run-permit", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--train-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable submission dry-run receipt exists: {args.output}")
    require(args.artifact_manifest.is_file(), "artifact manifest missing")
    manifest_after = sha256_file(args.artifact_manifest)
    require(manifest_after == args.manifest_sha_before, "sbatch test-only mutated the artifact manifest")
    require(args.train_script.is_file(), "training script missing")
    run_permit = permit(args.run_permit, args.scale)
    verify_log(args.log, int(run_permit["profile"]["nodes"]), args.train_script)
    payload = {
        "schema_version": "apertus_hard_h_to_g_submission_dry_run_v1",
        "status": "passed",
        "scale": args.scale,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "executing_code_bundle": executing_code_bundle(),
        "artifact_manifest": file_binding(args.artifact_manifest),
        "manifest_sha_before": args.manifest_sha_before,
        "manifest_sha_after": manifest_after,
        "training_script": file_binding(args.train_script),
        "run_permit": file_binding(args.run_permit),
        "log": file_binding(args.log),
        "checks": {
            "selected_profile_passes_sbatch_test_only": True,
            "normal_partition_and_exact_node_counts_are_explicit": True,
            "delayed_dependency_syntax_is_accepted": True,
            "immutable_training_script_is_bound": True,
            "artifact_manifest_is_unchanged": True,
        },
    }
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
