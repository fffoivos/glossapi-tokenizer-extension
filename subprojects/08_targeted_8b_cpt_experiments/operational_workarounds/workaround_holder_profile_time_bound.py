#!/usr/bin/env python3
"""Run a recovered holder with a profile-specific measured time bound.

The promoted two-node 1.5B manifest retained the one-node handoff estimate
(33,967 seconds), even though the only promoted runtime is the audited
two-node DP8 profile and the live allocation is five hours.  This adapter
keeps the immutable manifest, permit, science digest and canonical runner;
it overrides only the stale in-memory holder timing after checking the exact
campaign/profile/segment and the original value.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path


EXPECTED_PROFILE = "hard_h2g_1p5b_tp1_dp8_2node_mb4"
EXPECTED_ORIGINAL_TARGET_SECONDS = 33_967


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--segment", required=True)
    parser.add_argument("--canonical-permit", type=Path, required=True)
    parser.add_argument("--legacy-recovery-permit", type=Path, required=True)
    parser.add_argument("--target-runtime-seconds", type=int, required=True)
    parser.add_argument("--reserve-seconds", type=int, required=True)
    parser.add_argument("--minimum-train-seconds", type=int, required=True)
    parser.add_argument("--srun-wrapper-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.segment != "s3":
        raise ValueError("this workaround is restricted to 1.5B segment s3")
    if not 15_840 <= args.target_runtime_seconds <= 16_500:
        raise ValueError("target runtime must remain within the audited 4.4h plus bounded reserve envelope")
    if not 600 <= args.reserve_seconds <= 900:
        raise ValueError("reserve must remain between 10 and 15 minutes")
    if not 14_400 <= args.minimum_train_seconds <= 15_000:
        raise ValueError("minimum train time must remain within the audited two-node envelope")
    srun_wrapper = args.srun_wrapper_dir / "srun"
    if not srun_wrapper.is_file() or not os.access(srun_wrapper, os.X_OK):
        raise ValueError("executable experiment-local srun wrapper is missing")

    sys.path.insert(0, str(args.runner_root / "src"))
    sys.path.insert(0, str(args.runner_root / "src/_vendor/campaign_pydeps"))
    from apertus_cscs_campaign import engine
    from apertus_cscs_campaign.receipts import file_binding, read_json

    canonical = read_json(args.canonical_permit)
    recovery = read_json(args.legacy_recovery_permit)
    checks = {
        "canonical_schema": canonical.get("schema_version") == "apertus_checkpoint_permit_v2",
        "canonical_status": canonical.get("status") == "passed",
        "canonical_target": canonical.get("target_segment_id") == args.segment,
        "recovery_binding": canonical.get("recovery_permit") == file_binding(args.legacy_recovery_permit),
        "recovery_schema": recovery.get("schema_version") == "apertus_checkpoint_permit_v2",
        "recovery_status": recovery.get("status") == "passed",
        "same_campaign": canonical.get("campaign_id") == recovery.get("campaign_id"),
        "same_science": canonical.get("scientific_digest") == recovery.get("scientific_digest"),
        "same_source_attempt": canonical.get("source_attempt") == recovery.get("source_attempt"),
    }
    if not all(checks.values()):
        raise ValueError(f"legacy recovery permit binding drift: {checks}")

    original_verify = engine.verify_compiled
    original_latest = engine.latest_permit

    def verified_manifest(path: Path):
        manifest = copy.deepcopy(original_verify(path))
        if manifest.get("campaign_id") != "hard-h2g-1p5b-matched-r2":
            raise ValueError("campaign drift")
        if manifest.get("runtime", {}).get("profile_id") != EXPECTED_PROFILE:
            raise ValueError("promoted runtime-profile drift")
        segment = next(row for row in manifest["campaign"]["segments"] if row["id"] == args.segment)
        handoff = segment.get("handoff", {})
        if int(handoff.get("conservative_target_runtime_seconds", -1)) != EXPECTED_ORIGINAL_TARGET_SECONDS:
            raise ValueError("stale timing value changed; refuse rather than stack overrides")
        if int(segment.get("minimum_train_seconds", -1)) != 19_020:
            raise ValueError("stale minimum-train value changed; refuse rather than stack overrides")
        handoff["conservative_target_runtime_seconds"] = args.target_runtime_seconds
        handoff["reserve_seconds"] = args.reserve_seconds
        segment["minimum_train_seconds"] = args.minimum_train_seconds
        return manifest

    def select_permit(run_root: Path, target_segment_id: str):
        if run_root.resolve() == args.run_root.resolve() and target_segment_id == args.segment:
            return args.canonical_permit.resolve()
        return original_latest(run_root, target_segment_id)

    engine.verify_compiled = verified_manifest
    engine.latest_permit = select_permit
    os.environ["APERTUS_CAMPAIGN_MANIFEST"] = str(args.manifest.resolve())
    os.environ["APERTUS_CAMPAIGN_CODE_ROOT"] = str(args.runner_root.resolve())
    os.environ["APERTUS_CAMPAIGN_RUN_ROOT"] = str(args.run_root.resolve())
    os.environ["APERTUS_CAMPAIGN_TARGET_ID"] = args.segment
    os.environ["APERTUS_CAMPAIGN_QUALIFICATION_ONLY"] = "0"
    os.environ["PATH"] = f"{args.srun_wrapper_dir.resolve()}:{os.environ['PATH']}"
    return engine.run_holder_in_allocation(
        args.manifest,
        args.run_root,
        args.segment,
        poll_seconds=5,
    )


if __name__ == "__main__":
    raise SystemExit(main())
