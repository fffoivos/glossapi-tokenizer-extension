#!/usr/bin/env python3
"""Run one legacy proven segment from an audited postprocess recovery permit.

The frozen V99 runner predates typed terminal-postprocess adoption. This bridge
changes only predecessor selection: it verifies the exact passed recovery
permit and checkpoint reference, then delegates claim persistence and the
complete training path to the manifest-bound V99 engine.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-root", type=Path, required=True)
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--segment", required=True)
    parser.add_argument("--recovery-permit", type=Path, required=True)
    args = parser.parse_args()

    runner_root = args.runner_root.resolve()
    run_root = args.run_root.resolve()
    sys.path.insert(0, str(args.vendor_root.resolve()))
    sys.path.insert(0, str(runner_root / "src"))

    from apertus_cscs_campaign import engine  # noqa: PLC0415
    from apertus_cscs_campaign.receipts import (  # noqa: PLC0415
        file_binding,
        read_json,
        verify_binding,
    )

    manifest = engine.verify_compiled(args.manifest)
    if Path(str(manifest["code_bundle"]["root"])).resolve() != runner_root:
        raise ValueError("manifest code bundle differs from compatibility runner")
    if manifest["runtime"]["status"] != "proven":
        raise ValueError("postprocess recovery requires a proven runtime")
    segment = engine.find_segment(manifest, args.segment)
    release = engine.release_gate_state(segment)
    if release.get("state") != "released":
        raise ValueError(f"segment release gate is not released: {release}")
    prior_attempts = engine.segment_attempt_numbers(run_root, args.segment)
    for prior_attempt in prior_attempts:
        prior_root = engine.segment_attempt_root(run_root, args.segment, prior_attempt)
        execution = read_json(prior_root / "execution.json")
        if execution.get("status") != "failed" or execution.get("checkpoint") is not None:
            raise ValueError("target segment has a non-retryable prior attempt")

    permit = read_json(args.recovery_permit)
    checks = {
        "schema": permit.get("schema_version") == "apertus_checkpoint_permit_v2",
        "status": permit.get("status") == "passed",
        "campaign": permit.get("campaign_id") == manifest["campaign_id"],
        "science": permit.get("scientific_digest") == manifest["scientific_digest"],
        "contract": permit.get("contract_digest") == manifest["contract_digest"],
        "source": permit.get("source_segment_id") == "s3",
        "target": permit.get("target_segment_id") == args.segment,
        "source_attempt": permit.get("source_attempt") == 3,
        "target_start": int(segment["start_iteration"]) == 3218,
    }
    if not all(checks.values()):
        raise ValueError(f"recovery permit identity drift: {checks}")
    for label in ("checkpoint", "postprocess_recovery", "source_execution_receipt"):
        verify_binding(permit[label])

    reference_path = verify_binding(permit["checkpoint"])
    reference = read_json(reference_path)
    reference_checks = {
        "schema": reference.get("schema_version")
        == "apertus_hard_h_to_g_checkpoint_reference_v1",
        "status": reference.get("status") == "passed",
        "scale": reference.get("scale") == "8b",
        "phase": reference.get("phase") == 2,
        "update": reference.get("update") == 3218,
        "claimed_end": reference.get("claimed_end_update") == 3218,
    }
    if not all(reference_checks.values()):
        raise ValueError(f"checkpoint reference drift: {reference_checks}")
    for label in ("checkpoint_audit", "checkpoint_permit", "source_phase_cache_receipt"):
        verify_binding(reference[label])
    load_root = Path(str(reference["load_root"])).resolve()
    tracker = load_root / "latest_checkpointed_iteration.txt"
    if tracker.read_text(encoding="utf-8").strip() != "3218":
        raise ValueError("checkpoint tracker is not exactly update 3218")
    if not (load_root / "iter_0003218").is_dir():
        raise ValueError("update-3218 checkpoint directory is absent")

    attempt = max(prior_attempts, default=0) + 1
    root = engine.segment_attempt_root(run_root, args.segment, attempt)
    # The legacy canonical trainer keys its DCP runtime receipt only by Slurm
    # job id. A retry inside the same held allocation must preserve the prior
    # immutable receipt under a unique archival name before producing the new
    # attempt's identical check at the legacy path.
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    if not slurm_job_id.isdigit():
        raise ValueError("retry recovery requires a numeric SLURM_JOB_ID")
    compatibility = (
        run_root.parent.parent
        / "receipts"
        / "train_preflight"
        / f"dcp_metadata_compat_{slurm_job_id}.json"
    )
    if compatibility.exists():
        suffix = 1
        while True:
            archived = compatibility.with_name(
                f"{compatibility.stem}.retry_rotation_{suffix:03d}.json"
            )
            if not archived.exists():
                compatibility.replace(archived)
                break
            suffix += 1
    nonce = secrets.token_hex(8)
    claim = {
        "schema_version": "apertus_campaign_claim_v2",
        "status": "claimed",
        "recorded_at": engine.utc_now(),
        **engine.common_identity(
            manifest, args.segment, attempt, qualification_only=False
        ),
        "claim_nonce": nonce,
        "role": "train",
        "action": "submit_segment",
        "segment_id": args.segment,
        "start_iteration": int(segment["start_iteration"]),
        "end_iteration": int(segment["end_iteration"]),
        "load_checkpoint": str(reference_path),
        "recovery_permit": file_binding(args.recovery_permit),
    }
    engine.write_evidence(manifest, run_root, root / "claim.json", claim)

    updates = {
        "APERTUS_CAMPAIGN_ATTEMPT": str(attempt),
        "APERTUS_CAMPAIGN_CLAIM_NONCE": nonce,
        "APERTUS_CAMPAIGN_CODE_ROOT": str(runner_root),
        "APERTUS_CAMPAIGN_MANIFEST": str(args.manifest.resolve()),
        "APERTUS_CAMPAIGN_QUALIFICATION_ONLY": "0",
        "APERTUS_CAMPAIGN_QUALIFY_AND_CONTINUE": "0",
        "APERTUS_CAMPAIGN_RUN_ROOT": str(run_root),
        "APERTUS_CAMPAIGN_TARGET_ID": args.segment,
    }
    previous = {name: os.environ.get(name) for name in updates}
    os.environ.update(updates)
    try:
        return engine.run_training(
            args.manifest,
            run_root,
            args.segment,
            attempt,
            qualification_only=False,
            load_checkpoint_override=str(reference_path),
        )
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


if __name__ == "__main__":
    raise SystemExit(main())
