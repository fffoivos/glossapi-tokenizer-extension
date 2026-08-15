#!/usr/bin/env python3
"""Build canonical gate context from one completed exact-profile attempt."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from build_canonical_campaign_contracts import PROFILE_CHECKS
from contract_utils import (
    file_binding,
    read_json,
    require,
    require_file_binding,
    write_json_atomic,
)


def matching_input(manifest: dict[str, Any], item_id: str) -> dict[str, Any]:
    rows = manifest["campaign"]["science"]["immutable_inputs"]
    matches = [row for row in rows if row.get("id") == item_id]
    require(len(matches) == 1, f"immutable input is not unique: {item_id}")
    require_file_binding(matches[0])
    return matches[0]


def require_identity(
    value: dict[str, Any],
    manifest: dict[str, Any],
    *,
    schema: str,
    target: str,
    attempt: int,
) -> None:
    require(value.get("schema_version") == schema, f"{schema}: schema drift")
    require(value.get("campaign_id") == manifest["campaign_id"], "campaign drift")
    require(value.get("target_id", value.get("segment_id")) == target, "target drift")
    require(int(value.get("attempt", -1)) == attempt, "attempt drift")
    require(
        value.get("scientific_digest", manifest["scientific_digest"])
        == manifest["scientific_digest"],
        "scientific digest drift",
    )
    require(
        value.get("operational_digest", manifest["operational_digest"])
        == manifest["operational_digest"],
        "operational digest drift",
    )
    require(
        value.get("contract_digest") == manifest["contract_digest"],
        "contract digest drift",
    )


def validate_profile_receipt(
    path: Path,
    *,
    manifest: dict[str, Any],
    scale: str,
) -> dict[str, Any]:
    value = read_json(path)
    require(
        value.get("schema_version") == "apertus_hard_h_to_g_profile_benchmark_v1"
        and value.get("status") == "passed"
        and value.get("scale") == scale,
        "qualification profile receipt identity drift",
    )
    checks = value.get("checks")
    require(
        isinstance(checks, dict)
        and set(checks) == set(PROFILE_CHECKS)
        and all(checks[name] is True for name in PROFILE_CHECKS),
        "qualification profile checks did not all pass",
    )
    profile = value.get("profile")
    runtime = manifest["runtime"]
    require(isinstance(profile, dict), "qualification profile geometry missing")
    expected = {
        "nodes": runtime["slurm"]["nodes"],
        "gpus_per_node": runtime["slurm"]["gpus_per_node"],
        "tensor_parallel": runtime["parallelism"]["tensor"],
        "pipeline_parallel": runtime["parallelism"]["pipeline"],
        "context_parallel": runtime["parallelism"]["context"],
        "data_parallel": runtime["parallelism"]["data"],
    }
    require(
        all(int(profile.get(key, -1)) == int(expected[key]) for key in expected),
        "qualification profile geometry differs from candidate runtime",
    )
    measurement = value.get("measurement")
    require(
        isinstance(measurement, dict)
        and int(measurement.get("updates", -1)) == 256
        and int(measurement.get("discarded_warmup_updates", -1)) == 32
        and float(measurement.get("median_step_seconds", 0)) > 0
        and float(measurement.get("p90_step_seconds", 0)) > 0
        and float(measurement.get("tokens_per_gpu_hour", 0)) > 0,
        "qualification measurement is incomplete",
    )
    evidence = value.get("evidence")
    require(isinstance(evidence, list) and evidence, "profile evidence missing")
    for binding in evidence:
        require_file_binding(binding)

    scientific_binding = matching_input(manifest, "scientific_code_receipt")
    scientific_receipt = read_json(Path(str(scientific_binding["path"])))
    bundle = value.get("executing_code_bundle")
    require(
        isinstance(bundle, dict)
        and Path(str(bundle.get("root", ""))).resolve()
        == Path(str(scientific_receipt.get("root", ""))).resolve()
        and bundle.get("tree_sha256") == scientific_receipt.get("tree_sha256"),
        "profile benchmark scientific bundle drift",
    )
    return value


def build_context(
    *,
    manifest: dict[str, Any],
    run_root: Path,
    scale: str,
    segment_attempt_root: Any,
    segment_attempt_numbers: Any,
) -> dict[str, Any]:
    require(
        manifest["runtime"]["status"] == "candidate",
        "qualification requires a candidate runtime",
    )
    attempts = segment_attempt_numbers(run_root, "s0")
    qualifying = [
        attempt
        for attempt in attempts
        if (
            segment_attempt_root(run_root, "s0", attempt)
            / "qualification_checkpoint.json"
        ).is_file()
    ]
    require(len(qualifying) == 1, "qualification checkpoint attempt is not unique")
    attempt = qualifying[0]
    root = segment_attempt_root(run_root, "s0", attempt)
    required = {
        name: root / name
        for name in (
            "claim.json",
            "submission.json",
            "completion.json",
            "execution.json",
            "qualification_checkpoint.json",
            "profile_benchmark.json",
        )
    }
    require(
        all(path.is_file() for path in required.values()),
        "qualification attempt evidence is incomplete",
    )

    claim = read_json(required["claim.json"])
    require_identity(
        claim,
        manifest,
        schema="apertus_campaign_claim_v2",
        target="s0",
        attempt=attempt,
    )
    require(
        claim.get("qualification_only") is True
        and claim.get("role") == "train"
        and claim.get("action") == "submit_segment"
        and int(claim.get("start_iteration", -1)) == 0,
        "qualification claim mode drift",
    )

    submission = read_json(required["submission.json"])
    require_identity(
        submission,
        manifest,
        schema="apertus_campaign_submission_v2",
        target="s0",
        attempt=attempt,
    )
    job = submission.get("job")
    require(
        submission.get("status") == "submitted"
        and submission.get("qualification_only") is True
        and submission.get("role") == "train"
        and isinstance(job, dict)
        and isinstance(job.get("checks"), dict)
        and bool(job["checks"])
        and all(job["checks"].values()),
        "qualification Slurm audit did not pass",
    )

    completion = read_json(required["completion.json"])
    require(
        completion.get("schema_version") == "apertus_hard_h_to_g_training_completion_v1"
        and completion.get("status") == "checkpointed"
        and completion.get("campaign_id") == manifest["campaign_id"]
        and completion.get("segment_id") == "s0"
        and int(completion.get("attempt", -1)) == attempt
        and completion.get("contract_digest") == manifest["contract_digest"]
        and int(completion.get("observed_iteration", -1)) == 256,
        "qualification completion receipt drift",
    )

    execution = read_json(required["execution.json"])
    require_identity(
        execution,
        manifest,
        schema="apertus_campaign_training_attempt_v2",
        target="s0",
        attempt=attempt,
    )
    require(
        execution.get("status") == "gracefully_stopped"
        and execution.get("qualification_only") is True
        and int(execution.get("resume_iteration", -1)) == 256
        and int(execution.get("returncode", -1)) == 0
        and execution.get("minimum_runtime_satisfied") is True
        and execution.get("signal") == "SIGUSR1",
        "qualification execution did not stop at the bounded checkpoint",
    )

    checkpoint = read_json(required["qualification_checkpoint.json"])
    require(
        checkpoint.get("schema_version")
        == "apertus_hard_h_to_g_runtime_qualification_checkpoint_v1"
        and checkpoint.get("status") == "passed"
        and checkpoint.get("scale") == scale
        and int(checkpoint.get("update", -1)) == 256,
        "qualification checkpoint identity drift",
    )
    require_file_binding(checkpoint["tracker"])
    require_file_binding(checkpoint["metadata"])
    require_file_binding(
        checkpoint["profile_benchmark"],
        expected_path=required["profile_benchmark.json"],
    )
    require_file_binding(checkpoint["benchmark_contract"])
    expected_checkpoint = file_binding(required["qualification_checkpoint.json"])
    expected_profile = file_binding(required["profile_benchmark.json"])
    require(
        completion.get("checkpoint") == expected_checkpoint,
        "completion checkpoint binding drift",
    )
    require(
        completion.get("metrics") == expected_profile,
        "completion metrics binding drift",
    )
    require(
        execution.get("checkpoint") == expected_checkpoint,
        "execution checkpoint binding drift",
    )
    require(
        execution.get("completion_receipt")
        == file_binding(required["completion.json"]),
        "execution completion binding drift",
    )

    validate_profile_receipt(
        required["profile_benchmark.json"], manifest=manifest, scale=scale
    )
    return {
        "schema_version": "apertus_gate_context_v1",
        "status": "passed",
        "campaign_id": manifest["campaign_id"],
        "scale": scale,
        "qualification_attempt": attempt,
        "qualification_checkpoint": expected_checkpoint,
        "completion": file_binding(required["completion.json"]),
        "execution": file_binding(required["execution.json"]),
        "evidence": {
            "profile_measurement": expected_profile,
            "slurm_profile": file_binding(required["submission.json"]),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--scale", choices=("8b", "1p5b"), required=True)
    parser.add_argument("--canonical-runner-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    canonical_src = args.canonical_runner_root.resolve() / "src"
    require(canonical_src.is_dir(), "canonical runner source root missing")
    sys.path.insert(0, str(canonical_src))
    from apertus_cscs_campaign.engine import verify_compiled
    from apertus_cscs_campaign.state import (
        segment_attempt_numbers,
        segment_attempt_root,
    )

    manifest = verify_compiled(args.manifest.resolve())
    value = build_context(
        manifest=manifest,
        run_root=args.run_root.resolve(),
        scale=args.scale,
        segment_attempt_root=segment_attempt_root,
        segment_attempt_numbers=segment_attempt_numbers,
    )
    write_json_atomic(args.output, value)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
