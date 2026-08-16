#!/usr/bin/env python3
"""Qualify the fixed 1.5B profile and continue in the held allocation.

The canonical campaign runner owns the allocation and promotion. This adapter
runs the already-frozen scientific benchmark in that allocation and writes only
the two evidence objects consumed by the runner's fixed qualification gates.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from build_canonical_campaign_contracts import PROFILE_CHECKS, canonical_digest
from contract_utils import (
    file_binding,
    read_json,
    require,
    require_file_binding,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--canonical-runner-root", type=Path, required=True)
    parser.add_argument("--qualification-contract", type=Path, required=True)
    parser.add_argument("--qualification-root", type=Path, required=True)
    parser.add_argument("--qualification-context", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--contract-digest", required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--code-receipt", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--megatron-root", type=Path, required=True)
    parser.add_argument("--megatron-receipt", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--validation-receipt", type=Path, required=True)
    parser.add_argument("--extra-valid-sets", required=True)
    parser.add_argument("--new-greek-valid-sets", required=True)
    parser.add_argument(
        "--identity-only",
        action="store_true",
        help="validate the expanded qualification contract without requiring an allocation",
    )
    return parser.parse_args()


def matching_input(manifest: dict[str, Any], item_id: str, path: Path) -> None:
    rows = manifest["campaign"]["science"]["immutable_inputs"]
    matches = [row for row in rows if row.get("id") == item_id]
    require(len(matches) == 1, f"immutable input is not unique: {item_id}")
    require_file_binding(matches[0], expected_path=path)


def scontrol_value(raw: str, key: str) -> str:
    match = re.search(rf"(?:^|\s){re.escape(key)}=(\S+)", raw)
    return "" if match is None else match.group(1)


def main() -> int:
    args = parse_args()
    runner_src = args.canonical_runner_root.resolve() / "src"
    require(runner_src.is_dir(), "canonical runner source root missing")
    sys.path.insert(0, str(runner_src))
    from apertus_cscs_campaign.contracts import qualification_target
    from apertus_cscs_campaign.engine import verify_compiled
    from apertus_cscs_campaign.receipts import digest

    manifest_path = args.manifest.resolve()
    manifest = verify_compiled(manifest_path)
    require(manifest["runtime"]["status"] == "candidate", "runtime is not a candidate")
    require(
        manifest["campaign_id"] == args.campaign_id
        and manifest["contract_digest"] == args.contract_digest,
        "qualification campaign identity drift",
    )
    qualification_root = args.qualification_root.resolve()
    context_path = args.qualification_context.resolve()
    run_root = args.run_root.resolve()
    target_digest = digest(qualification_target(manifest["runtime"]))
    require(
        qualification_root == run_root / "qualification" / target_digest
        and context_path == qualification_root / "context.json",
        "canonical qualification path drift",
    )
    require(
        manifest["campaign"]["science"].get("runtime_requirements_sha256")
        == canonical_digest(manifest["runtime"]["qualification_scope"]),
        "runtime qualification scope digest drift",
    )

    matching_input(manifest, "scientific_code_receipt", args.code_receipt.resolve())
    matching_input(
        manifest, "qualification_contract", args.qualification_contract.resolve()
    )
    matching_input(manifest, "megatron_receipt", args.megatron_receipt.resolve())
    matching_input(
        manifest, "online_validation_receipt", args.validation_receipt.resolve()
    )

    contract = read_json(args.qualification_contract.resolve())
    require(
        contract.get("schema_version")
        == "apertus_hard_h_to_g_prelaunch_benchmark_contract_v1"
        and contract.get("status") == "frozen"
        and contract.get("kind") == "profile"
        and contract.get("scale") == "1p5b"
        and contract.get("profile_id") == "1p5b_tp1_1node",
        "fixed 1.5B qualification contract drift",
    )
    require(
        (
            int(contract.get("nodes", -1)),
            int(contract.get("tensor_parallel", -1)),
            int(contract.get("microbatch", -1)),
            int(contract.get("updates", -1)),
            str(contract.get("peak_lr")),
            str(contract.get("floor_lr")),
        )
        == (1, 1, 8, 256, "5.5e-5", "5.5e-6"),
        "fixed 1.5B qualification geometry drift",
    )
    if args.identity_only:
        print("qualification identity preflight: passed")
        return 0
    benchmark_root = qualification_root / "benchmark"
    require(
        Path(str(contract.get("output_root", ""))).resolve() == benchmark_root,
        "qualification benchmark output is not inside the canonical attempt root",
    )
    require(not benchmark_root.exists(), "qualification benchmark output already exists")

    runtime = manifest["runtime"]
    job_id = os.environ.get("SLURM_JOB_ID", "")
    raw_job = subprocess.run(
        ["scontrol", "show", "job", "-o", job_id],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    slurm_checks = {
        "job_id_present": bool(re.fullmatch(r"\d+", job_id)),
        "normal_partition": os.environ.get("SLURM_JOB_PARTITION") == "normal",
        "exact_live_nodes": int(os.environ.get("SLURM_NNODES", "0"))
        == int(runtime["slurm"]["nodes"]),
        "exact_profile_nodes": int(runtime["slurm"]["nodes"]) == 1,
        "exact_parallelism": runtime["parallelism"]
        == {"tensor": 1, "pipeline": 1, "context": 1, "data": 4},
        "scontrol_job_id": scontrol_value(raw_job, "JobId") == job_id,
        "scontrol_partition": scontrol_value(raw_job, "Partition") == "normal",
        "scontrol_nodes": scontrol_value(raw_job, "NumNodes") in {"1", "1-1"},
    }
    require(all(slurm_checks.values()), f"live Slurm profile drift: {slurm_checks}")

    child_env = os.environ.copy()
    child_env.update(
        {
            "H2G_CODE_ROOT": str(args.code_root.resolve()),
            "H2G_CODE_RECEIPT": str(args.code_receipt.resolve()),
            "H2G_STAGE_ROOT": str(args.stage_root.resolve()),
            "H2G_BENCHMARK_CONTRACT": str(args.qualification_contract.resolve()),
            "H2G_BENCHMARK_OUTPUT_ROOT": str(benchmark_root),
            "H2G_MEGATRON_DIR": str(args.megatron_root.resolve()),
            "H2G_MEGATRON_RECEIPT": str(args.megatron_receipt.resolve()),
            "H2G_VAL_DATA_DIR": str(args.validation_root.resolve()),
            "H2G_ONLINE_VALIDATION_RECEIPT": str(args.validation_receipt.resolve()),
            "H2G_EXTRA_VALID_SETS": args.extra_valid_sets,
            "H2G_NEW_GREEK_VALID_SETS": args.new_greek_valid_sets,
        }
    )
    subproject = (
        args.code_root.resolve() / "subprojects/08_targeted_8b_cpt_experiments"
    )
    subprocess.run(
        ["bash", str(subproject / "clariden/run_prelaunch_benchmark.sbatch")],
        check=True,
        env=child_env,
    )

    profile_receipt = qualification_root / "profile_benchmark.json"
    subprocess.run(
        [
            sys.executable,
            str(subproject / "scripts/finalize_profile_benchmark.py"),
            "--scale", "1p5b",
            "--profile-id", "1p5b_tp1_1node",
            "--nodes", "1",
            "--tensor-parallel", "1",
            "--microbatch", "8",
            "--experiment", str(subproject / "configs/hard_h_to_g_replication_v1.json"),
            "--allocation", str(subproject / "configs/hard_h_to_g_allocation_v1.json"),
            "--benchmark-contract", str(args.qualification_contract.resolve()),
            "--preflight", str(benchmark_root / "preflight.json"),
            "--throughput-log", str(benchmark_root / "throughput/driver.out"),
            "--reference-benchmark-contract", str(args.qualification_contract.resolve()),
            "--reference-throughput-log", str(benchmark_root / "throughput/driver.out"),
            "--uninterrupted-log", str(benchmark_root / "restart/uninterrupted/driver.out"),
            "--resumed-log", str(benchmark_root / "restart/resumed/driver.out"),
            "--phase2-uninterrupted-log", str(benchmark_root / "restart/phase2_uninterrupted/driver.out"),
            "--phase2-resumed-log", str(benchmark_root / "restart/phase2_resumed/driver.out"),
            "--output", str(profile_receipt),
        ],
        check=True,
        env=child_env,
    )
    profile = read_json(profile_receipt)
    require(
        profile.get("status") == "passed"
        and isinstance(profile.get("checks"), dict)
        and set(profile["checks"]) == set(PROFILE_CHECKS)
        and all(profile["checks"].values()),
        "profile qualification did not pass",
    )

    slurm_receipt = qualification_root / "slurm_profile.json"
    write_json_atomic(
        slurm_receipt,
        {
            "schema_version": "apertus_in_allocation_slurm_profile_v1",
            "status": "passed",
            "campaign_id": args.campaign_id,
            "contract_digest": args.contract_digest,
            "job_id": job_id,
            "runtime_profile_id": runtime["profile_id"],
            "checks": slurm_checks,
            "scontrol": raw_job,
        },
    )
    write_json_atomic(
        context_path,
        {
            "schema_version": "apertus_gate_context_v1",
            "status": "passed",
            "campaign_id": args.campaign_id,
            "contract_digest": args.contract_digest,
            "evidence": {
                "profile_measurement": file_binding(profile_receipt),
                "slurm_profile": file_binding(slurm_receipt),
            },
        },
    )
    print(context_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
