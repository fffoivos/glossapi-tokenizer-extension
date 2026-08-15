#!/usr/bin/env python3
"""Adapt the accepted historical 8B profile measurement to canonical gates."""

from __future__ import annotations

import argparse
import subprocess
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
from producer_bundle_compatibility import load_authority, require_accepted_producer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--canonical-runner-root", type=Path, required=True)
    parser.add_argument("--profile-benchmark", type=Path, required=True)
    parser.add_argument("--profile-promotion", type=Path, required=True)
    parser.add_argument("--producer-compatibility", type=Path, required=True)
    parser.add_argument("--slurm-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def matching_input(manifest: dict[str, Any], item_id: str, path: Path) -> None:
    rows = manifest["campaign"]["science"]["immutable_inputs"]
    matches = [row for row in rows if row.get("id") == item_id]
    require(len(matches) == 1, f"immutable input is not unique: {item_id}")
    require_file_binding(matches[0], expected_path=path)


def main() -> int:
    args = parse_args()
    runner_src = args.canonical_runner_root.resolve() / "src"
    require(runner_src.is_dir(), "canonical runner source root missing")
    sys.path.insert(0, str(runner_src))
    from apertus_cscs_campaign.engine import verify_compiled

    manifest = verify_compiled(args.manifest.resolve())
    require(
        manifest["runtime"]["status"] == "candidate"
        and manifest["campaign_id"] == "hard-h2g-8b-matched-r2",
        "8B candidate manifest drift",
    )
    runtime = manifest["runtime"]
    require(
        runtime["slurm"]["nodes"] == 16
        and runtime["parallelism"]
        == {"tensor": 2, "pipeline": 1, "context": 1, "data": 32},
        "8B candidate runtime geometry drift",
    )
    matching_input(manifest, "profile_promotion", args.profile_promotion.resolve())
    matching_input(
        manifest, "producer_compatibility", args.producer_compatibility.resolve()
    )

    compatibility = read_json(args.producer_compatibility.resolve())
    accepted = load_authority(
        args.producer_compatibility.resolve(),
        compatibility["executing_code_bundle"],
    )[1]
    promotion = read_json(args.profile_promotion.resolve())
    require(
        promotion.get("schema_version") == "apertus_hard_h_to_g_profile_promotion_v1"
        and promotion.get("status") == "promoted"
        and promotion.get("scale") == "8b"
        and promotion.get("selection", {}).get("profile_id") == "dp32_16node",
        "accepted 8B profile promotion drift",
    )
    require_accepted_producer(promotion, accepted, "8B profile promotion")
    benchmark_path = args.profile_benchmark.resolve()
    require(
        any(
            isinstance(binding, dict)
            and Path(str(binding.get("path", ""))).resolve() == benchmark_path
            and binding == file_binding(benchmark_path)
            for binding in promotion.get("candidate_receipts", [])
        ),
        "8B benchmark is not the promoted candidate",
    )
    benchmark = read_json(benchmark_path)
    require_accepted_producer(benchmark, accepted, "8B profile benchmark")
    require(
        benchmark.get("schema_version") == "apertus_hard_h_to_g_profile_benchmark_v1"
        and benchmark.get("status") == "passed"
        and benchmark.get("scale") == "8b"
        and benchmark.get("profile", {}).get("profile_id") == "dp32_16node",
        "8B profile benchmark identity drift",
    )
    require(
        benchmark.get("profile", {}).get("nodes") == 16
        and benchmark.get("profile", {}).get("tensor_parallel") == 2
        and benchmark.get("profile", {}).get("data_parallel") == 32,
        "8B profile benchmark geometry drift",
    )
    checks = benchmark.get("checks")
    require(
        isinstance(checks, dict)
        and set(checks) == set(PROFILE_CHECKS)
        and all(checks.values()),
        "8B profile benchmark checks are incomplete",
    )
    for binding in benchmark.get("evidence", []):
        require_file_binding(binding)

    contract_binding = benchmark.get("benchmark_contract")
    contract_path = require_file_binding(contract_binding)
    contract = read_json(contract_path)
    output_root = Path(str(contract.get("output_root", ""))).resolve()
    job_id_path = output_root / "slurm_job_id.txt"
    require(job_id_path.is_file(), "8B benchmark Slurm job-id evidence missing")
    job_id = job_id_path.read_text(encoding="utf-8").strip()
    accounting = subprocess.run(
        [
            "sacct", "-X", "-j", job_id, "--noheader", "--parsable2",
            "--format=JobIDRaw,State,Partition,AllocNodes,ElapsedRaw",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    rows = [row.split("|")[:5] for row in accounting if row.strip()]
    matches = [row for row in rows if row[0] == job_id]
    require(len(matches) == 1, f"8B benchmark accounting row drift: {rows}")
    row = matches[0]
    slurm_checks = {
        "job_id": row[0] == job_id,
        "completed": row[1].split()[0] == "COMPLETED",
        "normal_partition": row[2] == "normal",
        "exact_nodes": row[3] == "16",
        "positive_elapsed": int(row[4]) > 0,
        "contract_nodes": int(contract.get("nodes", -1)) == 16,
        "contract_profile": contract.get("profile_id") == "dp32_16node",
    }
    require(all(slurm_checks.values()), f"8B historical Slurm evidence drift: {slurm_checks}")
    write_json_atomic(
        args.slurm_output.resolve(),
        {
            "schema_version": "apertus_historical_slurm_profile_adapter_v1",
            "status": "passed",
            "campaign_id": manifest["campaign_id"],
            "contract_digest": manifest["contract_digest"],
            "job_id": job_id,
            "checks": slurm_checks,
            "accounting_row": row,
            "job_id_file": file_binding(job_id_path),
            "benchmark_contract": file_binding(contract_path),
        },
    )
    write_json_atomic(
        args.output.resolve(),
        {
            "schema_version": "apertus_gate_context_v1",
            "status": "passed",
            "campaign_id": manifest["campaign_id"],
            "contract_digest": manifest["contract_digest"],
            "evidence": {
                "profile_measurement": file_binding(benchmark_path),
                "slurm_profile": file_binding(args.slurm_output.resolve()),
            },
        },
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
