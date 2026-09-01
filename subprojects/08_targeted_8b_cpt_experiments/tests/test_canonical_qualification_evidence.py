from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.dont_write_bytecode = True
sys.path.insert(0, str(SCRIPTS))

from build_canonical_campaign_contracts import PROFILE_CHECKS
from build_canonical_qualification_context import (
    build_context,
    validate_profile_receipt,
)
from contract_utils import file_binding

EFFICIENCY_ROOT = Path(
    "/Users/foivoskarounos-zamparloukos/Projects/apertus-cscs-efficiency"
)


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def identity(manifest: dict[str, object], attempt: int) -> dict[str, object]:
    return {
        "campaign_id": manifest["campaign_id"],
        "target_id": "s0",
        "attempt": attempt,
        "scientific_digest": manifest["scientific_digest"],
        "operational_digest": manifest["operational_digest"],
        "contract_digest": manifest["contract_digest"],
        "qualification_only": True,
    }


def qualification_fixture(tmp_path: Path) -> tuple[dict[str, object], Path]:
    science = tmp_path / "science"
    science.mkdir()
    science_receipt = tmp_path / "science.json"
    write_json(
        science_receipt,
        {
            "schema_version": "apertus_mini_immutable_code_bundle_v1",
            "status": "frozen",
            "kind": "scientific",
            "root": str(science),
            "tree_sha256": "a" * 64,
        },
    )
    manifest: dict[str, object] = {
        "campaign_id": "hard-h2g-8b-matched-r2",
        "scientific_digest": "b" * 64,
        "operational_digest": "c" * 64,
        "contract_digest": "d" * 64,
        "runtime": {
            "status": "candidate",
            "slurm": {"nodes": 16, "gpus_per_node": 4},
            "parallelism": {"tensor": 2, "pipeline": 1, "context": 1, "data": 32},
        },
        "campaign": {
            "science": {
                "immutable_inputs": [
                    {"id": "scientific_code_receipt", **file_binding(science_receipt)}
                ]
            }
        },
    }
    run_root = tmp_path / "run"
    root = run_root / "segments/s0/attempts/attempt_000001"
    root.mkdir(parents=True)
    benchmark_contract = tmp_path / "benchmark.json"
    tracker = tmp_path / "checkpoint/latest_checkpointed_iteration.txt"
    metadata = tmp_path / "checkpoint/iter_0000256/.metadata"
    evidence = tmp_path / "driver.out"
    write_json(benchmark_contract, {"status": "frozen"})
    tracker.parent.mkdir(parents=True)
    tracker.write_text("256\n", encoding="utf-8")
    metadata.parent.mkdir(parents=True)
    metadata.write_bytes(b"dcp")
    evidence.write_text("machine evidence\n", encoding="utf-8")
    profile_path = root / "profile_benchmark.json"
    write_json(
        profile_path,
        {
            "schema_version": "apertus_hard_h_to_g_profile_benchmark_v1",
            "status": "passed",
            "scale": "8b",
            "executing_code_bundle": {"root": str(science), "tree_sha256": "a" * 64},
            "profile": {
                "profile_id": "dp32_16node",
                "nodes": 16,
                "gpus_per_node": 4,
                "tensor_parallel": 2,
                "pipeline_parallel": 1,
                "context_parallel": 1,
                "data_parallel": 32,
                "microbatch": 2,
            },
            "measurement": {
                "updates": 256,
                "discarded_warmup_updates": 32,
                "median_step_seconds": 8.5,
                "p90_step_seconds": 9.0,
                "tokens_per_gpu_hour": 27_000,
            },
            "checks": {name: True for name in PROFILE_CHECKS},
            "evidence": [file_binding(evidence)],
        },
    )
    checkpoint_path = root / "qualification_checkpoint.json"
    write_json(
        checkpoint_path,
        {
            "schema_version": "apertus_hard_h_to_g_runtime_qualification_checkpoint_v1",
            "status": "passed",
            "scale": "8b",
            "update": 256,
            "checkpoint_root": str(metadata.parent),
            "tracker": file_binding(tracker),
            "metadata": file_binding(metadata),
            "profile_benchmark": file_binding(profile_path),
            "benchmark_contract": file_binding(benchmark_contract),
        },
    )
    write_json(
        root / "claim.json",
        {
            "schema_version": "apertus_campaign_claim_v2",
            "status": "claimed",
            **identity(manifest, 1),
            "role": "train",
            "action": "submit_segment",
            "start_iteration": 0,
            "end_iteration": 952,
        },
    )
    write_json(
        root / "submission.json",
        {
            "schema_version": "apertus_campaign_submission_v2",
            "status": "submitted",
            **identity(manifest, 1),
            "role": "train",
            "job": {"job_id": "123", "checks": {"nodes": True, "account": True}},
        },
    )
    write_json(
        root / "completion.json",
        {
            "schema_version": "apertus_hard_h_to_g_training_completion_v1",
            "status": "checkpointed",
            "campaign_id": manifest["campaign_id"],
            "segment_id": "s0",
            "attempt": 1,
            "contract_digest": manifest["contract_digest"],
            "observed_iteration": 256,
            "checkpoint": file_binding(checkpoint_path),
            "metrics": file_binding(profile_path),
        },
    )
    write_json(
        root / "execution.json",
        {
            "schema_version": "apertus_campaign_training_attempt_v2",
            "status": "gracefully_stopped",
            **identity(manifest, 1),
            "segment_id": "s0",
            "resume_iteration": 256,
            "returncode": 0,
            "minimum_runtime_satisfied": True,
            "signal": "SIGUSR1",
            "checkpoint": file_binding(checkpoint_path),
            "completion_receipt": file_binding(root / "completion.json"),
        },
    )
    return manifest, run_root


def attempt_numbers(run_root: Path, segment: str) -> list[int]:
    roots = (run_root / "segments" / segment / "attempts").glob("attempt_*")
    return sorted(int(path.name.removeprefix("attempt_")) for path in roots)


def attempt_root(run_root: Path, segment: str, attempt: int) -> Path:
    return run_root / "segments" / segment / "attempts" / f"attempt_{attempt:06d}"


def test_context_binds_only_the_complete_qualification_attempt(tmp_path: Path) -> None:
    manifest, run_root = qualification_fixture(tmp_path)
    context = build_context(
        manifest=manifest,
        run_root=run_root,
        scale="8b",
        segment_attempt_root=attempt_root,
        segment_attempt_numbers=attempt_numbers,
    )
    assert context["status"] == "passed"
    assert context["qualification_attempt"] == 1
    assert set(context["evidence"]) == {"profile_measurement", "slurm_profile"}


def test_context_rejects_failed_profile_check(tmp_path: Path) -> None:
    manifest, run_root = qualification_fixture(tmp_path)
    profile = attempt_root(run_root, "s0", 1) / "profile_benchmark.json"
    value = json.loads(profile.read_text(encoding="utf-8"))
    value["checks"]["restart_next_step_parity"] = False
    write_json(profile, value)
    with pytest.raises(ValueError, match="checks did not all pass"):
        validate_profile_receipt(profile, manifest=manifest, scale="8b")


def test_context_rejects_ambiguous_qualification_attempt(tmp_path: Path) -> None:
    manifest, run_root = qualification_fixture(tmp_path)
    second = attempt_root(run_root, "s0", 2)
    second.mkdir(parents=True)
    (second / "qualification_checkpoint.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not unique"):
        build_context(
            manifest=manifest,
            run_root=run_root,
            scale="8b",
            segment_attempt_root=attempt_root,
            segment_attempt_numbers=attempt_numbers,
        )


def test_promoter_binds_exact_candidate_and_gate_set(tmp_path: Path) -> None:
    sys.path.insert(0, str(EFFICIENCY_ROOT / "src"))
    from apertus_cscs_campaign.contracts import compile_contracts
    from apertus_cscs_campaign.gates import produce_runtime_qualification
    from apertus_cscs_campaign.receipts import (
        atomic_json,
    )
    from apertus_cscs_campaign.receipts import (
        file_binding as canonical_binding,
    )

    example = EFFICIENCY_ROOT / "examples/debug_smoke"
    compiled_path = tmp_path / "compiled.json"
    compiled = compile_contracts(
        example / "campaign.json", example / "runtime.json", example / "evaluation.json"
    )
    atomic_json(compiled_path, compiled)
    gate_path = tmp_path / "gates.json"
    atomic_json(
        gate_path,
        {
            "schema_version": "apertus_campaign_gate_set_v2",
            "status": "passed",
            "campaign_id": compiled["campaign_id"],
            "scientific_digest": compiled["scientific_digest"],
            "operational_digest": compiled["operational_digest"],
            "contract_digest": compiled["contract_digest"],
            "code_tree_sha256": compiled["code_bundle"]["tree_sha256"],
            "context": canonical_binding(example / "runtime.json"),
            "gates": [{"id": "fixture", "type": "file_binding", "passed": True}],
        },
    )
    qualification = tmp_path / "qualification.json"
    produce_runtime_qualification(compiled_path, gate_path, qualification)
    output = tmp_path / "runtime-proven.json"
    env = os.environ.copy()
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "promote_canonical_runtime.py"),
            "--compiled-candidate",
            str(compiled_path),
            "--qualification-receipt",
            str(qualification),
            "--canonical-runner-root",
            str(EFFICIENCY_ROOT),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    proven = json.loads(output.read_text(encoding="utf-8"))
    assert proven["status"] == "proven"
    assert proven["qualification_receipt"] == file_binding(qualification)


def test_qualification_finalizer_is_debug_only_and_verifies_both_bundles() -> None:
    wrapper = (
        ROOT / "clariden/finalize_canonical_runtime_qualification_debug.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --partition=debug" in wrapper
    assert '"${SLURM_NNODES:-0}" == 1' in wrapper
    assert '--root "$H2G_CODE_ROOT"' in wrapper
    assert '--root "$APERTUS_CANONICAL_ROOT"' in wrapper
    assert "build_canonical_qualification_context.py" in wrapper
    assert "verify --manifest" in wrapper
    assert "qualify-runtime --manifest" in wrapper
    assert "promote_canonical_runtime.py" in wrapper
