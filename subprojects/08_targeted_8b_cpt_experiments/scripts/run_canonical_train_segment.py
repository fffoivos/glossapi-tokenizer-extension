#!/usr/bin/env python3
"""Bridge one canonical campaign claim to the frozen H-to-G trainer.

The canonical campaign runner owns attempts, Slurm submission, retries and
handoffs.  This adapter owns only experiment-specific translation: phase
assets, the existing trainer environment, checkpoint audit/permit production,
and the canonical completion receipt.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from contract_utils import file_binding, read_json, require, write_json_atomic

LATE_BOUND_PHASE_CACHE = "@phase_cache_receipt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("8b", "1p5b"), required=True)
    parser.add_argument("--phase", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--start-update", type=int, required=True)
    parser.add_argument("--end-update", type=int, required=True)
    parser.add_argument("--load-checkpoint", type=Path, required=True)
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--completion-receipt", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--segment-id", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--contract-digest", required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--code-receipt", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--phase-data-path-spec", type=Path, required=True)
    parser.add_argument("--phase-data-path", required=True)
    parser.add_argument("--phase-cache-receipt", type=Path, required=True)
    parser.add_argument("--phase-cache-root", type=Path, required=True)
    parser.add_argument("--phase-cache-tree-sha256", required=True)
    parser.add_argument("--megatron-root", type=Path, required=True)
    parser.add_argument("--megatron-receipt", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--validation-receipt", type=Path, required=True)
    parser.add_argument("--extra-valid-sets", required=True)
    parser.add_argument("--new-greek-valid-sets", required=True)
    parser.add_argument("--training-run-permit", type=Path, required=True)
    parser.add_argument("--authorization-gate", type=Path, required=True)
    parser.add_argument("--static-preflight-receipt", type=Path, required=True)
    parser.add_argument("--qualification-contract", type=Path, required=True)
    parser.add_argument("--initialization-permit", type=Path)
    parser.add_argument("--peak-lr", required=True)
    parser.add_argument("--floor-lr", required=True)
    parser.add_argument("--microbatch", type=int, required=True)
    parser.add_argument("--tensor-parallel", type=int, required=True)
    return parser.parse_args()


def load_checkpoint_reference(path: Path, *, scale: str, update: int) -> dict[str, Any]:
    value = read_json(path)
    require(
        value.get("schema_version") == "apertus_hard_h_to_g_checkpoint_reference_v1"
        and value.get("status") == "passed",
        "canonical checkpoint reference is not passing",
    )
    require(
        value.get("scale") == scale and int(value.get("update", -1)) == update,
        "checkpoint reference scale/update drift",
    )
    load_root = Path(str(value.get("load_root", "")))
    checkpoint_root = Path(str(value.get("checkpoint_root", "")))
    require(
        load_root.is_dir()
        and checkpoint_root == load_root / f"iter_{update:07d}"
        and checkpoint_root.is_dir(),
        "checkpoint reference payload drift",
    )
    for field in ("checkpoint_permit", "source_phase_cache_receipt"):
        binding = value.get(field)
        require(isinstance(binding, dict), f"checkpoint reference lacks {field}")
        bound = Path(str(binding.get("path", "")))
        require(
            bound.is_file() and binding == file_binding(bound),
            f"checkpoint reference {field} drift",
        )
    return value


def phase_range(phase: int) -> tuple[int, int]:
    return {1: (0, 2261), 2: (2261, 3218), 3: (3218, 3694)}[phase]


def resolve_phase_cache_arguments(args: argparse.Namespace) -> tuple[str, str]:
    """Resolve Phase-3 values that cannot exist before Phase 2 completes."""

    late_data = args.phase_data_path == LATE_BOUND_PHASE_CACHE
    late_hash = args.phase_cache_tree_sha256 == LATE_BOUND_PHASE_CACHE
    require(late_data == late_hash, "partial late-bound phase-cache declaration")
    if not late_data:
        return args.phase_data_path, args.phase_cache_tree_sha256
    require(
        args.phase == 3 and args.start_update in {3218, 3456},
        "late-bound phase-cache values are allowed only at a Phase-3 segment boundary",
    )
    receipt = read_json(args.phase_cache_receipt)
    require(
        receipt.get("schema_version") == "apertus_hard_h_to_g_phase_blend_cache_v1"
        and receipt.get("status") == "frozen"
        and int(receipt.get("phase", -1)) == 3,
        "late-bound Phase-3 cache receipt drift",
    )
    require(
        Path(str(receipt.get("cache_root", ""))).resolve()
        == args.phase_cache_root.resolve(),
        "late-bound Phase-3 cache-root drift",
    )
    require(
        receipt.get("data_path_spec") == file_binding(args.phase_data_path_spec),
        "late-bound Phase-3 data-path spec drift",
    )
    tokens = receipt.get("data_path_tokens")
    require(
        isinstance(tokens, list)
        and bool(tokens)
        and all(isinstance(token, str) and token for token in tokens),
        "late-bound Phase-3 data-path tokens missing",
    )
    cache_sha256 = str(receipt.get("cache_tree_sha256", ""))
    require(
        len(cache_sha256) == 64
        and all(character in "0123456789abcdef" for character in cache_sha256),
        "late-bound Phase-3 cache SHA-256 missing",
    )
    return shlex.join(tokens), cache_sha256


def bridge_triggers(canonical: Path, payload: Path, stop: threading.Event) -> None:
    """Copy canonical save/exit requests to the experiment trainer."""

    while not stop.wait(0.25):
        for name in ("save", "exit"):
            if not (canonical / name).exists():
                continue
            target = payload / "triggers"
            target.mkdir(parents=True, exist_ok=True)
            (target / name).touch(exist_ok=True)


def run_checked(argv: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(argv, check=True, env=env)


def validate_profile_qualification_contract(
    args: argparse.Namespace,
    contract: dict[str, Any],
    *,
    live_nodes: int,
) -> Path:
    require(
        args.phase == 1 and args.start_update == 0 and args.end_update >= 256,
        "runtime qualification must start at the Phase-1 origin",
    )
    require(
        contract.get("schema_version")
        == "apertus_hard_h_to_g_prelaunch_benchmark_contract_v1"
        and contract.get("status") == "frozen"
        and contract.get("kind") == "profile",
        "runtime qualification contract drift",
    )
    require(
        contract.get("scale") == args.scale and int(contract.get("updates", -1)) == 256,
        "runtime qualification scale/horizon drift",
    )
    require(
        int(contract.get("nodes", -1)) == live_nodes,
        "runtime qualification live-node drift",
    )
    require(
        int(contract.get("tensor_parallel", -1)) == args.tensor_parallel
        and int(contract.get("microbatch", -1)) == args.microbatch
        and str(contract.get("peak_lr")) == args.peak_lr
        and str(contract.get("floor_lr")) == args.floor_lr,
        "runtime qualification training geometry drift",
    )
    require(
        Path(str(contract.get("initialization_root", ""))).resolve()
        == args.load_checkpoint.resolve(),
        "runtime qualification initialization-root drift",
    )
    require(
        contract.get("phase_cache_receipt") == file_binding(args.phase_cache_receipt),
        "runtime qualification Phase-1 cache receipt drift",
    )
    require(
        Path(str(contract.get("phase_cache_root", ""))).resolve()
        == args.phase_cache_root.resolve(),
        "runtime qualification Phase-1 cache-root drift",
    )
    require(
        contract.get("phase_cache_tree_sha256") == args.phase_cache_tree_sha256,
        "runtime qualification Phase-1 cache hash drift",
    )
    require(
        str(contract.get("phase_data_path")) == args.phase_data_path,
        "runtime qualification Phase-1 data-path drift",
    )
    return Path(str(contract.get("output_root", ""))).resolve()


def run_profile_qualification(args: argparse.Namespace) -> int:
    """Run the frozen exact-profile prologue under the candidate campaign."""

    require(
        args.initialization_permit is not None and args.initialization_permit.is_file(),
        "runtime qualification lacks initialization evidence",
    )
    require(
        args.qualification_contract.is_file(), "runtime qualification contract missing"
    )
    contract = read_json(args.qualification_contract)
    output_root = validate_profile_qualification_contract(
        args,
        contract,
        live_nodes=int(os.environ.get("SLURM_NNODES", "0")),
    )
    require(
        not output_root.exists(), "runtime qualification output root already exists"
    )
    child_env = os.environ.copy()
    child_env.update(
        {
            "H2G_CODE_ROOT": str(args.code_root.resolve()),
            "H2G_CODE_RECEIPT": str(args.code_receipt.resolve()),
            "H2G_STAGE_ROOT": str(args.stage_root.resolve()),
            "H2G_BENCHMARK_CONTRACT": str(args.qualification_contract.resolve()),
            "H2G_BENCHMARK_OUTPUT_ROOT": str(output_root),
            "H2G_MEGATRON_DIR": str(args.megatron_root.resolve()),
            "H2G_MEGATRON_RECEIPT": str(args.megatron_receipt.resolve()),
            "H2G_VAL_DATA_DIR": str(args.validation_root.resolve()),
            "H2G_ONLINE_VALIDATION_RECEIPT": str(args.validation_receipt.resolve()),
            "H2G_EXTRA_VALID_SETS": args.extra_valid_sets,
            "H2G_NEW_GREEK_VALID_SETS": args.new_greek_valid_sets,
        }
    )
    benchmark_runner = (
        args.code_root
        / "subprojects/08_targeted_8b_cpt_experiments/clariden/run_prelaunch_benchmark.sbatch"
    )
    require(benchmark_runner.is_file(), "frozen profile benchmark runner missing")
    run_checked(["bash", str(benchmark_runner)], env=child_env)

    profile_receipt = args.attempt_root / "profile_benchmark.json"
    finalizer = (
        args.code_root
        / "subprojects/08_targeted_8b_cpt_experiments/scripts/finalize_profile_benchmark.py"
    )
    profile_id = str(contract["profile_id"])
    finalizer_argv = [
        sys.executable,
        str(finalizer),
        "--scale",
        args.scale,
        "--profile-id",
        profile_id,
        "--nodes",
        str(contract["nodes"]),
        "--tensor-parallel",
        str(contract["tensor_parallel"]),
        "--microbatch",
        str(contract["microbatch"]),
        "--experiment",
        str(
            args.code_root
            / "subprojects/08_targeted_8b_cpt_experiments/configs/hard_h_to_g_replication_v1.json"
        ),
        "--allocation",
        str(
            args.code_root
            / "subprojects/08_targeted_8b_cpt_experiments/configs/hard_h_to_g_allocation_v1.json"
        ),
        "--benchmark-contract",
        str(args.qualification_contract),
        "--preflight",
        str(output_root / "preflight.json"),
        "--throughput-log",
        str(output_root / "throughput/driver.out"),
        "--reference-benchmark-contract",
        str(args.qualification_contract),
        "--reference-throughput-log",
        str(output_root / "throughput/driver.out"),
        "--uninterrupted-log",
        str(output_root / "restart/uninterrupted/driver.out"),
        "--resumed-log",
        str(output_root / "restart/resumed/driver.out"),
        "--phase2-uninterrupted-log",
        str(output_root / "restart/phase2_uninterrupted/driver.out"),
        "--phase2-resumed-log",
        str(output_root / "restart/phase2_resumed/driver.out"),
        "--output",
        str(profile_receipt),
    ]
    run_checked(finalizer_argv, env=child_env)

    tracker = output_root / "throughput/checkpoints/latest_checkpointed_iteration.txt"
    checkpoint_root = output_root / "throughput/checkpoints/iter_0000256"
    metadata = checkpoint_root / ".metadata"
    require(
        tracker.is_file() and tracker.read_text(encoding="utf-8").strip() == "256",
        "runtime qualification checkpoint tracker drift",
    )
    require(metadata.is_file(), "runtime qualification DCP metadata missing")
    checkpoint_artifact = args.attempt_root / "qualification_checkpoint.json"
    write_json_atomic(
        checkpoint_artifact,
        {
            "schema_version": "apertus_hard_h_to_g_runtime_qualification_checkpoint_v1",
            "status": "passed",
            "scale": args.scale,
            "update": 256,
            "checkpoint_root": str(checkpoint_root),
            "tracker": file_binding(tracker),
            "metadata": file_binding(metadata),
            "profile_benchmark": file_binding(profile_receipt),
            "benchmark_contract": file_binding(args.qualification_contract),
        },
    )

    # A candidate campaign intentionally stops after the 256-update minimum
    # qualification window.  Signal the canonical parent so it records a real
    # resumable early stop rather than misclassifying the bounded run as a
    # failed production segment.
    os.kill(os.getppid(), signal.SIGUSR1)
    canonical_exit = args.attempt_root / "triggers/exit"
    deadline = time.monotonic() + 5
    while not canonical_exit.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    require(
        canonical_exit.is_file(), "canonical qualification stop signal was not observed"
    )
    run_checked(
        [
            sys.executable,
            "-B",
            "-m",
            "apertus_cscs_campaign.completion_adapter",
            "--manifest",
            str(Path(os.environ["APERTUS_CAMPAIGN_MANIFEST"])),
            "--run-root",
            str(args.run_root),
            "--segment",
            args.segment_id,
            "--attempt",
            str(args.attempt),
            "--observed-iteration",
            "256",
            "--checkpoint-artifact",
            str(checkpoint_artifact),
            "--metrics",
            str(profile_receipt),
            "--output",
            str(args.completion_receipt),
        ]
    )
    print(
        json.dumps(
            {"status": "qualification_checkpointed", "observed_iteration": 256},
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    args = parse_args()
    low, high = phase_range(args.phase)
    require(
        low <= args.start_update < args.end_update <= high,
        "canonical segment crosses a phase boundary",
    )
    require(
        os.environ.get("SLURM_JOB_PARTITION") == "normal",
        "canonical training requires normal",
    )
    require(args.attempt_root.is_dir(), "canonical attempt root missing")
    require(
        args.completion_receipt == args.attempt_root / "completion.json",
        "canonical completion path drift",
    )
    require(
        not args.completion_receipt.exists(),
        "canonical completion receipt already exists",
    )
    manifest = Path(os.environ.get("APERTUS_CAMPAIGN_MANIFEST", ""))
    canonical_root = Path(os.environ.get("APERTUS_CAMPAIGN_CODE_ROOT", ""))
    require(
        manifest.is_file() and canonical_root.is_dir(),
        "canonical runner environment missing",
    )
    if os.environ.get("APERTUS_CAMPAIGN_QUALIFICATION_ONLY") == "1":
        return run_profile_qualification(args)

    load_root: Path
    checkpoint_permit: Path | None = None
    source_phase_cache: Path | None = None
    canonical_resume = False
    if args.start_update == 0:
        require(args.load_checkpoint.is_dir(), "initialization load root missing")
        require(
            args.initialization_permit is not None
            and args.initialization_permit.is_file(),
            "initialization permit missing",
        )
        load_root = args.load_checkpoint.resolve()
    else:
        require(
            args.load_checkpoint.is_file(),
            "resumed canonical segment requires a checkpoint reference file",
        )
        reference = load_checkpoint_reference(
            args.load_checkpoint, scale=args.scale, update=args.start_update
        )
        load_root = Path(str(reference["load_root"])).resolve()
        checkpoint_permit = Path(str(reference["checkpoint_permit"]["path"])).resolve()
        source_phase_cache = Path(
            str(reference["source_phase_cache_receipt"]["path"])
        ).resolve()
        canonical_resume = bool(reference.get("gracefully_stopped", False))

    phase_data_path, phase_cache_tree_sha256 = resolve_phase_cache_arguments(args)

    payload_root = args.attempt_root / "payload"
    require(not payload_root.exists(), "immutable canonical payload root exists")
    canonical_triggers = args.attempt_root / "triggers"
    stop_bridge = threading.Event()
    bridge = threading.Thread(
        target=bridge_triggers,
        args=(canonical_triggers, payload_root, stop_bridge),
        daemon=True,
    )

    child_env = os.environ.copy()
    child_env.update(
        {
            "H2G_CODE_ROOT": str(args.code_root.resolve()),
            "H2G_CODE_RECEIPT": str(args.code_receipt.resolve()),
            "H2G_STAGE_ROOT": str(args.stage_root.resolve()),
            "H2G_MODEL_SCALE": args.scale,
            "H2G_PHASE": str(args.phase),
            "H2G_START_UPDATE": str(args.start_update),
            "H2G_EXIT_UPDATE": str(args.end_update),
            "H2G_LOAD_CHECKPOINT": str(load_root),
            "H2G_PHASE_DATA_PATH_SPEC": str(args.phase_data_path_spec.resolve()),
            "H2G_PHASE_DATA_PATH": phase_data_path,
            "H2G_PHASE_CACHE_RECEIPT": str(args.phase_cache_receipt.resolve()),
            "H2G_PHASE_CACHE_ROOT": str(args.phase_cache_root.resolve()),
            "H2G_PHASE_CACHE_TREE_SHA256": phase_cache_tree_sha256,
            "H2G_SEGMENT_OUTPUT_ROOT": str(payload_root),
            "H2G_PEAK_LR": args.peak_lr,
            "H2G_FLOOR_LR": args.floor_lr,
            "H2G_MICROBATCH_SIZE": str(args.microbatch),
            "H2G_TENSOR_PARALLEL_SIZE": str(args.tensor_parallel),
            "H2G_VAL_DATA_DIR": str(args.validation_root.resolve()),
            "H2G_ONLINE_VALIDATION_RECEIPT": str(args.validation_receipt.resolve()),
            "H2G_EXTRA_VALID_SETS": args.extra_valid_sets,
            "H2G_NEW_GREEK_VALID_SETS": args.new_greek_valid_sets,
            "H2G_MEGATRON_DIR": str(args.megatron_root.resolve()),
            "H2G_MEGATRON_RECEIPT": str(args.megatron_receipt.resolve()),
            "H2G_TRAINING_RUN_PERMIT": str(args.training_run_permit.resolve()),
            "H2G_AUTHORIZATION_GATE": str(args.authorization_gate.resolve()),
            "H2G_STATIC_PREFLIGHT": str(args.static_preflight_receipt.resolve()),
            "H2G_CANONICAL_RESUME": "1" if canonical_resume else "0",
        }
    )
    if args.start_update == 0:
        child_env["H2G_INITIALIZATION_PERMIT"] = str(
            args.initialization_permit.resolve()
        )
    else:
        child_env["H2G_CHECKPOINT_PERMIT"] = str(checkpoint_permit)
        child_env["H2G_SOURCE_PHASE_CACHE_RECEIPT"] = str(source_phase_cache)

    trainer = (
        args.code_root
        / "subprojects/08_targeted_8b_cpt_experiments/clariden/train_hard_h_to_g_segment.sbatch"
    )
    require(trainer.is_file(), "frozen H-to-G trainer missing")
    bridge.start()
    try:
        completed = subprocess.run(["bash", str(trainer)], env=child_env, check=False)
    finally:
        stop_bridge.set()
        bridge.join(timeout=2)
    require(
        completed.returncode == 0,
        f"H-to-G trainer failed with return code {completed.returncode}",
    )

    tracker = payload_root / "checkpoints/latest_checkpointed_iteration.txt"
    require(tracker.is_file(), "trainer exited without a checkpoint tracker")
    observed = int(tracker.read_text(encoding="utf-8").strip())
    require(
        args.start_update < observed <= args.end_update,
        "checkpoint cursor outside canonical claim",
    )
    graceful = observed < args.end_update
    require(
        not graceful or (canonical_triggers / "exit").exists(),
        "partial checkpoint lacks a canonical stop request",
    )

    checkpoint_root = payload_root / "checkpoints" / f"iter_{observed:07d}"
    audit_path = args.attempt_root / "checkpoint_audit.json"
    permit_path = args.attempt_root / "checkpoint_permit.json"
    train_log = args.attempt_root / "train.log"
    audit_argv = [
        sys.executable,
        str(
            args.code_root
            / "subprojects/08_targeted_8b_cpt_experiments/scripts/audit_training_checkpoint.py"
        ),
        "--scale",
        args.scale,
        "--source-phase",
        str(args.phase),
        "--update",
        str(observed),
        "--checkpoint-root",
        str(checkpoint_root),
        "--source-phase-cache-receipt",
        str(args.phase_cache_receipt),
        "--segment-preflight",
        str(
            args.stage_root
            / "receipts/train_preflight"
            / f"{args.scale}_p{args.phase}_{args.start_update}_{args.end_update}_{os.environ['SLURM_JOB_ID']}.json"
        ),
        "--training-log",
        str(train_log),
        "--output",
        str(audit_path),
    ]
    if graceful:
        audit_argv.append("--allow-graceful-stop")
    run_checked(audit_argv, env=child_env)
    run_checked(
        [
            sys.executable,
            str(
                args.code_root
                / "subprojects/08_targeted_8b_cpt_experiments/scripts/build_checkpoint_permit.py"
            ),
            "--scale",
            args.scale,
            "--source-phase",
            str(args.phase),
            "--update",
            str(observed),
            "--checkpoint-root",
            str(checkpoint_root),
            "--checkpoint-audit",
            str(audit_path),
            "--source-phase-cache-receipt",
            str(args.phase_cache_receipt),
            "--output",
            str(permit_path),
        ],
        env=child_env,
    )

    reference_path = args.attempt_root / "checkpoint_reference.json"
    write_json_atomic(
        reference_path,
        {
            "schema_version": "apertus_hard_h_to_g_checkpoint_reference_v1",
            "status": "passed",
            "scale": args.scale,
            "phase": args.phase,
            "update": observed,
            "claimed_end_update": args.end_update,
            "gracefully_stopped": graceful,
            "load_root": str((payload_root / "checkpoints").resolve()),
            "checkpoint_root": str(checkpoint_root.resolve()),
            "checkpoint_permit": file_binding(permit_path),
            "source_phase_cache_receipt": file_binding(args.phase_cache_receipt),
            "checkpoint_audit": file_binding(audit_path),
        },
    )

    run_checked(
        [
            sys.executable,
            "-B",
            "-m",
            "apertus_cscs_campaign.completion_adapter",
            "--manifest",
            str(manifest),
            "--run-root",
            str(args.run_root),
            "--segment",
            args.segment_id,
            "--attempt",
            str(args.attempt),
            "--observed-iteration",
            str(observed),
            "--checkpoint-artifact",
            str(reference_path),
            "--output",
            str(args.completion_receipt),
        ]
    )
    print(
        json.dumps(
            {
                "status": "completed" if not graceful else "checkpointed",
                "observed_iteration": observed,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
