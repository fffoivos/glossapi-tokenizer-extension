#!/usr/bin/env python3
"""Fail closed before a matched hard-H-to-G training segment starts."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shlex
from pathlib import Path

from build_checkpoint_permit import validate_permit as validate_checkpoint_permit
from build_training_run_permit import validate_permit as validate_training_run_permit
from contract_utils import (
    executing_code_bundle,
    file_binding,
    read_json,
    require,
    write_json_atomic,
)
from finalize_training_megatron import validate_runtime as validate_megatron_runtime
from freeze_hard_h_to_g_contract import artifacts_for_stage, validate_artifact_manifest
from freeze_online_validation_binaries import NAMES as ONLINE_PANEL_NAMES
from freeze_online_validation_binaries import NEW_GREEK as ONLINE_NEW_GREEK
from freeze_online_validation_binaries import (
    validate_receipt as validate_online_validation,
)
from freeze_phase_blend_cache import validate_receipt as validate_phase_cache
from producer_bundle_compatibility import load_authority, require_accepted_producer

CHECKPOINTS = {0, 238, 476, 714, 952, 1190, 1428, 1666, 1904, 2142, 2261, 2380, 2618, 2856, 3094, 3218, 3456, 3694}


def validate_segment_boundaries(
    *,
    phase: int,
    start_update: int,
    exit_update: int,
    one_update_resume_smoke: bool,
    canonical_resume: bool,
) -> None:
    if one_update_resume_smoke:
        require(
            phase == 3 and (start_update, exit_update) in {(3218, 3219), (3456, 3457)},
            "one-update resume-smoke boundary drift",
        )
    elif canonical_resume:
        require(
            start_update != 0
            and exit_update in CHECKPOINTS
            and start_update < exit_update,
            "canonical resume boundary drift",
        )
    else:
        require(
            start_update in CHECKPOINTS
            and exit_update in CHECKPOINTS
            and start_update < exit_update,
            "segment boundary drift",
        )
    low, high = {1: (0, 2261), 2: (2261, 3218), 3: (3218, 3694)}[phase]
    require(low <= start_update < exit_update <= high, "segment crosses a phase boundary")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--scale", choices=("8b", "1p5b"), required=True)
    parser.add_argument("--phase", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--start-update", type=int, required=True)
    parser.add_argument("--exit-update", type=int, required=True)
    parser.add_argument("--load-checkpoint", type=Path, required=True)
    parser.add_argument("--phase-data-path-spec", type=Path, required=True)
    parser.add_argument("--phase-data-path", required=True)
    parser.add_argument("--phase-cache-receipt", type=Path, required=True)
    parser.add_argument("--phase-cache-root", type=Path, required=True)
    parser.add_argument("--phase-cache-tree-sha256", required=True)
    parser.add_argument("--megatron-root", type=Path, required=True)
    parser.add_argument("--megatron-receipt", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--validation-receipt", type=Path, required=True)
    parser.add_argument("--runtime-compat-dir", type=Path, required=True)
    parser.add_argument("--runtime-compat-receipt", type=Path, required=True)
    parser.add_argument("--extra-valid-sets", required=True)
    parser.add_argument("--new-greek-valid-sets", required=True)
    parser.add_argument("--checkpoint-permit", type=Path)
    parser.add_argument("--source-phase-cache-receipt", type=Path)
    parser.add_argument("--initialization-permit", type=Path)
    parser.add_argument("--training-run-permit", type=Path, required=True)
    parser.add_argument("--authorization-gate", type=Path)
    parser.add_argument("--preauthorization-manifest", type=Path)
    parser.add_argument("--peak-lr", required=True)
    parser.add_argument("--floor-lr", required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--tensor-parallel", type=int, required=True)
    parser.add_argument("--microbatch", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--one-update-resume-smoke", action="store_true")
    parser.add_argument("--canonical-resume", action="store_true")
    parser.add_argument(
        "--preallocation-static",
        action="store_true",
        help="freeze the exact segment contract before sbatch; consumes no allocation",
    )
    parser.add_argument(
        "--static-preflight-receipt",
        type=Path,
        help="inside an allocation, require the exact preallocation contract to match",
    )
    return parser.parse_args()


def expected_authorization_stage(*, phase: int, start_update: int, resume_smoke: bool) -> str:
    if resume_smoke:
        require(phase == 3 and start_update in {3218, 3456}, "invalid resume-smoke authorization boundary")
        return "pre_main" if start_update == 3218 else "pre_extension"
    if phase in {1, 2}:
        return "pre_main"
    return "pre_extension" if start_update < 3456 else "pre_second_extension"


def validate_authorization_gate(
    path: Path,
    *,
    experiment_path: Path,
    expected_stage: str,
    scale: str,
) -> tuple[dict[str, object], set[tuple[str, str, str, int, str]]]:
    gate = read_json(path)
    current = executing_code_bundle()
    require(
        gate.get("schema_version") == "apertus_hard_h_to_g_frozen_contract_v2"
        and gate.get("status") == "launch_ready"
        and gate.get("mode") == "launch"
        and gate.get("gate_stage") == expected_stage
        and gate.get("scale") == (scale if expected_stage == "pre_main" else None)
        and gate.get("blockers") == []
        and gate.get("all_gates_receipt_backed") is True,
        "training authorization gate is not launch-ready for this segment",
    )
    require(gate.get("executing_code_bundle") == current, "training authorization-gate code-bundle drift")
    require(gate.get("experiment") == file_binding(experiment_path), "training authorization-gate experiment drift")
    require(
        int(gate.get("gate_count", -1))
        == len(artifacts_for_stage(expected_stage, scale if expected_stage == "pre_main" else None)),
        "training authorization-gate count drift",
    )
    manifest_binding = gate.get("artifact_manifest")
    compatibility_binding = gate.get("producer_bundle_compatibility")
    require(isinstance(manifest_binding, dict), "training authorization gate lacks artifact manifest")
    require(isinstance(compatibility_binding, dict), "training authorization gate lacks producer compatibility")
    manifest_path = Path(str(manifest_binding.get("path", "")))
    compatibility_path = Path(str(compatibility_binding.get("path", "")))
    require(manifest_path.is_file() and manifest_binding == file_binding(manifest_path), "authorization artifact-manifest binding drift")
    require(compatibility_path.is_file() and compatibility_binding == file_binding(compatibility_path), "authorization producer-compatibility binding drift")
    _, accepted = load_authority(compatibility_path, current)
    _, blockers = validate_artifact_manifest(
        manifest_path,
        artifacts_for_stage(expected_stage, scale if expected_stage == "pre_main" else None),
        accepted_producers=accepted,
        expected_scale=scale if expected_stage == "pre_main" else None,
    )
    require(not blockers, f"authorization artifact manifest no longer passes: {blockers}")
    return gate, accepted


def validate_preauthorization_manifest(
    path: Path,
    *,
    expected_stage: str,
    scale: str,
) -> set[tuple[str, str, str, int, str]]:
    manifest = read_json(path)
    current = executing_code_bundle()
    require(
        manifest.get("schema_version") == "apertus_hard_h_to_g_artifact_manifest_v1"
        and manifest.get("status") == "frozen"
        and manifest.get("gate_stage") == expected_stage
        and manifest.get("scale") == (scale if expected_stage == "pre_main" else None)
        and manifest.get("partial") is True,
        "technical preauthorization manifest drift",
    )
    compatibility_binding = manifest.get("producer_bundle_compatibility")
    require(isinstance(compatibility_binding, dict), "preauthorization manifest lacks producer compatibility")
    compatibility_path = Path(str(compatibility_binding.get("path", "")))
    require(
        compatibility_path.is_file() and compatibility_binding == file_binding(compatibility_path),
        "preauthorization producer-compatibility binding drift",
    )
    _, accepted = load_authority(compatibility_path, current)
    owner_roles = {"owner_production_authorization", "owner_extension_authorization"}
    required = [
        role
        for role in artifacts_for_stage(expected_stage, scale if expected_stage == "pre_main" else None)
        if role not in owner_roles
    ]
    _, blockers = validate_artifact_manifest(
        path,
        required,
        accepted_producers=accepted,
        expected_scale=scale if expected_stage == "pre_main" else None,
    )
    require(not blockers, f"technical preauthorization manifest failed: {blockers}")
    return accepted


def code_bundle_identities(
    accepted_producers: set[tuple[str, str, str, int, str]],
) -> set[tuple[str, str]]:
    """Project producer-authority keys into validators that bind root + tree."""

    return {(root, tree) for root, tree, _receipt, _bytes, _sha256 in accepted_producers}


def validate_initialization_load_contract(
    initialization: dict[str, object],
    *,
    experiment: dict[str, object],
    scale: str,
    tensor_parallel: int,
    load_root: Path,
) -> None:
    """Catch model/config/checkpoint geometry drift without loading weight bytes."""

    require(
        initialization.get("tensor_and_logit_roundtrip_exact") is True,
        "initialization tensor/logit roundtrip is not exact",
    )
    require(
        initialization.get("target_tensor_parallel") == tensor_parallel
        and initialization.get("target_pipeline_parallel") == 1,
        "initialization TP/PP geometry drift",
    )
    training = experiment.get("training")
    models = experiment.get("models")
    tokenizer = experiment.get("tokenizer")
    require(
        isinstance(training, dict) and isinstance(models, dict) and isinstance(tokenizer, dict),
        "experiment model/training/tokenizer contract missing",
    )
    model = models.get(scale)
    require(isinstance(model, dict), "experiment scale model contract missing")
    expected_training_geometry = {
        "max_position_embeddings": int(training["max_position_embeddings"]),
        "rope_theta": float(training["rope_theta"]),
    }
    require(
        initialization.get("training_geometry") == expected_training_geometry,
        "initialization training geometry drift",
    )
    require(
        initialization.get("megatron_commit") == training["runtime"]["megatron_upstream_commit"],
        "initialization Megatron revision drift",
    )
    config_binding = initialization.get("reference_config")
    require(isinstance(config_binding, dict), "initialization reference config binding missing")
    config_path = Path(str(config_binding.get("path", "")))
    require(
        config_path.is_file() and config_binding == file_binding(config_path),
        "initialization reference config binding drift",
    )
    config = read_json(config_path)
    expected_config = {
        "model_type": "apertus",
        "hidden_size": int(model["hidden_size"]),
        "num_hidden_layers": int(model["num_hidden_layers"]),
        "num_attention_heads": int(model["num_attention_heads"]),
        "num_key_value_heads": int(model["num_key_value_heads"]),
        "tie_word_embeddings": bool(model["tie_word_embeddings"]),
        "vocab_size": int(tokenizer["vocab_size"]),
        "max_position_embeddings": int(training["max_position_embeddings"]),
        "rope_theta": float(training["rope_theta"]),
    }
    require(
        all(config.get(name) == expected for name, expected in expected_config.items()),
        "initialization reference model/config geometry drift",
    )
    rope_scaling = config.get("rope_scaling")
    require(
        isinstance(rope_scaling, dict)
        and bool(training["rope_use_scaling"])
        and float(rope_scaling.get("factor", -1)) == float(training["rope_scaling_factor"]),
        "initialization RoPE scaling drift",
    )
    rows = initialization.get("megatron_files")
    require(isinstance(rows, list) and rows, "initialization Megatron file inventory missing")
    for row in rows:
        require(isinstance(row, dict), "initialization Megatron file row malformed")
        relative = Path(str(row.get("path", "")))
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe initialization file path")
        path = (load_root / relative).resolve()
        require(
            load_root.resolve() in path.parents
            and path.is_file()
            and path.stat().st_size == int(row.get("bytes", -1)),
            f"initialization checkpoint file size/path drift: {path}",
        )


def main() -> int:
    args = parse_args()
    require(not args.output.exists(), f"immutable preflight receipt exists: {args.output}")
    require(
        not (args.preallocation_static and args.static_preflight_receipt),
        "static creation cannot consume another static-preflight receipt",
    )
    if args.preallocation_static:
        require(args.authorization_gate is None, "static technical preflight must not self-authorize launch")
        require(args.preauthorization_manifest is not None, "static technical preflight requires a preauthorization manifest")
    else:
        require(args.authorization_gate is not None, "allocated preflight requires the final authorization gate")
        require(args.preauthorization_manifest is None, "allocated preflight must not substitute a partial manifest")
        require(
            args.static_preflight_receipt is not None,
            "allocated preflight requires the allocation-free static contract",
        )
    experiment = read_json(args.experiment)
    require(experiment.get("schema_version") == "apertus_hard_h_to_g_replication_v2", "experiment contract drift")
    authorization_stage = expected_authorization_stage(
        phase=args.phase,
        start_update=args.start_update,
        resume_smoke=args.one_update_resume_smoke,
    )
    if args.preallocation_static:
        accepted_producers = validate_preauthorization_manifest(
            args.preauthorization_manifest,
            expected_stage=authorization_stage,
            scale=args.scale,
        )
    else:
        _, accepted_producers = validate_authorization_gate(
            args.authorization_gate,
            experiment_path=args.experiment,
            expected_stage=authorization_stage,
            scale=args.scale,
        )
    validate_segment_boundaries(
        phase=args.phase,
        start_update=args.start_update,
        exit_update=args.exit_update,
        one_update_resume_smoke=args.one_update_resume_smoke,
        canonical_resume=args.canonical_resume,
    )
    accepted_code_bundles = code_bundle_identities(accepted_producers)
    require(args.load_checkpoint.is_dir(), "load checkpoint root missing")
    cache = read_json(args.phase_cache_receipt)
    validate_phase_cache(
        cache,
        phase=args.phase,
        data_path_spec=args.phase_data_path_spec,
        cache_root=args.phase_cache_root,
        accepted_code_bundles=accepted_code_bundles,
        verify_payload_hashes=False,
    )
    require(shlex.split(args.phase_data_path) == cache.get("data_path_tokens"), "target phase --data-path token drift")
    require(cache.get("cache_tree_sha256") == args.phase_cache_tree_sha256, "target phase cache SHA-256 environment drift")
    megatron = read_json(args.megatron_receipt)
    patch = Path(str(megatron.get("patch", {}).get("path", "")))
    validate_megatron_runtime(megatron, args.megatron_root, patch, require_helpers=True)
    online_validation = read_json(args.validation_receipt)
    validate_online_validation(
        online_validation,
        args.validation_root,
        accepted_code_bundles=accepted_code_bundles,
        verify_payload_hashes=False,
    )
    require(tuple(args.extra_valid_sets.split()) == ONLINE_PANEL_NAMES, "online validation environment inventory/order drift")
    require(frozenset(args.new_greek_valid_sets.split()) == ONLINE_NEW_GREEK, "online validation Greek-family environment drift")
    runtime_compat = read_json(args.runtime_compat_receipt)
    runtime_compat_file = args.runtime_compat_dir / "sitecustomize.py"
    require(
        runtime_compat.get("schema_version") == "apertus_hard_h_to_g_dcp_metadata_compat_v1"
        and runtime_compat.get("status") == "passed"
        and runtime_compat.get("torch_version") == "2.9.1"
        and runtime_compat.get("mcore_data_round_trip") is True
        and runtime_compat.get("shim_loaded") is True,
        "DCP metadata compatibility receipt did not pass",
    )
    require(
        runtime_compat.get("runtime_compat_dir") == str(args.runtime_compat_dir.resolve())
        and runtime_compat.get("sitecustomize") == file_binding(runtime_compat_file),
        "DCP metadata compatibility binding drift",
    )
    runtime_compat_contract = {
        "schema_version": runtime_compat["schema_version"],
        "status": runtime_compat["status"],
        "torch_version": runtime_compat.get("torch_version"),
        "mcore_data_round_trip": runtime_compat["mcore_data_round_trip"],
        "shim_loaded": runtime_compat["shim_loaded"],
        "runtime_compat_dir": runtime_compat["runtime_compat_dir"],
        "sitecustomize": runtime_compat["sitecustomize"],
    }
    if args.start_update > 0:
        require(args.initialization_permit is None, "resumed segment must not carry an initialization permit")
        require(args.checkpoint_permit is not None and args.checkpoint_permit.is_file(), "resumed segment lacks checkpoint permit")
        require(
            args.source_phase_cache_receipt is not None and args.source_phase_cache_receipt.is_file(),
            "resumed segment lacks source-phase cache receipt",
        )
        phase_low = {1: 0, 2: 2261, 3: 3218}[args.phase]
        source_phase = (
            args.phase - 1
            if args.phase > 1 and args.start_update == phase_low
            else args.phase
        )
        source_cache = read_json(args.source_phase_cache_receipt)
        validate_phase_cache(
            source_cache,
            phase=source_phase,
            data_path_spec=Path(str(source_cache.get("data_path_spec", {}).get("path", ""))),
            cache_root=Path(str(source_cache.get("cache_root", ""))),
            accepted_code_bundles=accepted_code_bundles,
            verify_payload_hashes=False,
        )
        permit = read_json(args.checkpoint_permit)
        checkpoint_root = args.load_checkpoint / f"iter_{args.start_update:07d}"
        require(checkpoint_root.is_dir(), "resumed segment load root lacks the permitted iteration directory")
        validate_checkpoint_permit(
            permit,
            scale=args.scale,
            source_phase=source_phase,
            update=args.start_update,
            checkpoint_root=checkpoint_root,
            source_phase_cache_receipt=args.source_phase_cache_receipt,
        )
        permit_audit_binding = permit.get("checkpoint_audit")
        if args.canonical_resume:
            require(isinstance(permit_audit_binding, dict), "canonical resume permit lacks checkpoint audit")
            permit_audit_path = Path(str(permit_audit_binding.get("path", "")))
            permit_audit = read_json(permit_audit_path)
            require(permit_audit_binding == file_binding(permit_audit_path), "canonical resume checkpoint-audit binding drift")
            require(permit_audit.get("gracefully_stopped") is True, "canonical resume source was not gracefully stopped")
        require(Path(str(permit.get("load_root", ""))).resolve() == args.load_checkpoint.resolve(), "resumed segment Megatron load-root drift")
    else:
        require(args.checkpoint_permit is None, "initial segment must not carry a checkpoint permit")
        require(args.source_phase_cache_receipt is None, "initial segment must not carry a source-phase cache receipt")
        require(args.initialization_permit is not None and args.initialization_permit.is_file(), "initial segment lacks initialization permit")
        initialization = read_json(args.initialization_permit)
        require(initialization.get("schema_version") == "apertus_targeted_init_roundtrip_v1" and initialization.get("status") == "passed", "initialization permit is not a passed roundtrip")
        require(initialization.get("scale") == args.scale, "initialization permit scale drift")
        require(Path(str(initialization.get("megatron_root", ""))).resolve() == args.load_checkpoint.resolve(), "initialization permit load-root drift")
        require_accepted_producer(initialization, accepted_producers, "initialization permit")
        validate_initialization_load_contract(
            initialization,
            experiment=experiment,
            scale=args.scale,
            tensor_parallel=args.tensor_parallel,
            load_root=args.load_checkpoint,
        )
    world = args.nodes * 4
    require(world % args.tensor_parallel == 0, "world size is not divisible by tensor parallel")
    data_parallel = world // args.tensor_parallel
    require(1024 % (data_parallel * args.microbatch) == 0, "global batch is not divisible by DP times microbatch")
    run_permit = read_json(args.training_run_permit)
    validate_training_run_permit(
        run_permit,
        scale=args.scale,
        nodes=args.nodes,
        tensor_parallel=args.tensor_parallel,
        microbatch=args.microbatch,
        peak_lr=args.peak_lr,
        floor_lr=args.floor_lr,
    )
    permit_profile = run_permit.get("profile")
    require(
        isinstance(permit_profile, dict)
        and int(permit_profile.get("data_parallel", -1)) == data_parallel
        and int(permit_profile.get("gradient_accumulation_microbatches", -1))
        == 1024 // (data_parallel * args.microbatch),
        "training run-permit parallelism drift",
    )
    segment_contract = {
        "scale": args.scale,
        "phase": args.phase,
        "start_update": args.start_update,
        "exit_update": args.exit_update,
        "nodes": args.nodes,
        "world_size": world,
        "tensor_parallel": args.tensor_parallel,
        "data_parallel": data_parallel,
        "microbatch": args.microbatch,
        "gradient_accumulation_microbatches": 1024 // (data_parallel * args.microbatch),
        "experiment": file_binding(args.experiment),
        "phase_cache_receipt": file_binding(args.phase_cache_receipt),
        "phase_cache_tree_sha256": args.phase_cache_tree_sha256,
        "phase_cache_root": str(args.phase_cache_root.resolve()),
        "source_phase_cache_receipt": file_binding(args.source_phase_cache_receipt) if args.source_phase_cache_receipt else None,
        "megatron_receipt": file_binding(args.megatron_receipt),
        "megatron_root": str(args.megatron_root.resolve()),
        "validation_receipt": file_binding(args.validation_receipt),
        "validation_root": str(args.validation_root.resolve()),
        "runtime_compat_dir": str(args.runtime_compat_dir.resolve()),
        # The proof receipt is regenerated for every real Slurm job and thus
        # necessarily has a different path, timestamp and job id than the
        # allocation-free proof. Compare only its invariant scientific
        # contract here; retain the exact receipt binding in the outer receipt.
        "runtime_compat_contract": runtime_compat_contract,
        "checkpoint_permit": file_binding(args.checkpoint_permit) if args.checkpoint_permit else None,
        "initialization_permit": file_binding(args.initialization_permit) if args.initialization_permit else None,
        "training_run_permit": file_binding(args.training_run_permit),
        "authorization_stage": authorization_stage,
        "peak_lr": args.peak_lr,
        "floor_lr": args.floor_lr,
        "load_checkpoint": str(args.load_checkpoint.resolve()),
        "phase_data_path_spec": file_binding(args.phase_data_path_spec),
        "phase_data_path": args.phase_data_path,
        "one_update_resume_smoke": args.one_update_resume_smoke,
        "canonical_resume": args.canonical_resume,
        "executing_code_bundle": executing_code_bundle(),
    }
    static_binding = None
    if args.static_preflight_receipt is not None:
        static = read_json(args.static_preflight_receipt)
        require(
            static.get("schema_version") == "apertus_hard_h_to_g_train_segment_preflight_v1"
            and static.get("status") == "passed"
            and static.get("preallocation_static") is True,
            "preallocation static preflight is not a passed static receipt",
        )
        require(
            static.get("segment_contract") == segment_contract,
            "allocated segment differs from its preallocation static preflight",
        )
        static_binding = file_binding(args.static_preflight_receipt)
    receipt = {
        "schema_version": "apertus_hard_h_to_g_train_segment_preflight_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "preallocation_static": args.preallocation_static,
        "static_preflight_receipt": static_binding,
        "authorization_gate": file_binding(args.authorization_gate) if args.authorization_gate else None,
        "preauthorization_manifest": file_binding(args.preauthorization_manifest) if args.preauthorization_manifest else None,
        "runtime_compat_receipt": file_binding(args.runtime_compat_receipt),
        "segment_contract": segment_contract,
        **segment_contract,
    }
    write_json_atomic(args.output, receipt)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
