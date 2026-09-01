#!/usr/bin/env python3
"""Freeze one profile or LR-pilot allocation before it can be submitted."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from contract_utils import (
    executing_code_bundle,
    file_binding,
    read_json,
    require,
    write_json_atomic,
)
from freeze_phase_blend_cache import validate_receipt as validate_phase_cache
from materialize_phase_cache import validate_overlay_receipt
from producer_bundle_compatibility import load_authority, require_accepted_producer

FLOORS = {"5.5e-5": "5.5e-6", "7.5e-5": "7.5e-6", "1.0e-4": "1.0e-5", "1.25e-4": "1.25e-5"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("profile", "lr"), required=True)
    parser.add_argument("--scale", choices=("8b", "1p5b"), required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--tensor-parallel", type=int, required=True)
    parser.add_argument("--microbatch", type=int, required=True)
    parser.add_argument("--peak-lr", required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--initialization-permit", type=Path, required=True)
    parser.add_argument("--producer-compatibility", type=Path, required=True)
    parser.add_argument("--phase1-cache-receipt", type=Path, required=True)
    parser.add_argument("--phase1-cache-overlay-receipt", type=Path, required=True)
    parser.add_argument("--phase2-cache-receipt", type=Path, required=True)
    parser.add_argument("--phase2-cache-overlay-receipt", type=Path, required=True)
    parser.add_argument("--profile-promotion", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable benchmark contract exists: {args.output}")
    require(not args.output_root.exists(), f"benchmark output root already exists: {args.output_root}")
    experiment = read_json(args.experiment)
    allocation = read_json(args.allocation)
    current = executing_code_bundle()
    _, accepted_producers = load_authority(args.producer_compatibility, current)
    accepted_code_bundles = {(root, tree) for root, tree, *_ in accepted_producers}
    require(experiment.get("schema_version") == "apertus_hard_h_to_g_replication_v2", "experiment contract drift")
    require(allocation.get("schema_version") == "apertus_hard_h_to_g_allocation_v1", "allocation contract drift")
    if args.scale == "8b":
        allowed = [allocation["profiles"]["8b"]]
    else:
        allowed = allocation["profiles"]["1p5b_candidates"]
    matches = [row for row in allowed if row["profile_id"] == args.profile_id]
    require(len(matches) == 1, "profile is outside frozen grid")
    profile = matches[0]
    require((args.nodes, args.tensor_parallel, args.microbatch) == (profile["nodes"], profile["tensor_parallel"], profile["microbatch"]), "benchmark decomposition drift")
    world = args.nodes * 4
    data_parallel = world // args.tensor_parallel
    require(world == args.tensor_parallel * data_parallel and 1024 % (data_parallel * args.microbatch) == 0, "benchmark global-batch geometry drift")
    if args.kind == "profile":
        require(args.profile_promotion is None, "profile benchmark cannot depend on its own promotion")
        expected_peak = "5.5e-5" if args.scale == "8b" else "1.0e-4"
        require(args.peak_lr == expected_peak, "profile benchmark peak LR drift")
        updates = 256
    else:
        require(args.scale == "1p5b" and args.peak_lr in experiment["learning_rate_pilot_1p5b"]["candidates"], "LR-pilot candidate drift")
        require(args.profile_promotion is not None and args.profile_promotion.is_file(), "LR pilot lacks promoted profile")
        promotion = read_json(args.profile_promotion)
        require(promotion.get("schema_version") == "apertus_hard_h_to_g_profile_promotion_v1" and promotion.get("status") == "promoted" and promotion.get("scale") == "1p5b", "LR-pilot profile promotion drift")
        selected = promotion.get("selection", {})
        require((selected.get("profile_id"), selected.get("nodes"), selected.get("tensor_parallel"), selected.get("microbatch")) == (args.profile_id, args.nodes, args.tensor_parallel, args.microbatch), "LR-pilot does not use promoted profile")
        updates = 238
    initialization = read_json(args.initialization_permit)
    require(initialization.get("schema_version") == "apertus_targeted_init_roundtrip_v1" and initialization.get("status") == "passed" and initialization.get("scale") == args.scale, "benchmark initialization permit drift")
    require_accepted_producer(initialization, accepted_producers, "benchmark initialization")
    phase_cache = read_json(args.phase1_cache_receipt)
    require_accepted_producer(phase_cache, accepted_producers, "Phase-1 benchmark cache")
    validate_phase_cache(
        phase_cache, phase=1,
        data_path_spec=Path(str(phase_cache.get("data_path_spec", {}).get("path", ""))),
        cache_root=Path(str(phase_cache.get("cache_root", ""))),
        accepted_code_bundles=accepted_code_bundles,
    )
    phase1_overlay = read_json(args.phase1_cache_overlay_receipt)
    require_accepted_producer(phase1_overlay, accepted_producers, "Phase-1 benchmark cache overlay")
    validate_overlay_receipt(
        phase1_overlay,
        phase=1,
        overlay_root=Path(str(phase1_overlay.get("overlay_root", ""))),
        accepted_code_bundles=accepted_code_bundles,
        require_pristine=True,
    )
    require(
        phase1_overlay.get("source_cache_receipt") == file_binding(args.phase1_cache_receipt),
        "Phase-1 benchmark overlay source drift",
    )
    phase2_cache = read_json(args.phase2_cache_receipt)
    require_accepted_producer(phase2_cache, accepted_producers, "Phase-2 benchmark cache")
    validate_phase_cache(
        phase2_cache, phase=2,
        data_path_spec=Path(str(phase2_cache.get("data_path_spec", {}).get("path", ""))),
        cache_root=Path(str(phase2_cache.get("cache_root", ""))),
        accepted_code_bundles=accepted_code_bundles,
    )
    phase2_overlay = read_json(args.phase2_cache_overlay_receipt)
    require_accepted_producer(phase2_overlay, accepted_producers, "Phase-2 benchmark cache overlay")
    validate_overlay_receipt(
        phase2_overlay,
        phase=2,
        overlay_root=Path(str(phase2_overlay.get("overlay_root", ""))),
        accepted_code_bundles=accepted_code_bundles,
        require_pristine=True,
    )
    require(
        phase2_overlay.get("source_cache_receipt") == file_binding(args.phase2_cache_receipt),
        "Phase-2 benchmark overlay source drift",
    )
    payload = {
        "schema_version": "apertus_hard_h_to_g_prelaunch_benchmark_contract_v1",
        "status": "frozen",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "executing_code_bundle": current,
        "producer_compatibility": file_binding(args.producer_compatibility),
        "kind": args.kind,
        "scale": args.scale,
        "profile_id": args.profile_id,
        "nodes": args.nodes,
        "gpus_per_node": 4,
        "tensor_parallel": args.tensor_parallel,
        "data_parallel": data_parallel,
        "microbatch": args.microbatch,
        "gradient_accumulation_microbatches": 1024 // (data_parallel * args.microbatch),
        "global_batch_sequences": 1024,
        "updates": updates,
        "peak_lr": args.peak_lr,
        "floor_lr": FLOORS[args.peak_lr],
        "experiment": file_binding(args.experiment),
        "allocation": file_binding(args.allocation),
        "initialization_permit": file_binding(args.initialization_permit),
        "initialization_root": initialization["megatron_root"],
        "phase_cache_receipt": file_binding(args.phase1_cache_receipt),
        "phase_cache_overlay_receipt": file_binding(args.phase1_cache_overlay_receipt),
        "phase_cache_root": phase1_overlay["overlay_root"],
        "phase_cache_tree_sha256": phase_cache["cache_tree_sha256"],
        "phase_data_path_spec": phase_cache["data_path_spec"],
        "phase_data_path": " ".join(phase_cache["data_path_tokens"]),
        "phase2_cache_receipt": file_binding(args.phase2_cache_receipt),
        "phase2_cache_overlay_receipt": file_binding(args.phase2_cache_overlay_receipt),
        "phase2_cache_root": phase2_overlay["overlay_root"],
        "phase2_cache_tree_sha256": phase2_cache["cache_tree_sha256"],
        "phase2_data_path_spec": phase2_cache["data_path_spec"],
        "phase2_data_path": " ".join(phase2_cache["data_path_tokens"]),
        "profile_promotion": file_binding(args.profile_promotion) if args.profile_promotion else None,
        "output_root": str(args.output_root.resolve()),
        "benchmark_access": {
            "greekmmlu": False,
            "native_suite": False,
            "source_conditioned_validation": args.kind == "lr",
        },
        "online_validation_panels": experiment["evaluation"]["historical_online_panels"],
        "online_validation_new_greek_panels": ["hplt", "openarchives", "greek_phd"],
        "restart_design": (
            "phase1_uninterrupted_0_to_2_and_resume_1_to_2_plus_"
            "phase2_entry_1_to_3_and_resume_2_to_3_with_phase_local_cursors_0_and_1024"
            if args.kind == "profile" else None
        ),
    }
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
