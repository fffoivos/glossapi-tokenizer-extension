#!/usr/bin/env python3
"""Fail closed inside a normal allocation before a profile/LR benchmark."""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path

from contract_utils import (
    executing_code_bundle,
    file_binding,
    read_json,
    require,
    write_json_atomic,
)
from finalize_training_megatron import validate_runtime as validate_megatron_runtime
from freeze_online_validation_binaries import (
    validate_receipt as validate_online_validation,
)
from freeze_phase_blend_cache import validate_receipt as validate_phase_cache
from materialize_phase_cache import validate_overlay_receipt
from producer_bundle_compatibility import load_authority, require_accepted_producer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--megatron-root", type=Path, required=True)
    parser.add_argument("--megatron-receipt", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--validation-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable benchmark preflight exists: {args.output}")
    require(not args.output_root.exists(), f"benchmark output root already exists: {args.output_root}")
    value = read_json(args.contract)
    require(value.get("schema_version") == "apertus_hard_h_to_g_prelaunch_benchmark_contract_v1", "benchmark contract schema drift")
    require(value.get("status") == "frozen" and value.get("kind") in {"profile", "lr"}, "benchmark contract status/kind drift")
    require(Path(str(value.get("output_root", ""))).resolve() == args.output_root.resolve(), "benchmark output-root drift")
    require(int(os.environ.get("SLURM_NNODES", "0")) == int(value.get("nodes", -1)), "live allocation node drift")
    require(os.environ.get("SLURM_JOB_PARTITION") == "normal", "benchmark allocation is not normal")
    current = executing_code_bundle()
    bundle = value.get("executing_code_bundle")
    require(isinstance(bundle, dict) and bundle.get("root") == current["root"] and bundle.get("tree_sha256") == current["tree_sha256"], "benchmark contract code-bundle drift")
    compatibility_binding = value.get("producer_compatibility")
    require(isinstance(compatibility_binding, dict), "benchmark producer-compatibility binding missing")
    compatibility_path = Path(str(compatibility_binding.get("path", "")))
    require(compatibility_path.is_file() and compatibility_binding == file_binding(compatibility_path), "benchmark producer-compatibility binding drift")
    _, accepted_producers = load_authority(compatibility_path, current)
    accepted_code_bundles = {(root, tree) for root, tree, *_ in accepted_producers}
    initialization_binding = value.get("initialization_permit")
    require(isinstance(initialization_binding, dict), "initialization binding missing")
    initialization_path = Path(str(initialization_binding.get("path", "")))
    require(initialization_path.is_file() and initialization_binding == file_binding(initialization_path), "initialization binding drift")
    initialization = read_json(initialization_path)
    require(initialization.get("schema_version") == "apertus_targeted_init_roundtrip_v1" and initialization.get("status") == "passed" and initialization.get("scale") == value.get("scale"), "initialization permit drift")
    require_accepted_producer(initialization, accepted_producers, "benchmark initialization")
    require(Path(str(initialization.get("megatron_root", ""))).resolve() == Path(str(value.get("initialization_root", ""))).resolve(), "initialization root drift")
    cache_binding = value.get("phase_cache_receipt")
    require(isinstance(cache_binding, dict), "Phase-1 cache binding missing")
    cache_path = Path(str(cache_binding.get("path", "")))
    require(cache_path.is_file() and cache_binding == file_binding(cache_path), "Phase-1 cache binding drift")
    cache = read_json(cache_path)
    require_accepted_producer(cache, accepted_producers, "Phase-1 benchmark cache")
    validate_phase_cache(cache, phase=1, data_path_spec=Path(str(cache["data_path_spec"]["path"])), cache_root=Path(str(cache["cache_root"])), accepted_code_bundles=accepted_code_bundles)
    overlay_binding = value.get("phase_cache_overlay_receipt")
    require(isinstance(overlay_binding, dict), "Phase-1 cache overlay binding missing")
    overlay_path = Path(str(overlay_binding.get("path", "")))
    require(overlay_path.is_file() and overlay_binding == file_binding(overlay_path), "Phase-1 cache overlay binding drift")
    overlay = read_json(overlay_path)
    require_accepted_producer(overlay, accepted_producers, "Phase-1 benchmark cache overlay")
    validate_overlay_receipt(
        overlay, phase=1, overlay_root=Path(str(overlay.get("overlay_root", ""))),
        accepted_code_bundles=accepted_code_bundles, require_pristine=True,
    )
    require(overlay.get("source_cache_receipt") == file_binding(cache_path), "Phase-1 cache overlay source drift")
    require(Path(str(value.get("phase_cache_root", ""))).resolve() == Path(str(overlay["overlay_root"])).resolve(), "Phase-1 cache overlay-root drift")
    require(cache.get("cache_tree_sha256") == value.get("phase_cache_tree_sha256"), "Phase-1 cache tree drift")
    phase2_binding = value.get("phase2_cache_receipt")
    require(isinstance(phase2_binding, dict), "Phase-2 cache binding missing")
    phase2_path = Path(str(phase2_binding.get("path", "")))
    require(phase2_path.is_file() and phase2_binding == file_binding(phase2_path), "Phase-2 cache binding drift")
    phase2 = read_json(phase2_path)
    require_accepted_producer(phase2, accepted_producers, "Phase-2 benchmark cache")
    validate_phase_cache(
        phase2, phase=2,
        data_path_spec=Path(str(phase2["data_path_spec"]["path"])),
        cache_root=Path(str(phase2["cache_root"])),
        accepted_code_bundles=accepted_code_bundles,
    )
    phase2_overlay_binding = value.get("phase2_cache_overlay_receipt")
    require(isinstance(phase2_overlay_binding, dict), "Phase-2 cache overlay binding missing")
    phase2_overlay_path = Path(str(phase2_overlay_binding.get("path", "")))
    require(phase2_overlay_path.is_file() and phase2_overlay_binding == file_binding(phase2_overlay_path), "Phase-2 cache overlay binding drift")
    phase2_overlay = read_json(phase2_overlay_path)
    require_accepted_producer(phase2_overlay, accepted_producers, "Phase-2 benchmark cache overlay")
    validate_overlay_receipt(
        phase2_overlay, phase=2, overlay_root=Path(str(phase2_overlay.get("overlay_root", ""))),
        accepted_code_bundles=accepted_code_bundles, require_pristine=True,
    )
    require(phase2_overlay.get("source_cache_receipt") == file_binding(phase2_path), "Phase-2 cache overlay source drift")
    require(Path(str(value.get("phase2_cache_root", ""))).resolve() == Path(str(phase2_overlay["overlay_root"])).resolve(), "Phase-2 cache overlay-root drift")
    require(phase2.get("cache_tree_sha256") == value.get("phase2_cache_tree_sha256"), "Phase-2 cache tree drift")
    megatron = read_json(args.megatron_receipt)
    require_accepted_producer(megatron, accepted_producers, "benchmark Megatron runtime")
    validate_megatron_runtime(megatron, args.megatron_root, Path(str(megatron.get("patch", {}).get("path", ""))))
    validation = read_json(args.validation_receipt)
    require_accepted_producer(validation, accepted_producers, "benchmark online validation")
    validate_online_validation(validation, args.validation_root, accepted_code_bundles=accepted_code_bundles)
    expected_panels = list(value.get("online_validation_panels", []))
    expected_new_greek = list(value.get("online_validation_new_greek_panels", []))
    require(expected_panels == validation.get("panel_names"), "benchmark online-validation panel order drift")
    require(expected_new_greek == ["hplt", "openarchives", "greek_phd"], "benchmark new-Greek panel order drift")
    require(os.environ.get("H2G_EXTRA_VALID_SETS", "").split() == expected_panels, "benchmark validation environment panel order drift")
    require(os.environ.get("H2G_NEW_GREEK_VALID_SETS", "").split() == expected_new_greek, "benchmark new-Greek validation environment drift")
    payload = {
        "schema_version": "apertus_hard_h_to_g_prelaunch_benchmark_preflight_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodes": int(os.environ.get("SLURM_NNODES", "0")),
        "contract": file_binding(args.contract),
        "producer_compatibility": file_binding(compatibility_path),
        "megatron_receipt": file_binding(args.megatron_receipt),
        "validation_receipt": file_binding(args.validation_receipt),
        "phase1_cache_receipt": file_binding(cache_path),
        "phase1_cache_overlay_receipt": file_binding(overlay_path),
        "phase2_cache_receipt": file_binding(phase2_path),
        "phase2_cache_overlay_receipt": file_binding(phase2_overlay_path),
        "executing_code_bundle": current,
    }
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
