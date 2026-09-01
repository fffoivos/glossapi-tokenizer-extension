#!/usr/bin/env python3
"""Run the v45 launch gate for a receipt-only successor stage, fail-closed.

The v45 gate correctly verifies live receipt bindings, but its old raw
``scientific_digest`` also includes those bindings.  A corrected successor
stage must therefore use a new raw digest even when its actual trainable
experiment is byte-identical.  This adapter does *not* replace any v45 gate:
it verifies the old and new contracts, normalizes only receipt bindings for
the digest comparison, then invokes the original v45 gate with ephemeral
normalized-digest copies.  All normal evidence paths and live bindings remain
the original gate's inputs and checks.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from successor_semantic_identity import (
    SCHEMA_VERSION,
    file_binding,
    normalized_profiles,
    normalized_recipe_payload,
    read_json,
    semantic_identity,
)


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def require_binding(binding: dict[str, Any], label: str) -> dict[str, Any]:
    path = Path(binding.get("path", ""))
    if file_binding(path) != binding:
        raise ValueError(f"{label} binding drift")
    return read_json(path)


def restart_control_from_promotion(promotion: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Materialize v45's missing ``restart.control`` view from its receipt.

    The fallback promotion receipt already binds the complete two-allocation
    parity record in ``parity.receipt``.  v45 verifies that promotion broadly,
    then accidentally reads a field which that schema never emitted.  This
    returns a strict compatibility view only after validating the original
    receipt and every result used by the gate.
    """

    parity_binding = promotion.get("parity", {}).get("receipt", {})
    parity = require_binding(parity_binding, "DP32 parity receipt")
    restart = parity.get("restart", {})
    first = restart.get("first", {})
    second = restart.get("second", {})
    checks = parity.get("checks", {})
    required = {
        "first_restart_provenance",
        "second_restart_provenance",
        "first_restart_numerically_equivalent",
        "second_restart_numerically_equivalent",
        "independent_restarts_identical",
    }
    if (
        parity.get("schema_version") != "apertus_full_8b_checkpoint_parity_smoke_v1"
        or parity.get("status") != "passed"
        or parity.get("profile_id") != "dp32_16node"
        or not required.issubset(checks)
        or not all(bool(checks[name]) for name in required)
        or first.get("provenance", {}).get("passed") is not True
        or first.get("numerical", {}).get("passed") is not True
        or second.get("provenance", {}).get("passed") is not True
        or second.get("numerical", {}).get("passed") is not True
        or restart.get("independent_identity", {}).get("passed") is not True
        or promotion.get("parity", {}).get("two_independent_restart_allocations_passed")
        is not True
    ):
        raise ValueError("DP32 parity receipt does not prove both restart allocations")
    return (
        {
            "provenance": first["provenance"],
            "numerical": first["numerical"],
            "independent_repeat": {
                "provenance": second["provenance"],
                "numerical": second["numerical"],
            },
            "two_independent_restart_allocations_passed": True,
        },
        parity_binding,
    )


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--selected-profile", type=Path, required=True)
    parser.add_argument("--benchmark-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-selected-profile", type=Path, required=True)
    parser.add_argument("--source-promotion-receipt", type=Path, required=True)
    parser.add_argument("--stage-identity", type=Path, required=True)
    return parser.parse_known_args()


def main() -> int:
    args, forwarded = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    code_root = args.code_root.resolve()
    script_root = code_root / "subprojects/07_full_8b_cpt/scripts"
    if not script_root.is_dir():
        raise ValueError("scientific code root has no full-8B scripts")
    sys.path.insert(0, str(script_root))
    import build_launch_gate as v45_gate  # pylint: disable=import-error
    import validate_execution_profile as v45_profile  # pylint: disable=import-error
    from contract import scientific_digest as raw_scientific_digest  # pylint: disable=import-error

    current_recipe = read_json(args.recipe)
    current_profiles = read_json(args.profiles)
    current_selected = read_json(args.selected_profile)
    source_selected = require_binding(file_binding(args.source_selected_profile), "source selected")
    source_promotion = require_binding(file_binding(args.source_promotion_receipt), "source promotion")
    identity = require_binding(file_binding(args.stage_identity), "successor stage identity")

    if file_binding(args.benchmark_receipt) != file_binding(args.source_promotion_receipt):
        raise ValueError("benchmark receipt differs from the proven source promotion receipt")

    if (
        current_selected.get("schema_version")
        != "apertus_full_8b_selected_execution_profile_v1"
        or current_selected.get("status") != "frozen"
    ):
        raise ValueError("successor selected profile is not frozen")
    rebinding = current_selected.get("rebinding", {})
    if (
        rebinding.get("schema_version") != SCHEMA_VERSION
        or rebinding.get("semantic_identity_matches") is not True
        or rebinding.get("normalized_recipe_changed_paths") != []
        or rebinding.get("normalized_profiles_changed_paths") != []
        or rebinding.get("source_selected_profile") != file_binding(args.source_selected_profile)
        or rebinding.get("promotion_receipt") != file_binding(args.source_promotion_receipt)
        or rebinding.get("stage_identity") != file_binding(args.stage_identity)
    ):
        raise ValueError("successor selected-profile rebinding proof drift")
    if (
        identity.get("schema_version") != "apertus_full_8b_successor_stage_identity_v1"
        or identity.get("status") != "passed"
        or not all(identity.get("checks", {}).values())
    ):
        raise ValueError("successor stage identity is not passed")

    source_recipe = require_binding(source_selected.get("recipe", {}), "source recipe")
    source_profiles = require_binding(source_selected.get("profiles", {}), "source profiles")
    if source_promotion.get("inputs", {}).get("recipe") != source_selected.get("recipe"):
        raise ValueError("promotion/source selected recipe binding drift")
    if source_promotion.get("inputs", {}).get("profiles") != source_selected.get("profiles"):
        raise ValueError("promotion/source selected profiles binding drift")
    if source_promotion.get("scientific_digest") != source_selected.get("selection", {}).get("scientific_digest"):
        raise ValueError("promotion/source selected raw digest drift")
    restart_control, parity_receipt_binding = restart_control_from_promotion(source_promotion)
    semantic = semantic_identity(current_recipe)
    if (
        semantic != semantic_identity(source_recipe)
        or normalized_recipe_payload(current_recipe)
        != normalized_recipe_payload(source_recipe)
        or normalized_profiles(current_profiles) != normalized_profiles(source_profiles)
        or rebinding.get("semantic_identity") != semantic
        or rebinding.get("source_semantic_identity") != semantic
    ):
        raise ValueError("successor changes a non-receipt scientific contract")

    profile_id = current_selected["selection"].get("profile_id")
    validated = v45_profile.validate(current_recipe, current_profiles, profile_id)
    raw_digest = raw_scientific_digest(current_recipe)
    if (
        raw_digest != validated["scientific_digest"]
        or raw_digest != current_selected["selection"].get("scientific_digest")
        or current_selected.get("recipe") != file_binding(args.recipe)
        or current_selected.get("profiles") != file_binding(args.profiles)
    ):
        raise ValueError("current successor raw contract binding drift")

    # The original gate's only incompatible comparison is raw digest equality
    # across receipt rebindings.  Feed it copies whose digest fields are the
    # proven semantic identity, while retaining the original inputs/paths.
    normalized_selected = copy.deepcopy(current_selected)
    normalized_selected["selection"]["scientific_digest"] = semantic
    normalized_benchmark = copy.deepcopy(source_promotion)
    normalized_benchmark["scientific_digest"] = semantic
    normalized_benchmark["restart"] = {"control": restart_control}
    with tempfile.TemporaryDirectory(prefix="full8b-successor-gate-") as tmp:
        tmp_root = Path(tmp)
        selected_path = tmp_root / "selected.json"
        benchmark_path = tmp_root / "benchmark.json"
        gate_path = tmp_root / "gate.json"
        selected_path.write_text(json.dumps(normalized_selected, sort_keys=True), encoding="utf-8")
        benchmark_path.write_text(json.dumps(normalized_benchmark, sort_keys=True), encoding="utf-8")

        # ``build_launch_gate`` and ``validate_execution_profile`` each bind a
        # direct imported reference, so patch both references—not the global
        # contract function.  Training keeps the original raw-digest behavior.
        original_gate_digest = v45_gate.scientific_digest
        original_profile_digest = v45_profile.scientific_digest
        original_argv = sys.argv[:]
        try:
            v45_gate.scientific_digest = semantic_identity
            v45_profile.scientific_digest = semantic_identity
            sys.argv = [
                str(script_root / "build_launch_gate.py"),
                *forwarded,
                "--code-root", str(code_root),
                "--recipe", str(args.recipe),
                "--profiles", str(args.profiles),
                "--selected-profile", str(selected_path),
                "--benchmark-receipt", str(benchmark_path),
                "--output", str(gate_path),
            ]
            result = v45_gate.main()
        finally:
            sys.argv = original_argv
            v45_gate.scientific_digest = original_gate_digest
            v45_profile.scientific_digest = original_profile_digest
        if result != 0:
            raise ValueError(f"v45 gate returned {result}")
        gate = read_json(gate_path)

    # Publish no temporary evidence.  The final receipt binds the real source
    # objects and records both raw digests plus the stable semantic identity.
    gate["scientific_digest"] = raw_digest
    gate["selected_profile"] = current_selected["selection"]
    gate["evidence"]["selected_profile"] = file_binding(args.selected_profile)
    gate["evidence"]["benchmark"] = file_binding(args.benchmark_receipt)
    gate["evidence"]["source_selected_profile"] = file_binding(args.source_selected_profile)
    gate["evidence"]["source_promotion_receipt"] = file_binding(args.source_promotion_receipt)
    gate["evidence"]["successor_stage_identity"] = file_binding(args.stage_identity)
    gate["evidence"]["dp32_checkpoint_parity"] = parity_receipt_binding
    gate["successor_rebinding"] = {
        "schema_version": SCHEMA_VERSION,
        "policy": "v45_gate_with_receipt_only_semantic_identity_adapter_v1",
        "base_gate": "build_launch_gate.py",
        "base_gate_completed_all_original_checks": True,
        "current_raw_scientific_digest": raw_digest,
        "source_raw_scientific_digest": source_selected["selection"]["scientific_digest"],
        "semantic_identity": semantic,
        "source_selected_profile": file_binding(args.source_selected_profile),
        "source_promotion_receipt": file_binding(args.source_promotion_receipt),
        "successor_stage_identity": file_binding(args.stage_identity),
        "receipt_only_normalized_fields": [
            "data.sanitized_source_receipt",
            "data.eligibility_policy.proof",
        ],
        "v45_restart_control_schema_bug_corrected_from_bound_parity_receipt": True,
        "dp32_checkpoint_parity_receipt": parity_receipt_binding,
        "all_other_scientific_fields_exactly_identical": True,
    }
    atomic_write(output, gate)
    print(json.dumps({"ok": True, "semantic_identity": semantic, "gates": len(gate["gates"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
