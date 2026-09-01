#!/usr/bin/env python3
"""Rebind a proven DP32 selection to scientifically identical successor contracts."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--old-selected", type=Path, required=True)
    parser.add_argument("--promotion-receipt", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--stage-identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.code_root / "subprojects/07_full_8b_cpt/scripts"))
    from contract import atomic_write_json, file_binding, read_json, scientific_digest
    from validate_execution_profile import validate
    from successor_semantic_identity import (
        SCHEMA_VERSION,
        changed_paths,
        normalized_profiles,
        normalized_recipe_payload,
        semantic_identity,
    )

    old = read_json(args.old_selected)
    promotion = read_json(args.promotion_receipt)
    identity = read_json(args.stage_identity)
    recipe = read_json(args.recipe)
    profiles = read_json(args.profiles)
    if old.get("schema_version") != "apertus_full_8b_selected_execution_profile_v1" or old.get("status") != "frozen":
        raise ValueError("old selected profile is not frozen")
    if promotion.get("schema_version") != "apertus_full_8b_dp32_fallback_selection_v1" or promotion.get("status") != "completed":
        raise ValueError("DP32 promotion receipt is not completed")
    if identity.get("schema_version") != "apertus_full_8b_successor_stage_identity_v1" or identity.get("status") != "passed" or not all(identity.get("checks", {}).values()):
        raise ValueError("successor stage identity did not pass")
    def bound_json(binding: dict, label: str) -> dict:
        path = Path(binding.get("path", ""))
        if file_binding(path) != binding:
            raise ValueError(f"{label} binding drift")
        return read_json(path)

    old_recipe = bound_json(old.get("recipe", {}), "old selected recipe")
    old_profiles = bound_json(old.get("profiles", {}), "old selected profiles")
    promotion_recipe = promotion.get("inputs", {}).get("recipe", {})
    promotion_profiles = promotion.get("inputs", {}).get("profiles", {})
    if promotion_recipe != old.get("recipe") or promotion_profiles != old.get("profiles"):
        raise ValueError("promotion receipt does not bind the source selected contracts")

    profile_id = old["selection"]["profile_id"]
    validated = validate(recipe, profiles, profile_id)
    digest = scientific_digest(recipe)
    if digest != validated["scientific_digest"]:
        raise ValueError("derived recipe scientific digest drift")
    old_digest = old["selection"].get("scientific_digest")
    if old_digest != promotion.get("scientific_digest"):
        raise ValueError("source selected/promotion raw digest drift")
    source_identity = semantic_identity(old_recipe)
    successor_identity = semantic_identity(recipe)
    if source_identity != successor_identity:
        raise ValueError("successor semantic identity differs from proven DP32 evidence")
    if normalized_recipe_payload(old_recipe) != normalized_recipe_payload(recipe):
        raise ValueError("recipe changed outside the receipt-only normalization")
    if normalized_profiles(old_profiles) != normalized_profiles(profiles):
        raise ValueError("profiles changed outside derivation bindings")
    new_geometry = profiles["profiles"][profile_id]
    selection = old["selection"]
    expected = {
        "profile_id": profile_id,
        "nodes": int(new_geometry["nodes"]),
        "gpus_per_node": int(new_geometry["gpus_per_node"]),
        "world_size": int(new_geometry["world_size"]),
        "data_parallel": int(new_geometry["data_parallel"]),
        "gradient_accumulation_steps": int(new_geometry["gradient_accumulation_steps"]),
        "segment_boundaries": list(map(int, new_geometry["segment_boundaries"])),
        "scientific_digest": digest,
        "status": new_geometry["status"],
    }
    old_expected = {**expected, "scientific_digest": old_digest}
    if selection != old_expected:
        raise ValueError({"selected_geometry": selection, "source_geometry": old_expected})
    rebound = copy.deepcopy(old)
    rebound["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    rebound["recipe"] = file_binding(args.recipe)
    rebound["profiles"] = file_binding(args.profiles)
    rebound["selection"] = expected
    rebound["rebinding"] = {
        "schema_version": SCHEMA_VERSION,
        "policy": "receipt_only_successor_rebinding_with_semantic_identity_v1",
        "source_selected_profile": file_binding(args.old_selected),
        "promotion_receipt": file_binding(args.promotion_receipt),
        "stage_identity": file_binding(args.stage_identity),
        "source_recipe": file_binding(Path(old["recipe"]["path"])),
        "source_profiles": file_binding(Path(old["profiles"]["path"])),
        "source_raw_scientific_digest": old_digest,
        "successor_raw_scientific_digest": digest,
        "raw_scientific_digest_changed_only_because_receipt_bindings_changed": digest != old_digest,
        "semantic_identity": successor_identity,
        "source_semantic_identity": source_identity,
        "semantic_identity_matches": True,
        "normalized_recipe_changed_paths": changed_paths(
            normalized_recipe_payload(old_recipe), normalized_recipe_payload(recipe)
        ),
        "normalized_profiles_changed_paths": changed_paths(
            normalized_profiles(old_profiles), normalized_profiles(profiles)
        ),
        "training_rebenchmark_required": False,
    }
    atomic_write_json(args.output, rebound)
    print(json.dumps({"ok": True, "profile": profile_id, "scientific_digest": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
