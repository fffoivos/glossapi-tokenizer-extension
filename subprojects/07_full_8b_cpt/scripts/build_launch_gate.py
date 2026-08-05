#!/usr/bin/env python3
"""Build the sole authoritative full-8B production launch gate."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from contract import atomic_write_json, file_binding, read_json, scientific_digest, sha256_file
from validate_execution_profile import validate


def status(path: Path, schemas: set[str]) -> dict:
    value = read_json(path)
    if value.get("schema_version") not in schemas:
        raise ValueError(f"{path}: schema drift")
    if str(value.get("status", "")).lower() not in {"accepted", "completed", "frozen", "passed", "promoted"}:
        raise ValueError(f"{path}: non-passing status")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-bundle-receipt", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--selected-profile", type=Path, required=True)
    parser.add_argument("--owner-decisions", type=Path, required=True)
    parser.add_argument("--hf-visibility", type=Path, required=True)
    parser.add_argument("--pool-receipt", type=Path, required=True)
    parser.add_argument("--packed-receipt", type=Path, required=True)
    parser.add_argument("--schedule-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--token-bytes-receipt", type=Path, required=True)
    parser.add_argument("--benchmark-receipt", type=Path, required=True)
    parser.add_argument("--initial-validation", type=Path, required=True)
    parser.add_argument("--initial-greekmmlu", type=Path, required=True)
    parser.add_argument("--initial-per-document-root", type=Path, required=True)
    parser.add_argument("--conversion-smoke", type=Path, required=True)
    parser.add_argument("--launch-environment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recipe = read_json(args.recipe)
    selected = status(args.selected_profile, {"apertus_full_8b_selected_execution_profile_v1"})
    validated = validate(recipe, read_json(args.profiles), selected["selection"]["profile_id"])
    if validated["scientific_digest"] != selected["selection"]["scientific_digest"] or validated["scientific_digest"] != scientific_digest(recipe):
        raise ValueError("selected profile scientific digest drift")
    owner = status(args.owner_decisions, {"apertus_full_8b_owner_decisions_v1"})
    required_owner = (
        "D0_selection_confirmed_after_per_document_rerun_or_explicit_point_estimate_acceptance",
        "libduth_permission_evidence_conflict_is_reconciled_or_explicitly_accepted",
        "explicit_production_launch_authorization_is_received",
    )
    if any(owner.get("decisions", {}).get(name, {}).get("accepted") is not True for name in required_owner):
        raise ValueError("owner decision gate drift")
    visibility = status(args.hf_visibility, {"apertus_full_8b_hf_visibility_v1"})
    if visibility.get("public") is not True:
        raise ValueError("dataset v2 is not public")
    pool = status(args.pool_receipt, {"apertus_schedule_pool_corpus_v1"})
    packed = status(args.packed_receipt, {"apertus_packed_sequence_corpus_v1"})
    schedule = status(args.schedule_manifest, {"apertus_data_order_schedules_v1"})
    validation = read_json(args.validation_manifest)
    if validation.get("schema_version") != "apertus_full_8b_validation_manifest_v1" or validation.get("status") != "frozen" or len(validation.get("panels", [])) != 13:
        raise ValueError("validation manifest drift")
    status(args.token_bytes_receipt, {"apertus_token_utf8_byte_lengths_v1"})
    benchmark = status(args.benchmark_receipt, {"apertus_full_8b_parallelism_benchmark_v1"})
    if benchmark.get("selected_profile") != selected["selection"]["profile_id"]:
        raise ValueError("benchmark/selected profile drift")
    status(args.initial_validation, {"apertus_full_8b_initial_validation_v1"})
    status(args.initial_greekmmlu, {"apertus_full_8b_initial_greekmmlu_v1"})
    status(args.conversion_smoke, {"apertus_full_8b_conversion_greekmmlu_smoke_v1"})
    environment = status(args.launch_environment, {"apertus_full_8b_launch_environment_v1"})
    if environment.get("nodes") != selected["selection"]["nodes"]:
        raise ValueError("scheduler snapshot/profile node drift")
    per_document = sorted(args.initial_per_document_root.glob("*.receipt.json"))
    if len(per_document) != 13:
        raise ValueError("initial per-document receipt count drift")
    for path in per_document:
        value = status(path, {"apertus_per_document_validation_v1"})
        if int(value.get("aggregate", {}).get("documents", 0)) <= 0:
            raise ValueError(f"empty per-document receipt: {path}")
    production_init = Path(recipe["initialization"]["production_verification_path"])
    roundtrip_init = Path(recipe["initialization"]["roundtrip_verification_path"])
    if sha256_file(production_init) != recipe["initialization"]["production_verification_file_sha256"] or sha256_file(roundtrip_init) != recipe["initialization"]["roundtrip_verification_file_sha256"]:
        raise ValueError("Token Distillation initialization receipt drift")
    gates = {name: True for name in recipe["launch_gates"]}
    evidence = {
        name: file_binding(path)
        for name, path in {
            "code_bundle": args.code_bundle_receipt,
            "recipe": args.recipe,
            "profiles": args.profiles,
            "selected_profile": args.selected_profile,
            "owner_decisions": args.owner_decisions,
            "hf_visibility": args.hf_visibility,
            "pool": args.pool_receipt,
            "packed": args.packed_receipt,
            "schedule": args.schedule_manifest,
            "validation": args.validation_manifest,
            "token_bytes": args.token_bytes_receipt,
            "benchmark": args.benchmark_receipt,
            "initial_validation": args.initial_validation,
            "initial_greekmmlu": args.initial_greekmmlu,
            "conversion_smoke": args.conversion_smoke,
            "launch_environment": args.launch_environment,
            "production_initialization": production_init,
            "roundtrip_initialization": roundtrip_init,
        }.items()
    }
    evidence["initial_per_document"] = [file_binding(path) for path in per_document]
    payload = {
        "schema_version": "apertus_full_8b_launch_gate_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "recipe_id": recipe["recipe_id"],
        "scientific_digest": scientific_digest(recipe),
        "selected_profile": selected["selection"],
        "gates": gates,
        "evidence": evidence,
        "checkpoint_averaging": False,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"ok": True, "gates": len(gates), "selected_profile": selected["selection"]["profile_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
