#!/usr/bin/env python3
"""Freeze the unseen/capacity/tokenization/cache authority for Phase 3."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

from contract_utils import executing_code_bundle, file_binding, read_json, require, write_json_atomic
from freeze_phase_blend_cache import PHASE3_COMPONENT_REQUESTED_SAMPLES, validate_receipt as validate_phase_cache


POOL_CONTRACT = {
    "openarchives": ("phase3_openarchives", "active_modern", 1_577_226_077, 386_991),
    "foreign_replay": ("phase3_foreign", "foreign_replay", 399_297_741, 97_973),
    "old_greek_replay": ("phase3_old_greek", "old_greek_replay", 19_964_888, 4_899),
}


def validate_tokenized(path: Path, stream: str, unseen_binding: dict[str, Any]) -> dict[str, Any]:
    value = read_json(path)
    require(value.get("schema_version") == "apertus_hard_h_to_g_tokenized_stream_v1", f"{stream}: tokenized schema drift")
    require(value.get("status") == "frozen" and value.get("stream") == stream, f"{stream}: tokenized identity drift")
    require(value.get("input_receipt") == unseen_binding, f"{stream}: unseen-catalog binding drift")
    index = value.get("index")
    catalog = value.get("document_catalog")
    require(isinstance(index, dict) and isinstance(catalog, dict), f"{stream}: token accounting missing")
    require(
        int(index.get("tokens_including_eod", -1)) == int(catalog.get("tokens_including_eod", -2)),
        f"{stream}: exact Megatron/catalog token count drift",
    )
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unseen-catalog-receipt", type=Path, required=True)
    parser.add_argument("--openarchives-tokenized-receipt", type=Path, required=True)
    parser.add_argument("--foreign-tokenized-receipt", type=Path, required=True)
    parser.add_argument("--old-greek-tokenized-receipt", type=Path, required=True)
    parser.add_argument("--phase3-cache-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable Phase-3 authority exists: {args.output}")
    current_bundle = executing_code_bundle()
    unseen = read_json(args.unseen_catalog_receipt)
    require(unseen.get("schema_version") == "apertus_phase3_unseen_catalog_receipt_v1", "Phase-3 unseen-catalog schema drift")
    require(unseen.get("status") == "passed", "Phase-3 unseen-catalog did not pass")
    policy = unseen.get("policy")
    require(
        isinstance(policy, dict)
        and policy.get("separate_phase3_blend") is True
        and policy.get("cursor_at_update_3218") == 0
        and policy.get("phase2_key_or_text_overlap_allowed") is False
        and policy.get("within_phase3_key_or_text_repetition_allowed") is False
        and policy.get("gptdataset_epoch_wrap_allowed") is False
        and policy.get("component_capacity_covers_c92402e_1p005_builder_margin") is True,
        "Phase-3 unseen/capacity policy drift",
    )
    pools = unseen.get("pools")
    require(isinstance(pools, dict) and set(pools) == set(POOL_CONTRACT), "Phase-3 pool set drift")
    unseen_binding = file_binding(args.unseen_catalog_receipt)
    tokenized_paths = {
        "openarchives": args.openarchives_tokenized_receipt,
        "foreign_replay": args.foreign_tokenized_receipt,
        "old_greek_replay": args.old_greek_tokenized_receipt,
    }
    tokenized: dict[str, dict[str, Any]] = {}
    for pool, (stream, _role, target, requested_samples) in POOL_CONTRACT.items():
        row = pools[pool]
        require(isinstance(row, dict), f"{pool}: unseen pool receipt malformed")
        require(int(row.get("target_tokens", -1)) == target, f"{pool}: target token drift")
        require(int(row.get("component_requested_samples_with_1p005_margin", -1)) == requested_samples, f"{pool}: component request drift")
        require(int(row.get("minimum_one_epoch_tokens", -1)) == requested_samples * 4096 + 1, f"{pool}: one-epoch minimum drift")
        require(int(row.get("required_capacity_tokens", -1)) >= requested_samples * 4096 + 1, f"{pool}: capacity rule drift")
        output = row.get("output")
        require(isinstance(output, dict) and int(output.get("tokens", -1)) >= int(row["required_capacity_tokens"]), f"{pool}: unseen capacity is insufficient")
        tokenized[pool] = validate_tokenized(tokenized_paths[pool], stream, unseen_binding)
        require(tokenized[pool].get("input") == {key: output[key] for key in ("path", "bytes", "sha256")}, f"{pool}: tokenized input/output binding drift")

    cache = read_json(args.phase3_cache_receipt)
    data_path = Path(str(cache.get("data_path_spec", {}).get("path", "")))
    cache_root = Path(str(cache.get("cache_root", "")))
    validate_phase_cache(cache, phase=3, data_path_spec=data_path, cache_root=cache_root)
    spec = read_json(data_path)
    components = spec.get("components")
    require(isinstance(components, list) and len(components) == 3, "Phase-3 data-path component set drift")
    expected_by_role = {
        role: file_binding(tokenized_paths[pool])
        for pool, (_stream, role, _target, _samples) in POOL_CONTRACT.items()
    }
    require(
        {str(row["role"]): row.get("tokenized_receipt") for row in components} == expected_by_role,
        "Phase-3 data-path/tokenized receipt binding drift",
    )
    require(cache.get("phase3_component_requested_samples") == PHASE3_COMPONENT_REQUESTED_SAMPLES, "Phase-3 cache component request drift")
    require(cache.get("phase3_component_built_samples") == PHASE3_COMPONENT_REQUESTED_SAMPLES, "Phase-3 cache component build drift")
    payload = {
        "schema_version": "apertus_hard_h_to_g_phase3_authority_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "executing_code_bundle": current_bundle,
        "unseen_catalog_receipt": unseen_binding,
        "tokenized_receipts": {pool: file_binding(path) for pool, path in tokenized_paths.items()},
        "phase3_data_path_spec": file_binding(data_path),
        "phase3_cache_receipt": file_binding(args.phase3_cache_receipt),
        "component_requested_samples": PHASE3_COMPONENT_REQUESTED_SAMPLES,
        "phase_local_cursor_at_entry": 0,
        "phase_local_cursor_at_update_3456": 243_712,
        "phase_local_cursor_at_update_3694": 487_424,
        "invariants": {
            "all_main_trajectory_documents_are_forbidden": True,
            "component_capacity_covers_exact_1p005_margin": True,
            "each_megatron_token_count_matches_its_document_catalog": True,
            "gptdataset_cache_has_one_epoch_and_no_document_wrap": True,
            "phase3_uses_a_separate_cursor_zero_cache": True,
        },
    }
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
