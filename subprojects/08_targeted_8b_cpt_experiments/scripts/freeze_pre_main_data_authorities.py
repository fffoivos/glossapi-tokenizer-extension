#!/usr/bin/env python3
"""Freeze four strict pre-main data authorities from the exact receipt chain."""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from contract_utils import executing_code_bundle, file_binding, read_json, require, write_json_atomic
from freeze_phase_blend_cache import validate_receipt as validate_phase_cache
from producer_bundle_compatibility import load_authority, require_accepted_producer


PASS = {"passed", "frozen"}


def receipt(path: Path, schema: str, label: str) -> dict[str, Any]:
    value = read_json(path)
    require(value.get("schema_version") == schema, f"{label}: schema drift")
    require(value.get("status") in PASS, f"{label}: receipt did not pass")
    return value


def binding_without_counts(value: dict[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in ("path", "bytes", "sha256")}


def byte_identity(value: dict[str, Any]) -> dict[str, Any]:
    """Return the fields that establish byte equality independent of location."""

    return {key: value[key] for key in ("bytes", "sha256")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-receipt", type=Path, required=True)
    parser.add_argument("--native-query-authority", type=Path, required=True)
    parser.add_argument("--source-views", type=Path, required=True)
    parser.add_argument("--hplt-selected", type=Path, required=True)
    parser.add_argument("--openarchives-selected", type=Path, required=True)
    parser.add_argument("--hplt-prepared", type=Path, required=True)
    parser.add_argument("--openarchives-prepared", type=Path, required=True)
    parser.add_argument("--hplt-stage-b", type=Path, required=True)
    parser.add_argument("--openarchives-stage-b", type=Path, required=True)
    parser.add_argument("--hplt-post-greekmmlu", type=Path, required=True)
    parser.add_argument("--openarchives-post-greekmmlu", type=Path, required=True)
    parser.add_argument("--replay-selected", type=Path, required=True)
    parser.add_argument("--replay-validation-filter", type=Path, required=True)
    parser.add_argument("--replay-scan-input", type=Path, required=True)
    parser.add_argument("--replay-native-filter", type=Path, required=True)
    parser.add_argument("--replay-greekmmlu-filter", type=Path, required=True)
    parser.add_argument("--replay-stage-b", type=Path, required=True)
    parser.add_argument("--replay-post-greekmmlu", type=Path, required=True)
    parser.add_argument("--replay-post-native", type=Path, required=True)
    parser.add_argument("--replay-split", type=Path, required=True)
    parser.add_argument("--validation-panels", type=Path, required=True)
    parser.add_argument("--hplt-tokenized", type=Path, required=True)
    parser.add_argument("--openarchives-tokenized", type=Path, required=True)
    parser.add_argument("--foreign-tokenized", type=Path, required=True)
    parser.add_argument("--old-greek-tokenized", type=Path, required=True)
    parser.add_argument("--phase1-cache", type=Path, required=True)
    parser.add_argument("--phase2-cache", type=Path, required=True)
    parser.add_argument("--producer-compatibility", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_dir.exists(), f"immutable authority directory exists: {args.output_dir}")
    current = executing_code_bundle()
    compatibility, accepted_producers = load_authority(args.producer_compatibility, current)

    query = receipt(args.query_receipt, "apertus_frozen_greekmmlu_queries_receipt_v1", "GreekMMLU queries")
    native_queries = receipt(args.native_query_authority, "apertus_native_suite_scan_authority_v1", "native query authority")
    source_views = receipt(args.source_views, "apertus_hard_h_to_g_source_views_v1", "source views")
    for label, value in (("GreekMMLU queries", query), ("native query authority", native_queries), ("source views", source_views)):
        require_accepted_producer(value, accepted_producers, label)
    invariants = source_views.get("invariants")
    require(
        isinstance(invariants, dict)
        and invariants.get("additional_global_deduplication_performed") is False
        and invariants.get("near_deduplication_performed") is False
        and invariants.get("published_native_exclusions_applied_before_e001") is True
        and invariants.get("reused_validation_panel_exact_text_exclusions_applied") is True,
        "source-view sanitation/exclusion policy drift",
    )

    selected = {
        "hplt": receipt(args.hplt_selected, "apertus_hard_h_to_g_modern_mix_v1", "HPLT selection"),
        "openarchives": receipt(args.openarchives_selected, "apertus_hard_h_to_g_modern_mix_v1", "OpenArchives selection"),
    }
    prepared = {
        "hplt": receipt(args.hplt_prepared, "apertus_hard_h_to_g_prepared_greek_stream_v1", "HPLT prepared"),
        "openarchives": receipt(args.openarchives_prepared, "apertus_hard_h_to_g_prepared_greek_stream_v1", "OpenArchives prepared"),
    }
    stage_b = {
        "hplt": receipt(args.hplt_stage_b, "apertus_hard_h_to_g_stage_b_stream_v1", "HPLT Stage B"),
        "openarchives": receipt(args.openarchives_stage_b, "apertus_hard_h_to_g_stage_b_stream_v1", "OpenArchives Stage B"),
    }
    selected_paths = {"hplt": args.hplt_selected, "openarchives": args.openarchives_selected}
    prepared_paths = {"hplt": args.hplt_prepared, "openarchives": args.openarchives_prepared}
    stage_paths = {"hplt": args.hplt_stage_b, "openarchives": args.openarchives_stage_b}
    target_tokens = {"hplt": 8_500_000_000, "openarchives": 3_700_000_000}
    for pool in ("hplt", "openarchives"):
        require_accepted_producer(selected[pool], accepted_producers, f"{pool} selection")
        require_accepted_producer(prepared[pool], accepted_producers, f"{pool} prepared")
        require_accepted_producer(stage_b[pool], accepted_producers, f"{pool} Stage B")
        require(selected[pool].get("pool") == pool and int(selected[pool].get("target_tokens", -1)) == target_tokens[pool], f"{pool}: historical selection geometry drift")
        recipe_binding = selected[pool].get("recipe_receipt")
        require(isinstance(recipe_binding, dict), f"{pool}: modern recipe binding missing")
        recipe_path = Path(str(recipe_binding.get("path", "")))
        require(recipe_path.is_file() and recipe_binding == file_binding(recipe_path), f"{pool}: modern recipe receipt binding drift")
        recipe_value = receipt(recipe_path, "apertus_hard_h_to_g_modern_mix_recipes_v1", f"{pool} modern recipes")
        require(recipe_value.get("source_view_receipt") == file_binding(args.source_views), f"{pool}: source-view lineage drift")
        require(prepared[pool].get("pool") == pool and prepared[pool].get("selected_mix_receipt") == file_binding(selected_paths[pool]), f"{pool}: prepared-selection lineage drift")
        require(prepared[pool].get("query_receipt") == file_binding(args.query_receipt), f"{pool}: query lineage drift")
        require(stage_b[pool].get("stream") == pool and stage_b[pool].get("mode") == "assert_noop", f"{pool}: v2 Stage-B no-op contract drift")
        require(stage_b[pool].get("input_receipt") == file_binding(prepared_paths[pool]), f"{pool}: Stage-B input receipt drift")
        require(stage_b[pool].get("input") == binding_without_counts(prepared[pool]["clean"]), f"{pool}: Stage-B input bytes drift")
        require(byte_identity(stage_b[pool]["output"]) == byte_identity(stage_b[pool]["input"]), f"{pool}: Stage-B was not byte-noop")
        require(
            int(stage_b[pool].get("counts", {}).get("changed_rows", -1)) == 0
            and int(stage_b[pool].get("counts", {}).get("input_rows", -1))
            == int(stage_b[pool].get("output", {}).get("rows", -2))
            and stage_b[pool].get("invariants", {}).get("asserted_byte_noop") is True,
            f"{pool}: Stage-B no-op row/invariant proof drift",
        )

    hplt_post = receipt(args.hplt_post_greekmmlu, "apertus_fresh_greekmmlu_stream_scan_v1", "HPLT post scan")
    oa_post = receipt(args.openarchives_post_greekmmlu, "apertus_fresh_greekmmlu_stream_scan_v1", "OpenArchives post scan")
    for pool, value, path in (("hplt", hplt_post, args.hplt_stage_b), ("openarchives", oa_post, args.openarchives_stage_b)):
        require(value.get("stream") == f"{pool}_post" and value.get("audit_only") is True, f"{pool}: post-scan identity drift")
        require(int(value.get("counts", {}).get("item_doc_pairs", -1)) == 0, f"{pool}: post-Stage-B GreekMMLU overlap")
        require(value.get("input") == binding_without_counts(stage_b[pool]["output"]), f"{pool}: post-scan did not audit exact Stage-B bytes")
        require_accepted_producer(value, accepted_producers, f"{pool} post scan")

    replay_selected = receipt(args.replay_selected, "apertus_hard_h_to_g_replay_mix_v1", "replay selection")
    replay_validation = receipt(args.replay_validation_filter, "apertus_replay_validation_exclusion_v1", "replay heldout filter")
    replay_scan_input = receipt(args.replay_scan_input, "apertus_replay_benchmark_scan_input_receipt_v1", "replay benchmark adapter")
    replay_native = receipt(args.replay_native_filter, "apertus_replay_native_suite_filter_v1", "replay native filter")
    replay_greek = receipt(args.replay_greekmmlu_filter, "apertus_fresh_greekmmlu_stream_scan_v1", "replay GreekMMLU filter")
    replay_stage = receipt(args.replay_stage_b, "apertus_hard_h_to_g_stage_b_stream_v1", "replay Stage B")
    replay_post_greek = receipt(args.replay_post_greekmmlu, "apertus_fresh_greekmmlu_stream_scan_v1", "replay post GreekMMLU")
    replay_post_native = receipt(args.replay_post_native, "apertus_native_suite_training_scan_exclusions_v1", "replay post native")
    replay_split = receipt(args.replay_split, "apertus_hard_h_to_g_replay_split_v1", "replay split")
    for label, value in (
        ("replay selection", replay_selected), ("replay heldout filter", replay_validation),
        ("replay benchmark adapter", replay_scan_input), ("replay native filter", replay_native),
        ("replay GreekMMLU filter", replay_greek), ("replay Stage B", replay_stage),
        ("replay post GreekMMLU", replay_post_greek), ("replay post native", replay_post_native),
        ("replay split", replay_split),
    ):
        require_accepted_producer(value, accepted_producers, label)
    require(replay_validation.get("input_manifest") == file_binding(args.replay_selected), "replay heldout filter lineage drift")
    require(replay_validation.get("validation_receipt") == file_binding(args.validation_panels), "replay heldout-panel binding drift")
    require(replay_scan_input.get("source_level_disjointness_escape_allowed") is False, "replay source-level scan escape enabled")
    adapter_binding = replay_scan_input.get("adapter_config")
    require(isinstance(adapter_binding, dict), "replay adapter-config binding missing")
    adapter_path = Path(str(adapter_binding.get("path", "")))
    require(adapter_path.is_file() and adapter_binding == file_binding(adapter_path), "replay adapter-config binding drift")
    adapter = receipt(adapter_path, "apertus_replay_scan_adapter_v1", "replay adapter config")
    require_accepted_producer(adapter, accepted_producers, "replay adapter config")
    require(adapter.get("mix_manifest") == file_binding(args.replay_validation_filter), "replay adapter did not consume the heldout-clean manifest")
    require(adapter.get("selected_replay") == replay_validation.get("output_binding"), "replay adapter did not consume heldout-clean bytes")
    require(replay_native.get("input_receipt") == file_binding(args.replay_scan_input), "replay native-filter adapter lineage drift")
    require(
        replay_greek.get("stream") == "replay_selected"
        and replay_greek.get("input") == binding_without_counts(replay_native["output"]),
        "replay GreekMMLU/native filter ordering drift",
    )
    require(replay_stage.get("stream") == "replay_selected" and replay_stage.get("mode") == "apply", "replay Stage-B mode drift")
    require(replay_stage.get("input_receipt") == file_binding(args.replay_greekmmlu_filter), "replay Stage-B did not follow both benchmark filters")
    require(replay_stage.get("input") == binding_without_counts(replay_greek["clean"]), "replay Stage-B did not consume exact GreekMMLU-clean bytes")
    require(replay_post_greek.get("stream") == "replay_selected_post" and replay_post_greek.get("audit_only") is True, "replay post GreekMMLU identity drift")
    require(int(replay_post_greek.get("counts", {}).get("item_doc_pairs", -1)) == 0, "replay post-Stage-B GreekMMLU overlap")
    require(replay_post_greek.get("input") == binding_without_counts(replay_stage["output"]), "replay post GreekMMLU bytes drift")
    require(int(replay_post_native.get("counts", {}).get("strong_match_rows", -1)) == 0, "replay post-Stage-B native-suite overlap")
    require(int(replay_post_native.get("exclusions", {}).get("rows", -1)) == 0, "replay post-Stage-B native exclusion set is not empty")
    require(replay_split.get("input_receipt") == file_binding(args.replay_stage_b), "replay split/Stage-B lineage drift")
    require(replay_split.get("post_greekmmlu_receipt") == file_binding(args.replay_post_greekmmlu), "replay split GreekMMLU audit drift")
    require(replay_split.get("post_native_scan_receipt") == file_binding(args.replay_post_native), "replay split native audit drift")

    validation = receipt(args.validation_panels, "apertus_hard_h_to_g_reused_validation_panels_v1", "validation panels")
    require_accepted_producer(validation, accepted_producers, "validation panels")
    require(int(validation.get("counts", {}).get("panels", -1)) == 13, "validation panel count drift")

    tokenized_paths = {
        "hplt": args.hplt_tokenized,
        "openarchives": args.openarchives_tokenized,
        "foreign": args.foreign_tokenized,
        "old_greek": args.old_greek_tokenized,
    }
    tokenized = {
        stream: receipt(path, "apertus_hard_h_to_g_tokenized_stream_v1", f"{stream} tokenized")
        for stream, path in tokenized_paths.items()
    }
    upstream_bindings = {
        "hplt": file_binding(args.hplt_stage_b),
        "openarchives": file_binding(args.openarchives_stage_b),
        "foreign": file_binding(args.replay_split),
        "old_greek": file_binding(args.replay_split),
    }
    expected_inputs = {
        "hplt": binding_without_counts(stage_b["hplt"]["output"]),
        "openarchives": binding_without_counts(stage_b["openarchives"]["output"]),
        "foreign": binding_without_counts(replay_split["outputs"]["foreign"]),
        "old_greek": binding_without_counts(replay_split["outputs"]["old_greek"]),
    }
    for stream, value in tokenized.items():
        require_accepted_producer(value, accepted_producers, f"{stream} tokenized")
        require(value.get("stream") == stream and value.get("input_receipt") == upstream_bindings[stream], f"{stream}: tokenized lineage drift")
        require(value.get("input") == expected_inputs[stream], f"{stream}: tokenized input-byte drift")
        require(value.get("invariants", {}).get("additional_deduplication") is False, f"{stream}: tokenization deduplication drift")

    phase_caches = {}
    accepted_cache_code_bundles = {(root, tree) for root, tree, *_ in accepted_producers}
    for phase, path in ((1, args.phase1_cache), (2, args.phase2_cache)):
        value = read_json(path)
        data_path = Path(str(value.get("data_path_spec", {}).get("path", "")))
        cache_root = Path(str(value.get("cache_root", "")))
        validate_phase_cache(
            value,
            phase=phase,
            data_path_spec=data_path,
            cache_root=cache_root,
            accepted_code_bundles=accepted_cache_code_bundles,
        )
        require_accepted_producer(value, accepted_producers, f"Phase-{phase} blend cache")
        phase_caches[phase] = value
    expected_active = {1: file_binding(args.hplt_tokenized), 2: file_binding(args.openarchives_tokenized)}
    expected_foreign = file_binding(args.foreign_tokenized)
    expected_old = file_binding(args.old_greek_tokenized)
    for phase, value in phase_caches.items():
        spec = read_json(Path(str(value["data_path_spec"]["path"])))
        by_role = {row["role"]: row["tokenized_receipt"] for row in spec["components"]}
        require(by_role == {
            "active_modern": expected_active[phase],
            "foreign_replay": expected_foreign,
            "old_greek_replay": expected_old,
        }, f"Phase-{phase} blend component lineage drift")

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", suffix=".partial", dir=args.output_dir.parent))
    try:
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        common = {
            "created_at": now,
            "executing_code_bundle": current,
            "producer_bundle_compatibility": file_binding(args.producer_compatibility),
        }
        benchmark_payload = {
            "schema_version": "apertus_hard_h_to_g_benchmark_union_authority_v1", "status": "passed", **common,
            "greekmmlu_queries": file_binding(args.query_receipt),
            "native_suite_query_authority": file_binding(args.native_query_authority),
            "source_view_native_exclusions": source_views["native_suite_exclusions"],
            "post_stage_b_scans": {
                "hplt_greekmmlu": file_binding(args.hplt_post_greekmmlu),
                "openarchives_greekmmlu": file_binding(args.openarchives_post_greekmmlu),
                "replay_greekmmlu": file_binding(args.replay_post_greekmmlu),
                "replay_native_suite": file_binding(args.replay_post_native),
            },
            "protipa": {"included": False, "reason": "owner_side_access_unavailable_at_contract_freeze"},
            "invariants": {"all_selected_stage_b_streams_are_greekmmlu_clean": True, "all_selected_replay_stage_b_bytes_are_native_suite_clean": True, "no_source_level_disjointness_escape": True},
        }
        dataset_payload = {
            "schema_version": "apertus_hard_h_to_g_dataset_authority_v1", "status": "passed", **common,
            "source_views": file_binding(args.source_views),
            "modern_selections": {pool: file_binding(path) for pool, path in selected_paths.items()},
            "prepared_greek": {pool: file_binding(path) for pool, path in prepared_paths.items()},
            "stage_b": {"hplt": file_binding(args.hplt_stage_b), "openarchives": file_binding(args.openarchives_stage_b), "replay": file_binding(args.replay_stage_b)},
            "replay_chain": {
                "selection": file_binding(args.replay_selected), "heldout_filter": file_binding(args.replay_validation_filter),
                "heterogeneous_adapter": file_binding(args.replay_scan_input), "native_filter": file_binding(args.replay_native_filter),
                "greekmmlu_filter": file_binding(args.replay_greekmmlu_filter), "split": file_binding(args.replay_split),
            },
            "tokenized_streams": {stream: file_binding(path) for stream, path in tokenized_paths.items()},
            "invariants": {"historical_selection_geometry_rebuilt": True, "stage_order_is_exact": True, "v2_stage_b_is_byte_noop": True, "replay_stage_b_follows_both_benchmark_filters": True, "additional_deduplication": False},
        }
        heldout_payload = {
            "schema_version": "apertus_hard_h_to_g_heldout_overlap_authority_v1", "status": "passed", **common,
            "validation_panels": file_binding(args.validation_panels),
            "source_views": file_binding(args.source_views),
            "replay_validation_filter": file_binding(args.replay_validation_filter),
            "tokenized_streams": {stream: file_binding(path) for stream, path in tokenized_paths.items()},
            "invariants": {"all_13_panels_frozen": True, "exact_panel_text_excluded_from_both_modern_views": True, "exact_panel_text_excluded_from_replay_before_benchmark_filters": True, "tokenized_inputs_bind_the_exact_heldout_clean_stage_b_bytes": True, "additional_deduplication": False},
        }
        blend_payload = {
            "schema_version": "apertus_hard_h_to_g_blend_cache_authority_v1", "status": "passed", **common,
            "phase_cache_receipts": {"1": file_binding(args.phase1_cache), "2": file_binding(args.phase2_cache)},
            "tokenized_streams": {stream: file_binding(path) for stream, path in tokenized_paths.items()},
            "shared_by_scales": ["8b", "1p5b"],
            "invariants": {"phase_specific_active_modern_streams": True, "identical_replay_components": True, "data_seed_20260609": True, "randomized_gptdataset": True, "no_shuffle_patch_disabled": True, "phase_2_cache_has_global_historical_horizon": True},
        }
        for name, payload in (
            ("benchmark_union_authority.json", benchmark_payload),
            ("dataset_authority.json", dataset_payload),
            ("heldout_overlap_authority.json", heldout_payload),
            ("blend_cache_authority.json", blend_payload),
        ):
            write_json_atomic(temporary / name, payload)
        os.rename(temporary, args.output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
