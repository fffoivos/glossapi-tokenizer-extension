#!/usr/bin/env python3
"""Validate and freeze the matched hard-H-to-G experiment contract.

Planning mode is intentionally useful before remote artifacts exist: it emits
the exact unresolved roles. Launch mode is fail-closed and accepts only a
receipt-bound artifact manifest whose files still exist and retain their
recorded hashes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

from contract_utils import executing_code_bundle, file_binding, read_json, require, sha256_file, write_json_atomic
from producer_bundle_compatibility import load_authority, require_accepted_producer


PASSING = {
    "accepted",
    "completed",
    "frozen",
    "launch_ready",
    "passed",
    "promoted",
    # LR selection receipts describe a completed decision with ``selected``.
    # Schema and role semantics are still checked below, so accepting this
    # terminal status does not weaken the scientific identity.
    "selected",
}
GLOBAL_BATCH_TOKEN_SLOTS = 4_194_304
DATA_PIPELINE = [
    "freeze_corrected_full8_validation_panels_and_exact_text_exclusions",
    "inventory_exact_source_labels_and_freeze_pristine_selected_rows",
    "apply_published_native_suite_exclusions_with_raw_text_hash_verification",
    "apply_reused_validation_panel_exact_text_exclusions_to_every_training_stream",
    "historical_8p5b_hplt_and_3p7b_openarchives_mix_selection_over_benchmark_clean_views",
    "historical_5b_replay_mix_selection_over_validation_clean_source_recipes",
    "e001_character_clean_on_selected_greek_streams",
    "regenerate_greekmmlu_queries_and_filter_selected_greek_streams",
    "normalize_selected_replay_then_filter_native_suite_then_filter_greekmmlu",
    "accept_published_v2_anonymization_receipt_without_reaudit_and_apply_stage_b_only_to_twice_filtered_replay",
    "audit_exact_replay_stage_b_bytes_for_zero_greekmmlu_and_native_suite_matches",
    "split_final_replay_into_foreign_and_old_greek",
    "tokenize_four_stage_b_streams_with_receipt_bound_historical_148480_megatron_preprocessing",
    "freeze_phase_1_and_phase_2_weighted_blends_and_gptdataset_index_caches",
    "enumerate_phase_2_realized_document_ids",
    "build_separate_phase_3_unseen_blend_and_index_cache",
]
CHECKPOINT_UPDATES = [
    0, 238, 476, 714, 952, 1190, 1428, 1666, 1904, 2142, 2261, 2380,
    2618, 2856, 3094, 3218, 3456, 3694,
]
PRE_MAIN_SHARED_ARTIFACTS = [
    "historical_asset_inventory",
    "benchmark_union_and_exclusions",
    "dataset_rebuild_and_lineage",
    "heldout_overlap_audit",
    "replay_split",
    "weighted_blend_and_index_caches",
    "training_megatron_runtime",
    "tokenizer_148480",
    "offline_validation_panels",
    "historical_online_validation_binaries",
    "sentinel_freeze",
    "evaluation_code_bundle",
    "legacy_public_evaluator_contract",
    "statistical_decision_contract",
]
PRE_MAIN_ARTIFACTS_BY_SCALE = {
    "8b": [
        "8b_init_roundtrip",
        "8b_restart_parity",
        "8b_training_run_permit",
    ],
    "1p5b": [
        "1p5b_td_init",
        "1p5b_init_roundtrip",
        "1p5b_profile_candidate",
        "1p5b_lr_selection",
        "1p5b_training_run_permit",
    ],
}
PRE_MAIN_TERMINAL_ARTIFACTS = [
    "production_timing",
    "allocation_schedule",
    "owner_production_authorization",
]
# Compatibility inventory for schema and role-map completeness.  A launch gate
# consumes only the shared roles, the selected scale's roles, and the terminal
# roles; it never requires the other scale's initialization or runtime gates.
PRE_MAIN_ARTIFACTS = list(dict.fromkeys(
    PRE_MAIN_SHARED_ARTIFACTS
    + PRE_MAIN_ARTIFACTS_BY_SCALE["8b"]
    + PRE_MAIN_ARTIFACTS_BY_SCALE["1p5b"]
    + PRE_MAIN_TERMINAL_ARTIFACTS
))
PRE_EXTENSION_ARTIFACTS = [
    "main_8b_update_3218_checkpoint_permit",
    "main_1p5b_update_3218_checkpoint_permit",
    "cross_scale_realized_sample_ledger_match_through_3218",
    "same_stack_sentinel_calibration_state",
    "phase_3_unseen_blend_and_capacity_receipt",
    "phase_local_cursor_guard_receipt",
    "constant_floor_scheduler_resume_receipt",
    "owner_extension_authorization",
]
PRE_SECOND_EXTENSION_ARTIFACTS = [
    "both_update_3456_checkpoint_permits",
    "phase_3_3456_to_3457_resume_receipts",
]
PRE_FINALIZATION_ARTIFACTS = [
    "all_required_source_panel_receipts",
    "all_required_greekmmlu_receipts",
    "all_required_native_suite_receipts",
    "selection_authorization_receipt",
]
ARTIFACTS_BY_STAGE = {
    "pre_main": PRE_MAIN_ARTIFACTS,
    "pre_extension": PRE_EXTENSION_ARTIFACTS,
    "pre_second_extension": PRE_SECOND_EXTENSION_ARTIFACTS,
    "pre_finalization": PRE_FINALIZATION_ARTIFACTS,
}
ALL_ARTIFACTS = sorted({role for roles in ARTIFACTS_BY_STAGE.values() for role in roles})
# Compatibility alias for callers that mean the main launch gate.
REQUIRED_ARTIFACTS = PRE_MAIN_ARTIFACTS


def artifacts_for_stage(gate_stage: str, scale: str | None = None) -> list[str]:
    require(gate_stage in ARTIFACTS_BY_STAGE, f"unknown gate stage: {gate_stage}")
    if gate_stage == "pre_main":
        require(scale in PRE_MAIN_ARTIFACTS_BY_SCALE, "pre-main gate requires an exact model scale")
        return (
            PRE_MAIN_SHARED_ARTIFACTS
            + PRE_MAIN_ARTIFACTS_BY_SCALE[str(scale)]
            + PRE_MAIN_TERMINAL_ARTIFACTS
        )
    require(scale is None, f"{gate_stage} is a joint matched-study gate and does not accept --scale")
    return ARTIFACTS_BY_STAGE[gate_stage]

# A manifest role is not a user-defined label.  Each role accepts only the
# fixed receipt schema below; otherwise a single unrelated passing receipt
# could be copied into every row and turn the launch gate into a rubber stamp.
ROLE_SCHEMAS = {
    "historical_asset_inventory": "apertus_hard_h_to_g_asset_inventory_v1",
    "benchmark_union_and_exclusions": "apertus_hard_h_to_g_benchmark_union_authority_v1",
    "dataset_rebuild_and_lineage": "apertus_hard_h_to_g_dataset_authority_v1",
    "heldout_overlap_audit": "apertus_hard_h_to_g_heldout_overlap_authority_v1",
    "replay_split": "apertus_hard_h_to_g_replay_split_v1",
    "weighted_blend_and_index_caches": "apertus_hard_h_to_g_blend_cache_authority_v1",
    "training_megatron_runtime": "apertus_targeted_training_megatron_v1",
    "tokenizer_148480": "apertus_historical_tokenizer_148480_v1",
    "8b_init_roundtrip": "apertus_targeted_init_roundtrip_v1",
    "1p5b_td_init": "apertus_1p5b_td_initialization_verification_v2",
    "1p5b_init_roundtrip": "apertus_targeted_init_roundtrip_v1",
    "offline_validation_panels": "apertus_hard_h_to_g_reused_validation_panels_v1",
    "historical_online_validation_binaries": "apertus_hard_h_to_g_online_validation_v1",
    "sentinel_freeze": "apertus_greekmmlu_sentinel_manifest_v1",
    "evaluation_code_bundle": "apertus_mini_immutable_code_bundle_v1",
    "legacy_public_evaluator_contract": "apertus_legacy_public_greekmmlu_receipt_v1",
    "statistical_decision_contract": "apertus_hard_h_to_g_statistical_decisions_v2",
    "8b_restart_parity": "apertus_hard_h_to_g_profile_promotion_v1",
    "1p5b_profile_candidate": "apertus_hard_h_to_g_prelaunch_benchmark_contract_v1",
    "1p5b_lr_selection": "apertus_hard_h_to_g_lr_selection_v1",
    "8b_training_run_permit": "apertus_hard_h_to_g_training_run_permit_v1",
    "1p5b_training_run_permit": "apertus_hard_h_to_g_training_run_permit_v1",
    "production_timing": "apertus_hard_h_to_g_production_timing_v1",
    "allocation_schedule": "apertus_hard_h_to_g_allocation_schedule_v1",
    "owner_production_authorization": "apertus_hard_h_to_g_owner_authorization_v1",
    "main_8b_update_3218_checkpoint_permit": "apertus_hard_h_to_g_checkpoint_permit_v2",
    "main_1p5b_update_3218_checkpoint_permit": "apertus_hard_h_to_g_checkpoint_permit_v2",
    "cross_scale_realized_sample_ledger_match_through_3218": "apertus_hard_h_to_g_cross_scale_ledger_match_v1",
    "same_stack_sentinel_calibration_state": "apertus_greekmmlu_sentinel_calibration_authority_v1",
    "phase_3_unseen_blend_and_capacity_receipt": "apertus_hard_h_to_g_phase3_authority_v1",
    "phase_local_cursor_guard_receipt": "apertus_hard_h_to_g_phase_cursor_authority_v1",
    "constant_floor_scheduler_resume_receipt": "apertus_hard_h_to_g_constant_floor_resume_authority_v1",
    "owner_extension_authorization": "apertus_hard_h_to_g_owner_authorization_v1",
    "both_update_3456_checkpoint_permits": "apertus_hard_h_to_g_checkpoint_pair_authority_v1",
    "phase_3_3456_to_3457_resume_receipts": "apertus_hard_h_to_g_resume_pair_authority_v1",
    "all_required_source_panel_receipts": "apertus_hard_h_to_g_source_panel_evaluation_authority_v1",
    "all_required_greekmmlu_receipts": "apertus_hard_h_to_g_greekmmlu_evaluation_authority_v1",
    "all_required_native_suite_receipts": "apertus_hard_h_to_g_native_suite_evaluation_authority_v1",
    "selection_authorization_receipt": "apertus_hard_h_to_g_selection_authorization_v1",
}
require(set(ROLE_SCHEMAS) == set(ALL_ARTIFACTS), "artifact role/schema map is incomplete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path)
    parser.add_argument("--producer-compatibility", type=Path)
    parser.add_argument("--mode", choices=("planning", "launch"), default="planning")
    parser.add_argument("--gate-stage", choices=tuple(ARTIFACTS_BY_STAGE), default="pre_main")
    parser.add_argument("--scale", choices=tuple(PRE_MAIN_ARTIFACTS_BY_SCALE))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate_experiment(value: dict[str, Any]) -> dict[str, Any]:
    require(
        value.get("schema_version") == "apertus_hard_h_to_g_replication_v2",
        "experiment schema drift",
    )
    target = value["historical_target"]
    require(target["training_code_revision"] == "c92402e39ef3c8e69ea378a59e79059dc14541f4", "training code drift")
    require(target["data_seed"] == 20260609, "data seed drift")
    require(target["curriculum_order_mode"] == "randomized", "sample order drift")
    require(target["megatron_gpt_dataset_no_shuffle"] == 0, "shuffle flag drift")
    require(target["historical_greekmmlu_query_jsonl_survives"] is False, "purged historical query status drift")
    require(target["historical_cluster_tokenizer_directory_is_empty"] is True, "historical tokenizer survival status drift")

    models = value["models"]
    require(models["8b"]["revision"] == "3162c99675aa588097cecd4a24b9aa1f712af477", "8B revision drift")
    require(models["1p5b"]["revision"] == "dbe8919b2f0389888bada6b3a19e81e0ef4286c1", "1.5B revision drift")
    require(models["8b"]["target_layer"] == 11, "8B TD layer drift")
    require(
        models["1p5b"]["target_layer"] == 6
        and models["1p5b"]["target_layer_hidden_state_index"] == 6,
        "1.5B TD layer/index drift",
    )
    require(not models["8b"]["tie_word_embeddings"] and not models["1p5b"]["tie_word_embeddings"], "embeddings became tied")

    tokenizer = value["tokenizer"]
    require(tokenizer["vocab_size"] == 148_480, "historical tokenizer vocab drift")
    require(tokenizer["make_vocab_size_divisible_by"] == 256, "tokenizer divisor drift")
    require(tokenizer["vocab_size"] % 256 == 0 and tokenizer["padding_tokens"] == 0, "tokenizer padding drift")
    require(tokenizer["tokenizer_json_sha256"] == "358ae3f29ac17c99769d6d437339e28657d5fcaed3486f8550feed3d6adfc394", "tokenizer hash drift")

    initialization = value["initialization"]
    td = initialization["token_distillation"]
    require(td["epochs"] == 1 and td["snippets_per_token"] == 25 and td["batch_size"] == 8 and td["learning_rate"] == 0.0001, "Token Distillation recipe drift")
    require(td["seed"] == 20260523 and td["dtype"] == "bfloat16", "Token Distillation execution drift")
    require(td["minimum_trained_token_fraction"] == 0.99 and td["preserve_base_rows_exact"] is True, "Token Distillation preservation gate drift")
    coverage = initialization["coverage_prepass"]
    require(coverage == {
        "target_extended_tokens": 2_000_000_000,
        "candidate_snippets_per_token": 100,
        "snippet_token_radius": 50,
        "seed": 20260523,
        "regenerated_text_is_named_difference": True,
    }, "TD coverage-prepass contract drift")
    norms = initialization["row_norm_gate"]
    require((norms["lower_percentile"], norms["upper_percentile"], norms["outward_padding_fraction"]) == (0.5, 99.5, 0.2), "TD row-norm band drift")
    require(norms["outward_padding_definition"] == "let_width_equal_p99p5_minus_p0p5_then_lower_equal_p0p5_minus_0p2_width_and_upper_equal_p99p5_plus_0p2_width", "TD row-norm padding definition drift")
    require(norms["freeze_before_1p5b_td"] is True, "TD row-norm contract is not pre-frozen")

    data = value["data"]
    require(data["source_dataset"]["revision"] == "987b8955fcd395c6219e39df9e64715457f69065", "dataset revision drift")
    require(data["source_dataset"]["already_anonymized"] is True, "anonymization status drift")
    require(data["source_dataset"]["additional_global_deduplication_allowed"] is False, "second dedup enabled")
    views = data["source_views"]
    require(views["hplt_source_dataset_exact"] == "HPLT/ell_Grek_ge8_no_mt_clean60", "HPLT source label drift")
    require(views["openarchives_source_dataset_regex"] == r"^openarchives\.gr", "OpenArchives source regex drift")
    require(views["population_change_from_historical_stream_is_named_reconstruction_difference"] is True, "population difference hidden")
    require(views["active_views_must_anti_join_greek_replay_natural_keys"] is True, "Greek replay anti-join disabled")
    overlap = data["native_suite_overlap"]
    require(overlap["application"] == "direct_exact_join_on_pristine_v2_rows_no_cluster_expansion", "overlap policy drift")
    require(overlap["row_text_hash_field"] == "document_text_sha256", "overlap hash field drift")
    require(overlap["row_text_hash_semantics"] == "sha256_raw_utf8_before_e001", "overlap hash semantics drift")
    require(overlap["hash_mismatch_action"] == "fail_build", "overlap hash mismatch became permissive")
    require(overlap["query_items"] == 80_446 and overlap["query_scored_examples"] == 83_970, "native-suite query inventory drift")
    require(overlap["frozen_examples_sha256"] == "51e1dc1565e44d891173a50b787bbe0a90916cf6e8d36c53b9c61106f65604df", "native-suite frozen examples drift")
    require(overlap["external_replay_post_filter_zero_scan_required"] is True, "native-suite replay residual scan disabled")
    greekmmlu = data["greekmmlu"]
    require(greekmmlu["exclusion_authority"] == "fresh_scan_of_every_rebuilt_training_stream", "stale GreekMMLU exclusion authority")
    require(greekmmlu["query_source"] == "regenerated_from_frozen_dataset_revision", "GreekMMLU query provenance drift")
    require((greekmmlu["k"], greekmmlu["max_gap_tokens"], greekmmlu["max_gap_tokens_short"]) == (8, 50, 5), "GreekMMLU scan geometry drift")
    replay_scan = data["replay_benchmark_scan"]
    require(replay_scan["all_selected_foreign_and_old_greek_rows_require_content_scan"] is True, "replay scan incomplete")
    require(replay_scan["source_level_disjointness_escape_allowed"] is False, "gameable replay scan escape enabled")
    require(replay_scan["scanner_adapter_role"] == "heterogeneous_replay_native_suite_and_greekmmlu_scanner", "replay adapter drift")
    require(replay_scan["run_on_partition"] == "debug", "replay scan routed off debug")
    require(replay_scan["native_suite_scanner_revision"] == "2a7eb9d8de342129f379575ec031f631cde304bc", "native-suite scanner revision drift")
    require(replay_scan["post_stage_b_zero_scans_required"] is True, "post-Stage-B replay scans disabled")
    cleaning = data["cleaning_and_anonymization"]
    require(cleaning["e001_implementation_sha256"] == "1997aa7a1248f9b946681e32bdcc98f33efc476fa28b53cc9bd8db5a3d883f63", "E001 implementation drift")
    require(cleaning["stage_b_masker_sha256"] == "8f489a175aeb47f2c0996431a9d1c6f93ec03d4f52d9ea33621b76facfc0e83c", "Stage-B masker drift")
    require(cleaning["one_pass_e001_plus_greekmmlu_composition_allowed"] is True and cleaning["one_pass_requires_exact_frozen_functions_and_equivalence_test"] is True, "one-pass composition contract drift")
    require(data["protipa"]["experiment_decision"] == "excluded_from_training_union_and_evaluation_before_manifest_freeze", "Protipa scope drift")
    replay_reconstruction = data["replay_reconstruction"]
    require(replay_reconstruction["acquisition_receipt_sha256"] == "9ea630cba9d70e60ad73626ee7d0671b9eb14d6a214be91efee5f4226437f6ba", "replay acquisition authority drift")
    require(replay_reconstruction["acquisition_policy"] == "all_matching_or_domain_separated_sha256_ranked_capacity_sample_v1", "replay acquisition policy drift")
    require(replay_reconstruction["acquisition_seed"] == 20260609 and replay_reconstruction["selected_files"] == 355, "replay acquisition selection drift")
    require(replay_reconstruction["preserve_historical_source_weights"] is True, "replay source weights may drift")
    require(replay_reconstruction["historical_document_identity_claimed"] is False, "reconstructed replay falsely claims historical document identity")
    require(bool(replay_reconstruction["named_reconstruction_difference"]), "replay reconstruction difference is unnamed")
    replay_mix = data["replay_mix_builder"]
    require(replay_mix["target_tokens"] == 5_000_000_000, "replay mix target drift")
    require(replay_mix["source_shard_count"] == 16 and replay_mix["target_tokens_per_shard"] == 312_500_000, "historical replay sharding drift")
    require(replay_mix["seed"] == 20260611, "historical replay mix seed drift")
    require(replay_mix["source_shard_rule"] == "eligible_source_row_index_modulo_16", "historical replay source-shard rule drift")
    require(replay_mix["concatenation_order"] == "source_shard_index_ascending", "historical replay concatenation drift")
    require(replay_mix["historical_algorithm_reproduced"] is True, "historical replay algorithm disabled")
    require(replay_mix["historical_document_identity_claimed"] is False, "reconstructed replay falsely claims historical selected rows")
    modern_mix = data["modern_mix_builder"]
    require(modern_mix["hplt_target_tokens"] == 8_500_000_000, "historical HPLT mix target drift")
    require(modern_mix["openarchives_target_tokens"] == 3_700_000_000, "historical OpenArchives mix target drift")
    require(modern_mix["source_shard_count"] == 16, "historical modern mix shard count drift")
    require(modern_mix["hplt_target_tokens_per_shard"] == 531_250_000, "historical HPLT shard target drift")
    require(modern_mix["openarchives_target_tokens_per_shard"] == 231_250_000, "historical OpenArchives shard target drift")
    require(modern_mix["seed"] == 20260611, "historical modern mix seed drift")
    require(modern_mix["source_shard_rule"] == "eligible_source_row_index_modulo_16", "historical modern mix source-shard rule drift")
    require(modern_mix["concatenation_order"] == "source_shard_index_ascending", "historical modern mix concatenation drift")
    require(modern_mix["historical_algorithm_reproduced"] is True, "historical modern mix algorithm disabled")
    require(modern_mix["lineage_extension_changes_selection"] is False, "modern mix lineage extension may change selection")
    require(modern_mix["selection_occurs_after_named_native_suite_validation_and_greek_replay_exclusions"] is True, "benchmark-clean modern selection ordering drift")
    require(modern_mix["historical_document_identity_claimed"] is False, "reconstructed modern mix falsely claims historical selected rows")
    require(bool(modern_mix["named_reconstruction_difference"]), "modern mix reconstruction difference is unnamed")
    tokenization = data["tokenization"]
    require(tokenization["schema"] == "apertus_hard_h_to_g_tokenized_stream_v1", "tokenization receipt schema drift")
    require(tokenization["tokenizer_json_sha256"] == "358ae3f29ac17c99769d6d437339e28657d5fcaed3486f8550feed3d6adfc394", "tokenization tokenizer drift")
    require(tokenization["megatron_commit"] == "c92402e39ef3c8e69ea378a59e79059dc14541f4", "tokenization Megatron drift")
    require(tokenization["tokenizer_type"] == "HuggingFaceTokenizer", "tokenization tokenizer type drift")
    require(tokenization["append_eod"] is True and tokenization["json_keys"] == ["text"], "tokenization text/EOD contract drift")
    require(tokenization["workers"] == 64, "tokenization worker geometry drift")
    require(tokenization["one_indexed_document_per_input_row"] is True, "tokenization document geometry drift")
    require(tokenization["streams"] == {
        "hplt": "hplt_only_ext_text_document",
        "openarchives": "glossapi_only_ext_text_document",
        "foreign": "foreign_replay_only_ext_text_document",
        "old_greek": "old_greek_replay_only_ext_text_document",
    }, "tokenization stream prefix drift")
    require(data["trainer_stack"] == "historical_megatron_weighted_blend", "trainer stack drift")
    require(data["explicit_schedule_reader_allowed"] is False, "explicit schedule reader enabled")
    require(data["pipeline"] == DATA_PIPELINE, "data pipeline ordering drift")
    require(data["blend_weights"] == {
        "active_modern": "1.0",
        "foreign_replay": "0.253164557",
        "old_greek_replay": "0.012658228",
    }, "replay blend drift")
    require(sum(data["historical_replay_split_rows"].values()) == data["historical_clean_document_counts_for_provenance_only"]["replay_before_split"], "historical replay row accounting drift")

    training = value["training"]
    require(training["sequence_length"] == 4096, "sequence length drift")
    require(training["global_batch_sequences"] == 1024, "global batch drift")
    require(training["global_batch_token_slots"] == GLOBAL_BATCH_TOKEN_SLOTS, "token batch drift")
    require(training["precision"] == "bf16" and training["main_gradients_dtype"] == "fp32", "training precision drift")
    require(training["optimizer"] == "AdEMAMix", "optimizer drift")
    require((training["beta1"], training["beta2"], training["beta3"], training["alpha"]) == ("0.9", "0.999", "0.999", "4.0"), "AdEMAMix drift")
    require(training["loss"] == "goldfish" and training["goldfish_k"] == training["goldfish_h"] == 50, "Goldfish drift")
    require(training["scheduler_nominal_tokens"] == 13_500_000_000, "scheduler anchor drift")
    require(training["scheduler_train_samples"] == 3_295_898, "scheduler sample anchor drift")
    require(training["scheduler_decay_samples"] == 659_179, "scheduler decay drift")
    require(training["extension_train_samples"] == 3_782_656, "extension train-sample horizon drift")
    require(training["extension_lr_schedule"] == "constant" and training["extension_lr_equals_nominal_min_lr"] is True, "extension LR schedule drift")
    require(training["extension_lr_warmup_samples"] == 0, "extension LR warmup drift")
    require(training["extension_alpha_warmup_updates"] == training["extension_beta3_warmup_updates"] == 3218, "extension optimizer ramp re-anchored")
    require(training["extension_override_opt_param_scheduler_expected"] is True, "extension scheduler override contract missing")
    require(training["peak_lr_8b"] == "5.5e-5" and training["terminal_lr_ratio"] == "0.1", "8B LR drift")
    require(
        training.get("peak_lr_by_scale") == {"8b": "5.5e-5", "1p5b": "5.5e-5"},
        "matched cross-scale LR drift",
    )
    require(training["rope_theta"] == 500_000 and training["max_position_embeddings"] == 4096, "RoPE geometry drift")
    require(training["cross_document_attention"] is False and training["attention_mask_reset_at_document_boundary"] is True, "historical attention-mask boundary behavior drift")
    require(training["eod_loss_masking"] is True and training["position_reset_at_document_boundary"] is True, "historical EOD/position boundary behavior drift")
    require(training["checkpoint_averaging"] is False, "checkpoint averaging enabled")
    runtime = training["runtime"]
    require(runtime["megatron_upstream_commit"] == target["training_code_revision"], "training Megatron revision drift")
    require(runtime["named_extra_validation_patch_sha256"] == "2e6810fa8b6c25597ccb3bcb9dc1ff5bf843ead2337e3edde0344605a23ec4c6", "named validation patch drift")
    require(runtime["exact_eval_iteration_patch_applied"] is False, "unapproved exact-evaluation patch enabled")
    require(runtime["trainer_scale_geometry_patch_required"] is True and runtime["trainer_and_runtime_guard_frozen_in_code_bundle"] is True, "scale-aware trainer freeze disabled")
    require(runtime["phase_cache_receipt_schema"] == "apertus_hard_h_to_g_phase_blend_cache_v1", "phase-cache schema drift")
    require(runtime["phase_data_path_schema"] == "apertus_hard_h_to_g_phase_data_path_v1", "phase data-path schema drift")
    require(runtime["explicit_data_cache_path_required"] is True, "explicit data-cache path disabled")
    require(runtime["strict_phase_local_guard"] == "scripts/phase_local_data_index_guard.py", "phase-local guard drift")
    require(runtime["training_run_permit_schema"] == "apertus_hard_h_to_g_training_run_permit_v1", "training run-permit schema drift")
    require(runtime["checkpoint_permit_schema"] == "apertus_hard_h_to_g_checkpoint_permit_v2", "checkpoint permit schema drift")

    schedule = value["schedule"]
    require(schedule["checkpoint_updates"] == CHECKPOINT_UPDATES, "checkpoint cadence drift")
    require(schedule["phase_1"]["optimizer_update_end"] == 2261, "phase-1 boundary drift")
    require(schedule["phase_2"]["optimizer_update_start"] == 2262 and schedule["phase_2"]["optimizer_update_end"] == 3218, "phase-2 boundary drift")
    require(schedule["phase_2"]["reset_data_index_on_entry"] is True, "historical boundary reset disabled")
    extension = schedule["extension"]
    require(extension["optimizer_update_start"] == 3219 and extension["optimizer_update_end"] == 3694, "extension horizon drift")
    require(extension["construction"] == "separate_frozen_weighted_blend_consumed_from_cursor_zero", "Phase-3 construction drift")
    require(extension["reset_data_index_on_entry"] is True, "Phase-3 cursor reset disabled")
    require(extension["phase_local_cursor_at_3218"] == 0, "3218 phase cursor drift")
    require(extension["phase_local_cursor_at_3456"] == (3456 - 3218) * 1024, "3456 phase cursor drift")
    require(extension["phase_local_cursor_at_3694"] == (3694 - 3218) * 1024, "3694 phase cursor drift")
    require(extension["gptdataset_requested_samples"] == 487_424, "Phase-3 dataset horizon drift")
    require(extension["component_requested_samples_with_1p005_margin"] == {
        "openarchives": 386_991,
        "foreign_replay": 97_973,
        "old_greek_replay": 4_899,
    }, "Phase-3 component construction margin drift")
    require(
        extension["capacity_rule"] == "each_phase3_component_has_at_least_requested_samples_times_4096_plus_one_token_in_one_epoch",
        "Phase-3 one-epoch capacity rule drift",
    )
    require(extension["scheduler_train_samples_remains"] == training["extension_train_samples"], "Phase-3 scheduler horizon drift")
    require(extension["main_trajectory_realized_documents_are_forbidden"] is True, "main-trajectory documents allowed into Phase 3")
    require(extension["within_phase_3_document_repetition_is_forbidden"] is True, "Phase-3 document repetition enabled")
    require(extension["exact_segment_exit_updates"] == [3456, 3694], "extension save-wall drift")

    evaluation = value["evaluation"]
    panel_authority = evaluation["offline_panel_authority"]
    require(panel_authority["manifest_sha256"] == "a4b1d696adf83b2c691a99565075ba0a70db4074f7fe91fe6fdbab094303c1d9", "offline panel manifest drift")
    require(panel_authority["panel_count"] == 13, "offline panel count drift")
    require(panel_authority["training_exclusion_semantics"] == "sha256_of_exact_stored_utf8_text", "offline panel exclusion policy drift")
    require(panel_authority["historical_h_to_g_loss_replication_claim_allowed"] is False, "new panels misrepresented as historical")
    native_code = evaluation["native_suite_code_authority"]
    require(native_code["branch"] == "agent/full8-results-analysis", "native-suite branch drift")
    require(native_code["revision"] == "2a7eb9d8de342129f379575ec031f631cde304bc", "native-suite code revision drift")
    sentinel = evaluation["greekmmlu_sentinel"]
    require(sentinel["sizes"] == [4096, 8192], "sentinel sizes drift")
    require(sentinel["early_calibration_updates"] == [0, 238, 476, 714], "sentinel early calibration drift")
    require(sentinel["late_resolution_updates"] == [2618, 2856, 3094, 3218], "sentinel late calibration drift")
    require(sentinel["bootstrap_replicates"] == 10_000 and sentinel["bootstrap_seed"] == 20260814, "sentinel bootstrap drift")
    require(sentinel["candidate_batch_size"] == 1, "candidate batch drift")
    require(sentinel["authorization_requires_both_early_and_late_tests"] is True, "sentinel authorization weakened")
    require(evaluation["legacy_public_evaluator"]["questions"] == 16_632, "legacy GreekMMLU panel drift")
    require(evaluation["legacy_public_evaluator"]["dtype"] == "bfloat16", "legacy evaluator dtype drift")

    require(
        value.get("learning_rate_1p5b") == {
            "mode": "fixed_matched_8b_recipe",
            "peak_lr": "5.5e-5",
            "floor_lr": "5.5e-6",
            "terminal_lr_ratio": "0.1",
            "source": "same_approved_scientific_recipe_as_8b",
            "lr_pilot_runs": 0,
            "benchmark_values_used_for_selection": False,
        },
        "1.5B fixed LR policy drift",
    )
    require(
        value.get("profile_selection") == {
            "minimum_updates": 256,
            "discard_warmup_updates": 32,
            "eligible_only_if_all_scientific_parity_checks_pass": True,
            "primary_metric": "tokens_per_gpu_hour",
            "near_tie_relative_fraction": 0.02,
            "near_tie_break": "lower_p90_step_seconds_then_fewer_nodes",
            "minimum_candidates": 2,
            "minimum_passing_candidates": 2,
            "all_three_candidates_preferred": True,
            "parity_thresholds": {
                "trajectory_loss_rmse_max": 0.05,
                "trajectory_loss_signed_mean_abs_max": 0.02,
                "trajectory_gradient_relative_median_max": 0.02,
                "restart_loss_abs_max": 1e-6,
                "restart_parameter_norm_abs_max": 1e-6,
                "restart_gradient_norm_atol": 0.001,
                "restart_gradient_norm_rtol": 0.02,
            },
        },
        "shared execution-profile selection rule drift",
    )
    require(
        value.get("profile_qualification_1p5b") == {
            "mode": "first_production_allocation_qualify_and_continue",
            "fixed_candidate_profile_id": "1p5b_tp1_1node",
            "candidate_comparison_runs": 0,
            "minimum_updates": 256,
            "discard_warmup_updates": 32,
            "production_continues_only_if_all_scientific_parity_checks_pass": True,
            "qualification_reuses_first_production_allocation": True,
            "parity_thresholds": {
                "trajectory_loss_rmse_max": 0.05,
                "trajectory_loss_signed_mean_abs_max": 0.02,
                "trajectory_gradient_relative_median_max": 0.02,
                "restart_loss_abs_max": 1e-6,
                "restart_parameter_norm_abs_max": 1e-6,
                "restart_gradient_norm_atol": 0.001,
                "restart_gradient_norm_rtol": 0.02,
            },
        },
        "1.5B execution-profile selection rule drift",
    )

    statistics = value["statistics"]
    require(statistics["replication_accuracy_margin"] == 0.015, "replication margin drift")
    require(statistics["trajectory_correlation"] == "spearman_on_adjacent_first_differences", "trajectory statistic drift")
    require(statistics["decision_states"] == ["pass", "fail", "inconclusive", "not_testable"], "decision states drift")
    launch = value["launch"]
    require(launch["required_pre_main_launch_artifacts"] == PRE_MAIN_ARTIFACTS, "pre-main compatibility artifact set drift")
    require(launch["pre_main_launch_scope"] == "independent_per_scale", "pre-main launch scope drift")
    require(launch["required_pre_main_shared_artifacts"] == PRE_MAIN_SHARED_ARTIFACTS, "pre-main shared artifact set drift")
    require(launch["required_pre_main_artifacts_by_scale"] == PRE_MAIN_ARTIFACTS_BY_SCALE, "pre-main scale artifact set drift")
    require(launch["required_pre_main_terminal_artifacts"] == PRE_MAIN_TERMINAL_ARTIFACTS, "pre-main terminal artifact set drift")
    require(
        launch["production_launch_authorized_by_scale"] == {"8b": False, "1p5b": False},
        "immutable experiment contract must not self-authorize production",
    )
    require(launch["required_pre_extension_artifacts"] == PRE_EXTENSION_ARTIFACTS, "pre-extension artifact set drift")
    require(launch["required_pre_second_extension_segment_artifacts"] == PRE_SECOND_EXTENSION_ARTIFACTS, "pre-second-extension artifact set drift")
    require(launch["required_pre_finalization_artifacts"] == PRE_FINALIZATION_ARTIFACTS, "pre-finalization artifact set drift")

    return {
        "checkpoint_token_slots": {
            str(update): update * GLOBAL_BATCH_TOKEN_SLOTS for update in CHECKPOINT_UPDATES
        },
        "realized_main_token_slots": 3218 * GLOBAL_BATCH_TOKEN_SLOTS,
        "realized_terminal_token_slots": 3694 * GLOBAL_BATCH_TOKEN_SLOTS,
        "extension_token_slots": (3694 - 3218) * GLOBAL_BATCH_TOKEN_SLOTS,
    }


def validate_allocation(value: dict[str, Any]) -> dict[str, Any]:
    require(value.get("schema_version") == "apertus_hard_h_to_g_allocation_v1", "allocation schema drift")
    require(value["allocation_limit_seconds"] == 43_200, "allocation limit drift")
    prep = value["preparation"]
    require(prep["partition"] == "debug" and prep["nodes"] == 1 and prep["maximum_wall_seconds"] <= 5400, "debug preparation drift")
    profile = value["profiles"]["8b"]
    require(profile["profile_id"] == "dp32_16node" and profile["nodes"] == 16, "8B profile drift")
    require(profile["tensor_parallel"] == 2 and profile["data_parallel"] == 32, "8B decomposition drift")
    require(profile["nodes"] * profile["gpus_per_node"] == profile["tensor_parallel"] * profile["data_parallel"], "8B world-size arithmetic drift")
    require(profile["ntasks_per_node"] == 4 and profile["gpus_per_task"] == 1 and profile["cpus_per_task"] == 72, "8B affinity drift")
    candidates = value["profiles"]["1p5b_candidates"]
    require([row["nodes"] for row in candidates] == [1, 2, 4], "1.5B candidate-node grid drift")
    for row in candidates:
        require(row["partition"] == "normal", "1.5B benchmark must use normal")
        require(row["nodes"] * row["gpus_per_node"] == row["tensor_parallel"] * row["data_parallel"], f"{row['profile_id']}: world-size arithmetic drift")
        require(row["microbatch"] == 8, f"{row['profile_id']}: microbatch drift")
        require(
            row["gradient_accumulation_microbatches"] == 1024 // (row["data_parallel"] * row["microbatch"]),
            f"{row['profile_id']}: gradient accumulation drift",
        )
    benchmark = value["profile_benchmark"]
    require(benchmark["minimum_updates"] >= 256 and benchmark["discard_warmup_updates"] >= 32, "profile benchmark too short")
    require(
        value.get("learning_rate_policy") == {
            "mode": "fixed_matched_8b_recipe",
            "allocations": 0,
            "peak_lr": "5.5e-5",
            "floor_lr": "5.5e-6",
            "terminal_lr_ratio": "0.1",
        },
        "allocation LR policy drift",
    )
    segmentation = value["segmentation"]
    require(segmentation["mode"] == "historical_segmented_default", "unproven single-allocation path selected")
    require(segmentation["at_most_one_pending_successor_total"] is True, "successor limit drift")
    require(segmentation["direct_normal_holder"] is True and segmentation["debug_timer_for_normal_holder"] is False, "holder routing drift")
    require(value["handoff_formula"]["values_status"] == "blocked_until_production_equivalent_measurements", "unmeasured handoff values were promoted")
    return {
        "8b_world_size": profile["nodes"] * profile["gpus_per_node"],
        "1p5b_candidate_world_sizes": {
            row["profile_id"]: row["nodes"] * row["gpus_per_node"] for row in candidates
        },
    }


def validate_code_bundle_receipt(receipt: dict[str, Any]) -> bool:
    """Full-hash an immutable scientific bundle described by its own receipt."""
    root = Path(str(receipt.get("root", ""))).resolve()
    rows = receipt.get("files")
    if (
        receipt.get("status") != "frozen"
        or receipt.get("kind") != "scientific"
        or not root.is_dir()
        or not isinstance(rows, list)
        or not rows
        or int(receipt.get("file_count", -1)) != len(rows)
    ):
        return False
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            return False
        relative = str(row.get("relative_path", ""))
        candidate = (root / relative).resolve()
        if (
            not relative
            or relative in seen
            or not candidate.is_file()
            or candidate.is_symlink()
            or root != candidate.parent and root not in candidate.parents
            or int(row.get("bytes", -1)) != candidate.stat().st_size
            or row.get("sha256") != sha256_file(candidate)
        ):
            return False
        seen.add(relative)
    canonical = json.dumps(rows, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest() == receipt.get("tree_sha256")


def role_semantics_match(
    role: str,
    receipt: dict[str, Any],
    *,
    expected_scale: str | None = None,
) -> bool:
    """Check role identities that cannot be expressed by receipt schema alone."""
    def binding_is_current(value: object) -> bool:
        if not isinstance(value, dict):
            return False
        path = Path(str(value.get("path", "")))
        return path.is_file() and value == file_binding(path)

    expected = ROLE_SCHEMAS[role]
    if receipt.get("schema_version") != expected:
        return False
    if role == "8b_init_roundtrip":
        return receipt.get("scale") == "8b"
    if role in {"1p5b_td_init", "1p5b_init_roundtrip"}:
        # The TD verification schema is 1.5B-specific and predates a scale key.
        return role == "1p5b_td_init" or receipt.get("scale") == "1p5b"
    if role == "8b_restart_parity":
        return (
            receipt.get("scale") == "8b"
            and receipt.get("selection", {}).get("profile_id") == "dp32_16node"
            and all(receipt.get("checks", {}).values())
            and receipt.get("checks", {}).get("phase2_entry_and_restart_parity") is True
        )
    if role == "1p5b_profile_candidate":
        return (
            receipt.get("scale") == "1p5b"
            and receipt.get("status") == "frozen"
            and receipt.get("kind") == "profile"
            and receipt.get("profile_id") == "1p5b_tp1_1node"
            and int(receipt.get("nodes", -1)) == 1
            and int(receipt.get("tensor_parallel", -1)) == 1
            and int(receipt.get("microbatch", -1)) == 8
            and int(receipt.get("updates", -1)) == 256
            and str(receipt.get("peak_lr")) == "5.5e-5"
            and str(receipt.get("floor_lr")) == "5.5e-6"
        )
    if role == "1p5b_lr_selection":
        return (
            receipt.get("scale") == "1p5b"
            and receipt.get("peak_lr") == "5.5e-5"
            and receipt.get("floor_lr") == "5.5e-6"
            and receipt.get("candidates") == ["5.5e-5"]
            and receipt.get("decision", {}).get("method") == "fixed_matched_8b_recipe"
            and receipt.get("decision", {}).get("lr_pilot_runs") == 0
            and all(receipt.get("checks", {}).values())
        )
    if role in {"8b_training_run_permit", "1p5b_training_run_permit"}:
        expected_scale = "8b" if role.startswith("8b") else "1p5b"
        profile = receipt.get("profile", {})
        lr = receipt.get("learning_rate", {})
        profile_matches = (
            profile.get("profile_id") == "dp32_16node"
            if expected_scale == "8b"
            else profile.get("profile_id") in {
                "1p5b_tp1_1node", "1p5b_tp1_2node", "1p5b_tp1_4node",
            }
        )
        return (
            receipt.get("scale") == expected_scale
            and profile.get("global_batch_sequences") == 1024
            and str(lr.get("terminal_ratio")) == "0.1"
            and profile_matches
            and str(lr.get("peak")) == "5.5e-5"
            and str(lr.get("floor")) == "5.5e-6"
        )
    if role == "main_8b_update_3218_checkpoint_permit":
        return receipt.get("scale") == "8b" and int(receipt.get("update", -1)) == 3218
    if role == "main_1p5b_update_3218_checkpoint_permit":
        return receipt.get("scale") == "1p5b" and int(receipt.get("update", -1)) == 3218
    if role == "owner_production_authorization":
        return (
            expected_scale in PRE_MAIN_ARTIFACTS_BY_SCALE
            and receipt.get("authorization_stage") == "pre_main"
            and receipt.get("scale") == expected_scale
        )
    if role == "owner_extension_authorization":
        return receipt.get("authorization_stage") == "pre_extension"
    if role == "benchmark_union_and_exclusions":
        invariants = receipt.get("invariants", {})
        return (
            receipt.get("protipa", {}).get("included") is False
            and invariants.get("all_selected_stage_b_streams_are_greekmmlu_clean") is True
            and invariants.get("all_selected_replay_stage_b_bytes_are_native_suite_clean") is True
            and invariants.get("no_source_level_disjointness_escape") is True
        )
    if role == "legacy_public_evaluator_contract":
        snapshot = receipt.get("snapshot")
        return (
            receipt.get("loader_change_scope") == "dataset_loading_only"
            and isinstance(receipt.get("snapshot_query_receipt"), dict)
            and isinstance(snapshot, dict)
            and isinstance(receipt.get("loader_parity_receipt"), dict)
            and isinstance(receipt.get("snapshot_adapter"), dict)
            and receipt.get("code_revision") == "cfdd0e7b00761a736be660867bf3d09733e24a92"
            and receipt.get("clean_panel_is_scientific_primary") is True
        )
    if role == "dataset_rebuild_and_lineage":
        invariants = receipt.get("invariants", {})
        return (
            invariants.get("stage_order_is_exact") is True
            and invariants.get("v2_stage_b_is_byte_noop") is True
            and invariants.get("replay_stage_b_follows_both_benchmark_filters") is True
            and invariants.get("additional_deduplication") is False
        )
    if role == "heldout_overlap_audit":
        invariants = receipt.get("invariants", {})
        return (
            invariants.get("all_13_panels_frozen") is True
            and invariants.get("exact_panel_text_excluded_from_both_modern_views") is True
            and invariants.get("exact_panel_text_excluded_from_replay_before_benchmark_filters") is True
            and invariants.get("tokenized_inputs_bind_the_exact_heldout_clean_stage_b_bytes") is True
            and invariants.get("additional_deduplication") is False
        )
    if role == "weighted_blend_and_index_caches":
        invariants = receipt.get("invariants", {})
        return (
            receipt.get("shared_by_scales") == ["8b", "1p5b"]
            and invariants.get("data_seed_20260609") is True
            and invariants.get("randomized_gptdataset") is True
            and invariants.get("no_shuffle_patch_disabled") is True
            and invariants.get("phase_2_cache_has_global_historical_horizon") is True
        )
    if role == "cross_scale_realized_sample_ledger_match_through_3218":
        return (
            receipt.get("matched_through_update") == 3218
            and receipt.get("global_consumed_samples") == 3_295_232
            and receipt.get("phase_2_local_consumed_samples") == 979_968
            and all(receipt.get("invariants", {}).values())
        )
    if role == "same_stack_sentinel_calibration_state":
        calibrations = receipt.get("calibrations", {})
        return (
            set(calibrations) == {"8b", "1p5b"}
            and all(row.get("decision_state") in {"4096_pass", "8192_pass", "full_panel_required"} for row in calibrations.values())
            and receipt.get("calibration_windows") == {
                "early": [0, 238, 476, 714],
                "late": [2618, 2856, 3094, 3218],
            }
        )
    if role == "phase_3_unseen_blend_and_capacity_receipt":
        invariants = receipt.get("invariants", {})
        return (
            receipt.get("component_requested_samples") == {
                "active_modern": 386_991,
                "foreign_replay": 97_973,
                "old_greek_replay": 4_899,
            }
            and receipt.get("phase_local_cursor_at_entry") == 0
            and receipt.get("phase_local_cursor_at_update_3456") == 243_712
            and receipt.get("phase_local_cursor_at_update_3694") == 487_424
            and all(invariants.values())
        )
    if role == "production_timing":
        method = receipt.get("method", {})
        scales = receipt.get("scales", {})
        return (
            expected_scale in PRE_MAIN_ARTIFACTS_BY_SCALE
            and receipt.get("scale") == expected_scale
            and method.get("production_equivalent_benchmark_updates") == 256
            and method.get("save_interval") == 119
            and method.get("evaluation_interval") == 25
            and method.get("queue_wait_excluded") is True
            and set(scales) == {expected_scale}
            and len(scales[expected_scale].get("segments", [])) == 6
            and receipt.get("clocks", {}).get("evidence_complete_pending_live_evaluation_backlog") is True
        )
    if role == "allocation_schedule":
        invariants = receipt.get("invariants", {})
        timing_binding = receipt.get("timing_receipt")
        expected_invariants = {
            "direct_normal_holder": True,
            "debug_timer_for_normal_holder": False,
            "source_trigger_uses_conservative_source_wall_time": True,
            "holder_verifies_checkpoint_permit_and_target_cache": True,
            "holder_requires_target_runtime_plus_reserve": True,
            "at_most_one_delayed_successor": True,
            "sbatch_test_only_passed_without_manifest_mutation": True,
        }
        return (
            expected_scale in PRE_MAIN_ARTIFACTS_BY_SCALE
            and receipt.get("scale") == expected_scale
            and receipt.get("allocation_seconds") == 43_200
            and receipt.get("reserve_seconds") == 1_200
            and receipt.get("maximum_pending_delayed_successors") == 1
            and binding_is_current(timing_binding)
            and len(receipt.get("successors", [])) == 5
            and all(row.get("scale") == expected_scale for row in receipt.get("successors", []))
            and invariants == expected_invariants
        )
    if role == "phase_local_cursor_guard_receipt":
        return set(receipt.get("smokes", {})) == {"8b", "1p5b"} and all(binding_is_current(value) for value in receipt["smokes"].values()) and all(receipt.get("checks", {}).values())
    if role == "constant_floor_scheduler_resume_receipt":
        return set(receipt.get("smokes", {})) == {"8b", "1p5b"} and all(binding_is_current(value) for value in receipt["smokes"].values()) and all(receipt.get("checks", {}).values())
    if role == "both_update_3456_checkpoint_permits":
        return receipt.get("update") == 3456 and set(receipt.get("permits", {})) == {"8b", "1p5b"} and all(binding_is_current(value) for value in receipt["permits"].values()) and all(receipt.get("checks", {}).values())
    if role == "phase_3_3456_to_3457_resume_receipts":
        return (
            receipt.get("start_update") == 3456
            and receipt.get("end_update") == 3457
            and set(receipt.get("smokes", {})) == {"8b", "1p5b"}
            and all(binding_is_current(value) for value in receipt["smokes"].values())
            and all(receipt.get("checks", {}).values())
        )
    return True


def validate_artifact_manifest(
    path: Path,
    required_roles: list[str],
    *,
    accepted_producers: set[tuple[str, str, str, int, str]] | None = None,
    expected_scale: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    value = read_json(path)
    require(value.get("schema_version") == "apertus_hard_h_to_g_artifact_manifest_v1", "artifact manifest schema drift")
    artifacts = value.get("artifacts", {})
    require(isinstance(artifacts, dict), "artifact manifest artifacts must be an object")
    if expected_scale is not None:
        require(expected_scale in PRE_MAIN_ARTIFACTS_BY_SCALE, "invalid expected manifest scale")
        require(value.get("gate_stage") == "pre_main", "scale-scoped manifest must be pre-main")
        require(value.get("scale") == expected_scale, "artifact manifest scale drift")
    blockers: list[str] = []
    for role in required_roles:
        row = artifacts.get(role)
        if not isinstance(row, dict):
            blockers.append(f"{role}:missing")
            continue
        artifact_path = Path(str(row.get("path", "")))
        if not artifact_path.is_file():
            blockers.append(f"{role}:file_missing")
            continue
        if row.get("sha256") != sha256_file(artifact_path) or int(row.get("bytes", -1)) != artifact_path.stat().st_size:
            blockers.append(f"{role}:binding_drift")
            continue
        receipt = read_json(artifact_path)
        if receipt.get("schema_version") != ROLE_SCHEMAS[role]:
            blockers.append(f"{role}:schema_{receipt.get('schema_version', 'missing')}")
            continue
        if str(receipt.get("status", "")).lower() not in PASSING:
            blockers.append(f"{role}:status_{receipt.get('status', 'missing')}")
            continue
        if not role_semantics_match(role, receipt, expected_scale=expected_scale):
            blockers.append(f"{role}:semantic_identity_drift")
            continue
        if role == "evaluation_code_bundle":
            if not validate_code_bundle_receipt(receipt):
                blockers.append(f"{role}:code_bundle_drift")
            continue
        if accepted_producers is not None:
            try:
                require_accepted_producer(receipt, accepted_producers, role)
            except (KeyError, TypeError, ValueError) as error:
                blockers.append(f"{role}:producer_bundle_not_compatibility_authorized:{error}")
                continue
        bundle = receipt.get("executing_code_bundle")
        if not isinstance(bundle, dict):
            blockers.append(f"{role}:producer_code_bundle_missing")
            continue
        bundle_root = Path(str(bundle.get("root", "")))
        bundle_receipt_row = bundle.get("receipt")
        if not bundle_root.is_dir() or not isinstance(bundle_receipt_row, dict):
            blockers.append(f"{role}:producer_code_bundle_missing")
            continue
        bundle_receipt_path = Path(str(bundle_receipt_row.get("path", "")))
        if not bundle_receipt_path.is_file():
            blockers.append(f"{role}:producer_code_receipt_missing")
            continue
        if (
            bundle_receipt_row.get("sha256") != sha256_file(bundle_receipt_path)
            or int(bundle_receipt_row.get("bytes", -1)) != bundle_receipt_path.stat().st_size
        ):
            blockers.append(f"{role}:producer_code_receipt_binding_drift")
            continue
        bundle_receipt = read_json(bundle_receipt_path)
        if (
            bundle_receipt.get("schema_version") != "apertus_mini_immutable_code_bundle_v1"
            or bundle_receipt.get("status") != "frozen"
            or bundle_receipt.get("kind") != "scientific"
            or Path(str(bundle_receipt.get("root", ""))).resolve() != bundle_root.resolve()
            or bundle_receipt.get("tree_sha256") != bundle.get("tree_sha256")
        ):
            blockers.append(f"{role}:producer_code_receipt_drift")
    unexpected = sorted(set(artifacts) - set(ALL_ARTIFACTS))
    require(not unexpected, f"unexpected artifact roles: {unexpected}")
    return value, blockers


def main() -> int:
    args = parse_args()
    require(not args.output.exists(), f"immutable output exists: {args.output}")
    experiment = read_json(args.experiment)
    allocation = read_json(args.allocation)
    derived = validate_experiment(experiment)
    allocation_derived = validate_allocation(allocation)

    required_artifacts = artifacts_for_stage(args.gate_stage, args.scale)
    blockers: list[str] = []
    manifest_binding = None
    producer_compatibility_binding = None
    if args.artifact_manifest is None:
        blockers.extend(f"{role}:missing" for role in required_artifacts)
    else:
        require(args.producer_compatibility is not None, "artifact manifest requires producer compatibility authority")
        current = executing_code_bundle()
        _, accepted_producers = load_authority(args.producer_compatibility, current)
        manifest = read_json(args.artifact_manifest)
        require(
            manifest.get("producer_bundle_compatibility") == file_binding(args.producer_compatibility),
            "artifact-manifest producer-compatibility binding drift",
        )
        require(
            manifest.get("executing_code_bundle") == current,
            "artifact-manifest current code-bundle drift",
        )
        _, blockers = validate_artifact_manifest(
            args.artifact_manifest,
            required_artifacts,
            accepted_producers=accepted_producers,
            expected_scale=args.scale,
        )
        manifest_binding = file_binding(args.artifact_manifest)
        producer_compatibility_binding = file_binding(args.producer_compatibility)
    blockers = sorted(set(blockers))
    if args.mode == "launch":
        require(not blockers, f"launch contract blocked: {blockers}")

    payload = {
        "schema_version": "apertus_hard_h_to_g_frozen_contract_v2",
        "status": "launch_ready" if not blockers else "blocked",
        "mode": args.mode,
        "gate_stage": args.gate_stage,
        "scale": args.scale,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "executing_code_bundle": executing_code_bundle(),
        "experiment": file_binding(args.experiment),
        "allocation": file_binding(args.allocation),
        "artifact_manifest": manifest_binding,
        "producer_bundle_compatibility": producer_compatibility_binding,
        "derived": {**derived, **allocation_derived},
        "blockers": blockers,
        "gate_count": len(required_artifacts),
        "all_gates_receipt_backed": not blockers,
    }
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
