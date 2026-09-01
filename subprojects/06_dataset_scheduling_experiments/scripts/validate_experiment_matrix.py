#!/usr/bin/env python3
"""Validate the five-arm full-corpus data-order screen."""

from __future__ import annotations

import argparse
import json
import math
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "configs" / "experiment_matrix.json"
EXPECTED_MODEL = "swiss-ai/Apertus-v1.1-0.5B"
EXPECTED_MODEL_REVISION = "1b7276176e564fc0cc7d7c3b991a8d653c8b8792"
EXPECTED_DATASET = "fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2"
EXPECTED_DATASET_REVISION = "3f97cec48af502f4996cf8ff20b02660e2dd3d31"
COVERAGE = "every_post_exclusion_eligible_greek_identity_exactly_once"

getcontext().prec = 80


def load_matrix(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("matrix root must be a JSON object")
    return value


def validate_matrix(value: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(
        value.get("schema_version") == "greek_data_order_screen_v1",
        "unexpected schema_version",
    )
    require(value.get("launch_authorized") is False, "design must not authorize launch")

    model = value.get("model", {})
    require(model.get("repo_id") == EXPECTED_MODEL, "base model drift")
    require(model.get("revision") == EXPECTED_MODEL_REVISION, "model revision drift")
    require(model.get("checkpoint_type") == "base_not_instruct", "must use base model")
    geometry = model.get("expected_native_geometry", {})
    require(
        geometry
        == {
            "architecture": "ApertusForCausalLM",
            "hidden_size": 1024,
            "intermediate_size": 6144,
            "num_hidden_layers": 20,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "hidden_activation": "xielu",
            "qk_norm": True,
            "post_norm": False,
            "rms_norm_eps": 1e-5,
            "attention_bias": False,
            "mlp_bias": False,
            "attention_dropout": 0.0,
            "hidden_dropout": 0.0,
            "initializer_range": 0.02,
            "tie_word_embeddings": True,
        },
        "Mini-native architecture geometry drift",
    )

    dataset = value.get("dataset", {})
    require(dataset.get("repo_id") == EXPECTED_DATASET, "dataset repo drift")
    require(dataset.get("revision") == EXPECTED_DATASET_REVISION, "dataset revision drift")
    require(dataset.get("coverage_contract") == COVERAGE, "dataset coverage drift")
    require(dataset.get("heldouts_excluded_before_schedule") is True, "heldouts must precede scheduling")
    require(
        dataset.get("greekmmlu_decontamination_before_schedule") is True,
        "GreekMMLU decontamination must precede scheduling",
    )
    require(
        dataset.get("document_identity") == ["source_dataset", "source_doc_id"],
        "document identity drift",
    )
    deduplication = dataset.get("deduplication_contract", {})
    require(
        deduplication.get("modern_greek")
        == "global_exact_content_uniqueness_required_and_receipted_over_text_sha256"
        and deduplication.get("replay")
        == "preserve_original_training_source_records_and_audit_exact_content_duplicates_without_removing_them"
        and deduplication.get("record_identity")
        == "source_document_cluster_id_plus_text_sha256"
        and deduplication.get(
            "source_document_cluster_ids_may_name_multiple_distinct_text_records"
        )
        is True,
        "training deduplication evidence contract drift",
    )

    selectors = dataset.get("selectors", {})
    require(selectors.get("hplt", {}).get("value") == "^HPLT/", "HPLT selector drift")
    require(
        selectors.get("glossapi_non_hplt", {}).get("operation") == "not_regex"
        and selectors.get("glossapi_non_hplt", {}).get("value") == "^HPLT/",
        "GlossAPI/non-HPLT selector must complement HPLT",
    )

    counts = dataset.get("planning_counts_before_final_exclusions", {})
    try:
        all_greek = int(counts["all_greek_tokens"])
        hplt = int(counts["hplt_tokens"])
        non_hplt = int(counts["glossapi_non_hplt_tokens"])
        require(hplt + non_hplt == all_greek, "Greek planning pools do not sum")
    except (KeyError, TypeError, ValueError):
        errors.append("invalid Greek planning counts")

    frozen_counts = dataset.get("frozen_post_exclusion_counts", {})
    try:
        frozen_hplt = int(frozen_counts["hplt_tokens"])
        frozen_non_hplt = int(frozen_counts["glossapi_non_hplt_tokens"])
        frozen_modern = int(frozen_counts["all_modern_greek_tokens"])
        frozen_foreign = int(frozen_counts["foreign_replay_tokens"])
        frozen_old = int(frozen_counts["old_greek_replay_tokens"])
        frozen_total = int(frozen_counts["all_active_tokens"])
        require(
            frozen_hplt + frozen_non_hplt == frozen_modern
            and frozen_modern + frozen_foreign + frozen_old == frozen_total
            and frozen_counts.get("operative_for_schedule_geometry") is True
            and frozen_counts.get("pool_corpus_receipt_sha256")
            == "76658cc8495b58a3a3dadc8aca6d16c7fed627ef8bced89d17a877f9f9125014",
            "frozen post-exclusion pool accounting drift",
        )
    except (KeyError, TypeError, ValueError):
        errors.append("invalid frozen post-exclusion pool counts")

    quota = dataset.get("modern_greek_quota_geometry", {})
    try:
        all_greek = Decimal(str(frozen_counts["all_modern_greek_tokens"]))
        hplt = Decimal(str(frozen_counts["hplt_tokens"]))
        non_hplt = Decimal(str(frozen_counts["glossapi_non_hplt_tokens"]))
        q_h = hplt / all_greek
        q_g = non_hplt / all_greek
        tolerance = Decimal("1e-49")
        require(
            abs(Decimal(quota["hplt_fraction"]) - q_h) < tolerance
            and abs(Decimal(quota["glossapi_non_hplt_fraction"]) - q_g) < tolerance
            and abs(
                Decimal(quota["gradual_curve_exponent"])
                - (Decimal(1) / q_g - Decimal(1))
            )
            < tolerance,
            "modern-Greek quota or gradual-curve geometry drift",
        )
        require(
            quota.get("gradual_window_count") == 128
            and quota.get("fixed_pool_permutations_across_arms") is True
            and quota.get("glossapi_internal_curriculum") is False
            and quota.get("source_dataset_used_as_order_key_inside_glossapi") is False,
            "gradual-window or internal GlossAPI-order contract drift",
        )
        require(
            quota.get("packing_io_order")
            == "selected_seeded_prefix_document_set_reordered_only_by_source_task_and_document_index_before_packing"
            and quota.get("packing_io_order_is_not_scientific_schedule_order") is True
            and quota.get("pool_permutation")
            == "one_deterministic_splitmix64_permutation_of_immutable_packed_sequence_ids_per_top_level_pool",
            "source-local packing versus scientific-order contract drift",
        )
    except (KeyError, TypeError, ValueError, ArithmeticError):
        errors.append("invalid modern-Greek quota geometry")

    replay = value.get("replay_control", {})
    require(replay.get("same_identity_manifest_across_arms") is True, "replay identities may not differ")
    require(replay.get("same_cumulative_token_positions_across_arms") is True, "replay placement may not differ")
    require(replay.get("stationary_across_schedule_phases") is True, "replay must be stationary")
    require(
        replay.get("same_sequence_ids_and_positions_across_arms") is True
        and replay.get("selected_without_replacement") is True
        and replay.get("original_content_multiplicity_preserved") is True
        and replay.get("duplicate_content_rate_measured_and_receipted") is True,
        "replay sequence identity/position contract drift",
    )
    try:
        selected = replay["selected_mix"]
        mixture_sum = sum(Decimal(selected[name]) for name in ("new_greek", "foreign_replay", "old_greek_replay"))
        require(mixture_sum == Decimal(1), "selected replay mixture must sum to one")
        require(
            replay.get("status") == "selected_and_frozen_by_experiment_design",
            "replay mixture must be explicitly selected before launch",
        )
    except (KeyError, TypeError, ValueError):
        errors.append("invalid selected replay mixture")

    training = value.get("training_control", {})
    try:
        sequence = int(training["sequence_length"])
        global_sequences = int(training["global_batch_sequences"])
        global_tokens = int(training["global_batch_tokens"])
        require(sequence * global_sequences == global_tokens, "global batch token geometry drift")
    except (KeyError, TypeError, ValueError):
        errors.append("invalid training geometry")
    require(
        training.get("same_nonfactor_optimizer_controls_across_all_cells") is True,
        "non-factor optimizer controls may not differ",
    )
    require(
        training.get("resume_global_iteration_across_data_boundaries") is True,
        "data boundaries must preserve global iteration",
    )
    require(
        training.get("reset_optimizer_at_data_boundary") is False
        and training.get("reset_scheduler_at_data_boundary") is False,
        "hard data boundaries may not reset optimizer or scheduler",
    )
    goldfish = training.get("goldfish_control", {})
    require(
        training.get("loss_objective") == "goldfish_k50_h50"
        and goldfish.get("k") == 50
        and goldfish.get("h") == 50
        and goldfish.get("hash_table_size") == 1000003
        and goldfish.get("hash_seed") == 2971215073
        and goldfish.get("mask_is_function_of_frozen_sequence_labels") is True
        and goldfish.get("require_identical_per_sequence_mask_hash_across_all_arms") is True
        and goldfish.get("require_added_token_hash_uniformity_gate") is True
        and goldfish.get("forbid_repacking_or_context_change_after_sequence_manifest_freeze") is True,
        "Goldfish mask comparability contract drift",
    )
    parallelism = training.get("parallelism", {})
    require(
        parallelism.get("tensor_parallel") == 1
        and parallelism.get("pipeline_parallel") == 1
        and parallelism.get("candidate_data_parallel") == [16, 32, 64, 128]
        and parallelism.get("candidate_nodes_per_arm") == [4, 8, 16, 32]
        and (
            parallelism.get("selected_data_parallel"),
            parallelism.get("selected_nodes_per_arm"),
        )
        in {(None, None), (16, 4), (32, 8), (64, 16), (128, 32)}
        and parallelism.get("one_training_process_per_physical_gpu") is True
        and parallelism.get("same_runtime_layout_across_schedule_arms") is True,
        "safe Mini strong-scaling contract drift",
    )
    require(
        parallelism.get("selected_data_parallel") == 16
        and parallelism.get("selected_nodes_per_arm") == 4
        and parallelism.get("selected_training_nodes_total") == 20
        and parallelism.get("selection_receipt")
        == "apertus-cscs-efficiency/evidence/mini_b2_five_arm_contention_20260802.json"
        and parallelism.get("selected_projected_end_to_end_training_hours")
        == 19.9979
        and parallelism.get("selected_normal_partition_segment_count") == 2
        and parallelism.get(
            "selected_normal_partition_segment_boundary_iterations"
        )
        == [19456]
        and parallelism.get(
            "selected_segment_boundary_is_regular_512_step_checkpoint"
        )
        is True,
        "B2-selected DP16 segment geometry drift",
    )
    launch_gates = value.get("launch_gates", [])
    require(
        "real_scheduled_data_dp16_b1_and_five_arm_dp16_b2_receipted"
        in launch_gates
        and "dp16_dp32_dp64_dp128_strong_scaling_benchmark_receipted"
        not in launch_gates,
        "B2-selected systems launch-gate drift",
    )
    require(
        parallelism.get("full_campaign_candidate_nodes")
        == {"dp16": 20, "dp32": 40, "dp64": 80, "dp128": 160}
        and parallelism.get("full_run_layout")
        == "one_aggregate_allocation_with_five_disjoint_equal_size_data_parallel_groups",
        "full-run five-arm allocation layout drift",
    )
    require(
        parallelism.get("full_run_fallback_layout")
        == "five_equal_independent_allocations_at_the_same_selected_data_parallel_size"
        and parallelism.get("resource_isolation") == "disjoint_gpu_sets_no_shared_gpu"
        and parallelism.get("implementation_policy")
        == "upstream_or_existing_project_runtime_paths_only",
        "conservative runtime fallback or isolation policy drift",
    )
    require(
        parallelism.get("wall_clock_target_hours_for_all_five_arms") == 36
        and parallelism.get("preferred_wall_clock_budget_hours_for_training") == 24
        and parallelism.get("preferred_single_normal_partition_segment_hours") == 12
        and parallelism.get("minimum_per_arm_active_tokens_per_second_for_36_hours")
        == 622916
        and parallelism.get("preferred_per_arm_active_tokens_per_second_for_24_hours")
        == 934374
        and parallelism.get("preferred_per_arm_active_tokens_per_second_for_12_hours")
        == 1868749
        and parallelism.get("strong_scaling_benchmark_data_parallel")
        == [16, 32, 64, 128],
        "36-hour throughput target or benchmark ladder drift",
    )
    require(
        parallelism.get("micro_batch_candidates_by_data_parallel")
        == {"16": [32, 16, 8, 4], "32": [16, 8, 4], "64": [8, 4], "128": [4]},
        "strong-scaling microbatch geometry drift",
    )
    required_speedups = {
        "bf16",
        "native_apertus_fused_kernels",
        "cross_entropy_loss_fusion",
        "distributed_optimizer_with_overlap_grad_reduce_and_param_gather",
        "disable_activation_recomputation_when_memory_safe",
        "selective_activation_recomputation_only_if_required",
        "fixed_length_sequence_packing",
        "pinned_memory_and_asynchronous_prefetch",
        "standard_nccl_data_parallelism",
        "torch_dist_async_checkpoint_save",
        "fully_parallel_distributed_checkpoint_load",
    }
    require(
        set(parallelism.get("allowed_standard_optimizations", []))
        == required_speedups,
        "conventional speedup allowlist drift",
    )
    require(
        set(parallelism.get("forbidden_experimental_optimizations", []))
        == {
            "multiple_training_models_per_physical_gpu",
            "cuda_mps_model_colocation",
            "vmap_or_stacked_parameter_population_training",
            "custom_grouped_gemm_multi_model_runtime",
            "new_custom_cuda_or_triton_kernels",
            "cross_model_shared_forward_or_optimizer_state",
        },
        "experimental multi-model runtime must remain forbidden",
    )
    require(
        training.get("rope_theta") == 500000
        and training.get("rope_type") == "default"
        and training.get("rope_scaling") == {"rope_type": "default"}
        and training.get("rope_scaling_effect") == "none"
        and training.get("max_position_embeddings") == 4096,
        "Mini-native RoPE geometry drift",
    )
    learning_rate = training.get("learning_rate", {})
    require(
        learning_rate.get("transfer_policy")
        == "preserve_cpt_to_pretraining_peak_lr_ratio_across_model_scales",
        "LR transfer policy drift",
    )
    require(
        learning_rate.get("recommended_mini_cpt_peak_lr") == "3e-4",
        "Mini CPT peak LR drift",
    )
    require(learning_rate.get("warmup_steps") == 800, "warmup-step scaling drift")
    require(
        learning_rate.get("warmup_tokens") == 1677721600,
        "warmup token-mass drift",
    )
    selected_lr = learning_rate.get("selected_treatment", {})
    require(
        selected_lr.get("lr_id") == "L0_wsd10"
        and selected_lr.get("schedule") == "wsd"
        and selected_lr.get("tail_start_lr_ratio_to_peak") == "1.0"
        and selected_lr.get("tail_end_lr_ratio_to_peak") == "0.10"
        and selected_lr.get("tail_shape") == "one_minus_sqrt"
        and learning_rate.get("tail_fraction") == "0.20"
        and learning_rate.get(
            "selected_schedule_applies_identically_to_all_five_data_orders"
        )
        is True
        and learning_rate.get("later_lr_floor_study_out_of_scope")
        == ["L1_wsd20", "L2_wsd30"],
        "fixed WSD-10 schedule contract drift",
    )
    branching = training.get("lr_branching", {})
    require(
        branching.get("enabled") is False,
        "primary round must not branch into LR treatments",
    )
    averaging = training.get("checkpoint_averaging", {})
    require(
        averaging.get("enabled") is False
        and averaging.get("endpoint_variants") == ["raw_final"]
        and averaging.get("sma_enabled") is False
        and averaging.get("ema_enabled") is False,
        "raw-final-only endpoint policy drift",
    )
    checkpoint_policy = training.get("checkpoint_policy", {})
    require(
        training.get("checkpoint_cadence_steps") == 512
        and training.get("checkpoint_cadence_tokens") == 1073741824
        and checkpoint_policy.get("full_state_saves_are_async") is True
        and checkpoint_policy.get("fast_validation_does_not_require_a_full_checkpoint")
        is True,
        "checkpoint cadence policy drift",
    )
    require(
        checkpoint_policy.get("greekmmlu_requires_exact_checkpoint_state") is True
        and checkpoint_policy.get(
            "prune_nonboundary_payload_only_after_all_evaluation_receipts_are_frozen"
        )
        is True
        and set(checkpoint_policy.get("mandatory_full_state_points", []))
        == {
            "every_512_steps_for_native_greekmmlu",
            "after_warmup",
            "each_12_hour_segment_boundary",
            "immediately_before_and_after_each_hard_data_transition",
            "matched_token_points_for_non_hard_arms",
            "cooldown_start",
            "final_endpoint",
        },
        "native GreekMMLU checkpoint materialization policy drift",
    )
    conversion = checkpoint_policy.get("greekmmlu_checkpoint_evaluation_path", {})
    require(
        conversion.get("source_checkpoint_format") == "torch_dist"
        and conversion.get("intermediate_conversion")
        == "SwissAI_Megatron_scripts/conversion/torchdist_2_torch.py"
        and conversion.get("hf_conversion")
        == "SwissAI_Megatron_tools/checkpoint/convert.py_loader_core_saver_swissai_hf"
        and conversion.get("hf_tokenizer_is_frozen_extended_tokenizer") is True
        and conversion.get("require_iteration_and_source_checkpoint_hash_binding") is True
        and conversion.get("require_conversion_receipt") is True
        and conversion.get("require_logit_equivalence_smoke_before_campaign") is True
        and conversion.get("evaluate_only_exact_converted_checkpoint") is True
        and conversion.get("conversion_and_evaluation_run_asynchronously_on_separate_nodes")
        is True,
        "native GreekMMLU exact-checkpoint conversion contract drift",
    )
    lr_smoke = learning_rate.get("common_stability_smoke", {})
    require(
        lr_smoke.get("steps") == 1024
        and lr_smoke.get("tokens") == 2147483648
        and lr_smoke.get("warmup_steps") == 800
        and lr_smoke.get("stable_peak_steps_observed") == 224
        and lr_smoke.get("restart_full_arms_from_frozen_td_initialization") is True
        and lr_smoke.get("must_not_select_schedule_arm") is True,
        "common LR stability-smoke contract drift",
    )

    arms = value.get("arms", [])
    require(isinstance(arms, list) and len(arms) == 5, "exactly five data-order arms are required")
    by_id = {
        arm.get("arm_id"): arm
        for arm in arms
        if isinstance(arm, dict) and isinstance(arm.get("arm_id"), str)
    }
    expected_arm_ids = {
        "D0_mixed",
        "D1_hard_h_to_g",
        "D2_hard_g_to_h",
        "D3_gradual_h_to_g",
        "D4_gradual_g_to_h",
    }
    require(set(by_id) == expected_arm_ids, "data-order arm IDs drift")

    common_order = value.get("data_order_common_contract", {})
    require(
        common_order.get("coverage_contract") == COVERAGE
        and common_order.get("same_document_and_loss_active_token_multiset_across_arms") is True
        and common_order.get("same_replay_sequence_ids_at_same_global_positions") is True
        and common_order.get("same_fixed_top_level_pool_permutations_across_arms") is True
        and common_order.get("glossapi_pool_is_aggregate_randomized_without_internal_curriculum") is True
        and common_order.get("top_level_pools_packed_independently_before_scheduling") is True
        and common_order.get("source_local_document_order_is_io_only") is True
        and common_order.get("same_seeded_prefix_document_set_before_io_reordering") is True
        and common_order.get("frozen_randomized_packed_sequence_catalog_is_scientific_pool_order") is True
        and common_order.get("immutable_sequence_payloads_and_ids_across_arms") is True
        and common_order.get("schedule_operates_only_on_sequence_ids") is True
        and common_order.get("repacking_after_schedule_interleave") is False
        and common_order.get("same_goldfish_mask_bitmap_for_each_sequence_id_across_arms") is True,
        "common exact-once data-order contract drift",
    )
    for arm in arms if isinstance(arms, list) else []:
        require(
            arm.get("common_contract_ref") == "data_order_common_contract"
            and arm.get("replay_control_ref") == "replay_control",
            "all arms must share data and replay controls",
        )

    mixed = by_id.get("D0_mixed", {})
    require(
        mixed.get("schedule_type") == "stationary_quota_mixture"
        and mixed.get("glossapi_fraction_within_modern_curve") == "q_G"
        and mixed.get("hplt_fraction_within_modern_curve") == "q_H"
        and mixed.get("window_count") == 128,
        "D0 stationary mixture contract drift",
    )

    hard_hg = by_id.get("D1_hard_h_to_g", {})
    hard_gh = by_id.get("D2_hard_g_to_h", {})
    hard_hg_phases = hard_hg.get("phases", [])
    hard_gh_phases = hard_gh.get("phases", [])
    require(
        [phase.get("modern_greek_pool") for phase in hard_hg_phases]
        == ["hplt", "glossapi_non_hplt"]
        and hard_hg.get("hard_transition_after_total_progress_fraction")
        == quota.get("hplt_fraction"),
        "D1 hard H-to-G contract drift",
    )
    require(
        [phase.get("modern_greek_pool") for phase in hard_gh_phases]
        == ["glossapi_non_hplt", "hplt"]
        and hard_gh.get("hard_transition_after_total_progress_fraction")
        == quota.get("glossapi_non_hplt_fraction"),
        "D2 hard G-to-H contract drift",
    )

    gradual_hg = by_id.get("D3_gradual_h_to_g", {})
    gradual_gh = by_id.get("D4_gradual_g_to_h", {})
    require(
        gradual_hg.get("glossapi_fraction_within_modern_curve") == "g(u)=u^a"
        and gradual_gh.get("glossapi_fraction_within_modern_curve") == "g(u)=(1-u)^a"
        and gradual_hg.get("window_count") == gradual_gh.get("window_count") == 128
        and gradual_hg.get("integral_glossapi_fraction")
        == gradual_gh.get("integral_glossapi_fraction")
        == "q_G"
        and gradual_gh.get("window_quota_formula") == "D3_window_quota_reversed",
        "gradual mirror/quota contract drift",
    )

    design = value.get("experiment_design", {})
    primary = design.get("primary_round", {})
    require(
        set(design.get("data_order_ids", [])) == expected_arm_ids
        and design.get("fixed_lr_id") == "L0_wsd10"
        and design.get("endpoint_variant_ids") == ["raw_final"]
        and design.get("lr_floor_experiment") == "out_of_scope_for_primary_round"
        and design.get("checkpoint_averaging") == "disabled",
        "primary experiment axis definitions drift",
    )
    require(
        primary.get("optimization_trajectories") == 5
        and primary.get("evaluated_endpoint_artifacts") == 5
        and primary.get("full_run_equivalents") == "5.0"
        and primary.get("aggregate_training_tokens_planning_value")
        == 403941528685
        and primary.get("aggregate_training_tokens_frozen_post_exclusion_value")
        == 403649695335,
        "five-arm run-count arithmetic drift",
    )
    try:
        n_data = len(design["data_order_ids"])
        n_endpoints = len(design["endpoint_variant_ids"])
        per_run_tokens = int(value["planning_estimate"]["total_tokens_rounded"])
        require(
            primary["optimization_trajectories"] == n_data
            and primary["evaluated_endpoint_artifacts"] == n_data * n_endpoints
            and Decimal(primary["full_run_equivalents"]) == Decimal(n_data)
            and primary["aggregate_training_tokens_planning_value"]
            == n_data * per_run_tokens,
            "five-arm dimensions do not reproduce declared counts",
        )
        require(
            primary["aggregate_training_tokens_frozen_post_exclusion_value"]
            == n_data * int(frozen_counts["all_active_tokens"]),
            "five-arm frozen post-exclusion token arithmetic drift",
        )
    except (KeyError, TypeError, ValueError, ArithmeticError):
        errors.append("invalid five-arm dimension arithmetic")

    comparison = value.get("comparison_contract", {})
    require(
        comparison.get("only_optimization_factors")
        == ["modern_greek_temporal_order"]
        and comparison.get("post_training_factor") == "none_raw_final_only",
        "factor definition contract drift",
    )
    must_match = set(comparison.get("must_match", []))
    required_controls = {
        "base_checkpoint_hash",
        "tokenizer_hash",
        "initialization_hash",
        "eligible_greek_identity_set",
        "eligible_greek_loss_active_token_multiset",
        "fixed_hplt_pool_permutation",
        "fixed_glossapi_pool_permutation",
        "replay_identity_set",
        "replay_sequence_ids_and_global_positions",
        "total_new_greek_tokens",
        "total_replay_tokens",
        "packing_and_document_masks",
        "global_batch_tokens",
        "optimizer_and_learning_rate_schedule",
        "selected_runtime_geometry",
        "loss_objective",
        "evaluation_inputs",
        "evaluation_code_revision",
    }
    require(must_match == required_controls, "must-match control set drift")

    estimate = value.get("planning_estimate", {})
    try:
        greek = Decimal(int(estimate["new_greek_tokens"]))
        fraction = Decimal(estimate["new_greek_fraction"])
        total = greek / fraction
        rounded_total = int(total.quantize(Decimal("1")))
        batch = int(estimate["global_batch_tokens"])
        require(int(estimate["total_tokens_rounded"]) == rounded_total, "planning total drift")
        require(int(estimate["iterations_ceiling"]) == math.ceil(total / batch), "planning iteration drift")
        require(
            abs(
                Decimal(estimate["hard_hplt_phase_tokens_fractional"])
                + Decimal(estimate["hard_glossapi_phase_tokens_fractional"])
                - total
            )
            < Decimal("1e-17"),
            "staged phase planning tokens do not sum to total",
        )
        require(
            int(estimate["all_replay_tokens_rounded_once"])
            == int((total * Decimal("0.21")).quantize(Decimal("1")))
            and int(estimate["primary_round_optimization_trajectories"]) == 5
            and int(estimate["primary_round_evaluated_endpoint_artifacts"]) == 5
            and estimate["primary_round_full_run_equivalents"] == "5.0"
            and int(estimate["primary_round_aggregate_tokens_rounded"])
            == 5 * rounded_total,
            "replay or five-arm planning arithmetic drift",
        )
    except (KeyError, TypeError, ValueError, ArithmeticError):
        errors.append("invalid planning estimate")
    require(estimate.get("not_valid_for_launch") is True, "planning estimate must be non-launchable")

    execution = value.get("frozen_execution_token_geometry", {})
    require(
        execution.get("modern_greek_tokens") == frozen_counts.get("all_modern_greek_tokens")
        and execution.get("hplt_tokens") == frozen_counts.get("hplt_tokens")
        and execution.get("glossapi_non_hplt_tokens")
        == frozen_counts.get("glossapi_non_hplt_tokens")
        and execution.get("foreign_replay_tokens")
        == frozen_counts.get("foreign_replay_tokens")
        and execution.get("old_greek_replay_tokens")
        == frozen_counts.get("old_greek_replay_tokens")
        and execution.get("total_active_tokens") == frozen_counts.get("all_active_tokens")
        and execution.get("five_arm_aggregate_active_tokens")
        == 5 * int(frozen_counts.get("all_active_tokens", -1))
        and execution.get("packed_corpus_receipt_sha256")
        == "4c3ae0a49f6733c32525b921d33fd97338bf7d09f00f1227d4cbaab0e24dfd2f"
        and execution.get("schedule_manifest_sha256")
        == "ffeaa69492b0a30768efb5c34a942e1b7d11ca5df0d962d001ae6387d6f20955"
        and execution.get("real_packed_sequences_per_arm") == 19709692
        and execution.get("loss_inactive_global_batch_filler_sequences_per_arm") == 260
        and execution.get("scheduled_sequences_per_arm") == 19709952
        and execution.get("scheduled_token_slots_per_arm") == 80731963392
        and execution.get("packing_and_global_batch_filler_token_slots") == 2024325
        and execution.get("optimizer_updates_per_arm") == 38496
        and execution.get(
            "exact_optimizer_updates_frozen_from_packed_sequence_and_global_batch_padding_receipt"
        )
        is True,
        "frozen execution token geometry drift",
    )

    evaluation = value.get("evaluation", {})
    split = evaluation.get("split_contract", {})
    require(
        split.get("split_before_packing") is True
        and split.get("document_cluster_level") is True
        and split.get("global_near_dedup_across_train_and_validation") is True,
        "validation split/leakage contract drift",
    )
    panels = set(evaluation.get("validation_panels", []))
    require(
        panels
        == {
            "hplt_heldout",
            "glossapi_aggregate_heldout",
            "glossapi_per_source_or_predeclared_family_heldouts",
            "neutral_external_modern_greek",
            "foreign_replay_per_language_heldouts",
            "old_greek_heldout",
            "added_token_stratified_heldout",
        },
        "source-conditioned validation panel drift",
    )
    greekmmlu = evaluation.get("greekmmlu", {})
    require(
        greekmmlu.get("dataset_repo_id") == "dascim/GreekMMLU"
        and greekmmlu.get("dataset_revision")
        == "6a03aa06b68beb932fb75edff3a34e50b3674649"
        and greekmmlu.get("dataset_config") == "All"
        and greekmmlu.get("dataset_split") == "test"
        and greekmmlu.get("benchmark_origin") == "natively_authored_greek"
        and greekmmlu.get("checkpoint_evaluation_is_mandatory") is True
        and greekmmlu.get("primary_for_0_5b_selection") is False
        and greekmmlu.get("zero_shot_primary") is True
        and greekmmlu.get("evaluation_scope")
        == "every_required_training_point_for_every_arm_not_endpoint_only"
        and greekmmlu.get("required_training_points_ref")
        == "evaluation.required_training_points"
        and greekmmlu.get(
            "same_question_ids_prompt_answer_order_and_scoring_code_across_all_checkpoints"
        )
        is True
        and greekmmlu.get("exact_checkpoint_conversion_contract_ref")
        == "training_control.checkpoint_policy.greekmmlu_checkpoint_evaluation_path"
        and greekmmlu.get("freeze_complete_evaluator_contract") is True
        and greekmmlu.get("metrics")
        == [
            "official_zero_shot_accuracy",
            "multiple_choice_cross_entropy_from_frozen_normalized_choice_scores",
            "correct_answer_continuation_bpb",
        ],
        "native GreekMMLU trajectory-evaluation contract drift",
    )
    require(
        evaluation.get("winner_rule", [])[:3]
        == [
            "apply_predeclared_foreign_old_greek_and_general_retention_noninferiority_margins",
            "among_passers_minimize_neutral_external_greek_bpb",
            "then_maximize_balanced_hplt_glossapi_relative_gain",
        ],
        "predeclared winner-selection hierarchy drift",
    )

    token_init = value.get("tokenizer_and_initialization", {})
    require(
        token_init.get("status") == "selected_requires_artifact_receipts_before_launch",
        "tokenizer/init policy drift",
    )
    tokenizer = token_init.get("tokenizer", {})
    require(
        tokenizer.get("policy")
        == "preserve_mini_base_ids_and_append_exact_production_greek_merge_chain",
        "Mini-compatible tokenizer overlay policy drift",
    )
    require(
        tokenizer.get("base_vocab_size") == 131072
        and tokenizer.get("added_token_count") == 17920
        and tokenizer.get("target_vocab_size") == 148992,
        "tokenizer vocabulary geometry drift",
    )
    require(
        tokenizer.get("target_vocab_size") == 148992
        and tokenizer.get("alignment_divisor") == 256
        and 148992 % 256 == 0
        and tokenizer.get("padding_tokens") == 0,
        "tokenizer must be divisible by 256 without padding",
    )
    pad_metadata = tokenizer.get("pad_metadata_reconciliation", {})
    require(
        pad_metadata.get("source_model_config_pad_token_id") == 3
        and pad_metadata.get("source_token_at_id_3") == "[INST]"
        and pad_metadata.get("existing_pad_token_id") == 10
        and pad_metadata.get("output_model_and_tokenizer_pad_token_id") == 10
        and pad_metadata.get("changes_token_ids_or_merges") is False,
        "Mini pad-metadata reconciliation drift",
    )
    initialization = token_init.get("initialization", {})
    require(initialization.get("tie_word_embeddings") is True, "Mini embeddings must remain tied")
    require(
        initialization.get("learn_output_with_separate_ce") is False,
        "tied TD must not use a separate output-only CE update",
    )
    require(
        initialization.get("forbid_reusing_apertus_8b_embedding_rows") is True,
        "8B embedding rows must be forbidden for the 0.5B model",
    )
    pilot_settings = initialization.get("common_pilot_settings", {})
    require(
        pilot_settings.get("shared_stratified_token_count") == 1024
        and pilot_settings.get("shared_token_ids_across_all_cells") is True,
        "tied TD cells must share one 1024-token pilot",
    )
    pilots = initialization.get("pilot_candidates", [])
    pilot_contract = {
        (
            pilot.get("target_layer"),
            tuple(pilot.get("loss_methods", [])),
        )
        for pilot in pilots
        if isinstance(pilot, dict)
    }
    require(
        pilot_contract
        == {
            (7, ("MSE-on-hiddens",)),
            (7, ("MSE-on-hiddens", "CE-auto-weighted")),
            (-1, ("MSE-on-hiddens",)),
            (-1, ("MSE-on-hiddens", "CE-auto-weighted")),
        },
        "tied TD pilot contract drift",
    )
    require(
        set(initialization.get("selection_gates", []))
        == {
            "all_non_new_rows_bitwise_preserved",
            "input_and_output_weights_share_storage",
            "all_embedding_values_finite",
            "max_new_row_norm_over_base_p999_at_most_4",
            "no_new_token_argmax_or_generation_collapse",
            "best_joint_heldout_bpb_on_hplt_non_hplt_and_polytonic_slices",
            "selected_macro_bpb_not_worse_than_frozen_tied_fvt_baseline",
        },
        "tied TD pilot selection-gate drift",
    )
    full_initialization = initialization.get("full_initialization_after_pilot", {})
    require(
        full_initialization.get("run_count") == 1
        and full_initialization.get("use_selected_layer_and_loss_profile") is True
        and full_initialization.get("requested_rows")
        == "complete_ordered_range_131072..148991"
        and full_initialization.get("minimum_trained_token_fraction") == "0.90"
        and full_initialization.get("low_coverage_fallback")
        == "retain_exact_fvt_subtoken_mean_row",
        "full tied TD run/fallback policy drift",
    )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", nargs="?", type=Path, default=DEFAULT_MATRIX)
    args = parser.parse_args()
    errors = validate_matrix(load_matrix(args.matrix))
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"PASS: {args.matrix}")
    print(
        "data_orders=5; fixed_lr=L0_wsd10; trajectories=5; "
        "endpoint_artifacts=5; full_run_equivalents=5; launch_authorized=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
