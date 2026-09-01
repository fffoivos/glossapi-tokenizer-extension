from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "validate_experiment_matrix.py"
SPEC = importlib.util.spec_from_file_location("validate_experiment_matrix", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ExperimentMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = MODULE.load_matrix(ROOT / "configs" / "experiment_matrix.json")

    def assert_invalid_with(self, value: dict, message: str) -> None:
        self.assertIn(message, MODULE.validate_matrix(value))

    def test_frozen_matrix_is_valid(self) -> None:
        self.assertEqual(MODULE.validate_matrix(self.matrix), [])

    def test_rejects_d1_h_to_g_phase_reversal(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["arms"][1]["phases"].reverse()
        self.assert_invalid_with(value, "D1 hard H-to-G contract drift")

    def test_rejects_d2_g_to_h_phase_reversal(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["arms"][2]["phases"].reverse()
        self.assert_invalid_with(value, "D2 hard G-to-H contract drift")

    def test_rejects_gradual_mirror_drift(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["arms"][4]["window_quota_formula"] = "independently_rounded"
        self.assert_invalid_with(value, "gradual mirror/quota contract drift")

    def test_rejects_gradual_exponent_drift(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["dataset"]["modern_greek_quota_geometry"][
            "gradual_curve_exponent"
        ] = "2.0"
        self.assert_invalid_with(
            value, "modern-Greek quota or gradual-curve geometry drift"
        )

    def test_rejects_replay_identity_drift(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["replay_control"]["same_identity_manifest_across_arms"] = False
        self.assert_invalid_with(value, "replay identities may not differ")

    def test_rejects_silent_replay_deduplication(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["replay_control"]["original_content_multiplicity_preserved"] = False
        self.assert_invalid_with(value, "replay sequence identity/position contract drift")

    def test_rejects_missing_modern_greek_dedup_evidence(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["dataset"]["deduplication_contract"]["modern_greek"] = "unknown"
        self.assert_invalid_with(value, "training deduplication evidence contract drift")

    def test_rejects_incomplete_greek_pool_accounting(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["dataset"]["planning_counts_before_final_exclusions"][
            "glossapi_non_hplt_tokens"
        ] -= 1
        self.assert_invalid_with(value, "Greek planning pools do not sum")

    def test_rejects_frozen_post_exclusion_pool_drift(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["dataset"]["frozen_post_exclusion_counts"]["hplt_tokens"] -= 1
        self.assert_invalid_with(value, "frozen post-exclusion pool accounting drift")

    def test_rejects_source_local_io_becoming_scientific_order(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["data_order_common_contract"]["source_local_document_order_is_io_only"] = False
        self.assert_invalid_with(value, "common exact-once data-order contract drift")

    def test_rejects_iteration_math_drift(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["planning_estimate"]["iterations_ceiling"] -= 1
        self.assert_invalid_with(value, "planning iteration drift")

    def test_rejects_frozen_schedule_iteration_drift(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["frozen_execution_token_geometry"]["optimizer_updates_per_arm"] -= 1
        self.assert_invalid_with(value, "frozen execution token geometry drift")

    def test_rejects_accidental_launch_authorization(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["launch_authorized"] = True
        self.assert_invalid_with(value, "design must not authorize launch")

    def test_rejects_8b_absolute_lr_transfer(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["training_control"]["learning_rate"][
            "recommended_mini_cpt_peak_lr"
        ] = "5.5e-5"
        self.assert_invalid_with(value, "Mini CPT peak LR drift")

    def test_rejects_fixed_lr_tail_shape_drift(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["training_control"]["learning_rate"]["selected_treatment"][
            "tail_shape"
        ] = "linear"
        self.assert_invalid_with(value, "fixed WSD-10 schedule contract drift")

    def test_rejects_checkpoint_averaging_reenabled(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["training_control"]["checkpoint_averaging"]["enabled"] = True
        self.assert_invalid_with(
            value, "raw-final-only endpoint policy drift"
        )

    def test_rejects_five_arm_run_count_drift(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["experiment_design"]["primary_round"]["optimization_trajectories"] = 4
        self.assert_invalid_with(
            value, "five-arm run-count arithmetic drift"
        )

    def test_rejects_b2_segment_boundary_drift(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["training_control"]["parallelism"][
            "selected_normal_partition_segment_boundary_iterations"
        ] = [19248]
        self.assert_invalid_with(value, "B2-selected DP16 segment geometry drift")

    def test_rejects_obsolete_full_dp_ladder_launch_gate(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["launch_gates"].remove(
            "real_scheduled_data_dp16_b1_and_five_arm_dp16_b2_receipted"
        )
        value["launch_gates"].append(
            "dp16_dp32_dp64_dp128_strong_scaling_benchmark_receipted"
        )
        self.assert_invalid_with(value, "B2-selected systems launch-gate drift")

    def test_rejects_validation_sequence_level_split(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["evaluation"]["split_contract"]["document_cluster_level"] = False
        self.assert_invalid_with(value, "validation split/leakage contract drift")

    def test_rejects_endpoint_only_greekmmlu(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["evaluation"]["greekmmlu"]["evaluation_scope"] = "final_endpoint_only"
        self.assert_invalid_with(
            value, "native GreekMMLU trajectory-evaluation contract drift"
        )

    def test_rejects_translated_greekmmlu_substitute(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["evaluation"]["greekmmlu"]["benchmark_origin"] = "translated_from_english"
        self.assert_invalid_with(
            value, "native GreekMMLU trajectory-evaluation contract drift"
        )

    def test_rejects_unpinned_native_greekmmlu_dataset(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["evaluation"]["greekmmlu"]["dataset_repo_id"] = "other/GreekMMLU"
        self.assert_invalid_with(
            value, "native GreekMMLU trajectory-evaluation contract drift"
        )

    def test_rejects_checkpoint_specific_greekmmlu_prompts(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["evaluation"]["greekmmlu"][
            "same_question_ids_prompt_answer_order_and_scoring_code_across_all_checkpoints"
        ] = False
        self.assert_invalid_with(
            value, "native GreekMMLU trajectory-evaluation contract drift"
        )

    def test_rejects_missing_greekmmlu_checkpoint_state(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["training_control"]["checkpoint_policy"][
            "greekmmlu_requires_exact_checkpoint_state"
        ] = False
        self.assert_invalid_with(
            value, "native GreekMMLU checkpoint materialization policy drift"
        )

    def test_rejects_unbound_greekmmlu_checkpoint_conversion(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["training_control"]["checkpoint_policy"][
            "greekmmlu_checkpoint_evaluation_path"
        ]["require_iteration_and_source_checkpoint_hash_binding"] = False
        self.assert_invalid_with(
            value, "native GreekMMLU exact-checkpoint conversion contract drift"
        )

    def test_rejects_checkpoint_pruning_before_evaluation_receipt(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["training_control"]["checkpoint_policy"][
            "prune_nonboundary_payload_only_after_all_evaluation_receipts_are_frozen"
        ] = False
        self.assert_invalid_with(
            value, "native GreekMMLU checkpoint materialization policy drift"
        )

    def test_rejects_repacking_that_changes_goldfish_masks(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["data_order_common_contract"]["repacking_after_schedule_interleave"] = True
        self.assert_invalid_with(value, "common exact-once data-order contract drift")

    def test_rejects_goldfish_hash_seed_drift(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["training_control"]["goldfish_control"]["hash_seed"] += 1
        self.assert_invalid_with(value, "Goldfish mask comparability contract drift")

    def test_rejects_rope_scaling_or_theta_drift(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["training_control"]["rope_theta"] = 12000000
        self.assert_invalid_with(value, "Mini-native RoPE geometry drift")

    def test_rejects_model_geometry_drift(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["model"]["expected_native_geometry"]["num_hidden_layers"] = 32
        self.assert_invalid_with(value, "Mini-native architecture geometry drift")

    def test_rejects_multi_model_gpu_colocation(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["training_control"]["parallelism"][
            "one_training_process_per_physical_gpu"
        ] = False
        self.assert_invalid_with(value, "safe Mini strong-scaling contract drift")

    def test_rejects_36_hour_throughput_target_drift(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["training_control"]["parallelism"][
            "minimum_per_arm_active_tokens_per_second_for_36_hours"
        ] = 1
        self.assert_invalid_with(
            value, "36-hour throughput target or benchmark ladder drift"
        )

    def test_rejects_removing_mps_prohibition(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["training_control"]["parallelism"][
            "forbidden_experimental_optimizations"
        ].remove("cuda_mps_model_colocation")
        self.assert_invalid_with(
            value, "experimental multi-model runtime must remain forbidden"
        )

    def test_rejects_shared_gpu_resource_policy(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["training_control"]["parallelism"]["resource_isolation"] = (
            "shared_gpu"
        )
        self.assert_invalid_with(
            value, "conservative runtime fallback or isolation policy drift"
        )

    def test_rejects_tokenizer_padding(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["tokenizer_and_initialization"]["tokenizer"]["padding_tokens"] = 256
        self.assert_invalid_with(
            value, "tokenizer must be divisible by 256 without padding"
        )

    def test_rejects_pad_metadata_drift_back_to_inst_id(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["tokenizer_and_initialization"]["tokenizer"][
            "pad_metadata_reconciliation"
        ]["output_model_and_tokenizer_pad_token_id"] = 3
        self.assert_invalid_with(value, "Mini pad-metadata reconciliation drift")

    def test_rejects_missing_tied_td_pilot(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["tokenizer_and_initialization"]["initialization"][
            "pilot_candidates"
        ].pop()
        self.assert_invalid_with(value, "tied TD pilot contract drift")

    def test_rejects_different_token_sets_across_td_cells(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["tokenizer_and_initialization"]["initialization"][
            "common_pilot_settings"
        ]["shared_token_ids_across_all_cells"] = False
        self.assert_invalid_with(
            value, "tied TD cells must share one 1024-token pilot"
        )

    def test_rejects_missing_fvt_baseline_non_regression_gate(self) -> None:
        value = copy.deepcopy(self.matrix)
        value["tokenizer_and_initialization"]["initialization"]["selection_gates"].remove(
            "selected_macro_bpb_not_worse_than_frozen_tied_fvt_baseline"
        )
        self.assert_invalid_with(value, "tied TD pilot selection-gate drift")


if __name__ == "__main__":
    unittest.main()
