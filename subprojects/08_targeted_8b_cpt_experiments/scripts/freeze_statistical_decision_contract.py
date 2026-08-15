#!/usr/bin/env python3
"""Freeze the predeclared replication, mirroring, and extension decisions."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

from contract_utils import (
    executing_code_bundle,
    file_binding,
    read_json,
    require,
    write_json_atomic,
)


def validate_statistics(experiment: dict[str, Any]) -> dict[str, Any]:
    statistics = experiment["statistics"]
    evaluation = experiment["evaluation"]
    require(statistics["bootstrap_replicates"] == 10_000, "bootstrap replicate drift")
    require(statistics["bootstrap_seed"] == 20260814, "bootstrap seed drift")
    require(statistics["replication_accuracy_margin"] == 0.015, "replication margin drift")
    require(statistics["replication_interval"] == "wilson_90_percent", "replication interval drift")
    require(statistics["trajectory_correlation"] == "spearman_on_adjacent_first_differences", "trajectory correlation drift")
    require(statistics["bootstrap_scope"] == "evaluation_panel_sampling_uncertainty_conditional_on_one_realized_training_run_per_scale", "bootstrap scope drift")
    require(statistics["trajectory_point_threshold"] == 0.45, "trajectory point threshold drift")
    require(statistics["trajectory_lower_confidence_threshold"] == 0.0, "trajectory confidence threshold drift")
    require(statistics["trajectory_fail_upper_confidence_threshold"] == 0.0, "trajectory fail threshold drift")
    require(statistics["pre_switch_updates"] == [0, 238, 476, 714, 952, 1190, 1428, 1666, 1904, 2142, 2261], "pre-switch window drift")
    require(statistics["post_switch_updates"] == [2261, 2380, 2618, 2856, 3094, 3218], "post-switch window drift")
    require(statistics["extension_updates"] == [3218, 3456, 3694], "extension window drift")
    require(statistics["immediate_switch_pair"] == [2261, 2380], "switch-pair drift")
    expected_metrics = [
        "greekmmlu_choice_nll", "balanced_greek_bpb", "hplt_bpb",
        "openarchives_macro_bpb", "foreign_replay_macro_bpb", "old_greek_bpb",
    ]
    require(statistics["slope_metrics"] == expected_metrics, "slope-metric inventory drift")
    require(statistics["immediate_switch_metrics"] == expected_metrics, "immediate-switch metric inventory drift")
    require(statistics["decision_states"] == ["pass", "fail", "inconclusive", "not_testable"], "decision-state drift")
    require(statistics["resampling_units"] == {
        "greekmmlu_choice_nll": "paired_question",
        "all_bpb_metrics": "paired_document_cluster",
    }, "resampling-unit drift")
    source_aggregation = statistics["source_panel_aggregation"]
    require(source_aggregation == {
        "hplt_bpb": ["hplt"],
        "openarchives_macro_bpb": ["openarchives", "greek_phd", "historical_polytonic"],
        "foreign_replay_macro_bpb": ["english", "de", "ru", "zh", "code", "math"],
        "old_greek_bpb": ["old_greek"],
        "neutral_external_modern_greek_bpb": ["neutral_external_modern_greek"],
        "family_combination": "unweighted_arithmetic_mean_of_family_bpb",
        "balanced_greek_bpb": "0.5_hplt_bpb_plus_0.5_openarchives_macro_bpb",
        "non_hplt_policy": "publish_aggregate_panel_but_do_not_double_count_it_as_a_source_family",
    }, "source-panel aggregation drift")
    forgetting = statistics["forgetting"]
    require(forgetting["panels"] == ["hplt", "foreign_replay_macro", "old_greek"], "forgetting-panel drift")
    require(forgetting["margin_formula_per_panel"] == "2 * median(document_bootstrap_standard_error_at_calibration_updates)", "forgetting margin drift")
    require(forgetting["minimum_is_recomputed_inside_every_bootstrap_replicate"] is True, "forgetting bootstrap minimum drift")
    require(forgetting["calibration_updates"] == [0, 238, 476, 714], "forgetting calibration window drift")
    require(forgetting["cross_scale_pass"] == "same_non_inconclusive_class_for_every_panel", "forgetting agreement pass rule drift")
    require(forgetting["cross_scale_fail"] == "at_least_one_panel_has_opposite_material_vs_no_material_classes", "forgetting agreement fail rule drift")
    require(forgetting["cross_scale_otherwise"] == "inconclusive", "forgetting agreement fallback drift")
    plateau = statistics["plateau_overlap"]
    require(plateau == {
        "pass": "nonempty_intersection",
        "fail": "empty_intersection",
        "inconclusive": "only_if_required_full_panel_scores_are_missing_or_invalid",
    }, "plateau decision drift")
    aggregation = statistics["goal_b_aggregation"]
    require(aggregation["primary"] == ["all_pre_post_slope_direction_tests", "all_immediate_switch_tests", "plateau_overlap", "retention_class_vector"], "Goal-B primary inventory drift")
    require(aggregation["secondary"] == ["greekmmlu_first_difference_spearman", "balanced_greek_bpb_first_difference_spearman"], "Goal-B secondary inventory drift")
    require(aggregation["pass"] == "all_primary_pass_and_at_least_one_secondary_pass_and_no_secondary_fail", "Goal-B pass rule drift")
    require(aggregation["fail"] == "any_primary_fail_or_both_secondary_fail", "Goal-B fail rule drift")
    require(aggregation["otherwise"] == "inconclusive", "Goal-B fallback drift")
    require(aggregation["extension_intervals_use_goal_c_tests_not_spearman"] is True, "extension correlation scope drift")
    goal_c = statistics["goal_c"]
    require(goal_c["margin_calibration_updates"] == [2618, 2856, 3094, 3218], "Goal-C margin window drift")
    require(goal_c["openarchives_improvement_positive_direction"] == "bpb_start_minus_bpb_end", "Goal-C adaptation direction drift")
    require(goal_c["retention_regression_positive_direction"] == "bpb_end_minus_bpb_start", "Goal-C retention direction drift")
    require(goal_c["margin_formula_per_panel"] == "2 * median(document_bootstrap_standard_error_at_margin_calibration_updates)", "Goal-C margin drift")
    require(goal_c["same_margin_for_each_interval_and_cumulative"] is True, "Goal-C margin reuse drift")
    require(goal_c["retention_panels"] == ["hplt", "foreign_replay_macro", "old_greek", "neutral_external_modern_greek"], "Goal-C retention panel drift")
    require(goal_c["pass"] == "openarchives_improvement_lower_bound_exceeds_margin_and_every_retention_regression_upper_bound_is_at_most_its_margin", "Goal-C pass rule drift")
    require(goal_c["fail"] == "openarchives_improvement_upper_bound_is_at_most_margin_or_any_retention_regression_lower_bound_exceeds_its_margin", "Goal-C fail rule drift")
    require(goal_c["otherwise"] == "inconclusive", "Goal-C fallback drift")
    require(goal_c["saturation"] == "first_interval_pass_and_second_interval_openarchives_improvement_fail_with_retention_noninferior", "Goal-C saturation rule drift")
    require(goal_c["inconclusive_second_interval_does_not_establish_saturation"] is True, "Goal-C inconclusive saturation drift")
    sentinel = evaluation["greekmmlu_sentinel"]
    require(sentinel["early_resolution_target"] == "0.5 * median(abs(adjacent_full_choice_nll_delta_at_0_238_476_714))", "sentinel early resolution formula drift")
    require(sentinel["late_resolution_target"] == "0.5 * median(abs(adjacent_full_choice_nll_delta_at_2618_2856_3094_3218))", "sentinel late resolution formula drift")
    require(sentinel["authorization_requires_both_early_and_late_tests"] is True, "sentinel resolution gate weakened")
    require(
        sentinel["cross_scale_comparison_panel"]
        == "largest_sentinel_passing_both_scales_else_full_clean_for_both",
        "cross-scale GreekMMLU panel policy drift",
    )
    return {
        "goal_a": {
            "reference": {"correct": 9969, "questions": 16632},
            "equivalence_margin": 0.015,
            "interval": "90_percent_wilson",
            "pass": "wilson_interval_fully_inside_reference_plus_or_minus_margin",
            "fail": "wilson_interval_fully_outside_equivalence_band",
            "otherwise": "inconclusive",
            "not_testable_if": "tokenizer_is_not_historical_148480_or_legacy_evaluator_drifts",
        },
        "goal_b": {
            "slope_windows": {
                "pre_switch": statistics["pre_switch_updates"],
                "post_switch": statistics["post_switch_updates"],
                "extension": statistics["extension_updates"],
            },
            "slope_gate": {
                "pass": "both_95_percent_intervals_exclude_zero_with_same_sign",
                "fail": "both_95_percent_intervals_exclude_zero_with_opposite_signs",
                "otherwise": "inconclusive",
            },
            "first_difference_spearman_gate": {
                "pass": "point_at_least_0p45_and_90_percent_lower_above_0",
                "fail": "90_percent_upper_below_0",
                "otherwise": "inconclusive",
            },
            "slope_metrics": statistics["slope_metrics"],
            "immediate_switch_metrics": statistics["immediate_switch_metrics"],
            "immediate_switch_pair": statistics["immediate_switch_pair"],
            "aggregation": aggregation,
            "plateau_overlap": plateau,
            "resampling_units": statistics["resampling_units"],
            "source_panel_aggregation": source_aggregation,
            "cumulative_improvement_correlation_forbidden": True,
        },
        "goal_c": {
            "updates": statistics["extension_updates"],
            **goal_c,
        },
        "forgetting": {
            **forgetting,
            "per_panel_quantity": "final_bpb_minus_minimum_prior_bpb",
        },
        "sentinel": {
            "sizes": evaluation["greekmmlu_sentinel"]["sizes"],
            "early_calibration_updates": sentinel["early_calibration_updates"],
            "late_resolution_updates": sentinel["late_resolution_updates"],
            "early_resolution_formula": sentinel["early_resolution_target"],
            "late_resolution_formula": sentinel["late_resolution_target"],
            "authorization_requires_both_tests": True,
            "cross_scale_comparison_panel": sentinel["cross_scale_comparison_panel"],
            "failure_state": evaluation["greekmmlu_sentinel"]["failure_state"],
            "selection_authorized_before_calibration": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable output exists: {args.output}")
    experiment = read_json(args.experiment)
    decisions = validate_statistics(experiment)
    payload = {
        "schema_version": "apertus_hard_h_to_g_statistical_decisions_v2",
        "status": "frozen",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "executing_code_bundle": executing_code_bundle(),
        "experiment": file_binding(args.experiment),
        "bootstrap": {"replicates": 10_000, "seed": 20260814},
        "bootstrap_scope": experiment["statistics"]["bootstrap_scope"],
        "decisions": decisions,
    }
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
