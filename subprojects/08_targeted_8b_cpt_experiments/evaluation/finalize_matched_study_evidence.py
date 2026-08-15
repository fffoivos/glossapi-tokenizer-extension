#!/usr/bin/env python3
"""Finalize joint source, GreekMMLU, native-suite, and statistical authorities."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from canonical_evidence import completed_result
from contract_utils import (
    executing_code_bundle,
    file_binding,
    read_json,
    require,
    require_file_binding,
    write_json_atomic,
)
from run_greekmmlu_evaluator import ALL_UPDATES
from validate_greekmmlu_sentinels import choice_nll

SCALES = ("8b", "1p5b")
UPDATES = tuple(sorted(ALL_UPDATES))
NATIVE_UPDATES = (0, 2261, 2618, 3218, 3456, 3694)
PANEL_NAMES = {
    "code",
    "de",
    "english",
    "greek_phd",
    "historical_polytonic",
    "hplt",
    "math",
    "neutral_external_modern_greek",
    "non_hplt",
    "old_greek",
    "openarchives",
    "ru",
    "zh",
}
METRIC_FAMILIES = {
    "hplt_bpb": ("hplt",),
    "openarchives_macro_bpb": ("openarchives", "greek_phd", "historical_polytonic"),
    "foreign_replay_macro_bpb": ("english", "de", "ru", "zh", "code", "math"),
    "old_greek_bpb": ("old_greek",),
    "neutral_external_modern_greek_bpb": ("neutral_external_modern_greek",),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            require(isinstance(row, dict), f"{path}:{line_number}: expected object")
            rows.append(row)
    return rows


def percentile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability, method="linear"))


def interval(values: np.ndarray, confidence: float) -> list[float]:
    tail = (1.0 - confidence) / 2.0
    return [percentile(values, tail), percentile(values, 1.0 - tail)]


def decision_same_sign(
    left: np.ndarray,
    right: np.ndarray,
    *,
    point_left: float | None = None,
    point_right: float | None = None,
) -> dict[str, Any]:
    left_ci = interval(left, 0.95)
    right_ci = interval(right, 0.95)
    observed_left = float(np.mean(left)) if point_left is None else float(point_left)
    observed_right = float(np.mean(right)) if point_right is None else float(point_right)
    left_nonzero = left_ci[0] > 0 or left_ci[1] < 0
    right_nonzero = right_ci[0] > 0 or right_ci[1] < 0
    same_sign = observed_left * observed_right > 0
    if left_nonzero and right_nonzero and same_sign:
        decision = "pass"
    elif left_nonzero and right_nonzero and not same_sign:
        decision = "fail"
    else:
        decision = "inconclusive"
    return {
        "8b": {"point": observed_left, "ci95": left_ci},
        "1p5b": {"point": observed_right, "ci95": right_ci},
        "decision": decision,
    }


def ols_slopes(values: np.ndarray, updates: list[int]) -> np.ndarray:
    x = np.asarray(updates, dtype=np.float64)
    x = x - x.mean()
    denominator = float(np.dot(x, x))
    require(denominator > 0, "invalid OLS update window")
    return (values @ x) / denominator


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return result


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = ranks(left)
    right_rank = ranks(right)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def spearman_gate(
    left: np.ndarray,
    right: np.ndarray,
    *,
    point_left: np.ndarray,
    point_right: np.ndarray,
) -> dict[str, Any]:
    require(left.shape == right.shape and left.ndim == 2, "Spearman bootstrap shape drift")
    require(
        point_left.shape == point_right.shape == left.shape[1:],
        "Spearman point shape drift",
    )
    correlations = np.asarray(
        [spearman(np.diff(left[row]), np.diff(right[row])) for row in range(left.shape[0])],
        dtype=np.float64,
    )
    point = spearman(np.diff(point_left), np.diff(point_right))
    ci90 = interval(correlations, 0.90)
    if point >= 0.45 and ci90[0] > 0:
        decision = "pass"
    elif ci90[1] < 0:
        decision = "fail"
    else:
        decision = "inconclusive"
    return {"point": point, "ci90": ci90, "decision": decision}


def collect_source_panels(run_root: Path, scale: str) -> tuple[dict[str, Any], dict[int, dict[str, Path]]]:
    authority: dict[str, Any] = {}
    documents: dict[int, dict[str, Path]] = {}
    for update in UPDATES:
        result_path, result = completed_result(
            run_root,
            evaluator_id="offline_panels",
            iteration=update,
            schema="apertus_hard_h_to_g_offline_panels_evaluation_v1",
            scale=scale,
        )
        panels = result.get("panels")
        require(isinstance(panels, list), f"{scale}@{update}: panel inventory missing")
        by_name = {str(row["name"]): row for row in panels}
        require(set(by_name) == PANEL_NAMES and len(by_name) == 13, f"{scale}@{update}: panel set drift")
        bound: dict[str, Any] = {}
        documents[update] = {}
        for name in sorted(by_name):
            row = by_name[name]
            receipt_path = require_file_binding(row["receipt"])
            document_path = require_file_binding(row["documents"])
            receipt = read_json(receipt_path)
            require(
                receipt.get("schema_version") == "apertus_per_document_validation_v1"
                and receipt.get("status") == "completed"
                and receipt.get("output", {}).get("rows") == len(read_jsonl(document_path)),
                f"{scale}@{update}/{name}: per-document receipt drift",
            )
            documents[update][name] = document_path
            bound[name] = {
                "receipt": file_binding(receipt_path),
                "documents": file_binding(document_path),
                "aggregate": row["aggregate"],
            }
        authority[str(update)] = {
            "canonical_result": file_binding(result_path),
            "panels": bound,
        }
    return authority, documents


def clustered_rows(path: Path) -> dict[str, tuple[float, int]]:
    clusters: dict[str, list[float | int]] = {}
    for row in read_jsonl(path):
        cluster = str(row.get("cluster_id") or row.get("doc_id") or "")
        require(cluster, f"document resampling identity missing: {path}")
        current = clusters.setdefault(cluster, [0.0, 0])
        current[0] = float(current[0]) + float(row["nll_numerator_nats"])
        current[1] = int(current[1]) + int(row["utf8_bytes"])
    require(clusters and all(int(value[1]) > 0 for value in clusters.values()), f"empty cluster panel: {path}")
    return {key: (float(value[0]), int(value[1])) for key, value in clusters.items()}


def family_arrays(
    documents: dict[str, dict[int, dict[str, Path]]], family: str
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    clusters_by_scale: dict[str, list[dict[str, tuple[float, int]]]] = {}
    keys: list[str] | None = None
    for scale in SCALES:
        rows = [clustered_rows(documents[scale][update][family]) for update in UPDATES]
        current_keys = sorted(rows[0])
        require(
            all(sorted(value) == current_keys for value in rows),
            f"{scale}/{family}: cluster identities change by checkpoint",
        )
        if keys is None:
            keys = current_keys
        else:
            require(keys == current_keys, f"{family}: cross-scale cluster identity drift")
        clusters_by_scale[scale] = rows
    require(keys is not None and keys, f"{family}: no clusters")
    numerators: dict[str, np.ndarray] = {}
    byte_counts: dict[str, np.ndarray] = {}
    for scale in SCALES:
        numerators[scale] = np.asarray(
            [[rows[key][0] for key in keys] for rows in clusters_by_scale[scale]],
            dtype=np.float64,
        )
        byte_counts[scale] = np.asarray(
            [[rows[key][1] for key in keys] for rows in clusters_by_scale[scale]],
            dtype=np.float64,
        )
    return numerators, byte_counts, keys


def metric_trajectory(
    documents: dict[str, dict[int, dict[str, Path]]],
    families: tuple[str, ...],
    *,
    replicates: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    points = {scale: np.zeros(len(UPDATES), dtype=np.float64) for scale in SCALES}
    boot = {
        scale: np.zeros((replicates, len(UPDATES)), dtype=np.float64) for scale in SCALES
    }
    for family_index, family in enumerate(families):
        numerators, byte_counts, keys = family_arrays(documents, family)
        for scale in SCALES:
            points[scale] += (
                numerators[scale].sum(axis=1)
                / (math.log(2.0) * byte_counts[scale].sum(axis=1))
                / len(families)
            )
        rng = np.random.default_rng(seed + family_index * 1_000_003)
        batch = 64
        probability = np.full(len(keys), 1.0 / len(keys), dtype=np.float64)
        for start in range(0, replicates, batch):
            width = min(batch, replicates - start)
            counts = rng.multinomial(len(keys), probability, size=width).astype(np.float64)
            for scale in SCALES:
                sampled_numerator = counts @ numerators[scale].T
                sampled_bytes = counts @ byte_counts[scale].T
                boot[scale][start : start + width] += (
                    sampled_numerator / (math.log(2.0) * sampled_bytes) / len(families)
                )
    return points, boot


def collect_greekmmlu(run_root: Path, scale: str) -> tuple[dict[str, Any], dict[int, Path]]:
    plateau_path, plateau = completed_result(
        run_root,
        evaluator_id="greekmmlu_plateau_confirmation",
        iteration=3694,
        schema="apertus_hard_h_to_g_greekmmlu_plateau_evaluation_v1",
        scale=scale,
    )
    source = plateau.get("source_evaluations")
    require(isinstance(source, dict) and sorted(map(int, source)) == list(UPDATES), f"{scale}: plateau source set drift")
    predictions: dict[int, Path] = {}
    bound: dict[str, Any] = {}
    expected_view = plateau.get("trajectory_view")
    joint_path = require_file_binding(plateau["joint_calibration_authority"])
    joint = read_json(joint_path)
    require(
        joint.get("schema_version")
        == "apertus_greekmmlu_sentinel_calibration_authority_v1"
        and joint.get("status") == "passed"
        and plateau.get("cross_scale_trajectory") == joint.get("cross_scale_trajectory"),
        f"{scale}: joint GreekMMLU authority drift",
    )
    for raw_update, row in source.items():
        update = int(raw_update)
        require(isinstance(row, dict) and row.get("view") == expected_view, f"{scale}@{update}: trajectory view drift")
        prediction_path = require_file_binding(row["predictions"])
        predictions[update] = prediction_path
        bound[str(update)] = row
    members = plateau.get("plateau_members")
    require(isinstance(members, list) and members, f"{scale}: plateau set missing")
    confirmation = plateau.get("full_panel_confirmation")
    if expected_view != "full_clean" and members != [3218]:
        require(
            isinstance(confirmation, dict)
            and int(confirmation.get("target_iteration", -1)) == int(members[0]),
            f"{scale}: required full plateau confirmation missing",
        )
        require_file_binding(confirmation["evaluation"])
    return (
        {
            "plateau_result": file_binding(plateau_path),
            "trajectory_view": expected_view,
            "joint_calibration_authority": file_binding(joint_path),
            "cross_scale_trajectory": plateau["cross_scale_trajectory"],
            "plateau_members": members,
            "full_panel_confirmation": confirmation,
            "source_evaluations": bound,
        },
        predictions,
    )


def greek_arrays(predictions: dict[str, dict[int, Path]], replicates: int, seed: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    values: dict[str, np.ndarray] = {}
    accuracy: dict[str, np.ndarray] = {}
    ids: list[str] | None = None
    for scale in SCALES:
        by_update = []
        correct_by_update = []
        for update in UPDATES:
            rows = read_jsonl(predictions[scale][update])
            current_ids = [str(row["example_id"]) for row in rows]
            require(len(current_ids) == len(set(current_ids)), f"{scale}@{update}: duplicate GreekMMLU ids")
            order = np.argsort(np.asarray(current_ids, dtype=object), kind="mergesort")
            sorted_ids = [current_ids[index] for index in order]
            if ids is None:
                ids = sorted_ids
            else:
                require(ids == sorted_ids, f"{scale}@{update}: GreekMMLU id drift")
            by_update.append([choice_nll(rows[index]) for index in order])
            correct_by_update.append([float(bool(rows[index]["correct"])) for index in order])
        values[scale] = np.asarray(by_update, dtype=np.float64)
        accuracy[scale] = np.asarray(correct_by_update, dtype=np.float64)
    require(ids is not None and ids, "GreekMMLU trajectory is empty")
    boot = {
        scale: np.zeros((replicates, len(UPDATES)), dtype=np.float64) for scale in SCALES
    }
    rng = np.random.default_rng(seed)
    probability = np.full(len(ids), 1.0 / len(ids), dtype=np.float64)
    for start in range(0, replicates, 64):
        width = min(64, replicates - start)
        counts = rng.multinomial(len(ids), probability, size=width).astype(np.float64)
        for scale in SCALES:
            boot[scale][start : start + width] = counts @ values[scale].T / len(ids)
    points = {scale: values[scale].mean(axis=1) for scale in SCALES}
    accuracies = {scale: accuracy[scale].mean(axis=1) for scale in SCALES}
    return points, boot, accuracies


def collect_native(run_root: Path, scale: str) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for update in NATIVE_UPDATES:
        result_path, result = completed_result(
            run_root,
            evaluator_id="native_greek_suite",
            iteration=update,
            schema="apertus_hard_h_to_g_native_suite_evaluation_v1",
            scale=scale,
        )
        matrix_path = require_file_binding(result["matrix"])
        filtered_path = require_file_binding(result["contamination_filtered"])
        exclusions_path = require_file_binding(result["exclusions"])
        matrix = read_json(matrix_path)
        filtered = read_json(filtered_path)
        require(
            matrix.get("status") == "completed"
            and len(matrix.get("checkpoint_receipts", [])) == 1
            and filtered.get("status") == "passed"
            and len(filtered.get("checkpoints", [])) == 1,
            f"{scale}@{update}: native-suite authority drift",
        )
        results[str(update)] = {
            "canonical_result": file_binding(result_path),
            "matrix": file_binding(matrix_path),
            "contamination_filtered": file_binding(filtered_path),
            "exclusions": file_binding(exclusions_path),
        }
    return results


def classify_margin(
    values: np.ndarray, margin: float, *, point: float | None = None
) -> dict[str, Any]:
    ci95 = interval(values, 0.95)
    if ci95[0] > margin:
        decision = "material_regression"
    elif ci95[1] <= margin:
        decision = "no_material_regression"
    else:
        decision = "inconclusive"
    return {
        "point": float(np.mean(values)) if point is None else float(point),
        "ci95": ci95,
        "margin": margin,
        "class": decision,
    }


def extension_decision(
    adaptation: np.ndarray,
    retention: dict[str, np.ndarray],
    margins: dict[str, float],
    *,
    adaptation_point: float,
    retention_points: dict[str, float],
) -> dict[str, Any]:
    adaptation_ci = interval(adaptation, 0.95)
    retention_rows = {
        metric: classify_margin(
            values, margins[metric], point=retention_points[metric]
        )
        for metric, values in retention.items()
    }
    adaptation_pass = adaptation_ci[0] > margins["openarchives_macro_bpb"]
    adaptation_fail = adaptation_ci[1] <= margins["openarchives_macro_bpb"]
    retention_pass = all(row["ci95"][1] <= row["margin"] for row in retention_rows.values())
    retention_fail = any(row["ci95"][0] > row["margin"] for row in retention_rows.values())
    if adaptation_pass and retention_pass:
        decision = "pass"
    elif adaptation_fail or retention_fail:
        decision = "fail"
    else:
        decision = "inconclusive"
    return {
        "decision": decision,
        "openarchives_improvement": {
            "point": float(adaptation_point),
            "ci95": adaptation_ci,
            "margin": margins["openarchives_macro_bpb"],
        },
        "retention": retention_rows,
        "retention_noninferior": retention_pass,
    }


def summarize_points(points: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {
        scale: {str(update): float(values[index]) for index, update in enumerate(UPDATES)}
        for scale, values in points.items()
    }


def make_selection(
    *,
    source_boot: dict[str, dict[str, np.ndarray]],
    source_points: dict[str, dict[str, np.ndarray]],
    greek_boot: dict[str, np.ndarray],
    greek_points: dict[str, np.ndarray],
    plateaus: dict[str, list[int]],
    legacy: dict[str, Any],
) -> dict[str, Any]:
    all_boot = {**source_boot, "greekmmlu_choice_nll": greek_boot}
    all_points = {**source_points, "greekmmlu_choice_nll": greek_points}
    pre = [UPDATES.index(value) for value in (0, 238, 476, 714, 952, 1190, 1428, 1666, 1904, 2142, 2261)]
    post = [UPDATES.index(value) for value in (2261, 2380, 2618, 2856, 3094, 3218)]
    slopes: dict[str, Any] = {}
    immediate: dict[str, Any] = {}
    for metric in (
        "greekmmlu_choice_nll",
        "balanced_greek_bpb",
        "hplt_bpb",
        "openarchives_macro_bpb",
        "foreign_replay_macro_bpb",
        "old_greek_bpb",
    ):
        metric_boot = all_boot[metric]
        pre_updates = [UPDATES[index] for index in pre]
        post_updates = [UPDATES[index] for index in post]
        slopes[metric] = {
            "pre_switch": decision_same_sign(
                ols_slopes(metric_boot["8b"][:, pre], pre_updates),
                ols_slopes(metric_boot["1p5b"][:, pre], pre_updates),
                point_left=ols_slopes(
                    all_points[metric]["8b"][None, pre], pre_updates
                )[0],
                point_right=ols_slopes(
                    all_points[metric]["1p5b"][None, pre], pre_updates
                )[0],
            ),
            "post_switch": decision_same_sign(
                ols_slopes(metric_boot["8b"][:, post], post_updates),
                ols_slopes(metric_boot["1p5b"][:, post], post_updates),
                point_left=ols_slopes(
                    all_points[metric]["8b"][None, post], post_updates
                )[0],
                point_right=ols_slopes(
                    all_points[metric]["1p5b"][None, post], post_updates
                )[0],
            ),
        }
        left, right = UPDATES.index(2261), UPDATES.index(2380)
        immediate[metric] = decision_same_sign(
            metric_boot["8b"][:, right] - metric_boot["8b"][:, left],
            metric_boot["1p5b"][:, right] - metric_boot["1p5b"][:, left],
            point_left=(
                all_points[metric]["8b"][right]
                - all_points[metric]["8b"][left]
            ),
            point_right=(
                all_points[metric]["1p5b"][right]
                - all_points[metric]["1p5b"][left]
            ),
        )

    correlation_end = UPDATES.index(3218) + 1
    correlation = {
        metric: spearman_gate(
            all_boot[metric]["8b"][:, :correlation_end],
            all_boot[metric]["1p5b"][:, :correlation_end],
            point_left=all_points[metric]["8b"][:correlation_end],
            point_right=all_points[metric]["1p5b"][:correlation_end],
        )
        for metric in ("greekmmlu_choice_nll", "balanced_greek_bpb")
    }
    forgetting: dict[str, Any] = {}
    forgetting_decisions = []
    calibration_indices = [UPDATES.index(value) for value in (0, 238, 476, 714)]
    final_index = UPDATES.index(3218)
    for metric in ("hplt_bpb", "foreign_replay_macro_bpb", "old_greek_bpb"):
        per_scale = {}
        classes = []
        for scale in SCALES:
            boot = all_boot[metric][scale]
            margin = 2.0 * statistics.median(
                float(np.std(boot[:, index], ddof=0)) for index in calibration_indices
            )
            values = boot[:, final_index] - np.min(boot[:, : final_index + 1], axis=1)
            point_values = all_points[metric][scale]
            row = classify_margin(
                values,
                margin,
                point=(
                    point_values[final_index]
                    - np.min(point_values[: final_index + 1])
                ),
            )
            per_scale[scale] = row
            classes.append(row["class"])
        if "inconclusive" in classes:
            decision = "inconclusive"
        elif classes[0] == classes[1]:
            decision = "pass"
        else:
            decision = "fail"
        per_scale["cross_scale_decision"] = decision
        forgetting[metric] = per_scale
        forgetting_decisions.append(decision)
    plateau_overlap = sorted(set(plateaus["8b"]) & set(plateaus["1p5b"]))
    plateau_decision = "pass" if plateau_overlap else "fail"
    primary = [
        *(row[window]["decision"] for row in slopes.values() for window in ("pre_switch", "post_switch")),
        *(row["decision"] for row in immediate.values()),
        plateau_decision,
        *forgetting_decisions,
    ]
    secondary = [row["decision"] for row in correlation.values()]
    if all(value == "pass" for value in primary) and "pass" in secondary and "fail" not in secondary:
        goal_b = "pass"
    elif "fail" in primary or secondary.count("fail") == 2:
        goal_b = "fail"
    else:
        goal_b = "inconclusive"

    extension_indices = [UPDATES.index(value) for value in (3218, 3456, 3694)]
    margin_indices = [UPDATES.index(value) for value in (2618, 2856, 3094, 3218)]
    goal_c: dict[str, Any] = {}
    for scale in SCALES:
        margins = {
            metric: 2.0 * statistics.median(
                float(np.std(all_boot[metric][scale][:, index], ddof=0)) for index in margin_indices
            )
            for metric in (
                "openarchives_macro_bpb",
                "hplt_bpb",
                "foreign_replay_macro_bpb",
                "old_greek_bpb",
                "neutral_external_modern_greek_bpb",
            )
        }
        intervals = {}
        for label, start, end in (
            ("3218_to_3456", extension_indices[0], extension_indices[1]),
            ("3456_to_3694", extension_indices[1], extension_indices[2]),
            ("3218_to_3694", extension_indices[0], extension_indices[2]),
        ):
            adaptation = (
                all_boot["openarchives_macro_bpb"][scale][:, start]
                - all_boot["openarchives_macro_bpb"][scale][:, end]
            )
            retention = {
                metric: all_boot[metric][scale][:, end] - all_boot[metric][scale][:, start]
                for metric in (
                    "hplt_bpb",
                    "foreign_replay_macro_bpb",
                    "old_greek_bpb",
                    "neutral_external_modern_greek_bpb",
                )
            }
            intervals[label] = extension_decision(
                adaptation,
                retention,
                margins,
                adaptation_point=(
                    all_points["openarchives_macro_bpb"][scale][start]
                    - all_points["openarchives_macro_bpb"][scale][end]
                ),
                retention_points={
                    metric: (
                        all_points[metric][scale][end]
                        - all_points[metric][scale][start]
                    )
                    for metric in retention
                },
            )
        saturation = (
            intervals["3218_to_3456"]["decision"] == "pass"
            and intervals["3456_to_3694"]["openarchives_improvement"]["ci95"][1]
            <= margins["openarchives_macro_bpb"]
            and intervals["3456_to_3694"]["retention_noninferior"]
        )
        goal_c[scale] = {"margins": margins, "intervals": intervals, "saturation": saturation}

    return {
        "goal_a": legacy,
        "goal_b": {
            "decision": goal_b,
            "slope_direction": slopes,
            "immediate_switch": immediate,
            "first_difference_spearman": correlation,
            "forgetting": forgetting,
            "plateau": {
                "8b": plateaus["8b"],
                "1p5b": plateaus["1p5b"],
                "intersection": plateau_overlap,
                "decision": plateau_decision,
            },
        },
        "goal_c": goal_c,
        "point_trajectories": {
            metric: summarize_points(value) for metric, value in all_points.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--8b-run-root", type=Path, required=True)
    parser.add_argument("--1p5b-run-root", type=Path, required=True)
    parser.add_argument("--statistical-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_dir.exists(), f"immutable final evidence exists: {args.output_dir}")
    statistics_contract = read_json(args.statistical_contract)
    require(
        statistics_contract.get("schema_version") == "apertus_hard_h_to_g_statistical_decisions_v2"
        and statistics_contract.get("status") == "frozen"
        and statistics_contract.get("bootstrap") == {"replicates": 10_000, "seed": 20260814},
        "statistical decision contract drift",
    )
    expected_aggregation = {
        key: list(value) for key, value in METRIC_FAMILIES.items()
    }
    frozen_aggregation = statistics_contract["decisions"]["goal_b"]["source_panel_aggregation"]
    require(
        all(frozen_aggregation[key] == value for key, value in expected_aggregation.items()),
        "source metric family contract drift",
    )
    run_roots = {"8b": args.__dict__["8b_run_root"], "1p5b": args.__dict__["1p5b_run_root"]}
    bundle = executing_code_bundle()
    args.output_dir.mkdir(parents=True)

    source_rows: dict[str, Any] = {}
    source_documents: dict[str, dict[int, dict[str, Path]]] = {}
    for scale in SCALES:
        source_rows[scale], source_documents[scale] = collect_source_panels(run_roots[scale], scale)
    source_path = args.output_dir / "source_panel_authority.json"
    write_json_atomic(
        source_path,
        {
            "schema_version": "apertus_hard_h_to_g_source_panel_evaluation_authority_v1",
            "status": "completed",
            "executing_code_bundle": bundle,
            "statistical_contract": file_binding(args.statistical_contract),
            "scales": source_rows,
            "updates": list(UPDATES),
            "panel_names": sorted(PANEL_NAMES),
            "all_required_source_panel_receipts_present": True,
        },
    )

    greek_rows: dict[str, Any] = {}
    greek_predictions: dict[str, dict[int, Path]] = {}
    for scale in SCALES:
        greek_rows[scale], greek_predictions[scale] = collect_greekmmlu(run_roots[scale], scale)
    require(
        greek_rows["8b"]["trajectory_view"] == greek_rows["1p5b"]["trajectory_view"]
        and greek_rows["8b"]["cross_scale_trajectory"]
        == greek_rows["1p5b"]["cross_scale_trajectory"]
        and greek_rows["8b"]["joint_calibration_authority"]
        == greek_rows["1p5b"]["joint_calibration_authority"],
        "cross-scale GreekMMLU trajectory panel drift",
    )
    legacy_path, legacy_result = completed_result(
        run_roots["8b"],
        evaluator_id="legacy_public_greekmmlu",
        iteration=3218,
        schema="apertus_hard_h_to_g_legacy_public_evaluation_v1",
        scale="8b",
    )
    compatibility_path = require_file_binding(legacy_result["compatibility_result"])
    compatibility = read_json(compatibility_path)
    require(
        compatibility.get("schema_version") == "apertus_legacy_public_greekmmlu_result_v1"
        and compatibility.get("status") == "completed"
        and compatibility.get("decision") in {"pass", "fail", "inconclusive"},
        "legacy compatibility evidence drift",
    )
    greek_path = args.output_dir / "greekmmlu_authority.json"
    write_json_atomic(
        greek_path,
        {
            "schema_version": "apertus_hard_h_to_g_greekmmlu_evaluation_authority_v1",
            "status": "completed",
            "executing_code_bundle": bundle,
            "statistical_contract": file_binding(args.statistical_contract),
            "scales": greek_rows,
            "legacy_8b_update_3218": {
                "canonical_result": file_binding(legacy_path),
                "compatibility_result": file_binding(compatibility_path),
                "decision": compatibility["decision"],
            },
            "all_required_greekmmlu_receipts_present": True,
        },
    )

    native_rows = {scale: collect_native(run_roots[scale], scale) for scale in SCALES}
    native_path = args.output_dir / "native_suite_authority.json"
    write_json_atomic(
        native_path,
        {
            "schema_version": "apertus_hard_h_to_g_native_suite_evaluation_authority_v1",
            "status": "completed",
            "executing_code_bundle": bundle,
            "scales": native_rows,
            "updates": list(NATIVE_UPDATES),
            "all_required_native_suite_receipts_present": True,
        },
    )

    source_points: dict[str, dict[str, np.ndarray]] = {}
    source_boot: dict[str, dict[str, np.ndarray]] = {}
    for metric_index, (metric, families) in enumerate(METRIC_FAMILIES.items()):
        source_points[metric], source_boot[metric] = metric_trajectory(
            source_documents,
            families,
            replicates=10_000,
            seed=20260814 + metric_index * 10_000_019,
        )
    source_points["balanced_greek_bpb"] = {
        scale: 0.5 * source_points["hplt_bpb"][scale]
        + 0.5 * source_points["openarchives_macro_bpb"][scale]
        for scale in SCALES
    }
    source_boot["balanced_greek_bpb"] = {
        scale: 0.5 * source_boot["hplt_bpb"][scale]
        + 0.5 * source_boot["openarchives_macro_bpb"][scale]
        for scale in SCALES
    }
    greek_points, greek_boot, greek_accuracy = greek_arrays(
        greek_predictions, 10_000, 20260814
    )
    selection = make_selection(
        source_boot=source_boot,
        source_points=source_points,
        greek_boot=greek_boot,
        greek_points=greek_points,
        plateaus={scale: greek_rows[scale]["plateau_members"] for scale in SCALES},
        legacy={
            "decision": compatibility["decision"],
            "observed": compatibility["observed"],
            "reference": compatibility["reference"],
        },
    )
    selection["greekmmlu_accuracy_trajectories"] = summarize_points(greek_accuracy)
    selection_path = args.output_dir / "selection_authorization.json"
    write_json_atomic(
        selection_path,
        {
            "schema_version": "apertus_hard_h_to_g_selection_authorization_v1",
            "status": "completed",
            "executing_code_bundle": bundle,
            "statistical_contract": file_binding(args.statistical_contract),
            "source_panel_authority": file_binding(source_path),
            "greekmmlu_authority": file_binding(greek_path),
            "native_suite_authority": file_binding(native_path),
            "bootstrap": {
                "replicates": 10_000,
                "seed": 20260814,
                "scope": "evaluation_panel_sampling_uncertainty_conditional_on_one_realized_training_run_per_scale",
                "source_unit": "paired_document_cluster",
                "greekmmlu_unit": "paired_question",
            },
            "selection_authorized": True,
            "authorization_scope": (
                "authorizes publication of the predeclared statistical decisions only; "
                "does not authorize a training continuation or a scientific winner when "
                "a decision is fail or inconclusive"
            ),
            "decisions": selection,
        },
    )
    print(selection_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
