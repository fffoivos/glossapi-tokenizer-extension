#!/usr/bin/env python3
"""Measure exact GreekMMLU response trajectories and wrong-cell displacement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested_accuracy_baseline(
    states: np.ndarray,
    labels: np.ndarray,
    initial_nll: np.ndarray,
    example_ids: np.ndarray,
) -> np.ndarray:
    """Match every group accuracy using a frozen iteration-0 difficulty order."""
    baseline = np.zeros_like(states, dtype=np.bool_)
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        order = np.lexsort((example_ids[indices], initial_nll[indices]))
        ranked = indices[order]
        counts = states[:, indices].sum(axis=1).astype(int)
        for t, count in enumerate(counts):
            baseline[t, ranked[:count]] = True
    return baseline


def observed_displacement(signal: np.ndarray, wrong: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    checkpoints = signal.shape[0]
    names, codes = np.unique(labels, return_inverse=True)
    minimum_side = max(3, int(math.ceil(checkpoints / 4)))
    splits = np.arange(minimum_side, checkpoints - minimum_side + 1, dtype=int)
    values = np.zeros((len(splits), len(names)), dtype=np.float64)
    capacity = np.zeros_like(values)
    net_wrong = np.zeros_like(values)
    cumulative = np.cumsum(signal, axis=0, dtype=np.float64)
    total = cumulative[-1]
    cumulative_wrong = np.cumsum(wrong, axis=0, dtype=np.float64)
    total_wrong = cumulative_wrong[-1]
    for row_index, split in enumerate(splits):
        delta = (total - cumulative[split - 1]) / (checkpoints - split) - cumulative[split - 1] / split
        values[row_index] = 0.5 * np.bincount(codes, weights=np.abs(delta), minlength=len(names))
        before_wrong = cumulative_wrong[split - 1] / split
        after_wrong = (total_wrong - cumulative_wrong[split - 1]) / (checkpoints - split)
        before_mass = np.bincount(codes, weights=before_wrong, minlength=len(names))
        after_mass = np.bincount(codes, weights=after_wrong, minlength=len(names))
        capacity[row_index] = np.minimum(before_mass, after_mass)
        net_wrong[row_index] = after_mass - before_mass
    peak_indices = np.argmax(values, axis=0)
    return {
        "names": names,
        "codes": codes,
        "splits": splits,
        "values": values,
        "capacity": capacity,
        "net_wrong": net_wrong,
        "peak_indices": peak_indices,
    }


def permutation_peaks(
    signal: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    replicates: int,
    seed: int,
) -> np.ndarray:
    names, codes = np.unique(labels, return_inverse=True)
    checkpoints = signal.shape[0]
    rng = np.random.default_rng(seed)
    peaks = np.zeros((replicates, len(names)), dtype=np.float64)
    for replicate in range(replicates):
        permuted = signal[rng.permutation(checkpoints)]
        cumulative = np.cumsum(permuted, axis=0, dtype=np.float64)
        total = cumulative[-1]
        current = np.zeros(len(names), dtype=np.float64)
        for split in splits:
            delta = (total - cumulative[split - 1]) / (checkpoints - split) - cumulative[split - 1] / split
            displacement = 0.5 * np.bincount(codes, weights=np.abs(delta), minlength=len(names))
            current = np.maximum(current, displacement)
        peaks[replicate] = current
    return peaks


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    count = len(p_values)
    order = np.argsort(p_values, kind="stable")
    adjusted = np.empty(count, dtype=np.float64)
    running = 1.0
    for reverse_rank in range(count - 1, -1, -1):
        index = order[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, float(p_values[index]) * count / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def trajectory_for_group(
    indices: np.ndarray,
    states: np.ndarray,
    choices: np.ndarray,
    choice_nll: np.ndarray,
    correct_answer_bits: np.ndarray,
    correct_answer_bytes: np.ndarray,
    iterations: np.ndarray,
) -> dict[str, Any]:
    group_states = states[:, indices]
    group_choices = choices[:, indices]
    trajectories = []
    byte_denominator = float(correct_answer_bytes[indices].sum())
    if byte_denominator <= 0:
        raise ValueError("category has no positive correct-answer UTF-8 bytes")
    for t, iteration in enumerate(iterations):
        current = group_states[t]
        row: dict[str, Any] = {
            "iteration": int(iteration),
            "n": int(len(indices)),
            "correct": int(current.sum()),
            "accuracy": float(current.mean()),
            "choice_nll": float(choice_nll[t, indices].mean()),
            "correct_answer_bpb": float(correct_answer_bits[t, indices].sum() / byte_denominator),
        }
        if t == 0:
            row["vs_previous"] = None
        else:
            previous = group_states[t - 1]
            row["vs_previous"] = {
                "newly_correct": int((~previous & current).sum()),
                "newly_wrong": int((previous & ~current).sum()),
                "correctness_flips": int((previous != current).sum()),
                "answer_choice_flips": int((group_choices[t - 1] != group_choices[t]).sum()),
            }
        trajectories.append(row)

    accuracy = group_states.mean(axis=1)
    nll = choice_nll[:, indices].mean(axis=1)
    best_accuracy_index = int(np.argmax(accuracy))
    best_nll_index = int(np.argmin(nll))
    global_best_index = int(np.where(iterations == 9536)[0][0]) if 9536 in iterations else best_accuracy_index
    final = group_states[-1]
    global_best = group_states[global_best_index]
    ever_correctness_changed = (group_states[1:] != group_states[:-1]).any(axis=0)
    ever_choice_changed = (group_choices[1:] != group_choices[:-1]).any(axis=0)
    return {
        "n": int(len(indices)),
        "trajectory": trajectories,
        "initial_accuracy": float(accuracy[0]),
        "best_accuracy": float(accuracy[best_accuracy_index]),
        "best_accuracy_iteration": int(iterations[best_accuracy_index]),
        "final_accuracy": float(accuracy[-1]),
        "final_minus_initial_accuracy": float(accuracy[-1] - accuracy[0]),
        "final_minus_best_accuracy": float(accuracy[-1] - accuracy[best_accuracy_index]),
        "best_choice_nll": float(nll[best_nll_index]),
        "best_choice_nll_iteration": int(iterations[best_nll_index]),
        "final_choice_nll": float(nll[-1]),
        "final_minus_best_choice_nll": float(nll[-1] - nll[best_nll_index]),
        "stable_correct": int(group_states.all(axis=0).sum()),
        "stable_wrong": int((~group_states).all(axis=0).sum()),
        "transient_correctness": int((~group_states.all(axis=0) & ~(~group_states).all(axis=0)).sum()),
        "ever_correctness_changed": int(ever_correctness_changed.sum()),
        "ever_answer_choice_changed": int(ever_choice_changed.sum()),
        "correctness_flip_events": int((group_states[1:] != group_states[:-1]).sum()),
        "answer_choice_flip_events": int((group_choices[1:] != group_choices[:-1]).sum()),
        "global_peak_to_final": {
            "reference_iteration": int(iterations[global_best_index]),
            "newly_correct": int((~global_best & final).sum()),
            "newly_wrong": int((global_best & ~final).sum()),
            "paired_replacements": int(min((~global_best & final).sum(), (global_best & ~final).sum())),
            "correct_set_churn": int((global_best != final).sum()),
            "net_correct": int(final.sum() - global_best.sum()),
        },
    }


def analyze_window_displacement(
    axis: str,
    labels: np.ndarray,
    states: np.ndarray,
    difficulty_nll: np.ndarray,
    example_ids: np.ndarray,
    iterations: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    baseline = nested_accuracy_baseline(states, labels, difficulty_nll, example_ids)
    for label in np.unique(labels):
        indices = labels == label
        if not np.array_equal(states[:, indices].sum(axis=1), baseline[:, indices].sum(axis=1)):
            raise AssertionError(f"{axis}/{label}: window accuracy baseline mismatch")
    wrong = (~states).astype(np.float64)
    signal = wrong - (~baseline).astype(np.float64)
    observed = observed_displacement(signal, wrong, labels)
    null = permutation_peaks(signal, labels, observed["splits"], replicates, seed)
    peak_indices = observed["peak_indices"]
    raw = observed["values"][peak_indices, np.arange(len(observed["names"]))]
    p_values = (1 + (null >= raw[None, :]).sum(axis=0)) / (replicates + 1)
    q_values = benjamini_hochberg(p_values.astype(float))
    rows = {}
    for group_index, label in enumerate(observed["names"]):
        peak_index = int(peak_indices[group_index])
        split = int(observed["splits"][peak_index])
        capacity = float(observed["capacity"][peak_index, group_index])
        rows[str(label)] = {
            "window_first_iteration": int(iterations[0]),
            "window_final_iteration": int(iterations[-1]),
            "window_checkpoints": int(len(iterations)),
            "raw_cells": float(raw[group_index]),
            "fraction_of_available_wrong_mass": float(raw[group_index] / capacity) if capacity else 0.0,
            "best_split_index": split,
            "past_last_iteration": int(iterations[split - 1]),
            "future_first_iteration": int(iterations[split]),
            "past_checkpoints": split,
            "future_checkpoints": int(len(iterations) - split),
            "net_wrong_cells_at_split": float(observed["net_wrong"][peak_index, group_index]),
            "null_median_cells": float(np.median(null[:, group_index])),
            "null_q95_cells": float(np.quantile(null[:, group_index], 0.95)),
            "null_q975_cells": float(np.quantile(null[:, group_index], 0.975)),
            "supported_above_q975_cells": float(max(0.0, raw[group_index] - np.quantile(null[:, group_index], 0.975))),
            "permutation_p_high": float(p_values[group_index]),
            "bh_q_value_within_axis": float(q_values[group_index]),
            "curve": [
                {"split_index": int(split_value), "cells": float(observed["values"][i, group_index])}
                for i, split_value in enumerate(observed["splits"])
            ],
        }
    return rows


def analyze_axis(
    axis: str,
    labels: np.ndarray,
    states: np.ndarray,
    choices: np.ndarray,
    choice_nll: np.ndarray,
    correct_answer_bits: np.ndarray,
    correct_answer_bytes: np.ndarray,
    example_ids: np.ndarray,
    iterations: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    baseline = nested_accuracy_baseline(states, labels, choice_nll[0], example_ids)
    for label in np.unique(labels):
        indices = labels == label
        if not np.array_equal(states[:, indices].sum(axis=1), baseline[:, indices].sum(axis=1)):
            raise AssertionError(f"{axis}/{label}: accuracy baseline mismatch")
    wrong = (~states).astype(np.float64)
    signal = wrong - (~baseline).astype(np.float64)
    observed = observed_displacement(signal, wrong, labels)
    null = permutation_peaks(signal, labels, observed["splits"], replicates, seed)
    peak_indices = observed["peak_indices"]
    raw = observed["values"][peak_indices, np.arange(len(observed["names"]))]
    p_values = (1 + (null >= raw[None, :]).sum(axis=0)) / (replicates + 1)
    q_values = benjamini_hochberg(p_values.astype(float))
    rows: dict[str, Any] = {}
    for group_index, label in enumerate(observed["names"]):
        indices = np.flatnonzero(labels == label)
        peak_index = int(peak_indices[group_index])
        split = int(observed["splits"][peak_index])
        capacity = float(observed["capacity"][peak_index, group_index])
        trajectory = trajectory_for_group(
            indices,
            states,
            choices,
            choice_nll,
            correct_answer_bits,
            correct_answer_bytes,
            iterations,
        )
        trajectory["displacement"] = {
            "raw_cells": float(raw[group_index]),
            "fraction_of_available_wrong_mass": float(raw[group_index] / capacity) if capacity else 0.0,
            "best_split_index": split,
            "past_last_iteration": int(iterations[split - 1]),
            "future_first_iteration": int(iterations[split]),
            "past_checkpoints": split,
            "future_checkpoints": int(len(iterations) - split),
            "net_wrong_cells_at_split": float(observed["net_wrong"][peak_index, group_index]),
            "null_median_cells": float(np.median(null[:, group_index])),
            "null_q95_cells": float(np.quantile(null[:, group_index], 0.95)),
            "null_q975_cells": float(np.quantile(null[:, group_index], 0.975)),
            "supported_above_q975_cells": float(max(0.0, raw[group_index] - np.quantile(null[:, group_index], 0.975))),
            "permutation_p_high": float(p_values[group_index]),
            "bh_q_value_within_axis": float(q_values[group_index]),
            "curve": [
                {"split_index": int(split_value), "cells": float(observed["values"][i, group_index])}
                for i, split_value in enumerate(observed["splits"])
            ],
        }
        rows[str(label)] = trajectory
    return {
        "axis": axis,
        "category_count": len(rows),
        "permutation_replicates": replicates,
        "categories": rows,
    }


def weighted_correlation(rows: list[dict[str, Any]], x_key: str, y_key: str) -> float | None:
    if len(rows) < 2:
        return None
    weights = np.asarray([row["n"] for row in rows], dtype=float)
    x = np.asarray([row[x_key] for row in rows], dtype=float)
    y = np.asarray([row[y_key] for row in rows], dtype=float)
    x_mean = float(np.average(x, weights=weights))
    y_mean = float(np.average(y, weights=weights))
    covariance = float(np.average((x - x_mean) * (y - y_mean), weights=weights))
    variance_x = float(np.average((x - x_mean) ** 2, weights=weights))
    variance_y = float(np.average((y - y_mean) ** 2, weights=weights))
    if variance_x <= 0 or variance_y <= 0:
        return None
    return covariance / math.sqrt(variance_x * variance_y)


def build_findings(axes: dict[str, Any]) -> dict[str, Any]:
    whole = axes["whole"]["categories"]["All clean GreekMMLU"]
    subjects = axes["subject"]["categories"]
    levels = axes["level"]["categories"]
    subject_rows = []
    for name, row in subjects.items():
        subject_rows.append({
            "name": name,
            "n": row["n"],
            "displacement_share": row["displacement"]["fraction_of_available_wrong_mass"],
            "displacement_cells": row["displacement"]["raw_cells"],
            "q": row["displacement"]["bh_q_value_within_axis"],
            "post_peak_displacement_share": row["post_global_peak_displacement"]["fraction_of_available_wrong_mass"],
            "post_peak_displacement_cells": row["post_global_peak_displacement"]["raw_cells"],
            "post_peak_q": row["post_global_peak_displacement"]["bh_q_value_within_axis"],
            "final_minus_best_accuracy": row["final_minus_best_accuracy"],
            "final_minus_best_choice_nll": row["final_minus_best_choice_nll"],
        })
    stable_subjects = [row for row in subject_rows if row["n"] >= 100]
    significant = [row for row in subject_rows if row["q"] < 0.05]
    post_peak_significant = [row for row in subject_rows if row["post_peak_q"] < 0.05]
    return {
        "whole": {
            "raw_displacement_cells": whole["displacement"]["raw_cells"],
            "displacement_share": whole["displacement"]["fraction_of_available_wrong_mass"],
            "permutation_p": whole["displacement"]["permutation_p_high"],
            "best_split": [whole["displacement"]["past_last_iteration"], whole["displacement"]["future_first_iteration"]],
            "ever_correctness_changed_fraction": whole["ever_correctness_changed"] / whole["n"],
            "ever_answer_choice_changed_fraction": whole["ever_answer_choice_changed"] / whole["n"],
            "accuracy_best_to_final_pp": 100 * whole["final_minus_best_accuracy"],
            "nll_best_to_final": whole["final_minus_best_choice_nll"],
            "post_peak_raw_displacement_cells": whole["post_global_peak_displacement"]["raw_cells"],
            "post_peak_displacement_share": whole["post_global_peak_displacement"]["fraction_of_available_wrong_mass"],
            "post_peak_permutation_p": whole["post_global_peak_displacement"]["permutation_p_high"],
            "post_peak_best_split": [whole["post_global_peak_displacement"]["past_last_iteration"], whole["post_global_peak_displacement"]["future_first_iteration"]],
            "peak_to_final_paired_replacements": whole["global_peak_to_final"]["paired_replacements"],
            "peak_to_final_net_correct": whole["global_peak_to_final"]["net_correct"],
        },
        "subjects_significant_after_bh": len(significant),
        "significant_subjects": sorted(significant, key=lambda row: row["q"]),
        "post_peak_subjects_significant_after_bh": len(post_peak_significant),
        "post_peak_significant_subjects": sorted(post_peak_significant, key=lambda row: row["post_peak_q"]),
        "largest_displacement_share_subjects_n_ge_100": sorted(stable_subjects, key=lambda row: row["displacement_share"], reverse=True)[:8],
        "largest_post_peak_displacement_share_subjects_n_ge_100": sorted(stable_subjects, key=lambda row: row["post_peak_displacement_share"], reverse=True)[:8],
        "largest_peak_to_final_regressions_n_ge_100": sorted(stable_subjects, key=lambda row: row["final_minus_best_accuracy"])[:8],
        "subject_displacement_vs_peak_to_final_accuracy_correlation_weighted_n_ge_100": weighted_correlation(stable_subjects, "displacement_share", "final_minus_best_accuracy"),
        "subject_post_peak_displacement_vs_peak_to_final_accuracy_correlation_weighted_n_ge_100": weighted_correlation(stable_subjects, "post_peak_displacement_share", "final_minus_best_accuracy"),
        "levels_significant_after_bh": [name for name, row in levels.items() if row["displacement"]["bh_q_value_within_axis"] < 0.05],
        "post_peak_levels_significant_after_bh": [name for name, row in levels.items() if row["post_global_peak_displacement"]["bh_q_value_within_axis"] < 0.05],
    }


def write_csv(path: Path, axes: dict[str, Any]) -> None:
    columns = [
        "axis", "category", "n", "initial_accuracy", "best_accuracy", "best_accuracy_iteration",
        "final_accuracy", "final_minus_best_accuracy", "best_choice_nll", "best_choice_nll_iteration",
        "final_choice_nll", "final_minus_best_choice_nll", "raw_displacement_cells",
        "displacement_share", "null_q975_cells", "permutation_p", "bh_q", "past_last_iteration",
        "future_first_iteration", "post_peak_displacement_cells", "post_peak_displacement_share",
        "post_peak_permutation_p", "post_peak_bh_q", "ever_correctness_changed", "ever_answer_choice_changed",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for axis, axis_payload in axes.items():
            for name, row in axis_payload["categories"].items():
                displacement = row["displacement"]
                post_peak = row["post_global_peak_displacement"]
                writer.writerow({
                    "axis": axis,
                    "category": name,
                    "n": row["n"],
                    "initial_accuracy": row["initial_accuracy"],
                    "best_accuracy": row["best_accuracy"],
                    "best_accuracy_iteration": row["best_accuracy_iteration"],
                    "final_accuracy": row["final_accuracy"],
                    "final_minus_best_accuracy": row["final_minus_best_accuracy"],
                    "best_choice_nll": row["best_choice_nll"],
                    "best_choice_nll_iteration": row["best_choice_nll_iteration"],
                    "final_choice_nll": row["final_choice_nll"],
                    "final_minus_best_choice_nll": row["final_minus_best_choice_nll"],
                    "raw_displacement_cells": displacement["raw_cells"],
                    "displacement_share": displacement["fraction_of_available_wrong_mass"],
                    "null_q975_cells": displacement["null_q975_cells"],
                    "permutation_p": displacement["permutation_p_high"],
                    "bh_q": displacement["bh_q_value_within_axis"],
                    "past_last_iteration": displacement["past_last_iteration"],
                    "future_first_iteration": displacement["future_first_iteration"],
                    "post_peak_displacement_cells": post_peak["raw_cells"],
                    "post_peak_displacement_share": post_peak["fraction_of_available_wrong_mass"],
                    "post_peak_permutation_p": post_peak["permutation_p_high"],
                    "post_peak_bh_q": post_peak["bh_q_value_within_axis"],
                    "ever_correctness_changed": row["ever_correctness_changed"],
                    "ever_answer_choice_changed": row["ever_answer_choice_changed"],
                })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=1999)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()

    payload = np.load(args.history, allow_pickle=False)
    required = {
        "states", "choices", "choice_nll", "correct_answer_bits", "correct_answer_bytes",
        "iterations", "example_ids", "subjects", "levels", "answer_indices", "num_choices",
    }
    missing = required - set(payload.files)
    if missing:
        raise ValueError(f"history matrix lacks fields: {sorted(missing)}")
    states = payload["states"].astype(np.bool_)
    choices = payload["choices"].astype(np.int8)
    choice_nll = payload["choice_nll"].astype(np.float64)
    correct_answer_bits = payload["correct_answer_bits"].astype(np.float64)
    correct_answer_bytes = payload["correct_answer_bytes"].astype(np.int64)
    iterations = payload["iterations"].astype(int)
    example_ids = payload["example_ids"].astype(str)
    subjects = payload["subjects"].astype(str)
    levels = payload["levels"].astype(str)
    if states.shape != (19, 16_159):
        raise ValueError(f"unexpected exact-history geometry: {states.shape}")
    if choices.shape != states.shape or choice_nll.shape != states.shape or correct_answer_bits.shape != states.shape:
        raise ValueError("response matrix geometry drift")
    if len(np.unique(example_ids)) != len(example_ids):
        raise ValueError("duplicate example IDs")
    if np.any(correct_answer_bytes < 0):
        raise ValueError("negative answer byte count")

    axes = {
        "whole": analyze_axis(
            "whole", np.full(len(example_ids), "All clean GreekMMLU"), states, choices, choice_nll,
            correct_answer_bits, correct_answer_bytes, example_ids, iterations, args.permutations, args.seed + 1,
        ),
        "subject": analyze_axis(
            "subject", subjects, states, choices, choice_nll, correct_answer_bits, correct_answer_bytes,
            example_ids, iterations, args.permutations, args.seed + 2,
        ),
        "level": analyze_axis(
            "level", levels, states, choices, choice_nll, correct_answer_bits, correct_answer_bytes,
            example_ids, iterations, args.permutations, args.seed + 3,
        ),
    }
    peak_index = int(np.where(iterations == 9536)[0][0])
    axis_labels = {
        "whole": np.full(len(example_ids), "All clean GreekMMLU"),
        "subject": subjects,
        "level": levels,
    }
    for axis_index, (axis, labels) in enumerate(axis_labels.items()):
        post_peak = analyze_window_displacement(
            f"{axis}_post_global_peak",
            labels,
            states[peak_index:],
            choice_nll[0],
            example_ids,
            iterations[peak_index:],
            args.permutations,
            args.seed + 101 + axis_index,
        )
        if set(post_peak) != set(axes[axis]["categories"]):
            raise AssertionError(f"{axis}: post-peak category identity drift")
        for name, result in post_peak.items():
            axes[axis]["categories"][name]["post_global_peak_displacement"] = result
    output = {
        "schema_version": "full8b_greekmmlu_response_displacement_by_category_v1",
        "status": "completed",
        "history": {"path": str(args.history.resolve()), "bytes": args.history.stat().st_size, "sha256": sha256_file(args.history)},
        "contract": {
            "questions": int(states.shape[1]),
            "checkpoints": int(states.shape[0]),
            "iterations": iterations.tolist(),
            "category_axes": ["native subject label", "native educational level", "whole clean benchmark"],
            "accuracy_adjustment": "Within each category, match every checkpoint's exact correct count using a fixed difficulty order defined only by iteration-0 choice NLL, with example ID as deterministic tie-breaker.",
            "displacement": "Maximum guarded past-future half-L1 distance between accuracy-adjusted wrong-cell occupancy vectors; units are equivalent wrong cells displaced.",
            "windows": ["all 19 checkpoints", "10 checkpoints from the global accuracy peak at iteration 9536 through the endpoint"],
            "noise_floor": "Checkpoint-order permutation preserving each complete questionnaire state, checkpoint accuracy, per-question response frequency and its paired accuracy baseline.",
            "multiple_testing": "Benjamini-Hochberg correction within each category axis.",
            "permutations": args.permutations,
            "seed": args.seed,
        },
        "axes": axes,
        "findings": build_findings(axes),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output_json) + ".partial")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(args.output_json)
    write_csv(args.output_csv, axes)
    print(json.dumps({
        "ok": True,
        "json": str(args.output_json.resolve()),
        "csv": str(args.output_csv.resolve()),
        "subjects": axes["subject"]["category_count"],
        "levels": axes["level"]["category_count"],
        "permutations": args.permutations,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
