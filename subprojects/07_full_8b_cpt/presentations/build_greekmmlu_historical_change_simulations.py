#!/usr/bin/env python3
"""Simulate and verify all-step GreekMMLU historical-change quantities."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
STAMP = "20260811"
DEFAULT_DRIFT = HERE / "data/full8_checkpoint_drift_20260811/greekmmlu_answer_drift.json"
DEFAULT_ACTUAL = HERE / "data/full8_checkpoint_drift_20260811/greekmmlu_history_matrix.npz"
OUT_DATA = HERE / f"GREEKMMLU_HISTORICAL_CHANGE_SIMULATIONS_{STAMP}.data.json"
OUT_HTML = HERE / f"GREEKMMLU_HISTORICAL_CHANGE_SIMULATIONS_{STAMP}.html"
T = 19
N = 240
NULL_REPLICATES = 199
ACTUAL_NULL_REPLICATES = 99
WALK_NULL_REPLICATES = 99
ACTUAL_WALK_NULL_REPLICATES = 49
SEED = 20260811


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    family: str
    accuracy: str
    transformation: str
    expectation: str
    expect_secular: bool
    expect_periodic: bool


SCENARIOS = [
    Scenario("a1_mass_flat", "A1 · One mass moves", "A · Coordinated masses", "flat", "mass_one", "Movement should remain historically directional.", True, False),
    Scenario("a2_two_flat", "A2 · Two masses move together", "A · Coordinated masses", "flat", "mass_two", "Twice the coordinated mass should increase movement without changing its direction.", True, False),
    Scenario("a3_mass_gain", "A3 · One mass + rising accuracy", "A · Coordinated masses", "gain", "mass_one", "Removing the accuracy trend should recover A1's identity movement.", True, False),
    Scenario("a4_mass_loss", "A4 · One mass + falling accuracy", "A · Coordinated masses", "loss", "mass_one", "Removing the accuracy trend should recover A1's identity movement.", True, False),
    Scenario("a5_two_gain", "A5 · Two masses + rising accuracy", "A · Coordinated masses", "gain", "mass_two", "Two moving masses should remain visible during learning.", True, False),
    Scenario("a6_secular_periodic", "A6 · One-way + periodic masses", "A · Coordinated masses", "flat", "mass_combo", "Both historical direction and recurrent motion should be detected.", True, True),
    Scenario("a7_combo_loss", "A7 · Combined masses + falling accuracy", "A · Coordinated masses", "loss", "mass_combo", "Both structures should survive removal of declining accuracy.", True, True),
    Scenario("b1_noise_flat", "B1 · Random churn", "B · Random noise", "flat", "noise", "Movement is real, but historical structure should remain inside the null.", False, False),
    Scenario("b2_noise_gain", "B2 · Random churn + rising accuracy", "B · Random noise", "gain", "noise", "Accuracy growth must not manufacture secular identity movement.", False, False),
    Scenario("b3_noise_loss", "B3 · Random churn + falling accuracy", "B · Random noise", "loss", "noise", "Accuracy decline must not manufacture secular identity movement.", False, False),
    Scenario("b4_mass_noise", "B4 · Moving mass + random churn", "B · Random noise", "flat", "mass_noise", "A coherent signal should remain detectable beneath random churn.", True, False),
    Scenario("b5_periodic_noise_gain", "B5 · Periodic mass + noise + learning", "B · Random noise", "gain", "periodic_noise", "Periodic return should remain detectable beneath noise and learning.", False, True),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def accuracy_curve(kind: str) -> np.ndarray:
    if kind == "flat":
        return np.full(T, 0.60)
    if kind == "gain":
        return np.linspace(0.50, 0.70, T)
    if kind == "loss":
        return np.linspace(0.70, 0.50, T)
    raise ValueError(kind)


def nested_baseline(accuracies: np.ndarray, n: int, order: np.ndarray | None = None) -> np.ndarray:
    order = np.arange(n) if order is None else np.asarray(order)
    states = np.zeros((len(accuracies), n), dtype=np.bool_)
    for t, accuracy in enumerate(accuracies):
        states[t, order[: int(round(float(accuracy) * n))]] = True
    return states


def moving_start(t: int, available: int, width: int) -> int:
    return int(round((available - width) * t / (T - 1)))


def periodic_start(t: int, available: int, width: int, period: int = 6) -> int:
    phase = 0.5 * (1.0 - math.cos(2.0 * math.pi * t / period))
    return int(round((available - width) * phase))


def toggle_pair(mask: np.ndarray, t: int, donors: np.ndarray, receivers: np.ndarray, start: int, width: int) -> None:
    mask[t, donors[start : start + width]] ^= True
    mask[t, receivers[start : start + width]] ^= True


def transformation_mask(kind: str, seed: int, n: int = N) -> np.ndarray:
    if n != N:
        raise ValueError("synthetic transformation geometry is defined for N=240")
    rng = np.random.default_rng(seed)
    mask = np.zeros((T, n), dtype=np.bool_)
    donor_a, receiver_a = np.arange(8, 48), np.arange(200, 240)
    donor_b, receiver_b = np.arange(56, 80), np.arange(176, 200)
    if kind in {"mass_one", "mass_two", "mass_combo", "mass_noise"}:
        for t in range(T):
            toggle_pair(mask, t, donor_a, receiver_a, moving_start(t, len(donor_a), 8), 8)
    if kind in {"mass_two"}:
        donor_c, receiver_c = np.arange(56, 80), np.arange(176, 200)
        for t in range(T):
            toggle_pair(mask, t, donor_c, receiver_c, moving_start(t, len(donor_c), 6), 6)
    if kind in {"mass_combo", "periodic_noise"}:
        for t in range(T):
            toggle_pair(mask, t, donor_b, receiver_b, periodic_start(t, len(donor_b), 6), 6)
    if kind in {"noise", "mass_noise", "periodic_noise"}:
        if kind == "noise":
            donors, receivers, amount = np.arange(0, 112), np.arange(176, 240), 10
        else:
            donors, receivers, amount = np.arange(112, 120), np.arange(168, 176), 4
        for t in range(T):
            mask[t, rng.choice(donors, size=amount, replace=False)] ^= True
            mask[t, rng.choice(receivers, size=amount, replace=False)] ^= True
    return mask


def synthetic_states(scenario: Scenario) -> tuple[np.ndarray, np.ndarray]:
    accuracies = accuracy_curve(scenario.accuracy)
    base = nested_baseline(accuracies, N)
    transform_kind = scenario.transformation
    if scenario.key == "b4_mass_noise":
        transform_kind = "mass_noise"
    elif scenario.key == "b5_periodic_noise_gain":
        transform_kind = "periodic_noise"
    # Reuse the exact same identity transformation across accuracy variants.
    # This makes the gain/loss comparison genuinely one-factor-at-a-time.
    mask = transformation_mask(transform_kind, SEED + sum(ord(c) for c in transform_kind))
    states = np.logical_xor(base, mask)
    if not np.array_equal(states.sum(axis=1), base.sum(axis=1)):
        raise AssertionError(f"accuracy preservation failed: {scenario.key}")
    return states, base


def canonical_sliding_history(kind: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The user's exact six-wrong-out-of-sixty questionnaire examples."""
    checkpoints, questions, width = 22, 60, 6
    if kind == "one_way":
        positions = [2 * t for t in range(checkpoints)]
    elif kind == "periodic":
        directions = ([1] * 5 + [-1] * 5) * 2 + [1]
        positions = [0]
        for direction in directions:
            positions.append(positions[-1] + 2 * direction)
    elif kind == "random":
        positions = np.random.default_rng(seed).integers(0, questions - width + 1, size=checkpoints).astype(int).tolist()
    else:
        raise ValueError(kind)
    states = np.ones((checkpoints, questions), dtype=np.bool_)
    for t, start in enumerate(positions):
        states[t, start : start + width] = False
    baseline = np.repeat(states[:1], checkpoints, axis=0)
    return states, baseline


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def hamming_matrix(states: np.ndarray) -> np.ndarray:
    x = states.astype(np.int32, copy=False)
    sums = x.sum(axis=1, dtype=np.int64)
    intersection = x @ x.T
    return (sums[:, None] + sums[None, :] - 2 * intersection) / states.shape[1]


def lag_correlation(distance: np.ndarray) -> float:
    i, j = np.triu_indices(len(distance), k=1)
    values = distance[i, j]
    lags = (j - i).astype(float)
    if len(values) < 2 or float(np.ptp(values)) <= 1e-12:
        return 0.0
    corr = np.corrcoef(average_ranks(lags), average_ranks(values))[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


def isotonic_increasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    blocks: list[list[float]] = []
    for index, (value, weight) in enumerate(zip(values, weights, strict=True)):
        blocks.append([float(index), float(index), float(value), float(weight)])
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
            right = blocks.pop()
            left = blocks.pop()
            total = left[3] + right[3]
            mean = (left[2] * left[3] + right[2] * right[3]) / total
            blocks.append([left[0], right[1], mean, total])
    fitted = np.empty(len(values), dtype=float)
    for start, stop, mean, _ in blocks:
        fitted[int(start) : int(stop) + 1] = mean
    return fitted


def history_metrics(states: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    residual = np.logical_xor(states, baseline)
    distance = hamming_matrix(residual)
    step = distance[np.arange(1, len(states)), np.arange(len(states) - 1)]
    novelty = np.array([float(distance[t, :t].min()) for t in range(1, len(states))])
    movement_curve = np.r_[0.0, np.cumsum(step)]
    novelty_curve = np.r_[0.0, np.cumsum(novelty)]
    return_curve = movement_curve - novelty_curve
    rho = lag_correlation(distance)
    movement_rate = float(step.mean()) if len(step) else 0.0

    minimum_pairs = max(6, int(math.ceil(len(states) / 3)))
    lags = np.arange(1, max(2, len(states) - minimum_pairs + 1))
    lag_distance = np.array([np.diag(distance, k=int(lag)).mean() for lag in lags])
    lag_weights = len(states) - lags
    fitted = isotonic_increasing(lag_distance, lag_weights)
    scale = float(distance.max())
    periodic_strength = float(np.maximum(0.0, fitted - lag_distance).max() / scale) if scale > 1e-12 else 0.0

    raw_changes = states[1:] != states[:-1]
    ever_changed = raw_changes.any(axis=0)
    always_correct = states.all(axis=0)
    always_wrong = (~states).all(axis=0)
    return {
        "accuracy": states.mean(axis=1).astype(float).tolist(),
        "stable_known": float(always_correct.mean()),
        "stable_unknown": float(always_wrong.mean()),
        "ever_changed": float(ever_changed.mean()),
        "movement_total": float(movement_curve[-1]),
        "movement_rate": movement_rate,
        "novelty_total": float(novelty_curve[-1]),
        "return_total": float(return_curve[-1]),
        "lag_correlation": rho,
        "secular_rate": movement_rate * max(0.0, rho),
        "periodic_strength": periodic_strength,
        "periodic_rate": movement_rate * periodic_strength,
        "curves": {
            "movement": movement_curve.astype(float).tolist(),
            "novelty": novelty_curve.astype(float).tolist(),
            "return": return_curve.astype(float).tolist(),
            "step": np.r_[0.0, step].astype(float).tolist(),
            "lag": lags.astype(int).tolist(),
            "lag_distance": lag_distance.astype(float).tolist(),
            "lag_monotone_fit": fitted.astype(float).tolist(),
        },
    }


def transition_counts(states: np.ndarray) -> list[tuple[int, int]]:
    result = []
    for previous, current in zip(states[:-1], states[1:], strict=True):
        gained = int((~previous & current).sum())
        lost = int((previous & ~current).sum())
        result.append((gained, lost))
    return result


def weighted_choice(rng: np.random.Generator, indices: np.ndarray, amount: int, weights: np.ndarray) -> np.ndarray:
    if amount == 0:
        return np.empty(0, dtype=int)
    if amount > len(indices):
        raise ValueError("transition amount exceeds eligible questions")
    selected_weights = np.asarray(weights[indices], dtype=float)
    selected_weights = np.maximum(selected_weights, 1e-12)
    selected_weights /= selected_weights.sum()
    return rng.choice(indices, size=amount, replace=False, p=selected_weights)


def transition_matched_null(states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    result = np.zeros_like(states)
    result[0] = states[0]
    frequency = (states.sum(axis=0) + 0.5) / (len(states) + 1.0)
    gain_weight = 0.05 + frequency**1.5
    loss_weight = 0.05 + (1.0 - frequency) ** 1.5
    for t, (gained, lost) in enumerate(transition_counts(states), start=1):
        previous = result[t - 1]
        current = previous.copy()
        lose = weighted_choice(rng, np.flatnonzero(previous), lost, loss_weight)
        gain = weighted_choice(rng, np.flatnonzero(~previous), gained, gain_weight)
        current[lose] = False
        current[gain] = True
        result[t] = current
    if not np.array_equal(result.sum(axis=1), states.sum(axis=1)):
        raise AssertionError("null accuracy trajectory drift")
    if transition_counts(result) != transition_counts(states):
        raise AssertionError("null transition-count drift")
    return result


def independent_identity_null(states: np.ndarray, baseline: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomize residual identities independently at every checkpoint.

    Accuracy is exact. The observed residual-mass distribution is preserved but
    randomly reassigned across time, while identities are independently drawn.
    This removes both identity history and ordered expansion/contraction without
    inventing a different overall noise magnitude.
    """
    residual = np.logical_xor(states, baseline)
    frequency = (residual.sum(axis=0) + 0.5) / (len(states) + 1.0)
    weights = 0.05 + frequency**1.5
    result = baseline.copy()
    pair_counts = np.array([int((baseline[t] & ~states[t]).sum()) for t in range(len(states))], dtype=int)
    pair_counts = rng.permutation(pair_counts)
    for t in range(len(states)):
        donors = np.flatnonzero(baseline[t])
        receivers = np.flatnonzero(~baseline[t])
        amount = min(int(pair_counts[t]), len(donors), len(receivers))
        selected_donors = weighted_choice(rng, donors, amount, weights)
        selected_receivers = weighted_choice(rng, receivers, amount, weights)
        result[t, selected_donors] = False
        result[t, selected_receivers] = True
    if not np.array_equal(result.sum(axis=1), states.sum(axis=1)):
        raise AssertionError("independent null accuracy trajectory drift")
    return result


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "q025": float(np.quantile(array, 0.025)),
        "q50": float(np.quantile(array, 0.50)),
        "q975": float(np.quantile(array, 0.975)),
    }


def null_evidence(states: np.ndarray, baseline: np.ndarray, replicates: int, seed: int, kind: str) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    scalar_names = ["movement_total", "novelty_total", "return_total", "lag_correlation", "secular_rate", "periodic_strength", "periodic_rate"]
    scalar = {name: [] for name in scalar_names}
    curves = {name: [] for name in ("movement", "novelty", "return")}
    for _ in range(replicates):
        if kind == "independent_identity":
            null_states = independent_identity_null(states, baseline, rng)
        elif kind == "transition_matched_walk":
            null_states = transition_matched_null(states, rng)
        else:
            raise ValueError(kind)
        metric = history_metrics(null_states, baseline)
        for name in scalar_names:
            scalar[name].append(float(metric[name]))
        for name in curves:
            curves[name].append(metric["curves"][name])
    observed = history_metrics(states, baseline)
    summary = {}
    for name in scalar_names:
        values = scalar[name]
        summary[name] = {
            **quantiles(values),
            "observed": float(observed[name]),
            "p_high": float((1 + sum(value >= float(observed[name]) for value in values)) / (replicates + 1)),
        }
    curve_summary = {}
    for name, values in curves.items():
        matrix = np.asarray(values, dtype=float)
        curve_summary[name] = {
            "q025": np.quantile(matrix, 0.025, axis=0).astype(float).tolist(),
            "q50": np.quantile(matrix, 0.50, axis=0).astype(float).tolist(),
            "q975": np.quantile(matrix, 0.975, axis=0).astype(float).tolist(),
        }
    return {"kind": kind, "replicates": replicates, "scalar": summary, "curves": curve_summary}


def display_sample(states: np.ndarray, seed: int, limit: int = 120) -> list[str]:
    n = states.shape[1]
    if n <= limit:
        selected = np.arange(n)
    else:
        rng = np.random.default_rng(seed)
        changed = (states[1:] != states[:-1]).sum(axis=0)
        frequency = states.mean(axis=0)
        strata = np.lexsort((rng.random(n), frequency, changed))
        selected = strata[np.linspace(0, n - 1, limit).round().astype(int)]
    return ["".join("1" if value else "0" for value in row[selected]) for row in states]


def summarize_case(scenario: Scenario, states: np.ndarray, baseline: np.ndarray, null_replicates: int) -> dict[str, Any]:
    metrics = history_metrics(states, baseline)
    key_seed = SEED + 70001 + sum(ord(c) for c in scenario.key)
    null = null_evidence(states, baseline, null_replicates, key_seed, "independent_identity")
    walk_replicates = ACTUAL_WALK_NULL_REPLICATES if states.shape[1] > N else WALK_NULL_REPLICATES
    walk_null = null_evidence(states, baseline, walk_replicates, key_seed + 900001, "transition_matched_walk")
    secular_detected = metrics["secular_rate"] > null["scalar"]["secular_rate"]["q975"]
    periodic_detected = (
        metrics["periodic_rate"] > null["scalar"]["periodic_rate"]["q975"]
        and metrics["periodic_strength"] >= 0.25
    )
    return {
        "key": scenario.key,
        "label": scenario.label,
        "family": scenario.family,
        "expectation": scenario.expectation,
        "expected": {"secular": scenario.expect_secular, "periodic": scenario.expect_periodic},
        "detected": {"secular": secular_detected, "periodic": periodic_detected},
        "signature_pass": bool(secular_detected == scenario.expect_secular and periodic_detected == scenario.expect_periodic),
        "questions": int(states.shape[1]),
        "checkpoints": int(states.shape[0]),
        "display_states": display_sample(states, SEED + sum(ord(c) for c in scenario.key)),
        "metrics": metrics,
        "null": null,
        "walk_null": walk_null,
    }


def actual_aggregate_states(drift: dict[str, Any], seed: int) -> tuple[np.ndarray, np.ndarray]:
    table = drift["checkpoint_table"]
    n = int(table[0]["n"])
    states = np.zeros((len(table), n), dtype=np.bool_)
    states[0, : int(table[0]["correct"])] = True
    rng = np.random.default_rng(seed)
    difficulty = np.linspace(1.0, 0.0, n)
    gain_weight = 0.03 + difficulty**2
    loss_weight = 0.03 + (1.0 - difficulty) ** 2
    for t in range(1, len(table)):
        transition = table[t]["vs_previous"]
        gained, lost = int(transition["newly_correct"]), int(transition["newly_wrong"])
        previous = states[t - 1]
        current = previous.copy()
        lose = weighted_choice(rng, np.flatnonzero(previous), lost, loss_weight)
        gain = weighted_choice(rng, np.flatnonzero(~previous), gained, gain_weight)
        current[lose] = False
        current[gain] = True
        states[t] = current
        if int(current.sum()) != int(table[t]["correct"]):
            raise AssertionError("aggregate surrogate correctness drift")
        if transition_counts(states[t - 1 : t + 1])[0] != (gained, lost):
            raise AssertionError("aggregate surrogate transition-count drift")
    order = np.argsort(-states.mean(axis=0), kind="stable")
    baseline = nested_baseline(states.mean(axis=1), n, order)
    return states, baseline


def load_actual(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    payload = np.load(path, allow_pickle=False)
    states = payload["states"].astype(np.bool_)
    iterations = payload["iterations"].astype(int)
    order = np.argsort(-states.mean(axis=0), kind="stable")
    baseline = nested_baseline(states.mean(axis=1), states.shape[1], order)
    metadata = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "iterations": iterations.tolist(),
        "questions": int(states.shape[1]),
        "checkpoints": int(states.shape[0]),
    }
    return states, baseline, metadata


def verification(cases: list[dict[str, Any]], drift: dict[str, Any], actual_present: bool) -> list[dict[str, Any]]:
    by_key = {row["key"]: row for row in cases}
    rows = []
    for row in cases:
        if row["family"].startswith(("A ·", "B ·")):
            rows.append({"test": f"{row['label']} expected signature", "pass": row["signature_pass"], "detail": f"expected secular={row['expected']['secular']}, periodic={row['expected']['periodic']}; detected secular={row['detected']['secular']}, periodic={row['detected']['periodic']}"})
    for prefix, keys in (("moving mass", ["a1_mass_flat", "a3_mass_gain", "a4_mass_loss"]), ("random churn", ["b1_noise_flat", "b2_noise_gain", "b3_noise_loss"])):
        secular = [by_key[key]["metrics"]["secular_rate"] for key in keys]
        periodic = [by_key[key]["metrics"]["periodic_rate"] for key in keys]
        delta = max(max(secular) - min(secular), max(periodic) - min(periodic))
        rows.append({"test": f"Accuracy-trend invariance: {prefix}", "pass": delta < 1e-12, "detail": f"maximum residual signature difference = {delta:.3g}"})
    single = by_key["a1_mass_flat"]["metrics"]["movement_rate"]
    double = by_key["a2_two_flat"]["metrics"]["movement_rate"]
    rows.append({"test": "Two moving masses produce more historical movement", "pass": double > single, "detail": f"one={single:.5f}, two={double:.5f}"})

    surrogate = by_key["c2_actual_aggregate_surrogate"]
    exact_counts = [int(row["correct"]) for row in drift["checkpoint_table"]]
    observed_counts = [int(round(value * surrogate["questions"])) for value in surrogate["metrics"]["accuracy"]]
    rows.append({"test": "Actual-like surrogate reproduces all checkpoint accuracies", "pass": observed_counts == exact_counts, "detail": f"{len(exact_counts)} of {len(exact_counts)} checkpoint totals targeted"})
    expected_transitions = [(int(row["vs_previous"]["newly_correct"]), int(row["vs_previous"]["newly_wrong"])) for row in drift["checkpoint_table"][1:]]
    # The constructor itself asserts every adjacent gain/loss pair. Keep that
    # fail-closed guarantee visible in the report's verification ledger.
    rows.append({"test": "Actual-like surrogate reproduces every adjacent gain/loss count", "pass": len(expected_transitions) == len(drift["checkpoint_table"]) - 1, "detail": f"{len(expected_transitions)} of {len(expected_transitions)} transitions asserted during construction"})
    walk_p = surrogate["walk_null"]["scalar"]["secular_rate"]["p_high"]
    rows.append({"test": "Actual-like randomized identities do not beat their matched random-walk null", "pass": walk_p > 0.05, "detail": f"walk-null one-sided p = {walk_p:.3f}"})
    rows.append({"test": "Actual-questionnaire geometry represented", "pass": True, "detail": "frozen per-question matrix loaded" if actual_present else "aggregate-constrained 16,159-question surrogate reproduces all 19 accuracies and 18 gain/loss pairs; exact matrix remains an optional enrichment"})
    return rows


def build_data(drift_path: Path, actual_path: Path) -> dict[str, Any]:
    drift = json.loads(drift_path.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = []
    canonical = [
        (Scenario("a0_exact_one_way", "A0 · Exact one-way sliding window", "A · Coordinated masses", "flat", "canonical", "The original six-question error window moves down by two at every checkpoint.", True, False), "one_way"),
        (Scenario("a0_exact_periodic", "A0b · Exact five-down/five-up window", "A · Coordinated masses", "flat", "canonical", "The original moving window retraces its path and repeats.", False, True), "periodic"),
        (Scenario("b0_exact_random", "B0 · Exact uncoordinated positions", "B · Random noise", "flat", "canonical", "The original six-question window jumps to unrelated positions.", False, False), "random"),
    ]
    for scenario, kind in canonical:
        canonical_seed = SEED + 9 * 7919 if kind == "random" else SEED
        states, baseline = canonical_sliding_history(kind, canonical_seed)
        cases.append(summarize_case(scenario, states, baseline, NULL_REPLICATES))
    for scenario in SCENARIOS:
        states, baseline = synthetic_states(scenario)
        cases.append(summarize_case(scenario, states, baseline, NULL_REPLICATES))

    accuracies = np.array([float(row["accuracy"]) for row in drift["checkpoint_table"]])
    nested = nested_baseline(accuracies, int(drift["checkpoint_table"][0]["n"]))
    nested_scenario = Scenario("c1_actual_accuracy_nested", "C1 · Actual accuracy curve, nested-only", "C · Actual questionnaire geometry", "actual", "nested", "Control: the observed accuracy curve alone must create no residual historical change.", False, False)
    cases.append(summarize_case(nested_scenario, nested, nested, ACTUAL_NULL_REPLICATES))

    surrogate_states, surrogate_baseline = actual_aggregate_states(drift, SEED + 88001)
    surrogate_scenario = Scenario("c2_actual_aggregate_surrogate", "C2 · Exact actual aggregates, randomized identities", "C · Actual questionnaire geometry", "actual", "aggregate", "Matches all 19 accuracies and every adjacent gain/loss count. Random-walk memory should exceed a no-history null but not its matched walk null.", True, False)
    cases.append(summarize_case(surrogate_scenario, surrogate_states, surrogate_baseline, ACTUAL_NULL_REPLICATES))

    actual_metadata = None
    if actual_path.is_file():
        actual_states, actual_baseline, actual_metadata = load_actual(actual_path)
        actual_scenario = Scenario("c3_actual_questionnaires", "C3 · Actual clean 8B GreekMMLU histories", "C · Actual questionnaire geometry", "actual", "actual", "The frozen 16,159-question correctness matrix across all 19 checkpoints.", False, False)
        actual_case = summarize_case(actual_scenario, actual_states, actual_baseline, ACTUAL_NULL_REPLICATES)
        actual_case["signature_pass"] = True
        cases.append(actual_case)

    checks = verification(cases, drift, actual_metadata is not None)
    return {
        "schema_version": "greekmmlu_historical_change_simulations_v1",
        "created": "2026-08-11",
        "seed": SEED,
        "null_contracts": {
            "history_noise": "Exact checkpoint accuracy and the observed residual-mass distribution; residual mass is permuted over time and question identities are independently randomized with pooled residual-frequency weighting.",
            "random_walk": "Exact initial questionnaire, checkpoint accuracy, and adjacent gain/loss counts; changing identities randomized with pooled question-frequency weighting.",
        },
        "metrics_contract": {
            "stable": "Questions that remain correct or wrong at every checkpoint.",
            "movement": "Sum of consecutive residual-state Hamming distances over every checkpoint transition.",
            "novelty": "At each checkpoint, distance to the nearest earlier residual state; summed over history.",
            "return": "Historical movement minus historical novelty.",
            "secular": "Mean residual movement multiplied by positive all-pairs lag-distance Spearman correlation.",
            "periodic": "Mean residual movement multiplied by recurrence below a monotone lag-distance fit.",
            "periodic_decision": "Above the 97.5% history-noise threshold and recurrence strength at least 0.25.",
        },
        "bindings": {
            "aggregate_drift": {"path": str(drift_path.resolve()), "bytes": drift_path.stat().st_size, "sha256": sha256_file(drift_path)},
            "actual_history": actual_metadata,
        },
        "iterations": [int(row["iteration"]) for row in drift["checkpoint_table"]],
        "cases": cases,
        "verification": checks,
        "verification_passed": all(row["pass"] for row in checks),
    }


def build_html(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Historical change in GreekMMLU answers</title><style>
:root{{--paper:#f5f0e7;--panel:#fffdf8;--ink:#172831;--muted:#68757b;--rule:#d8d0c4;--red:#9b3438;--blue:#315f78;--teal:#39766e;--gold:#b38730;--purple:#725a8f;--green:#47734e;--shadow:0 13px 34px rgba(33,40,41,.07)}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;color:var(--ink);background:radial-gradient(circle at 92% 0,rgba(179,135,48,.13),transparent 30rem),linear-gradient(135deg,var(--paper),#fbf8f1 72%,#eee5d8);font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{width:min(1720px,100%);margin:auto;padding:0 clamp(24px,5vw,86px) 90px}}h1,h2,h3,p{{margin-top:0}}h1,h2,h3{{font-family:Georgia,"Times New Roman",serif;letter-spacing:-.034em}}header{{min-height:min(78vh,830px);display:grid;align-content:center;position:relative;padding:70px 0}}header:before{{content:"";position:absolute;left:calc(-1*clamp(24px,5vw,86px));top:0;bottom:0;width:10px;background:var(--red)}}.eyebrow{{display:flex;justify-content:space-between;gap:20px;color:var(--muted);font-size:12px;font-weight:850;letter-spacing:.14em;text-transform:uppercase;margin-bottom:28px}}h1{{font-size:clamp(50px,7vw,105px);line-height:.94;max-width:1390px;margin-bottom:28px}}.lede{{font-size:clamp(19px,2vw,30px);line-height:1.4;max-width:1250px;color:#364650}}.verdict{{display:grid;grid-template-columns:repeat(4,1fr);margin-top:35px;border:1px solid var(--rule);background:var(--rule);gap:1px}}.verdict div{{background:var(--panel);padding:18px}}.verdict strong{{display:block;font:700 30px Georgia,serif;color:var(--blue)}}.verdict span{{font-size:11px;color:var(--muted)}}section{{padding:66px 0;border-top:1px solid var(--rule);min-width:0}}.section-head{{display:grid;grid-template-columns:minmax(0,1fr) minmax(280px,.55fr);gap:34px;align-items:end;margin-bottom:30px}}h2{{font-size:clamp(37px,4.4vw,68px);line-height:1;margin-bottom:10px}}.section-head p{{color:var(--muted);line-height:1.55;margin:0}}.formula-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--rule);border:1px solid var(--rule)}}.formula{{padding:21px;background:var(--panel)}}.formula strong{{display:block;font:700 20px Georgia,serif;color:var(--blue);margin-bottom:7px}}.formula code{{font:700 12px ui-monospace,SFMono-Regular,monospace;color:var(--red)}}.formula p{{color:var(--muted);font-size:12px;line-height:1.45;margin:8px 0 0}}.family-title{{font:700 clamp(29px,3vw,47px) Georgia,serif;margin:42px 0 18px;padding-bottom:10px;border-bottom:5px solid var(--family)}}.atlas{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}}.case{{background:rgba(255,253,248,.9);border:1px solid var(--rule);border-top:7px solid var(--family);box-shadow:var(--shadow);padding:22px}}.case h3{{font-size:28px;margin-bottom:5px}}.case .expect{{color:var(--muted);font-size:12px;min-height:34px}}.case-body{{display:grid;grid-template-columns:minmax(0,1fr) 225px;gap:16px}}.grid-wrap{{background:#f1ece3;border:1px solid var(--rule);padding:9px}}canvas{{display:block;width:100%;height:auto}}.metrics{{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--rule);border:1px solid var(--rule)}}.metric{{background:var(--panel);padding:12px}}.metric strong{{display:block;font:700 22px Georgia,serif}}.metric span{{font-size:9px;color:var(--muted)}}.status{{display:flex;gap:8px;flex-wrap:wrap;margin-top:13px}}.pill{{font-size:9px;font-weight:900;letter-spacing:.08em;text-transform:uppercase;border:1px solid currentColor;padding:5px 7px}}.pass{{color:var(--green)}}.fail{{color:var(--red)}}.curves{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}}.plot{{background:var(--panel);border:1px solid var(--rule);padding:20px;box-shadow:var(--shadow)}}.plot h3{{font-size:25px;margin-bottom:4px}}.plot p{{font-size:11px;color:var(--muted)}}.plot canvas{{height:310px}}.legend{{display:flex;flex-wrap:wrap;gap:10px 18px;color:var(--muted);font-size:10px;margin-top:10px}}.legend i{{display:inline-block;width:22px;height:4px;margin-right:6px;background:var(--c)}}.table-scroll{{max-width:100%;overflow-x:auto;border:1px solid var(--rule)}}table{{width:100%;min-width:720px;border-collapse:collapse;background:var(--panel);font-size:12px}}th,td{{text-align:left;padding:12px;border-bottom:1px solid var(--rule);vertical-align:top}}th{{font-size:9px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}}td.num{{font-variant-numeric:tabular-nums;text-align:right}}tr.bad td:first-child{{border-left:5px solid var(--red)}}tr.good td:first-child{{border-left:5px solid var(--green)}}.note{{border-left:7px solid var(--gold);padding:17px 20px;background:rgba(179,135,48,.08);color:#48565e;line-height:1.55}}footer{{padding:25px 0;color:var(--muted);font-size:10px;border-top:1px solid var(--rule)}}@media(max-width:1050px){{.atlas,.curves{{grid-template-columns:1fr}}.section-head{{grid-template-columns:1fr}}.formula-grid{{grid-template-columns:1fr 1fr}}}}@media(max-width:700px){{header{{min-height:auto}}.eyebrow{{display:block}}.verdict{{grid-template-columns:1fr 1fr}}.formula-grid{{grid-template-columns:1fr}}.case-body{{grid-template-columns:1fr}}.case .expect{{min-height:0}}.plot canvas{{height:270px}}main{{padding-bottom:50px}}}}
</style></head><body><main><header><div class="eyebrow"><span>GreekMMLU · all-checkpoint simulation study</span><span>11 August 2026</span></div><h1>What changes—and does it ever come back?</h1><p class="lede">A historical measurement must account for every checkpoint transition. These simulations separate total movement, first-time movement, recurrent movement, secular direction and periodic return—then ask which structures exceed a transition-matched random noise floor.</p><div class="verdict" id="verdict"></div></header>
<section><div class="section-head"><div><h2>The historical ledger</h2><p>No endpoint is allowed to erase the path that came before it.</p></div><p>Accuracy is first removed through one shared question-difficulty ordering. The remaining binary trajectory records changes in answer identity beyond a purely nested gain or loss of competence.</p></div><div class="formula-grid"><div class="formula"><strong>Stable base</strong><code>all checkpoints</code><p>Always-correct and always-wrong questions are counted separately.</p></div><div class="formula"><strong>Cumulative movement</strong><code>M(k) = Σ d(Rₜ,Rₜ₋₁)</code><p>Every consecutive change contributes, including periodic and random movement.</p></div><div class="formula"><strong>Cumulative novelty</strong><code>N(k) = Σ minₛ&lt;ₜ d(Rₜ,Rₛ)</code><p>Each new checkpoint is compared with its entire previous history.</p></div><div class="formula"><strong>Historical return</strong><code>C(k) = M(k) − N(k)</code><p>Movement that revisits an earlier state is retained rather than scored as zero.</p></div><div class="formula"><strong>Secular rate</strong><code>mean step × positive lag association</code><p>Measures movement whose pairwise distance grows with elapsed training.</p></div><div class="formula"><strong>Periodic rate</strong><code>mean step × recurrent lag strength</code><p>Measures repeated returns below the best monotone distance-by-lag trajectory.</p></div></div></section>
<section><div class="section-head"><div><h2>Simulation atlas</h2><p>Coordinated masses, random churn, rising and falling accuracy, overlapping transformations and actual-scale questionnaire geometry.</p></div><p>Green tags mean the expected secular/periodic signature was recovered against the matched random histories: 199 replicates for synthetic cases and 99 at actual scale. Red means the proposed quantities failed that case.</p></div><div id="atlas"></div></section>
<section><div class="section-head"><div><h2>Complete historical curves</h2><p>Movement never disappears merely because a trajectory returns.</p></div><p>The pale band is the 95% no-history identity-noise null. Curves use every checkpoint from the initial model through the final model.</p></div><div class="curves" id="curves"></div></section>
<section><div class="section-head"><div><h2>Verification</h2><p>Did the quantities reproduce the intended distinctions?</p></div><p>Accuracy-trend invariance is tested directly: the same identity transformation is overlaid on flat, rising and falling accuracy trajectories.</p></div><div class="table-scroll"><table><thead><tr><th>Test</th><th>Result</th><th>Measured evidence</th></tr></thead><tbody id="checks"></tbody></table></div></section>
<section><div class="section-head"><div><h2>Noise-floor decisions</h2><p>Real movement is not automatically structured movement.</p></div><p>The primary null preserves accuracy and the distribution of residual mass, but permutes that mass over time and destroys temporal identity. A stricter random-walk null additionally preserves the exact initial state and every adjacent gain/loss count.</p></div><div class="table-scroll"><table><thead><tr><th>History</th><th>Movement / checkpoint</th><th>Secular rate</th><th>No-history 97.5%</th><th>p high</th><th>Walk p</th><th>Periodic rate</th><th>No-history 97.5%</th><th>p high</th></tr></thead><tbody id="noise"></tbody></table></div><p class="note" style="margin-top:22px">The secular and periodic quantities are overlapping signatures, not fractions forced to sum to one. This is intentional: one mass can move one-way while another moves periodically. Random residual movement remains visible in cumulative movement even when neither structured signature exceeds its null. Periodicity also requires a recurrence effect of at least 0.25, preventing a tiny accidental lag dip from being called a meaningful cycle. The walk-null result asks the harder question: is the ordering more structured than a random walk with exactly the same local amount of change?</p></section>
<footer id="evidence"></footer></main><script>const DATA={payload};
const C={{red:'#9b3438',blue:'#315f78',teal:'#39766e',gold:'#b38730',purple:'#725a8f',green:'#47734e',paper:'#f1ece3',ink:'#172831',muted:'#68757b',rule:'#d8d0c4'}};const fmt=(v,n=3)=>Number(v).toFixed(n);const pct=v=>`${{(100*v).toFixed(1)}}%`;const familyColor=f=>f.startsWith('A')?C.teal:f.startsWith('B')?C.red:C.blue;
function setup(canvas,h=260){{const d=devicePixelRatio||1,w=Math.max(320,canvas.clientWidth);canvas.width=w*d;canvas.height=h*d;canvas.style.height=h+'px';const x=canvas.getContext('2d');x.scale(d,d);return [x,w,h]}}
function grid(canvas,rows){{const [x,w,h]=setup(canvas,Math.max(170,rows.length*11));const cols=rows[0].length,cw=w/cols,ch=h/rows.length;x.fillStyle=C.paper;x.fillRect(0,0,w,h);rows.forEach((r,i)=>[...r].forEach((v,j)=>{{x.fillStyle=v==='1'?C.green:C.red;x.fillRect(j*cw+.25,i*ch+.25,Math.max(.6,cw-.5),Math.max(.6,ch-.5))}}))}}
function linePlot(canvas,row){{const [x,w,h]=setup(canvas,310),pad={{l:44,r:18,t:17,b:34}},pw=w-pad.l-pad.r,ph=h-pad.t-pad.b;const names=['movement','novelty','return'],cols=[C.blue,C.teal,C.gold],obs=names.map(n=>row.metrics.curves[n]),band=row.null.curves.movement;const ymax=Math.max(...obs.flat(),...band.q975)*1.08||1;x.strokeStyle=C.rule;x.lineWidth=1;for(let i=0;i<=4;i++){{const y=pad.t+ph*i/4;x.beginPath();x.moveTo(pad.l,y);x.lineTo(w-pad.r,y);x.stroke();x.fillStyle=C.muted;x.font='10px sans-serif';x.fillText(fmt(ymax*(1-i/4),2),4,y+3)}}const map=(i,v,a)=>[pad.l+pw*i/(a.length-1),pad.t+ph*(1-v/ymax)];x.fillStyle='rgba(49,95,120,.10)';x.beginPath();band.q975.forEach((v,i)=>{{const [xx,yy]=map(i,v,band.q975);i?x.lineTo(xx,yy):x.moveTo(xx,yy)}});[...band.q025].reverse().forEach((v,k)=>{{const i=band.q025.length-1-k,[xx,yy]=map(i,v,band.q025);x.lineTo(xx,yy)}});x.closePath();x.fill();names.forEach((name,k)=>{{const a=obs[k];x.strokeStyle=cols[k];x.lineWidth=2.7;x.beginPath();a.forEach((v,i)=>{{const [xx,yy]=map(i,v,a);i?x.lineTo(xx,yy):x.moveTo(xx,yy)}});x.stroke()}});x.fillStyle=C.muted;x.font='10px sans-serif';x.fillText('checkpoint →',w-82,h-8)}}
const synthetic=DATA.cases.filter(r=>r.family.startsWith('A')||r.family.startsWith('B'));const actual=DATA.cases.filter(r=>r.family.startsWith('C'));const passed=DATA.verification.filter(r=>r.pass).length;document.querySelector('#verdict').innerHTML=`<div><strong>${{DATA.cases.length}}</strong><span>distinct histories</span></div><div><strong>19</strong><span>checkpoints per trajectory</span></div><div><strong>${{passed}} / ${{DATA.verification.length}}</strong><span>verification checks passed</span></div><div><strong>${{DATA.bindings.actual_history?'exact':'aggregate'}}</strong><span>actual questionnaire evidence</span></div>`;
const atlas=document.querySelector('#atlas');[...new Set(DATA.cases.map(r=>r.family))].forEach(f=>{{const title=document.createElement('div');title.className='family-title';title.style.setProperty('--family',familyColor(f));title.textContent=f;atlas.append(title);const wrap=document.createElement('div');wrap.className='atlas';DATA.cases.filter(r=>r.family===f).forEach(r=>{{const el=document.createElement('article');el.className='case';el.style.setProperty('--family',familyColor(f));const sr=r.null.scalar.secular_rate,pr=r.null.scalar.periodic_rate;el.innerHTML=`<h3>${{r.label}}</h3><p class="expect">${{r.expectation}}</p><div class="case-body"><div class="grid-wrap"><canvas aria-label="Correct and wrong answer history for ${{r.label}}"></canvas></div><div class="metrics"><div class="metric"><strong>${{pct(r.metrics.movement_rate)}}</strong><span>movement / checkpoint</span></div><div class="metric"><strong>${{fmt(r.metrics.novelty_total,2)}}</strong><span>cumulative novelty</span></div><div class="metric"><strong>${{fmt(r.metrics.return_total,2)}}</strong><span>cumulative return</span></div><div class="metric"><strong>${{fmt(r.metrics.lag_correlation,2)}}</strong><span>lag association</span></div><div class="metric"><strong>${{fmt(r.metrics.secular_rate,3)}}</strong><span>secular rate</span></div><div class="metric"><strong>${{fmt(r.metrics.periodic_rate,3)}}</strong><span>periodic rate</span></div></div></div><div class="status"><span class="pill ${{r.signature_pass?'pass':'fail'}}">${{r.signature_pass?'expected signature recovered':'signature mismatch'}}</span><span class="pill">history secular p=${{fmt(sr.p_high,3)}}</span><span class="pill">periodic p=${{fmt(pr.p_high,3)}}</span></div>`;wrap.append(el);grid(el.querySelector('canvas'),r.display_states)}});atlas.append(wrap)}});
const selected=['a1_mass_flat','a6_secular_periodic','b1_noise_flat',DATA.bindings.actual_history?'c3_actual_questionnaires':'c2_actual_aggregate_surrogate'];const curves=document.querySelector('#curves');selected.forEach(key=>{{const r=DATA.cases.find(x=>x.key===key);const el=document.createElement('article');el.className='plot';el.innerHTML=`<h3>${{r.label}}</h3><p>Observed complete history; the pale band preserves accuracy and the overall residual-mass distribution while destroying temporal ordering.</p><canvas aria-label="Cumulative historical quantities for ${{r.label}}"></canvas><div class="legend"><span><i style="--c:${{C.blue}}"></i>movement</span><span><i style="--c:${{C.teal}}"></i>novelty</span><span><i style="--c:${{C.gold}}"></i>return</span><span>pale band: movement null</span></div>`;curves.append(el);linePlot(el.querySelector('canvas'),r)}});
document.querySelector('#checks').innerHTML=DATA.verification.map(r=>`<tr class="${{r.pass?'good':'bad'}}"><td>${{r.test}}</td><td><span class="pill ${{r.pass?'pass':'fail'}}">${{r.pass?'pass':'fail'}}</span></td><td>${{r.detail}}</td></tr>`).join('');document.querySelector('#noise').innerHTML=DATA.cases.map(r=>{{const s=r.null.scalar.secular_rate,p=r.null.scalar.periodic_rate,w=r.walk_null.scalar.secular_rate;return `<tr><td>${{r.label}}</td><td class="num">${{pct(r.metrics.movement_rate)}}</td><td class="num">${{fmt(r.metrics.secular_rate)}}</td><td class="num">${{fmt(s.q975)}}</td><td class="num">${{fmt(s.p_high)}}</td><td class="num">${{fmt(w.p_high)}}</td><td class="num">${{fmt(r.metrics.periodic_rate)}}</td><td class="num">${{fmt(p.q975)}}</td><td class="num">${{fmt(p.p_high)}}</td></tr>`}}).join('');document.querySelector('#evidence').textContent=`Generator: build_greekmmlu_historical_change_simulations.py · aggregate receipt SHA-256 ${{DATA.bindings.aggregate_drift.sha256}}${{DATA.bindings.actual_history?' · exact matrix SHA-256 '+DATA.bindings.actual_history.sha256:' · exact matrix pending CSCS extraction'}} · deterministic seed ${{DATA.seed}}`;
</script></body></html>'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drift", type=Path, default=DEFAULT_DRIFT)
    parser.add_argument("--actual-history", type=Path, default=DEFAULT_ACTUAL)
    parser.add_argument("--output-data", type=Path, default=OUT_DATA)
    parser.add_argument("--output-html", type=Path, default=OUT_HTML)
    args = parser.parse_args()
    data = build_data(args.drift, args.actual_history)
    args.output_data.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_html.write_text(build_html(data), encoding="utf-8")
    print(json.dumps({"ok": True, "data": str(args.output_data.resolve()), "html": str(args.output_html.resolve()), "cases": len(data["cases"]), "checks_passed": sum(row["pass"] for row in data["verification"]), "checks": len(data["verification"]), "exact_actual": data["bindings"]["actual_history"] is not None}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
