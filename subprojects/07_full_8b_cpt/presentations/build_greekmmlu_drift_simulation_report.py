#!/usr/bin/env python3
"""Stress-test candidate GreekMMLU answer-identity drift measures.

The simulation separates a scalar competence path (a nested item-difficulty
baseline) from balanced identity transformations. It then compares:

1. raw HDI on binary correctness vectors;
2. marginal-adjusted HDI using pairwise excess turnover; and
3. residual HDI after removing a frozen common baseline.

Outputs are deterministic and self-contained: JSON, CSV and one HTML report.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


HERE = Path(__file__).resolve().parent
STAMP = "20260811"
OUT_JSON = HERE / f"GREEKMMLU_DRIFT_SIMULATION_STRESS_TEST_{STAMP}.data.json"
OUT_CSV = HERE / f"GREEKMMLU_DRIFT_SIMULATION_STRESS_TEST_{STAMP}.summary.csv"
OUT_HTML = HERE / f"GREEKMMLU_DRIFT_SIMULATION_STRESS_TEST_{STAMP}.html"

N_ITEMS = 512
MASTER_CHECKPOINTS = 81
CHECKPOINT_COUNTS = [6, 9, 11, 21, 41, 81]
REPLICATES = 64
ORDER_PERMUTATIONS = 128
BASELINE_REPLICATES = 128
BASELINE_NOISE = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40]


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    family: str
    expected: str
    description: str
    ability: str
    transform: str


SCENARIOS = [
    Scenario("static", "Static null", "No identity drift", "none", "No competence change and no identity transformation.", "flat", "none"),
    Scenario("nested_gain", "Nested competence gain", "No identity drift", "none", "The correct set only expands along one fixed difficulty order.", "gain", "none"),
    Scenario("nested_loss", "Nested competence loss", "No identity drift", "none", "The correct set only contracts along the same fixed difficulty order.", "loss", "none"),
    Scenario("periodic_ability", "Periodic competence", "No identity drift", "none", "Accuracy rises and falls, but always along the same nested item ordering.", "periodic", "none"),
    Scenario("monotonic_swap", "Monotonic identity replacement", "Secular drift", "high", "Balanced correct↔wrong swaps accumulate without reversal at fixed accuracy.", "flat", "monotonic"),
    Scenario("periodic_swap", "Periodic identity reversal", "Periodic redistribution", "low", "A fixed transformation grows and then exactly retraces twice.", "flat", "periodic"),
    Scenario("quasi_periodic", "Near-periodic redistribution", "Periodic redistribution", "low-medium", "The same transformation repeatedly expands and contracts with imperfect timing.", "flat", "quasi"),
    Scenario("travelling_window", "Travelling error window", "Periodic redistribution", "medium", "A constant-size transformation moves through question identities and wraps around.", "flat", "window"),
    Scenario("independent_churn", "Independent checkpoint churn", "Interference", "low", "Each checkpoint receives an unrelated balanced set of swapped identities.", "flat", "random"),
    Scenario("markov_churn", "Persistent random churn", "Interference", "low-medium", "A random active set changes slowly, creating smooth but non-directional interference.", "flat", "markov"),
    Scenario("abrupt_shift", "Abrupt permanent replacement", "Secular drift", "high", "One irreversible identity replacement occurs halfway through the run.", "flat", "abrupt"),
    Scenario("gain_plus_monotonic", "Gain + monotonic replacement", "Overlapping", "high", "Scalar competence improves while an independent identity transformation accumulates.", "gain", "monotonic"),
    Scenario("periodic_ability_plus_monotonic", "Periodic ability + monotonic replacement", "Overlapping", "high", "Overall accuracy oscillates while the residual knowledge pattern drifts one way.", "periodic", "monotonic"),
    Scenario("gain_plus_periodic", "Gain + periodic replacement", "Overlapping", "low-medium", "Competence improves while a separate identity transformation cycles out and back.", "gain", "periodic"),
    Scenario("mono_plus_periodic", "Monotonic + periodic transformations", "Overlapping", "medium-high", "A one-way transformation and a disjoint periodic transformation operate simultaneously.", "flat", "mono_periodic"),
    Scenario("two_axis_loop", "Closed two-transformation loop", "Overlapping periodic", "low", "Two disjoint periodic transformations trace a loop and return to their initial state.", "flat", "loop"),
]


def ability_counts(kind: str, count: int = MASTER_CHECKPOINTS) -> np.ndarray:
    u = np.linspace(0.0, 1.0, count)
    if kind == "flat":
        values = np.full(count, 320.0)
    elif kind == "gain":
        values = 260.0 + 120.0 * u
    elif kind == "loss":
        values = 380.0 - 120.0 * u
    elif kind == "periodic":
        values = 320.0 + 60.0 * np.sin(4.0 * np.pi * u)
    else:
        raise ValueError(kind)
    return np.rint(values).astype(int)


def nested_baseline(counts: np.ndarray, order: np.ndarray | None = None) -> np.ndarray:
    order = np.arange(N_ITEMS) if order is None else order
    out = np.zeros((len(counts), N_ITEMS), dtype=np.bool_)
    for t, k in enumerate(counts):
        out[t, order[: int(k)]] = True
    return out


def pair_banks(rng: np.random.Generator) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    donors_a = rng.permutation(np.arange(0, 48))
    receivers_a = rng.permutation(np.arange(464, 512))
    donors_b = rng.permutation(np.arange(48, 96))
    receivers_b = rng.permutation(np.arange(416, 464))
    return (donors_a, receivers_a), (donors_b, receivers_b)


def apply_prefix(mask: np.ndarray, bank: tuple[np.ndarray, np.ndarray], counts: np.ndarray) -> None:
    donors, receivers = bank
    for t, count in enumerate(counts.astype(int)):
        if count:
            mask[t, donors[:count]] ^= True
            mask[t, receivers[:count]] ^= True


def apply_active(mask: np.ndarray, bank: tuple[np.ndarray, np.ndarray], active_by_t: list[np.ndarray]) -> None:
    donors, receivers = bank
    for t, active in enumerate(active_by_t):
        if len(active):
            mask[t, donors[active]] ^= True
            mask[t, receivers[active]] ^= True


def make_transform(kind: str, rng: np.random.Generator) -> np.ndarray:
    u = np.linspace(0.0, 1.0, MASTER_CHECKPOINTS)
    bank_a, bank_b = pair_banks(rng)
    mask = np.zeros((MASTER_CHECKPOINTS, N_ITEMS), dtype=np.bool_)
    if kind == "none":
        return mask
    if kind == "monotonic":
        apply_prefix(mask, bank_a, np.rint(48 * u))
    elif kind == "periodic":
        counts = np.rint(24 * (1 - np.cos(4 * np.pi * u)))
        apply_prefix(mask, bank_a, counts)
    elif kind == "quasi":
        phase = 3.35 * np.pi * u + 0.38 * np.sin(5 * np.pi * u)
        counts = np.rint(48 * np.abs(np.sin(phase)))
        apply_prefix(mask, bank_a, counts)
    elif kind == "window":
        width = 16
        active = []
        for x in u:
            start = int(round(x * 96)) % 48
            active.append(np.array([(start + j) % 48 for j in range(width)], dtype=int))
        apply_active(mask, bank_a, active)
    elif kind == "random":
        active = [np.sort(rng.choice(48, size=24, replace=False)) for _ in range(MASTER_CHECKPOINTS)]
        apply_active(mask, bank_a, active)
    elif kind == "markov":
        current = set(int(x) for x in rng.choice(48, size=24, replace=False))
        active = []
        for _ in range(MASTER_CHECKPOINTS):
            active.append(np.array(sorted(current), dtype=int))
            leaving = rng.choice(np.array(sorted(current)), size=2, replace=False)
            remaining = np.array(sorted(set(range(48)) - current), dtype=int)
            entering = rng.choice(remaining, size=2, replace=False)
            current.difference_update(int(x) for x in leaving)
            current.update(int(x) for x in entering)
        apply_active(mask, bank_a, active)
    elif kind == "abrupt":
        apply_prefix(mask, bank_a, np.where(u < 0.5, 0, 48))
    elif kind == "mono_periodic":
        apply_prefix(mask, bank_a, np.rint(48 * u))
        apply_prefix(mask, bank_b, np.rint(24 * (1 - np.cos(4 * np.pi * u))))
    elif kind == "loop":
        counts_a = np.rint(24 + 24 * np.cos(2 * np.pi * u))
        counts_b = np.rint(24 + 24 * np.sin(2 * np.pi * u))
        apply_prefix(mask, bank_a, counts_a)
        apply_prefix(mask, bank_b, counts_b)
    else:
        raise ValueError(kind)
    return mask


def make_trajectory(scenario: Scenario, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    counts = ability_counts(scenario.ability)
    baseline = nested_baseline(counts)
    transform = make_transform(scenario.transform, rng)
    return np.logical_xor(baseline, transform), baseline


def hamming_matrix(states: np.ndarray) -> np.ndarray:
    x = states.astype(np.int16)
    totals = x.sum(axis=1, dtype=np.int32)
    intersections = x.astype(np.int32) @ x.astype(np.int32).T
    return (totals[:, None] + totals[None, :] - 2 * intersections) / states.shape[1]


def excess_matrix(states: np.ndarray, raw_distance: np.ndarray) -> np.ndarray:
    accuracy = states.mean(axis=1)
    forced = np.abs(accuracy[:, None] - accuracy[None, :])
    return np.maximum(0.0, raw_distance - forced)


def permutation_orders(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((ORDER_PERMUTATIONS, size), dtype=int)
    out[:, 0] = 0
    base = np.arange(1, size)
    for row in range(ORDER_PERMUTATIONS):
        out[row, 1:] = rng.permutation(base)
    return out


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, including exact tie handling."""
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


def lag_distance_correlation(distance: np.ndarray) -> float:
    """Spearman association between elapsed checkpoints and state distance."""
    i, j = np.triu_indices(len(distance), k=1)
    values = distance[i, j]
    if len(values) < 2 or float(values.max() - values.min()) <= 1e-12:
        return 0.0
    lags = (j - i).astype(float)
    rx = average_ranks(lags)
    ry = average_ranks(values)
    corr = np.corrcoef(rx, ry)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


def hdi_from_distance(distance: np.ndarray, orders: np.ndarray) -> dict[str, float]:
    size = len(distance)
    adjacent = distance[np.arange(1, size), np.arange(0, size - 1)]
    path = float(adjacent.sum())
    novelty = np.array([float(distance[t, :t].min()) for t in range(1, size)])
    fresh = float(novelty.sum())
    diameter = float(distance.max())
    hnr = fresh / path if path > 1e-12 else 0.0
    frontier = diameter / path if path > 1e-12 else 0.0
    if path > 1e-12:
        perm_paths = distance[orders[:, 1:], orders[:, :-1]].sum(axis=1)
        order = float((np.count_nonzero(perm_paths + 1e-12 >= path) + 1) / (len(perm_paths) + 1))
    else:
        order = 1.0
    lag_corr = lag_distance_correlation(distance)
    endpoint_distance = float(distance[0, -1])
    endpoint_efficiency = endpoint_distance / path if path > 1e-12 else 0.0
    return {
        "path": path,
        "fresh": fresh,
        "diameter": diameter,
        "hnr": hnr,
        "frontier": frontier,
        "order": order,
        "hdi": hnr * frontier * order,
        "lag_corr": lag_corr,
        "endpoint_distance": endpoint_distance,
        "endpoint_efficiency": endpoint_efficiency,
        "secular_drift": endpoint_distance * max(0.0, lag_corr),
    }


def trajectory_metrics(states: np.ndarray, baseline: np.ndarray, indices: np.ndarray, seed: int) -> dict[str, dict[str, float]]:
    x = states[indices]
    b = baseline[indices]
    raw_d = hamming_matrix(x)
    adjusted_d = excess_matrix(x, raw_d)
    residual_d = hamming_matrix(np.logical_xor(x, b))
    orders = permutation_orders(len(indices), seed)
    return {
        "raw": hdi_from_distance(raw_d, orders),
        "adjusted": hdi_from_distance(adjusted_d, orders),
        "residual": hdi_from_distance(residual_d, orders),
        "accuracy": {
            "start": float(x[0].mean()),
            "end": float(x[-1].mean()),
            "range": float(x.mean(axis=1).max() - x.mean(axis=1).min()),
        },
    }


def quantiles(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(a)),
        "q10": float(np.quantile(a, 0.10)),
        "q90": float(np.quantile(a, 0.90)),
        "mean": float(a.mean()),
    }


def summarize(records: list[dict]) -> list[dict]:
    rows = []
    for scenario in SCENARIOS:
        for checkpoints in CHECKPOINT_COUNTS:
            subset = [r for r in records if r["scenario"] == scenario.key and r["checkpoints"] == checkpoints]
            row = {
                "scenario": scenario.key,
                "label": scenario.label,
                "family": scenario.family,
                "expected": scenario.expected,
                "description": scenario.description,
                "checkpoints": checkpoints,
            }
            for variant in ("raw", "adjusted", "residual"):
                for metric in ("hdi", "hnr", "frontier", "order", "path", "diameter", "lag_corr", "endpoint_distance", "endpoint_efficiency", "secular_drift"):
                    row[f"{variant}_{metric}"] = quantiles([r[variant][metric] for r in subset])
            for metric in ("start", "end", "range"):
                row[f"accuracy_{metric}"] = quantiles([r["accuracy"][metric] for r in subset])
            rows.append(row)
    return rows


def noisy_order(rng: np.random.Generator, noise: float) -> np.ndarray:
    if noise == 0:
        return np.arange(N_ITEMS)
    score = np.arange(N_ITEMS, dtype=float) + rng.normal(0.0, noise * N_ITEMS, size=N_ITEMS)
    return np.argsort(score)


def baseline_sensitivity() -> list[dict]:
    targets = [next(s for s in SCENARIOS if s.key == key) for key in ("nested_gain", "gain_plus_monotonic")]
    indices = np.linspace(0, MASTER_CHECKPOINTS - 1, 21).round().astype(int)
    rows = []
    for target in targets:
        for noise in BASELINE_NOISE:
            values = []
            paths = []
            for rep in range(BASELINE_REPLICATES):
                seed = 880000 + rep * 101 + sum(ord(c) for c in target.key)
                states, true_baseline = make_trajectory(target, seed)
                counts = ability_counts(target.ability)
                estimate = nested_baseline(counts, noisy_order(np.random.default_rng(seed + 77), noise))
                residual = np.logical_xor(states[indices], estimate[indices])
                distance = hamming_matrix(residual)
                orders = permutation_orders(len(indices), seed + 99)
                metric = hdi_from_distance(distance, orders)
                values.append(metric["hdi"])
                paths.append(metric["path"])
            rows.append({
                "scenario": target.key,
                "label": target.label,
                "rank_noise": noise,
                "residual_hdi": quantiles(values),
                "residual_path": quantiles(paths),
            })
    return rows


def triangle_counterexample() -> dict:
    # A={q1}, B={q1,q2}, C={q2}. Both A and C are nested in B, but A and C swap identity.
    states = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.bool_)
    raw = hamming_matrix(states)
    adjusted = excess_matrix(states, raw)
    return {
        "states": {"A": [1, 0], "B": [1, 1], "C": [0, 1]},
        "E_AB": float(adjusted[0, 1]),
        "E_BC": float(adjusted[1, 2]),
        "E_AC": float(adjusted[0, 2]),
        "triangle_holds": bool(adjusted[0, 2] <= adjusted[0, 1] + adjusted[1, 2] + 1e-12),
    }


def run_simulations() -> dict:
    records = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        for rep in range(REPLICATES):
            seed = 20260811 + scenario_index * 100003 + rep * 7919
            states, baseline = make_trajectory(scenario, seed)
            for checkpoints in CHECKPOINT_COUNTS:
                indices = np.linspace(0, MASTER_CHECKPOINTS - 1, checkpoints).round().astype(int)
                metrics = trajectory_metrics(states, baseline, indices, seed + checkpoints * 1009)
                records.append({"scenario": scenario.key, "replicate": rep, "checkpoints": checkpoints, **metrics})
    summary = summarize(records)
    return {
        "metadata": {
            "created": "2026-08-11",
            "items": N_ITEMS,
            "master_checkpoints": MASTER_CHECKPOINTS,
            "checkpoint_counts": CHECKPOINT_COUNTS,
            "scenarios": len(SCENARIOS),
            "replicates_per_scenario": REPLICATES,
            "trajectory_evaluations": len(records),
            "order_permutations_per_metric": ORDER_PERMUTATIONS,
            "baseline_replicates": BASELINE_REPLICATES,
            "seed_root": 20260811,
        },
        "scenario_summary": summary,
        "baseline_sensitivity": baseline_sensitivity(),
        "triangle_counterexample": triangle_counterexample(),
    }


def compact(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def write_csv(data: dict) -> None:
    rows = [r for r in data["scenario_summary"] if r["checkpoints"] == 21]
    fields = [
        "scenario", "label", "family", "expected", "raw_hdi", "adjusted_hdi", "residual_hdi", "residual_secular_drift",
        "raw_path", "adjusted_path", "residual_path", "accuracy_start", "accuracy_end", "accuracy_range",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "scenario": row["scenario"],
                "label": row["label"],
                "family": row["family"],
                "expected": row["expected"],
                "raw_hdi": compact(row["raw_hdi"]["median"]),
                "adjusted_hdi": compact(row["adjusted_hdi"]["median"]),
                "residual_hdi": compact(row["residual_hdi"]["median"]),
                "residual_secular_drift": compact(row["residual_secular_drift"]["median"]),
                "raw_path": compact(row["raw_path"]["median"]),
                "adjusted_path": compact(row["adjusted_path"]["median"]),
                "residual_path": compact(row["residual_path"]["median"]),
                "accuracy_start": compact(row["accuracy_start"]["median"]),
                "accuracy_end": compact(row["accuracy_end"]["median"]),
                "accuracy_range": compact(row["accuracy_range"]["median"]),
            })


def build_html(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Stress-testing historical knowledge drift</title>
<style>
:root{{--paper:#f5f0e7;--panel:#fffdf8;--ink:#172831;--muted:#68757b;--rule:#d8d0c4;--red:#9b3438;--blue:#315f78;--teal:#39766e;--gold:#b38730;--purple:#725a8f;--green:#47734e;--shadow:0 14px 34px rgba(33,40,41,.07)}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:radial-gradient(circle at 92% 0,rgba(179,135,48,.13),transparent 29rem),linear-gradient(135deg,var(--paper),#fbf8f1 73%,#eee5d8);color:var(--ink);font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}.page{{width:min(1720px,100%);margin:auto;padding:0 clamp(24px,5vw,88px) 100px}}h1,h2,h3,p{{margin-top:0}}h1,h2,h3{{font-family:Georgia,"Times New Roman",serif;letter-spacing:-.034em}}h1{{font-size:clamp(52px,7vw,108px);line-height:.94;max-width:1450px;margin-bottom:28px}}h2{{font-size:clamp(36px,4.6vw,70px);line-height:1;max-width:1320px;margin-bottom:16px}}h3{{font-size:clamp(21px,2vw,31px)}}p,li,td,th{{font-size:clamp(14px,1vw,18px);line-height:1.5}}.hero{{min-height:min(92vh,1060px);display:grid;align-content:center;position:relative;padding:72px 0}}.hero:before{{content:"";position:absolute;left:calc(-1*clamp(24px,5vw,88px));top:0;bottom:0;width:10px;background:var(--red)}}.eyebrow{{display:flex;justify-content:space-between;gap:20px;color:var(--muted);font-size:12px;font-weight:850;letter-spacing:.14em;text-transform:uppercase;margin-bottom:30px}}.title-rule{{width:114px;height:8px;background:var(--red);margin-bottom:28px}}.lede{{max-width:1220px;color:#364650;font-size:clamp(20px,2vw,31px);line-height:1.4}}.finding{{max-width:1430px;border-left:7px solid var(--gold);padding:4px 0 4px 27px;margin-top:38px;font:700 clamp(23px,2.6vw,41px)/1.22 Georgia,serif}}.nav{{display:flex;flex-wrap:wrap;gap:10px 24px;margin-top:43px;padding-top:18px;border-top:1px solid var(--rule)}}.nav a{{color:var(--muted);font-size:13px;font-weight:780;text-decoration:none}}.section{{padding:clamp(74px,9vw,138px) 0 0}}.head{{display:grid;grid-template-columns:8px minmax(0,1fr);gap:24px;margin-bottom:34px}}.bar{{background:var(--accent,var(--red));min-height:86px}}.kicker{{color:var(--accent,var(--red));font-size:12px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;margin-bottom:9px}}.intro{{max-width:1200px;color:var(--muted)}}.verdict{{padding:clamp(24px,3vw,42px);border:1px solid var(--rule);border-top:9px solid var(--red);background:rgba(255,253,248,.9);box-shadow:var(--shadow)}}.verdict h3{{font-size:clamp(29px,3vw,49px)}}.verdict-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--rule);border:1px solid var(--rule);margin-top:26px}}.verdict-grid div{{background:var(--panel);padding:21px}}.verdict-grid strong{{display:block;font:700 clamp(24px,2.4vw,40px) Georgia,serif;color:var(--red)}}.verdict-grid span{{color:var(--muted);font-size:12px}}.equations{{display:grid;gap:18px}}.equation-card{{display:grid;grid-template-columns:270px minmax(0,1fr);gap:26px;padding:25px 0;border-top:1px solid var(--rule)}}.equation-card:first-child{{border-top:7px solid var(--blue)}}.equation-name{{font:700 27px Georgia,serif}}.equation-name small{{display:block;margin-top:6px;color:var(--muted);font:750 11px Inter,sans-serif;text-transform:uppercase;letter-spacing:.09em}}.symbolic{{display:flex;align-items:center;flex-wrap:wrap;gap:.18em;padding:15px 19px;background:var(--panel);border:1px solid var(--rule);font:700 clamp(19px,1.65vw,31px)/1.4 Georgia,serif;overflow-x:auto}}.symbolic sub,.symbolic sup{{font-size:.56em}}.frac{{display:inline-grid;grid-template-rows:auto auto;text-align:center;line-height:1.05;margin:0 .12em}}.frac span:first-child{{padding:0 .25em .12em;border-bottom:1.5px solid currentColor}}.frac span:last-child{{padding:.12em .25em 0}}.plain{{color:var(--muted);margin:9px 0 0}}.callout{{padding:25px;border-left:7px solid var(--gold);background:rgba(179,135,48,.08)}}.callout strong{{font-family:Georgia,serif;font-size:22px}}.chart-card{{padding:clamp(18px,2vw,31px);border:1px solid var(--rule);background:rgba(255,253,248,.88);box-shadow:var(--shadow);margin-bottom:30px}}.chart-title{{display:flex;justify-content:space-between;gap:26px;align-items:end;margin-bottom:22px}}.chart-title p{{max-width:720px;color:var(--muted);margin-bottom:4px}}.metric-legend{{display:flex;flex-wrap:wrap;gap:10px 20px;color:var(--muted);font-size:12px}}.metric-legend span:before{{content:"";display:inline-block;width:24px;height:6px;margin-right:7px;transform:translateY(-2px);background:var(--c)}}.bar-table{{display:grid;gap:9px}}.bar-row{{display:grid;grid-template-columns:280px minmax(0,1fr) 220px;gap:16px;align-items:center;padding-bottom:9px;border-bottom:1px solid rgba(216,208,196,.7)}}.bar-label strong{{display:block;font-size:13px}}.bar-label span{{color:var(--muted);font-size:10px}}.tracks{{display:grid;gap:3px}}.track{{height:9px;background:#ebe5dc;position:relative}}.track i{{display:block;height:100%;background:var(--c);width:calc(var(--v)*100%)}}.values{{display:grid;grid-template-columns:repeat(4,1fr);gap:5px;font:700 12px Georgia,serif;text-align:right}}.small-multiples{{display:grid;grid-template-columns:repeat(2,1fr);gap:25px}}.plot{{min-width:0}}.plot h3{{font-size:22px;margin-bottom:5px}}svg{{display:block;width:100%;height:auto;overflow:visible}}.axis{{stroke:#a9a196;stroke-width:1}}.grid{{stroke:#ddd6cc;stroke-width:1;stroke-dasharray:3 4}}.axis-text{{fill:var(--muted);font:10px Inter,sans-serif}}.plot-line{{fill:none;stroke-width:2.5}}.plot-dot{{stroke:var(--panel);stroke-width:1.5}}.table-wrap{{overflow:auto;border:1px solid var(--rule)}}table{{border-collapse:collapse;width:100%;min-width:1220px}}th,td{{padding:12px 14px;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top}}th{{background:#e9e1d5;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.07em;position:sticky;top:0}}td:first-child{{font-weight:800}}.num{{font-variant-numeric:tabular-nums;text-align:right}}.good{{color:var(--green);font-weight:850}}.bad{{color:var(--red);font-weight:850}}.warn{{color:var(--gold);font-weight:850}}.scenario-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--rule);border:1px solid var(--rule)}}.scenario-grid article{{background:var(--panel);padding:18px}}.scenario-grid h3{{font-size:19px;margin-bottom:7px}}.scenario-grid p{{color:var(--muted);font-size:12px;margin:0}}.scenario-grid .tag{{display:inline-block;margin-bottom:9px;color:var(--teal);font-size:10px;font-weight:900;text-transform:uppercase;letter-spacing:.08em}}.counterexample{{display:grid;grid-template-columns:1fr 1fr;gap:28px}}.state-box{{padding:26px;border-top:7px solid var(--purple);background:var(--panel);border-left:1px solid var(--rule);border-right:1px solid var(--rule);border-bottom:1px solid var(--rule)}}.state-row{{display:grid;grid-template-columns:30px 1fr auto;gap:12px;align-items:center;margin:10px 0}}.bits{{display:flex;gap:6px}}.bit{{width:34px;height:34px;display:grid;place-items:center;color:white;font-weight:850;background:var(--red)}}.bit.one{{background:var(--green)}}.recommendation{{padding:30px;border-left:8px solid var(--teal);background:rgba(57,118,110,.09);font:700 clamp(22px,2vw,34px)/1.3 Georgia,serif}}.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:28px}}.panel{{padding:25px;border:1px solid var(--rule);border-top:7px solid var(--blue);background:rgba(255,253,248,.76)}}.panel ul{{padding-left:20px;margin-bottom:0}}.evidence{{color:var(--muted);font-size:12px;margin-top:30px}}code{{overflow-wrap:anywhere}}@media(max-width:1100px){{.verdict-grid,.scenario-grid{{grid-template-columns:1fr 1fr}}.equation-card,.counterexample,.two-col{{grid-template-columns:1fr}}.bar-row{{grid-template-columns:220px minmax(0,1fr) 180px}}}}@media(max-width:690px){{.hero{{min-height:auto}}.eyebrow,.chart-title{{display:block}}.verdict-grid,.small-multiples,.scenario-grid{{grid-template-columns:1fr}}.bar-row{{grid-template-columns:1fr}}.values{{text-align:left}}.page{{padding-bottom:58px}}.section{{padding-top:76px}}.symbolic{{font-size:18px}}}}
</style></head><body><main class="page">
<header class="hero"><div class="eyebrow"><span>GreekMMLU · knowledge-identity analysis</span><span>11 August 2026</span></div><div class="title-rule"></div><h1>Does the model know different things—or simply more things?</h1><p class="lede">A deterministic stress test of historical drift metrics across scalar learning, forgetting, periodicity, random churn, irreversible replacement and overlapping transformations.</p><div class="finding">Raw HDI confuses learning with drift. Residual HDI fixes that, but still assigns substantial drift to a smooth closed loop. Competence level, residual movement and persistent historical displacement must remain separate.</div><nav class="nav"><a href="#verdict">Verdict</a><a href="#definitions">Equations</a><a href="#simulations">Simulations</a><a href="#checkpoint-count">Checkpoint count</a><a href="#baseline">Common baseline</a><a href="#recommendation">Recommendation</a></nav></header>

<section class="section" id="verdict" style="--accent:var(--red)"><div class="head"><div class="bar"></div><div><div class="kicker">01 · Direct answer</div><h2>The current metric does not yet capture the intended notion.</h2><p class="intro">It detects coherent movement through binary answer states, but it cannot tell whether that movement was forced by total accuracy—and smoothness alone does not rule out a loop.</p></div></div><div class="verdict"><h3>Pure nested gain is the decisive competence counterexample.</h3><p>Suppose every later checkpoint retains every previously correct answer and adds new correct answers in one stable difficulty order. Nothing is exchanged or forgotten. Raw Hamming movement is nevertheless one-way, historically novel and permutation-coherent—so raw HDI approaches 1.</p><div class="verdict-grid" id="verdict-grid"></div><div class="callout" id="loop-verdict" style="margin-top:24px"></div></div></section>

<section class="section" id="definitions" style="--accent:var(--blue)"><div class="head"><div class="bar"></div><div><div class="kicker">02 · Definitions</div><h2>Subtract “how much” before measuring “what.”</h2><p class="intro">The first correction is exact and baseline-free for each pair of checkpoints. A historical geometry, however, requires a common reference.</p></div></div><div class="equations">
<article class="equation-card"><div class="equation-name">Raw change<small>all answer turnover</small></div><div><div class="symbolic"><i>D</i><sub>ts</sub> = <span class="frac"><span>1</span><span>N</span></span> &Sigma;<sub>i=1</sub><sup>N</sup> 𝟙[<i>x</i><sub>t,i</sub> ≠ <i>x</i><sub>s,i</sub>]</div><p class="plain">This includes both net improvement or decline and exchanges in question identity.</p></div></article>
<article class="equation-card"><div class="equation-name">Competence-only floor<small>minimum forced movement</small></div><div><div class="symbolic"><i>B</i><sub>ts</sub> = |<i>p</i><sub>t</sub> − <i>p</i><sub>s</sub>| &nbsp; where &nbsp; <i>p</i><sub>t</sub> = <span class="frac"><span>1</span><span>N</span></span>&Sigma;<sub>i=1</sub><sup>N</sup><i>x</i><sub>t,i</sub></div><p class="plain">If the smaller correct set is contained in the larger, this is all the movement required by the accuracy change.</p></div></article>
<article class="equation-card"><div class="equation-name">Excess turnover<small>baseline-free pair statistic</small></div><div><div class="symbolic"><i>E</i><sub>ts</sub> = <i>D</i><sub>ts</sub> − |<i>p</i><sub>t</sub> − <i>p</i><sub>s</sub>| = 2 min(<i>G</i><sub>ts</sub>, <i>L</i><sub>ts</sub>)</div><p class="plain">G is the fraction gained and L the fraction lost. E is zero for a purely nested gain or loss and positive only when gains and losses coexist.</p></div></article>
<article class="equation-card"><div class="equation-name">Frozen common baseline<small>historical residual state</small></div><div><div class="symbolic"><i>b</i><sub>t,i</sub> = 𝟙[<i>δ</i><sub>i</sub> ≤ <i>α</i><sub>t</sub>] &nbsp;;&nbsp; <i>z</i><sub>t,i</sub> = <i>x</i><sub>t,i</sub> ⊕ <i>b</i><sub>t,i</sub></div><p class="plain">δ is a fixed question-difficulty order. α changes only the predicted amount known. The residual z records which question identities depart from that common competence baseline.</p></div></article>
<article class="equation-card"><div class="equation-name">Residual drift<small>apply historical geometry here</small></div><div><div class="symbolic"><i>D</i><sup>res</sup><sub>ts</sub> = <i>d</i><sub>H</sub>(<i>z</i><sub>t</sub>,<i>z</i><sub>s</sub>) &nbsp;;&nbsp; <i>RHDI</i> = <i>HNR</i>(<i>D</i><sup>res</sup>) × <i>FE</i>(<i>D</i><sup>res</sup>) × <i>C</i><sub>order</sub>(<i>D</i><sup>res</sup>)</div><p class="plain">This is the revised candidate for coherent identity drift. The report still exposes all three components; the product is not treated as a literature standard.</p></div></article>
<article class="equation-card"><div class="equation-name">Secular residual drift<small>persistent amount × temporal trend</small></div><div><div class="symbolic"><i>ρ</i><sub>lag</sub> = Spearman(<i>D</i><sup>res</sup><sub>ts</sub>, |<i>τ</i><sub>t</sub>−<i>τ</i><sub>s</sub>|) &nbsp;;&nbsp; <i>SRD</i> = <i>D</i><sup>res</sup><sub>0T</sub> × max(0,<i>ρ</i><sub>lag</sub>)</div><p class="plain">D<sup>res</sup><sub>0T</sub> is the residual endpoint displacement. ρ<sub>lag</sub> asks whether checkpoints farther apart in training time are systematically farther apart in what they know. SRD is zero for an exact closed loop and retains absolute effect size.</p></div></article>
</div><div class="callout"><strong>Important impossibility:</strong> pairwise excess turnover E is useful, but it is not a metric. If every nested gain/loss were assigned zero distance without a shared baseline, triangle inequality can fail. Therefore E can describe adjacent churn; it cannot safely replace a common baseline for recurrence, frontier or path geometry.</div></section>

<section class="section" id="simulations" style="--accent:var(--gold)"><div class="head"><div class="bar"></div><div><div class="kicker">03 · 6,144 trajectory evaluations</div><h2>Sixteen mechanisms, sixty-four seeds, six checkpoint densities.</h2><p class="intro">Every trajectory contains 512 questions. A fixed nested difficulty baseline controls accuracy; balanced transformations then alter question identity without changing the intended competence level.</p></div></div><div class="scenario-grid" id="scenario-grid"></div><figure class="chart-card"><div class="chart-title"><div><h3>At 21 checkpoints: three HDIs versus secular residual drift</h3><p>Bars show medians over 64 reproducible seeded realizations; exact values appear at right. SRD is an absolute fraction of question states, whereas each HDI is a scale-free trajectory-shape score.</p></div><div class="metric-legend"><span style="--c:var(--red)">raw HDI</span><span style="--c:var(--gold)">pair-adjusted HDI</span><span style="--c:var(--teal)">residual HDI</span><span style="--c:var(--blue)">secular residual drift</span></div></div><div class="bar-table" id="metric-bars"></div></figure><div class="table-wrap"><table id="scenario-table"><thead><tr><th>Case</th><th>Intuitive class</th><th>Accuracy start→end</th><th>Raw HDI</th><th>Adjusted HDI</th><th>Residual HDI</th><th>Secular residual drift</th><th>Residual path</th><th>Reading</th></tr></thead><tbody></tbody></table></div></section>

<section class="section" id="checkpoint-count" style="--accent:var(--purple)"><div class="head"><div class="bar"></div><div><div class="kicker">04 · Checkpoint-count sensitivity</div><h2>The metric is not invariant to how often we look.</h2><p class="intro">The same underlying 81-point trajectories are thinned to 6, 9, 11, 21, 41 and 81 checkpoints. Periodic and stochastic paths are especially sensitive because sparse sampling can miss reversals or alias cycles.</p></div></div><div class="chart-card"><div class="chart-title"><div><h3>Median score versus number of observed checkpoints</h3><p>Representative cases. HDI panels use 0–1; secular residual drift uses 0–0.20 because it retains absolute magnitude.</p></div><div class="metric-legend" id="scenario-legend"></div></div><div class="small-multiples"><div class="plot"><h3>Raw HDI</h3><svg id="raw-density" viewBox="0 0 520 330" role="img" aria-label="Raw HDI by checkpoint count"></svg></div><div class="plot"><h3>Pair-adjusted HDI</h3><svg id="adjusted-density" viewBox="0 0 520 330" role="img" aria-label="Pair-adjusted HDI by checkpoint count"></svg></div><div class="plot"><h3>Residual HDI</h3><svg id="residual-density" viewBox="0 0 520 330" role="img" aria-label="Residual HDI by checkpoint count"></svg></div><div class="plot"><h3>Secular residual drift</h3><svg id="secular-density" viewBox="0 0 520 330" role="img" aria-label="Secular residual drift by checkpoint count"></svg></div></div></div><div class="two-col"><article class="panel"><h3>What changes with denser checkpoints?</h3><ul><li>More checkpoints reveal reversals that a sparse grid can miss.</li><li>Random threshold crossings accumulate path length and reduce frontier efficiency.</li><li>Permutation coherence changes because the null distribution has a different number of states.</li><li>Binary correctness amplifies small score changes around the answer boundary.</li></ul></article><article class="panel"><h3>Comparability rule</h3><ul><li>Evaluate every run on one frozen token-indexed checkpoint grid.</li><li>Report sensitivity under predeclared thinning of that grid.</li><li>Report transition churn per billion tokens, not per checkpoint.</li><li>Use continuous answer margins for the primary residual trajectory.</li></ul></article></div></section>

<section class="section" id="baseline" style="--accent:var(--teal)"><div class="head"><div class="bar"></div><div><div class="kicker">05 · Common-baseline sensitivity</div><h2>Yes, use a common baseline—but freeze and audit it.</h2><p class="intro">An oracle difficulty order removes pure nested gain exactly. A misspecified order creates false residual motion, so baseline quality must be treated as part of the measurement contract.</p></div></div><div class="counterexample"><div class="state-box"><h3>Why pairwise adjustment cannot define history</h3><div class="state-row"><strong>A</strong><div class="bits"><span class="bit one">1</span><span class="bit">0</span></div><span>{{q₁}}</span></div><div class="state-row"><strong>B</strong><div class="bits"><span class="bit one">1</span><span class="bit one">1</span></div><span>{{q₁,q₂}}</span></div><div class="state-row"><strong>C</strong><div class="bits"><span class="bit">0</span><span class="bit one">1</span></div><span>{{q₂}}</span></div><div class="symbolic"><i>E</i>(A,B)=0, &nbsp; <i>E</i>(B,C)=0, &nbsp; but &nbsp; <i>E</i>(A,C)=1</div><p class="plain">A and C are both nested inside B, yet they disagree completely. Zero + zero cannot bound one: triangle inequality fails.</p></div><div class="chart-card" style="margin:0"><h3>Baseline rank error</h3><p class="plain">Residual HDI at 21 checkpoints as the frozen item order is increasingly corrupted.</p><svg id="baseline-noise" viewBox="0 0 620 380" role="img" aria-label="Residual HDI sensitivity to baseline rank noise"></svg></div></div><div class="two-col" style="margin-top:30px"><article class="panel"><h3>Preferred real-data baseline</h3><p>Use each question’s continuous correct-option margin and fit a frozen-difficulty model:</p><div class="symbolic"><i>m</i><sub>t,i</sub> = <i>α</i><sub>t</sub> − <i>δ</i><sub>i</sub> + <i>r</i><sub>t,i</sub></div><p>α captures “more or less overall”; δ is fixed question difficulty; r is identity-specific residual knowledge. Compute temporal distances on r, not on thresholded accuracy alone.</p></article><article class="panel"><h3>Freeze without leakage</h3><ul><li>Estimate δ from the pre-CPT checkpoint, an external model panel, or a separately frozen calibration set.</li><li>Do not estimate the baseline from the full trajectory and then evaluate the same trajectory without disclosure.</li><li>Stratify or weight by subject and educational level so composition does not masquerade as drift.</li><li>Retain binary excess turnover as a transparent secondary diagnostic.</li></ul></article></div></section>

<section class="section" id="recommendation" style="--accent:var(--teal)"><div class="head"><div class="bar"></div><div><div class="kicker">06 · Recommended measurement contract</div><h2>Separate level, churn and historical drift.</h2></div></div><div class="recommendation">Do not report raw HDI—or residual HDI alone—as “knowledge drift.” Primary trajectory: continuous residual distances relative to one frozen item-difficulty baseline. If one scalar is required, report secular residual drift (persistent endpoint displacement × lag-distance trend), beside its two components. Secondary: pairwise excess turnover.</div><div class="two-col" style="margin-top:30px"><article class="panel"><h3>Per checkpoint transition</h3><ul><li>Net competence change: Δp or change in mean correct-option margin.</li><li>Gains G and losses L separately.</li><li>Excess turnover E = 2 min(G,L).</li><li>Excess-turnover rate per billion training tokens.</li></ul></article><article class="panel"><h3>Across the whole history</h3><ul><li>Residual distance matrix on continuous margins.</li><li>Endpoint displacement and lag-distance association.</li><li>Residual HNR, frontier efficiency and order coherence as diagnostics.</li><li>Checkpoint-thinning and rolling-endpoint sensitivity.</li></ul></article></div><div class="callout" style="margin-top:28px"><strong>SRD is a candidate, not a magic scalar.</strong> It deliberately counts only persistent displacement, so its value depends on the declared endpoint. Report it on the full run and on predeclared rolling horizons; never let it replace the residual distance matrix or the gain/loss table.</div><p class="evidence">Reproducible artifact: <code>{OUT_JSON.name}</code> · summary: <code>{OUT_CSV.name}</code> · generator: <code>{Path(__file__).name}</code>. Synthetic results only; no Apertus checkpoint predictions are altered or re-evaluated here.</p></section>
</main><script>const DATA={payload};
const $=s=>document.querySelector(s), fmt=(v,d=2)=>Number(v).toFixed(d), median=(r,k)=>r[k].median;
const at21=DATA.scenario_summary.filter(r=>r.checkpoints===21);
const nested=at21.find(r=>r.scenario==='nested_gain');
const loop=at21.find(r=>r.scenario==='two_axis_loop');
$('#verdict-grid').innerHTML=`<div><strong>${{fmt(median(nested,'raw_hdi'))}}</strong><span>raw HDI · false positive</span></div><div><strong>${{fmt(median(nested,'adjusted_hdi'))}}</strong><span>pair-adjusted HDI</span></div><div><strong>${{fmt(median(nested,'residual_hdi'))}}</strong><span>common-baseline residual HDI</span></div>`;
$('#loop-verdict').innerHTML=`<strong>A second counterexample remains after baseline correction.</strong> The closed two-transformation loop scores residual HDI = ${{fmt(median(loop,'residual_hdi'))}}, even though its endpoint returns exactly to its initial residual state. Secular residual drift = ${{fmt(median(loop,'residual_secular_drift'))}} because persistent endpoint displacement is zero.`;
$('#scenario-grid').innerHTML=at21.map(r=>`<article><span class="tag">${{r.family}}</span><h3>${{r.label}}</h3><p>${{r.description}}</p></article>`).join('');
const colors=['#9b3438','#b38730','#39766e','#315f78'];
$('#metric-bars').innerHTML=at21.map(r=>{{const vals=[median(r,'raw_hdi'),median(r,'adjusted_hdi'),median(r,'residual_hdi'),median(r,'residual_secular_drift')];return `<div class="bar-row"><div class="bar-label"><strong>${{r.label}}</strong><span>${{r.family}}</span></div><div class="tracks">${{vals.map((v,i)=>`<div class="track" style="--c:${{colors[i]}}"><i style="--v:${{Math.min(1,v)}}"></i></div>`).join('')}}</div><div class="values">${{vals.map(v=>`<span>${{fmt(v)}}</span>`).join('')}}</div></div>`}}).join('');
const reading=r=>{{if(r.family==='No identity drift')return median(r,'raw_hdi')>.15?'Raw metric confuses level with identity.':'Correct null.';if(r.scenario==='two_axis_loop')return 'Residual HDI remains too high; endpoint persistence corrects it.';if(r.family.includes('Periodic')||r.scenario==='gain_plus_periodic')return 'Recurrence suppresses persistent displacement.';if(r.family==='Interference')return 'Novel patterns are not necessarily directional.';return 'Residual trajectory isolates the identity component.'}};
$('#scenario-table tbody').innerHTML=at21.map(r=>`<tr><td>${{r.label}}</td><td>${{r.expected}}</td><td>${{fmt(median(r,'accuracy_start')*100,1)}}% → ${{fmt(median(r,'accuracy_end')*100,1)}}%</td><td class="num">${{fmt(median(r,'raw_hdi'))}}</td><td class="num">${{fmt(median(r,'adjusted_hdi'))}}</td><td class="num">${{fmt(median(r,'residual_hdi'))}}</td><td class="num">${{fmt(median(r,'residual_secular_drift'))}}</td><td class="num">${{fmt(median(r,'residual_path'))}}</td><td>${{reading(r)}}</td></tr>`).join('');
const NS='http://www.w3.org/2000/svg'; function node(name,attrs={{}},text=''){{const e=document.createElementNS(NS,name);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text)e.textContent=text;return e}}
function lineChart(svg,series,xvals,ymax=1){{const W=520,H=330,m={{l:48,r:15,t:17,b:42}},x=v=>m.l+(v-xvals[0])/(xvals.at(-1)-xvals[0])*(W-m.l-m.r),y=v=>H-m.b-v/ymax*(H-m.t-m.b);[0,.25,.5,.75,1].forEach(f=>{{const v=f*ymax;svg.append(node('line',{{x1:m.l,y1:y(v),x2:W-m.r,y2:y(v),class:'grid'}}));svg.append(node('text',{{x:m.l-8,y:y(v)+3,'text-anchor':'end',class:'axis-text'}},fmt(v,2)))}});xvals.forEach(v=>svg.append(node('text',{{x:x(v),y:H-m.b+20,'text-anchor':'middle',class:'axis-text'}},String(v))));svg.append(node('line',{{x1:m.l,y1:m.t,x2:m.l,y2:H-m.b,class:'axis'}}));svg.append(node('line',{{x1:m.l,y1:H-m.b,x2:W-m.r,y2:H-m.b,class:'axis'}}));series.forEach(s=>{{const pts=s.values.map((v,i)=>`${{x(xvals[i])}},${{y(v)}}`).join(' ');svg.append(node('polyline',{{points:pts,class:'plot-line',stroke:s.color}}));s.values.forEach((v,i)=>svg.append(node('circle',{{cx:x(xvals[i]),cy:y(v),r:3.6,fill:s.color,class:'plot-dot'}})))}});svg.append(node('text',{{x:(m.l+W-m.r)/2,y:H-4,'text-anchor':'middle',class:'axis-text'}},'observed checkpoints'))}}
const selected=['nested_gain','monotonic_swap','periodic_swap','independent_churn','markov_churn','two_axis_loop']; const palette=['#9b3438','#39766e','#b38730','#725a8f','#315f78','#7d5d3f'];
$('#scenario-legend').innerHTML=selected.map((key,i)=>`<span style="--c:${{palette[i]}}">${{at21.find(r=>r.scenario===key).label}}</span>`).join('');
['raw','adjusted','residual'].forEach(kind=>{{const series=selected.map((key,i)=>({{color:palette[i],values:DATA.metadata.checkpoint_counts.map(c=>median(DATA.scenario_summary.find(r=>r.scenario===key&&r.checkpoints===c),kind+'_hdi'))}}));lineChart($('#'+kind+'-density'),series,DATA.metadata.checkpoint_counts)}});
const secularSeries=selected.map((key,i)=>({{color:palette[i],values:DATA.metadata.checkpoint_counts.map(c=>median(DATA.scenario_summary.find(r=>r.scenario===key&&r.checkpoints===c),'residual_secular_drift'))}}));lineChart($('#secular-density'),secularSeries,DATA.metadata.checkpoint_counts,.2);
function baselineChart(){{const svg=$('#baseline-noise'),W=620,H=380,m={{l:52,r:20,t:22,b:50}},rows=DATA.baseline_sensitivity,xvals=[...new Set(rows.map(r=>r.rank_noise))],x=v=>m.l+v/.4*(W-m.l-m.r),y=v=>H-m.b-v*(H-m.t-m.b);[0,.25,.5,.75,1].forEach(v=>{{svg.append(node('line',{{x1:m.l,y1:y(v),x2:W-m.r,y2:y(v),class:'grid'}}));svg.append(node('text',{{x:m.l-8,y:y(v)+3,'text-anchor':'end',class:'axis-text'}},fmt(v,2)))}});xvals.forEach(v=>svg.append(node('text',{{x:x(v),y:H-m.b+20,'text-anchor':'middle',class:'axis-text'}},fmt(v,2))));svg.append(node('line',{{x1:m.l,y1:m.t,x2:m.l,y2:H-m.b,class:'axis'}}));svg.append(node('line',{{x1:m.l,y1:H-m.b,x2:W-m.r,y2:H-m.b,class:'axis'}}));[['nested_gain','#9b3438'],['gain_plus_monotonic','#39766e']].forEach(([key,color])=>{{const s=rows.filter(r=>r.scenario===key);svg.append(node('polyline',{{points:s.map(r=>`${{x(r.rank_noise)}},${{y(r.residual_hdi.median)}}`).join(' '),class:'plot-line',stroke:color}}));s.forEach(r=>{{svg.append(node('line',{{x1:x(r.rank_noise),x2:x(r.rank_noise),y1:y(r.residual_hdi.q10),y2:y(r.residual_hdi.q90),stroke:color,'stroke-width':1.2}}));svg.append(node('circle',{{cx:x(r.rank_noise),cy:y(r.residual_hdi.median),r:4,fill:color,class:'plot-dot'}}))}});svg.append(node('text',{{x:x(s.at(-1).rank_noise)-4,y:y(s.at(-1).residual_hdi.median)-8,'text-anchor':'end',fill:color,class:'axis-text'}},s[0].label))}});svg.append(node('text',{{x:(m.l+W-m.r)/2,y:H-7,'text-anchor':'middle',class:'axis-text'}},'noise added to frozen question-difficulty rank'))}}baselineChart();
</script></body></html>'''


def main() -> None:
    data = run_simulations()
    OUT_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(data)
    OUT_HTML.write_text(build_html(data), encoding="utf-8")
    print(json.dumps({
        "json": str(OUT_JSON),
        "csv": str(OUT_CSV),
        "html": str(OUT_HTML),
        "trajectory_evaluations": data["metadata"]["trajectory_evaluations"],
    }, indent=2))


if __name__ == "__main__":
    main()
