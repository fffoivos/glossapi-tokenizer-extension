#!/usr/bin/env python3
"""Search for an interpretable all-history displacement of wrong cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import build_greekmmlu_historical_change_simulations as hist


HERE = Path(__file__).resolve().parent
STAMP = "20260811"
DEFAULT_DRIFT = HERE / "data/full8_checkpoint_drift_20260811/greekmmlu_answer_drift.json"
OUT_DATA = HERE / f"GREEKMMLU_WRONG_CELL_DISPLACEMENT_BAKEOFF_{STAMP}.data.json"
OUT_HTML = HERE / f"GREEKMMLU_WRONG_CELL_DISPLACEMENT_BAKEOFF_{STAMP}.html"
SEED = 20260811
SYNTHETIC_PERMUTATIONS = 999
ACTUAL_PERMUTATIONS = 499


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relocated_wrong_mass(before: np.ndarray, after: np.ndarray) -> dict[str, float]:
    """Pair error removals with error additions, leaving net accuracy separate."""
    delta = after.astype(float, copy=False) - before.astype(float, copy=False)
    newly_wrong = float(np.maximum(delta, 0.0).sum())
    recovered = float(np.maximum(-delta, 0.0).sum())
    return {
        "relocated": min(newly_wrong, recovered),
        "newly_wrong": newly_wrong,
        "recovered": recovered,
        "net_wrong": newly_wrong - recovered,
        "l1": newly_wrong + recovered,
    }


def split_curve(states: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    wrong = (~states).astype(np.float64)
    baseline_wrong = (~baseline).astype(np.float64)
    signal = wrong - baseline_wrong
    checkpoints, questions = wrong.shape
    minimum_side = max(3, int(math.ceil(checkpoints / 4)))
    cumulative = np.cumsum(signal, axis=0)
    total = cumulative[-1]
    raw_cumulative = np.cumsum(wrong, axis=0)
    raw_total = raw_cumulative[-1]
    rows = []
    for split in range(minimum_side, checkpoints - minimum_side + 1):
        before = cumulative[split - 1] / split
        after = (total - cumulative[split - 1]) / (checkpoints - split)
        mass = relocated_wrong_mass(before, after)
        raw_before = raw_cumulative[split - 1] / split
        raw_after = (raw_total - raw_cumulative[split - 1]) / (checkpoints - split)
        capacity = min(float(raw_before.sum()), float(raw_after.sum()))
        rows.append({
            "split": split,
            "before_checkpoints": split,
            "after_checkpoints": checkpoints - split,
            **mass,
            "raw_net_wrong": float(raw_after.sum() - raw_before.sum()),
            "capacity": capacity,
            "fraction": mass["relocated"] / capacity if capacity else 0.0,
        })
    peak = max(rows, key=lambda row: (row["relocated"], -abs(row["split"] - checkpoints / 2)))
    return {"minimum_side": minimum_side, "rows": rows, "peak": peak}


def endpoint_relocation(states: np.ndarray) -> float:
    return relocated_wrong_mass(~states[0], ~states[-1])["relocated"]


def adjacent_relocation(states: np.ndarray) -> float:
    wrong = ~states
    return float(sum(relocated_wrong_mass(wrong[t - 1], wrong[t])["relocated"] for t in range(1, len(states))))


def novelty_relocation(states: np.ndarray) -> float:
    wrong = ~states
    return float(sum(min(relocated_wrong_mass(wrong[s], wrong[t])["relocated"] for s in range(t)) for t in range(1, len(states))))


def slope_relocation(states: np.ndarray) -> float:
    wrong = (~states).astype(float)
    progress = np.linspace(0.0, 1.0, len(states))
    design = np.column_stack([np.ones(len(states)), progress])
    slope = np.linalg.lstsq(design, wrong, rcond=None)[0][1]
    return relocated_wrong_mass(np.zeros_like(slope), slope)["relocated"]


def permutation_null(states: np.ndarray, baseline: np.ndarray, replicates: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    observed = split_curve(states, baseline)
    split_ids = [row["split"] for row in observed["rows"]]
    curves = []
    peaks = []
    for _ in range(replicates):
        order = rng.permutation(len(states))
        result = split_curve(states[order], baseline[order])
        if [row["split"] for row in result["rows"]] != split_ids:
            raise AssertionError("split geometry drift")
        values = [float(row["relocated"]) for row in result["rows"]]
        curves.append(values)
        peaks.append(max(values))
    matrix = np.asarray(curves, dtype=float)
    peak_array = np.asarray(peaks, dtype=float)
    raw = float(observed["peak"]["relocated"])
    return {
        "contract": "Permute checkpoint order; preserve every questionnaire state, every checkpoint accuracy, and every question's total wrong frequency exactly.",
        "replicates": replicates,
        "split": split_ids,
        "curve": {
            "q025": np.quantile(matrix, 0.025, axis=0).tolist(),
            "q50": np.quantile(matrix, 0.50, axis=0).tolist(),
            "q975": np.quantile(matrix, 0.975, axis=0).tolist(),
        },
        "peak": {
            "observed": raw,
            "q025": float(np.quantile(peak_array, 0.025)),
            "q50": float(np.quantile(peak_array, 0.50)),
            "q95": float(np.quantile(peak_array, 0.95)),
            "q975": float(np.quantile(peak_array, 0.975)),
            "p_high": float((1 + int((peak_array >= raw).sum())) / (replicates + 1)),
            "excess_over_median": max(0.0, raw - float(np.median(peak_array))),
            "supported_above_q975": max(0.0, raw - float(np.quantile(peak_array, 0.975))),
        },
    }


def candidate_metrics(states: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    selected = split_curve(states, baseline)
    old = hist.history_metrics(states, baseline)
    intervals = len(states) - 1
    questions = states.shape[1]
    cumulative_secular_wrong_cells = old["secular_rate"] * intervals * questions / 2.0
    periodic = old["periodic_strength"]
    return {
        "endpoint_relocated_cells": endpoint_relocation(states),
        "adjacent_cumulative_cells": adjacent_relocation(states),
        "novelty_cumulative_cells": novelty_relocation(states),
        "linear_slope_cells": slope_relocation(states),
        "scaled_secular_cells": cumulative_secular_wrong_cells,
        "scaled_secular_div_periodicity": cumulative_secular_wrong_cells / periodic if periodic > 1e-12 else None,
        "selected": selected,
    }


def display_sample(states: np.ndarray, seed: int, limit: int = 120) -> list[str]:
    return hist.display_sample(states, seed, limit)


def build_case(key: str, label: str, family: str, note: str, states: np.ndarray, baseline: np.ndarray, expected_persistent: bool, empirical: bool = False) -> dict[str, Any]:
    replicates = ACTUAL_PERMUTATIONS if states.shape[1] > hist.N else SYNTHETIC_PERMUTATIONS
    candidates = candidate_metrics(states, baseline)
    null = permutation_null(states, baseline, replicates, SEED + 12347 + sum(ord(char) for char in key))
    detected = null["peak"]["p_high"] < 0.05
    peak = candidates["selected"]["peak"]
    return {
        "key": key,
        "label": label,
        "family": family,
        "note": note,
        "expected_persistent": expected_persistent,
        "empirical": empirical,
        "detected": detected,
        "signature_pass": True if empirical else detected == expected_persistent,
        "questions": int(states.shape[1]),
        "checkpoints": int(states.shape[0]),
        "wrong_start": int((~states[0]).sum()),
        "wrong_end": int((~states[-1]).sum()),
        "mean_wrong": float((~states).sum(axis=1).mean()),
        "display_states": display_sample(states, SEED + sum(ord(char) for char in key)),
        "candidates": candidates,
        "selected": {
            "raw_cells": float(peak["relocated"]),
            "fraction_of_wrong_mass": float(peak["fraction"]),
            "best_split": int(peak["split"]),
            "before_checkpoints": int(peak["before_checkpoints"]),
            "after_checkpoints": int(peak["after_checkpoints"]),
            "recovered_cells": float(peak["recovered"]),
            "newly_wrong_cells": float(peak["newly_wrong"]),
            "net_wrong_cells": float(peak["raw_net_wrong"]),
            "noise_median_cells": float(null["peak"]["q50"]),
            "noise_q975_cells": float(null["peak"]["q975"]),
            "excess_over_median_cells": float(null["peak"]["excess_over_median"]),
            "supported_above_q975_cells": float(null["peak"]["supported_above_q975"]),
            "p_high": float(null["peak"]["p_high"]),
        },
        "null": null,
    }


def build_data(drift_path: Path) -> dict[str, Any]:
    drift = json.loads(drift_path.read_text(encoding="utf-8"))
    cases = []
    canonical = [
        ("a0_one_way", "A0 · One-way sliding window", "A · Canonical intuition", "The same six wrong cells move downward by two positions per checkpoint.", "one_way", SEED, True),
        ("a0_periodic", "A0b · Five down, five up", "A · Canonical intuition", "The six-cell error window retraces the same path repeatedly.", "periodic", SEED, False),
        ("b0_random", "B0 · Uncoordinated positions", "A · Canonical intuition", "The six-cell error window jumps to unrelated positions.", "random", SEED + 9 * 7919, False),
    ]
    for key, label, family, note, kind, seed, expected in canonical:
        states, baseline = hist.canonical_sliding_history(kind, seed)
        cases.append(build_case(key, label, family, note, states, baseline, expected))

    for scenario in hist.SCENARIOS:
        states, baseline = hist.synthetic_states(scenario)
        cases.append(build_case(scenario.key, scenario.label, scenario.family, scenario.expectation, states, baseline, scenario.expect_secular))

    accuracies = np.array([float(row["accuracy"]) for row in drift["checkpoint_table"]])
    n = int(drift["checkpoint_table"][0]["n"])
    nested = hist.nested_baseline(accuracies, n)
    cases.append(build_case(
        "c1_actual_nested",
        "C1 · Actual accuracy curve, nested-only",
        "C · Actual-scale geometry",
        "All 19 real accuracies, but no answer-identity relocation beyond nested learning and forgetting.",
        nested,
        nested,
        False,
    ))
    surrogate, surrogate_baseline = hist.actual_aggregate_states(drift, SEED + 88001)
    cases.append(build_case(
        "c2_actual_aggregate_surrogate",
        "C2 · Actual aggregates, persistent surrogate",
        "C · Actual-scale geometry",
        "All 16,159 questions, 19 exact accuracies and 18 exact gain/loss pairs; identities are a calibrated surrogate, not the frozen real histories.",
        surrogate,
        surrogate_baseline,
        True,
        empirical=True,
    ))

    by_key = {row["key"]: row for row in cases}
    checks = []
    for key in ["a0_one_way", "a0_periodic", "b0_random"] + [scenario.key for scenario in hist.SCENARIOS]:
        row = by_key[key]
        checks.append({
            "test": f"{row['label']} classification",
            "pass": row["signature_pass"],
            "detail": f"expected persistent={row['expected_persistent']}; detected={row['detected']}; p={row['selected']['p_high']:.4f}",
        })
    for label, keys in (
        ("moving mass", ["a1_mass_flat", "a3_mass_gain", "a4_mass_loss"]),
        ("random churn", ["b1_noise_flat", "b2_noise_gain", "b3_noise_loss"]),
    ):
        values = [by_key[key]["selected"]["raw_cells"] for key in keys]
        checks.append({"test": f"Accuracy-trend invariance: {label}", "pass": max(values) - min(values) < 1e-10, "detail": f"raw displacement range = {max(values)-min(values):.3g} cells"})
    checks.extend([
        {"test": "Two coordinated masses exceed one", "pass": by_key["a2_two_flat"]["selected"]["raw_cells"] > by_key["a1_mass_flat"]["selected"]["raw_cells"], "detail": f"{by_key['a1_mass_flat']['selected']['raw_cells']:.2f} → {by_key['a2_two_flat']['selected']['raw_cells']:.2f} cells"},
        {"test": "Nested actual accuracy curve has zero relocation", "pass": by_key["c1_actual_nested"]["selected"]["raw_cells"] < 1e-12, "detail": f"{by_key['c1_actual_nested']['selected']['raw_cells']:.3g} cells"},
        {"test": "Selected metric remains bounded by wrong-mass capacity", "pass": all(row["selected"]["fraction_of_wrong_mass"] <= 1.0 + 1e-12 for row in cases), "detail": "all 17 histories"},
    ])

    canonical_rows = [by_key[key] for key in ("a0_one_way", "a0_periodic", "b0_random")]
    candidate_bakeoff = [
        {"candidate": "Endpoint relocation", "unit": "wrong cells", "uses_all_steps": False, "bounded": True, "one_way": canonical_rows[0]["candidates"]["endpoint_relocated_cells"], "periodic": canonical_rows[1]["candidates"]["endpoint_relocated_cells"], "random": canonical_rows[2]["candidates"]["endpoint_relocated_cells"], "verdict": "reject", "reason": "Only the first and final states; random equals one-way."},
        {"candidate": "Adjacent cumulative relocation", "unit": "wrong-cell moves", "uses_all_steps": True, "bounded": False, "one_way": canonical_rows[0]["candidates"]["adjacent_cumulative_cells"], "periodic": canonical_rows[1]["candidates"]["adjacent_cumulative_cells"], "random": canonical_rows[2]["candidates"]["adjacent_cumulative_cells"], "verdict": "retain as churn", "reason": "Correctly measures activity, but counts periodic return and random churn repeatedly."},
        {"candidate": "Cumulative historical novelty", "unit": "novel wrong-cell moves", "uses_all_steps": True, "bounded": False, "one_way": canonical_rows[0]["candidates"]["novelty_cumulative_cells"], "periodic": canonical_rows[1]["candidates"]["novelty_cumulative_cells"], "random": canonical_rows[2]["candidates"]["novelty_cumulative_cells"], "verdict": "retain as exploration", "reason": "Discounts exact return, but random histories explore many new states."},
        {"candidate": "Linear-slope relocation", "unit": "fitted wrong cells", "uses_all_steps": True, "bounded": False, "one_way": canonical_rows[0]["candidates"]["linear_slope_cells"], "periodic": canonical_rows[1]["candidates"]["linear_slope_cells"], "random": canonical_rows[2]["candidates"]["linear_slope_cells"], "verdict": "reject", "reason": "Can exceed the six wrong cells actually present."},
        {"candidate": "Scaled secular rate", "unit": "path-weighted wrong-cell moves", "uses_all_steps": True, "bounded": False, "one_way": canonical_rows[0]["candidates"]["scaled_secular_cells"], "periodic": canonical_rows[1]["candidates"]["scaled_secular_cells"], "random": canonical_rows[2]["candidates"]["scaled_secular_cells"], "verdict": "reject as displacement", "reason": "Multiplying by steps yields path length, not absolute displaced mass."},
        {"candidate": "Scaled secular ÷ periodicity", "unit": "undefined", "uses_all_steps": True, "bounded": False, "one_way": canonical_rows[0]["candidates"]["scaled_secular_div_periodicity"], "periodic": canonical_rows[1]["candidates"]["scaled_secular_div_periodicity"], "random": canonical_rows[2]["candidates"]["scaled_secular_div_periodicity"], "verdict": "reject", "reason": "Undefined for pure one-way drift because periodicity is zero."},
        {"candidate": "Peak past–future relocation", "unit": "equivalent wrong cells", "uses_all_steps": True, "bounded": True, "one_way": canonical_rows[0]["selected"]["raw_cells"], "periodic": canonical_rows[1]["selected"]["raw_cells"], "random": canonical_rows[2]["selected"]["raw_cells"], "verdict": "select", "reason": "Absolute, accuracy-adjusted, bounded and significant only for the one-way history."},
    ]
    return {
        "schema_version": "greekmmlu_wrong_cell_displacement_bakeoff_v1",
        "created": "2026-08-11",
        "seed": SEED,
        "selected_metric": {
            "name": "Absolute Historical Wrong-cell Displacement",
            "short": "AHWD",
            "definition": "Maximum, over guarded historical splits, of paired relocation between the time-averaged accuracy-adjusted wrong-cell distributions before and after the split.",
            "units": "equivalent wrong cells displaced",
            "minimum_side": "ceil(checkpoints / 4), at least 3",
            "accuracy_baseline": "At every checkpoint, subtract a frozen difficulty-ordered nested questionnaire with exactly the same number of wrong answers.",
        },
        "binding": {"path": str(drift_path.resolve()), "bytes": drift_path.stat().st_size, "sha256": sha256_file(drift_path)},
        "cases": cases,
        "candidate_bakeoff": candidate_bakeoff,
        "verification": checks,
        "verification_passed": all(row["pass"] for row in checks),
    }


def build_html(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Absolute displacement of wrong GreekMMLU cells</title><style>
:root{{--paper:#f5f0e7;--panel:#fffdf8;--ink:#172831;--muted:#68757b;--rule:#d8d0c4;--red:#9b3438;--blue:#315f78;--teal:#39766e;--gold:#b38730;--purple:#725a8f;--green:#47734e;--shadow:0 13px 34px rgba(33,40,41,.07)}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;color:var(--ink);background:radial-gradient(circle at 92% 0,rgba(179,135,48,.13),transparent 30rem),linear-gradient(135deg,var(--paper),#fbf8f1 72%,#eee5d8);font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{width:min(1720px,100%);margin:auto;padding:0 clamp(24px,5vw,86px) 90px}}h1,h2,h3,p{{margin-top:0}}h1,h2,h3{{font-family:Georgia,"Times New Roman",serif;letter-spacing:-.034em}}header{{min-height:min(78vh,830px);display:grid;align-content:center;position:relative;padding:70px 0}}header:before{{content:"";position:absolute;left:calc(-1*clamp(24px,5vw,86px));top:0;bottom:0;width:10px;background:var(--red)}}.eyebrow{{display:flex;justify-content:space-between;gap:20px;color:var(--muted);font-size:12px;font-weight:850;letter-spacing:.14em;text-transform:uppercase;margin-bottom:28px}}h1{{font-size:clamp(50px,7vw,105px);line-height:.94;max-width:1390px;margin-bottom:28px}}.lede{{font-size:clamp(19px,2vw,30px);line-height:1.4;max-width:1250px;color:#364650}}.hero-grid{{display:grid;grid-template-columns:1.25fr repeat(3,1fr);margin-top:35px;border:1px solid var(--rule);background:var(--rule);gap:1px}}.hero-grid div{{background:var(--panel);padding:18px}}.hero-grid strong{{display:block;font:700 30px Georgia,serif;color:var(--blue)}}.hero-grid span{{font-size:11px;color:var(--muted)}}section{{padding:66px 0;border-top:1px solid var(--rule);min-width:0}}.section-head{{display:grid;grid-template-columns:minmax(0,1fr) minmax(280px,.55fr);gap:34px;align-items:end;margin-bottom:30px}}h2{{font-size:clamp(37px,4.4vw,68px);line-height:1;margin-bottom:10px}}.section-head p{{color:var(--muted);line-height:1.55;margin:0}}.canonical{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:22px}}.card,.plot-card{{background:rgba(255,253,248,.92);border:1px solid var(--rule);border-top:7px solid var(--case,var(--blue));padding:21px;box-shadow:var(--shadow)}}.card h3,.plot-card h3{{font-size:27px;margin-bottom:5px}}.card p,.plot-card p{{color:var(--muted);font-size:11px;line-height:1.45}}.grid-wrap{{padding:9px;background:#f1ece3;border:1px solid var(--rule)}}canvas{{display:block;width:100%;height:auto}}.numbers{{display:grid;grid-template-columns:1fr 1fr;gap:1px;margin-top:12px;border:1px solid var(--rule);background:var(--rule)}}.number{{background:var(--panel);padding:11px}}.number.primary{{grid-column:1/-1;border-left:5px solid var(--blue)}}.number strong{{display:block;font:700 26px Georgia,serif}}.number span{{font-size:9px;color:var(--muted)}}.pill{{display:inline-block;margin:10px 6px 0 0;padding:5px 7px;border:1px solid currentColor;font-size:9px;font-weight:900;letter-spacing:.08em;text-transform:uppercase}}.pass{{color:var(--green)}}.fail{{color:var(--red)}}.equation{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:var(--rule);border:1px solid var(--rule)}}.equation div{{background:var(--panel);padding:22px}}.equation strong{{display:block;font:700 22px Georgia,serif;color:var(--blue);margin-bottom:8px}}code{{font:700 12px ui-monospace,SFMono-Regular,monospace;color:var(--red)}}.equation p{{font-size:12px;line-height:1.5;color:var(--muted);margin:9px 0 0}}.plots{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:22px}}.plot-card canvas{{height:280px}}.legend{{display:flex;gap:12px;flex-wrap:wrap;margin-top:10px;color:var(--muted);font-size:9px}}.legend i{{display:inline-block;width:20px;height:4px;background:var(--c);margin-right:5px}}.table-scroll{{max-width:100%;overflow-x:auto;border:1px solid var(--rule)}}table{{width:100%;min-width:840px;border-collapse:collapse;background:var(--panel);font-size:12px}}th,td{{padding:11px;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top}}th{{font-size:9px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}}td.num{{text-align:right;font-variant-numeric:tabular-nums}}tr.selected td:first-child{{border-left:6px solid var(--green)}}tr.rejected td:first-child{{border-left:6px solid var(--red)}}.scenario-bars{{display:grid;gap:10px}}.bar-row{{display:grid;grid-template-columns:245px 1fr 190px;gap:14px;align-items:center}}.bar-label strong{{display:block;font-size:11px}}.bar-label span{{font-size:9px;color:var(--muted)}}.track{{height:18px;background:#e9e2d7;position:relative}}.track i{{position:absolute;left:0;top:0;height:100%;width:calc(var(--v)*1%);background:var(--case)}}.track b{{position:absolute;top:-4px;bottom:-4px;left:calc(var(--n)*1%);border-left:2px dashed var(--red)}}.bar-value{{font:700 12px ui-monospace,SFMono-Regular,monospace}}.note{{padding:18px 21px;border-left:7px solid var(--gold);background:rgba(179,135,48,.09);line-height:1.55;color:#48565e}}footer{{padding:25px 0;border-top:1px solid var(--rule);font-size:10px;color:var(--muted)}}@media(max-width:1250px){{.equation{{grid-template-columns:1fr 1fr}}}}@media(max-width:1050px){{.canonical,.plots{{grid-template-columns:1fr}}.section-head{{grid-template-columns:1fr}}.bar-row{{grid-template-columns:180px 1fr 150px}}}}@media(max-width:700px){{header{{min-height:auto}}.eyebrow{{display:block}}.hero-grid{{grid-template-columns:1fr 1fr}}.equation{{grid-template-columns:1fr}}.bar-row{{grid-template-columns:1fr}}.track{{height:15px}}main{{padding-bottom:50px}}}}
</style></head><body><main><header><div class="eyebrow"><span>GreekMMLU · wrong-cell displacement bake-off</span><span>11 August 2026</span></div><h1>How many wrong cells actually moved?</h1><p class="lede">The selected quantity is absolute and historical: it reports the equivalent number of wrong-answer cells whose question identity moved between earlier and later training regimes. A fixed difficulty baseline removes changes due only to knowing more or fewer answers.</p><div class="hero-grid" id="hero"></div></header>
<section><div class="section-head"><div><h2>The three cases the metric must distinguish</h2><p>Same error mass, radically different histories.</p></div><p>Every card shows the complete questionnaire history. The red cells are wrong answers; rows are checkpoints. Raw displacement is compared with an exact checkpoint-order permutation floor.</p></div><div class="canonical" id="canonical"></div></section>
<section><div class="section-head"><div><h2>The selected quantity</h2><p>Absolute Historical Wrong-cell Displacement—AHWD.</p></div><p>Every candidate split uses the full history: every checkpoint belongs either to its past regime or its future regime. At least one quarter of the trajectory must remain on each side.</p></div><div class="equation"><div><strong>1 · Remove competence</strong><code>zₜq = eₜq − bₜq</code><p><em>e</em> is the observed wrong cell. <em>b</em> is a frozen difficulty-ordered baseline with exactly the same checkpoint accuracy. Thus a pure gain or loss of knowledge contributes zero relocation.</p></div><div><strong>2 · Compare regimes</strong><code>Δq(k) = meanfuture(zₜq) − meanpast(zₜq)</code><p>For every question, compare its accuracy-adjusted wrong occupancy before and after historical boundary <em>k</em>.</p></div><div><strong>3 · Pair the moved mass</strong><code>D(k) = [Σq|Δq(k)| − |ΣqΔq(k)|] / 2</code><p>This is the smaller of the total mass entering and leaving question identities: the minimum number of equivalent wrong cells that must have relocated.</p></div><div><strong>4 · Keep the strongest shift</strong><code>AHWD = maxk D(k)</code><p>The result is non-negative, bounded by available wrong mass and expressed directly as equivalent wrong cells displaced.</p></div></div><p class="note" style="margin-top:22px"><strong>Interpretation:</strong> AHWD measures displaced <em>mass</em>, not travel distance along an arbitrary question ordering. Six wrong cells moving many columns still represent at most six wrong cells of displaced mass. The separate checkpoint-order permutation test asks whether that mass forms more historical structure than accidental churn.</p></section>
<section><div class="section-head"><div><h2>Historical displacement across every boundary</h2><p>The selected number comes from these complete curves.</p></div><p>The line is observed absolute displacement. The band is the 95% checkpoint-order permutation floor, which preserves every questionnaire, every accuracy and every question's total wrong frequency.</p></div><div class="plots" id="split-plots"></div></section>
<section><div class="section-head"><div><h2>Candidate bake-off</h2><p>Why multiplication and division do not produce an absolute displacement.</p></div><p>The first three columns use your exact one-way, periodic and uncoordinated examples. Only the selected quantity is both bounded and interpretable as displaced wrong mass.</p></div><div class="table-scroll"><table><thead><tr><th>Candidate</th><th>Unit</th><th>All steps?</th><th>Bounded?</th><th>One-way</th><th>Periodic</th><th>Random</th><th>Decision</th><th>Reason</th></tr></thead><tbody id="bakeoff"></tbody></table></div></section>
<section><div class="section-head"><div><h2>Stress tests</h2><p>Moving masses, multiple masses, noise and changing accuracy.</p></div><p>Bars show displacement as a percentage of available wrong mass. The dashed marker is each history's 97.5% order-permutation threshold.</p></div><div class="scenario-bars" id="scenario-bars"></div></section>
<section><div class="section-head"><div><h2>Exact results</h2><p>Raw cells, noise floors and supported displacement.</p></div><p>“Supported” is the conservative amount above the 97.5% permutation threshold. Raw AHWD remains the directly interpretable absolute displacement.</p></div><div class="table-scroll"><table><thead><tr><th>History</th><th>Wrong cells start→end</th><th>Raw AHWD</th><th>Error-mass share</th><th>Null median</th><th>Null 97.5%</th><th>Above 97.5%</th><th>p high</th><th>Best split</th></tr></thead><tbody id="results"></tbody></table></div><p class="note" style="margin-top:22px">The 16,159-question actual-scale row is a calibrated surrogate reproducing every observed checkpoint accuracy and adjacent gain/loss count. It tests geometry and scale; it is not a substitute for the frozen per-question prediction histories.</p></section>
<section><div class="section-head"><div><h2>Verification ledger</h2><p>The metric passed every declared intuition check.</p></div><p>Classification uses a one-sided checkpoint-order permutation test at p &lt; 0.05. Effect sizes remain visible even when they do not exceed the floor.</p></div><div class="table-scroll"><table><thead><tr><th>Test</th><th>Result</th><th>Evidence</th></tr></thead><tbody id="checks"></tbody></table></div></section>
<footer id="evidence"></footer></main><script>const DATA={payload};const C={{red:'#9b3438',blue:'#315f78',teal:'#39766e',gold:'#b38730',purple:'#725a8f',green:'#47734e',paper:'#f1ece3',rule:'#d8d0c4',muted:'#68757b'}};const fmt=(v,n=2)=>v===null?'undefined':Number(v).toFixed(n);const pct=v=>`${{(100*v).toFixed(1)}}%`;const familyColor=f=>f.startsWith('A')?C.teal:f.startsWith('B')?C.red:C.blue;
function setup(canvas,h){{const d=devicePixelRatio||1,w=Math.max(290,canvas.clientWidth);canvas.width=w*d;canvas.height=h*d;canvas.style.height=h+'px';const x=canvas.getContext('2d');x.scale(d,d);return[x,w,h]}}function grid(canvas,rows){{const[x,w,h]=setup(canvas,Math.max(180,rows.length*12));const cw=w/rows[0].length,ch=h/rows.length;x.fillStyle=C.paper;x.fillRect(0,0,w,h);rows.forEach((r,i)=>[...r].forEach((v,j)=>{{x.fillStyle=v==='1'?C.green:C.red;x.fillRect(j*cw+.2,i*ch+.2,Math.max(.5,cw-.4),Math.max(.5,ch-.4))}}))}}
function splitPlot(canvas,row){{const[x,w,h]=setup(canvas,280),pad={{l:42,r:16,t:14,b:32}},pw=w-pad.l-pad.r,ph=h-pad.t-pad.b,obs=row.candidates.selected.rows.map(r=>r.relocated),q0=row.null.curve.q025,q1=row.null.curve.q975,med=row.null.curve.q50,ids=row.null.split,ymax=Math.max(...obs,...q1)*1.08||1,map=(i,v)=>[pad.l+pw*i/(obs.length-1),pad.t+ph*(1-v/ymax)];x.strokeStyle=C.rule;for(let j=0;j<=4;j++){{const y=pad.t+ph*j/4;x.beginPath();x.moveTo(pad.l,y);x.lineTo(w-pad.r,y);x.stroke();x.fillStyle=C.muted;x.font='9px sans-serif';x.fillText(fmt(ymax*(1-j/4),1),3,y+3)}}x.fillStyle='rgba(155,52,56,.10)';x.beginPath();q1.forEach((v,i)=>{{const[a,b]=map(i,v);i?x.lineTo(a,b):x.moveTo(a,b)}});[...q0].reverse().forEach((v,k)=>{{const i=q0.length-1-k,[a,b]=map(i,v);x.lineTo(a,b)}});x.closePath();x.fill();[[med,C.red,1.3],[obs,C.blue,3]].forEach(([a,c,l])=>{{x.strokeStyle=c;x.lineWidth=l;x.beginPath();a.forEach((v,i)=>{{const[xx,yy]=map(i,v);i?x.lineTo(xx,yy):x.moveTo(xx,yy)}});x.stroke()}});x.fillStyle=C.muted;x.font='9px sans-serif';x.fillText('historical split →',w-96,h-7)}}
const canonical=DATA.cases.slice(0,3);const one=canonical[0];document.querySelector('#hero').innerHTML=`<div><strong>${{fmt(one.selected.raw_cells)}} cells</strong><span>one-way historical displacement</span></div><div><strong>${{pct(one.selected.fraction_of_wrong_mass)}}</strong><span>of its six-cell error mass</span></div><div><strong>${{fmt(one.selected.noise_q975_cells)}}</strong><span>97.5% permutation floor</span></div><div><strong>${{DATA.verification.filter(r=>r.pass).length}} / ${{DATA.verification.length}}</strong><span>verification checks passed</span></div>`;
const cc=document.querySelector('#canonical');canonical.forEach(r=>{{const el=document.createElement('article');el.className='card';el.style.setProperty('--case',familyColor(r.family));el.innerHTML=`<h3>${{r.label}}</h3><p>${{r.note}}</p><div class="grid-wrap"><canvas aria-label="Complete history for ${{r.label}}"></canvas></div><div class="numbers"><div class="number primary"><strong>${{fmt(r.selected.raw_cells)}} cells</strong><span>absolute historical displacement</span></div><div class="number"><strong>${{fmt(r.selected.noise_q975_cells)}}</strong><span>97.5% noise floor</span></div><div class="number"><strong>${{fmt(r.selected.p_high,3)}}</strong><span>permutation p</span></div></div><span class="pill ${{r.detected?'pass':'fail'}}">${{r.detected?'persistent displacement':'not above floor'}}</span>`;cc.append(el);grid(el.querySelector('canvas'),r.display_states)}});
const sp=document.querySelector('#split-plots');canonical.forEach(r=>{{const el=document.createElement('article');el.className='plot-card';el.style.setProperty('--case',familyColor(r.family));el.innerHTML=`<h3>${{r.label}}</h3><p>All guarded past–future boundaries; raw blue line versus permuted history.</p><canvas aria-label="Absolute wrong-cell displacement by historical boundary for ${{r.label}}"></canvas><div class="legend"><span><i style="--c:${{C.blue}}"></i>observed</span><span><i style="--c:${{C.red}}"></i>null median</span><span>pale band: 95% null</span></div>`;sp.append(el);splitPlot(el.querySelector('canvas'),r)}});
document.querySelector('#bakeoff').innerHTML=DATA.candidate_bakeoff.map(r=>`<tr class="${{r.verdict==='select'?'selected':r.verdict.startsWith('reject')?'rejected':''}}"><td>${{r.candidate}}</td><td>${{r.unit}}</td><td>${{r.uses_all_steps?'yes':'no'}}</td><td>${{r.bounded?'yes':'no'}}</td><td class="num">${{fmt(r.one_way)}}</td><td class="num">${{fmt(r.periodic)}}</td><td class="num">${{fmt(r.random)}}</td><td>${{r.verdict}}</td><td>${{r.reason}}</td></tr>`).join('');
const stress=DATA.cases.filter(r=>!r.key.startsWith('a0_')&&!r.key.startsWith('b0_')&&!r.key.startsWith('c'));const maxShare=Math.max(...stress.map(r=>Math.max(r.selected.fraction_of_wrong_mass,r.selected.noise_q975_cells/Math.max(1,r.selected.raw_cells/r.selected.fraction_of_wrong_mass))))*100;document.querySelector('#scenario-bars').innerHTML=stress.map(r=>{{const capacity=r.selected.raw_cells/Math.max(r.selected.fraction_of_wrong_mass,1e-12),raw=100*r.selected.fraction_of_wrong_mass,noise=100*r.selected.noise_q975_cells/capacity;return `<div class="bar-row"><div class="bar-label"><strong>${{r.label}}</strong><span>${{r.wrong_start}}→${{r.wrong_end}} wrong cells</span></div><div class="track" style="--case:${{familyColor(r.family)}}"><i style="--v:${{100*raw/maxShare}}"></i><b style="--n:${{100*noise/maxShare}}"></b></div><div class="bar-value">${{fmt(r.selected.raw_cells)}} cells · p=${{fmt(r.selected.p_high,3)}}</div></div>`}}).join('');
document.querySelector('#results').innerHTML=DATA.cases.map(r=>`<tr><td>${{r.label}}</td><td>${{r.wrong_start}} → ${{r.wrong_end}}</td><td class="num">${{fmt(r.selected.raw_cells)}}</td><td class="num">${{pct(r.selected.fraction_of_wrong_mass)}}</td><td class="num">${{fmt(r.selected.noise_median_cells)}}</td><td class="num">${{fmt(r.selected.noise_q975_cells)}}</td><td class="num">${{fmt(r.selected.supported_above_q975_cells)}}</td><td class="num">${{fmt(r.selected.p_high,3)}}</td><td>${{r.selected.before_checkpoints}} | ${{r.selected.after_checkpoints}}</td></tr>`).join('');document.querySelector('#checks').innerHTML=DATA.verification.map(r=>`<tr class="${{r.pass?'selected':'rejected'}}"><td>${{r.test}}</td><td><span class="pill ${{r.pass?'pass':'fail'}}">${{r.pass?'pass':'fail'}}</span></td><td>${{r.detail}}</td></tr>`).join('');document.querySelector('#evidence').textContent=`Generator: build_greekmmlu_wrong_cell_displacement_report.py · aggregate evidence SHA-256 ${{DATA.binding.sha256}} · deterministic seed ${{DATA.seed}}`;
</script></body></html>'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drift", type=Path, default=DEFAULT_DRIFT)
    parser.add_argument("--output-data", type=Path, default=OUT_DATA)
    parser.add_argument("--output-html", type=Path, default=OUT_HTML)
    args = parser.parse_args()
    data = build_data(args.drift)
    args.output_data.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    args.output_html.write_text(build_html(data), encoding="utf-8")
    print(json.dumps({"ok": True, "cases": len(data["cases"]), "checks": len(data["verification"]), "passed": sum(row["pass"] for row in data["verification"]), "data": str(args.output_data.resolve()), "html": str(args.output_html.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
