#!/usr/bin/env python3
"""Build a receipt-bound comparison of the completed 8B D0 run and 0.5B factorial."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ARMS = (
    "D0_mixed",
    "D1_hard_h_to_g",
    "D2_hard_g_to_h",
    "D3_gradual_h_to_g",
    "D4_gradual_g_to_h",
)
LEARNING = (
    "hplt", "non_hplt", "openarchives", "greek_phd",
    "historical_polytonic", "neutral_external_modern_greek",
)
RETENTION = ("english", "code", "math", "de", "ru", "zh", "old_greek")
LABELS = {
    "hplt": "HPLT broad Greek", "non_hplt": "GlossAPI / non-HPLT",
    "openarchives": "OpenArchives", "greek_phd": "Greek PhD",
    "historical_polytonic": "Historical polytonic",
    "neutral_external_modern_greek": "Neutral external Greek",
    "english": "English", "code": "Code", "math": "Math",
    "de": "German", "ru": "Russian", "zh": "Chinese",
    "old_greek": "Old Greek",
}
MINI_TOKENS_PER_UPDATE = 2_097_152
FULL_TOKENS_PER_UPDATE = 4_194_304

MINI_HASHES = {
    "core_campaign_summary.json": "eb8a2db1949bbc2be3108c273d947f58ec086e88b13fd7ed9e7ed7338bf8591e",
    "dataset_order_selection_analysis.json": "581f59294fd4f728bd1ff318dfa2c1d4fb52914b0d49e6001b83ae8a6e677b19",
    "full_endpoint_validation_receipt.json": "6e49e02e67e0baf75acd965e49445fd5bf1e657552d9bd3cfec5187953311146",
    "greekmmlu_trajectory.json": "74e56f40ab05e982aac7dc6cebcae1422e3fa97db15525b035f92581d26211cb",
    "validation_trajectory.json": "bfb7c7073adee541f4e3ed74548b343cf58d60b93249a54f2a900937d5972298",
}
def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite(child, f"{path}[{index}]")


def build_payload(repo: Path) -> dict[str, Any]:
    mini_root = repo / "subprojects/06_dataset_scheduling_experiments/presentations/data/dataset_order_20260805"
    mini_inputs: dict[str, dict] = {}
    bindings = []
    for name, expected in MINI_HASHES.items():
        path = mini_root / name
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"0.5B evidence drift: {path}: {actual} != {expected}")
        mini_inputs[name] = read_json(path)
        bindings.append({"role": f"mini_{name}", "path": str(path.resolve()), "sha256": actual})

    full_path = repo / "subprojects/07_full_8b_cpt/presentations/FULL8_SANITIZED_CPT_FINAL_RESULTS_20260811.data.json"
    full = read_json(full_path)
    full_sha = sha256_file(full_path)
    bindings.append({"role": "full8_completed_campaign", "path": str(full_path.resolve()), "sha256": full_sha})
    if (
        full.get("completion", {}).get("status") != "completed"
        or int(full["meta"]["snapshot_iteration"]) != int(full["meta"]["planned_updates"])
        or int(full["meta"]["greekmmlu_complete_milestones"]) != 19
    ):
        raise ValueError("8B final campaign evidence is incomplete")

    mini_gm_raw = mini_inputs["greekmmlu_trajectory.json"]
    mini_val_raw = mini_inputs["validation_trajectory.json"]
    core = mini_inputs["core_campaign_summary.json"]
    selection = mini_inputs["dataset_order_selection_analysis.json"]
    if mini_gm_raw.get("row_count") != 415 or mini_val_raw.get("row_count") != 5395:
        raise ValueError("0.5B trajectory row count drift")
    if core.get("status") != "completed" or selection.get("status") != "completed":
        raise ValueError("0.5B campaign is not complete")
    if selection["decision"]["provisional_observed_leader"] != "D0_mixed":
        raise ValueError("0.5B selection evidence drift")

    mini_gm = {arm: [] for arm in ARMS}
    for row in mini_gm_raw["rows"]:
        mini_gm[row["arm_id"]].append({
            "iteration": int(row["iteration"]),
            "tokens_b": int(row["iteration"]) * MINI_TOKENS_PER_UPDATE / 1e9,
            "accuracy": float(row["clean_accuracy"]),
            "nll": float(row["clean_choice_nll"]),
            "bpb": float(row["clean_correct_answer_bpb"]),
        })
    for arm in ARMS:
        mini_gm[arm].sort(key=lambda row: row["iteration"])
        if len(mini_gm[arm]) != 83 or mini_gm[arm][0]["iteration"] != 0 or mini_gm[arm][-1]["iteration"] != 38_496:
            raise ValueError(f"0.5B GreekMMLU trajectory drift: {arm}")

    full_gm = []
    for row in full["new_greekmmlu"]:
        full_gm.append({
            "iteration": int(row["iteration"]),
            "tokens_b": int(row["iteration"]) * FULL_TOKENS_PER_UPDATE / 1e9,
            "accuracy": float(row["clean_accuracy"]),
            "nll": float(row["clean_choice_nll"]),
            "bpb": float(row["clean_correct_answer_bpb"]),
        })
    full_gm.sort(key=lambda row: row["iteration"])
    if len(full_gm) != 19 or full_gm[0]["iteration"] != 0 or full_gm[-1]["iteration"] != 18_284:
        raise ValueError("8B GreekMMLU checkpoint set is incomplete")

    mini_validation = {arm: {panel: [] for panel in (*LEARNING, *RETENTION)} for arm in ARMS}
    for row in mini_val_raw["rows"]:
        panel = row["panel"]
        if panel in mini_validation[row["arm_id"]]:
            mini_validation[row["arm_id"]][panel].append({
                "iteration": int(row["iteration"]),
                "tokens_b": int(row["iteration"]) * MINI_TOKENS_PER_UPDATE / 1e9,
                "bpb": float(row["bpb"]),
            })
    full_validation = {}
    for panel in (*LEARNING, *RETENTION):
        rows = full["new_validation"][panel]
        full_validation[panel] = [{
            "iteration": int(row["iteration"]),
            "tokens_b": int(row["iteration"]) * FULL_TOKENS_PER_UPDATE / 1e9,
            "bpb": float(row["bpb"]),
        } for row in rows]
        if len(rows) < 2:
            raise ValueError(f"insufficient 8B validation points: {panel}")
    if any(len(mini_validation[arm][panel]) != 83 for arm in ARMS for panel in (*LEARNING, *RETENTION)):
        raise ValueError("0.5B validation trajectory length drift")

    mini_initial_rows = {
        row["panel"]: row for row in mini_val_raw["rows"]
        if row["arm_id"] == "D0_mixed" and int(row["iteration"]) == 0
    }
    panel_geometry = {}
    geometry_keys = ("base_target_count", "added_target_count", "base_target_bytes", "added_target_bytes")
    for panel in (*LEARNING, *RETENTION):
        a = mini_initial_rows[panel]
        b = full["new_validation"][panel][0]
        panel_geometry[panel] = {
            "exact_match": all(round(float(a[key]), 6) == round(float(b[key]), 6) for key in geometry_keys),
            "mini": {key: float(a[key]) for key in geometry_keys},
            "full8": {key: float(b[key]) for key in geometry_keys},
        }
    exact_panel_matches = sum(row["exact_match"] for row in panel_geometry.values())

    mini_endpoints = {}
    for arm in ARMS:
        rows = mini_gm[arm]
        endpoint = rows[-1]
        best_nll = min(rows, key=lambda row: row["nll"])
        best_accuracy = max(rows, key=lambda row: row["accuracy"])
        mini_endpoints[arm] = {
            **endpoint,
            "best_nll": best_nll["nll"], "best_nll_tokens_b": best_nll["tokens_b"],
            "best_accuracy": best_accuracy["accuracy"],
            "best_accuracy_tokens_b": best_accuracy["tokens_b"],
            "nll_rebound": endpoint["nll"] - best_nll["nll"],
        }
    full_endpoint = full_gm[-1]
    full_best_nll = min(full_gm, key=lambda row: row["nll"])
    full_best_accuracy = max(full_gm, key=lambda row: row["accuracy"])
    target_tokens = full_endpoint["tokens_b"]
    mini_matched = {
        arm: min(mini_gm[arm], key=lambda row: abs(row["tokens_b"] - target_tokens))
        for arm in ARMS
    }
    comparison = {
        "target_tokens_b": target_tokens,
        "full8": full_endpoint,
        "mini": mini_matched,
        "full8_vs_mini_d0": {
            "accuracy_pp": 100 * (full_endpoint["accuracy"] - mini_matched["D0_mixed"]["accuracy"]),
            "nll": full_endpoint["nll"] - mini_matched["D0_mixed"]["nll"],
            "bpb": full_endpoint["bpb"] - mini_matched["D0_mixed"]["bpb"],
        },
    }

    payload = {
        "meta": {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "mini_final_tokens_b": 80_729_939_067 / 1e9,
            "mini_updates": 38_496,
            "full_planned_tokens_b": int(full["meta"]["active_tokens"]) / 1e9,
            "full_snapshot_tokens_b": float(full["meta"]["snapshot_tokens_b"]),
            "full_snapshot_iteration": int(full["meta"]["snapshot_iteration"]),
            "full_latest_greekmmlu_iteration": full_endpoint["iteration"],
            "full_latest_greekmmlu_tokens_b": full_endpoint["tokens_b"],
            "full_greekmmlu_points": len(full_gm),
            "mini_greekmmlu_points": sum(len(rows) for rows in mini_gm.values()),
            "clean_n": 16_159,
            "panel_geometry_exact_matches": exact_panel_matches,
            "panel_geometry_count": len(panel_geometry),
            "raw_points_no_smoothing": True,
        },
        "contract": {
            "mini": {
                "model": "swiss-ai/Apertus-v1.1-0.5B",
                "layers": 20, "hidden": 1024, "heads": 16, "kv_heads": 4,
                "tied_embeddings": True,
                "initialization": "mini-specific tied Token Distillation",
                "tokens": 80_729_939_067, "tokens_per_update": MINI_TOKENS_PER_UPDATE,
                "updates": 38_496, "peak_lr": 1.5e-4, "floor_lr": 1.5e-5,
                "warmup_updates": 800, "cooldown_start": 30_796,
                "data": "decontaminated; before Apertus-standard PII masking; replay multiplicity preserved",
            },
            "full8": {
                "model": "swiss-ai/Apertus-8B-2509",
                "layers": 32, "hidden": 4096, "heads": 32, "kv_heads": 8,
                "tied_embeddings": False,
                "initialization": "untied layer-11 Token Distillation + output calibration",
                "tokens": int(full["meta"]["active_tokens"]),
                "tokens_per_update": FULL_TOKENS_PER_UPDATE,
                "updates": int(full["meta"]["planned_updates"]),
                "peak_lr": 5.5e-5, "floor_lr": 5.5e-6,
                "warmup_updates": 400, "cooldown_start": 14_627,
                "data": "decontaminated; Apertus PII masking; global exact post-mask dedup",
            },
            "shared": {
                "tokenizer_sha": "bbb08e71929b519c5c2362338b0fc6a0e99955cb8fdbf0729ae1311117e6561b",
                "vocab": 148_992, "sequence_length": 4096,
                "mix": "79% Modern Greek / 20% foreign replay / 1% Old Greek",
                "optimizer": "AdEMAMix; beta1 .9, beta2 .999, beta3 .999, alpha 4, WD .1, clip .1",
                "objective": "Goldfish k50/h50; cross-document attention and EOD-loss masking",
                "greekmmlu": "dascim/GreekMMLU @ 6a03aa…; clean n=16,159; float32; zero-shot",
            },
        },
        "mini_greekmmlu": mini_gm,
        "full_greekmmlu": full_gm,
        "mini_validation": mini_validation,
        "full_validation": full_validation,
        "mini_endpoints": mini_endpoints,
        "full_endpoint": {
            **full_endpoint,
            "best_nll": full_best_nll["nll"],
            "best_nll_tokens_b": full_best_nll["tokens_b"],
            "best_accuracy": full_best_accuracy["accuracy"],
            "best_accuracy_tokens_b": full_best_accuracy["tokens_b"],
            "nll_rebound": full_endpoint["nll"] - full_best_nll["nll"],
        },
        "comparison": comparison,
        "panel_geometry": panel_geometry,
        "selection": {
            "provisional_observed_leader": selection["decision"]["provisional_observed_leader"],
            "winner_selected": selection["decision"]["winner_selected"],
            "passing_arms": selection["point_estimate_selection"]["passing_arms"],
        },
        "data_delta": full["data"],
        "labels": LABELS,
        "bindings": bindings,
    }
    finite(payload)
    return payload


TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Apertus CPT across scale — completed 0.5B factorial and 8B D0</title>
  <style>
    :root{--paper:#f7f3ea;--white:#fffdf8;--ink:#102131;--muted:#65717b;--rule:#d6cfc2;--red:#c64b45;--blue:#315f86;--teal:#277e74;--gold:#bd8730;--purple:#775ba6;--green:#3f7f57;--shadow:0 18px 48px rgba(20,30,38,.085)}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;color:var(--ink);background:radial-gradient(circle at 92% 1%,rgba(189,135,48,.12),transparent 24rem),linear-gradient(135deg,var(--paper),#fbf8f1 72%,#f1eadf);font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:#8d2e32;text-decoration-thickness:1px;text-underline-offset:3px}h1,h2,h3,p{margin-top:0}h1,h2{font-family:Georgia,"Times New Roman",serif;letter-spacing:-.035em}h1{max-width:1300px;font-size:clamp(52px,7vw,108px);line-height:.95;margin-bottom:24px}h2{max-width:1180px;font-size:clamp(34px,4.4vw,68px);line-height:1.02;margin-bottom:18px}h3{font:700 clamp(22px,2.2vw,34px)/1.08 Georgia,serif}p,li,td,th{font-size:clamp(14px,1.05vw,18px);line-height:1.5}.page{width:min(1680px,100%);margin:auto;padding:0 clamp(24px,5vw,82px) 90px}.hero{min-height:min(88vh,1100px);display:grid;align-content:center;position:relative;padding:72px 0}.hero:before{content:"";position:absolute;left:calc(-1 * clamp(24px,5vw,82px));top:0;bottom:0;width:10px;background:var(--red)}.eyebrow{display:flex;justify-content:space-between;gap:24px;color:var(--muted);font-size:12px;font-weight:850;letter-spacing:.14em;text-transform:uppercase;margin-bottom:30px}.title-rule{width:104px;height:8px;background:var(--red);margin-bottom:26px}.lede{max-width:1120px;color:#34424e;font-size:clamp(20px,2vw,31px);line-height:1.43}.finding{max-width:1370px;margin:42px 0 0;padding:23px 0 0 28px;border-left:7px solid var(--gold);font:700 clamp(25px,2.8vw,43px)/1.22 Georgia,serif}.nav{display:flex;flex-wrap:wrap;gap:10px 22px;margin-top:43px;padding-top:18px;border-top:1px solid var(--rule)}.nav a{color:var(--muted);font-size:13px;font-weight:780;text-decoration:none}
    section{padding:clamp(72px,9vw,132px) 0 0}.head{display:grid;grid-template-columns:8px minmax(0,1fr);gap:23px;margin-bottom:32px}.bar{background:var(--accent,var(--red));min-height:82px}.kicker{color:var(--accent,var(--red));font-size:12px;font-weight:900;letter-spacing:.13em;text-transform:uppercase;margin-bottom:9px}.intro{max-width:1120px;color:var(--muted)}.figure{margin:0 0 30px;padding:clamp(14px,1.8vw,26px);border:1px solid var(--rule);background:rgba(255,253,248,.84);box-shadow:var(--shadow)}.figure svg{display:block;width:100%;height:auto}.figure figcaption{display:flex;justify-content:space-between;gap:24px;color:var(--muted);font-size:12px;line-height:1.45;margin-top:12px}.hero-chart{padding:24px 28px 20px}.hero-chart svg{min-height:600px}.chart-title{font:700 21px Georgia,serif}.pair{display:grid;grid-template-columns:1fr 1fr;gap:28px}.facets{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}.facet{padding:16px;border-top:6px solid var(--facet,var(--blue));background:rgba(255,253,248,.72)}.facet h3{font-size:21px;margin-bottom:3px}.facet .sub{color:var(--muted);font-size:12px;margin-bottom:7px}.facet svg{width:100%;height:auto}.legend{display:flex;flex-wrap:wrap;gap:10px 23px;margin:0 0 20px;color:var(--muted);font-size:13px}.legend span:before{content:"";display:inline-block;width:23px;height:4px;margin-right:7px;transform:translateY(-2px);background:var(--legend)}.legend .thick:before{height:6px}.callout{margin:26px 0 0;padding:18px 0 0 23px;border-left:6px solid var(--gold);color:#3b4954}.callout strong{color:var(--ink)}table{width:100%;border-collapse:collapse;background:rgba(255,255,255,.22)}th,td{padding:11px 13px;border-bottom:1px solid var(--rule);vertical-align:top}th{color:var(--muted);font-size:11px;letter-spacing:.07em;text-transform:uppercase;text-align:left}td:not(:first-child),th:not(:first-child){text-align:right;font-variant-numeric:tabular-nums}.table-wrap{overflow-x:auto}.best{color:var(--green);font-weight:850}.warn{color:#8d2e32;font-weight:850}.contract td:first-child{font-weight:800}.contract td:not(:first-child),.contract th:not(:first-child){text-align:left}.metric-strip{display:grid;grid-template-columns:repeat(4,1fr);margin:32px 0;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule)}.metric-strip article{padding:23px;border-left:1px solid var(--rule)}.metric-strip article:first-child{border-left:0;padding-left:0}.metric-strip .n{font:700 clamp(30px,3vw,50px)/1 Georgia,serif}.metric-strip .l{color:var(--muted);font-size:12px;margin-top:7px}.two-col{display:grid;grid-template-columns:1fr 1fr;gap:44px}.audit{padding-top:16px;border-top:6px solid var(--audit,var(--blue))}.audit code{font-size:12px;overflow-wrap:anywhere}.small{color:var(--muted);font-size:13px}.status{display:inline-block;border-radius:999px;padding:5px 10px;background:#f3dcda;color:#872f32;font-size:11px;font-weight:900;letter-spacing:.07em;text-transform:uppercase}
    @media(max-width:1050px){.pair,.facets,.two-col{grid-template-columns:1fr}.metric-strip{grid-template-columns:1fr 1fr}.hero-chart svg{min-height:auto}}@media(max-width:650px){.hero{min-height:auto}.eyebrow,.figure figcaption{flex-direction:column}.metric-strip{grid-template-columns:1fr}.metric-strip article{border-left:0;border-top:1px solid var(--rule);padding:18px 0}.page{padding-bottom:55px}section{padding-top:72px}}@media print{body{background:white}.page{width:100%;padding:0 30px 40px}.hero{min-height:auto}.figure{box-shadow:none}.nav{display:none}}
  </style>
</head>
<body><main class="page">
  <header class="hero">
    <div class="eyebrow"><span>APERTUS CPT · CROSS-SCALE EVIDENCE</span><span>BOTH CAMPAIGNS COMPLETE · 11 AUG 2026</span></div><div class="title-rule"></div>
    <h1>What the five-arm 0.5B screen predicts—and what the 8B run actually does</h1>
    <p class="lede">The completed 0.5B factorial isolates data order across five otherwise matched 80.73B-token trajectories. The completed 8B campaign carries forward stationary D0 mixing on a larger untied model and a newly anonymized, globally post-mask-deduplicated 76.69B-token corpus.</p>
    <p class="finding">At matched late checkpoints near 76B tokens, 8B reaches 54.85% clean GreekMMLU accuracy and 1.1221 choice NLL versus 42.25% and 1.2705 for 0.5B D0: +12.61 accuracy points and −0.1484 NLL. Yet 8B correct-answer BPB is 0.0082 higher, showing that answer discrimination and absolute continuation likelihood do not move identically across scale.</p>
    <nav class="nav"><a href="#contract">Contract</a><a href="#mmlu">Large GreekMMLU plots</a><a href="#learning">Learning loss</a><a href="#forgetting">Forgetting</a><a href="#interpretation">Interpretation</a><a href="#evidence">Evidence</a></nav>
  </header>

  <section id="contract" style="--accent:var(--blue)"><div class="head"><span class="bar"></span><div><div class="kicker">01 · Comparison contract</div><h2>Same question; different model, initialization, and cleaned corpus</h2><p class="intro">The 0.5B arm comparison is causal for temporal order because its five arms share all non-factor controls. The 8B comparison is observational: D0 was carried forward, but model geometry, embedding tying, absolute LR, global batch, anonymization, and exact-dedup scope changed.</p></div></div>
    <div class="pair"><figure class="figure"><svg id="mix" role="img"></svg><figcaption><span>Five 0.5B schedule arms; 8B uses D0 only.</span><span>Foreign 20% and Old Greek 1% remain stationary.</span></figcaption></figure><figure class="figure"><svg id="lr" role="img"></svg><figcaption><span>Relative WSD-10 geometry is nearly identical.</span><span>Both warmups consume 1.678B tokens.</span></figcaption></figure></div>
    <div class="table-wrap"><table class="contract"><thead><tr><th>Dimension</th><th>0.5B factorial</th><th>Completed 8B D0</th><th>Comparison consequence</th></tr></thead><tbody id="contract-body"></tbody></table></div>
    <p class="callout"><strong>Dataset delta.</strong> The 8B rerun masks Apertus-standard email, IP, and validated IBAN patterns and then applies global exact post-mask deduplication. This removes 4.044B active tokens (5.01%) relative to the 0.5B schedule corpus and shifts the Modern-Greek HPLT/GlossAPI split from 69.06/30.94 to 68.52/31.48. This was not a factor in the five-arm experiment.</p>
  </section>

  <section id="mmlu" style="--accent:var(--red)"><div class="head"><span class="bar"></span><div><div class="kicker">02 · Primary benchmark evidence</div><h2>GreekMMLU, large enough to read the trajectory</h2><p class="intro">Every point uses the same native Greek benchmark revision, clean 16,159-question subset, zero-shot answer ranking, and float32 evaluator. Raw checkpoint points are connected without smoothing. The 8B series includes the corrected update-0 RoPE anchor.</p></div></div>
    <div class="legend"><span class="thick" style="--legend:var(--red)">8B · D0 mixed</span><span style="--legend:var(--blue)">0.5B · D0 mixed</span><span style="--legend:var(--teal)">0.5B · hard H→G</span><span style="--legend:var(--gold)">0.5B · hard G→H</span><span style="--legend:var(--purple)">0.5B · gradual H→G</span><span style="--legend:var(--green)">0.5B · gradual G→H</span></div>
    <figure class="figure hero-chart"><svg id="gm-accuracy" role="img"></svg><figcaption><span>Clean zero-shot accuracy · higher is better.</span><span>All raw checkpoints; active-token x-axis.</span></figcaption></figure>
    <figure class="figure hero-chart"><svg id="gm-nll" role="img"></svg><figcaption><span>Clean multiple-choice NLL · lower is better.</span><span>Continuous metric; primary small-model diagnostic.</span></figcaption></figure>
    <figure class="figure hero-chart"><svg id="gm-bpb" role="img"></svg><figcaption><span>Clean correct-answer continuation BPB · lower is better.</span><span>Absolute answer likelihood can diverge from choice discrimination.</span></figcaption></figure>
    <div class="metric-strip" id="metric-strip"></div>
    <h3>0.5B schedule-only zoom</h3><p class="intro">The cross-scale separation compresses the actual data-order differences. These full-width zooms retain all 83 matched checkpoints per 0.5B arm.</p>
    <figure class="figure hero-chart"><svg id="mini-accuracy" role="img"></svg><figcaption><span>0.5B accuracy zoom.</span><span>Visual separation is not statistical significance.</span></figcaption></figure>
    <figure class="figure hero-chart"><svg id="mini-nll" role="img"></svg><figcaption><span>0.5B choice-NLL zoom.</span><span>D0 is the best endpoint NLL, not the best NLL seen at any earlier checkpoint.</span></figcaption></figure>
    <div class="table-wrap"><table><thead><tr><th>Trajectory</th><th>Token point</th><th>Accuracy</th><th>Choice NLL</th><th>Answer BPB</th><th>Best NLL</th><th>Late NLL rebound</th></tr></thead><tbody id="gm-table"></tbody></table></div>
  </section>

  <section id="learning" style="--accent:var(--teal)"><div class="head"><span class="bar"></span><div><div class="kicker">03 · Source-conditioned learning</div><h2>Greek loss falls at both scales</h2><p class="intro">Each panel plots BPB change from that run’s first available source-loss measurement. This normalization compares trajectory shape while avoiding a false claim that the two packed heldout geometries are identical. Negative values indicate learning.</p></div></div><div class="facets" id="learning-facets"></div>
    <p class="callout"><strong>What transfers:</strong> D0 produces broad reductions on HPLT, GlossAPI, specialist Greek, polytonic material, and neutral external Greek at both scales. The 8B curves generally descend faster in active-token space, consistent with greater capacity—not proof that the sanitized corpus is intrinsically better.</p>
  </section>

  <section id="forgetting" style="--accent:var(--gold)"><div class="head"><span class="bar"></span><div><div class="kicker">04 · Retention and forgetting</div><h2>Replay loss rises while Old Greek continues learning</h2><p class="intro">Forgetting is plotted as current BPB minus the best BPB observed earlier in the same trajectory. Zero means the panel is at its best-so-far value. The complete trajectories are shown; no tail crop or smoothing is used.</p></div></div><div class="facets" id="retention-facets"></div>
    <p class="callout"><strong>Shared pattern:</strong> English, code, mathematics, German, Russian, and Chinese all finish above their best-so-far BPB under D0 at both scales, while Old Greek finishes at its best value because it receives the fixed 1% replay stream. At the completed 8B endpoint, Chinese has the largest rise-from-best (+0.0647 BPB), followed by German (+0.0418) and English (+0.0291).</p>
  </section>

  <section id="interpretation" style="--accent:var(--purple)"><div class="head"><span class="bar"></span><div><div class="kicker">05 · Interpretation</div><h2>The 0.5B screen supports D0; it does not guarantee monotonic 8B benchmarks</h2></div></div>
    <div class="two-col"><article class="audit" style="--audit:var(--green)"><div class="kicker">Convergent evidence</div><h3>What replicated conceptually</h3><ul><li>D0 remains the least assumption-heavy schedule and was the 0.5B observed all-round leader after retention screening.</li><li>Both scales learn every Greek source family under stationary mixing.</li><li>Both scales show non-monotonic GreekMMLU NLL: all five 0.5B endpoints and the completed 8B endpoint are worse than their own best-NLL checkpoints.</li><li>At matched late checkpoints near 76B tokens, 8B is +12.61 accuracy points and −0.1484 choice NLL relative to 0.5B D0.</li></ul></article>
      <article class="audit" style="--audit:var(--red)"><div class="kicker">Non-identification</div><h3>What cannot be attributed</h3><ul><li>8B versus 0.5B is not a controlled scale-only comparison: architecture, tied/untied TD, absolute LR, global batch, and sanitized corpus differ.</li><li>None of the 13 fast source panels has an exact target-count/byte geometry match across scales; absolute cross-scale BPB ranks are descriptive.</li><li>The 8B run uses 5.01% fewer active corpus tokens and a post-mask exact-dedup pass absent from the 0.5B factorial.</li><li>No claim of statistical superiority across model scales is needed; the observed margins are point estimates from distinct training contracts.</li></ul></article></div>
    <p class="callout"><strong>Best completed reading:</strong> 8B learns materially more Greek multiple-choice discrimination than any 0.5B arm, but most of its benchmark gain is already present by about 10B tokens and its best checkpoint remains around 40B. Both scales finish above their own best NLL, supporting a general late-trajectory plateau/rebound rather than an anomaly unique to 8B.</p>
  </section>

  <section id="evidence" style="--accent:var(--ink)"><div class="head"><span class="bar"></span><div><div class="kicker">06 · Evidence boundary</div><h2>Receipt-bound, complete, and reproducible</h2><p class="intro">The report embeds every plotted point and records the local evidence hashes. Both campaigns are complete. The 8B input is the final campaign artifact with all 19 GreekMMLU milestones and all 39 document-local validation receipts.</p></div></div><div class="two-col"><article class="audit" style="--audit:var(--blue)"><h3>Counts and checks</h3><p id="counts"></p><p class="small">All data are finite; series are numerically ordered; endpoint calculations are reproduced from raw receipts; no smoothing is applied.</p></article><article class="audit" style="--audit:var(--gold)"><h3>Bound inputs</h3><div id="bindings"></div></article></div>
    <p class="small" style="margin-top:38px"><span class="status">Both campaigns complete</span> Cross-scale comparisons remain descriptive because the model and corpus contracts differ.</p>
  </section>
  <footer style="margin-top:100px;padding-top:24px;border-top:1px solid var(--rule);color:var(--muted);font-size:12px">Apertus Greek CPT · completed 0.5B five-arm factorial versus completed 8B D0 · generated from frozen local evidence and exact CSCS receipts.</footer>
</main>
<script id="report-data" type="application/json">__REPORT_DATA__</script>
<script>
const D=JSON.parse(document.getElementById('report-data').textContent),NS='http://www.w3.org/2000/svg';
const COLORS={full8:'#c64b45',D0_mixed:'#315f86',D1_hard_h_to_g:'#277e74',D2_hard_g_to_h:'#bd8730',D3_gradual_h_to_g:'#775ba6',D4_gradual_g_to_h:'#3f7f57',ink:'#102131',muted:'#65717b',rule:'#d6cfc2',paper:'#fffdf8'};
const LEARNING=['hplt','non_hplt','openarchives','greek_phd','historical_polytonic','neutral_external_modern_greek'];
const RETENTION=['english','code','math','de','ru','zh','old_greek'];
const ARM_LABEL={D0_mixed:'0.5B D0',D1_hard_h_to_g:'0.5B D1',D2_hard_g_to_h:'0.5B D2',D3_gradual_h_to_g:'0.5B D3',D4_gradual_g_to_h:'0.5B D4'};
const DASH={D0_mixed:'',D1_hard_h_to_g:'8 5',D2_hard_g_to_h:'4 4',D3_gradual_h_to_g:'11 4 2 4',D4_gradual_g_to_h:'2 4'};
function el(name,attrs={},text=''){const n=document.createElementNS(NS,name);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text)n.textContent=text;return n}
function path(points,x,y){return points.map((p,i)=>(i?'L':'M')+x(p[0]).toFixed(2)+','+y(p[1]).toFixed(2)).join(' ')}
function domain(values,pad=.06){let lo=Math.min(...values),hi=Math.max(...values);let d=Math.max(hi-lo,Math.abs(lo)*.02,1e-6);return[lo-d*pad,hi+d*pad]}
function lineChart(id,metric,title,desc,{miniOnly=false,height=710}={}){
 const svg=document.getElementById(id),W=1540,H=height,m={l:105,r:185,t:68,b:82},series=[];
 Object.keys(D.mini_greekmmlu).forEach(a=>series.push({id:a,label:ARM_LABEL[a],color:COLORS[a],dash:DASH[a],width:a==='D0_mixed'?3.2:2.25,rows:D.mini_greekmmlu[a]}));
 if(!miniOnly)series.push({id:'full8',label:'8B D0',color:COLORS.full8,dash:'',width:5.2,rows:D.full_greekmmlu});
 const all=series.flatMap(s=>s.rows.map(r=>r[metric])),[y0,y1]=domain(all,miniOnly?.12:.06),xmax=D.meta.mini_final_tokens_b;
 const x=v=>m.l+v/xmax*(W-m.l-m.r),y=v=>H-m.b-(v-y0)/(y1-y0)*(H-m.t-m.b);svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.append(el('title',{},title));svg.append(el('desc',{},desc));
 for(let i=0;i<6;i++){const v=y0+(y1-y0)*i/5;svg.append(el('line',{x1:m.l,x2:W-m.r,y1:y(v),y2:y(v),stroke:COLORS.rule}));const txt=metric==='accuracy'?(v*100).toFixed(1)+'%':v.toFixed(3);svg.append(el('text',{x:m.l-13,y:y(v)+4,fill:COLORS.muted,'font-size':12,'text-anchor':'end'},txt))}
 [0,10,20,30,40,50,60,70,80].forEach(v=>{svg.append(el('line',{x1:x(v),x2:x(v),y1:m.t,y2:H-m.b,stroke:COLORS.rule,opacity:.45}));svg.append(el('text',{x:x(v),y:H-m.b+30,fill:COLORS.muted,'font-size':12,'text-anchor':'middle'},v))});
 [[1.678,'shared warmup end',COLORS.gold],[61.35,'8B cooldown',COLORS.full8],[64.58,'0.5B cooldown',COLORS.blue]].forEach(([v,l,c],i)=>{svg.append(el('line',{x1:x(v),x2:x(v),y1:m.t,y2:H-m.b,stroke:c,'stroke-width':1.5,'stroke-dasharray':'5 5',opacity:.8}));if(!miniOnly||i!==1)svg.append(el('text',{x:x(v)+(i===1?-6:6),y:m.t-17+i*15,fill:c,'font-size':10,'font-weight':850,'text-anchor':i===1?'end':'start'},l))});
 const ends=[];series.forEach(s=>{const pts=s.rows.map(r=>[r.tokens_b,r[metric]]);svg.append(el('path',{d:path(pts,x,y),fill:'none',stroke:s.color,'stroke-width':s.width,'stroke-dasharray':s.dash,'stroke-linecap':'round','stroke-linejoin':'round'}));pts.forEach(p=>svg.append(el('circle',{cx:x(p[0]),cy:y(p[1]),r:s.id==='full8'?3.8:1.35,fill:s.color,opacity:s.id==='full8'?1:.62})));const last=pts.at(-1);ends.push({s,v:last[1],py:y(last[1]),px:x(last[0])})});
 const miniEnds=ends.filter(e=>e.s.id!=='full8').sort((a,b)=>a.py-b.py),minGap=16;miniEnds.forEach((e,i)=>e.ly=Math.max(m.t+4,e.py));for(let i=1;i<miniEnds.length;i++)miniEnds[i].ly=Math.max(miniEnds[i].ly,miniEnds[i-1].ly+minGap);let overflow=miniEnds.at(-1).ly-(H-m.b-3);if(overflow>0)miniEnds.forEach(e=>e.ly-=overflow);miniEnds.forEach(e=>{const lx=W-m.r+12;svg.append(el('line',{x1:e.px+5,x2:lx-4,y1:e.py,y2:e.ly,stroke:e.s.color}));const f=metric==='accuracy'?(e.v*100).toFixed(2)+'%':e.v.toFixed(4);svg.append(el('text',{x:lx,y:e.ly+4,fill:e.s.color,'font-size':11,'font-weight':850},`${e.s.label} ${f}`))});
 const f8=ends.find(e=>e.s.id==='full8');if(f8){const f=metric==='accuracy'?(f8.v*100).toFixed(2)+'%':f8.v.toFixed(4);svg.append(el('line',{x1:f8.px+5,x2:f8.px+18,y1:f8.py,y2:f8.py,stroke:f8.s.color,'stroke-width':2}));svg.append(el('text',{x:f8.px+23,y:f8.py+4,fill:f8.s.color,'font-size':13,'font-weight':900},`8B D0 ${f}`))}
 svg.append(el('line',{x1:m.l,x2:W-m.r,y1:H-m.b,y2:H-m.b,stroke:COLORS.ink}));svg.append(el('line',{x1:m.l,x2:m.l,y1:m.t,y2:H-m.b,stroke:COLORS.ink}));svg.append(el('text',{x:(m.l+W-m.r)/2,y:H-20,fill:COLORS.muted,'font-size':14,'font-weight':750,'text-anchor':'middle'},'Active training tokens (billions)'));svg.append(el('text',{x:m.l,y:28,fill:COLORS.ink,'font-size':21,'font-family':'Georgia','font-weight':700},title));
}
function drawMix(){const svg=document.getElementById('mix'),W=760,H=420,m={l:65,r:25,t:45,b:58},x=u=>m.l+u*(W-m.l-m.r),y=v=>H-m.b-v/.79*(H-m.t-m.b),q=.3094306438,a=2.231742;svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.append(el('title',{},'GlossAPI share across the five 0.5B data-order arms'));svg.append(el('desc',{},'Only temporal order changes. The 8B run uses stationary D0 mixing.'));[0,.2,.4,.6,.79].forEach(v=>{svg.append(el('line',{x1:m.l,x2:W-m.r,y1:y(v),y2:y(v),stroke:COLORS.rule}));svg.append(el('text',{x:m.l-9,y:y(v)+4,fill:COLORS.muted,'font-size':11,'text-anchor':'end'},(100*v).toFixed(0)+'%'))});const specs={D0_mixed:u=>.79*q,D1_hard_h_to_g:u=>u<1-q?0:.79,D2_hard_g_to_h:u=>u<q?.79:0,D3_gradual_h_to_g:u=>.79*Math.pow(u,a),D4_gradual_g_to_h:u=>.79*Math.pow(1-u,a)};Object.entries(specs).forEach(([id,f])=>{const pts=Array.from({length:260},(_,i)=>[i/259,f(i/259)]);svg.append(el('path',{d:path(pts,x,y),fill:'none',stroke:COLORS[id],'stroke-width':id==='D0_mixed'?4:2.6,'stroke-dasharray':DASH[id]}))});svg.append(el('line',{x1:m.l,x2:W-m.r,y1:H-m.b,y2:H-m.b,stroke:COLORS.ink}));svg.append(el('text',{x:W/2,y:H-15,fill:COLORS.muted,'font-size':13,'text-anchor':'middle'},'Normalized training progress'));svg.append(el('text',{x:m.l,y:24,fill:COLORS.ink,'font-size':18,'font-family':'Georgia','font-weight':700},'GlossAPI share of all active tokens'))}
function drawLR(){const svg=document.getElementById('lr'),W=760,H=420,m={l:70,r:30,t:45,b:58},x=u=>m.l+u*(W-m.l-m.r),y=v=>H-m.b-v*(H-m.t-m.b),lr=u=>u<.02078?.1+.9*u/.02078:u<.8?1:.1+.9*Math.sqrt((1-u)/.2);svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.append(el('title',{},'Relative learning-rate geometry at both model scales'));svg.append(el('desc',{},'Both schedules warm from one tenth to peak over 1.678 billion tokens, remain stable to 80 percent, then use one-minus-square-root decay to one tenth.'));[.1,.4,.7,1].forEach(v=>{svg.append(el('line',{x1:m.l,x2:W-m.r,y1:y(v),y2:y(v),stroke:COLORS.rule}));svg.append(el('text',{x:m.l-10,y:y(v)+4,fill:COLORS.muted,'font-size':11,'text-anchor':'end'},v.toFixed(1)+'×'))});const pts=Array.from({length:320},(_,i)=>[i/319,lr(i/319)]);svg.append(el('path',{d:path(pts,x,y),fill:'none',stroke:COLORS.ink,'stroke-width':5}));svg.append(el('text',{x:x(.4),y:y(1)-15,fill:COLORS.D0_mixed,'font-size':12,'font-weight':850,'text-anchor':'middle'},'0.5B peak 1.5e−4'));svg.append(el('text',{x:x(.4),y:y(1)+18,fill:COLORS.full8,'font-size':12,'font-weight':850,'text-anchor':'middle'},'8B peak 5.5e−5'));svg.append(el('line',{x1:m.l,x2:W-m.r,y1:H-m.b,y2:H-m.b,stroke:COLORS.ink}));svg.append(el('text',{x:W/2,y:H-15,fill:COLORS.muted,'font-size':13,'text-anchor':'middle'},'Normalized training progress'));svg.append(el('text',{x:m.l,y:24,fill:COLORS.ink,'font-size':18,'font-family':'Georgia','font-weight':700},'Learning rate / peak'))}
function changeFacet(panel,mode){const article=document.createElement('article');article.className='facet';article.style.setProperty('--facet',mode==='learning'?COLORS.teal:COLORS.gold);article.innerHTML=`<h3>${D.labels[panel]}</h3><div class="sub">${mode==='learning'?'Δ BPB from first measurement · lower is better':'BPB rise from best-so-far · lower is better'}</div>`;const svg=el('svg'),W=720,H=330,m={l:70,r:88,t:24,b:52},series=[];Object.keys(D.mini_validation).forEach(a=>{const rows=D.mini_validation[a][panel],base=rows[0].bpb,best=[];let b=Infinity;const pts=rows.map(r=>{b=Math.min(b,r.bpb);return[r.tokens_b,mode==='learning'?r.bpb-base:r.bpb-b]});series.push({id:a,color:COLORS[a],dash:DASH[a],width:a==='D0_mixed'?2.7:1.55,pts})});{const rows=D.full_validation[panel],base=rows[0].bpb;let b=Infinity;series.push({id:'full8',color:COLORS.full8,dash:'',width:4,pts:rows.map(r=>{b=Math.min(b,r.bpb);return[r.tokens_b,mode==='learning'?r.bpb-base:r.bpb-b]})})}const vals=series.flatMap(s=>s.pts.map(p=>p[1])),[y0,y1]=domain(vals,.08),x=v=>m.l+v/D.meta.mini_final_tokens_b*(W-m.l-m.r),y=v=>H-m.b-(v-y0)/(y1-y0)*(H-m.t-m.b);svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.append(el('title',{},`${D.labels[panel]} ${mode} across 0.5B schedules and 8B D0`));svg.append(el('desc',{},'Raw full-horizon source validation trajectories by active training tokens.'));for(let i=0;i<4;i++){const v=y0+(y1-y0)*i/3;svg.append(el('line',{x1:m.l,x2:W-m.r,y1:y(v),y2:y(v),stroke:COLORS.rule}));svg.append(el('text',{x:m.l-8,y:y(v)+4,fill:COLORS.muted,'font-size':10,'text-anchor':'end'},(v>=0?'+':'')+v.toFixed(3)))}[0,20,40,60,80].forEach(v=>svg.append(el('text',{x:x(v),y:H-m.b+24,fill:COLORS.muted,'font-size':10,'text-anchor':'middle'},v)));series.forEach(s=>{svg.append(el('path',{d:path(s.pts,x,y),fill:'none',stroke:s.color,'stroke-width':s.width,'stroke-dasharray':s.dash,'stroke-linejoin':'round'}));const last=s.pts.at(-1);if(s.id==='full8'||s.id==='D0_mixed'){svg.append(el('circle',{cx:x(last[0]),cy:y(last[1]),r:s.id==='full8'?4:3,fill:s.color}));svg.append(el('text',{x:x(last[0])+(s.id==='full8'?7:7),y:y(last[1])+(s.id==='full8'?-7:12),fill:s.color,'font-size':9.5,'font-weight':850},`${s.id==='full8'?'8B':'0.5 D0'} ${(last[1]>=0?'+':'')+last[1].toFixed(3)}`))}});svg.append(el('line',{x1:m.l,x2:W-m.r,y1:y(0),y2:y(0),stroke:COLORS.ink,'stroke-width':1.2,opacity:.7}));svg.append(el('text',{x:(m.l+W-m.r)/2,y:H-14,fill:COLORS.muted,'font-size':11,'text-anchor':'middle'},'Active tokens (B)'));article.append(svg);return article}
function fillContract(){const a=D.contract.mini,b=D.contract.full8,rows=[['Base model / geometry',`${a.model}; 20L, h=1,024, 16 heads`,`${b.model}; 32L, h=4,096, 32 heads`,'Scale and architecture confound'],['Embedding adaptation',`${a.initialization}; tied`,`${b.initialization}; untied`,'Different TD geometry'],['Active data',`${(a.tokens/1e9).toFixed(3)}B`,`${(b.tokens/1e9).toFixed(3)}B`,'8B corpus is 5.01% smaller'],['Batch / update',`${(a.tokens_per_update/1e6).toFixed(3)}M tokens; ${a.updates.toLocaleString()} updates`,`${(b.tokens_per_update/1e6).toFixed(3)}M tokens; ${b.updates.toLocaleString()} updates`,'Same token-scale schedule geometry'],['Learning rate',`${a.peak_lr.toExponential(1)} → ${a.floor_lr.toExponential(1)}`,`${b.peak_lr.toExponential(1)} → ${b.floor_lr.toExponential(1)}`,'Both WSD-10; absolute LR differs'],['Data treatment',a.data,b.data,'Not a data-identical comparison'],['Tokenizer / objective',`148,992; 4,096; Goldfish k50/h50`,`148,992; 4,096; Goldfish k50/h50`,'Core text objective aligned'],['GreekMMLU',D.contract.shared.greekmmlu,D.contract.shared.greekmmlu,'Direct benchmark contract aligned']];const body=document.getElementById('contract-body');rows.forEach(r=>{const tr=document.createElement('tr');r.forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.append(td)});body.append(tr)})}
function fillMetrics(){const f=D.full_endpoint,c=D.comparison.full8_vs_mini_d0,m=D.comparison.mini.D0_mixed,items=[[`${(f.accuracy*100).toFixed(2)}%`,`8B clean accuracy at ${f.tokens_b.toFixed(1)}B`],[`${c.accuracy_pp.toFixed(2)} pp`,`8B − 0.5B D0 at ${m.tokens_b.toFixed(1)}B / ${f.tokens_b.toFixed(1)}B`],[f.best_nll.toFixed(4),`8B best NLL at ${f.best_nll_tokens_b.toFixed(1)}B`],[`+${f.nll_rebound.toFixed(4)}`,'8B NLL rebound from best to endpoint']];document.getElementById('metric-strip').innerHTML=items.map(([n,l])=>`<article><div class="n">${n}</div><div class="l">${l}</div></article>`).join('')}
function fillTable(){const body=document.getElementById('gm-table'),rows=[['8B · D0 endpoint',D.full_endpoint],...Object.keys(D.mini_endpoints).map(a=>[ARM_LABEL[a]+' endpoint',D.mini_endpoints[a]])];rows.forEach(([name,r],i)=>{const tr=document.createElement('tr'),vals=[name,`${r.tokens_b.toFixed(2)}B`,`${(r.accuracy*100).toFixed(2)}%`,r.nll.toFixed(4),r.bpb.toFixed(4),`${r.best_nll.toFixed(4)} @ ${r.best_nll_tokens_b.toFixed(1)}B`,`+${r.nll_rebound.toFixed(4)}`];vals.forEach((v,j)=>{const td=document.createElement('td');td.textContent=v;if(i===0&&j===0)td.className='warn';tr.append(td)});body.append(tr)})}
function fillEvidence(){document.getElementById('counts').innerHTML=`0.5B: <strong>${D.meta.mini_greekmmlu_points}</strong> GreekMMLU receipts and 5,395 source-loss rows. 8B: <strong>${D.meta.full_greekmmlu_points}</strong> clean GreekMMLU points through update ${D.meta.full_latest_greekmmlu_iteration.toLocaleString()} and source loss through update ${D.meta.full_snapshot_iteration.toLocaleString()}. Exact cross-scale source-panel geometry matches: <strong>${D.meta.panel_geometry_exact_matches}/${D.meta.panel_geometry_count}</strong>.`;document.getElementById('bindings').innerHTML=D.bindings.map(b=>`<p><strong>${b.role}</strong><br><code>${b.path}</code><br><code>sha256 ${b.sha256}</code></p>`).join('')}
drawMix();drawLR();lineChart('gm-accuracy','accuracy','Clean GreekMMLU accuracy','Five complete 0.5B data-order trajectories and the completed 8B stationary-mix trajectory.',{height:730});lineChart('gm-nll','nll','Clean GreekMMLU choice NLL','Continuous multiple-choice loss across model scale and data order.',{height:730});lineChart('gm-bpb','bpb','Clean GreekMMLU correct-answer BPB','Byte-normalized absolute correct-answer continuation likelihood.',{height:730});lineChart('mini-accuracy','accuracy','0.5B clean accuracy — schedule zoom','All five 0.5B schedules over the complete 80.73B-token horizon.',{miniOnly:true,height:650});lineChart('mini-nll','nll','0.5B clean choice NLL — schedule zoom','All five 0.5B schedules over the complete 80.73B-token horizon.',{miniOnly:true,height:650});LEARNING.forEach(p=>document.getElementById('learning-facets').append(changeFacet(p,'learning')));RETENTION.forEach(p=>document.getElementById('retention-facets').append(changeFacet(p,'retention')));fillContract();fillMetrics();fillTable();fillEvidence();if(location.hash)setTimeout(()=>document.querySelector(location.hash)?.scrollIntoView(),200);
</script></body></html>'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("FULL8_VS_0P5B_FIVE_ARM_COMPARISON_20260811.html"))
    parser.add_argument("--data-output", type=Path, default=Path(__file__).with_name("FULL8_VS_0P5B_FIVE_ARM_COMPARISON_20260811.data.json"))
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    payload = build_payload(repo)
    args.output = args.output.resolve()
    args.data_output = args.data_output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.replace("__REPORT_DATA__", encoded)
    if "__REPORT_DATA__" in html:
        raise RuntimeError("report data placeholder was not replaced")
    args.output.write_text(html, encoding="utf-8")
    args.data_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": True, "html": str(args.output), "data": str(args.data_output),
        "html_bytes": args.output.stat().st_size,
        "full8_latest_greekmmlu": payload["meta"]["full_latest_greekmmlu_iteration"],
        "mini_greekmmlu_points": payload["meta"]["mini_greekmmlu_points"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
