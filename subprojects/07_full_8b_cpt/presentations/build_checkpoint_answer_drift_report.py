#!/usr/bin/env python3
"""Build the single-page GreekMMLU answer-drift and source-exposure report."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def receipt(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def logged_argument(text: str, name: str) -> float:
    match = re.search(
        rf"^\s*0:\s+{re.escape(name)}\s+\.+\s+(\S+)\s*$",
        text,
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"executed optimizer log does not contain {name}")
    return float(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drift",
        type=Path,
        default=ROOT / "data/full8_checkpoint_drift_20260811/greekmmlu_answer_drift.json",
    )
    parser.add_argument(
        "--exposure",
        type=Path,
        default=ROOT / "data/full8_checkpoint_drift_20260811/checkpoint_source_exposure.json",
    )
    parser.add_argument(
        "--peer",
        type=Path,
        default=ROOT / "data/greekmmlu_peer_models_recovered_20260619.json",
    )
    parser.add_argument(
        "--full-results",
        type=Path,
        default=ROOT / "FULL8_SANITIZED_CPT_FINAL_RESULTS_20260811.data.json",
    )
    parser.add_argument(
        "--optimizer-log",
        type=Path,
        default=(
            ROOT.parents[2]
            / ".codex_tmp/full8_current_comparison/new/logs/segment_4_training.log.pre_final_snapshot"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "FULL8_GREEKMMLU_DRIFT_AND_DATA_EXPOSURE_20260811.html",
    )
    args = parser.parse_args()

    drift = read_json(args.drift)
    exposure = read_json(args.exposure)
    peer = read_json(args.peer)
    full = read_json(args.full_results)
    optimizer_log_text = args.optimizer_log.read_text(encoding="utf-8")
    optimizer = {
        name: logged_argument(optimizer_log_text, name)
        for name in (
            "adam_beta1",
            "adam_beta2",
            "ademamix_alpha",
            "ademamix_alpha_warmup",
            "ademamix_beta3",
            "ademamix_beta3_warmup",
            "clip_grad",
            "lr",
            "min_lr",
            "weight_decay",
        )
    }
    constants = full["constants"]
    schedule_checks = {
        "lr": float(constants["peak_lr"]),
        "min_lr": float(constants["minimum_lr"]),
        "ademamix_alpha_warmup": float(constants["new_updates"]),
        "ademamix_beta3_warmup": float(constants["new_updates"]),
    }
    for name, expected in schedule_checks.items():
        if optimizer[name] != expected:
            raise ValueError(
                f"executed optimizer value {name}={optimizer[name]} differs from report contract {expected}"
            )
    metrics = full["new_greekmmlu"]
    if [int(row["iteration"]) for row in metrics] != [int(row["iteration"]) for row in drift["checkpoint_table"]]:
        raise ValueError("GreekMMLU checkpoint coverage differs between answer and metric evidence")
    if [int(row["iteration"]) for row in metrics] != [int(row["iteration"]) for row in exposure["checkpoint_summary"]]:
        raise ValueError("GreekMMLU checkpoint coverage differs from source exposure")

    best = max(metrics, key=lambda row: float(row["clean_accuracy"]))
    final = metrics[-1]
    peak_to_final = drift["best_to_final"]
    persistence = drift["correctness_persistence"]
    subject_rows = drift["subject_final_vs_best"]
    source_rows = exposure["sources"]
    register_rows = exposure["hplt_register_level_1"]
    checkpoint_progress = exposure["checkpoint_summary"]
    glossapi_rows = [
        row
        for row in source_rows
        if row["label"] != "HPLT/ell_Grek_ge8_no_mt_clean60"
        and row["label"] != "greek_replay_apertus_original"
        and row["label"] != "code_starcoderdata_subset"
        and row["label"] != "math_finemath"
        and not row["label"].startswith("replay_")
    ]
    glossapi_tokens = sum(int(row["total_tokens"]) for row in glossapi_rows)
    expected_glossapi_tokens = int(full["data"]["new_modern"]["glossapi_non_hplt_tokens"])
    if glossapi_tokens != expected_glossapi_tokens:
        raise ValueError(
            f"classified GlossAPI/non-HPLT tokens {glossapi_tokens} != contract {expected_glossapi_tokens}"
        )
    glossapi_trajectory = [
        {
            "iteration": int(checkpoint["iteration"]),
            "seen_tokens": sum(int(row["trajectory"][index]["seen_tokens"]) for row in glossapi_rows),
        }
        for index, checkpoint in enumerate(checkpoint_progress)
    ]
    for point in glossapi_trajectory:
        point["seen_token_fraction"] = point["seen_tokens"] / glossapi_tokens
    large_source_gaps = [
        abs(float(point["seen_token_fraction"]) - float(checkpoint["corpus_token_fraction"]))
        for row in source_rows
        if int(row["total_tokens"]) >= 1_000_000_000
        for point, checkpoint in zip(row["trajectory"][1:-1], checkpoint_progress[1:-1], strict=True)
    ]
    register_gaps = [
        abs(float(point["seen_token_fraction"]) - float(checkpoint["corpus_token_fraction"]))
        for row in register_rows
        for point, checkpoint in zip(row["trajectory"][1:-1], checkpoint_progress[1:-1], strict=True)
    ]
    payload = {
        "drift": drift,
        "exposure": exposure,
        "peer": peer,
        "metrics": metrics,
        "hyperparameters": {
            "optimizer": "AdEMAMix",
            "total_updates": int(constants["new_updates"]),
            "tokens_per_update": int(constants["tokens_per_update"]),
            "warmup_end": int(constants["warmup_end"]),
            "cooldown_start": int(constants["new_cooldown_start"]),
            "cooldown_shape": "1-sqrt",
            "peak_lr": optimizer["lr"],
            "minimum_lr": optimizer["min_lr"],
            "beta1": optimizer["adam_beta1"],
            "beta2": optimizer["adam_beta2"],
            "beta3_final": optimizer["ademamix_beta3"],
            "beta3_warmup": int(optimizer["ademamix_beta3_warmup"]),
            "alpha_final": optimizer["ademamix_alpha"],
            "alpha_warmup": int(optimizer["ademamix_alpha_warmup"]),
            "weight_decay": optimizer["weight_decay"],
            "clip_grad": optimizer["clip_grad"],
            "beta3_schedule": "linear in EMA half-life from beta1 to beta3_final",
            "alpha_schedule": "linear from zero to alpha_final",
            "schedule_reference_url": "https://github.com/apple/ml-ademamix/blob/main/pytorch/ademamix.py",
            "hplt_trajectory": source_rows[0]["trajectory"],
            "glossapi_total_tokens": glossapi_tokens,
            "glossapi_trajectory": glossapi_trajectory,
        },
        "derived": {
            "best_iteration": int(best["iteration"]),
            "best_clean_accuracy": float(best["clean_accuracy"]),
            "best_full_accuracy": float(best["accuracy"]),
            "final_clean_accuracy": float(final["clean_accuracy"]),
            "final_full_accuracy": float(final["accuracy"]),
            "peak_to_final_newly_correct": int(peak_to_final["newly_correct"]),
            "peak_to_final_newly_wrong": int(peak_to_final["newly_wrong"]),
            "peak_to_final_flip_rate": float(peak_to_final["answer_choice_flip_rate"]),
            "peak_to_final_correct_churn": float(peak_to_final["correct_set_churn_rate"]),
            "transient_correct": int(persistence["transiently_correct"]),
            "transient_correct_fraction": int(persistence["transiently_correct"]) / int(drift["clean_subset"]["n"]),
            "subjects_down_peak_to_final": sum(row["final_minus_best_checkpoint"] < 0 for row in subject_rows),
            "subjects_up_peak_to_final": sum(row["final_minus_best_checkpoint"] > 0 for row in subject_rows),
            "source_count": len(source_rows),
            "register_count": len(register_rows),
            "large_source_max_abs_progress_gap": max(large_source_gaps),
            "hplt_register_max_abs_progress_gap": max(register_gaps),
        },
        "bindings": {
            "drift": receipt(args.drift),
            "exposure": receipt(args.exposure),
            "peer": receipt(args.peer),
            "full_results": receipt(args.full_results),
            "executed_optimizer_log": receipt(args.optimizer_log),
            "answer_drift_analyzer": receipt(
                ROOT.parent / "analysis/analyze_greekmmlu_answer_drift.py"
            ),
            "source_exposure_analyzer": receipt(
                ROOT.parent / "analysis/analyze_checkpoint_source_exposure.py"
            ),
            "ademamix_schedule_audit": receipt(
                ROOT.parents[1]
                / "03_apertus_extension_and_embedding_adaptation/_archive/2026-05-24_2B_bakeoff_review/AUDIT_FINDINGS.md"
            ),
        },
    }
    data_path = args.output.with_suffix(".data.json")
    data_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")

    html = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GreekMMLU answer drift and exact corpus exposure · Apertus 8B</title>
<style>
:root{--paper:#f5f0e7;--panel:#fffdf8;--ink:#17242d;--muted:#66747d;--rule:#d7cec0;--red:#9b3438;--blue:#315f78;--teal:#39766e;--gold:#b38730;--green:#47734e;--orange:#b65e33;--shadow:0 16px 42px rgba(42,38,31,.08)}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;color:var(--ink);background:radial-gradient(circle at 93% 1%,rgba(179,135,48,.13),transparent 27rem),linear-gradient(135deg,var(--paper),#fbf8f1 74%,#efe7da);font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.page{width:min(1740px,100%);margin:auto;padding:0 clamp(24px,5vw,86px) 96px}h1,h2,h3,p{margin-top:0}h1,h2,h3{font-family:Georgia,"Times New Roman",serif;letter-spacing:-.032em}h1{max-width:1450px;font-size:clamp(52px,7vw,108px);line-height:.94;margin-bottom:25px}h2{max-width:1280px;font-size:clamp(35px,4.5vw,70px);line-height:1.01;margin-bottom:16px}h3{font-size:clamp(21px,2vw,32px)}p,li,td,th{font-size:clamp(14px,1vw,18px);line-height:1.5}.hero{min-height:min(91vh,1120px);display:grid;align-content:center;position:relative;padding:70px 0}.hero:before{content:"";position:absolute;left:calc(-1 * clamp(24px,5vw,86px));top:0;bottom:0;width:10px;background:var(--red)}.eyebrow{display:flex;justify-content:space-between;gap:22px;color:var(--muted);font-size:12px;font-weight:850;letter-spacing:.14em;text-transform:uppercase;margin-bottom:30px}.title-rule{width:112px;height:8px;background:var(--red);margin-bottom:27px}.lede{max-width:1190px;color:#364650;font-size:clamp(20px,2vw,31px);line-height:1.42}.finding{max-width:1450px;margin:38px 0 0;padding:22px 0 0 28px;border-left:7px solid var(--gold);font:700 clamp(24px,2.6vw,42px)/1.2 Georgia,serif}.nav{display:flex;flex-wrap:wrap;gap:10px 24px;margin-top:42px;padding-top:18px;border-top:1px solid var(--rule)}.nav a{color:var(--muted);font-size:13px;font-weight:780;text-decoration:none}.section{padding:clamp(75px,9vw,138px) 0 0}.head{display:grid;grid-template-columns:8px minmax(0,1fr);gap:24px;margin-bottom:32px}.bar{background:var(--accent,var(--red));min-height:84px}.kicker{color:var(--accent,var(--red));font-size:12px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;margin-bottom:9px}.intro{max-width:1190px;color:var(--muted)}.metric-strip{display:grid;grid-template-columns:repeat(4,1fr);margin:34px 0;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule)}.metric-strip article{padding:24px;border-left:1px solid var(--rule)}.metric-strip article:first-child{border-left:0;padding-left:0}.metric-strip .n{font:700 clamp(31px,3vw,52px)/1 Georgia,serif}.metric-strip .l{color:var(--muted);font-size:12px;margin-top:8px}.figure{margin:0 0 30px;padding:clamp(16px,2vw,28px);border:1px solid var(--rule);background:rgba(255,253,248,.86);box-shadow:var(--shadow)}.figure svg{display:block;width:100%;height:auto}.hero-chart svg{min-height:610px}.figure figcaption{display:flex;justify-content:space-between;gap:25px;color:var(--muted);font-size:12px;line-height:1.45;margin-top:12px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:28px}.notes{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}.note{padding:23px;border-top:7px solid var(--blue);background:rgba(255,253,248,.72)}.note.good{border-color:var(--green)}.note.caution{border-color:var(--gold)}.note h3{font-size:23px;margin-bottom:8px}.table-wrap{overflow:auto;max-height:760px;border:1px solid var(--rule);background:rgba(255,255,255,.28)}table{border-collapse:separate;border-spacing:0;width:max-content;min-width:100%}th,td{padding:10px 12px;border-bottom:1px solid var(--rule);border-right:1px solid rgba(215,206,192,.48);vertical-align:top}th{position:sticky;top:0;z-index:3;background:#eee7dc;color:var(--muted);font-size:11px;letter-spacing:.055em;text-transform:uppercase;text-align:right}th:first-child,td:first-child{position:sticky;left:0;z-index:2;background:#f7f2e9;text-align:left;min-width:210px;font-weight:750}th:first-child{z-index:4;background:#e9e1d5}td:not(:first-child){text-align:right;font-variant-numeric:tabular-nums}.small{font-size:12px;color:var(--muted)}.status{display:inline-block;border-radius:999px;padding:5px 10px;background:#f0d8d5;color:#872f32;font-size:11px;font-weight:900;letter-spacing:.07em;text-transform:uppercase}.legend{display:flex;flex-wrap:wrap;gap:10px 24px;margin:0 0 18px;color:var(--muted);font-size:13px}.legend span:before{content:"";display:inline-block;width:24px;height:5px;margin-right:8px;transform:translateY(-2px);background:var(--legend)}code{font-size:12px;overflow-wrap:anywhere}.evidence{display:grid;grid-template-columns:1fr 1fr;gap:24px}.evidence article{padding-top:15px;border-top:6px solid var(--teal)}
.schedule-chart svg{min-height:900px}
@media(max-width:1050px){.pair,.notes,.evidence{grid-template-columns:1fr}.metric-strip{grid-template-columns:1fr 1fr}.hero-chart svg,.schedule-chart svg{min-height:auto}.evidence article{min-width:0}.evidence code,.evidence .small{display:block;overflow-wrap:anywhere;word-break:break-word}}@media(max-width:650px){.hero{min-height:auto}.eyebrow,.figure figcaption{flex-direction:column}.metric-strip{grid-template-columns:1fr}.metric-strip article{border-left:0;border-top:1px solid var(--rule);padding:18px 0}.page{padding-bottom:56px}.section{padding-top:75px}}
@page{size:A3 landscape;margin:12mm}
@media print{body{background:white}.page{width:100%;max-width:none;padding:0}.hero{min-height:auto;padding:24px 0}.hero:before{left:-12px}.section{padding-top:38px;break-before:page}.figure{box-shadow:none;break-inside:avoid}.figure svg{min-height:0;max-height:160mm;width:100%}.figure figcaption{break-before:avoid}.head,.metric-strip,.note,.evidence article{break-inside:avoid}h2,h3,.intro{break-after:avoid}.notes{break-inside:avoid}.table-wrap{max-height:none;overflow:visible}.table-wrap table{width:100%;table-layout:auto}.table-wrap th,.table-wrap td{padding:6px 7px;font-size:9px}.table-wrap th:first-child,.table-wrap td:first-child{position:static;min-width:118px}.table-wrap th{position:static}thead{display:table-header-group}.nav{display:none}}
</style></head><body><main class="page">
<header class="hero"><div class="eyebrow"><span>Apertus 8B · D0 stationary CPT</span><span>Evidence frozen 11 August 2026</span></div><div class="title-rule"></div><h1>The score plateau hides moving answers.</h1><p class="lede">A question-level audit of all 19 clean GreekMMLU checkpoints, joined to exact source-token and document exposure from the immutable D0 schedule. The report also recovers the earlier Gemma, Krikri and Qwen comparison—with its historical-evidence limits made explicit.</p><div class="finding"><span id="hero-finding"></span></div><div class="nav"><a href="#optimizer">Training geometry</a><a href="#trajectory">Benchmark trajectory</a><a href="#answers">Answer drift</a><a href="#subjects">Subject areas</a><a href="#sources">Sources seen</a><a href="#hplt">HPLT site types</a><a href="#peers">Peer models</a><a href="#evidence">Evidence</a></div></header>

<section class="section" id="optimizer" style="--accent:#725a8f">
<div class="head"><div class="bar"></div><div><div class="kicker">01 · Training geometry at the 40B checkpoint</div><h2>Five clocks align near halfway—but none changes phase there.</h2><p class="intro">The benchmark peak is placed against the executed learning-rate schedule, AdEMAMix’s two full-run warmups, and exact HPLT/GlossAPI exposure. Curves are reconstructed from the executed optimizer configuration; GreekMMLU and exposure points are measured.</p></div></div>
<div class="metric-strip" id="optimizer-metrics"></div>
<figure class="figure schedule-chart"><svg id="optimizer-geometry" role="img"></svg><figcaption><span>Solid points: measured benchmark and exposure evidence. Schedule curves: reconstructed from executed flags and the AdEMAMix half-life scheduler.</span><span>Red line: update 9,536 / 39.996B · grey line: exact corpus halfway.</span></figcaption></figure>
<h3>How much data has been seen—and how far backward does each β look?</h3><p class="intro">The left side shows cumulative source tokens. The right side shows each exponential moving average’s half-life in training tokens. A half-life is not a hard cutoff: gradients older than the bar still contribute, but at progressively smaller weights. α changes the strength of the slow β3 branch; it does not change β3’s memory length.</p>
<figure class="figure schedule-chart"><svg id="beta-history-tokens" role="img"></svg><figcaption><span>Left: exact cumulative tokens consumed from HPLT, GlossAPI/non-HPLT, and replay. Right: β1, β3 and β2 memory half-lives.</span><span>Purple α badge: coefficient applied to the slow β3 directional average.</span></figcaption></figure>
<div class="notes"><article class="note caution"><h3>No 40B switch</h3><p>Learning rate is still at its peak, the cooldown has not started, and α and β3 are only partway through smooth full-run ramps. The coincidence is therefore an optimum along moving schedules—not a configured phase boundary.</p></article><article class="note"><h3>Slow history gains influence</h3><p>At the peak, α gives the slow first-moment branch a coefficient of about 2.09 while β3 gives it a roughly 364-update half-life. This is a mixing coefficient, not a claim that the slow branch contributes exactly twice the update magnitude.</p></article><article class="note good"><h3>All data clocks are halfway</h3><p>HPLT and GlossAPI exposure track total corpus progress almost exactly. The graph therefore cannot identify HPLT’s 50% point as the cause of the benchmark peak.</p></article><article class="note caution"><h3>Adaptation versus stability</h3><p>Peak LR keeps individual steps plastic while the increasingly weighted slow EMA makes them depend on a longer history. That combination can keep source loss improving while benchmark decision boundaries continue to move.</p></article></div>
</section>

<section class="section" id="trajectory" style="--accent:var(--red)"><div class="head"><div class="bar"></div><div><div class="kicker">02 · Benchmark trajectory</div><h2>Accuracy peaks early; continuous loss does not settle identically.</h2><p class="intro">Every point uses the same decontaminated 16,159-question subset. Accuracy is a discrete argmax; choice NLL is the continuous probability assigned to the correct option relative to the alternatives.</p></div></div><div class="metric-strip" id="headline-metrics"></div><figure class="figure hero-chart"><svg id="accuracy" role="img"></svg><figcaption><span>Clean zero-shot accuracy · higher is better.</span><span>Exact active-token x-axis.</span></figcaption></figure><figure class="figure hero-chart"><svg id="nll" role="img"></svg><figcaption><span>Clean multiple-choice NLL · lower is better.</span><span>Same 19 checkpoints.</span></figcaption></figure></section>

<section class="section" id="answers" style="--accent:var(--gold)"><div class="head"><div class="bar"></div><div><div class="kicker">03 · Question-level state changes</div><h2>A flat score is not a stable answer set.</h2><p class="intro">At each transition, “newly correct” means wrong at the preceding checkpoint and correct now; “newly wrong” is the reverse. Choice flips count any change in the selected option, even when both choices are wrong.</p></div></div><figure class="figure hero-chart"><svg id="answer-flow" role="img"></svg><figcaption><span>Green: newly correct; red: newly wrong.</span><span>Bars are percentages of the fixed 16,159 questions.</span></figcaption></figure><div class="table-wrap"><table id="answer-table"></table></div></section>

<section class="section" id="subjects" style="--accent:var(--blue)"><div class="head"><div class="bar"></div><div><div class="kicker">04 · Areas of knowledge</div><h2>Which subjects drift—and when?</h2><p class="intro">The chart fixes the reference at update 9,536—the global accuracy peak—and asks which subject groups gained or lost by the final checkpoint. The first table then exposes every subject at every checkpoint, so local reversals are not hidden by the endpoint summary. Small subjects are shown but should be read with their sample size.</p></div></div><figure class="figure"><svg id="subjects-chart" role="img"></svg><figcaption><span>Final minus update-9,536 accuracy.</span><span>Positive is improvement; negative is regression.</span></figcaption></figure><h3>Subject accuracy at every checkpoint</h3><div class="legend"><span style="--legend:rgba(155,52,56,.55)">below that row’s mean</span><span style="--legend:#f7f2e9">near that row’s mean</span><span style="--legend:rgba(71,115,78,.58)">above that row’s mean</span><span>Red border = row minimum · green border = row maximum</span></div><div class="table-wrap"><table id="subject-checkpoint-table"></table></div><p class="small">Each cell is accuracy on the fixed clean subset for that subject. Color intensity is normalized within its own row, so it reveals when that subject performed relatively well or poorly; it does not compare difficulty across subjects. Hover for the row mean and exact deltas.</p><h3 style="margin-top:34px">Subject endpoint summary</h3><div class="table-wrap"><table id="subjects-table"></table></div><h3 style="margin-top:34px">Educational-level accuracy at every checkpoint</h3><p class="small">The same row-relative coloring and best/worst borders are applied here.</p><div class="table-wrap"><table id="level-checkpoint-table"></table></div></section>

<section class="section" id="sources" style="--accent:var(--teal)"><div class="head"><div class="bar"></div><div><div class="kicker">05 · Exact corpus exposure</div><h2>What fraction of every source had actually been seen?</h2><p class="intro">These are reconstructed loss-active tokens from the frozen packed sequence IDs—not nominal 79/20/1 quotas. “Touched” documents have at least one intersecting sequence consumed; “fully seen” documents have all intersecting sequences consumed.</p></div></div><figure class="figure hero-chart"><svg id="source-lines" role="img"></svg><figcaption><span>Largest source families; diagonal means perfectly uniform exposure.</span><span>Full source × checkpoint table follows.</span></figcaption></figure><div class="table-wrap"><table id="source-table"></table></div><p class="small">Cell values are exact seen-token percentages. Hover a cell for touched- and fully-seen-document percentages.</p></section>

<section class="section" id="hplt" style="--accent:var(--orange)"><div class="head"><div class="bar"></div><div><div class="kicker">06 · HPLT metadata categories</div><h2>Which HPLT content types had been seen?</h2><p class="intro">The available “type” field is HPLT’s preserved <code>register_level_1</code> text-register classifier, such as Narrative or Informational description/explanation. It describes the content, not necessarily the whole website, and it was not inferred after training.</p></div></div><figure class="figure hero-chart"><svg id="register-lines" role="img"></svg><figcaption><span>Token exposure of HPLT register categories.</span><span>Exact packed-document join.</span></figcaption></figure><div class="table-wrap"><table id="register-table"></table></div></section>

<section class="section" id="peers" style="--accent:var(--red)"><div class="head"><div class="bar"></div><div><div class="kicker">07 · Historical model context</div><h2>Gemma, Krikri, Qwen—and our earlier pilot.</h2><p class="intro">The peer values were recovered from the preserved June 19 plot after the raw CSCS evaluation payloads had been pruned. They used the full 16,632-question split. The current run is receipt-backed and shown separately; the two evidence classes are context, not a controlled leaderboard.</p></div></div><p><span class="status">Recovered plot evidence</span></p><figure class="figure hero-chart"><svg id="peer-chart" role="img"></svg><figcaption><span>Historical plot values plus current full-split best/final for orientation.</span><span>Public peers may contain GreekMMLU training overlap.</span></figcaption></figure><div class="table-wrap"><table id="peer-table"></table></div></section>

<section class="section" id="interpretation" style="--accent:var(--gold)"><div class="head"><div class="bar"></div><div><div class="kicker">08 · Interpretation</div><h2>What the combined evidence says.</h2></div></div><div class="notes"><article class="note caution"><h3>Plateau ≠ convergence</h3><p>The aggregate accuracy wanders around a plateau while hundreds of individual questions cross the decision boundary between adjacent checkpoints.</p></article><article class="note"><h3>Drift has subject structure</h3><p>Most subject groups are below their update-9,536 value at the endpoint, but a minority continue improving. The endpoint is a different capability mix, not simply a uniformly worse copy.</p></article><article class="note good"><h3>D0 exposure is auditable</h3><p id="source-interpretation"></p></article><article class="note caution"><h3>No causal attribution from one run</h3><p>The join can test whether a category had been seen; it cannot prove that a local exposure increment caused a subject score change. That requires controlled schedules or source-order seeds.</p></article></div></section>

<section class="section" id="evidence" style="--accent:var(--teal)"><div class="head"><div class="bar"></div><div><div class="kicker">09 · Reproducibility</div><h2>Evidence boundaries and receipts.</h2></div></div><div class="evidence" id="evidence-grid"></div></section>
</main><script type="application/json" id="report-data">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('report-data').textContent),NS='http://www.w3.org/2000/svg';
const C={ink:'#17242d',muted:'#66747d',rule:'#d7cec0',red:'#9b3438',blue:'#315f78',teal:'#39766e',gold:'#b38730',green:'#47734e',orange:'#b65e33',grey:'#8a9296'};
const expByIter=Object.fromEntries(D.exposure.checkpoint_summary.map(r=>[r.iteration,r]));
const tokensB=i=>expByIter[i].seen_active_tokens/1e9, pct=v=>(100*v).toFixed(2)+'%', num=v=>Number(v).toLocaleString();
function el(name,attrs={},text=''){const n=document.createElementNS(NS,name);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text!=='')n.textContent=text;return n}
function domain(vals,p=.08){let a=Math.min(...vals),b=Math.max(...vals);if(a===b){a-=1;b+=1}const d=(b-a)*p;return[a-d,b+d]}
function path(points,x,y){return points.map((p,i)=>(i?'L':'M')+x(p[0]).toFixed(2)+','+y(p[1]).toFixed(2)).join(' ')}
function lineChart(id,series,opt){const svg=document.getElementById(id),W=1540,H=opt.height||630,m={l:105,r:190,t:68,b:82};const all=series.flatMap(s=>s.points),xs=all.map(p=>p[0]),ys=all.map(p=>p[1]);let [y0,y1]=opt.domain||domain(ys,.10),x0=Math.min(...xs),x1=Math.max(...xs);const x=v=>m.l+(v-x0)/(x1-x0)*(W-m.l-m.r),y=v=>H-m.b-(v-y0)/(y1-y0)*(H-m.t-m.b);svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.append(el('title',{},opt.title));svg.append(el('desc',{},opt.desc));for(let i=0;i<6;i++){const v=y0+(y1-y0)*i/5;svg.append(el('line',{x1:m.l,x2:W-m.r,y1:y(v),y2:y(v),stroke:C.rule}));svg.append(el('text',{x:m.l-12,y:y(v)+4,fill:C.muted,'font-size':12,'text-anchor':'end'},opt.yfmt(v)))}for(let i=0;i<6;i++){const v=x0+(x1-x0)*i/5;svg.append(el('text',{x:x(v),y:H-m.b+30,fill:C.muted,'font-size':12,'text-anchor':'middle'},v.toFixed(1)))}if(opt.diagonal){const lo=Math.max(x0,y0),hi=Math.min(x1,y1);svg.append(el('path',{d:path([[lo,lo],[hi,hi]],x,y),fill:'none',stroke:C.ink,'stroke-width':2,'stroke-dasharray':'8 7',opacity:.55}))}const endpoints=[];series.forEach(s=>{svg.append(el('path',{d:path(s.points,x,y),fill:'none',stroke:s.color,'stroke-width':s.width||3,'stroke-dasharray':s.dash||'','stroke-linecap':'round','stroke-linejoin':'round'}));s.points.forEach(p=>svg.append(el('circle',{cx:x(p[0]),cy:y(p[1]),r:s.radius||3,fill:s.color,opacity:.95})));endpoints.push({s,p:s.points.at(-1),py:y(s.points.at(-1)[1])})});if(opt.labels){endpoints.sort((a,b)=>a.py-b.py);const gap=16;endpoints.forEach((e,i)=>e.ly=Math.max(m.t+6,e.py,i?endpoints[i-1].ly+gap:m.t+6));const over=endpoints.at(-1).ly-(H-m.b-4);if(over>0)endpoints.forEach(e=>e.ly-=over);endpoints.forEach(e=>{const lx=W-m.r+12;svg.append(el('line',{x1:x(e.p[0])+5,x2:lx-4,y1:e.py,y2:e.ly,stroke:e.s.color}));svg.append(el('text',{x:lx,y:e.ly+4,fill:e.s.color,'font-size':11,'font-weight':850},e.s.label))})}svg.append(el('line',{x1:m.l,x2:W-m.r,y1:H-m.b,y2:H-m.b,stroke:C.ink}));svg.append(el('line',{x1:m.l,x2:m.l,y1:m.t,y2:H-m.b,stroke:C.ink}));svg.append(el('text',{x:(m.l+W-m.r)/2,y:H-20,fill:C.muted,'font-size':14,'font-weight':750,'text-anchor':'middle'},opt.xlabel));svg.append(el('text',{x:m.l,y:28,fill:C.ink,'font-size':22,'font-family':'Georgia','font-weight':700},opt.title))}
function optimizerGeometry(){
 const svg=document.getElementById('optimizer-geometry'),hp=D.hyperparameters,W=1540,H=1035,m={l:164,r:68,t:86,b:72},totalB=expByIter[hp.total_updates].seen_active_tokens/1e9,x=v=>m.l+v/totalB*(W-m.l-m.r),bestI=D.derived.best_iteration,bestB=tokensB(bestI),halfB=totalB/2,warmB=totalB*hp.warmup_end/hp.total_updates,coolB=totalB*hp.cooldown_start/hp.total_updates;
 const f=b=>Math.log(.5)/Math.log(b+1e-8)-1,finv=t=>Math.pow(.5,1/(t+1)),beta3At=u=>{if(u>=hp.beta3_warmup)return hp.beta3_final;const a=u/hp.beta3_warmup;return finv((1-a)*f(hp.beta1)+a*f(hp.beta3_final))},alphaAt=u=>hp.alpha_final*Math.min(1,u/hp.alpha_warmup),lrAt=u=>{if(u<=hp.warmup_end)return hp.minimum_lr+(hp.peak_lr-hp.minimum_lr)*u/hp.warmup_end;if(u<hp.cooldown_start)return hp.peak_lr;const q=(u-hp.cooldown_start)/(hp.total_updates-hp.cooldown_start);return hp.minimum_lr+(hp.peak_lr-hp.minimum_lr)*(1-Math.sqrt(Math.max(0,Math.min(1,q))))},hlB=b=>Math.log(.5)/Math.log(b)*hp.tokens_per_update/1e9;
 const bestBeta=beta3At(bestI),bestAlpha=alphaAt(bestI),bestHalfSteps=Math.log(.5)/Math.log(bestBeta),bestHalfB=bestHalfSteps*hp.tokens_per_update/1e9;
 document.getElementById('optimizer-metrics').innerHTML=`<article><div class=n>${(hp.peak_lr*1e5).toFixed(2)}×10⁻⁵</div><div class=l>learning rate at 40B · still peak</div></article><article><div class=n>${bestAlpha.toFixed(2)}</div><div class=l>effective AdEMAMix α · target ${hp.alpha_final.toFixed(1)}</div></article><article><div class=n>${bestBeta.toFixed(6)}</div><div class=l>effective β3 · ${bestHalfSteps.toFixed(0)}-update / ${bestHalfB.toFixed(2)}B-token half-life</div></article><article><div class=n>${pct(expByIter[bestI].corpus_token_fraction)}</div><div class=l>complete corpus consumed · HPLT and GlossAPI match</div></article>`;
 svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.append(el('title',{},'GreekMMLU, learning rate, AdEMAMix schedules and source exposure on one token axis'));svg.append(el('desc',{},'Five aligned panels show the GreekMMLU accuracy peak near 40 billion tokens, constant peak learning rate, rising alpha, rising beta3 half-life, and uniform HPLT and GlossAPI exposure.'));
 const plotTop=m.t,plotBottom=H-m.b;svg.append(el('rect',{x:x(coolB),y:plotTop,width:x(totalB)-x(coolB),height:plotBottom-plotTop,fill:C.gold,opacity:.075}));
 for(let i=0;i<=8;i++){const v=totalB*i/8;svg.append(el('line',{x1:x(v),x2:x(v),y1:plotTop,y2:plotBottom,stroke:C.rule,opacity:.62}));svg.append(el('text',{x:x(v),y:H-m.b+29,fill:C.muted,'font-size':11,'text-anchor':'middle'},v.toFixed(1)))}
 const paneH=142,gap=37,panes=[
  {label:'GreekMMLU clean accuracy',domain:[.34,.59],fmt:v=>(100*v).toFixed(0)+'%',series:[{label:'measured accuracy',color:C.red,width:4,points:D.metrics.map(r=>[tokensB(r.iteration),r.clean_accuracy]),dots:true}]},
  {label:'Learning rate',domain:[0,hp.peak_lr*1.08],fmt:v=>(v*1e5).toFixed(1)+'e−5',series:[{label:'reconstructed WSD-10',color:C.blue,width:4,points:Array.from({length:241},(_,i)=>{const u=hp.total_updates*i/240;return[totalB*i/240,lrAt(u)]})}]},
  {label:'AdEMAMix α',domain:[0,hp.alpha_final*1.08],fmt:v=>v.toFixed(1),series:[{label:'slow-EMA mixing coefficient',color:'#725a8f',width:4,points:Array.from({length:241},(_,i)=>{const u=hp.total_updates*i/240;return[totalB*i/240,alphaAt(u)]})}]},
  {label:'EMA memory half-life',domain:[0,3.15],fmt:v=>v.toFixed(1)+'B',series:[{label:'β3 slow first moment',color:C.teal,width:4,points:Array.from({length:241},(_,i)=>{const u=hp.total_updates*i/240;return[totalB*i/240,hlB(beta3At(u))]})},{label:'β2 second moment',color:C.gold,width:2,dash:'8 6',points:[[0,hlB(hp.beta2)],[totalB,hlB(hp.beta2)]]},{label:'β1 fast first moment',color:C.grey,width:2,dash:'3 6',points:[[0,hlB(hp.beta1)],[totalB,hlB(hp.beta1)]]}]},
  {label:'Exact token exposure',domain:[0,1],fmt:v=>(100*v).toFixed(0)+'%',series:[{label:'whole corpus',color:C.ink,width:2,dash:'7 6',points:D.exposure.checkpoint_summary.map(r=>[tokensB(r.iteration),r.corpus_token_fraction])},{label:'HPLT',color:C.orange,width:4,points:hp.hplt_trajectory.map(r=>[tokensB(r.iteration),r.seen_token_fraction]),dots:true},{label:'GlossAPI / non-HPLT',color:C.green,width:3,points:hp.glossapi_trajectory.map(r=>[tokensB(r.iteration),r.seen_token_fraction]),dots:true}]}
 ];
 panes.forEach((pane,pi)=>{const top=plotTop+pi*(paneH+gap),bottom=top+paneH,y=v=>bottom-(v-pane.domain[0])/(pane.domain[1]-pane.domain[0])*paneH;svg.append(el('text',{x:m.l-14,y:top+16,fill:C.ink,'font-size':14,'font-weight':850,'text-anchor':'end'},pane.label));for(let j=0;j<3;j++){const v=pane.domain[0]+(pane.domain[1]-pane.domain[0])*j/2;svg.append(el('line',{x1:m.l,x2:W-m.r,y1:y(v),y2:y(v),stroke:C.rule}));svg.append(el('text',{x:m.l-12,y:y(v)+4,fill:C.muted,'font-size':10,'text-anchor':'end'},pane.fmt(v)))}pane.series.forEach((s,si)=>{svg.append(el('path',{d:path(s.points,x,y),fill:'none',stroke:s.color,'stroke-width':s.width,'stroke-dasharray':s.dash||'','stroke-linecap':'round','stroke-linejoin':'round'}));if(s.dots)s.points.forEach(p=>svg.append(el('circle',{cx:x(p[0]),cy:y(p[1]),r:3.2,fill:s.color})));svg.append(el('text',{x:m.l+10+si*245,y:top+18,fill:s.color,'font-size':10,'font-weight':850},s.label))});svg.append(el('line',{x1:m.l,x2:W-m.r,y1:bottom,y2:bottom,stroke:C.ink,opacity:.55}))});
 [{v:halfB,color:C.grey,dash:'6 6',label:`50% corpus · ${halfB.toFixed(2)}B`,y:28,anchor:'end',dx:-8},{v:bestB,color:C.red,dash:'',label:`best measured · ${bestB.toFixed(3)}B`,y:48,anchor:'start',dx:8},{v:warmB,color:C.teal,dash:'4 5',label:`LR warmup end · ${warmB.toFixed(2)}B`,y:69,anchor:'start',dx:6},{v:coolB,color:C.gold,dash:'7 5',label:`cooldown begins · ${coolB.toFixed(2)}B`,y:69,anchor:'start',dx:6}].forEach(a=>{svg.append(el('line',{x1:x(a.v),x2:x(a.v),y1:plotTop-5,y2:plotBottom,stroke:a.color,'stroke-width':a.color===C.red?3:2,'stroke-dasharray':a.dash,opacity:.9}));svg.append(el('text',{x:x(a.v)+a.dx,y:a.y,fill:a.color,'font-size':11,'font-weight':850,'text-anchor':a.anchor},a.label))});
 svg.append(el('text',{x:(m.l+W-m.r)/2,y:H-18,fill:C.muted,'font-size':14,'font-weight':750,'text-anchor':'middle'},'Exact active training tokens (billions)'));
}optimizerGeometry();
function betaHistoryTokens(){
 const svg=document.getElementById('beta-history-tokens'),hp=D.hyperparameters,W=1540,H=900,totalB=expByIter[hp.total_updates].seen_active_tokens/1e9,left={x0:170,x1:765},right={x0:945,x1:1375},alphaX=1460,memoryMax=3.08,iterations=[4768,9536,14304,hp.total_updates],rowTop=130,rowGap=182;
 const sourceX=v=>left.x0+v/totalB*(left.x1-left.x0),memoryX=v=>right.x0+v/memoryMax*(right.x1-right.x0),f=b=>Math.log(.5)/Math.log(b+1e-8)-1,finv=t=>Math.pow(.5,1/(t+1)),beta3At=u=>{if(u>=hp.beta3_warmup)return hp.beta3_final;const a=u/hp.beta3_warmup;return finv((1-a)*f(hp.beta1)+a*f(hp.beta3_final))},alphaAt=u=>hp.alpha_final*Math.min(1,u/hp.alpha_warmup),halfLifeB=b=>Math.log(.5)/Math.log(b)*hp.tokens_per_update/1e9;
 const hpltByIter=Object.fromEntries(hp.hplt_trajectory.map(r=>[r.iteration,r.seen_tokens/1e9])),glossByIter=Object.fromEntries(hp.glossapi_trajectory.map(r=>[r.iteration,r.seen_tokens/1e9]));
 svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.append(el('title',{},'Cumulative source tokens and optimizer memory half-lives at four training checkpoints'));svg.append(el('desc',{},'At 20, 40, 60 and 76.7 billion tokens, stacked bars show exact source tokens consumed while adjacent bars show beta1, beta3 and beta2 exponential-memory half-lives. Alpha badges show the coefficient on the slow beta3 branch.'));
 svg.append(el('text',{x:left.x0,y:30,fill:C.ink,'font-size':21,'font-family':'Georgia','font-weight':700},'Exact cumulative tokens consumed'));
 svg.append(el('text',{x:right.x0,y:30,fill:C.ink,'font-size':21,'font-family':'Georgia','font-weight':700},'Gradient-history half-life'));
 [[C.orange,'HPLT'],[C.green,'GlossAPI / non-HPLT'],[C.blue,'foreign + Old-Greek replay']].forEach((q,i)=>{svg.append(el('rect',{x:left.x0+i*190,y:51,width:23,height:8,fill:q[0]}));svg.append(el('text',{x:left.x0+30+i*190,y:61,fill:q[0],'font-size':10,'font-weight':850},q[1]))});
 [[C.red,'β1 fast signed direction'],['#725a8f','β3 slow signed direction'],[C.gold,'β2 squared magnitude']].forEach((q,i)=>{svg.append(el('line',{x1:right.x0+i*168,x2:right.x0+22+i*168,y1:55,y2:55,stroke:q[0],'stroke-width':5}));svg.append(el('text',{x:right.x0+29+i*168,y:59,fill:q[0],'font-size':9,'font-weight':850},q[1]))});
 [0,20,40,60,totalB].forEach(v=>{svg.append(el('line',{x1:sourceX(v),x2:sourceX(v),y1:82,y2:H-62,stroke:C.rule,opacity:.7}));svg.append(el('text',{x:sourceX(v),y:H-37,fill:C.muted,'font-size':10,'text-anchor':'middle'},v.toFixed(v===totalB?1:0)+'B'))});
 [0,1,2,3].forEach(v=>{svg.append(el('line',{x1:memoryX(v),x2:memoryX(v),y1:82,y2:H-62,stroke:C.rule,opacity:.7}));svg.append(el('text',{x:memoryX(v),y:H-37,fill:C.muted,'font-size':10,'text-anchor':'middle'},v.toFixed(0)+'B'))});
 iterations.forEach((iter,index)=>{const y=rowTop+index*rowGap,total=expByIter[iter].seen_active_tokens/1e9,h=hpltByIter[iter],g=glossByIter[iter],replay=total-h-g,beta3=beta3At(iter),histories=[{name:'β1',value:halfLifeB(hp.beta1),color:C.red,y:y+57},{name:'β3',value:halfLifeB(beta3),color:'#725a8f',y:y+91},{name:'β2',value:halfLifeB(hp.beta2),color:C.gold,y:y+125}],alpha=alphaAt(iter),highlight=iter===D.derived.best_iteration;
  svg.append(el('rect',{x:130,y:y-34,width:1370,height:154,rx:7,fill:highlight?'rgba(155,52,56,.055)':'rgba(255,253,248,.34)',stroke:highlight?C.red:C.rule,'stroke-width':highlight?2:1}));
  svg.append(el('text',{x:142,y:y-9,fill:highlight?C.red:C.ink,'font-size':16,'font-family':'Georgia','font-weight':850},`${total.toFixed(index===3?3:1)}B tokens seen${highlight?' · GreekMMLU peak':''}`));
  let cursor=0;[[h,C.orange,'H'],[g,C.green,'G'],[replay,C.blue,'R']].forEach(([value,color,label])=>{const x0=sourceX(cursor),x1=sourceX(cursor+value);svg.append(el('rect',{x:x0,y:y+20,width:Math.max(0,x1-x0),height:38,fill:color,opacity:.86}));if(x1-x0>25)svg.append(el('text',{x:(x0+x1)/2,y:y+44,fill:'#fff','font-size':9,'font-weight':900,'text-anchor':'middle'},`${label} ${value.toFixed(2)}B`));cursor+=value});
  svg.append(el('line',{x1:left.x0,x2:sourceX(total),y1:y+64,y2:y+64,stroke:C.ink,opacity:.65}));
  histories.forEach(q=>{svg.append(el('text',{x:right.x0-12,y:q.y+4,fill:q.color,'font-size':10,'font-weight':900,'text-anchor':'end'},q.name));svg.append(el('line',{x1:right.x0,x2:memoryX(q.value),y1:q.y,y2:q.y,stroke:q.color,'stroke-width':q.name==='β3'?8:5,'stroke-linecap':'round'}));svg.append(el('circle',{cx:memoryX(q.value),cy:q.y,r:q.name==='β3'?5:3.5,fill:q.color}));svg.append(el('text',{x:memoryX(q.value)+8,y:q.y+4,fill:q.color,'font-size':10,'font-weight':850},`${q.value.toFixed(3)}B`))});
  const radius=8+alpha*2.6;svg.append(el('circle',{cx:alphaX,cy:y+91,r:radius,fill:'#725a8f',opacity:.82}));svg.append(el('text',{x:alphaX,y:y+95,fill:'#fff','font-size':10,'font-weight':900,'text-anchor':'middle'},alpha.toFixed(2)));svg.append(el('text',{x:alphaX,y:y+125,fill:'#725a8f','font-size':10,'font-weight':850,'text-anchor':'middle'},'α × slow β3'));
 });
 svg.append(el('text',{x:(left.x0+left.x1)/2,y:H-10,fill:C.muted,'font-size':12,'font-weight':750,'text-anchor':'middle'},'Absolute cumulative training tokens'));
 svg.append(el('text',{x:(right.x0+right.x1)/2,y:H-10,fill:C.muted,'font-size':12,'font-weight':750,'text-anchor':'middle'},'Tokens corresponding to one EMA half-life'));
}betaHistoryTokens();
document.getElementById('hero-finding').textContent=`From the accuracy peak to the endpoint, ${num(D.derived.peak_to_final_newly_wrong)} formerly correct answers are lost and ${num(D.derived.peak_to_final_newly_correct)} new ones are gained. The net score falls—but ${pct(D.derived.peak_to_final_correct_churn)} of all questions change correctness state.`;
document.getElementById('headline-metrics').innerHTML=`<article><div class=n>${pct(D.derived.best_clean_accuracy)}</div><div class=l>best clean accuracy · update ${num(D.derived.best_iteration)}</div></article><article><div class=n>${pct(D.derived.final_clean_accuracy)}</div><div class=l>final clean accuracy</div></article><article><div class=n>${pct(D.derived.peak_to_final_flip_rate)}</div><div class=l>selected answer changed · peak → final</div></article><article><div class=n>${pct(D.derived.transient_correct_fraction)}</div><div class=l>correct at some, but not all, checkpoints</div></article>`;
document.getElementById('source-interpretation').textContent=`The stationary D0 schedule is empirically uniform: among every source of at least 1B tokens, the largest departure from overall corpus progress is only ${(100*D.derived.large_source_max_abs_progress_gap).toFixed(2)} percentage points. Across HPLT register categories, the maximum is ${(100*D.derived.hplt_register_max_abs_progress_gap).toFixed(2)} points.`;
const metricPoints=(field)=>D.metrics.map(r=>[tokensB(r.iteration),r[field]]);lineChart('accuracy',[{label:'clean accuracy',color:C.red,width:5,radius:4,points:metricPoints('clean_accuracy')}],{title:'GreekMMLU accuracy · full horizon',desc:'Decontaminated GreekMMLU accuracy at 19 checkpoints.',yfmt:v=>(100*v).toFixed(1)+'%',xlabel:'Exact active training tokens (billions)'});lineChart('nll',[{label:'clean choice NLL',color:C.blue,width:5,radius:4,points:metricPoints('clean_choice_nll')}],{title:'GreekMMLU choice NLL · full horizon',desc:'Decontaminated multiple-choice negative log likelihood at 19 checkpoints.',yfmt:v=>v.toFixed(3),xlabel:'Exact active training tokens (billions)'});
function answerFlow(){const svg=document.getElementById('answer-flow'),rows=D.drift.checkpoint_table.slice(1),W=1540,H=620,m={l:105,r:40,t:62,b:88},rN=D.drift.clean_subset.n,x=v=>m.l+v/tokensB(rows.at(-1).iteration)*(W-m.l-m.r),max=Math.max(...rows.flatMap(r=>[r.vs_previous.newly_correct,r.vs_previous.newly_wrong]))/rN*1.18,y=v=>H/2-v/max*(H/2-m.t);svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.append(el('title',{},'Newly correct and newly wrong GreekMMLU answers between checkpoints'));for(let i=1;i<=3;i++){const v=max*i/3;[v,-v].forEach(q=>{svg.append(el('line',{x1:m.l,x2:W-m.r,y1:y(q),y2:y(q),stroke:C.rule}));svg.append(el('text',{x:m.l-10,y:y(q)+4,fill:C.muted,'font-size':11,'text-anchor':'end'},(100*Math.abs(q)).toFixed(1)+'%'))})}svg.append(el('line',{x1:m.l,x2:W-m.r,y1:y(0),y2:y(0),stroke:C.ink,'stroke-width':1.4}));const bw=(W-m.l-m.r)/rows.length*.31;rows.forEach(r=>{const cx=x(tokensB(r.iteration)),g=r.vs_previous.newly_correct/rN,l=-r.vs_previous.newly_wrong/rN;svg.append(el('rect',{x:cx-bw-1,y:y(g),width:bw,height:y(0)-y(g),fill:C.green,opacity:.86}));svg.append(el('rect',{x:cx+1,y:y(0),width:bw,height:y(l)-y(0),fill:C.red,opacity:.83}));svg.append(el('text',{x:cx,y:H-m.b+26,fill:C.muted,'font-size':10,'text-anchor':'middle'},tokensB(r.iteration).toFixed(1))) });svg.append(el('text',{x:(m.l+W-m.r)/2,y:H-22,fill:C.muted,'font-size':14,'text-anchor':'middle'},'Exact active training tokens (billions)'));svg.append(el('text',{x:m.l,y:26,fill:C.ink,'font-size':22,'font-family':'Georgia','font-weight':700},'Correctness crossings between adjacent checkpoints'))}answerFlow();
function table(node,headers,rows){node.innerHTML=`<thead><tr>${headers.map(h=>`<th>${h}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map((v,i)=>`<td>${v}</td>`).join('')}</tr>`).join('')}</tbody>`}
table(document.getElementById('answer-table'),['checkpoint','active tokens','accuracy','newly correct','newly wrong','net correct','choice flips','correct-set churn','prior-correct retained'],D.drift.checkpoint_table.map((r,i)=>{const v=r.vs_previous;return[num(r.iteration),tokensB(r.iteration).toFixed(3)+'B',pct(r.accuracy),v?num(v.newly_correct):'—',v?num(v.newly_wrong):'—',v?(v.net_correct_change>0?'+':'')+num(v.net_correct_change):'—',v?pct(v.answer_choice_flip_rate):'—',v?pct(v.correct_set_churn_rate):'—',v?pct(v.prior_correct_retention):'—']}));
function subjects(){const rows=[...D.drift.subject_final_vs_best].sort((a,b)=>a.final_minus_best_checkpoint-b.final_minus_best_checkpoint),svg=document.getElementById('subjects-chart'),W=1540,H=rows.length*34+110,m={l:330,r:90,t:45,b:55},lim=Math.max(...rows.map(r=>Math.abs(r.final_minus_best_checkpoint)))*1.12,x=v=>m.l+(v+lim)/(2*lim)*(W-m.l-m.r),step=(H-m.t-m.b)/rows.length;svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.append(el('title',{},'Subject accuracy change from global peak checkpoint to final'));svg.append(el('line',{x1:x(0),x2:x(0),y1:m.t,y2:H-m.b,stroke:C.ink,'stroke-width':1.4}));rows.forEach((r,i)=>{const y=m.t+i*step+3,h=Math.max(12,step-7),v=r.final_minus_best_checkpoint,xx=x(v);svg.append(el('rect',{x:Math.min(x(0),xx),y,width:Math.abs(xx-x(0)),height:h,fill:v>=0?C.green:C.red,opacity:.8}));svg.append(el('text',{x:m.l-12,y:y+h*.72,'text-anchor':'end',fill:C.ink,'font-size':11},r.subject));svg.append(el('text',{x:v>=0?xx+6:xx-6,y:y+h*.72,'text-anchor':v>=0?'start':'end',fill:v>=0?C.green:C.red,'font-size':10,'font-weight':800},(100*v).toFixed(1)+' pp'))});svg.append(el('text',{x:W/2,y:H-15,'text-anchor':'middle',fill:C.muted,'font-size':13},'Final minus peak accuracy (percentage points)'))}subjects();
table(document.getElementById('subjects-table'),['subject','n','initial','best across run','best update','at global peak','final','final − global peak'],Object.entries(D.drift.subjects).sort((a,b)=>b[1].n-a[1].n).map(([name,r])=>{const gp=r.trajectory.find(x=>x.iteration===D.derived.best_iteration);return[name,num(r.n),pct(r.initial_accuracy),pct(r.best_accuracy),num(r.best_iteration),pct(gp.accuracy),pct(r.final_accuracy),(100*(r.final_accuracy-gp.accuracy)).toFixed(2)+' pp']}));
function trajectoryMatrix(id,groups,label){const checkpoints=D.drift.checkpoint_table.map(r=>r.iteration),headers=[label,'n',...checkpoints.map(i=>`u${num(i)}<br><span class=small>${tokensB(i).toFixed(1)}B</span>`)];const body=Object.entries(groups).sort((a,b)=>b[1].n-a[1].n).map(([name,r])=>{const values=r.trajectory.map(p=>p.accuracy),mean=values.reduce((a,b)=>a+b,0)/values.length,best=Math.max(...values),worst=Math.min(...values),scale=Math.max(best-mean,mean-worst,1e-12),flat=Math.abs(best-worst)<1e-12;return[name,num(r.n),...r.trajectory.map((p,i)=>{const previous=i?r.trajectory[i-1].accuracy:null,fromPrevious=previous===null?null:p.accuracy-previous,fromMean=p.accuracy-mean,strength=Math.min(1,Math.abs(fromMean)/scale),alpha=strength<.015?0:.08+.30*strength,color=fromMean>=0?`rgba(71,115,78,${alpha.toFixed(3)})`:`rgba(155,52,56,${alpha.toFixed(3)})`,isBest=!flat&&Math.abs(p.accuracy-best)<1e-12,isWorst=!flat&&Math.abs(p.accuracy-worst)<1e-12,border=isBest?`inset 0 0 0 3px ${C.green}`:isWorst?`inset 0 0 0 3px ${C.red}`:'none',weight=isBest||isWorst?850:500,previousText=fromPrevious===null?'initial checkpoint':`${fromPrevious>=0?'+':''}${(100*fromPrevious).toFixed(2)} pp from preceding checkpoint`,title=`row mean ${pct(mean)} · ${fromMean>=0?'+':''}${(100*fromMean).toFixed(2)} pp versus mean · ${previousText}`;return `<span title="${title}" style="display:block;margin:-10px -12px;padding:10px 12px;background:${color};box-shadow:${border};font-weight:${weight}">${pct(p.accuracy)}</span>`})]});table(document.getElementById(id),headers,body)}
trajectoryMatrix('subject-checkpoint-table',D.drift.subjects,'subject');trajectoryMatrix('level-checkpoint-table',D.drift.levels,'level');
const palette=[C.red,C.blue,C.teal,C.gold,C.orange,C.green,'#725a8f','#886a45','#4f748c','#8b4f69'];
function exposureSeries(rows,count){return rows.slice(0,count).map((r,i)=>({label:r.label,color:palette[i%palette.length],width:i<3?4:2.6,radius:2.5,points:r.trajectory.map((p,j)=>[D.exposure.checkpoint_summary[j].corpus_token_fraction,p.seen_token_fraction])}))}
lineChart('source-lines',exposureSeries(D.exposure.sources,9),{title:'Cumulative source exposure versus corpus progress',desc:'Largest source families by token mass.',domain:[0,1],diagonal:true,labels:true,yfmt:v=>(100*v).toFixed(0)+'%',xlabel:'Fraction of the complete training stream consumed'});
function exposureTable(id,rows){const headers=['source / category','tokens','documents',...D.exposure.checkpoint_summary.map(r=>tokensB(r.iteration).toFixed(1)+'B')];const body=rows.map(r=>[r.label,(r.total_tokens/1e9).toFixed(3)+'B',num(r.total_documents),...r.trajectory.map(p=>`<span title="Touched ${pct(p.touched_document_fraction)} · fully seen ${pct(p.fully_seen_document_fraction)}">${pct(p.seen_token_fraction)}</span>`)]);table(document.getElementById(id),headers,body)}exposureTable('source-table',D.exposure.sources);
lineChart('register-lines',exposureSeries(D.exposure.hplt_register_level_1,8),{title:'Cumulative HPLT web-register exposure',desc:'Largest HPLT register level 1 categories by token mass.',domain:[0,1],diagonal:true,labels:true,yfmt:v=>(100*v).toFixed(0)+'%',xlabel:'Fraction of the complete training stream consumed'});exposureTable('register-table',D.exposure.hplt_register_level_1);
function peers(){const hist=D.peer.models.map(r=>({model:r.model,accuracy:r.accuracy,evidence:'historical rendered plot',caveat:r.caveat}));hist.push({model:'Current sanitized 8B · best',accuracy:D.derived.best_full_accuracy,evidence:'current receipt',caveat:`update ${num(D.derived.best_iteration)}`},{model:'Current sanitized 8B · final',accuracy:D.derived.final_full_accuracy,evidence:'current receipt',caveat:'terminal checkpoint'});hist.sort((a,b)=>b.accuracy-a.accuracy);const svg=document.getElementById('peer-chart'),W=1540,H=650,m={l:330,r:100,t:55,b:65},x=v=>m.l+v/.72*(W-m.l-m.r),step=(H-m.t-m.b)/hist.length;svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.append(el('title',{},'Historical peer GreekMMLU accuracy and current run context'));[0,.2,.4,.6].forEach(v=>{svg.append(el('line',{x1:x(v),x2:x(v),y1:m.t,y2:H-m.b,stroke:C.rule}));svg.append(el('text',{x:x(v),y:H-m.b+26,'text-anchor':'middle',fill:C.muted,'font-size':11},pct(v)))});const ref=D.peer.reference.accuracy;svg.append(el('line',{x1:x(ref),x2:x(ref),y1:m.t-10,y2:H-m.b,stroke:C.gold,'stroke-width':3,'stroke-dasharray':'7 6'}));svg.append(el('text',{x:x(ref)+6,y:m.t-18,fill:C.gold,'font-size':11,'font-weight':850},`historical Apertus base ${pct(ref)}`));hist.forEach((r,i)=>{const y=m.t+i*step+5,h=Math.max(20,step-10),current=r.evidence==='current receipt';svg.append(el('rect',{x:m.l,y,width:x(r.accuracy)-m.l,height:h,fill:current?C.teal:C.red,opacity:current?.88:.72}));svg.append(el('text',{x:m.l-12,y:y+h*.7,'text-anchor':'end',fill:C.ink,'font-size':13,'font-weight':current?850:650},r.model));svg.append(el('text',{x:x(r.accuracy)+8,y:y+h*.7,fill:current?C.teal:C.red,'font-size':13,'font-weight':900},pct(r.accuracy)))});svg.append(el('text',{x:W/2,y:H-15,'text-anchor':'middle',fill:C.muted,'font-size':13},'GreekMMLU accuracy · full 16,632-question split'));
table(document.getElementById('peer-table'),['model','accuracy','evidence class','note'],hist.map(r=>[r.model,pct(r.accuracy),r.evidence,r.caveat]));}peers();
const bind=Object.entries(D.bindings).map(([k,v])=>`<article><h3>${k.replaceAll('_',' ')}</h3><p><code>${v.path}</code></p><p class=small>SHA-256 ${v.sha256} · ${num(v.bytes)} bytes</p></article>`).join('');document.getElementById('evidence-grid').innerHTML=bind+`<article><h3>Exact schedule join</h3><p>${num(D.exposure.classification.modern_retained_documents_joined)} retained Modern-Greek documents joined across ${num(D.exposure.classification.modern_input_files)} frozen parquet files.</p><p class=small>${D.exposure.definition.seen_tokens}</p></article><article><h3>Historical peer limit</h3><p>${D.peer.evidence.raw_payload_status}</p><p class=small>${D.peer.comparability}</p></article>`;
</script></body></html>'''.replace("__DATA__", encoded)
    args.output.write_text(html, encoding="utf-8")
    print(json.dumps({"ok": True, "html": str(args.output.resolve()), "data": str(data_path.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
