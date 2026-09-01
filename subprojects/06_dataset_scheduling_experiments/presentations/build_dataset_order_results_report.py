#!/usr/bin/env python3
"""Build the self-contained five-arm dataset-order experiment report."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "dataset_order_20260805"
OUTPUT = HERE / "DATA_ORDER_MIX_RESULTS_20260805.html"

EXPECTED_HASHES = {
    "core_campaign_summary.json": "eb8a2db1949bbc2be3108c273d947f58ec086e88b13fd7ed9e7ed7338bf8591e",
    "dataset_order_selection_analysis.json": "581f59294fd4f728bd1ff318dfa2c1d4fb52914b0d49e6001b83ae8a6e677b19",
    "full_endpoint_validation_receipt.json": "6e49e02e67e0baf75acd965e49445fd5bf1e657552d9bd3cfec5187953311146",
    "greekmmlu_trajectory.json": "74e56f40ab05e982aac7dc6cebcae1422e3fa97db15525b035f92581d26211cb",
    "validation_trajectory.json": "bfb7c7073adee541f4e3ed74548b343cf58d60b93249a54f2a900937d5972298",
}

ARMS = [
    "D0_mixed",
    "D1_hard_h_to_g",
    "D2_hard_g_to_h",
    "D3_gradual_h_to_g",
    "D4_gradual_g_to_h",
]

PANELS = [
    "hplt",
    "non_hplt",
    "greek_phd",
    "openarchives",
    "historical_polytonic",
    "neutral_external_modern_greek",
    "old_greek",
    "english",
    "code",
    "math",
    "de",
    "ru",
    "zh",
]

GLOSSAPI_PANELS = ["non_hplt", "greek_phd", "openarchives", "historical_polytonic"]
REPLAY_PANELS = ["old_greek", "english", "code", "math", "de", "ru", "zh"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_checked(name: str) -> dict:
    path = DATA_DIR / name
    actual = sha256(path)
    expected = EXPECTED_HASHES[name]
    if actual != expected:
        raise RuntimeError(f"hash mismatch for {name}: {actual} != {expected}")
    with path.open() as handle:
        return json.load(handle)


def assert_finite(value: object, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite value at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite(child, f"{path}[{index}]")


def build_payload() -> dict:
    validation = load_checked("validation_trajectory.json")
    greekmmlu = load_checked("greekmmlu_trajectory.json")
    core = load_checked("core_campaign_summary.json")
    selection = load_checked("dataset_order_selection_analysis.json")
    full = load_checked("full_endpoint_validation_receipt.json")

    assert validation["schema_version"] == "apertus_mini_validation_trajectory_v1"
    assert validation["status"] == "completed"
    assert validation["row_count"] == validation["expected_row_count"] == 5395
    assert greekmmlu["schema_version"] == "apertus_mini_greekmmlu_trajectory_v1"
    assert greekmmlu["status"] == "completed"
    assert greekmmlu["row_count"] == greekmmlu["expected_row_count"] == 415
    assert greekmmlu["evaluation_namespace"] == "fp32_v1"
    assert greekmmlu["authoritative_evaluation_dtype"] == "float32"
    assert core["schema_version"] == "apertus_mini_core_campaign_summary_v1"
    assert core["status"] == "completed" and core["winner_selected"] is False
    assert selection["schema_version"] == "apertus_dataset_order_selection_analysis_v1"
    assert selection["status"] == "completed"
    assert selection["decision"]["winner_selected"] is False
    assert selection["decision"]["provisional_observed_leader"] == "D0_mixed"
    assert selection["point_estimate_selection"]["passing_arms"] == [
        "D0_mixed",
        "D1_hard_h_to_g",
        "D3_gradual_h_to_g",
    ]
    assert full["schema_version"] == "apertus_mini_full_endpoint_validation_v1"
    assert full["status"] == "completed"
    assert full["iteration"] == 38496
    assert full["row_count"] == full["expected_row_count"] == 65
    assert_finite(validation["rows"], "validation.rows")
    assert_finite(greekmmlu["rows"], "greekmmlu.rows")
    assert_finite(core, "core")
    assert_finite(full["rows"], "full.rows")

    validation_by: dict[str, dict[str, list[list[float]]]] = {
        arm: {panel: [] for panel in PANELS} for arm in ARMS
    }
    for row in validation["rows"]:
        validation_by[row["arm_id"]][row["panel"]].append([row["iteration"], row["bpb"]])
    for arm in ARMS:
        for panel in PANELS:
            validation_by[arm][panel].sort(key=lambda item: item[0])
            assert len(validation_by[arm][panel]) == 83
            assert validation_by[arm][panel][0][0] == 0
            assert validation_by[arm][panel][-1][0] == 38496

    full_by: dict[str, dict[str, float]] = {arm: {} for arm in ARMS}
    for row in full["rows"]:
        full_by[row["arm_id"]][row["panel"]] = row["bpb"]
    assert all(set(full_by[arm]) == set(PANELS) for arm in ARMS)

    greekmmlu_by: dict[str, list[list[float]]] = {arm: [] for arm in ARMS}
    for row in greekmmlu["rows"]:
        greekmmlu_by[row["arm_id"]].append(
            [
                row["iteration"],
                row["clean_accuracy"],
                row["clean_choice_nll"],
                row["clean_correct_answer_bpb"],
            ]
        )
        assert row["clean_n"] == 16159
    for arm in ARMS:
        greekmmlu_by[arm].sort(key=lambda item: item[0])
        assert len(greekmmlu_by[arm]) == 83
        assert greekmmlu_by[arm][0][0] == 0 and greekmmlu_by[arm][-1][0] == 38496

    summary_by: dict[str, dict[str, dict]] = {arm: {} for arm in ARMS}
    for row in core["loss_learning_and_forgetting"]:
        summary_by[row["arm_id"]][row["panel"]] = {
            "initial": row["initial_bpb"],
            "best": row["best_bpb"],
            "best_iteration": row["best_iteration"],
            "fast_final": row["fast_final_bpb"],
            "full_final": row["full_final_bpb"],
            "forgetting": row["forgetting_bpb"],
            "relative_improvement": row["relative_improvement_from_initial"],
        }
    assert all(set(summary_by[arm]) == set(PANELS) for arm in ARMS)

    q_h = 0.6905693562215841
    q_g = 0.3094306437784159
    derived: dict[str, dict[str, float | int]] = {}
    for arm in ARMS:
        gloss_macro = mean(full_by[arm][panel] for panel in GLOSSAPI_PANELS)
        replay_final = mean(full_by[arm][panel] for panel in REPLAY_PANELS)
        replay_forgetting = mean(summary_by[arm][panel]["forgetting"] for panel in REPLAY_PANELS)
        gm = greekmmlu_by[arm]
        best_acc = max(gm, key=lambda row: row[1])
        best_nll = min(gm, key=lambda row: row[2])
        endpoint = gm[-1]
        derived[arm] = {
            "hplt": full_by[arm]["hplt"],
            "gloss_macro": gloss_macro,
            "balanced_greek": 0.5 * full_by[arm]["hplt"] + 0.5 * gloss_macro,
            "natural_greek": q_h * full_by[arm]["hplt"] + q_g * gloss_macro,
            "neutral": full_by[arm]["neutral_external_modern_greek"],
            "replay_final_macro": replay_final,
            "replay_forgetting_macro": replay_forgetting,
            "gm_accuracy": endpoint[1],
            "gm_choice_nll": endpoint[2],
            "gm_answer_bpb": endpoint[3],
            "gm_best_accuracy": best_acc[1],
            "gm_best_accuracy_iteration": best_acc[0],
            "gm_best_choice_nll": best_nll[2],
            "gm_best_choice_nll_iteration": best_nll[0],
        }

    return {
        "meta": {
            "run_id": "mini_cpt5_20260803T074854Z",
            "final_iteration": 38496,
            "active_tokens_per_arm": 80729939067,
            "checkpoint_count_per_arm": 83,
            "validation_panel_count": 13,
            "validation_binding_count": 5395,
            "greekmmlu_binding_count": 415,
            "greekmmlu_clean_n": 16159,
            "peak_lr": 0.00015,
            "minimum_lr": 0.000015,
            "warmup_end": 800,
            "hard_g_to_h": 11912,
            "hard_h_to_g": 26584,
            "cooldown_start": 30796,
            "q_h": q_h,
            "q_g": q_g,
            "gradual_exponent": 2.231741975484828,
            "checkpoint_averaging": False,
            "evaluation_dtype": "float32",
            "input_hashes": EXPECTED_HASHES,
        },
        "validation": validation_by,
        "full": full_by,
        "greekmmlu": greekmmlu_by,
        "summary": summary_by,
        "derived": derived,
        "selection": selection,
    }


TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>Five data orders — learning, forgetting, and GreekMMLU</title>
  <style>
    :root {
      --paper:#f7f3ea; --paper-deep:#eee6d8; --white:#fffdf8; --ink:#102131; --muted:#65717b; --rule:#d6cfc2;
      --red:#d6534d; --red-dark:#8d2e32; --blue:#356f9f; --teal:#277e74; --gold:#c18b32; --purple:#775ba6; --green:#3f7f57;
      --shadow:0 18px 50px rgba(20,30,38,.09);
    }
    *{box-sizing:border-box} html{scroll-behavior:smooth} body{margin:0;color:var(--ink);background:radial-gradient(circle at 94% 2%,rgba(193,139,50,.12),transparent 22rem),linear-gradient(135deg,var(--paper),#fbf8f1 72%,#f1eadf);font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
    a{color:var(--red-dark);text-decoration-thickness:1px;text-underline-offset:3px} code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.92em;overflow-wrap:anywhere} h1,h2,h3,p{margin-top:0} h1,h2{font-family:Georgia,"Times New Roman",serif;letter-spacing:-.035em} h1{max-width:1100px;margin-bottom:24px;font-size:clamp(52px,7vw,106px);line-height:.96} h2{max-width:1080px;margin-bottom:20px;font-size:clamp(34px,4.4vw,66px);line-height:1.03} h3{margin-bottom:5px;font-size:clamp(16px,1.4vw,22px)} p,li,td,th{font-size:clamp(14px,1.1vw,18px);line-height:1.48}
    .page{width:min(1500px,100%);margin:0 auto;padding:0 clamp(24px,5vw,78px) 90px}.hero{min-height:90vh;display:grid;align-content:center;position:relative;padding:72px 0 56px}.hero::before{content:"";position:absolute;left:calc(-1 * clamp(24px,5vw,78px));top:0;bottom:0;width:9px;background:var(--red)}
    .eyebrow{display:flex;justify-content:space-between;gap:24px;margin-bottom:30px;color:var(--muted);font-size:12px;font-weight:800;letter-spacing:.14em;text-transform:uppercase}.title-rule{width:92px;height:7px;margin-bottom:25px;background:var(--red)}.lede{max-width:1010px;color:#34424e;font-size:clamp(20px,2vw,30px);line-height:1.43}.finding{max-width:1240px;margin:42px 0 0;padding:22px 0 0 28px;border-left:7px solid var(--gold);font:700 clamp(25px,3vw,44px)/1.2 Georgia,serif}
    .contents{display:flex;flex-wrap:wrap;gap:9px 20px;margin-top:44px;padding-top:17px;border-top:1px solid var(--rule)}.contents a{color:var(--muted);font-size:13px;font-weight:750;text-decoration:none}.contents a:hover{color:var(--red-dark)}
    section.report{position:relative;padding:clamp(68px,9vw,128px) 0 0}.section-head{display:grid;grid-template-columns:8px minmax(0,1fr);gap:22px;align-items:start;margin-bottom:30px}.section-line{width:8px;height:100%;min-height:76px;background:var(--accent,var(--red))}.kicker{margin-bottom:9px;color:var(--accent,var(--red));font-size:12px;font-weight:850;letter-spacing:.13em;text-transform:uppercase}.section-intro{max-width:1050px;margin-bottom:0;color:var(--muted)}
    .figure{margin:0;padding:clamp(14px,1.8vw,24px);border:1px solid var(--rule);background:rgba(255,253,248,.82);box-shadow:var(--shadow)}.figure svg{display:block;width:100%;height:auto}.figure figcaption{display:flex;justify-content:space-between;gap:24px;margin-top:13px;color:var(--muted);font-size:12px;line-height:1.4}.figure-pair{display:grid;grid-template-columns:1.2fr .8fr;gap:clamp(18px,2vw,28px);align-items:stretch}.figure-pair .figure{height:100%}
    .method-table,.results-table{width:100%;margin-top:30px;border-collapse:collapse;background:rgba(255,255,255,.24)}.method-table th,.method-table td,.results-table th,.results-table td{padding:11px 13px;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top}.method-table th,.results-table th{color:var(--muted);font-size:12px;letter-spacing:.06em;text-transform:uppercase}.results-table td:not(:first-child),.results-table th:not(:first-child){text-align:right;font-variant-numeric:tabular-nums}.best{color:var(--green);font-weight:850}.warning{color:var(--red-dark);font-weight:800}
    .legend{display:flex;flex-wrap:wrap;gap:9px 22px;margin:0 0 18px 30px;color:var(--muted);font-size:13px}.legend span::before{content:"";display:inline-block;width:21px;height:4px;margin-right:7px;transform:translateY(-2px);background:var(--legend)}.legend .full::before{width:10px;height:10px;transform:translateY(1px) rotate(45deg);border:2px solid var(--ink);background:var(--white)}
    .facet-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:clamp(16px,2vw,28px)}.facet{min-width:0;padding:14px 15px 10px;border-top:5px solid var(--facet,var(--blue));background:rgba(255,253,248,.62)}.facet header{display:flex;justify-content:space-between;gap:16px;align-items:baseline;margin-bottom:6px}.facet header span{color:var(--muted);font-size:12px}.facet svg{display:block;width:100%;height:auto}
    .annotation{margin:25px 0 0;padding:16px 0 0 22px;border-left:5px solid var(--gold);color:#3c4a55}.annotation strong{color:var(--ink)}.zoom-block{margin-top:44px;padding-top:30px;border-top:1px solid var(--rule)}.zoom-head{display:grid;grid-template-columns:minmax(0,1fr) minmax(300px,.7fr);gap:28px;align-items:end;margin-bottom:20px}.zoom-head h3{margin:0;font:700 clamp(25px,2.6vw,39px)/1.08 Georgia,serif}.zoom-head p{margin:0;color:var(--muted)}
    .metric-grid{display:grid;grid-template-columns:1fr;gap:26px}.synthesis{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));margin-top:34px;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule)}.synthesis article{padding:24px clamp(12px,1.7vw,25px);border-left:1px solid var(--rule)}.synthesis article:first-child{border-left:0;padding-left:0}.synthesis article:last-child{padding-right:0}.synthesis .arm{color:var(--arm);font:700 clamp(25px,2.6vw,39px)/1 Georgia,serif}.synthesis .state{margin:8px 0 9px;font-size:12px;font-weight:850;letter-spacing:.08em;text-transform:uppercase}.synthesis p{margin:0;color:var(--muted);font-size:14px}
    .decision-banner{display:grid;grid-template-columns:minmax(220px,.7fr) minmax(0,1.5fr);gap:clamp(24px,4vw,58px);align-items:center;margin:0 0 34px;padding:clamp(22px,3vw,38px);border:1px solid var(--rule);border-left:9px solid var(--green);background:rgba(255,253,248,.86);box-shadow:var(--shadow)}.decision-banner .verdict{color:var(--green);font:700 clamp(36px,5vw,72px)/.95 Georgia,serif}.decision-banner .status{margin:10px 0 0;color:var(--muted);font-size:12px;font-weight:850;letter-spacing:.1em;text-transform:uppercase}.decision-banner p{margin:0}.badge{display:inline-block;min-width:54px;padding:4px 8px;border-radius:999px;text-align:center;font-size:11px;font-weight:900;letter-spacing:.06em;text-transform:uppercase}.badge.pass{color:#235a39;background:#dcecdf}.badge.fail{color:#872f32;background:#f3dcda}.ci-note{margin-top:14px;color:var(--muted);font-size:13px}
    .audit{display:grid;grid-template-columns:1fr 1fr;gap:clamp(24px,4vw,58px)}.audit article{padding-top:17px;border-top:7px solid var(--audit)}.audit .label{color:var(--audit);font-weight:850;letter-spacing:.11em;text-transform:uppercase}.audit h3{margin:10px 0 12px;font:700 clamp(27px,3vw,44px)/1.05 Georgia,serif}.hashes{font-size:12px;line-height:1.7;word-break:break-all}footer.page-footer{margin-top:100px;padding-top:24px;border-top:1px solid var(--rule);color:var(--muted);font-size:12px;line-height:1.55}
    .sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
    @media(max-width:1050px){.figure-pair,.facet-grid,.audit,.zoom-head,.decision-banner{grid-template-columns:1fr}.synthesis{grid-template-columns:1fr 1fr}.synthesis article{border-top:1px solid var(--rule)}.synthesis article:nth-child(odd){border-left:0}.synthesis article:first-child{border-top:0}.results-wrap{overflow-x:auto}}
    @media(max-width:650px){.hero{min-height:auto}.eyebrow,.figure figcaption{flex-direction:column}.synthesis{grid-template-columns:1fr}.synthesis article,.synthesis article:nth-child(odd){border-left:0;padding:20px 0}.method-table{display:block;overflow-x:auto}.results-table.compact-mobile,.results-table.compact-mobile tbody,.results-table.compact-mobile tr,.results-table.compact-mobile td{display:block;width:100%}.results-table.compact-mobile thead{display:none}.results-table.compact-mobile tr{padding:12px 0;border-bottom:1px solid var(--rule)}.results-table.compact-mobile td{display:flex;justify-content:space-between;gap:20px;padding:5px 0;border:0;text-align:right}.results-table.compact-mobile td::before{content:attr(data-label);color:var(--muted);font-size:10px;font-weight:800;letter-spacing:.06em;text-transform:uppercase}.results-table.compact-mobile td:first-child{display:block;color:var(--ink);font-size:17px;font-weight:850;text-align:left}.results-table.compact-mobile td:first-child::before{display:none}}
    @media print{body{background:white}.page{width:100%;padding:0 28px 40px}.hero{min-height:auto;padding-top:40px}section.report{break-inside:avoid;padding-top:50px}.contents{display:none}.figure,.facet{box-shadow:none}}
  </style>
</head>
<body>
  <main class="page">
    <header class="hero">
      <div class="eyebrow"><span>APERTUS 0.5B · DATA-ORDER FACTORIAL</span><span>COMPLETE CAMPAIGN · 05 AUG 2026</span></div>
      <div class="title-rule"></div>
      <h1>Order changes what the model remembers</h1>
      <p class="lede">Five trajectories consumed the same 80.730B-token stream once, with identical HPLT, GlossAPI/non-HPLT, foreign replay, Old-Greek replay, packing, optimizer, and WSD-10 schedule. Only the temporal order of the two Modern-Greek pools changed.</p>
      <p class="finding">D0 Mixed is the observed all-round leader: it passes the source-retention safety screen, leads the predeclared endpoint hierarchy, and has lower clean GreekMMLU choice NLL than every ordered arm under paired question bootstrap. It is not yet a statistically resolved winner because the frozen source-loss outputs lack document-cluster rows.</p>
      <nav class="contents" aria-label="Report sections"><a href="#factor">Mix geometry</a><a href="#learning">Greek learning</a><a href="#forgetting">Retention and forgetting</a><a href="#greekmmlu">GreekMMLU</a><a href="#synthesis">Applied gate</a><a href="#audit">Evidence</a></nav>
    </header>

    <section class="report" id="factor" style="--accent:var(--blue)">
      <div class="section-head"><span class="section-line"></span><div><div class="kicker">01 · Intervention geometry</div><h2>Exactly one factor: when HPLT and GlossAPI appear</h2><p class="section-intro">The left plot shows GlossAPI/non-HPLT as a share of all loss-active tokens. HPLT always fills the remainder of the fixed 79% Modern-Greek stream; foreign replay remains 20% and Old-Greek replay 1% at every point. The right plot shows the one LR schedule shared by all five arms.</p></div></div>
      <div class="figure-pair">
        <figure class="figure"><svg id="mix-chart" role="img" aria-labelledby="mix-title mix-desc"></svg><figcaption><span>HPLT share = 79% − plotted GlossAPI share.</span><span>Gradual curves use 128 quota-corrected windows.</span></figcaption></figure>
        <figure class="figure"><svg id="lr-chart" role="img" aria-labelledby="lr-title lr-desc"></svg><figcaption><span>Warmup 800 · cooldown 30,796.</span><span>Peak 1.5e−4 · floor 1.5e−5.</span></figcaption></figure>
      </div>
      <table class="method-table" aria-label="Fixed experiment contract"><thead><tr><th>Same data</th><th>Same training</th><th>Same evaluation</th><th>Only treatment</th></tr></thead><tbody><tr><td>63.777B Modern-Greek + 16.146B foreign replay + 0.807B Old-Greek replay per arm; each frozen sequence identity consumed once.</td><td>Apertus v1.1 0.5B, extended 148,992-token tokenizer, tied token-distillation initialization, AdEMAMix, Goldfish objective, 4,096 context, 2.097M global tokens/update.</td><td>83 matched checkpoints per arm; 13 fixed source panels; decontaminated native GreekMMLU; authoritative checkpoint scoring in float32.</td><td>D0 mixed; D1 hard H→G; D2 hard G→H; D3 gradual H→G; D4 gradual G→H. No checkpoint averaging.</td></tr></tbody></table>
    </section>

    <section class="report" id="learning" style="--accent:var(--teal)">
      <div class="section-head"><span class="section-line"></span><div><div class="kicker">02 · Modern-Greek learning</div><h2>Complete source-conditioned loss trajectories</h2><p class="section-intro">Every raw checkpoint measurement is shown from initialization through update 38,496. Lines are the frequent fast panel; open diamonds are the larger final endpoint evaluation. Lower bits per UTF-8 byte (BPB) is better. Vertical rules mark the two hard switches and the shared LR cooldown.</p></div></div>
      <div class="legend" aria-label="Data order legend"><span style="--legend:#102131">D0 Mixed</span><span style="--legend:#d6534d">D1 Hard H→G</span><span style="--legend:#356f9f">D2 Hard G→H</span><span style="--legend:#277e74">D3 Gradual H→G</span><span style="--legend:#c18b32">D4 Gradual G→H</span><span class="full">Full endpoint</span></div>
      <div class="facet-grid" id="learning-grid"></div>
      <p class="annotation"><strong>Recency is large and source-specific.</strong> D1 finishes best on every curated GlossAPI family but pays +0.0473 BPB on HPLT and +0.0184 on neutral Greek versus D0. D2 finishes best on HPLT but gives back +0.0823 BPB on aggregate non-HPLT and +0.0855 on historical polytonic.</p>
      <div class="zoom-block"><div class="zoom-head"><h3>Endpoint effects magnified · Δ BPB versus D0</h3><p>This derived endpoint view uses the same full-evaluation values as the open diamonds above. Teal means lower loss than the stationary mixture; red means higher. It makes small separations readable without replacing the full trajectories.</p></div><figure class="figure"><svg id="greek-heatmap" role="img" aria-labelledby="greek-heat-title greek-heat-desc"></svg><figcaption><span>Columns are independent source-conditioned panels.</span><span>Observed deltas only; paired uncertainty is not yet applied.</span></figcaption></figure></div>
      <div class="zoom-block"><div class="zoom-head"><h3>The adaptation–retention frontier</h3><p>Each endpoint is positioned by broad-HPLT loss and the macro-average of four curated GlossAPI families. The desirable direction is down and left. Mirror schedules trace the recency tradeoff; D0 and D3 are the observed balanced endpoints.</p></div><figure class="figure"><svg id="tradeoff-chart" role="img" aria-labelledby="tradeoff-title tradeoff-desc"></svg><figcaption><span>GlossAPI macro: non-HPLT, Greek PhD, OpenArchives, historical polytonic.</span><span>Axes use full endpoint BPB.</span></figcaption></figure></div>
    </section>

    <section class="report" id="forgetting" style="--accent:var(--red-dark)">
      <div class="section-head"><span class="section-line"></span><div><div class="kicker">03 · Retention and forgetting</div><h2>Replay sources over the entire run</h2><p class="section-intro">Absolute BPB remains primary. “Forgetting” is the fast-panel loss at the last checkpoint minus that arm’s best fast-panel loss anywhere in training—not merely movement during the cooldown. All seven requested replay sources are shown.</p></div></div>
      <div class="legend" aria-label="Data order legend"><span style="--legend:#102131">D0 Mixed</span><span style="--legend:#d6534d">D1 Hard H→G</span><span style="--legend:#356f9f">D2 Hard G→H</span><span style="--legend:#277e74">D3 Gradual H→G</span><span style="--legend:#c18b32">D4 Gradual G→H</span><span class="full">Full endpoint</span></div>
      <div class="facet-grid" id="retention-grid"></div>
      <div class="zoom-block"><div class="zoom-head"><h3>Small endpoint gaps, placed on a zero-centered scale</h3><p>The complete trajectories above show when forgetting occurs. This secondary view shows full endpoint BPB minus D0 for each replay panel so the much smaller between-arm differences remain visible.</p></div><figure class="figure"><svg id="retention-delta-chart" role="img" aria-labelledby="retention-delta-title retention-delta-desc"></svg><figcaption><span>Negative is lower loss than D0; positive is higher.</span><span>D3 nearly matches D0 and is slightly lower on four foreign panels.</span></figcaption></figure></div>
      <p class="annotation"><strong>Ordering influences replay less than it influences Greek source loss, but not uniformly.</strong> D0 has the lowest observed replay-forgetting macro-average (0.09475 BPB); D3 is essentially adjacent (0.09487). Hard or reverse HPLT-ending schedules show larger observed replay forgetting, especially in German, English, math, and Chinese.</p>
    </section>

    <section class="report" id="greekmmlu" style="--accent:var(--purple)">
      <div class="section-head"><span class="section-line"></span><div><div class="kicker">04 · Native Greek GreekMMLU</div><h2>All 83 checkpoint evaluations—not only the endpoint</h2><p class="section-intro">The decontaminated native-Greek subset contains 16,159 questions. Accuracy is familiar but discontinuous; choice NLL and correct-answer BPB remain informative when answer ranking does not change. Raw points are preserved. No smoothing or checkpoint averaging is used.</p></div></div>
      <div class="legend" aria-label="Data order legend"><span style="--legend:#102131">D0 Mixed</span><span style="--legend:#d6534d">D1 Hard H→G</span><span style="--legend:#356f9f">D2 Hard G→H</span><span style="--legend:#277e74">D3 Gradual H→G</span><span style="--legend:#c18b32">D4 Gradual G→H</span></div>
      <div class="metric-grid"><figure class="figure"><svg id="gm-accuracy" role="img" aria-labelledby="gm-accuracy-title gm-accuracy-desc"></svg><figcaption><span>Clean zero-shot accuracy · higher is better.</span><span>D3 has the highest observed final accuracy.</span></figcaption></figure><figure class="figure"><svg id="gm-nll" role="img" aria-labelledby="gm-nll-title gm-nll-desc"></svg><figcaption><span>Multiple-choice cross-entropy · lower is better.</span><span>D0 has the lowest observed final choice NLL.</span></figcaption></figure><figure class="figure"><svg id="gm-bpb" role="img" aria-labelledby="gm-bpb-title gm-bpb-desc"></svg><figcaption><span>Correct-answer continuation BPB · lower is better.</span><span>D2 has the lowest observed final answer BPB.</span></figcaption></figure></div>
      <div class="results-wrap"><table class="results-table compact-mobile" id="greekmmlu-table" aria-label="GreekMMLU endpoint and best checkpoint metrics"><thead><tr><th>Arm</th><th>Final accuracy</th><th>Final choice NLL</th><th>Final answer BPB</th><th>Best accuracy checkpoint</th><th>Best NLL checkpoint</th></tr></thead><tbody></tbody></table></div>
      <p class="annotation"><strong>The metrics disagree about the endpoint.</strong> D3 leads clean accuracy at 42.37%; D0 leads final choice NLL at 1.2869; D2 leads correct-answer BPB at 0.1856. Several arms reached better choice NLL earlier and then regressed, so checkpoint trajectories matter. These are descriptive one-seed differences, not a declared winner.</p>
    </section>

    <section class="report" id="synthesis" style="--accent:var(--green)">
      <div class="section-head"><span class="section-line"></span><div><div class="kicker">05 · Applied decision gate</div><h2>D0 leads the screen; D3 is the confirmatory challenger</h2><p class="section-intro">The frozen 5% common-stability retention margin is reused here transparently as a safety sensitivity—not relabelled as a separately frozen final-winner margin. The primary source hierarchy is applied only among arms that pass it. GreekMMLU intervals use 10,000 paired question bootstraps.</p></div></div>
      <div class="decision-banner"><div><div class="verdict">D0 Mixed</div><div class="status">Observed leader · winner not formally declared</div></div><p>Passes the 5% source-retention safety screen; ranks first among passers on neutral external Greek, then balanced HPLT/GlossAPI; and beats every alternative on clean GreekMMLU choice NLL with paired 95% intervals excluding zero. D3 is the closest balanced schedule to carry into a seed-level confirmation.</p></div>
      <div class="results-wrap"><table class="results-table compact-mobile" id="gate-table" aria-label="Source retention safety gate"><thead><tr><th>Arm</th><th>Safety gate</th><th>Worst retained panel</th><th>Worst relative change vs D0</th><th>Point hierarchy rank</th></tr></thead><tbody></tbody></table></div>
      <div class="zoom-block"><div class="zoom-head"><h3>GreekMMLU choice NLL favors D0 consistently</h3><p>Each interval is candidate minus D0 on the same 16,159 clean questions. Positive values mean the candidate has higher—and therefore worse—choice NLL.</p></div><figure class="figure"><svg id="nll-ci-chart" role="img" aria-labelledby="nll-ci-title nll-ci-desc"></svg><figcaption><span>10,000 paired bootstrap samples · percentile 95% interval.</span><span>All four intervals exclude zero in D0’s favor.</span></figcaption></figure></div>
      <div class="results-wrap"><table class="results-table compact-mobile" id="uncertainty-table" aria-label="Paired clean GreekMMLU uncertainty versus D0"><thead><tr><th>Arm vs D0</th><th>Accuracy delta [95% CI]</th><th>Choice NLL delta [95% CI]</th><th>Answer BPB delta [95% CI]</th><th>McNemar p</th></tr></thead><tbody></tbody></table></div>
      <p class="ci-note">Accuracy does not separate D0 from D1, D3, or D4; D2 is lower. D3’s observed +0.24 percentage-point accuracy lead has a paired 95% interval from −0.22 to +0.71 points. Continuous choice NLL is the cleaner signal here.</p>
      <div class="results-wrap"><table class="results-table compact-mobile" id="synthesis-table" aria-label="Dataset-order endpoint synthesis"><thead><tr><th>Arm</th><th>HPLT BPB</th><th>GlossAPI macro</th><th>Balanced Greek</th><th>Neutral Greek</th><th>Replay forgetting</th><th>GreekMMLU acc.</th><th>GreekMMLU NLL</th></tr></thead><tbody></tbody></table></div>
      <div class="synthesis">
        <article style="--arm:#102131"><div class="arm">D0</div><div class="state">Stationary mixed</div><p>Observed best balanced-Greek aggregate, best replay-forgetting macro, and best final GreekMMLU choice NLL.</p></article>
        <article style="--arm:#d6534d"><div class="arm">D1</div><div class="state">Hard H→G</div><p>Strongest curated endpoint, but largest broad-HPLT and neutral-Greek recency cost.</p></article>
        <article style="--arm:#356f9f"><div class="arm">D2</div><div class="state">Fails safety screen</div><p>Strongest HPLT endpoint and answer BPB, but historical-polytonic loss is 13.1% above D0.</p></article>
        <article style="--arm:#277e74"><div class="arm">D3</div><div class="state">Gradual H→G</div><p>Near-D0 balanced loss, second-best curated macro, and highest observed final GreekMMLU accuracy.</p></article>
        <article style="--arm:#c18b32"><div class="arm">D4</div><div class="state">Fails safety screen</div><p>Near-D2 HPLT and neutral endpoint, but historical-polytonic loss is 7.17% above D0.</p></article>
      </div>
      <p class="annotation"><strong>Decision status: observed leader, not declared winner.</strong> D0 is first under the stated hierarchy after the source safety screen, and its GreekMMLU choice-NLL advantage is paired-question robust. Formal winner selection remains blocked only where the frozen evidence is genuinely insufficient: document-cluster intervals for primary source BPB and a pre-endpoint numeric margin for the general benchmark suite.</p>
    </section>

    <section class="report" id="audit" style="--accent:var(--gold)">
      <div class="section-head"><span class="section-line"></span><div><div class="kicker">06 · Evidence and limits</div><h2>Frozen inputs, exact bindings, no hidden variants</h2></div></div>
      <div class="audit"><article style="--audit:var(--teal)"><div class="label">Now applied</div><h3>Safety gate + paired questions</h3><p>The analysis reproduces every clean GreekMMLU endpoint from the saved per-question predictions, then performs 10,000 paired bootstraps and exact McNemar tests. The 5% source-retention safety sensitivity is evaluated on eight named panels.</p><p>Run: <code>mini_cpt5_20260803T074854Z</code></p></article><article style="--audit:var(--red)"><div class="label">Evidence boundary</div><h3>Aggregate BPB is not a bootstrap sample</h3><p>The validation receipts save one aggregate per panel, not document-cluster numerators and byte denominators. Checkpoint points are temporally dependent and are not substituted for document clusters. General benchmark scores are reported descriptively because no numeric final noninferiority margin was frozen.</p><p>Evaluation authority: <code>fp32_v1 / float32</code>.</p></article></div>
      <table class="method-table"><thead><tr><th>Frozen evidence</th><th>SHA-256</th></tr></thead><tbody><tr><td><a href="data/dataset_order_20260805/validation_trajectory.json">validation_trajectory.json</a></td><td class="hashes">bfb7c7073adee541f4e3ed74548b343cf58d60b93249a54f2a900937d5972298</td></tr><tr><td><a href="data/dataset_order_20260805/greekmmlu_trajectory.json">greekmmlu_trajectory.json</a></td><td class="hashes">74e56f40ab05e982aac7dc6cebcae1422e3fa97db15525b035f92581d26211cb</td></tr><tr><td><a href="data/dataset_order_20260805/full_endpoint_validation_receipt.json">full_endpoint_validation_receipt.json</a></td><td class="hashes">6e49e02e67e0baf75acd965e49445fd5bf1e657552d9bd3cfec5187953311146</td></tr><tr><td><a href="data/dataset_order_20260805/core_campaign_summary.json">core_campaign_summary.json</a></td><td class="hashes">eb8a2db1949bbc2be3108c273d947f58ec086e88b13fd7ed9e7ed7338bf8591e</td></tr><tr><td><a href="data/dataset_order_20260805/dataset_order_selection_analysis.json">dataset_order_selection_analysis.json</a></td><td class="hashes">581f59294fd4f728bd1ff318dfa2c1d4fb52914b0d49e6001b83ae8a6e677b19</td></tr></tbody></table>
    </section>

    <footer class="page-footer">Source contract: <a href="../configs/experiment_matrix.json">experiment_matrix.json</a> · Earlier visual reference: <a href="LR_FLOOR_EXPERIMENT_RESULTS_20260802.html">LR floor report</a><br>Authoritative remote run root: <code>/capstor/scratch/cscs/fffoivos/runs/06_dataset_scheduling_experiments/mini_cpt5_20260803T074854Z</code></footer>
  </main>

  <script id="report-data" type="application/json">__REPORT_DATA__</script>
  <script>
    "use strict";
    const NS="http://www.w3.org/2000/svg";
    const C={ink:"#102131",muted:"#65717b",rule:"#d6cfc2",paper:"#fffdf8",red:"#d6534d",blue:"#356f9f",teal:"#277e74",gold:"#c18b32",purple:"#775ba6",green:"#3f7f57"};
    const A=[{id:"D0_mixed",short:"D0",name:"Mixed",color:C.ink,dash:""},{id:"D1_hard_h_to_g",short:"D1",name:"Hard H→G",color:C.red,dash:""},{id:"D2_hard_g_to_h",short:"D2",name:"Hard G→H",color:C.blue,dash:""},{id:"D3_gradual_h_to_g",short:"D3",name:"Gradual H→G",color:C.teal,dash:"8 4"},{id:"D4_gradual_g_to_h",short:"D4",name:"Gradual G→H",color:C.gold,dash:"8 4"}];
    const D=JSON.parse(document.getElementById("report-data").textContent);
    const N={hplt:"HPLT · broad Greek",non_hplt:"GlossAPI · aggregate non-HPLT",greek_phd:"Greek PhD",openarchives:"OpenArchives",historical_polytonic:"Historical polytonic",neutral_external_modern_greek:"Neutral external Modern Greek",old_greek:"Old-Greek replay",english:"English replay",code:"Code replay",math:"Math replay",de:"German replay",ru:"Russian replay",zh:"Chinese replay"};
    const S={hplt:"HPLT",non_hplt:"non-HPLT",greek_phd:"Greek PhD",openarchives:"OpenArchives",historical_polytonic:"Polytonic",neutral_external_modern_greek:"Neutral",old_greek:"Old Greek",english:"English",code:"Code",math:"Math",de:"German",ru:"Russian",zh:"Chinese"};
    const learning=["hplt","non_hplt","greek_phd","openarchives","historical_polytonic","neutral_external_modern_greek"];
    const retention=["old_greek","english","code","math","de","ru","zh"];
    const finalIter=D.meta.final_iteration;

    function el(name,attrs={},text=null){const n=document.createElementNS(NS,name);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text!==null)n.textContent=text;return n}
    function linePath(points,x,y){return points.map((p,i)=>`${i?"L":"M"}${x(p[0]).toFixed(2)},${y(p[1]).toFixed(2)}`).join(" ")}
    function fmt(v,d=4){return `${v<0?"−":"+"}${Math.abs(v).toFixed(d)}`}
    function sci(v){return v.toExponential(1).replace(".0e","e").replace("e-0","e−")}
    function paddedDomain(values,floor=.002){const lo=Math.min(...values),hi=Math.max(...values),p=Math.max((hi-lo)*.065,floor);return[lo-p,hi+p]}
    function endpointLabels(items,top,bottom,gap=13){items.sort((a,b)=>a.py-b.py);items.forEach((d,i)=>d.ly=i?Math.max(d.py,items[i-1].ly+gap):d.py);let over=items.at(-1).ly-bottom;if(over>0)items.forEach(d=>d.ly-=over);let under=top-items[0].ly;if(under>0)items.forEach(d=>d.ly+=under);return items}
    function addPhaseRules(svg,x,y0,y1,labels=true){[[D.meta.hard_g_to_h,C.blue,"D2 switch"],[D.meta.hard_h_to_g,C.red,"D1 switch"],[D.meta.cooldown_start,C.purple,"LR decay"]].forEach(([it,color,label],i)=>{svg.append(el("line",{x1:x(it),x2:x(it),y1:y0,y2:y1,stroke:color,"stroke-width":1.25,"stroke-dasharray":i===2?"2 4":"5 4",opacity:.75}));if(labels)svg.append(el("text",{x:x(it)+4,y:y0+12+i*12,fill:color,"font-size":9,"font-weight":750},label))})}

    function drawMix(){const svg=document.getElementById("mix-chart"),W=840,H=500,m={l:74,r:90,t:55,b:64},x=u=>m.l+u*(W-m.l-m.r),y=v=>H-m.b-v/.82*(H-m.t-m.b);svg.setAttribute("viewBox",`0 0 ${W} ${H}`);svg.append(el("title",{id:"mix-title"},"GlossAPI share across five temporal data orders"));svg.append(el("desc",{id:"mix-desc"},"The stationary arm keeps GlossAPI at 24.45 percent of all tokens. Hard and gradual schedules move between zero and seventy-nine percent while replay remains fixed."));[0,.2,.4,.6,.79].forEach(t=>{svg.append(el("line",{x1:m.l,x2:W-m.r,y1:y(t),y2:y(t),stroke:C.rule}));svg.append(el("text",{x:m.l-10,y:y(t)+4,fill:C.muted,"font-size":12,"text-anchor":"end"},`${Math.round(t*100)}%`))});[0,.25,.5,.75,1].forEach(t=>svg.append(el("text",{x:x(t),y:H-m.b+27,fill:C.muted,"font-size":12,"text-anchor":"middle"},`${Math.round(t*100)}%`)));
      const qH=D.meta.q_h,qG=D.meta.q_g,a=D.meta.gradual_exponent;const value=(id,u)=>id==="D0_mixed"?.79*qG:id==="D1_hard_h_to_g"?(u<qH?0:.79):id==="D2_hard_g_to_h"?(u<qG?.79:0):id==="D3_gradual_h_to_g"?.79*Math.pow(u,a):.79*Math.pow(1-u,a);
      A.forEach(arm=>{const pts=Array.from({length:257},(_,i)=>{const u=i/256;return[u,value(arm.id,u)]});svg.append(el("path",{d:linePath(pts,x,y),fill:"none",stroke:arm.color,"stroke-width":arm.id==="D0_mixed"?4:3.3,"stroke-dasharray":arm.dash,"stroke-linejoin":"round"}));});
      const labs=[{a:A[0],u:.48},{a:A[1],u:.78},{a:A[2],u:.73},{a:A[3],u:.63},{a:A[4],u:.63}];labs.forEach(d=>{const v=value(d.a.id,d.u);svg.append(el("rect",{x:x(d.u)-3,y:y(v)-12,width:82,height:18,rx:3,fill:C.paper,opacity:.88}));svg.append(el("text",{x:x(d.u)+2,y:y(v)+1,fill:d.a.color,"font-size":11,"font-weight":850},`${d.a.short} ${d.a.name}`))});
      [qG,qH].forEach((u,i)=>{svg.append(el("line",{x1:x(u),x2:x(u),y1:m.t,y2:H-m.b,stroke:i?C.red:C.blue,"stroke-width":1.4,"stroke-dasharray":"5 4",opacity:.65}));svg.append(el("text",{x:x(u),y:m.t-12,fill:i?C.red:C.blue,"font-size":10,"font-weight":800,"text-anchor":"middle"},i?"D1 · 69.06%":"D2 · 30.94%"))});
      svg.append(el("line",{x1:m.l,x2:W-m.r,y1:H-m.b,y2:H-m.b,stroke:C.ink}));svg.append(el("line",{x1:m.l,x2:m.l,y1:m.t,y2:H-m.b,stroke:C.ink}));svg.append(el("text",{x:(m.l+W-m.r)/2,y:H-15,fill:C.muted,"font-size":13,"font-weight":700,"text-anchor":"middle"},"Training progress"));svg.append(el("text",{x:19,y:(m.t+H-m.b)/2,fill:C.muted,"font-size":13,"font-weight":700,"text-anchor":"middle",transform:`rotate(-90 19 ${(m.t+H-m.b)/2})`},"GlossAPI share of all tokens"));}

    function oneMinusSqrt(z,start,end){return end+(start-end)*(1-Math.sqrt(Math.max(0,Math.min(1,z))))}
    function lrAt(it){const peak=D.meta.peak_lr,min=D.meta.minimum_lr;if(it<=D.meta.warmup_end)return min+(peak-min)*it/D.meta.warmup_end;if(it<=D.meta.cooldown_start)return peak;return oneMinusSqrt((it-D.meta.cooldown_start)/(finalIter-D.meta.cooldown_start),peak,min)}
    function drawLR(){const svg=document.getElementById("lr-chart"),W=590,H=500,m={l:82,r:28,t:55,b:64},x=v=>m.l+v/finalIter*(W-m.l-m.r),y=v=>H-m.b-v/(D.meta.peak_lr*1.08)*(H-m.t-m.b);svg.setAttribute("viewBox",`0 0 ${W} ${H}`);svg.append(el("title",{id:"lr-title"},"Fixed WSD-10 learning-rate schedule"));svg.append(el("desc",{id:"lr-desc"},"All five arms warm from fifteen micro to one hundred fifty micro over eight hundred updates, remain stable, and decay to ten percent of peak after update 30796."));[[0,D.meta.warmup_end,C.gold,.12,"WARMUP"],[D.meta.warmup_end,D.meta.cooldown_start,C.teal,.06,"STABLE"],[D.meta.cooldown_start,finalIter,C.purple,.1,"COOLDOWN"]].forEach(([a,b,c,o,l])=>{svg.append(el("rect",{x:x(a),y:m.t,width:x(b)-x(a),height:H-m.t-m.b,fill:c,opacity:o}));svg.append(el("text",{x:(x(a)+x(b))/2,y:m.t+18,fill:C.muted,"font-size":10,"font-weight":800,"text-anchor":"middle"},l))});[0,D.meta.minimum_lr,D.meta.peak_lr/2,D.meta.peak_lr].forEach(t=>{svg.append(el("line",{x1:m.l,x2:W-m.r,y1:y(t),y2:y(t),stroke:C.rule}));svg.append(el("text",{x:m.l-9,y:y(t)+4,fill:C.muted,"font-size":11,"text-anchor":"end"},t?sci(t):"0"))});[0,800,11912,26584,30796,38496].forEach(t=>svg.append(el("text",{x:x(t),y:H-m.b+25,fill:C.muted,"font-size":9,"text-anchor":"middle"},t.toLocaleString())));const pts=Array.from({length:320},(_,i)=>{const it=finalIter*i/319;return[it,lrAt(it)]});svg.append(el("path",{d:linePath(pts,x,y),fill:"none",stroke:C.ink,"stroke-width":4,"stroke-linejoin":"round","stroke-linecap":"round"}));svg.append(el("circle",{cx:x(finalIter),cy:y(D.meta.minimum_lr),r:4,fill:C.red}));svg.append(el("line",{x1:m.l,x2:W-m.r,y1:H-m.b,y2:H-m.b,stroke:C.ink}));svg.append(el("line",{x1:m.l,x2:m.l,y1:m.t,y2:H-m.b,stroke:C.ink}));svg.append(el("text",{x:(m.l+W-m.r)/2,y:H-15,fill:C.muted,"font-size":13,"font-weight":700,"text-anchor":"middle"},"Training update"));}

    function lossFacet(panel,kind){
      const article=document.createElement("article"); article.className="facet"; article.style.setProperty("--facet",kind==="learning"?C.teal:C.red);
      const header=document.createElement("header"),title=document.createElement("h3"),meta=document.createElement("span"),svg=document.createElementNS(NS,"svg");
      title.textContent=N[panel]; meta.textContent="absolute BPB · lower is better"; header.append(title,meta); article.append(header,svg);
      const W=700,H=345,m={l:61,r:113,t:24,b:48},all=A.flatMap(a=>D.validation[a.id][panel].map(p=>p[1]).concat([D.full[a.id][panel]]));
      const [yMin,yMax]=paddedDomain(all,(panel==="zh"||panel==="de")?0.006:0.002),x=v=>m.l+v/finalIter*(W-m.l-m.r),y=v=>H-m.b-(v-yMin)/(yMax-yMin)*(H-m.t-m.b);
      svg.setAttribute("viewBox",`0 0 ${W} ${H}`); svg.setAttribute("role","img"); svg.setAttribute("aria-label",`${N[panel]} validation BPB for all five data orders from initialization through update ${finalIter}`);
      for(let j=0;j<4;j++){const t=yMin+(yMax-yMin)*j/3;svg.append(el("line",{x1:m.l,x2:W-m.r,y1:y(t),y2:y(t),stroke:C.rule}));svg.append(el("text",{x:m.l-8,y:y(t)+4,fill:C.muted,"font-size":10,"text-anchor":"end"},t.toFixed(t<1?3:2)))}
      [0,11912,19456,26584,30796,38496].forEach(t=>svg.append(el("text",{x:x(t),y:H-m.b+24,fill:C.muted,"font-size":9,"text-anchor":"middle"},t===0?"0":`${(t/1000).toFixed(t%1000?1:0)}k`)));
      addPhaseRules(svg,x,m.t,H-m.b,false); const endpoints=[];
      A.forEach(a=>{
        const pts=D.validation[a.id][panel];
        svg.append(el("path",{d:linePath(pts,x,y),fill:"none",stroke:a.color,"stroke-width":a.id==="D0_mixed"?3.2:2.55,"stroke-dasharray":a.dash,"stroke-linejoin":"round","stroke-linecap":"round"}));
        pts.forEach(p=>svg.append(el("circle",{cx:x(p[0]),cy:y(p[1]),r:1.15,fill:a.color,opacity:.55})));
        const v=D.full[a.id][panel],px=x(finalIter),py=y(v);
        svg.append(el("polygon",{points:`${px},${py-5} ${px+5},${py} ${px},${py+5} ${px-5},${py}`,fill:C.paper,stroke:a.color,"stroke-width":2})); endpoints.push({a,v,py});
      });
      endpointLabels(endpoints,m.t+3,H-m.b-3,13).forEach(d=>{const lx=W-m.r+9,px=x(finalIter);svg.append(el("line",{x1:px+6,x2:lx-3,y1:d.py,y2:d.ly,stroke:d.a.color,"stroke-width":1}));svg.append(el("text",{x:lx,y:d.ly+3,fill:d.a.color,"font-size":9.5,"font-weight":850},`${d.a.short} ${d.v.toFixed(4)}`))});
      svg.append(el("line",{x1:m.l,x2:W-m.r,y1:H-m.b,y2:H-m.b,stroke:C.ink})); svg.append(el("line",{x1:m.l,x2:m.l,y1:m.t,y2:H-m.b,stroke:C.ink})); return article;
    }

    function drawHeatmap(){
      const svg=document.getElementById("greek-heatmap"),panels=learning,arms=A.slice(1),W=1280,H=520,m={l:190,r:30,t:130,b:55};
      const cw=(W-m.l-m.r)/panels.length,rh=(H-m.t-m.b)/arms.length,vals=arms.flatMap(a=>panels.map(p=>D.full[a.id][p]-D.full.D0_mixed[p])),lim=Math.max(...vals.map(Math.abs));
      svg.setAttribute("viewBox",`0 0 ${W} ${H}`); svg.append(el("title",{id:"greek-heat-title"},"Endpoint Greek BPB differences versus stationary mixing")); svg.append(el("desc",{id:"greek-heat-desc"},"Four ordered schedules are compared with D0 across HPLT, four curated Greek panels, and neutral Greek. Teal cells are improvements and red cells are regressions."));
      panels.forEach((p,j)=>{const tx=m.l+(j+.5)*cw,ty=m.t-14;svg.append(el("text",{x:tx,y:ty,fill:C.ink,"font-size":12,"font-weight":800,"text-anchor":"end",transform:`rotate(-38 ${tx} ${ty})`},S[p]))});
      arms.forEach((a,i)=>{
        const cy=m.t+(i+.5)*rh; svg.append(el("text",{x:m.l-18,y:cy+5,fill:a.color,"font-size":14,"font-weight":850,"text-anchor":"end"},`${a.short} · ${a.name}`));
        panels.forEach((p,j)=>{const v=D.full[a.id][p]-D.full.D0_mixed[p],strength=Math.abs(v)/lim,color=v<=0?C.teal:C.red;svg.append(el("rect",{x:m.l+j*cw+2,y:m.t+i*rh+2,width:cw-4,height:rh-4,rx:3,fill:color,opacity:.12+.78*strength}));svg.append(el("text",{x:m.l+(j+.5)*cw,y:cy+5,fill:strength>.58?C.paper:C.ink,"font-size":12,"font-weight":850,"text-anchor":"middle"},fmt(v)))})
      });
      svg.append(el("text",{x:(m.l+W-m.r)/2,y:H-15,fill:C.muted,"font-size":13,"font-weight":700,"text-anchor":"middle"},"Full endpoint BPB difference versus D0 · lower is better"));
    }

    function drawTradeoff(){const svg=document.getElementById("tradeoff-chart"),W=1280,H=610,m={l:105,r:90,t:55,b:82},xs=A.map(a=>D.derived[a.id].hplt),ys=A.map(a=>D.derived[a.id].gloss_macro),[x0,x1]=paddedDomain(xs,.004),[y0,y1]=paddedDomain(ys,.006),x=v=>m.l+(v-x0)/(x1-x0)*(W-m.l-m.r),y=v=>H-m.b-(v-y0)/(y1-y0)*(H-m.t-m.b);svg.setAttribute("viewBox",`0 0 ${W} ${H}`);svg.append(el("title",{id:"tradeoff-title"},"HPLT versus curated GlossAPI endpoint BPB"));svg.append(el("desc",{id:"tradeoff-desc"},"The hard schedules occupy opposite ends of the broad versus curated Greek tradeoff. D0 and D3 sit near the lower-left balance point."));for(let j=0;j<5;j++){const tx=x0+(x1-x0)*j/4,ty=y0+(y1-y0)*j/4;svg.append(el("line",{x1:x(tx),x2:x(tx),y1:m.t,y2:H-m.b,stroke:C.rule}));svg.append(el("line",{x1:m.l,x2:W-m.r,y1:y(ty),y2:y(ty),stroke:C.rule}));svg.append(el("text",{x:x(tx),y:H-m.b+28,fill:C.muted,"font-size":12,"text-anchor":"middle"},tx.toFixed(3)));svg.append(el("text",{x:m.l-12,y:y(ty)+4,fill:C.muted,"font-size":12,"text-anchor":"end"},ty.toFixed(3)))}[["D1_hard_h_to_g","D2_hard_g_to_h"],["D3_gradual_h_to_g","D4_gradual_g_to_h"]].forEach(pair=>{const p=pair.map(id=>D.derived[id]);svg.append(el("line",{x1:x(p[0].hplt),y1:y(p[0].gloss_macro),x2:x(p[1].hplt),y2:y(p[1].gloss_macro),stroke:C.rule,"stroke-width":2,"stroke-dasharray":"5 5"}))});const offsets={D0_mixed:[-16,-18],D1_hard_h_to_g:[13,-12],D2_hard_g_to_h:[13,-12],D3_gradual_h_to_g:[13,19],D4_gradual_g_to_h:[13,19]};A.forEach(a=>{const d=D.derived[a.id],px=x(d.hplt),py=y(d.gloss_macro),[dx,dy]=offsets[a.id];svg.append(el("circle",{cx:px,cy:py,r:a.id==="D0_mixed"?10:8,fill:a.color,stroke:C.paper,"stroke-width":3}));svg.append(el("text",{x:px+dx,y:py+dy,fill:a.color,"font-size":14,"font-weight":900,"text-anchor":dx<0?"end":"start"},`${a.short} · ${d.hplt.toFixed(3)}, ${d.gloss_macro.toFixed(3)}`))});svg.append(el("text",{x:(m.l+W-m.r)/2,y:H-18,fill:C.muted,"font-size":14,"font-weight":700,"text-anchor":"middle"},"HPLT full endpoint BPB · lower ←"));svg.append(el("text",{x:24,y:(m.t+H-m.b)/2,fill:C.muted,"font-size":14,"font-weight":700,"text-anchor":"middle",transform:`rotate(-90 24 ${(m.t+H-m.b)/2})`},"GlossAPI macro BPB · lower ↓"));svg.append(el("text",{x:m.l+12,y:H-m.b-16,fill:C.green,"font-size":12,"font-weight":850},"DESIRABLE DIRECTION ↙"))}

    function drawRetentionDelta(){const svg=document.getElementById("retention-delta-chart"),arms=A.slice(1),W=1280,H=650,m={l:190,r:90,t:65,b:78},vals=arms.flatMap(a=>retention.map(p=>D.full[a.id][p]-D.full.D0_mixed[p])),lim=Math.max(...vals.map(Math.abs))*1.15,x=v=>m.l+(v+lim)/(2*lim)*(W-m.l-m.r),rh=(H-m.t-m.b)/retention.length;svg.setAttribute("viewBox",`0 0 ${W} ${H}`);svg.append(el("title",{id:"retention-delta-title"},"Replay endpoint BPB differences versus D0"));svg.append(el("desc",{id:"retention-delta-desc"},"Dots show each ordered schedule's endpoint loss difference from stationary mixing for Old Greek, English, code, math, German, Russian, and Chinese."));for(let j=0;j<=6;j++){const t=-lim+2*lim*j/6;svg.append(el("line",{x1:x(t),x2:x(t),y1:m.t-10,y2:H-m.b,stroke:Math.abs(t)<1e-10?C.ink:C.rule,"stroke-width":Math.abs(t)<1e-10?2:1}));svg.append(el("text",{x:x(t),y:H-m.b+29,fill:C.muted,"font-size":12,"text-anchor":"middle"},Math.abs(t)<1e-10?"D0":fmt(t,3)))}retention.forEach((p,i)=>{const cy=m.t+(i+.5)*rh;svg.append(el("line",{x1:m.l,x2:W-m.r,y1:cy+rh*.48,y2:cy+rh*.48,stroke:C.rule,opacity:.55}));svg.append(el("text",{x:m.l-18,y:cy+5,fill:C.ink,"font-size":14,"font-weight":800,"text-anchor":"end"},S[p]));arms.forEach((a,j)=>{const v=D.full[a.id][p]-D.full.D0_mixed[p],py=cy+(j-1.5)*10;svg.append(el("line",{x1:x(0),x2:x(v),y1:py,y2:py,stroke:a.color,"stroke-width":1.5,opacity:.55}));svg.append(el("circle",{cx:x(v),cy:py,r:5.2,fill:a.color,stroke:C.paper,"stroke-width":1.3}))})});arms.forEach((a,i)=>{const lx=m.l+i*190;svg.append(el("circle",{cx:lx,cy:29,r:5,fill:a.color}));svg.append(el("text",{x:lx+10,y:33,fill:a.color,"font-size":12,"font-weight":850},`${a.short} ${a.name}`))});svg.append(el("text",{x:(m.l+W-m.r)/2,y:H-15,fill:C.muted,"font-size":14,"font-weight":700,"text-anchor":"middle"},"Full endpoint BPB difference versus D0"))}

    function drawGM(id,index,title,better,formatter){
      const svg=document.getElementById(id),W=1280,H=520,m={l:92,r:164,t:48,b:67},all=A.flatMap(a=>D.greekmmlu[a.id].map(r=>r[index]));
      const [yMin,yMax]=paddedDomain(all,index===1?0.004:0.01),x=v=>m.l+v/finalIter*(W-m.l-m.r),y=v=>H-m.b-(v-yMin)/(yMax-yMin)*(H-m.t-m.b);
      svg.setAttribute("viewBox",`0 0 ${W} ${H}`); svg.append(el("title",{id:`${id}-title`},title)); svg.append(el("desc",{id:`${id}-desc`},`${title} across all 83 matched checkpoints for five data-order schedules. ${better}.`));
      for(let j=0;j<5;j++){const t=yMin+(yMax-yMin)*j/4;svg.append(el("line",{x1:m.l,x2:W-m.r,y1:y(t),y2:y(t),stroke:C.rule}));svg.append(el("text",{x:m.l-12,y:y(t)+4,fill:C.muted,"font-size":12,"text-anchor":"end"},formatter(t)))}
      [0,800,11912,19456,26584,30796,38496].forEach(t=>svg.append(el("text",{x:x(t),y:H-m.b+27,fill:C.muted,"font-size":11,"text-anchor":"middle"},t===0?"0":`${(t/1000).toFixed(t%1000?1:0)}k`)));
      addPhaseRules(svg,x,m.t,H-m.b,true); const ends=[];
      A.forEach(a=>{
        const pts=D.greekmmlu[a.id].map(r=>[r[0],r[index]]);
        svg.append(el("path",{d:linePath(pts,x,y),fill:"none",stroke:a.color,"stroke-width":a.id==="D0_mixed"?3.2:2.5,"stroke-dasharray":a.dash,"stroke-linejoin":"round","stroke-linecap":"round"}));
        pts.forEach(p=>svg.append(el("circle",{cx:x(p[0]),cy:y(p[1]),r:1.35,fill:a.color,opacity:.62})));
        const last=pts.at(-1); ends.push({a,v:last[1],py:y(last[1])}); svg.append(el("circle",{cx:x(last[0]),cy:y(last[1]),r:4,fill:a.color,stroke:C.paper,"stroke-width":1.5}));
      });
      endpointLabels(ends,m.t+4,H-m.b-4,14).forEach(d=>{const lx=W-m.r+11,px=x(finalIter);svg.append(el("line",{x1:px+5,x2:lx-3,y1:d.py,y2:d.ly,stroke:d.a.color,"stroke-width":1}));svg.append(el("text",{x:lx,y:d.ly+4,fill:d.a.color,"font-size":11,"font-weight":850},`${d.a.short} ${formatter(d.v)}`))});
      svg.append(el("line",{x1:m.l,x2:W-m.r,y1:H-m.b,y2:H-m.b,stroke:C.ink})); svg.append(el("line",{x1:m.l,x2:m.l,y1:m.t,y2:H-m.b,stroke:C.ink})); svg.append(el("text",{x:(m.l+W-m.r)/2,y:H-15,fill:C.muted,"font-size":14,"font-weight":700,"text-anchor":"middle"},"Training update"));
    }

    function drawNLLCI(){
      const svg=document.getElementById("nll-ci-chart"),arms=A.slice(1),W=1280,H=430,m={l:235,r:180,t:58,b:72};
      const rows=arms.map(a=>({a,...D.selection.greekmmlu_paired_vs_d0[a.id].delta_candidate_minus_d0.choice_nll}));
      const low=Math.min(0,...rows.map(r=>r.percentile_95_ci[0]))-.002,high=Math.max(...rows.map(r=>r.percentile_95_ci[1]))+.004,x=v=>m.l+(v-low)/(high-low)*(W-m.l-m.r),rh=(H-m.t-m.b)/rows.length;
      svg.setAttribute("viewBox",`0 0 ${W} ${H}`);svg.append(el("title",{id:"nll-ci-title"},"Paired clean GreekMMLU choice NLL difference versus D0"));svg.append(el("desc",{id:"nll-ci-desc"},"All ordered schedules have positive candidate-minus-D0 choice NLL differences, with paired 95 percent bootstrap intervals above zero."));
      svg.append(el("rect",{x:x(0),y:m.t-20,width:W-m.r-x(0),height:H-m.t-m.b+20,fill:C.green,opacity:.06}));
      for(let j=0;j<=5;j++){const t=low+(high-low)*j/5;svg.append(el("line",{x1:x(t),x2:x(t),y1:m.t-18,y2:H-m.b,stroke:Math.abs(t)<.0005?C.ink:C.rule,"stroke-width":Math.abs(t)<.0005?2:1}));svg.append(el("text",{x:x(t),y:H-m.b+27,fill:C.muted,"font-size":12,"text-anchor":"middle"},t.toFixed(3)))}
      svg.append(el("line",{x1:x(0),x2:x(0),y1:m.t-18,y2:H-m.b,stroke:C.ink,"stroke-width":2}));svg.append(el("text",{x:x(0),y:H-m.b+27,fill:C.ink,"font-size":12,"font-weight":850,"text-anchor":"middle"},"0"));
      rows.forEach((r,i)=>{const cy=m.t+(i+.5)*rh,lo=r.percentile_95_ci[0],hi=r.percentile_95_ci[1];svg.append(el("text",{x:m.l-18,y:cy+5,fill:r.a.color,"font-size":15,"font-weight":850,"text-anchor":"end"},`${r.a.short} · ${r.a.name}`));svg.append(el("line",{x1:x(lo),x2:x(hi),y1:cy,y2:cy,stroke:r.a.color,"stroke-width":5,"stroke-linecap":"round"}));svg.append(el("line",{x1:x(lo),x2:x(lo),y1:cy-9,y2:cy+9,stroke:r.a.color,"stroke-width":2}));svg.append(el("line",{x1:x(hi),x2:x(hi),y1:cy-9,y2:cy+9,stroke:r.a.color,"stroke-width":2}));svg.append(el("circle",{cx:x(r.point),cy,r:7,fill:r.a.color,stroke:C.paper,"stroke-width":2}));svg.append(el("text",{x:W-m.r+13,y:cy+4,fill:r.a.color,"font-size":12,"font-weight":850},`+${r.point.toFixed(4)}`))});
      svg.append(el("text",{x:(m.l+W-m.r)/2,y:H-16,fill:C.muted,"font-size":14,"font-weight":700,"text-anchor":"middle"},"Candidate − D0 clean choice NLL · positive means D0 is better →"));svg.append(el("text",{x:x(0)+8,y:m.t-29,fill:C.green,"font-size":11,"font-weight":850},"D0 LOWER NLL"));
    }

    function fillDecisionTables(){
      const gateBody=document.querySelector("#gate-table tbody"),ranking=D.selection.point_estimate_selection.lexicographic_order;
      A.forEach(a=>{const row=D.selection.source_retention_safety_gate[a.id],tr=document.createElement("tr"),rank=ranking.indexOf(a.id);const cells=[`${a.short} · ${a.name}`,null,a.id==="D0_mixed"?"Reference":S[row.worst_panel],a.id==="D0_mixed"?"0.00%":`${(row.worst_relative_regression*100).toFixed(2)}%`,rank<0?"Screened out":`#${rank+1}`],labels=["Arm","Safety gate","Worst retained panel","Worst relative change vs D0","Point hierarchy rank"];cells.forEach((value,i)=>{const td=document.createElement("td");td.dataset.label=labels[i];if(i===1){const badge=document.createElement("span");badge.className=`badge ${row.status}`;badge.textContent=row.status;td.append(badge)}else td.textContent=value;tr.append(td)});gateBody.append(tr)});
      const body=document.querySelector("#uncertainty-table tbody");A.slice(1).forEach(a=>{const r=D.selection.greekmmlu_paired_vs_d0[a.id],d=r.delta_candidate_minus_d0,ci=(m,digits=4,scale=1)=>`${(m.point*scale).toFixed(digits)} [${(m.percentile_95_ci[0]*scale).toFixed(digits)}, ${(m.percentile_95_ci[1]*scale).toFixed(digits)}]`,values=[`${a.short} · ${a.name}`,`${ci(d.accuracy,2,100)} pp`,ci(d.choice_nll),ci(d.correct_answer_bpb),r.mcnemar_accuracy.exact_two_sided_p.toFixed(3)],labels=["Arm vs D0","Accuracy delta [95% CI]","Choice NLL delta [95% CI]","Answer BPB delta [95% CI]","McNemar p"],tr=document.createElement("tr");values.forEach((value,i)=>{const td=document.createElement("td");td.textContent=value;td.dataset.label=labels[i];if(i===2&&d.choice_nll.percentile_95_ci[0]>0)td.classList.add("warning");tr.append(td)});body.append(tr)});
    }

    function markBest(table,columns){columns.forEach(({index,direction})=>{const cells=[...table.querySelectorAll(`tbody tr td:nth-child(${index+1})`)],vals=cells.map(c=>Number(c.dataset.value)),best=direction==="max"?Math.max(...vals):Math.min(...vals);cells.forEach((c,i)=>{if(Math.abs(vals[i]-best)<1e-12)c.classList.add("best")})})}
    function fillTables(){const gmBody=document.querySelector("#greekmmlu-table tbody"),synBody=document.querySelector("#synthesis-table tbody");A.forEach(a=>{const d=D.derived[a.id],tr=document.createElement("tr"),cells=[`${a.short} · ${a.name}`,`${(d.gm_accuracy*100).toFixed(2)}%`,d.gm_choice_nll.toFixed(4),d.gm_answer_bpb.toFixed(4),`${(d.gm_best_accuracy*100).toFixed(2)}% @ ${d.gm_best_accuracy_iteration.toLocaleString()}`,`${d.gm_best_choice_nll.toFixed(4)} @ ${d.gm_best_choice_nll_iteration.toLocaleString()}`],values=[null,d.gm_accuracy,d.gm_choice_nll,d.gm_answer_bpb,d.gm_best_accuracy,d.gm_best_choice_nll],labels=["Arm","Final accuracy","Final choice NLL","Final answer BPB","Best accuracy checkpoint","Best NLL checkpoint"];cells.forEach((v,i)=>{const td=document.createElement("td");td.textContent=v;td.dataset.label=labels[i];if(values[i]!==null)td.dataset.value=values[i];tr.append(td)});gmBody.append(tr);
        const sr=document.createElement("tr"),scells=[`${a.short} · ${a.name}`,d.hplt.toFixed(4),d.gloss_macro.toFixed(4),d.balanced_greek.toFixed(4),d.neutral.toFixed(4),d.replay_forgetting_macro.toFixed(4),`${(d.gm_accuracy*100).toFixed(2)}%`,d.gm_choice_nll.toFixed(4)],svals=[null,d.hplt,d.gloss_macro,d.balanced_greek,d.neutral,d.replay_forgetting_macro,d.gm_accuracy,d.gm_choice_nll],slabels=["Arm","HPLT BPB","GlossAPI macro","Balanced Greek","Neutral Greek","Replay forgetting","GreekMMLU accuracy","GreekMMLU NLL"];scells.forEach((v,i)=>{const td=document.createElement("td");td.textContent=v;td.dataset.label=slabels[i];if(svals[i]!==null)td.dataset.value=svals[i];sr.append(td)});synBody.append(sr)});markBest(document.getElementById("greekmmlu-table"),[{index:1,direction:"max"},{index:2,direction:"min"},{index:3,direction:"min"},{index:4,direction:"max"},{index:5,direction:"min"}]);markBest(document.getElementById("synthesis-table"),[{index:1,direction:"min"},{index:2,direction:"min"},{index:3,direction:"min"},{index:4,direction:"min"},{index:5,direction:"min"},{index:6,direction:"max"},{index:7,direction:"min"}])}

    drawMix();drawLR();learning.forEach(p=>document.getElementById("learning-grid").append(lossFacet(p,"learning")));drawHeatmap();drawTradeoff();retention.forEach(p=>document.getElementById("retention-grid").append(lossFacet(p,"retention")));drawRetentionDelta();drawGM("gm-accuracy",1,"Clean GreekMMLU zero-shot accuracy","Higher is better",v=>`${(v*100).toFixed(1)}%`);drawGM("gm-nll",2,"Clean GreekMMLU multiple-choice cross-entropy","Lower is better",v=>v.toFixed(3));drawGM("gm-bpb",3,"Clean GreekMMLU correct-answer BPB","Lower is better",v=>v.toFixed(3));drawNLLCI();fillDecisionTables();fillTables();
  </script>
</body>
</html>
'''


def main() -> None:
    payload = build_payload()
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.replace("__REPORT_DATA__", encoded)
    if "__REPORT_DATA__" in html:
        raise RuntimeError("report data placeholder was not replaced")
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
