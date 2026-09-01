#!/usr/bin/env python3
"""Build the single-page report for the stopped 8B exploratory prefix."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any


ENDPOINT = 7_152
PLANNED_UPDATES = 19_248
TOKENS_PER_UPDATE = 4_194_304
PEAK_LR = 5.5e-5
MIN_LR = 5.5e-6
WARMUP_END = 400
COOLDOWN_START = 15_398

LEARNING = (
    "hplt",
    "non_hplt",
    "openarchives",
    "greek_phd",
    "historical_polytonic",
    "neutral_external_modern_greek",
)
RETENTION = ("english", "code", "math", "de", "ru", "zh", "old_greek")
LABELS = {
    "hplt": "HPLT broad Greek",
    "non_hplt": "GlossAPI / non-HPLT",
    "openarchives": "OpenArchives",
    "greek_phd": "Greek PhD",
    "historical_polytonic": "Historical polytonic",
    "neutral_external_modern_greek": "Neutral external Greek",
    "english": "English",
    "code": "Code",
    "math": "Math",
    "de": "German",
    "ru": "Russian",
    "zh": "Chinese",
    "old_greek": "Old Greek",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_parser(repo_root: Path):
    path = repo_root / "subprojects/06_dataset_scheduling_experiments/evaluation/collect_validation_trajectory.py"
    spec = importlib.util.spec_from_file_location("full8_validation_parser", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_log, path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_validation(repo_root: Path, evidence: Path) -> tuple[dict[str, list[list[float]]], list[dict[str, Any]]]:
    initial_path = evidence / "initial_validation_receipt.json"
    initial = read_json(initial_path)
    if initial.get("schema_version") != "apertus_full_8b_initial_validation_v1" or initial.get("status") != "completed":
        raise ValueError("exact update-0 validation receipt is incomplete")
    parse_log, parser_path = load_parser(repo_root)
    bindings = [
        {"role": "validation_parser", "path": str(parser_path.resolve()), "sha256": sha256_file(parser_path)},
        {"role": "initial_validation", "path": str(initial_path.resolve()), "sha256": sha256_file(initial_path)},
    ]
    rows: dict[tuple[int, str], dict[str, Any]] = {
        (0, row["panel"]): row for row in initial["panels"]
    }
    selected_logs = sorted((evidence / "logs").glob("segments_*_training.log"))
    if len(selected_logs) != 3:
        raise ValueError(f"expected three authoritative segment logs, found {len(selected_logs)}")
    for path in selected_logs:
        bindings.append({"role": "training_log", "path": str(path.resolve()), "sha256": sha256_file(path)})
        for row in parse_log(path):
            iteration = int(row["iteration"])
            if iteration <= ENDPOINT:
                rows[(iteration, row["panel"])] = row
    panels = set(LEARNING) | set(RETENTION)
    result: dict[str, list[list[float]]] = {}
    for panel in sorted(panels):
        values = [
            [iteration, float(row["bpb"])]
            for (iteration, name), row in sorted(rows.items())
            if name == panel
        ]
        if not values or values[0][0] != 0 or values[-1][0] != ENDPOINT:
            raise ValueError(f"validation trajectory is incomplete for {panel}")
        result[panel] = values
    return result, bindings


def collect_greekmmlu(evidence: Path) -> tuple[list[dict[str, float]], list[dict[str, Any]]]:
    rows: list[dict[str, float]] = []
    bindings: list[dict[str, Any]] = []
    for directory in sorted((evidence / "greekmmlu").glob("iter_*"), key=lambda p: int(p.name.split("_")[1])):
        matches = list(directory.glob("*headline.json"))
        if len(matches) != 1:
            continue
        path = matches[0]
        value = read_json(path)
        if not isinstance(value, list) or len(value) != 1 or int(value[0]["n"]) != 16_632:
            raise ValueError(f"invalid clean GreekMMLU headline: {path}")
        iteration = int(directory.name.split("_")[1])
        row = value[0]
        rows.append(
            {
                "iteration": iteration,
                "accuracy": float(row["accuracy"]),
                "choice_nll": float(row["choice_nll"]),
                "correct_answer_bpb": float(row["correct_answer_bpb"]),
                "n": int(row["n"]),
            }
        )
        bindings.append({"role": "greekmmlu_headline", "iteration": iteration, "path": str(path.resolve()), "sha256": sha256_file(path)})
    if not rows or rows[0]["iteration"] != 0:
        raise ValueError("update-0 GreekMMLU anchor is absent")
    return rows, bindings


def summarize(values: list[list[float]]) -> dict[str, float]:
    initial = values[0][1]
    endpoint = values[-1][1]
    minimum = min(value for _, value in values)
    min_iteration = min(values, key=lambda row: row[1])[0]
    return {
        "initial": initial,
        "endpoint": endpoint,
        "minimum": minimum,
        "minimum_iteration": min_iteration,
        "net_change": endpoint - initial,
        "forgetting_from_best": endpoint - minimum,
    }


def build_payload(repo_root: Path, evidence: Path) -> dict[str, Any]:
    validation, bindings = collect_validation(repo_root, evidence)
    greekmmlu, greek_bindings = collect_greekmmlu(evidence)
    endpoint_present = any(row["iteration"] == ENDPOINT for row in greekmmlu)
    summaries = {panel: summarize(values) for panel, values in validation.items()}
    return {
        "meta": {
            "title": "8B Modern-Greek CPT — exploratory prefix",
            "endpoint_iteration": ENDPOINT,
            "endpoint_tokens": ENDPOINT * TOKENS_PER_UPDATE,
            "planned_updates": PLANNED_UPDATES,
            "planned_token_slots": PLANNED_UPDATES * TOKENS_PER_UPDATE,
            "progress_fraction": ENDPOINT / PLANNED_UPDATES,
            "tokens_per_update": TOKENS_PER_UPDATE,
            "peak_lr": PEAK_LR,
            "minimum_lr": MIN_LR,
            "warmup_end": WARMUP_END,
            "cooldown_start": COOLDOWN_START,
            "greekmmlu_endpoint_present": endpoint_present,
            "decontaminated": True,
            "anonymized": False,
            "validation_content_disjoint": True,
            "raw_points_no_smoothing": True,
        },
        "labels": LABELS,
        "learning_panels": LEARNING,
        "retention_panels": RETENTION,
        "validation": validation,
        "summaries": summaries,
        "greekmmlu": greekmmlu,
        "bindings": bindings + greek_bindings,
    }


STYLE = r"""
:root{--paper:#f4f0e7;--paper2:#fbf8f1;--ink:#272724;--muted:#6e6a62;--line:#cbc3b4;--teal:#167a75;--orange:#c86a38;--blue:#526a8c;--wine:#8b4850;--gold:#a98538;--green:#5d7756}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.48}
main{max-width:1440px;margin:auto;padding:42px 48px 90px}h1,h2,h3{font-family:Georgia,"Times New Roman",serif;font-weight:500;margin:0}h1{font-size:clamp(42px,6vw,84px);line-height:.94;max-width:1100px;letter-spacing:-.04em}h2{font-size:34px;margin-bottom:8px}h3{font-size:20px}.eyebrow{font-size:12px;text-transform:uppercase;letter-spacing:.18em;color:var(--wine);font-weight:750}.lede{font-family:Georgia,serif;font-size:23px;max-width:900px;color:#4a4741}.hero{padding:50px 0 42px;border-bottom:1px solid var(--ink)}.hero-meta,.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:32px}.metric{background:rgba(255,255,255,.35);border-top:3px solid var(--teal);padding:18px}.metric b{display:block;font-family:Georgia,serif;font-size:30px;font-weight:500}.metric span{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}.warning{margin-top:22px;padding:17px 20px;border-left:4px solid var(--orange);background:#f4e5d5}.section{padding:52px 0 14px;border-bottom:1px solid var(--line)}.section-intro{max-width:940px;color:var(--muted);margin:0 0 26px}.chart-shell{background:var(--paper2);border:1px solid var(--line);padding:18px;margin:18px 0}.chart-title{display:flex;justify-content:space-between;gap:16px;align-items:baseline;margin-bottom:5px}.chart-title small{color:var(--muted)}svg.chart{width:100%;height:auto;display:block}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.mini{background:var(--paper2);border-top:3px solid var(--teal);padding:16px}.mini.retention{border-top-color:var(--orange)}.mini h3{display:flex;justify-content:space-between;gap:8px}.mini h3 span{font-family:Inter,sans-serif;font-size:12px;color:var(--muted);font-weight:500}.legend{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin:9px 0}.swatch{display:inline-block;width:18px;height:3px;vertical-align:middle;margin-right:6px}.table-wrap{overflow:auto;background:var(--paper2);border:1px solid var(--line)}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:10px 12px;border-bottom:1px solid #ddd5c8;font-size:13px}th:first-child,td:first-child{text-align:left}th{position:sticky;top:0;background:#e9e3d8;text-transform:uppercase;letter-spacing:.06em;font-size:11px}.positive{color:var(--orange)}.negative{color:var(--teal)}.notes{display:grid;grid-template-columns:1.1fr .9fr;gap:28px}.note{background:var(--paper2);padding:20px;border-left:3px solid var(--blue)}code{font-size:.9em}.footer{padding-top:34px;color:var(--muted);font-size:12px}.axis{stroke:#a9a195;stroke-width:1}.gridline{stroke:#ddd6ca;stroke-width:1}.tick{font:11px Inter,sans-serif;fill:#756f65}.series{fill:none;stroke:var(--teal);stroke-width:2}.series.orange{stroke:var(--orange)}.series.blue{stroke:var(--blue)}.series.ghost{stroke:#aaa397;stroke-dasharray:5 5}.point{fill:var(--paper2);stroke:var(--teal);stroke-width:1.5}.vline{stroke:var(--wine);stroke-width:1;stroke-dasharray:4 4}.annotation{font:11px Inter,sans-serif;fill:var(--wine)}
@media(max-width:900px){main{padding:28px 20px 70px}.hero-meta,.cards,.grid,.notes{grid-template-columns:1fr}.metric b{font-size:26px}}
"""


SCRIPT = r"""
const D=JSON.parse(document.getElementById('report-data').textContent);
const NS='http://www.w3.org/2000/svg';
function s(tag,attrs={}){const e=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);return e}
function fmt(v,d=3){return Number(v).toFixed(d)}
function plot(el, series, opt={}){
  const W=1000,H=300,m={l:66,r:22,t:18,b:42}; const svg=s('svg',{viewBox:`0 0 ${W} ${H}`,class:'chart','aria-label':opt.label||'chart'});el.appendChild(svg);
  const all=series.flatMap(x=>x.values); let xs=all.map(x=>x[0]),ys=all.map(x=>x[1]);
  let xmin=opt.xmin??Math.min(...xs),xmax=opt.xmax??Math.max(...xs),ymin=opt.ymin??Math.min(...ys),ymax=opt.ymax??Math.max(...ys);if(ymax===ymin){ymax+=1;ymin-=1} const pad=(ymax-ymin)*.08;ymin-=pad;ymax+=pad;
  const X=x=>m.l+(x-xmin)/(xmax-xmin)*(W-m.l-m.r),Y=y=>m.t+(ymax-y)/(ymax-ymin)*(H-m.t-m.b);
  for(let i=0;i<=4;i++){let y=ymin+(ymax-ymin)*i/4,py=Y(y);svg.appendChild(s('line',{x1:m.l,y1:py,x2:W-m.r,y2:py,class:'gridline'}));let t=s('text',{x:m.l-9,y:py+4,'text-anchor':'end',class:'tick'});t.textContent=opt.percent?`${(100*y).toFixed(1)}%`:y.toFixed(opt.yDigits??3);svg.appendChild(t)}
  for(let i=0;i<=4;i++){let x=xmin+(xmax-xmin)*i/4,px=X(x);let t=s('text',{x:px,y:H-14,'text-anchor':'middle',class:'tick'});t.textContent=opt.xTokens?`${(x*D.meta.tokens_per_update/1e9).toFixed(1)}B`:Math.round(x).toLocaleString();svg.appendChild(t)}
  svg.appendChild(s('line',{x1:m.l,y1:m.t,x2:m.l,y2:H-m.b,class:'axis'}));svg.appendChild(s('line',{x1:m.l,y1:H-m.b,x2:W-m.r,y2:H-m.b,class:'axis'}));
  if(opt.marker!==undefined){let x=X(opt.marker);svg.appendChild(s('line',{x1:x,y1:m.t,x2:x,y2:H-m.b,class:'vline'}));let t=s('text',{x:x+5,y:m.t+12,class:'annotation'});t.textContent=opt.markerLabel||opt.marker;svg.appendChild(t)}
  for(const item of series){let d=item.values.map((p,i)=>`${i?'L':'M'}${X(p[0]).toFixed(2)},${Y(p[1]).toFixed(2)}`).join(' ');svg.appendChild(s('path',{d,class:`series ${item.className||''}`}));if(item.points)for(const p of item.values){svg.appendChild(s('circle',{cx:X(p[0]),cy:Y(p[1]),r:3.2,class:'point'}))}}
}
function runningGap(values){let best=Infinity;return values.map(([x,y])=>{best=Math.min(best,y);return[x,y-best]})}
function title(panel){return D.labels[panel]||panel}
for(const el of document.querySelectorAll('[data-panel-absolute]')){const p=el.dataset.panelAbsolute;plot(el,[{values:D.validation[p]}],{xTokens:true,marker:D.meta.endpoint_iteration,markerLabel:'chosen endpoint'});}
for(const el of document.querySelectorAll('[data-panel-forgetting]')){const p=el.dataset.panelForgetting;plot(el,[{values:runningGap(D.validation[p]),className:'orange'}],{xTokens:true,ymin:0,marker:D.meta.endpoint_iteration,markerLabel:'endpoint',yDigits:4});}
const lr=[];for(let u=0;u<=D.meta.planned_updates;u+=48){let v;if(u<=D.meta.warmup_end)v=D.meta.minimum_lr+(D.meta.peak_lr-D.meta.minimum_lr)*u/D.meta.warmup_end;else if(u<=D.meta.cooldown_start)v=D.meta.peak_lr;else{let q=(u-D.meta.cooldown_start)/(D.meta.planned_updates-D.meta.cooldown_start);v=D.meta.minimum_lr+(D.meta.peak_lr-D.meta.minimum_lr)*(1-Math.sqrt(Math.min(1,q)))}lr.push([u,v])}lr.push([D.meta.planned_updates,D.meta.minimum_lr]);
const observed=lr.filter(x=>x[0]<=D.meta.endpoint_iteration);if(observed.at(-1)[0]!==D.meta.endpoint_iteration)observed.push([D.meta.endpoint_iteration,D.meta.peak_lr]);plot(document.getElementById('lr-chart'),[{values:lr,className:'ghost'},{values:observed,className:'blue'}],{xmin:0,xmax:D.meta.planned_updates,xTokens:true,marker:D.meta.endpoint_iteration,markerLabel:'stopped',yDigits:6});
const gm=D.greekmmlu;plot(document.getElementById('gm-acc'),[{values:gm.map(x=>[x.iteration,x.accuracy]),points:true}],{xTokens:true,percent:true,marker:D.meta.endpoint_iteration,markerLabel:D.meta.greekmmlu_endpoint_present?'7152':'7152 pending'});plot(document.getElementById('gm-nll'),[{values:gm.map(x=>[x.iteration,x.choice_nll]),points:true,className:'blue'}],{xTokens:true,marker:D.meta.endpoint_iteration,markerLabel:D.meta.greekmmlu_endpoint_present?'7152':'7152 pending'});plot(document.getElementById('gm-bpb'),[{values:gm.map(x=>[x.iteration,x.correct_answer_bpb]),points:true,className:'orange'}],{xTokens:true,marker:D.meta.endpoint_iteration,markerLabel:D.meta.greekmmlu_endpoint_present?'7152':'7152 pending'});
"""


def table_rows(payload: dict[str, Any], panels: tuple[str, ...]) -> str:
    rows = []
    for panel in panels:
        value = payload["summaries"][panel]
        net_class = "negative" if value["net_change"] < 0 else "positive"
        rows.append(
            f"<tr><td>{LABELS[panel]}</td><td>{value['initial']:.4f}</td><td>{value['minimum']:.4f}</td>"
            f"<td>{value['endpoint']:.4f}</td><td class='{net_class}'>{value['net_change']:+.4f}</td>"
            f"<td class='positive'>{value['forgetting_from_best']:+.4f}</td><td>{int(value['minimum_iteration']):,}</td></tr>"
        )
    return "".join(rows)


def running_gap(values: list[list[float]]) -> list[list[float]]:
    best = math.inf
    result = []
    for x_value, y_value in values:
        best = min(best, y_value)
        result.append([x_value, y_value - best])
    return result


def svg_plot(
    series: list[tuple[list[list[float]], str, bool]],
    *,
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    marker: float | None = None,
    marker_label: str = "",
    x_tokens: bool = True,
    percent: bool = False,
    y_digits: int = 3,
) -> str:
    width, height = 1000, 300
    left, right, top, bottom = 66, 22, 18, 42
    points = [point for values, _, _ in series for point in values]
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x_min = min(x_values) if xmin is None else xmin
    x_max = max(x_values) if xmax is None else xmax
    y_min = min(y_values) if ymin is None else ymin
    y_max = max(y_values) if ymax is None else ymax
    if y_max == y_min:
        y_min -= 1
        y_max += 1
    padding = (y_max - y_min) * 0.08
    if ymin is None:
        y_min -= padding
    if ymax is None:
        y_max += padding

    def x_position(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (width - left - right)

    def y_position(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * (height - top - bottom)

    body: list[str] = []
    for index in range(5):
        value = y_min + (y_max - y_min) * index / 4
        y_pos = y_position(value)
        label = f"{100 * value:.1f}%" if percent else f"{value:.{y_digits}f}"
        body.append(f"<line x1='{left}' y1='{y_pos:.2f}' x2='{width-right}' y2='{y_pos:.2f}' class='gridline'/><text x='{left-9}' y='{y_pos+4:.2f}' text-anchor='end' class='tick'>{label}</text>")
    for index in range(5):
        value = x_min + (x_max - x_min) * index / 4
        x_pos = x_position(value)
        label = f"{value*TOKENS_PER_UPDATE/1e9:.1f}B" if x_tokens else f"{round(value):,}"
        body.append(f"<text x='{x_pos:.2f}' y='{height-14}' text-anchor='middle' class='tick'>{label}</text>")
    body.append(f"<line x1='{left}' y1='{top}' x2='{left}' y2='{height-bottom}' class='axis'/><line x1='{left}' y1='{height-bottom}' x2='{width-right}' y2='{height-bottom}' class='axis'/>")
    if marker is not None:
        marker_x = x_position(marker)
        body.append(f"<line x1='{marker_x:.2f}' y1='{top}' x2='{marker_x:.2f}' y2='{height-bottom}' class='vline'/><text x='{marker_x+5:.2f}' y='{top+12}' class='annotation'>{marker_label}</text>")
    for values, class_name, show_points in series:
        path = " ".join(
            f"{'M' if index == 0 else 'L'}{x_position(point[0]):.2f},{y_position(point[1]):.2f}"
            for index, point in enumerate(values)
        )
        body.append(f"<path d='{path}' class='series {class_name}'/>")
        if show_points:
            body.extend(
                f"<circle cx='{x_position(point[0]):.2f}' cy='{y_position(point[1]):.2f}' r='3.2' class='point'/>"
                for point in values
            )
    return f"<svg class='chart' viewBox='0 0 {width} {height}' role='img'>{''.join(body)}</svg>"


def build_html(payload: dict[str, Any]) -> str:
    meta = payload["meta"]
    learning_cards = "".join(
        f"<article class='mini'><h3>{LABELS[p]}<span>absolute BPB</span></h3>{svg_plot([(payload['validation'][p], '', False)], marker=ENDPOINT, marker_label='endpoint')}</article>"
        for p in LEARNING
    )
    retention_cards = "".join(
        f"<article class='mini retention'><h3>{LABELS[p]}<span>absolute BPB</span></h3>{svg_plot([(payload['validation'][p], '', False)], marker=ENDPOINT, marker_label='endpoint')}</article>"
        for p in RETENTION
    )
    forgetting_cards = "".join(
        f"<article class='mini retention'><h3>{LABELS[p]}<span>BPB above running minimum</span></h3>{svg_plot([(running_gap(payload['validation'][p]), 'orange', False)], ymin=0, marker=ENDPOINT, marker_label='endpoint', y_digits=4)}</article>"
        for p in RETENTION
    )
    lr: list[list[float]] = []
    for update in range(0, PLANNED_UPDATES + 1, 48):
        if update <= WARMUP_END:
            value = MIN_LR + (PEAK_LR - MIN_LR) * update / WARMUP_END
        elif update <= COOLDOWN_START:
            value = PEAK_LR
        else:
            progress = (update - COOLDOWN_START) / (PLANNED_UPDATES - COOLDOWN_START)
            value = MIN_LR + (PEAK_LR - MIN_LR) * (1 - math.sqrt(min(1, progress)))
        lr.append([update, value])
    lr.append([PLANNED_UPDATES, MIN_LR])
    observed_lr = [row for row in lr if row[0] <= ENDPOINT]
    if observed_lr[-1][0] != ENDPOINT:
        observed_lr.append([ENDPOINT, PEAK_LR])
    lr_chart = svg_plot(
        [(lr, "ghost", False), (observed_lr, "blue", False)],
        xmin=0,
        xmax=PLANNED_UPDATES,
        marker=ENDPOINT,
        marker_label="stopped",
        y_digits=6,
    )
    greekmmlu = payload["greekmmlu"]
    endpoint_label = "7,152" if meta["greekmmlu_endpoint_present"] else "7,152 pending"
    gm_accuracy = svg_plot(
        [([[row["iteration"], row["accuracy"]] for row in greekmmlu], "", True)],
        marker=ENDPOINT,
        marker_label=endpoint_label,
        percent=True,
    )
    gm_nll = svg_plot(
        [([[row["iteration"], row["choice_nll"]] for row in greekmmlu], "blue", True)],
        marker=ENDPOINT,
        marker_label=endpoint_label,
    )
    gm_bpb = svg_plot(
        [([[row["iteration"], row["correct_answer_bpb"]] for row in greekmmlu], "orange", True)],
        marker=ENDPOINT,
        marker_label=endpoint_label,
    )
    gm_status = "included" if meta["greekmmlu_endpoint_present"] else "still running"
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>8B Greek CPT — exploratory prefix</title><style>{STYLE}</style></head><body><main>
<header class='hero'><div class='eyebrow'>Apertus 8B · stopped exploratory control · 6 August 2026</div><h1>What the first 30B tokens taught—and forgot</h1><p class='lede'>A complete, unsmoothed view of the mixed-data run through the predeclared checkpoint at update 7,152. This is an exploratory control, not the production result.</p>
<div class='hero-meta'><div class='metric'><b>{meta['endpoint_tokens']/1e9:.3f}B</b><span>token slots consumed</span></div><div class='metric'><b>{100*meta['progress_fraction']:.1f}%</b><span>of planned trajectory</span></div><div class='metric'><b>13</b><span>content-disjoint panels</span></div><div class='metric'><b>{gm_status}</b><span>GreekMMLU @ 7,152</span></div></div>
<div class='warning'><strong>Interpretation boundary.</strong> The training text was GreekMMLU-decontaminated but had not received the required Apertus PII anonymization pass. We stopped the run for that reason. The validation panels themselves are the corrected, exact-content-disjoint set. Results diagnose the trajectory; they do not authorize publishing or continuing this checkpoint.</div></header>

<section class='section'><div class='eyebrow'>Optimization context</div><h2>The LR never entered cooldown</h2><p class='section-intro'>The solid line is the LR actually traversed: warmup to 5.5×10⁻⁵ by update 400, then a stable plateau. The dashed continuation is the planned WSD-10 tail. Because the run stopped at 7,152, no observation here says anything about the 10% LR floor.</p><div class='chart-shell'><div class='chart-title'><h3>Planned and observed learning rate</h3><small>x-axis: cumulative token slots</small></div>{lr_chart}<div class='legend'><span><i class='swatch' style='background:var(--blue)'></i>observed</span><span><i class='swatch' style='background:#aaa397'></i>unobserved plan</span></div></div></section>

<section class='section'><div class='eyebrow'>Learning</div><h2>Greek learning is broad, not confined to GlossAPI</h2><p class='section-intro'>Absolute bits per UTF-8 byte from update 0 through 7,152. Every point emitted by the fast panel is shown; no smoothing, interpolation, or cherry-picked tail window is used.</p><div class='grid'>{learning_cards}</div><div class='table-wrap'><table><thead><tr><th>Panel</th><th>Initial</th><th>Best</th><th>At 7,152</th><th>Net Δ</th><th>Above best</th><th>Best update</th></tr></thead><tbody>{table_rows(payload, LEARNING)}</tbody></table></div></section>

<section class='section'><div class='eyebrow'>Retention</div><h2>Absolute loss shows capability; zoomed loss shows forgetting</h2><p class='section-intro'>The first grid keeps the full absolute BPB trajectory. The second subtracts each panel’s running minimum, which exposes small regressions that disappear against the large y-range. Old Greek is included with the corrected clean panel—not the rejected 99.16%-overlapping inherited panel.</p><h3>Absolute validation BPB</h3><div class='grid'>{retention_cards}</div><h3 style='margin-top:30px'>Forgetting: current BPB minus best BPB seen so far</h3><div class='grid'>{forgetting_cards}</div><div class='table-wrap'><table><thead><tr><th>Panel</th><th>Initial</th><th>Best</th><th>At 7,152</th><th>Net Δ</th><th>Forgetting</th><th>Best update</th></tr></thead><tbody>{table_rows(payload, RETENTION)}</tbody></table></div></section>

<section class='section'><div class='eyebrow'>Native Greek evaluation</div><h2>GreekMMLU improves sharply, then oscillates</h2><p class='section-intro'>The clean 16,632-question subset is scored continuously as well as by accuracy. Choice NLL and correct-answer BPB remain informative when accuracy moves by only a few tenths of a point. The exact 7,152 result is inserted only after its receipt exists.</p><div class='grid'><article class='mini'><h3>Accuracy<span>higher is better</span></h3>{gm_accuracy}</article><article class='mini'><h3>Choice NLL<span>lower is better</span></h3>{gm_nll}</article><article class='mini'><h3>Correct-answer BPB<span>lower is better</span></h3>{gm_bpb}</article></div></section>

<section class='section'><div class='eyebrow'>What can be concluded now</div><h2>A useful trajectory, not a reusable endpoint</h2><div class='notes'><div class='note'><h3>Learning signal</h3><p>The source-conditioned curves show substantial learning on HPLT, GlossAPI/non-HPLT, institutional, academic, historical, and neutral Greek. The neutral set improves as well, so the signal is not merely memorization of the last source distribution.</p></div><div class='note'><h3>Retention signal</h3><p>Foreign panels show small but visible departures from their best observed BPB. The zoomed charts are the correct view for comparing forgetting; absolute curves alone can make those changes look flat.</p></div><div class='note'><h3>GreekMMLU signal</h3><p>The large early improvement is real on the decontaminated subset, but later checkpoints are non-monotonic. We should select using continuous metrics plus source loss—not a single accuracy maximum.</p></div><div class='note'><h3>Next experiment</h3><p>Rebuild the complete Modern-Greek and replay stream with the receipt-bound anonymization pass, global post-mask deduplication, and raw-validation collision removal; then recreate the 79/20/1 schedule and restart from the original Token-Distilled initialization.</p></div></div></section>

<section class='section'><div class='eyebrow'>Methods & provenance</div><h2>Evidence contract</h2><p class='section-intro'>Update 7,152 equals {meta['endpoint_tokens']:,} token slots. Only successful authoritative segment logs are used, and later observations produced while the graceful stop completed are excluded. Validation points are raw fast-panel measurements; the forthcoming per-document endpoint adds cluster-level uncertainty without rewriting this trajectory.</p><div class='table-wrap'><table><thead><tr><th>Evidence role</th><th>Iteration</th><th>SHA-256</th><th>Path</th></tr></thead><tbody>{''.join(f"<tr><td>{row['role']}</td><td>{row.get('iteration','—')}</td><td><code>{row['sha256'][:16]}…</code></td><td style='text-align:left'><code>{row['path']}</code></td></tr>" for row in payload['bindings'])}</tbody></table></div></section>
<footer class='footer'>Generated from receipt-bound artifacts. Full JSON evidence is embedded below; the report has no network dependencies.</footer>
<script id='report-data' type='application/json'>{data}</script></main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-output", type=Path)
    args = parser.parse_args()
    payload = build_payload(args.repo_root.resolve(), args.evidence_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html(payload), encoding="utf-8")
    data_output = args.data_output or args.output.with_suffix(".data.json")
    data_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(args.output.resolve()), "data": str(data_output.resolve()), "greekmmlu_points": len(payload["greekmmlu"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
