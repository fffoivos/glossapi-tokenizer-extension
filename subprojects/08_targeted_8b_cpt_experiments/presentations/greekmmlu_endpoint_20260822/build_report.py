#!/usr/bin/env python3
"""Build the trajectory-first 1.5B versus 8B HPLT-to-OA report."""

from __future__ import annotations

import hashlib
import html
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
OUTPUT = ROOT / "GREEKMMLU_H2G_CROSS_SCALE_RESULTS_20260822.html"

INK = "#152b3c"
MUTED = "#687985"
GRID = "#d9d5cd"
PAPER = "#f5f1e8"
WHITE = "#fffdf8"
BLUE = "#276a9f"
RED = "#c84b42"
TEAL = "#21857a"
GOLD = "#c6922e"
PURPLE = "#7b5aa6"
SLATE = "#506879"

TOKENS_PER_UPDATE = 4_194_304
TOTAL_UPDATE = 3694
HPLT_END = 2261
OA_END = 3218
CHECKPOINTS = [0, 238, 476, 714, 952, 1190, 1428, 1666, 1904, 2142, 2261,
               2380, 2618, 2856, 3094, 3218, 3456, 3694]


def load(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def esc(value: object) -> str:
    return html.escape(str(value))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokens_b(update: int | float) -> float:
    return update * TOKENS_PER_UPDATE / 1e9


def pct(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}%"


def path_points(points: list[tuple[float, float]], x, y) -> str:
    return " ".join(("M" if i == 0 else "L") + f" {x(px):.2f} {y(py):.2f}"
                    for i, (px, py) in enumerate(points))


def chart_shell(width: int, height: int, title: str, desc: str) -> list[str]:
    ident = hashlib.sha1(title.encode()).hexdigest()[:10]
    return [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-labelledby="t-{ident} d-{ident}">',
        f'<title id="t-{ident}">{esc(title)}</title>',
        f'<desc id="d-{ident}">{esc(desc)}</desc>',
    ]


def phase_background(out: list[str], x, top: float, bottom: float) -> None:
    phases = [
        (0, HPLT_END, BLUE, 0.055, "HPLT"),
        (HPLT_END, OA_END, TEAL, 0.060, "OpenArchives"),
        (OA_END, TOTAL_UPDATE, GOLD, 0.075, "extension"),
    ]
    for start, end, color, opacity, label in phases:
        x0, x1 = x(start), x(end)
        out.append(f'<rect x="{x0:.2f}" y="{top}" width="{x1-x0:.2f}" '
                   f'height="{bottom-top}" fill="{color}" opacity="{opacity}"/>')
        out.append(f'<text x="{(x0+x1)/2:.2f}" y="{top+18}" text-anchor="middle" '
                   f'class="phase-note" fill="{color}">{label}</text>')
    for u in (HPLT_END, OA_END):
        out.append(f'<line x1="{x(u):.2f}" y1="{top}" x2="{x(u):.2f}" y2="{bottom}" '
                   f'stroke="{INK}" stroke-width="1.3" stroke-dasharray="7 6" opacity=".55"/>')


def schedule_lr_svg(traj: dict) -> str:
    width, height = 1180, 390
    left, right, top, bottom = 76, 34, 44, 72
    split = 178
    plot_w = width - left - right
    x = lambda u: left + u / TOTAL_UPDATE * plot_w
    out = chart_shell(width, height, "Matched data and learning-rate schedule",
                      "Both model scales consume the same data phases and use the same learning-rate trajectory.")
    phases = [(0, HPLT_END, BLUE, "HPLT", "9.483B"),
              (HPLT_END, OA_END, TEAL, "OpenArchives", "4.014B"),
              (OA_END, TOTAL_UPDATE, GOLD, "unseen OA + replay", "1.996B")]
    for start, end, color, label, amount in phases:
        x0, x1 = x(start), x(end)
        out.append(f'<rect x="{x0:.1f}" y="72" width="{x1-x0:.1f}" height="64" rx="2" fill="{color}" opacity=".91"/>')
        out.append(f'<text x="{(x0+x1)/2:.1f}" y="101" text-anchor="middle" class="phase-label">{esc(label)}</text>')
        out.append(f'<text x="{(x0+x1)/2:.1f}" y="124" text-anchor="middle" class="phase-small">{amount}</text>')
    for u, label in [(0, "TD init"), (HPLT_END, "phase switch"), (OA_END, "replication endpoint"), (TOTAL_UPDATE, "final")]:
        out.append(f'<line x1="{x(u):.1f}" y1="56" x2="{x(u):.1f}" y2="151" stroke="{INK}" stroke-width="1.7"/>')
        anchor = "start" if u == 0 else "end" if u == TOTAL_UPDATE else "middle"
        out.append(f'<text x="{x(u):.1f}" y="169" text-anchor="{anchor}" class="axis-strong">{tokens_b(u):.3f}B</text>')
        out.append(f'<text x="{x(u):.1f}" y="187" text-anchor="{anchor}" class="note">{label}</text>')

    lr = [(u, v) for u, v in traj["scales"]["8b"]["learning_rate"]]
    ymax = max(v for _, v in lr) * 1.12
    y0, y1 = split + 42, height - bottom
    y = lambda v: y1 - v / ymax * (y1 - y0)
    phase_background(out, x, y0, y1)
    for frac in (0, .5, 1):
        val = ymax * frac
        yy = y(val)
        out.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="{GRID}"/>')
        out.append(f'<text x="{left-10}" y="{yy+5:.1f}" text-anchor="end" class="axis">{val*1e5:.1f}</text>')
    out.append(f'<path d="{path_points(lr, x, y)}" fill="none" stroke="{PURPLE}" stroke-width="4"/>')
    out.append(f'<text x="{left}" y="{y0-12}" class="panel-title">Learning rate</text>')
    out.append(f'<text x="{left+116}" y="{y0-12}" class="note">×10⁻⁵ · identical at both scales</text>')
    for u in (0, 952, HPLT_END, OA_END, TOTAL_UPDATE):
        out.append(f'<text x="{x(u):.1f}" y="{height-32}" text-anchor="middle" class="axis">{tokens_b(u):.1f}B</text>')
    out.append(f'<text x="{(left+width-right)/2:.1f}" y="{height-8}" text-anchor="middle" class="axis-strong">consumed token slots</text>')
    out.append("</svg>")
    return "".join(out)


def normalized_learning_svg(traj: dict) -> str:
    panels = [("hplt", "HPLT heldout"), ("openarchives", "OpenArchives heldout"), ("greek_phd", "Greek PhD heldout")]
    width, height = 1180, 520
    left, right, top, bottom = 72, 28, 62, 72
    gap = 24
    panel_w = (width - left - right - 2 * gap) / 3
    plot_h = height - top - bottom
    out = chart_shell(width, height, "Within-model normalized learning trajectories",
                      "Loss reduction is normalized from zero at update 25 to one hundred percent at the final measured point, allowing comparison of trajectory shape rather than absolute model-scale score.")
    for pi, (panel, label) in enumerate(panels):
        px0 = left + pi * (panel_w + gap)
        px1 = px0 + panel_w
        x = lambda u, a=px0, b=px1: a + u / TOTAL_UPDATE * (b - a)
        y = lambda v: top + plot_h - (v + .08) / 1.20 * plot_h
        phase_background(out, x, top, top + plot_h)
        for tick in (0, .25, .5, .75, 1):
            yy = y(tick)
            out.append(f'<line x1="{px0}" y1="{yy:.1f}" x2="{px1}" y2="{yy:.1f}" stroke="{GRID}"/>')
            if pi == 0:
                out.append(f'<text x="{px0-9}" y="{yy+4:.1f}" text-anchor="end" class="axis">{tick*100:.0f}%</text>')
        out.append(f'<text x="{px0}" y="32" class="panel-title">{label}</text>')
        for scale, color, model in [("1p5b", RED, "1.5B"), ("8b", BLUE, "8B")]:
            raw = traj["scales"][scale]["validation_loss"][panel]
            first, last = raw[0][1], raw[-1][1]
            values = [(u, (first-v)/(first-last)) for u, v in raw]
            out.append(f'<path d="{path_points(values, x, y)}" fill="none" stroke="{color}" stroke-width="3.2"/>')
            u, v = values[-1]
            out.append(f'<circle cx="{x(u):.1f}" cy="{y(v):.1f}" r="5" fill="{color}"/>')
        for u in (0, HPLT_END, OA_END, TOTAL_UPDATE):
            out.append(f'<text x="{x(u):.1f}" y="{height-39}" text-anchor="middle" class="axis">{tokens_b(u):.1f}</text>')
    out.append(f'<text x="{left}" y="{height-9}" class="axis-strong">B token slots</text>')
    out.append(f'<line x1="{width-230}" y1="24" x2="{width-198}" y2="24" stroke="{RED}" stroke-width="4"/><text x="{width-190}" y="29" class="note">1.5B</text>')
    out.append(f'<line x1="{width-118}" y1="24" x2="{width-86}" y2="24" stroke="{BLUE}" stroke-width="4"/><text x="{width-78}" y="29" class="note">8B</text>')
    out.append("</svg>")
    return "".join(out)


def absolute_learning_svg(traj: dict) -> str:
    panels = [("hplt", "HPLT"), ("openarchives", "OpenArchives"), ("greek_phd", "Greek PhD")]
    width, height = 1180, 580
    left, right, top, bottom = 72, 30, 58, 64
    gap = 26
    panel_w = (width-left-right-2*gap)/3
    plot_h = height-top-bottom
    out = chart_shell(width, height, "Absolute heldout-loss trajectories",
                      "The 8B model has lower absolute heldout loss, while both scales bend at the same data-phase boundaries.")
    for pi, (panel, label) in enumerate(panels):
        px0=left+pi*(panel_w+gap); px1=px0+panel_w
        all_vals=[v for scale in ("1p5b","8b") for _,v in traj["scales"][scale]["validation_loss"][panel]]
        lo=min(all_vals); hi=max(all_vals); pad=(hi-lo)*.08
        lo-=pad; hi+=pad
        x=lambda u,a=px0,b=px1:a+u/TOTAL_UPDATE*(b-a)
        y=lambda v:top+plot_h-(v-lo)/(hi-lo)*plot_h
        phase_background(out,x,top,top+plot_h)
        for k in range(5):
            val=lo+(hi-lo)*k/4; yy=y(val)
            out.append(f'<line x1="{px0}" y1="{yy:.1f}" x2="{px1}" y2="{yy:.1f}" stroke="{GRID}"/>')
            out.append(f'<text x="{px0-9}" y="{yy+4:.1f}" text-anchor="end" class="axis">{val:.2f}</text>')
        out.append(f'<text x="{px0}" y="30" class="panel-title">{label}</text>')
        for scale,color in [("1p5b",RED),("8b",BLUE)]:
            points=traj["scales"][scale]["validation_loss"][panel]
            out.append(f'<path d="{path_points(points,x,y)}" fill="none" stroke="{color}" stroke-width="3"/>')
        for u in (0,HPLT_END,OA_END,TOTAL_UPDATE):
            out.append(f'<text x="{x(u):.1f}" y="{height-30}" text-anchor="middle" class="axis">{tokens_b(u):.1f}</text>')
    out.append(f'<text x="{left}" y="{height-7}" class="axis-strong">loss · lower is better</text>')
    out.append("</svg>")
    return "".join(out)


def training_loss_svg(traj: dict) -> str:
    width,height=1180,500
    left,right,top,bottom=76,34,48,70
    x=lambda u:left+u/TOTAL_UPDATE*(width-left-right)
    all_vals=[]; curves={}
    for scale in ("1p5b","8b"):
        raw=traj["scales"][scale]["training_loss"]
        smooth=[]
        window=35
        for i in range(0,len(raw),8):
            chunk=[v for _,v in raw[max(0,i-window):min(len(raw),i+window+1)]]
            smooth.append((raw[i][0],sum(chunk)/len(chunk)))
        curves[scale]=smooth; all_vals.extend(v for _,v in smooth if _>=25)
    lo=min(all_vals)*.95;hi=min(4.0,max(all_vals))*1.03
    y=lambda v:top+(hi-v)/(hi-lo)*(height-top-bottom)
    out=chart_shell(width,height,"Optimization-loss trajectories",
                    "Smoothed training loss for both scales across the identical curriculum and learning-rate schedule.")
    phase_background(out,x,top,height-bottom)
    for k in range(6):
        val=lo+(hi-lo)*k/5;yy=y(val)
        out.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="{GRID}"/>')
        out.append(f'<text x="{left-10}" y="{yy+5:.1f}" text-anchor="end" class="axis">{val:.2f}</text>')
    for scale,color,label in [("1p5b",RED,"1.5B"),("8b",BLUE,"8B")]:
        pts=[p for p in curves[scale] if p[0]>=25]
        out.append(f'<path d="{path_points(pts,x,y)}" fill="none" stroke="{color}" stroke-width="3.4"/>')
        u,v=pts[-1]
        out.append(f'<text x="{x(u)-8:.1f}" y="{y(v)-10:.1f}" text-anchor="end" class="value-small" fill="{color}">{label} {v:.3f}</text>')
    for u in (0,952,HPLT_END,OA_END,TOTAL_UPDATE):
        out.append(f'<text x="{x(u):.1f}" y="{height-35}" text-anchor="middle" class="axis">{tokens_b(u):.1f}B</text>')
    out.append(f'<text x="{(left+width-right)/2:.1f}" y="{height-8}" text-anchor="middle" class="axis-strong">consumed token slots</text>')
    out.append("</svg>")
    return "".join(out)


def forgetting_svg(traj: dict) -> str:
    panels=[("english","English",RED),("de","German",GOLD),("ru","Russian",PURPLE),("zh","Chinese",TEAL),("code","Code",SLATE)]
    width,height=1180,600
    left,right,top,bottom=72,34,74,70
    gap=50; panel_w=(width-left-right-gap)/2; plot_h=height-top-bottom
    out=chart_shell(width,height,"Retention trajectories as loss above each panel's best-so-far value",
                    "Zero means the model is at its best observed loss so far; upward movement is forgetting relative to that model and panel's own historical minimum.")
    for si,(scale,label) in enumerate([("1p5b","Apertus 1.5B"),("8b","Apertus 8B")]):
        px0=left+si*(panel_w+gap);px1=px0+panel_w
        x=lambda u,a=px0,b=px1:a+u/TOTAL_UPDATE*(b-a)
        curves=[]; ymax=0
        for panel,_,color in panels:
            running=math.inf;pts=[]
            for u,v in traj["scales"][scale]["validation_loss"][panel]:
                running=min(running,v);pts.append((u,v-running))
            curves.append((panel,color,pts));ymax=max(ymax,max(v for _,v in pts))
        ymax=max(.05,ymax*1.08)
        y=lambda v:top+plot_h-v/ymax*plot_h
        phase_background(out,x,top,top+plot_h)
        for k in range(5):
            val=ymax*k/4;yy=y(val)
            out.append(f'<line x1="{px0}" y1="{yy:.1f}" x2="{px1}" y2="{yy:.1f}" stroke="{GRID}"/>')
            out.append(f'<text x="{px0-9}" y="{yy+4:.1f}" text-anchor="end" class="axis">+{val:.3f}</text>')
        out.append(f'<text x="{px0}" y="32" class="panel-title">{label}</text>')
        for _,color,pts in curves:
            out.append(f'<path d="{path_points(pts,x,y)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for u in (0,HPLT_END,OA_END,TOTAL_UPDATE):
            out.append(f'<text x="{x(u):.1f}" y="{height-34}" text-anchor="middle" class="axis">{tokens_b(u):.1f}</text>')
    legend_x=left
    for i,(_,label,color) in enumerate(panels):
        lx=legend_x+i*142
        out.append(f'<line x1="{lx}" y1="{height-12}" x2="{lx+26}" y2="{height-12}" stroke="{color}" stroke-width="4"/>')
        out.append(f'<text x="{lx+34}" y="{height-7}" class="note">{label}</text>')
    out.append("</svg>")
    return "".join(out)


def greekmmlu_coverage_svg(one: dict, eight: dict) -> str:
    width,height=1180,430
    left,right,top,bottom=72,48,76,72
    x=lambda u:left+u/TOTAL_UPDATE*(width-left-right)
    rows=[("1.5B",one,RED,170),("8B",eight,BLUE,280)]
    out=chart_shell(width,height,"GreekMMLU checkpoint coverage",
                    "The frozen design requested checkpoint trajectories, but only the final clean-panel result exists for each model scale.")
    phase_background(out,x,top,height-bottom)
    for label,metrics,color,y in rows:
        out.append(f'<text x="{left-18}" y="{y+6}" text-anchor="end" class="model-label">{label}</text>')
        out.append(f'<line x1="{left}" y1="{y}" x2="{width-right}" y2="{y}" stroke="{GRID}" stroke-width="2"/>')
        for u in CHECKPOINTS:
            out.append(f'<circle cx="{x(u):.1f}" cy="{y}" r="5" fill="{WHITE}" stroke="#adb6bd" stroke-width="1.5"/>')
        out.append(f'<circle cx="{x(TOTAL_UPDATE):.1f}" cy="{y}" r="10" fill="{color}" stroke="{WHITE}" stroke-width="3"/>')
        out.append(f'<text x="{x(TOTAL_UPDATE)-12:.1f}" y="{y-17}" text-anchor="end" class="value-small" fill="{color}">{pct(metrics["accuracy"])}</text>')
    for u in (0,HPLT_END,OA_END,TOTAL_UPDATE):
        out.append(f'<text x="{x(u):.1f}" y="{height-35}" text-anchor="middle" class="axis">{tokens_b(u):.1f}B</text>')
    out.append(f'<text x="{left}" y="45" class="panel-title">17 planned intermediate points · 0 completed</text>')
    out.append(f'<text x="{left}" y="{height-8}" class="axis-strong">Open circles are missing evaluations; filled circles are completed full-clean evaluations.</text>')
    out.append("</svg>")
    return "".join(out)


def historical_svg(hist: dict) -> str:
    pts=[(u,m["micro"]) for u,m in hist["arms"]["td"]["native_mcq"]]
    width,height=1180,420
    left,right,top,bottom=76,48,50,72
    x=lambda u:left+u/3218*(width-left-right)
    lo=.47;hi=.61
    y=lambda v:top+(hi-v)/(hi-lo)*(height-top-bottom)
    out=chart_shell(width,height,"Historical 8B GreekMMLU trajectory — context only",
                    "The earlier token-distillation HPLT-to-OpenArchives run used the public 16632-question evaluator and therefore must not be numerically merged with the current clean 16159-question endpoints.")
    for tick in (.48,.52,.56,.60):
        yy=y(tick)
        out.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="{GRID}"/>')
        out.append(f'<text x="{left-10}" y="{yy+5:.1f}" text-anchor="end" class="axis">{tick*100:.0f}%</text>')
    out.append(f'<rect x="{x(0):.1f}" y="{top}" width="{x(2261)-x(0):.1f}" height="{height-top-bottom}" fill="{BLUE}" opacity=".05"/>')
    out.append(f'<rect x="{x(2261):.1f}" y="{top}" width="{x(3218)-x(2261):.1f}" height="{height-top-bottom}" fill="{TEAL}" opacity=".06"/>')
    out.append(f'<line x1="{x(2261):.1f}" y1="{top}" x2="{x(2261):.1f}" y2="{height-bottom}" stroke="{INK}" stroke-dasharray="7 6" opacity=".55"/>')
    out.append(f'<path d="{path_points(pts,x,y)}" fill="none" stroke="{BLUE}" stroke-width="4"/>')
    for u,v in pts:
        out.append(f'<circle cx="{x(u):.1f}" cy="{y(v):.1f}" r="4.5" fill="{BLUE}"/>')
    best=max(pts,key=lambda p:p[1])
    out.append(f'<text x="{x(best[0])-8:.1f}" y="{y(best[1])-13:.1f}" text-anchor="end" class="value-small" fill="{BLUE}">available predecessor peak {pct(best[1])}</text>')
    for u in (0,952,2261,3218):
        out.append(f'<text x="{x(u):.1f}" y="{height-35}" text-anchor="middle" class="axis">{tokens_b(u):.1f}B</text>')
    out.append(f'<text x="{left}" y="{height-8}" class="note">This surviving curve is the earlier two-arm TD study (ending 58.66%), not the later β₂-selected run whose endpoint was 59.94%.</text>')
    out.append("</svg>")
    return "".join(out)


def key_table(traj: dict) -> str:
    steps=[25,2250,3200,3450,3675]
    names={25:"first measured",2250:"HPLT boundary",3200:"OA endpoint",3450:"extension 1",3675:"final measured"}
    rows=[]
    for panel,label in [("hplt","HPLT"),("openarchives","OpenArchives"),("greek_phd","Greek PhD")]:
        for scale,display in [("1p5b","1.5B"),("8b","8B")]:
            vals=dict(traj["scales"][scale]["validation_loss"][panel])
            cells="".join(f'<td class="num">{vals[s]:.4f}</td>' for s in steps)
            rows.append(f'<tr><td>{label}</td><td>{display}</td>{cells}</tr>')
    heads="".join(f'<th>{names[s]}<span>{tokens_b(s):.2f}B</span></th>' for s in steps)
    return f'<table><thead><tr><th>panel</th><th>scale</th>{heads}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def phase_delta_table(traj: dict) -> str:
    steps=(2250,3200,3675)
    rows=[]
    for panel,label in [("hplt","HPLT"),("openarchives","OpenArchives"),("greek_phd","Greek PhD")]:
        for scale,display in [("1p5b","1.5B"),("8b","8B")]:
            vals=dict(traj["scales"][scale]["validation_loss"][panel])
            oa=vals[steps[1]]-vals[steps[0]]
            ext=vals[steps[2]]-vals[steps[1]]
            def cell(v):
                cls="gain" if v<0 else "regress"
                return f'<td class="num {cls}">{v:+.4f}</td>'
            rows.append(f'<tr><td>{label}</td><td>{display}</td>{cell(oa)}{cell(ext)}</tr>')
    return '<table><thead><tr><th>panel</th><th>scale</th><th>OpenArchives phase<span>2250 → 3200</span></th><th>extension<span>3200 → 3675</span></th></tr></thead><tbody>'+"".join(rows)+"</tbody></table>"


def main() -> None:
    traj=load("training_trajectories.json")
    d1=load("1p5b_summary.json")
    d8=load("8b_summary.json")
    hist=load("historical_cpt_2arm_summary.json")
    o1=d1["views"]["full_clean"]["metrics"]["overall"]
    o8=d8["views"]["full_clean"]["metrics"]["overall"]

    # Shape difference: mean absolute distance between curves after normalizing
    # each model from its own first to final validation loss.
    shape=[]
    for panel in ("hplt","openarchives","greek_phd"):
        curves=[]
        for scale in ("1p5b","8b"):
            raw=[v for _,v in traj["scales"][scale]["validation_loss"][panel]]
            curves.append([(raw[0]-v)/(raw[0]-raw[-1]) for v in raw])
        shape.append(sum(abs(a-b) for a,b in zip(*curves))/len(curves[0]))

    evidence=[]
    for name in ("training_trajectories.json","1p5b_summary.json","8b_summary.json","historical_cpt_2arm_summary.json","hard_h_to_g_replication_v1.json"):
        p=EVIDENCE/name
        evidence.append(f'<tr><td><code>{name}</code></td><td><code>{sha256(p)}</code></td><td class="num">{p.stat().st_size:,}</td></tr>')

    document=f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Matched H→G CPT · 1.5B and 8B trajectories</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Source+Sans+3:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,500;8..60,650&display=swap');
:root{{--ink:{INK};--muted:{MUTED};--grid:{GRID};--paper:{PAPER};--white:{WHITE};--blue:{BLUE};--red:{RED};--teal:{TEAL};--gold:{GOLD};}}
*{{box-sizing:border-box}} html{{scroll-behavior:smooth}} body{{margin:0;background:var(--paper);color:var(--ink);font-family:'Source Sans 3',system-ui,sans-serif;line-height:1.48}}
.page{{max-width:1320px;margin:auto;padding:44px 48px 90px}} header{{border-top:8px solid var(--ink);padding-top:28px;margin-bottom:28px}}
.eyebrow{{font:500 12px 'IBM Plex Mono',monospace;letter-spacing:.13em;text-transform:uppercase;color:var(--teal)}} h1{{font:650 clamp(38px,5.2vw,72px)/.98 'Source Serif 4',serif;max-width:1050px;margin:12px 0 18px;letter-spacing:-.03em}}
.dek{{font-size:22px;max-width:970px;color:#405465;margin:0}} .meta{{display:flex;gap:20px;flex-wrap:wrap;margin-top:26px;font:500 12px 'IBM Plex Mono',monospace;color:var(--muted)}}
.verdict{{display:grid;grid-template-columns:1.25fr .75fr;gap:22px;margin:34px 0}} .callout{{background:var(--ink);color:var(--white);padding:28px 30px;border-radius:10px}} .callout h2{{font:650 31px/1.08 'Source Serif 4',serif;margin:0 0 12px}} .callout p{{font-size:18px;margin:0;color:#dbe4e8}}
.status{{background:#fff7e3;border:1px solid #d8b76b;border-left:8px solid var(--gold);padding:24px;border-radius:8px}} .status strong{{font:650 22px 'Source Serif 4',serif;display:block;margin-bottom:8px}} section{{margin-top:52px}}
.section-head{{display:grid;grid-template-columns:90px 1fr;gap:18px;border-top:1px solid #aeb7bd;padding-top:18px;margin-bottom:22px}} .section-no{{font:500 13px 'IBM Plex Mono',monospace;color:var(--teal)}} h2{{font:650 36px/1.1 'Source Serif 4',serif;margin:0}} .section-head p{{grid-column:2;margin:8px 0 0;max-width:930px;color:#526572;font-size:18px}}
.figure{{background:var(--white);border:1px solid #ded9d0;border-radius:10px;padding:18px 20px 12px;box-shadow:0 10px 26px rgba(31,51,65,.055);overflow:hidden}} .chart{{width:100%;height:auto;display:block}} figcaption{{margin:10px 8px 4px;color:var(--muted);font-size:14px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:22px}} .metric-card{{background:var(--white);border-top:5px solid var(--blue);padding:22px;border-radius:6px}} .metric-card.red{{border-color:var(--red)}} .metric-card .big{{font:650 40px 'Source Serif 4',serif}} .metric-card .small{{color:var(--muted)}}
.finding{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin:20px 0}} .finding article{{background:#edf3f1;border-left:5px solid var(--teal);padding:19px 20px}} .finding strong{{display:block;font:650 21px 'Source Serif 4',serif;margin-bottom:6px}}
.warning{{background:#fff4ee;border-left:7px solid var(--red);padding:22px 24px;margin:22px 0}} .warning strong{{font:650 22px 'Source Serif 4',serif;display:block;margin-bottom:7px}}
table{{width:100%;border-collapse:collapse;background:var(--white);font-size:14px}} th,td{{padding:11px 12px;border-bottom:1px solid #e0ddd5;text-align:left}} th{{background:#e9eef0;font-weight:650;vertical-align:bottom}} th span{{display:block;font-size:11px;color:var(--muted);font-weight:500}} td.num{{font-family:'IBM Plex Mono',monospace;text-align:right}} td.gain{{background:#e6f2ec;color:#176b54}} td.regress{{background:#f8e9e4;color:#a33e36}} code{{font:12px 'IBM Plex Mono',monospace;word-break:break-all}}
.method{{background:#e9edef;padding:24px 28px;border-radius:8px}} .method p{{margin:.5em 0}} footer{{border-top:1px solid #aeb7bd;margin-top:54px;padding-top:18px;color:var(--muted);font-size:13px}}
.axis,.note,.phase-note{{font:12px 'Source Sans 3',sans-serif;fill:var(--muted)}} .axis-strong{{font:600 12px 'Source Sans 3',sans-serif;fill:var(--ink)}} .panel-title{{font:650 18px 'Source Serif 4',serif;fill:var(--ink)}} .model-label{{font:650 17px 'Source Sans 3',sans-serif;fill:var(--ink)}} .value-small{{font:700 14px 'IBM Plex Mono',monospace}} .phase-label{{font:650 16px 'Source Sans 3',sans-serif;fill:white}} .phase-small{{font:500 12px 'IBM Plex Mono',monospace;fill:white}} .phase-note{{font-weight:650;font-size:11px}}
@media(max-width:760px){{.page{{padding:26px 18px 70px}} .verdict,.grid2,.finding{{grid-template-columns:1fr}} .section-head{{grid-template-columns:58px 1fr}} h2{{font-size:29px}} .dek{{font-size:18px}} .figure{{padding:8px 5px}} table{{font-size:12px}} th,td{{padding:8px 7px}} .wide-table{{overflow-x:auto}}}}
</style></head><body><main class="page">
<header><div class="eyebrow">Hard HPLT → OpenArchives · cross-scale evidence · 22 August 2026</div>
<h1>Compare the trajectories, not the endpoint heights</h1>
<p class="dek">Apertus 1.5B and 8B followed the same 15.494B-token curriculum. Their heldout-loss curves show strongly aligned learning dynamics. The completed GreekMMLU evidence, however, contains only one checkpoint per scale—so it cannot yet establish whether the smaller model mirrors the larger model's benchmark trajectory.</p>
<div class="meta"><span>3,694 updates</span><span>4.194M token slots/update</span><span>148 validation measurements/panel/scale</span><span>16,159 clean GreekMMLU questions at final only</span></div></header>

<div class="verdict"><div class="callout"><h2>What the existing trajectories do show</h2><p>The cross-scale response to the data order is similar: HPLT loss bottoms at the HPLT→OA switch, OpenArchives and Greek-PhD loss fall sharply during the OA phase, and both continue improving in the extension. After normalization within each model, mean curve disagreement is only {sum(shape)/len(shape)*100:.2f}% of the observed improvement span.</p></div>
<div class="status"><strong>GreekMMLU mirroring: unanswered</strong>Final clean accuracy is {pct(o1['accuracy'])} at 1.5B and {pct(o8['accuracy'])} at 8B. These are comparable endpoints, not trajectories. No intermediate clean/sentinel scores completed.</div></div>

<section><div class="section-head"><div class="section-no">01 · EXPOSURE</div><div><h2>One curriculum, one learning-rate path</h2><p>Both models saw the same token schedule. Phase boundaries are therefore causal landmarks shared by both trajectories, rather than post-hoc alignment points.</p></div></div>
<figure class="figure">{schedule_lr_svg(traj)}<figcaption>Learning-rate values are read directly from the completed TensorBoard event streams, not reconstructed from prose.</figcaption></figure></section>

<section><div class="section-head"><div class="section-no">02 · SHAPE</div><div><h2>The learning curves bend together</h2><p>Each curve below is normalized against its own first and final measured loss. This removes the expected scale advantage of 8B and compares the temporal shape—the scientific question the endpoint bars obscured.</p></div></div>
<figure class="figure">{normalized_learning_svg(traj)}<figcaption>0% is each model's loss at update 25; 100% is its loss at update 3,675. Curve-shape mean absolute gaps: HPLT {shape[0]*100:.2f}%, OpenArchives {shape[1]*100:.2f}%, Greek PhD {shape[2]*100:.2f}% of each observed improvement span.</figcaption></figure>
<div class="finding"><article><strong>HPLT phase</strong>Both scales learn broad Greek quickly and approach their HPLT minima at the phase boundary.</article><article><strong>OpenArchives phase</strong>Both scales trade a small HPLT regression for a large curated-Greek gain.</article><article><strong>Extension</strong>OA and Greek-PhD loss keep falling; HPLT stabilizes or recovers rather than continuing to drift upward.</article></div></section>

<section><div class="section-head"><div class="section-no">03 · LEVEL</div><div><h2>Absolute loss still separates model capacity</h2><p>The normalized view answers whether the response shapes align. The absolute view confirms the expected capacity difference: 8B remains lower-loss throughout on all three Greek panels.</p></div></div>
<figure class="figure">{absolute_learning_svg(traj)}<figcaption>Same tokenizer, panels and objective at both scales; absolute loss is therefore directly readable, though not the right measure of trajectory mirroring.</figcaption></figure>
<div class="wide-table">{key_table(traj)}</div>
<h3>Phase-local changes</h3><div class="wide-table">{phase_delta_table(traj)}</div></section>

<section><div class="section-head"><div class="section-no">04 · OPTIMIZATION</div><div><h2>The dense training traces tell the same phase story</h2><p>Both optimization losses fall through HPLT, step down when OpenArchives begins, and rise modestly across the extension blend. The synchronized bends are schedule-driven; the vertical gap is model capacity.</p></div></div>
<figure class="figure">{training_loss_svg(traj)}<figcaption>Centered 71-update moving average, sampled every eight updates for display. All 3,694 underlying observations per scale are retained in the evidence JSON.</figcaption></figure></section>

<section><div class="section-head"><div class="section-no">05 · RETENTION</div><div><h2>Forgetting is real, but not monotone through the extension</h2><p>Loss above each panel's own best-so-far value isolates forgetting from absolute model quality. Both scales move away from early minima on foreign and code panels; the extension partially reverses several trajectories.</p></div></div>
<figure class="figure">{forgetting_svg(traj)}<figcaption>Online validation panels are shown exactly as trained. The inherited Old-Greek panel is omitted from this forgetting figure because it is not a reliable independent retention panel.</figcaption></figure></section>

<section><div class="section-head"><div class="section-no">06 · GREEKMMLU</div><div><h2>The benchmark trajectory was not measured</h2><p>The frozen design asked for 18 checkpoint positions and a shared sentinel/full-panel decision. The panels were materialized, but the intermediate scoring jobs did not complete. A line joining the two final model scores would be a scale comparison, not a trajectory.</p></div></div>
<figure class="figure">{greekmmlu_coverage_svg(o1,o8)}<figcaption>This is the critical correction to the previous report: absence is shown explicitly instead of replacing a trajectory analysis with endpoint bars.</figcaption></figure>
<div class="grid2"><div class="metric-card red"><div class="eyebrow">Apertus 1.5B · final</div><div class="big">{pct(o1['accuracy'])}</div><div class="small">{o1['correct']:,}/{o1['n']:,} correct · choice NLL {o1['choice_nll']:.4f} · correct-answer BPB {o1['correct_answer_bpb']:.4f}</div></div>
<div class="metric-card"><div class="eyebrow">Apertus 8B · final</div><div class="big">{pct(o8['accuracy'])}</div><div class="small">{o8['correct']:,}/{o8['n']:,} correct · choice NLL {o8['choice_nll']:.4f} · correct-answer BPB {o8['correct_answer_bpb']:.4f}</div></div></div>
<div class="warning"><strong>Conclusion boundary</strong>We can say the smaller model mirrors the larger model on source-conditioned loss dynamics. We cannot say it mirrors the larger model on GreekMMLU peak timing, plateau width, or checkpoint-to-checkpoint change until the same frozen benchmark panel is scored at intermediate checkpoints.</div></section>

<section><div class="section-head"><div class="section-no">07 · HISTORY</div><div><h2>The predecessor 8B run had the trajectory we are missing here</h2><p>A surviving earlier TD study shows GreekMMLU rising over training with local reversals. It explains why a single final point is inadequate, but it cannot be numerically merged with the current clean-panel results.</p></div></div>
<figure class="figure">{historical_svg(hist)}<figcaption>Historical public-panel trajectory, n=16,632. Current results use the decontaminated n=16,159 panel and a distinct corpus realization. The later β₂-selected 59.94% result survives as an endpoint, not a complete per-question trajectory.</figcaption></figure></section>

<section><div class="section-head"><div class="section-no">08 · EVIDENCE</div><div><h2>What is measured, and what is not</h2><p>The report is self-contained and generated from copied immutable evidence. No score is inferred from a plot and no missing checkpoint is interpolated.</p></div></div>
<div class="method"><p><strong>Measured:</strong> complete training loss and LR at updates 1–3,694; nine online validation panels every 25 updates from 25–3,675; final clean GreekMMLU predictions and aggregate metrics for both scales.</p><p><strong>Not measured:</strong> current-run intermediate GreekMMLU accuracy, choice NLL or BPB; therefore no current cross-scale GreekMMLU shape correlation or peak-location comparison.</p><p><strong>Normalization:</strong> for model m and panel s, displayed learning progress is (L<sub>m,s</sub>(25) − L<sub>m,s</sub>(t)) / (L<sub>m,s</sub>(25) − L<sub>m,s</sub>(3675)). This compares shape without pretending the two models should have the same absolute loss.</p></div>
<div class="wide-table"><table><thead><tr><th>evidence file</th><th>SHA-256</th><th>bytes</th></tr></thead><tbody>{''.join(evidence)}</tbody></table></div></section>

<footer>Generated 22 August 2026 · Apertus hard HPLT→OpenArchives cross-scale study · trajectory-first revision</footer>
</main></body></html>"""
    OUTPUT.write_text(document,encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
