#!/usr/bin/env python3
"""Build a single-page academic report for full-panel and stable-LR evidence."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
OUTPUT = ROOT / "HARD_H2G_FULL_PANEL_AND_STABLE_LR_20260822.html"
TOKENS = 4_194_304
INK, MUTED, GRID, PAPER, WHITE = "#152b3c", "#687985", "#d9d5cd", "#f5f1e8", "#fffdf8"
BLUE, RED, TEAL, GOLD, PURPLE, GREEN = "#276a9f", "#c84b42", "#21857a", "#c6922e", "#7b5aa6", "#447a55"


def load() -> dict:
    return json.loads((EVIDENCE / "analysis.json").read_text(encoding="utf-8"))


def esc(value) -> str:
    return html.escape(str(value))


def pct(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}%"


def pp(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f} pp"


def tokens_b(update: int) -> float:
    return update * TOKENS / 1e9


def svg_head(width: int, height: int, title: str, desc: str) -> list[str]:
    ident = hashlib.sha1(title.encode()).hexdigest()[:10]
    return [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-labelledby="t{ident} d{ident}">',
        f'<title id="t{ident}">{esc(title)}</title><desc id="d{ident}">{esc(desc)}</desc>',
    ]


def path(points, x, y) -> str:
    return " ".join(("M" if i == 0 else "L") + f" {x(a):.2f} {y(b):.2f}" for i, (a, b) in enumerate(points))


def phase_bands(out: list[str], x, top: float, bottom: float, end: int = 3694) -> None:
    phases = [(0, 2261, BLUE, "HPLT"), (2261, 3218, TEAL, "OpenArchives"), (3218, end, GOLD, "extension")]
    for start, stop, color, label in phases:
        if start >= end:
            continue
        stop = min(stop, end)
        x0, x1 = x(start), x(stop)
        out.append(f'<rect x="{x0:.2f}" y="{top}" width="{x1-x0:.2f}" height="{bottom-top}" fill="{color}" opacity=".055"/>')
        out.append(f'<text x="{(x0+x1)/2:.2f}" y="{top+17}" text-anchor="middle" class="phase" fill="{color}">{label}</text>')
    for update in (2261, 3218):
        if update <= end:
            out.append(f'<line x1="{x(update):.2f}" y1="{top}" x2="{x(update):.2f}" y2="{bottom}" stroke="{INK}" stroke-dasharray="7 6" opacity=".45"/>')


def lr_svg(data: dict) -> str:
    width, height = 1180, 410
    left, right, top, bottom = 78, 34, 46, 68
    x = lambda u: left + u / 3694 * (width-left-right)
    ymax = 5.9e-5
    y = lambda v: top + (ymax-v)/ymax*(height-top-bottom)
    out = svg_head(width, height, "Learning-rate intervention geometry", "The arms share the original schedule through update 2499. The branch holds peak learning rate to update 3218 while the original arm decays.")
    phase_bands(out, x, top, height-bottom)
    for value in (0, 1e-5, 3e-5, 5.5e-5):
        yy = y(value)
        out.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="{GRID}"/>')
        out.append(f'<text x="{left-10}" y="{yy+4:.1f}" text-anchor="end" class="axis">{value*1e5:.1f}</text>')
    decay = [(int(u), float(v)) for u, v in data["learning_rate"]["decayed_full"]]
    out.append(f'<path d="{path(decay,x,y)}" fill="none" stroke="{PURPLE}" stroke-width="3.5"/>')
    stable = [(2499, 5.5e-5), (3218, 5.5e-5)]
    out.append(f'<path d="{path(stable,x,y)}" fill="none" stroke="{GREEN}" stroke-width="4.2" stroke-dasharray="10 5"/>')
    out.append(f'<line x1="{x(2499):.1f}" y1="{top}" x2="{x(2499):.1f}" y2="{height-bottom}" stroke="{RED}" stroke-width="1.6"/>')
    out.append(f'<text x="{x(2499)+7:.1f}" y="{top+38}" class="direct" fill="{RED}">branch · u2499</text>')
    out.append(f'<text x="{x(2820):.1f}" y="{y(5.5e-5)-12:.1f}" class="direct" fill="{GREEN}">stable 5.5×10⁻⁵</text>')
    out.append(f'<text x="{x(3040):.1f}" y="{y(2.1e-5):.1f}" class="direct" fill="{PURPLE}">WSD decay</text>')
    for update in (0, 952, 1904, 2261, 2499, 3218, 3694):
        out.append(f'<text x="{x(update):.1f}" y="{height-31}" text-anchor="middle" class="axis">{tokens_b(update):.1f}B</text>')
    out.append(f'<text x="{left}" y="{top-17}" class="panel-title">learning rate ×10⁻⁵</text>')
    out.append(f'<text x="{(left+width-right)/2:.1f}" y="{height-7}" text-anchor="middle" class="axis-strong">consumed token slots</text></svg>')
    return "".join(out)


def full_score_svg(data: dict) -> str:
    rows = data["decayed_full_panel"]
    points = [(r["update"], r["accuracy"]) for r in rows]
    width, height = 1180, 520
    left, right, top, bottom = 82, 42, 48, 72
    lo, hi = .485, .59
    x = lambda u: left + u/3694*(width-left-right)
    y = lambda v: top + (hi-v)/(hi-lo)*(height-top-bottom)
    out = svg_head(width, height, "Full public GreekMMLU trajectory", "All seventeen existing 8B checkpoints scored in float32 on all 16632 public GreekMMLU questions.")
    phase_bands(out, x, top, height-bottom)
    for value in (.50, .52, .54, .56, .58):
        yy=y(value); out.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="{GRID}"/>')
        out.append(f'<text x="{left-11}" y="{yy+4:.1f}" text-anchor="end" class="axis">{pct(value,0)}</text>')
    out.append(f'<path d="{path(points,x,y)}" fill="none" stroke="{BLUE}" stroke-width="4"/>')
    for u,v in points:
        out.append(f'<circle cx="{x(u):.1f}" cy="{y(v):.1f}" r="5" fill="{WHITE}" stroke="{BLUE}" stroke-width="3"/>')
    peak=max(rows,key=lambda r:r["accuracy"])
    out.append(f'<line x1="{x(peak["update"]):.1f}" y1="{y(peak["accuracy"]):.1f}" x2="{x(peak["update"])+74:.1f}" y2="{y(peak["accuracy"])-42:.1f}" stroke="{RED}"/>')
    out.append(f'<text x="{x(peak["update"])+80:.1f}" y="{y(peak["accuracy"])-45:.1f}" class="callout" fill="{RED}">peak {pct(peak["accuracy"],2)} · u{peak["update"]}</text>')
    for update in (0, 952, 1904, 2261, 2618, 3218, 3694):
        out.append(f'<text x="{x(update):.1f}" y="{height-32}" text-anchor="middle" class="axis">{tokens_b(update):.1f}B</text>')
    out.append(f'<text x="{left}" y="{top-18}" class="panel-title">accuracy · 16,632 questions</text>')
    out.append(f'<text x="{(left+width-right)/2:.1f}" y="{height-7}" text-anchor="middle" class="axis-strong">consumed token slots</text></svg>')
    return "".join(out)


def paired_score_svg(data: dict) -> str:
    rows=data["paired"]
    width,height=1180,500; left,right,top,bottom=86,46,48,76
    lo=min(min(r["stable"]["accuracy"],r["decayed"]["accuracy"]) for r in rows)-.004
    hi=max(max(r["stable"]["accuracy"],r["decayed"]["accuracy"]) for r in rows)+.004
    x=lambda u:left+(u-2499)/(3218-2499)*(width-left-right)
    y=lambda v:top+(hi-v)/(hi-lo)*(height-top-bottom)
    out=svg_head(width,height,"Paired stable-LR versus decayed trajectory","The same examples are evaluated at the same data cursors; only the learning-rate schedule differs after update 2499.")
    for k in range(6):
        val=lo+(hi-lo)*k/5; yy=y(val)
        out.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="{GRID}"/>')
        out.append(f'<text x="{left-10}" y="{yy+4:.1f}" text-anchor="end" class="axis">{pct(val,1)}</text>')
    for key,color,label,dash in (("stable",GREEN,"stable LR","10 5"),("decayed",PURPLE,"WSD decay","")):
        pts=[(r["update"],r[key]["accuracy"]) for r in rows]
        dash_attr=f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<path d="{path(pts,x,y)}" fill="none" stroke="{color}" stroke-width="4"{dash_attr}/>')
        for u,v in pts:
            out.append(f'<circle cx="{x(u):.1f}" cy="{y(v):.1f}" r="6" fill="{WHITE}" stroke="{color}" stroke-width="3"/>')
        u,v=pts[-1]; out.append(f'<text x="{x(u)-7:.1f}" y="{y(v)+(20 if key=="decayed" else -11):.1f}" text-anchor="end" class="direct" fill="{color}">{label} {pct(v,2)}</text>')
    for r in rows:
        out.append(f'<text x="{x(r["update"]):.1f}" y="{height-39}" text-anchor="middle" class="axis-strong">u{r["update"]}</text>')
        out.append(f'<text x="{x(r["update"]):.1f}" y="{height-21}" text-anchor="middle" class="axis">{tokens_b(r["update"]):.2f}B</text>')
    out.append(f'<text x="{left}" y="{top-17}" class="panel-title">full-public accuracy · zoomed y-axis {pct(lo,1)}–{pct(hi,1)}</text></svg>')
    return "".join(out)


def continuous_metrics_svg(data: dict) -> str:
    rows=data["decayed_full_panel"]
    panels=[("choice_nll","choice NLL"),("correct_answer_bpb","correct-answer BPB")]
    width,height=1180,480; left,right,top,bottom=82,36,58,64; gap=54
    pw=(width-left-right-gap)/2; ph=height-top-bottom
    out=svg_head(width,height,"Continuous full-panel metrics","Choice negative log likelihood and correct-answer bits per byte across all seventeen 8B checkpoints; lower is better.")
    for i,(metric,label) in enumerate(panels):
        x0=left+i*(pw+gap); vals=[float(r[metric]) for r in rows]; lo=min(vals); hi=max(vals); pad=(hi-lo)*.09; lo-=pad; hi+=pad
        x=lambda u,a=x0:a+u/3694*pw; y=lambda v:top+(hi-v)/(hi-lo)*ph
        phase_bands(out,x,top,top+ph)
        for k in range(5):
            value=lo+(hi-lo)*k/4; yy=y(value)
            out.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0+pw}" y2="{yy:.1f}" stroke="{GRID}"/>')
            out.append(f'<text x="{x0-8}" y="{yy+4:.1f}" text-anchor="end" class="axis">{value:.3f}</text>')
        pts=[(r["update"],float(r[metric])) for r in rows]
        out.append(f'<path d="{path(pts,x,y)}" fill="none" stroke="{BLUE if i==0 else TEAL}" stroke-width="3.4"/>')
        for u,v in pts: out.append(f'<circle cx="{x(u):.1f}" cy="{y(v):.1f}" r="4" fill="{WHITE}" stroke="{BLUE if i==0 else TEAL}" stroke-width="2.3"/>')
        best=min(rows,key=lambda r:float(r[metric])); out.append(f'<text x="{x(best["update"]):.1f}" y="{y(float(best[metric]))-11:.1f}" text-anchor="middle" class="direct" fill="{BLUE if i==0 else TEAL}">best u{best["update"]} · {float(best[metric]):.3f}</text>')
        out.append(f'<text x="{x0}" y="{top-21}" class="panel-title">{label} · lower is better</text>')
        for u in (0,2261,3218,3694): out.append(f'<text x="{x(u):.1f}" y="{height-28}" text-anchor="middle" class="axis">{tokens_b(u):.1f}B</text>')
    out.append('</svg>')
    return "".join(out)


def loss_grid_svg(data: dict) -> str:
    roles=[("hplt","HPLT · retention"),("openarchives","OpenArchives · adaptation"),("greek_phd","Greek PhD · transfer"),
           ("english","English"),("de","German"),("ru","Russian"),("zh","Chinese"),("code","Code"),("old_greek","Old Greek")]
    width,height=1180,1120; cols=3; rows=3; left,right,top,bottom=68,25,58,55; gx,gy=28,42
    pw=(width-left-right-gx*(cols-1))/cols; ph=(height-top-bottom-gy*(rows-1))/rows
    out=svg_head(width,height,"All nine validation-loss trajectories","Absolute heldout loss over the complete original trajectory, with the stable-LR branch overlaid from update 2499 through 3218.")
    for i,(panel,label) in enumerate(roles):
        col,row=i%cols,i//cols; x0=left+col*(pw+gx); y0=top+row*(ph+gy)
        full=[(int(u),float(v)) for u,v in data["validation"]["decayed_full"][panel]]
        branch_s=[(int(u),float(v)) for u,v in data["validation"]["paired_branch"][panel]["stable"]]
        branch_d=[(int(u),float(v)) for u,v in data["validation"]["paired_branch"][panel]["decayed"]]
        vals=[v for _,v in full]+[v for _,v in branch_s]; lo=min(vals); hi=max(vals); pad=max((hi-lo)*.09,.002); lo-=pad; hi+=pad
        x=lambda u,a=x0:a+u/3694*pw; y=lambda v,a=y0:a+(hi-v)/(hi-lo)*ph
        phase_bands(out,x,y0,y0+ph)
        for k in range(4):
            value=lo+(hi-lo)*k/3; yy=y(value)
            out.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0+pw}" y2="{yy:.1f}" stroke="{GRID}"/>')
            out.append(f'<text x="{x0-7}" y="{yy+4:.1f}" text-anchor="end" class="axis">{value:.3f}</text>')
        out.append(f'<text x="{x0}" y="{y0-19}" class="panel-title">{esc(label)}</text>')
        out.append(f'<path d="{path(full,x,y)}" fill="none" stroke="{PURPLE}" stroke-width="2.4"/>')
        out.append(f'<path d="{path(branch_s,x,y)}" fill="none" stroke="{GREEN}" stroke-width="3.4" stroke-dasharray="9 4"/>')
        out.append(f'<path d="{path(branch_d,x,y)}" fill="none" stroke="{PURPLE}" stroke-width="3.4"/>')
        if row==rows-1:
            for u in (0,2261,3218,3694): out.append(f'<text x="{x(u):.1f}" y="{y0+ph+18}" text-anchor="middle" class="axis">{tokens_b(u):.1f}</text>')
    out.append(f'<line x1="{width-320}" y1="24" x2="{width-286}" y2="24" stroke="{PURPLE}" stroke-width="4"/><text x="{width-279}" y="29" class="note">decayed</text>')
    out.append(f'<line x1="{width-188}" y1="24" x2="{width-154}" y2="24" stroke="{GREEN}" stroke-width="4" stroke-dasharray="9 4"/><text x="{width-147}" y="29" class="note">stable branch</text>')
    out.append('</svg>')
    return "".join(out)


def table(rows: list[list[str]], headers: list[str]) -> str:
    return '<div class="wide-table"><table><thead><tr>'+''.join(f'<th>{h}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{cell}</td>' for cell in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def main() -> None:
    data=load(); decayed={r["update"]:r for r in data["decayed_full_panel"]}; stable={r["update"]:r for r in data["stable_full_panel"]}
    intervals=data["stable_trajectory"]["interval_accuracy_change_pp"]
    last=intervals[-1]; noise=data["stable_trajectory"]["noise_reference_pp"]
    if last > noise:
        direction="materially rising"
    elif last > 0:
        direction="numerically rising, but by less than the noise reference"
    else:
        direction="flat or falling"
    primary=(f"The stable-LR arm is {direction} at the endpoint: its last interval changed by {last:+.3f} pp, "
             f"against the predeclared ≈0.4 pp checkpoint noise reference.")
    paired_rows=[]
    for r in data["paired"]:
        paired_rows.append([f'u{r["update"]}',f'{tokens_b(r["update"]):.3f}B',pct(r["decayed"]["accuracy"],3),pct(r["stable"]["accuracy"],3),pp(r["stable_minus_decayed_accuracy_pp"]),f'{r["stable"]["choice_nll"]:.5f}',f'{r["stable"]["correct_answer_bpb"]:.5f}'])
    legacy_rows=[]
    for r in data["legacy_replication"]["results"]:
        legacy_rows.append([f'u{r["update"]}',f'{r["correct"]:,}/{r["n"]:,}',pct(r["accuracy"],3)])
    endpoint_loss=[]
    for panel,values in data["validation"]["paired_branch"].items():
        endpoint_loss.append([panel,f'{values["decayed"][-1][1]:.6f}',f'{values["stable"][-1][1]:.6f}',f'{values["endpoint_stable_minus_decayed"]:+.6f}'])
    source_rows=[]
    analysis_path=EVIDENCE/"analysis.json"
    for name,binding in data["sources"].items(): source_rows.append([esc(name),f'<code>{esc(binding["path"])}</code>',f'<code>{binding["sha256"]}</code>',f'{binding["bytes"]:,}'])
    source_rows.insert(0,["analysis",'<code>evidence/analysis.json</code>',f'<code>{hashlib.sha256(analysis_path.read_bytes()).hexdigest()}</code>',f'{analysis_path.stat().st_size:,}'])
    allocation_rows = [
        ["3151839", "4 / 16", "3:28:39", "55.64", "A1/A2 + early branch scoring"],
        ["3152592", "16 / 64", "2:52:29", "183.98", "B2 training, finalization, exports"],
        ["3153569", "4 / 16", "0:00:52", "0.23", "mis-shaped request; relinquished"],
        ["3153706", "4 / 16", "0:14:54", "3.97", "paired endpoint scoring"],
    ]
    css=f"""
    :root{{--ink:{INK};--muted:{MUTED};--grid:{GRID};--paper:{PAPER};--white:{WHITE};--blue:{BLUE};--red:{RED};--teal:{TEAL};--gold:{GOLD};--purple:{PURPLE};--green:{GREEN}}}
    *{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}} main,header,footer{{width:min(1280px,calc(100% - 42px));margin:auto}} header{{padding:62px 0 34px;border-bottom:1px solid var(--grid)}} .eyebrow,.kicker{{text-transform:uppercase;letter-spacing:.14em;font-size:.76rem;font-weight:750;color:var(--teal)}} h1,h2{{font-family:Georgia,serif;font-weight:500;line-height:1.08}} h1{{font-size:clamp(2.5rem,6vw,5.4rem);max-width:1100px;margin:.12em 0}} h2{{font-size:clamp(1.8rem,3.5vw,3.15rem);margin:.15em 0 .42em}} h3{{font:700 1rem/1.3 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}} .deck{{max-width:920px;font:1.25rem/1.5 Georgia,serif;color:#344b5b}} section{{padding:48px 0;border-bottom:1px solid var(--grid)}} .section-no{{color:var(--red);font-weight:800;letter-spacing:.13em}} .finding{{border-left:5px solid var(--teal);padding:17px 22px;background:rgba(255,253,248,.7);font:1.35rem/1.48 Georgia,serif}} .chart{{display:block;width:100%;height:auto;background:var(--white);border:1px solid var(--grid);box-shadow:0 8px 24px rgba(21,43,60,.07);margin:20px 0 10px}} figcaption,.note{{color:var(--muted);font-size:.88rem}} .axis{{font-size:12px;fill:{MUTED}}}.axis-strong{{font-size:12px;font-weight:700;fill:{INK}}}.phase{{font-size:11px;font-weight:800;letter-spacing:.08em}}.panel-title{{font-size:14px;font-weight:800;fill:{INK}}}.direct{{font-size:13px;font-weight:800}}.callout{{font-size:14px;font-weight:800}} .two{{display:grid;grid-template-columns:1fr 1fr;gap:24px}} .contract{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}} .contract div{{padding:15px;border-top:3px solid var(--blue);background:rgba(255,253,248,.65)}} .contract strong{{display:block;font-family:Georgia,serif;font-size:1.3rem}} .wide-table{{overflow-x:auto;background:var(--white);border:1px solid var(--grid)}} table{{border-collapse:collapse;width:100%;font-size:.9rem}} th,td{{padding:10px 12px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}} th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}} th{{background:#eae6dc;text-transform:uppercase;letter-spacing:.05em;font-size:.72rem}} code{{font-size:.76rem;white-space:normal;overflow-wrap:anywhere}} .evidence-table table{{min-width:940px}} .evidence-table code{{white-space:nowrap;overflow-wrap:normal}} .warn{{border:1px solid #d7b77c;background:#fff7e5;padding:18px}} .stop{{border-left:5px solid var(--red);padding:18px 22px;background:#fff7f3}} footer{{padding:34px 0 60px;color:var(--muted)}} @media(max-width:720px){{main,header,footer{{width:min(100% - 22px,1280px)}}header{{padding-top:34px}}section{{padding:32px 0}}.two,.contract{{grid-template-columns:1fr}}figure{{margin-inline:0}}h1{{font-size:2.55rem}}.deck{{font-size:1.05rem}}.axis,.axis-strong{{font-size:14px}}.panel-title,.direct,.callout{{font-size:16px}}}} @media print{{body{{background:white}}.chart{{box-shadow:none}}section{{break-inside:avoid}}}}
    """
    html_out=f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Hard H→G · full panel and stable LR</title><style>{css}</style></head><body>
    <header><div class="eyebrow">Apertus 8B · controlled branch experiment · 22 August 2026</div><h1>Did cooldown create the GreekMMLU peak?</h1><p class="deck">A full 16,632-question rescore of the Hard H→G trajectory, plus a matched branch that holds peak learning rate after update 2,499. Same model state, data cursor, examples and optimizer recipe; only LR decay changes.</p><p class="finding">{esc(primary)}</p></header><main>
    <section><div class="section-no">01 · CONTRACT</div><h2>One intervention, paired at the same data cursors</h2><div class="contract"><div><span class="kicker">branch</span><strong>u2,499</strong>last saved pre-decay state</div><div><span class="kicker">stable arm</span><strong>5.5×10⁻⁵</strong>constant through u3,218</div><div><span class="kicker">evaluation</span><strong>16,632</strong>full public GreekMMLU</div><div><span class="kicker">integrity</span><strong>0 / 0</strong>skipped / non-finite</div></div><figure>{lr_svg(data)}<figcaption>Logged WSD curve; stable branch is the realized constant-LR intervention. Phase boundaries and the exact branch point are shown.</figcaption></figure></section>
    <section><div class="section-no">02 · PRIMARY TRAJECTORY</div><h2>The corrected full-panel curve peaks in early decay</h2><figure>{full_score_svg(data)}<figcaption>All 17 existing 8B checkpoints, FP32 common evaluator. Primary trajectory statistics are interpreted through u3,218; the post-endpoint extension remains LR-confounded sensitivity evidence.</figcaption></figure><figure>{continuous_metrics_svg(data)}<figcaption>Continuous metrics use the same questions and scorer. They remain informative when accuracy changes are small or cross a decision boundary.</figcaption></figure><div class="two"><p class="finding">Peak: <strong>{pct(decayed[2618]["accuracy"],3)}</strong> at u2,618. Endpoint u3,218: <strong>{pct(decayed[3218]["accuracy"],3)}</strong>, a {pp(100*(decayed[3218]["accuracy"]-decayed[2618]["accuracy"]))} change.</p><p class="warn">The older 16,159-question clean-subset cross-scale report remains a sensitivity analysis. It is not silently combined with this full-public 8B panel.</p></div></section>
    <section><div class="section-no">03 · CAUSAL BRANCH</div><h2>Holding peak LR does not restore a rising endpoint trajectory</h2><figure>{paired_score_svg(data)}<figcaption>Zoomed paired comparison; raw points are retained. The y-axis is intentionally narrow and labeled. A level advantage is not treated as proof that no-decay is superior—the preregistered question is the stable arm's own slope.</figcaption></figure>{table(paired_rows,["checkpoint","tokens","decayed acc.","stable acc.","stable−decay","stable choice NLL","stable BPB"])}<p class="stop"><strong>B3 gate:</strong> {esc(primary)} The extension 3,219→3,694 remains unauthorized and was not launched.</p></section>
    <section><div class="section-no">04 · LEARNING & FORGETTING</div><h2>All validation panels, across the complete horizon</h2><figure>{loss_grid_svg(data)}<figcaption>Absolute loss, lower is better. Purple is the original decayed run; dashed green exists only after the branch. Full-horizon context is primary, with the branch difference visible without discarding earlier learning and forgetting history.</figcaption></figure><h3>Branch endpoint loss at u3,200</h3>{table(endpoint_loss,["panel","decayed","stable","stable−decay"])}<p class="note">Positive stable−decay means higher loss for stable LR. These are deterministic heldout trajectories, not confidence intervals.</p></section>
    <section><div class="section-no">05 · LEGACY REPLICATION</div><h2>The historical GreekMMLU peak did not replicate within the frozen band</h2><div class="two"><div>{table(legacy_rows,["checkpoint","correct","legacy BF16 accuracy"])} </div><p class="finding">Historical selected β₂ target: <strong>59.963%</strong>. Current best like-for-like result: <strong>57.900%</strong>. Delta: <strong>−2.062 pp</strong>, outside the preregistered ±1.0 pp band. The band was not widened after observing the result.</p></div><p class="note">Pinned evaluator revision cfdd0e7b; BF16; max input 3,072; candidate and example batch 16; all 16,632 questions.</p></section>
    <section><div class="section-no">06 · INTERPRETATION</div><h2>What the experiment does—and does not—show</h2><div class="two"><div><h3>Supported</h3><p>The full-panel peak is at u2,618, just after decay begins. Stable LR modestly changes the level relative to WSD at first, but the branch's own GreekMMLU accuracy falls at every paired checkpoint. That weakens the hypothesis that cooldown alone caused the peak.</p></div><div><h3>Not supported</h3><p>No general claim that WSD is optimal: one data-order seed and one constant-LR branch were tested. The final 1.353 pp separation exceeds the ≈0.4 pp checkpoint noise reference, but it does not identify the best alternative cooldown. Training/validation loss cannot substitute for the benchmark trajectory.</p></div></div><div class="warn"><strong>Parity scope:</strong> every export passed exact tensor mapping, but runtime semantic parity remained an explicit warning. Results are suitable for the frozen within-trajectory comparison used here; the caveat is not erased.</div></section>
    <section><div class="section-no">07 · EVIDENCE</div><h2>Frozen inputs and reviewer trail</h2><div class="evidence-table">{table(source_rows,["evidence","path","SHA-256","bytes"])}</div><h3>Allocated compute ledger</h3>{table(allocation_rows,["job","nodes / GPUs","elapsed","GPU-h","role"])}<p>The append-only execution narrative is <code>HARD_H2G_FULL_PANEL_AND_NO_DECAY_EXECUTION_NOTES_20260822.md</code>. Reusable operational findings are filed as efficiency issues #146, #147, #149, #150, #151 and #152; the rank-zero full-geometry wrapper is PR #148.</p></section>
    </main><footer>Generated from frozen receipts and raw logs · self-contained single-page academic report · no checkpoint averaging · no B3 launch</footer></body></html>"""
    OUTPUT.write_text(html_out,encoding="utf-8")
    print(OUTPUT)


if __name__=="__main__": main()
