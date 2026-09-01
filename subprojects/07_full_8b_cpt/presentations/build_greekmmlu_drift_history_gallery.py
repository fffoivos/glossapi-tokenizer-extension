#!/usr/bin/env python3
"""Build a visual questionnaire-history gallery for drift archetypes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from build_greekmmlu_drift_simulation_report import (
    excess_matrix,
    hamming_matrix,
    hdi_from_distance,
    permutation_orders,
)


HERE = Path(__file__).resolve().parent
STAMP = "20260811"
OUT_HTML = HERE / f"GREEKMMLU_DRIFT_HISTORY_GALLERY_{STAMP}.html"
OUT_JSON = HERE / f"GREEKMMLU_DRIFT_HISTORY_GALLERY_{STAMP}.data.json"
N = 60
T = 22


@dataclass(frozen=True)
class Case:
    key: str
    title: str
    family: str
    note: str
    ability: str
    transform: str


CASES = [
    Case("calibration_0", "Minimum drift", "Calibration ladder", "No answers are replaced; SRD is exactly zero.", "calibration", "calibration_0"),
    Case("calibration_20", "Twenty-percent drift", "Calibration ladder", "Six correct/wrong pairs are progressively exchanged at constant accuracy.", "calibration", "calibration_20"),
    Case("calibration_40", "Forty-percent drift", "Calibration ladder", "Twelve correct/wrong pairs are progressively exchanged at constant accuracy.", "calibration", "calibration_40"),
    Case("calibration_60", "Sixty-percent drift", "Calibration ladder", "Eighteen correct/wrong pairs are progressively exchanged at constant accuracy.", "calibration", "calibration_60"),
    Case("calibration_80", "Eighty-percent drift", "Calibration ladder", "Twenty-four correct/wrong pairs are progressively exchanged at constant accuracy.", "calibration", "calibration_80"),
    Case("calibration_100", "Maximum drift", "Calibration ladder", "All thirty correct/wrong pairs are progressively exchanged; the final state is the exact complement.", "calibration", "calibration_100"),
    Case("slide_one_way", "A · One-way sliding window", "Original sliding-window examples", "The first 10% is wrong; the six-question window moves down by two every checkpoint.", "window", "slide_one_way"),
    Case("slide_periodic", "B · Five down, five up", "Original sliding-window examples", "The same window moves down five times, back up five times, and repeats.", "window", "slide_periodic"),
    Case("slide_quasi", "C · Six down, seven up, five down, three up", "Original sliding-window examples", "The direction is recurrent but the excursions have unequal lengths.", "window", "slide_quasi"),
    Case("slide_random", "D · Uncoordinated window positions", "Original sliding-window examples", "The same six wrong answers jump to unrelated positions at each checkpoint.", "window", "slide_random"),
    Case("static", "Static answers", "No identity drift", "Nothing changes.", "flat", "none"),
    Case("gain", "Knows progressively more", "No identity drift", "Correct answers only accumulate.", "gain", "none"),
    Case("loss", "Knows progressively less", "No identity drift", "Correct answers only disappear.", "loss", "none"),
    Case("ability_cycle", "Accuracy rises and falls", "No identity drift", "The same difficulty order expands and contracts.", "periodic", "none"),
    Case("one_way", "One-way answer replacement", "Secular identity drift", "Old correct answers are steadily exchanged for new ones.", "flat", "monotonic"),
    Case("abrupt", "Abrupt permanent replacement", "Secular identity drift", "One stable answer pattern is replaced halfway through.", "flat", "abrupt"),
    Case("exact_cycle", "Exact periodic return", "Periodic identity change", "The transformation moves out and retraces exactly.", "flat", "periodic"),
    Case("quasi_cycle", "Near-periodic return", "Periodic identity change", "The transformation repeatedly expands and contracts imperfectly.", "flat", "quasi"),
    Case("travelling", "Travelling answer window", "Periodic identity change", "A fixed-size block moves through question identities.", "flat", "window"),
    Case("random", "Independent random churn", "Interference", "Each checkpoint swaps a new unrelated group of answers.", "flat", "random"),
    Case("markov", "Persistent random churn", "Interference", "Most swapped answers persist, while a few change each checkpoint.", "flat", "markov"),
    Case("gain_one_way", "Learning + one-way replacement", "Overlapping transformations", "The model knows more overall while identities drift.", "gain", "monotonic"),
    Case("ability_one_way", "Oscillating accuracy + one-way replacement", "Overlapping transformations", "Accuracy cycles while identity replacement continues.", "periodic", "monotonic"),
    Case("gain_cycle", "Learning + periodic replacement", "Overlapping transformations", "The model knows more overall while identities cycle.", "gain", "periodic"),
    Case("mono_cycle", "One-way + periodic replacement", "Overlapping transformations", "A secular transformation and a disjoint cycle overlap.", "flat", "mono_periodic"),
    Case("closed_loop", "Closed two-transformation loop", "Overlapping transformations", "Two disjoint cycles trace a loop and return to the start.", "flat", "loop"),
]


def ability_counts(kind: str) -> np.ndarray:
    u = np.linspace(0, 1, T)
    if kind == "flat":
        values = np.full(T, 38.0)
    elif kind == "gain":
        values = 28 + 14 * u
    elif kind == "loss":
        values = 42 - 14 * u
    elif kind == "periodic":
        values = 35 + 7 * np.sin(4 * np.pi * u)
    elif kind == "window":
        values = np.full(T, 54.0)
    else:
        raise ValueError(kind)
    return np.rint(values).astype(int)


def baseline(counts: np.ndarray) -> np.ndarray:
    return np.arange(N)[None, :] < counts[:, None]


def apply_prefix(mask: np.ndarray, donors: np.ndarray, receivers: np.ndarray, counts: np.ndarray) -> None:
    for t, count in enumerate(counts.astype(int)):
        if count:
            mask[t, donors[:count]] ^= True
            mask[t, receivers[:count]] ^= True


def transform(kind: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = np.linspace(0, 1, T)
    da, ra = rng.permutation(np.arange(0, 8)), rng.permutation(np.arange(52, 60))
    db, rb = rng.permutation(np.arange(8, 16)), rng.permutation(np.arange(44, 52))
    mask = np.zeros((T, N), dtype=np.bool_)
    if kind == "none":
        return mask
    if kind == "monotonic":
        apply_prefix(mask, da, ra, np.rint(8 * u))
    elif kind == "abrupt":
        apply_prefix(mask, da, ra, np.where(u < 0.5, 0, 8))
    elif kind == "periodic":
        apply_prefix(mask, da, ra, np.rint(4 * (1 - np.cos(4 * np.pi * u))))
    elif kind == "quasi":
        phase = 3.35 * np.pi * u + 0.38 * np.sin(5 * np.pi * u)
        apply_prefix(mask, da, ra, np.rint(8 * np.abs(np.sin(phase))))
    elif kind == "window":
        width = 3
        for t, x in enumerate(u):
            start = int(round(x * 8)) % 8
            active = np.array([(start + j) % 8 for j in range(width)])
            mask[t, da[active]] ^= True
            mask[t, ra[active]] ^= True
    elif kind == "random":
        for t in range(T):
            active = rng.choice(8, size=4, replace=False)
            mask[t, da[active]] ^= True
            mask[t, ra[active]] ^= True
    elif kind == "markov":
        active = set(int(x) for x in rng.choice(8, size=4, replace=False))
        for t in range(T):
            ids = np.array(sorted(active))
            mask[t, da[ids]] ^= True
            mask[t, ra[ids]] ^= True
            leave = int(rng.choice(np.array(sorted(active))))
            enter = int(rng.choice(np.array(sorted(set(range(8)) - active))))
            active.remove(leave)
            active.add(enter)
    elif kind == "mono_periodic":
        apply_prefix(mask, da, ra, np.rint(8 * u))
        apply_prefix(mask, db, rb, np.rint(4 * (1 - np.cos(4 * np.pi * u))))
    elif kind == "loop":
        apply_prefix(mask, da, ra, np.rint(4 + 4 * np.cos(2 * np.pi * u)))
        apply_prefix(mask, db, rb, np.rint(4 + 4 * np.sin(2 * np.pi * u)))
    else:
        raise ValueError(kind)
    return mask


def sliding_window_history(kind: str, seed: int) -> np.ndarray:
    """Return the user's canonical six-wrong-out-of-sixty histories."""
    width = 6
    if kind == "slide_one_way":
        positions = [2 * t for t in range(T)]
    elif kind == "slide_periodic":
        directions = ([1] * 5 + [-1] * 5) * 2 + [1]
        positions = [0]
        for direction in directions:
            positions.append(positions[-1] + 2 * direction)
    elif kind == "slide_quasi":
        directions = [1] * 6 + [-1] * 7 + [1] * 5 + [-1] * 3
        positions = [14]
        for direction in directions:
            positions.append(positions[-1] + 2 * direction)
    elif kind == "slide_random":
        rng = np.random.default_rng(seed)
        positions = [int(x) for x in rng.integers(0, N - width + 1, size=T)]
    else:
        raise ValueError(kind)
    states = np.ones((T, N), dtype=np.bool_)
    for t, start in enumerate(positions):
        states[t, start : start + width] = False
    return states


def calibration_history(kind: str) -> np.ndarray:
    """Constant-accuracy paths with endpoint replacement from 0% through 100%."""
    percent = int(kind.rsplit("_", 1)[1])
    final_pairs = round((percent / 100) * (N // 2))
    steps = 4
    states = np.zeros((steps, N), dtype=np.bool_)
    states[:, : N // 2] = True
    for t in range(steps):
        pairs = round(final_pairs * t / (steps - 1))
        if pairs:
            states[t, :pairs] = False
            states[t, N // 2 : N // 2 + pairs] = True
    return states


def build_data() -> dict:
    rows = []
    for index, case in enumerate(CASES):
        seed = 20260811 + index * 7919
        if case.transform.startswith("calibration_"):
            states = calibration_history(case.transform)
            base = np.repeat(states[:1], len(states), axis=0)
        elif case.transform.startswith("slide_"):
            states = sliding_window_history(case.transform, seed)
            base = np.repeat(states[:1], len(states), axis=0)
        else:
            counts = ability_counts(case.ability)
            base = baseline(counts)
            states = np.logical_xor(base, transform(case.transform, seed))
        raw_d = hamming_matrix(states)
        adjusted_d = excess_matrix(states, raw_d)
        residual_d = hamming_matrix(np.logical_xor(states, base))
        orders = permutation_orders(len(states), seed + 1009)
        raw = hdi_from_distance(raw_d, orders)
        adjusted = hdi_from_distance(adjusted_d, orders)
        residual = hdi_from_distance(residual_d, orders)
        rows.append({
            "key": case.key,
            "title": case.title,
            "family": case.family,
            "note": case.note,
            "states": ["".join("1" if x else "0" for x in row) for row in states],
            "checkpoints": len(states),
            "accuracy": [float(row.mean()) for row in states],
            "scores": {
                "raw_hdi": raw["hdi"],
                "adjusted_hdi": adjusted["hdi"],
                "residual_hdi": residual["hdi"],
                "srd": residual["secular_drift"],
            },
        })
    return {"items": N, "maximum_checkpoints": T, "cases": rows, "seed_root": 20260811}


def build_html(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Questionnaire histories and drift scores</title><style>
:root{{--paper:#f5f0e7;--panel:#fffdf8;--ink:#172831;--muted:#68757b;--rule:#d8d0c4;--red:#9b3438;--blue:#315f78;--teal:#39766e;--gold:#b38730;--purple:#725a8f;--green:#47734e;--shadow:0 13px 32px rgba(33,40,41,.07)}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;color:var(--ink);background:radial-gradient(circle at 92% 0,rgba(179,135,48,.13),transparent 30rem),linear-gradient(135deg,var(--paper),#fbf8f1 72%,#eee5d8);font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{width:min(1680px,100%);margin:auto;padding:0 clamp(24px,5vw,86px) 90px}}h1,h2,p{{margin-top:0}}h1,h2{{font-family:Georgia,"Times New Roman",serif;letter-spacing:-.034em}}header{{min-height:min(72vh,760px);display:grid;align-content:center;position:relative;padding:68px 0}}header:before{{content:"";position:absolute;left:calc(-1*clamp(24px,5vw,86px));top:0;bottom:0;width:10px;background:var(--red)}}.eyebrow{{display:flex;justify-content:space-between;gap:20px;color:var(--muted);font-size:12px;font-weight:850;letter-spacing:.14em;text-transform:uppercase;margin-bottom:28px}}h1{{font-size:clamp(50px,7vw,105px);line-height:.94;max-width:1320px;margin-bottom:26px}}.lede{{font-size:clamp(19px,2vw,30px);line-height:1.4;max-width:1160px;color:#364650}}.legend{{display:flex;flex-wrap:wrap;gap:13px 24px;margin-top:30px;padding-top:17px;border-top:1px solid var(--rule);color:var(--muted);font-size:12px}}.legend span:before{{content:"";display:inline-block;width:24px;height:8px;margin-right:8px;background:var(--c)}}.gallery{{display:grid;gap:32px}}.section-label{{padding:54px 0 4px;border-bottom:1px solid var(--rule);font:700 clamp(30px,3.5vw,55px)/1 Georgia,"Times New Roman",serif;letter-spacing:-.035em}}.section-label:first-child{{padding-top:0}}.history{{padding:clamp(18px,2vw,29px);border:1px solid var(--rule);border-top:8px solid var(--case,var(--blue));background:rgba(255,253,248,.88);box-shadow:var(--shadow)}}.history-head{{display:flex;justify-content:space-between;gap:22px;align-items:start;margin-bottom:18px}}.history h2{{font-size:clamp(25px,2.5vw,39px);margin-bottom:5px}}.history-head p{{color:var(--muted);margin:0}}.tag{{color:var(--case);font-size:10px;font-weight:900;letter-spacing:.1em;text-transform:uppercase;white-space:nowrap;padding-top:8px}}.body{{display:grid;grid-template-columns:minmax(0,1fr) 335px;gap:27px;align-items:stretch}}.questionnaire{{min-width:0;padding:13px;border:1px solid var(--rule);background:#f1ece3}}canvas{{display:block;width:100%;height:auto;image-rendering:pixelated}}.axis-note{{display:flex;justify-content:space-between;color:var(--muted);font-size:10px;margin-top:7px}}.scores{{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--rule);border:1px solid var(--rule)}}.score{{background:var(--panel);padding:16px}}.score.primary{{grid-column:1/-1;border-left:6px solid var(--blue);background:rgba(49,95,120,.08)}}.score strong{{display:block;font:700 30px Georgia,serif;color:var(--score,var(--ink))}}.score span{{display:block;color:var(--muted);font-size:10px;line-height:1.3}}.score small{{display:block;margin-top:3px;color:var(--muted);font-size:9px}}footer{{margin-top:52px;padding-top:18px;border-top:1px solid var(--rule);color:var(--muted);font-size:11px}}@media(max-width:920px){{.body{{grid-template-columns:1fr}}.scores{{grid-template-columns:repeat(4,1fr)}}.score.primary{{grid-column:auto}}}}@media(max-width:650px){{header{{min-height:auto}}.eyebrow,.history-head{{display:block}}.tag{{display:block;margin-bottom:8px}}.scores{{grid-template-columns:1fr 1fr}}.score.primary{{grid-column:1/-1}}main{{padding-bottom:55px}}}}
</style></head><body><main><header><div class="eyebrow"><span>GreekMMLU · synthetic questionnaire histories</span><span>11 August 2026</span></div><h1>Histories of correct and wrong answers</h1><p class="lede">The first six histories define the full 0–1 drift scale. The next four are your original sliding-window examples. Each score is calculated from exactly the answers shown.</p><div class="legend"><span style="--c:var(--green)">correct</span><span style="--c:var(--red)">wrong</span><span>top row = first checkpoint</span><span>bottom row = final checkpoint</span></div><div style="display:flex;flex-wrap:wrap;gap:1px;margin-top:24px;background:var(--rule);border:1px solid var(--rule)"><div style="flex:1 1 250px;padding:17px;background:var(--panel)"><strong style="display:block;font:700 30px Georgia,serif;color:var(--blue)">0.00 · minimum</strong><span style="color:var(--muted);font-size:11px">No persistent residual answer-identity change.</span></div><div style="flex:2 1 360px;padding:17px;background:linear-gradient(90deg,rgba(49,95,120,.05),rgba(49,95,120,.28))"><strong style="display:block;font:700 17px Georgia,serif">0.20 → 0.40 → 0.60 → 0.80</strong><span style="color:var(--muted);font-size:11px">Increasing fractions of the questionnaire are permanently replaced.</span></div><div style="flex:1 1 250px;padding:17px;background:var(--panel)"><strong style="display:block;font:700 30px Georgia,serif;color:var(--blue)">1.00 · maximum</strong><span style="color:var(--muted);font-size:11px">Every identity is replaced and distance grows monotonically with time.</span></div></div></header><section class="gallery" id="gallery"></section><footer>60-question synthetic questionnaires · 4–22 checkpoints · deterministic seed 20260811 · generator: {Path(__file__).name}</footer></main><script>const DATA={payload};
const COLORS={{correct:'#47734e',wrong:'#9b3438',paper:'#f1ece3',ink:'#68757b',line:'#fffdf8'}};const familyColor=f=>f==='Calibration ladder'?'#315f78':f==='Original sliding-window examples'?'#9b3438':f==='No identity drift'?'#315f78':f==='Secular identity drift'?'#39766e':f==='Periodic identity change'?'#b38730':f==='Interference'?'#725a8f':'#9b3438';const fmt=v=>Number(v).toFixed(2);
function draw(canvas,rows){{const dpr=Math.max(1,window.devicePixelRatio||1),left=28,top=4,cw=10,ch=rows.length<=4?20:9,W=left+DATA.items*cw,H=top+rows.length*ch;canvas.width=W*dpr;canvas.height=H*dpr;canvas.style.aspectRatio=`${{W}}/${{H}}`;const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);ctx.fillStyle=COLORS.paper;ctx.fillRect(0,0,W,H);ctx.font='8px Inter, sans-serif';ctx.textAlign='right';ctx.fillStyle=COLORS.ink;rows.forEach((row,t)=>{{if(t===0||t===Math.floor((rows.length-1)/2)||t===rows.length-1)ctx.fillText('t'+t,left-5,top+t*ch+12);[...row].forEach((v,i)=>{{ctx.fillStyle=v==='1'?COLORS.correct:COLORS.wrong;ctx.fillRect(left+i*cw,top+t*ch,cw-1,ch-1)}})}});ctx.strokeStyle='rgba(255,253,248,.95)';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(left,top+Math.floor(rows.length/2)*ch-.5);ctx.lineTo(W,top+Math.floor(rows.length/2)*ch-.5);ctx.stroke()}}
const gallery=document.querySelector('#gallery');DATA.cases.forEach((c,index)=>{{if(index===0||index===6||index===10){{const heading=document.createElement('div');heading.className='section-label';heading.textContent=index===0?'Calibration: minimum to maximum drift':index===6?'Your original sliding-window cases':'Additional stress cases';gallery.append(heading)}}const article=document.createElement('article');article.className='history';article.style.setProperty('--case',familyColor(c.family));const number=String(index+1).padStart(2,'0');article.innerHTML=`<div class="history-head"><div><span style="display:inline-block;margin:0 10px 7px 0;color:var(--muted);font-size:10px;font-weight:900;letter-spacing:.12em;text-transform:uppercase">Example ${{number}}</span><span class="tag">${{c.family}}</span><h2>${{c.title}}</h2><p>${{c.note}}</p></div></div><div class="body"><div class="questionnaire"><canvas aria-label="Example ${{number}}: ${{c.title}} questionnaire history"></canvas><div class="axis-note"><span>Q1</span><span>questions →</span><span>Q${{DATA.items}}</span></div></div><div class="scores"><div class="score primary" style="--score:var(--blue)"><strong>${{fmt(c.scores.srd)}}</strong><span>Secular residual drift</span><small>persistent identity change</small></div><div class="score" style="--score:var(--red)"><strong>${{fmt(c.scores.raw_hdi)}}</strong><span>Raw HDI</span></div><div class="score" style="--score:var(--gold)"><strong>${{fmt(c.scores.adjusted_hdi)}}</strong><span>Pair-adjusted HDI</span></div><div class="score" style="--score:var(--teal)"><strong>${{fmt(c.scores.residual_hdi)}}</strong><span>Residual HDI</span></div><div class="score"><strong>${{(c.accuracy[0]*100).toFixed(0)}}→${{(c.accuracy.at(-1)*100).toFixed(0)}}%</strong><span>Accuracy</span></div></div></div>`;gallery.append(article);draw(article.querySelector('canvas'),c.states)}});
</script></body></html>'''


def main() -> None:
    data = build_data()
    OUT_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_HTML.write_text(build_html(data), encoding="utf-8")
    print(json.dumps({"html": str(OUT_HTML), "data": str(OUT_JSON), "cases": len(data["cases"])}, indent=2))


if __name__ == "__main__":
    main()
