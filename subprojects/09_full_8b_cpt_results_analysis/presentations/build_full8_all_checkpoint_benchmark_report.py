#!/usr/bin/env python3
"""Build the complete 19-checkpoint native-Greek benchmark report.

The three early/40B/final points are read from their strict post-hoc rescoring
receipts. The remaining 16 points are read from the completed strict matrix on
Clariden. This keeps every plotted native-suite point on one frozen filtered
question set per benchmark.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
SUBPROJECT = HERE.parent
STRICT3 = SUBPROJECT / "evidence/d0_0p5b_vs_full8_native_greek_3cp_20260814/full8_filtered"
GREEKMMLU = HERE / "FULL8_RESULTS.data.json"
OUT_HTML = HERE / "FULL8_ALL_CHECKPOINT_NATIVE_BENCHMARKS_20260819.html"
OUT_DATA = HERE / "FULL8_ALL_CHECKPOINT_NATIVE_BENCHMARKS_20260819.data.json"

CHECKPOINTS = [0, 400, 1192, 2384, 3576, 4768, 5960, 7152, 8344, 9536, 10728,
               11920, 13112, 14304, 14627, 15496, 16688, 17880, 18284]
STRICT3_ITERS = {0, 9536, 18284}
CORE = ["asep_mcqa", "demosqa", "gpcr", "medical_mcqa", "oyxoy_metaphor",
        "oyxoy_nli", "oyxoy_wic", "oyxoy_wsd_definition"]
LABELS = {
    "asep_mcqa": "ASEP MCQA", "demosqa": "DemosQA", "gpcr": "GPCR",
    "medical_mcqa": "Medical MCQA", "oyxoy_metaphor": "OYXOY · metaphor",
    "oyxoy_nli": "OYXOY · NLI", "oyxoy_wic": "OYXOY · WiC",
    "oyxoy_wsd_definition": "OYXOY · WSD", "oyxoy_nli_exact_set": "NLI exact-set control",
}
REMOTE = r'''import csv, glob, json, os
roots = [
 "/iopsstor/scratch/cscs/fffoivos/evals/full8_native_greek_peak_window_20260817/matrix_v1_issue94",
 "/iopsstor/scratch/cscs/fffoivos/evals/full8_remaining12_checkpoint_release_20260817/matrix_v1",
]
out = {}
for root in roots:
 for p in glob.glob(root + "/iter_*/combined/metrics.csv"):
  it = int(os.path.basename(os.path.dirname(os.path.dirname(p))).split("_")[1])
  for row in csv.DictReader(open(p)):
   if row["subject"] == "__all__": out.setdefault(str(it), {})[row["benchmark"]] = row
print(json.dumps(out))'''


def row_metrics(row: dict[str, str]) -> dict[str, float | int | None]:
    def number(key: str) -> float | None:
        value = row.get(key, "")
        return float(value) if value not in ("", None) else None
    return {"n": int(row["n"]), "accuracy": number("accuracy"),
            "choice_nll": number("choice_nll"), "correct_answer_bpb": number("correct_answer_bpb"),
            "balanced_accuracy": number("balanced_accuracy"), "binary_macro_f1": number("binary_macro_f1")}


def local_strict_points() -> dict[str, dict[str, dict[str, float | int | None]]]:
    result: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for iteration in STRICT3_ITERS:
        path = STRICT3 / f"iter_{iteration:07d}" / "strict_filtered_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        result[str(iteration)] = {row["benchmark"]: row_metrics(row) for row in csv.DictReader(path.open()) if row["subject"] == "__all__"}
    return result


def remote_strict_points() -> dict[str, dict[str, dict[str, float | int | None]]]:
    encoded = base64.b64encode(REMOTE.encode()).decode()
    proc = subprocess.run(["ssh", "clariden", f"echo {encoded} | base64 -d | python3"], check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw = json.loads(proc.stdout)
    return {it: {bench: row_metrics(row) for bench, row in rows.items()} for it, rows in raw.items()}


def build_data() -> dict:
    points = local_strict_points()
    points.update(remote_strict_points())
    if sorted(map(int, points)) != CHECKPOINTS:
        raise ValueError(f"checkpoint mismatch: {sorted(map(int, points))}")
    for it, rows in points.items():
        missing = set(CORE) - set(rows)
        if missing:
            raise ValueError(f"{it}: missing {sorted(missing)}")
    gm = {p["iteration"]: p for p in json.loads(GREEKMMLU.read_text())["new_greekmmlu"]}
    if sorted(gm) != CHECKPOINTS:
        raise ValueError("GreekMMLU checkpoint coverage mismatch")
    token_per_update = 4_194_304
    return {
        "schema_version": "apertus_full8_all_checkpoint_native_benchmark_report_v1",
        "schedule": {"cooldown_start_iteration": 14627,
                     "cooldown_start_tokens_b": 14627 * token_per_update / 1e9,
                     "post_start_checkpoint_count": sum(it > 14627 for it in CHECKPOINTS),
                     "including_start_checkpoint_count": sum(it >= 14627 for it in CHECKPOINTS)},
        "checkpoints": [{"iteration": it, "tokens_b": it * token_per_update / 1e9,
                         "label": "init" if it == 0 else f"{it:,}"} for it in CHECKPOINTS],
        "greekmmlu": [{"iteration": it, "accuracy": gm[it]["clean_accuracy"],
                        "choice_nll": gm[it]["clean_choice_nll"], "n": gm[it]["clean_n"]} for it in CHECKPOINTS],
        "benchmarks": [{"id": ident, "label": LABELS[ident],
                        "points": [{"iteration": it, **points[str(it)][ident]} for it in CHECKPOINTS]}
                       for ident in CORE],
        "auxiliary": [{"id": "oyxoy_nli_exact_set", "label": LABELS["oyxoy_nli_exact_set"],
                       "points": [{"iteration": it, **points[str(it)]["oyxoy_nli_exact_set"]} for it in CHECKPOINTS]}],
        "evidence": {
            "strict_anchor": str(STRICT3),
            "peak_matrix": "/iopsstor/scratch/cscs/fffoivos/evals/full8_native_greek_peak_window_20260817/matrix_v1_issue94",
            "remaining_matrix": "/iopsstor/scratch/cscs/fffoivos/evals/full8_remaining12_checkpoint_release_20260817/matrix_v1",
            "remaining_receipt": "/iopsstor/scratch/cscs/fffoivos/evals/full8_remaining12_checkpoint_release_20260817/matrix_v1/matrix_receipt.json",
            "remaining_receipt_sha256": "1a077bd4 (contract); completed matrix verified independently: 252/252 shards",
            "contract": str(SUBPROJECT / "evaluation/native_greek_3cp_contract.json"),
        },
    }


TEMPLATE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Apertus 8B · complete checkpoint benchmark results</title>
<style>
:root{--paper:#f5f0e7;--panel:#fffdf8;--ink:#17242d;--muted:#66747d;--rule:#d7cec0;--red:#9b3438;--blue:#315f78;--teal:#39766e;--gold:#b38730;--green:#47734e;--purple:#725a8f;--shadow:0 16px 42px rgba(42,38,31,.08)}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 92% 0,rgba(179,135,48,.14),transparent 27rem),linear-gradient(135deg,var(--paper),#fbf8f1 75%,#eee6d8);font:16px/1.55 Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.page{max-width:1740px;margin:auto;padding:0 clamp(24px,5vw,86px) 96px}h1,h2,h3{font-family:Georgia,"Times New Roman",serif;letter-spacing:-.032em;line-height:1.02;margin:0 0 16px}h1{font-size:clamp(52px,7vw,104px);max-width:1430px}h2{font-size:clamp(34px,4.2vw,66px)}h3{font-size:25px}.hero{min-height:min(89vh,1000px);display:grid;align-content:center;padding:72px 0;position:relative}.hero:before{content:"";position:absolute;left:calc(-1 * clamp(24px,5vw,86px));top:0;bottom:0;width:10px;background:var(--red)}.eyebrow,.kicker{color:var(--muted);font-size:12px;font-weight:900;letter-spacing:.14em;text-transform:uppercase}.eyebrow{display:flex;justify-content:space-between;gap:20px;margin-bottom:30px}.rule{height:8px;width:110px;background:var(--red);margin-bottom:26px}.lede{font-size:clamp(20px,2vw,31px);max-width:1250px;color:#3b4a53}.finding{margin-top:36px;border-left:7px solid var(--gold);padding:18px 0 0 28px;font:700 clamp(23px,2.5vw,40px)/1.22 Georgia,serif;max-width:1450px}.nav{display:flex;gap:10px 24px;flex-wrap:wrap;margin-top:38px;border-top:1px solid var(--rule);padding-top:17px}.nav a{color:var(--muted);font-size:13px;font-weight:800;text-decoration:none}.section{padding-top:110px}.head{display:grid;grid-template-columns:8px 1fr;gap:24px;margin-bottom:30px}.bar{background:var(--accent);min-height:86px}.kicker{color:var(--accent);margin-bottom:8px}.intro{max-width:1200px;color:var(--muted);font-size:18px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid var(--rule);border-bottom:1px solid var(--rule);margin:30px 0}.metric{padding:22px;border-left:1px solid var(--rule)}.metric:first-child{border-left:0;padding-left:0}.num{font:700 clamp(30px,3vw,50px)/1 Georgia,serif}.label{font-size:12px;color:var(--muted);margin-top:8px}.figure{margin:0 0 30px;padding:clamp(14px,2vw,28px);background:rgba(255,253,248,.84);border:1px solid var(--rule);box-shadow:var(--shadow)}svg{display:block;width:100%;height:auto}.figure figcaption{margin-top:13px;color:var(--muted);font-size:12px;display:flex;justify-content:space-between;gap:18px}.legend{display:flex;gap:10px 22px;flex-wrap:wrap;margin:0 0 15px;color:var(--muted);font-size:13px}.sw{display:inline-block;width:24px;height:5px;vertical-align:middle;margin-right:7px;background:var(--c)}.notes{display:grid;grid-template-columns:repeat(3,1fr);gap:22px}.note{padding:22px;background:rgba(255,253,248,.72);border-top:7px solid var(--blue)}.note.gold{border-color:var(--gold)}.note.red{border-color:var(--red)}.note.green{border-color:var(--green)}.note p{margin:0;color:#40505a}.tablewrap{overflow:auto;border:1px solid var(--rule);background:rgba(255,255,255,.26)}table{border-collapse:separate;border-spacing:0;width:100%;min-width:880px}th,td{padding:11px 13px;border-right:1px solid rgba(215,206,192,.55);border-bottom:1px solid var(--rule);text-align:right;font-variant-numeric:tabular-nums}th{background:#eee7dc;color:var(--muted);font-size:11px;letter-spacing:.05em;text-transform:uppercase}th:first-child,td:first-child{text-align:left;font-weight:800}.evidence{display:grid;grid-template-columns:1fr 1fr;gap:24px}.evidence article{border-top:6px solid var(--teal);padding-top:14px;min-width:0}.evidence code{font-size:12px;word-break:break-all}.pill{display:inline-block;border-radius:999px;padding:5px 10px;background:#dce9dc;color:#355b3b;font-size:11px;font-weight:900;letter-spacing:.08em;text-transform:uppercase}@media(max-width:900px){.metrics,.notes,.evidence{grid-template-columns:1fr 1fr}.figure figcaption,.eyebrow{flex-direction:column}}@media(max-width:600px){.metrics,.notes,.evidence{grid-template-columns:1fr}.metric{border-left:0;border-top:1px solid var(--rule);padding:17px 0}.hero{min-height:auto}.section{padding-top:76px}}@media print{body{background:#fff}.page{max-width:none;padding:0}.hero{min-height:auto;padding:30px 0}.section{padding-top:42px;break-before:page}.figure{box-shadow:none;break-inside:avoid}.note{break-inside:avoid}}
</style></head><body><main class="page"><header class="hero"><div class="eyebrow"><span>Apertus 8B · decontaminated native-Greek checkpoint matrix</span><span>19 checkpoints · evidence closed 19 August 2026</span></div><div class="rule"></div><h1>The GreekMMLU peak is real, but other Greek capabilities peak at different times.</h1><p class="lede">All 19 saved CPT checkpoints are now scored on the same strict, post-hoc decontaminated native-Greek benchmark subsets. The report separates continuous language-model evidence from discrete benchmark capability, rather than collapsing unlike tasks into one average.</p><div class="finding" id="finding"></div><nav class="nav"><a href="#contract">Contract</a><a href="#greekmmlu">GreekMMLU</a><a href="#mcq">MCQ suite</a><a href="#oyxoy">OYXOY</a><a href="#table">Exact results</a><a href="#interpretation">Interpretation</a><a href="#evidence">Evidence</a></nav></header>
<section class="section" id="contract" style="--accent:var(--purple)"><div class="head"><div class="bar"></div><div><div class="kicker">01 · evaluation contract</div><h2>One causal-LM trajectory; one strict subset per task.</h2><p class="intro">The scorer uses frozen Greek prompts, option order, FP32 candidate likelihood arithmetic and zero-shot scoring. The three original anchor checkpoints are read from their later strict rescoring; the sixteen additional checkpoints were evaluated directly on that same filtered subset.</p></div></div><div class="metrics" id="metrics"></div><div class="notes"><article class="note green"><h3>Coverage closed</h3><p>The matrix now covers every one of the 19 saved 8B checkpoints: initialization through update 18,284 / 76.689B token slots.</p></article><article class="note gold"><h3>Fair comparison</h3><p>All plotted native-suite points use fixed retained example counts per benchmark. The older raw three-point scores are not mixed into this trajectory.</p></article><article class="note red"><h3>Not a training intervention</h3><p>Filtering corrects reporting for training exposure discovered after the run. It does not change model weights or retroactively remove corpus documents.</p></article></div></section>
<section class="section" id="greekmmlu" style="--accent:var(--red)"><div class="head"><div class="bar"></div><div><div class="kicker">02 · GreekMMLU through the full run</div><h2>Accuracy reaches 56.81% at 39.997B; choice NLL reaches its minimum at the same checkpoint.</h2><p class="intro">GreekMMLU is independently decontaminated (16,159 questions) and serves as the broad native-Greek anchor. Points are raw measurements; lines only join evaluations.</p></div></div><div class="legend"><span><i class="sw" style="--c:var(--red)"></i>clean accuracy · higher is better</span><span><i class="sw" style="--c:var(--blue)"></i>clean choice NLL · lower is better</span></div><figure class="figure"><svg id="gm" role="img"></svg><figcaption><span>Full horizon: initialization to the 76.689B-token terminal checkpoint.</span><span id="gm-cap"></span></figcaption></figure></section>
<section class="section" id="mcq" style="--accent:var(--blue)"><div class="head"><div class="bar"></div><div><div class="kicker">03 · native MCQ suite</div><h2>Medical MCQA aligns with the mid-run peak; ASEP, DemosQA and GPCR retain distinct late behavior.</h2><p class="intro">Each panel has its own y-scale so that within-benchmark movement is visible. The common x-axis preserves exact temporal alignment.</p></div></div><div class="legend"><span><i class="sw" style="--c:var(--blue)"></i>accuracy · higher is better</span><span><i class="sw" style="--c:var(--gold)"></i>choice NLL · lower is better</span></div><figure class="figure"><svg id="mcq-chart" role="img"></svg><figcaption><span>Strict decontaminated subsets: ASEP n=1,180 · DemosQA n=599 · GPCR n=194 · Medical n=419.</span><span>Panel scales are local; do not compare vertical positions across tasks.</span></figcaption></figure></section>
<section class="section" id="oyxoy" style="--accent:var(--teal)"><div class="head"><div class="bar"></div><div><div class="kicker">04 · lexical and semantic tasks</div><h2>OYXOY exposes a different kind of post-peak decline.</h2><p class="intro">The OYXOY tasks are scored as zero-shot causal-LM decisions, not as the benchmark’s supervised encoder setup. WiC and metaphor have strong class imbalance; their raw accuracy needs the balanced-accuracy columns in the exact table below.</p></div></div><div class="legend"><span><i class="sw" style="--c:var(--teal)"></i>accuracy</span><span><i class="sw" style="--c:var(--gold)"></i>choice NLL</span></div><figure class="figure"><svg id="oyxoy-chart" role="img"></svg><figcaption><span>Strict retained subsets: metaphor n=2,042 · NLI n=5,244 · WiC n=54,217 · WSD n=9,999.</span><span>For balanced binary accuracy, 50% marks chance; exact values are tabulated below.</span></figcaption></figure></section>
<section class="section" id="table" style="--accent:var(--gold)"><div class="head"><div class="bar"></div><div><div class="kicker">05 · benchmark optima</div><h2>No macro-average: the winning checkpoint depends on the capability measured.</h2><p class="intro">This table reports each task’s best and final strict-filtered score. It intentionally does not average tasks of radically different size, class balance and semantic scope.</p></div></div><div class="tablewrap"><table id="optima"></table></div></section>
<section class="section" id="interpretation" style="--accent:var(--green)"><div class="head"><div class="bar"></div><div><div class="kicker">06 · interpretation and limits</div><h2>The run improves the language-model objective longer than it improves every downstream capability.</h2></div></div><div class="notes" id="notes"></div></section>
<section class="section" id="evidence" style="--accent:var(--teal)"><div class="head"><div class="bar"></div><div><div class="kicker">07 · reproducibility</div><h2>Receipt-backed matrix and strict rescoring anchors.</h2></div></div><div class="evidence" id="evidence-grid"></div></section></main><script type="application/json" id="data">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),C={ink:'#17242d',muted:'#66747d',rule:'#d7cec0',red:'#9b3438',blue:'#315f78',teal:'#39766e',gold:'#b38730',green:'#47734e',purple:'#725a8f'};const NS='http://www.w3.org/2000/svg';
const el=(n,a={},t='')=>{let e=document.createElementNS(NS,n);Object.entries(a).forEach(([k,v])=>e.setAttribute(k,v));if(t!==''&&t!==null)e.textContent=t;return e};const pct=v=>(v*100).toFixed(2)+'%';const f=v=>Number(v).toFixed(3);const n=v=>Number(v).toLocaleString('en-US');const pts=D.checkpoints, xMax=pts.at(-1).tokens_b, coolIndex=pts.findIndex(p=>p.iteration===D.schedule.cooldown_start_iteration);
function line(points,color,dash=''){return el('path',{d:points.map((p,i)=>(i?'L':'M')+p[0].toFixed(1)+','+p[1].toFixed(1)).join(' '),fill:'none',stroke:color,'stroke-width':3.5,'stroke-dasharray':dash,'stroke-linecap':'round','stroke-linejoin':'round'})}
function axes(svg,g,range,fmt,zero){const{x,y,w,h}=g;for(let i=0;i<4;i++){let v=range[0]+(range[1]-range[0])*i/3,yy=y+h-(v-range[0])/(range[1]-range[0])*h;svg.append(el('line',{x1:x,x2:x+w,y1:yy,y2:yy,stroke:C.rule}));svg.append(el('text',{x:x-8,y:yy+4,'text-anchor':'end',fill:C.muted,'font-size':10},fmt(v)))}if(zero!==undefined&&zero>=range[0]&&zero<=range[1]){let yy=y+h-(zero-range[0])/(range[1]-range[0])*h;svg.append(el('line',{x1:x,x2:x+w,y1:yy,y2:yy,stroke:C.gold,'stroke-width':1.6,'stroke-dasharray':'5 5'}))}}
function addSeries(svg,values,g,range,color,fmt,mark=true){let X=i=>g.x+i/(values.length-1)*g.w,Y=v=>g.y+g.h-(v-range[0])/(range[1]-range[0])*g.h,ps=values.map((v,i)=>[X(i),Y(v)]);svg.append(line(ps,color));if(mark)ps.forEach(p=>svg.append(el('circle',{cx:p[0],cy:p[1],r:3.8,fill:color,stroke:'#fffdf8','stroke-width':1.6})));return ps}
function labels(svg,g){pts.forEach((p,i)=>{if(i%2&&i!==pts.length-1)return;svg.append(el('text',{x:g.x+i/(pts.length-1)*g.w,y:g.y+g.h+24,'text-anchor':i===0?'start':i===pts.length-1?'end':'middle',fill:C.muted,'font-size':9},p.tokens_b.toFixed(0)+'B'))})}
function cooldown(svg,g,label=false){let xx=g.x+coolIndex/(pts.length-1)*g.w;svg.append(el('rect',{x:xx,y:g.y,width:g.x+g.w-xx,height:g.h,fill:'rgba(179,135,48,.12)'}));svg.append(el('line',{x1:xx,x2:xx,y1:g.y,y2:g.y+g.h,stroke:C.gold,'stroke-width':1.7,'stroke-dasharray':'5 5'}));if(label)svg.append(el('text',{x:xx+8,y:g.y+16,fill:'#795d21','font-size':11,'font-weight':900},'1-sqrt LR cooldown'))}
let bestGM=D.greekmmlu.reduce((a,b)=>a.accuracy>b.accuracy?a:b), minGM=D.greekmmlu.reduce((a,b)=>a.choice_nll<b.choice_nll?a:b);document.getElementById('finding').textContent=`GreekMMLU peaks at update ${bestGM.iteration.toLocaleString()} / ${pts.find(p=>p.iteration===bestGM.iteration).tokens_b.toFixed(3)}B token slots (${pct(bestGM.accuracy)}; NLL ${f(minGM.choice_nll)}). The same midpoint is best on Medical MCQA, while GPCR continues improving later—evidence against a single universal “best checkpoint.”`;
document.getElementById('metrics').innerHTML=`<div class=metric><div class=num>19</div><div class=label>saved 8B checkpoints scored</div></div><div class=metric><div class=num>8</div><div class=label>core native-Greek benchmark views</div></div><div class=metric><div class=num>${n(D.benchmarks.reduce((s,b)=>s+b.points[0].n,0))}</div><div class=label>strict retained examples per checkpoint</div></div><div class=metric><div class=num>252</div><div class=label>remaining-matrix shards independently audited</div></div>`;
function gmChart(){let s=document.getElementById('gm'),W=1540,H=640,m={l:110,r:110,t:52,b:70},g={x:m.l,y:m.t,w:W-m.l-m.r,h:H-m.t-m.b};s.setAttribute('viewBox',`0 0 ${W} ${H}`);s.append(el('title',{},'GreekMMLU clean accuracy and choice NLL at 19 checkpoints'));let acc=D.greekmmlu.map(p=>p.accuracy),loss=D.greekmmlu.map(p=>p.choice_nll),ra=[Math.min(...acc)-.015,Math.max(...acc)+.015],rl=[Math.min(...loss)-.035,Math.max(...loss)+.035];cooldown(s,g,true);axes(s,g,ra,pct);for(let i=0;i<4;i++){let v=rl[0]+(rl[1]-rl[0])*i/3,yy=g.y+g.h-(v-rl[0])/(rl[1]-rl[0])*g.h;s.append(el('text',{x:W-m.r+8,y:yy+4,fill:C.blue,'font-size':10},f(v)))}let pa=addSeries(s,acc,g,ra,C.red),pl=loss.map((v,i)=>[g.x+i/(loss.length-1)*g.w,g.y+g.h-(v-rl[0])/(rl[1]-rl[0])*g.h]);s.append(line(pl,C.blue,'7 5'));pl.forEach(p=>s.append(el('circle',{cx:p[0],cy:p[1],r:3.2,fill:C.blue,stroke:'#fffdf8','stroke-width':1.4})));[9536,14627,18284].forEach((iteration,j)=>{let k=pts.findIndex(p=>p.iteration===iteration),p=pa[k],gm=D.greekmmlu[k],above=j!==1;s.append(el('rect',{x:p[0]-53,y:p[1]+(above?-50:14),width:106,height:34,rx:4,fill:'#fffdf8',stroke:j===0?C.red:C.gold}));s.append(el('text',{x:p[0],y:p[1]+(above?-29:35),'text-anchor':'middle',fill:j===0?C.red:'#795d21','font-size':10,'font-weight':900},`${pts[k].tokens_b.toFixed(1)}B · ${pct(gm.accuracy)}`))});labels(s,g)}gmChart();document.getElementById('gm-cap').textContent=`Clean subset n=${n(D.greekmmlu[0].n)} · ${D.schedule.post_start_checkpoint_count} checkpoints strictly after cooldown start (${D.schedule.including_start_checkpoint_count} including the 61.350B boundary) · endpoint ${pct(D.greekmmlu.at(-1).accuracy)} / NLL ${f(D.greekmmlu.at(-1).choice_nll)}.`;
function multiples(id,items,kind){let s=document.getElementById(id),W=1540,cols=2,pw=705,ph=310,gapX=58,gapY=52,rows=Math.ceil(items.length/cols),H=50+rows*(ph+gapY),left=44;s.setAttribute('viewBox',`0 0 ${W} ${H}`);s.append(el('title',{},`${kind} benchmark trajectories over 19 checkpoints`));items.forEach((b,i)=>{let ox=left+(i%cols)*(pw+gapX),oy=44+Math.floor(i/cols)*(ph+gapY),g={x:ox+54,y:oy+60,w:pw-72,h:ph-105},acc=b.points.map(p=>p.accuracy),nll=b.points.map(p=>p.choice_nll).filter(v=>v!==null),lo=Math.min(...acc)-.025,hi=Math.max(...acc)+.025;s.append(el('rect',{x:ox,y:oy,width:pw,height:ph,fill:'rgba(255,253,248,.38)',stroke:C.rule}));s.append(el('text',{x:ox+15,y:oy+25,fill:C.ink,'font-size':18,'font-family':'Georgia','font-weight':850},b.label));s.append(el('text',{x:ox+15,y:oy+44,fill:C.muted,'font-size':10,'font-weight':750},`strict n=${n(b.points[0].n)} · local vertical scale`));cooldown(s,g,i===0);axes(s,g,[lo,hi],pct,kind==='oyxoy'&&b.id!=='oyxoy_nli'?0.5:undefined);addSeries(s,acc,g,[lo,hi],kind==='oyxoy'?C.teal:C.blue,pct);if(nll.length===acc.length){let rn=[Math.min(...nll)-.025,Math.max(...nll)+.025],ps=b.points.map((p,j)=>[g.x+j/(b.points.length-1)*g.w,g.y+g.h-(p.choice_nll-rn[0])/(rn[1]-rn[0])*g.h]);s.append(line(ps,C.gold,'7 5'));ps.forEach(p=>s.append(el('circle',{cx:p[0],cy:p[1],r:3,fill:C.gold,stroke:'#fffdf8','stroke-width':1.2})));for(let z=0;z<3;z++){let v=rn[0]+(rn[1]-rn[0])*z/2,yy=g.y+g.h-(v-rn[0])/(rn[1]-rn[0])*g.h;s.append(el('text',{x:g.x+g.w+7,y:yy+4,fill:C.gold,'font-size':9},f(v)))}}labels(s,g)});}
multiples('mcq-chart',D.benchmarks.slice(0,4),'mcq');multiples('oyxoy-chart',D.benchmarks.slice(4),'oyxoy');
function opt(metric,better){return b=>{let a=b.points.filter(p=>p[metric]!==null);return a.length?a.reduce((x,p)=>better(p[metric],x[metric])?p:x):null}}function fmt(v){return v===null?'—':pct(v)}function fmtn(v){return v===null?'—':f(v)}let rows=D.benchmarks.map(b=>{let a=opt('accuracy',(x,y)=>x>y)(b),l=opt('choice_nll',(x,y)=>x<y)(b),bal=opt('balanced_accuracy',(x,y)=>x>y)(b),last=b.points.at(-1);return `<tr><td>${b.label}</td><td>${n(b.points[0].n)}</td><td>${a.iteration.toLocaleString()} · ${pts.find(p=>p.iteration===a.iteration).tokens_b.toFixed(1)}B</td><td>${fmt(a.accuracy)}</td><td>${last.iteration.toLocaleString()} · ${fmt(last.accuracy)}</td><td>${l===null?'—':l.iteration.toLocaleString()+' · '+fmtn(l.choice_nll)}</td><td>${fmtn(last.choice_nll)}</td><td>${bal===null?'—':bal.iteration.toLocaleString()+' · '+fmt(bal.balanced_accuracy)}</td><td>${fmt(last.balanced_accuracy)}</td></tr>`}).join('');document.getElementById('optima').innerHTML=`<thead><tr><th>benchmark</th><th>strict n</th><th>best accuracy checkpoint</th><th>best accuracy</th><th>final accuracy</th><th>lowest NLL checkpoint</th><th>final NLL</th><th>best balanced accuracy</th><th>final balanced accuracy</th></tr></thead><tbody>${rows}</tbody>`;
let medical=D.benchmarks.find(b=>b.id==='medical_mcqa'),gpcr=D.benchmarks.find(b=>b.id==='gpcr'),final=D.checkpoints.at(-1);document.getElementById('notes').innerHTML=`<article class="note green"><h3>Mid-run is a useful selection point</h3><p>The 40B GreekMMLU optimum agrees with the best Medical MCQA score and with the lowest GreekMMLU choice NLL. It is a substantive checkpoint-selection candidate, not an accuracy accident alone.</p></article><article class="note gold"><h3>Late training is not uniformly harmful</h3><p>GPCR's strict accuracy reaches its maximum at update ${opt('accuracy',(x,y)=>x>y)(gpcr).iteration.toLocaleString()}, and ASEP/DemosQA have their own trajectories. Dataset-level loss and one benchmark cannot fully specify capability.</p></article><article class="note red"><h3>Interpret cautiously</h3><p>These are one-seed, fixed-prompt measurements. The report has no confidence intervals or replicated trajectories; it shows effect sizes and timing, not statistical significance.</p></article>`;
let E=D.evidence;document.getElementById('evidence-grid').innerHTML=`<article><h3>Frozen contract</h3><code>${E.contract}</code><p>Greek prompts, option order, FP32 likelihood scorer and zero-shot methodology are frozen here.</p></article><article><h3>Strict anchor rescoring</h3><code>${E.strict_anchor}</code><p>Initialization, 40B and terminal points use these strict-filtered receipts—not the older raw matrix values.</p></article><article><h3>Peak-window matrix</h3><code>${E.peak_matrix}</code><p>Four narrow-window checkpoint evaluations, all on the same retained subsets.</p></article><article><h3>Completed remaining matrix</h3><code>${E.remaining_matrix}</code><p><span class=pill>Completed and independently audited</span><br>${E.remaining_receipt_sha256}</p></article>`;
</script></body></html>'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT_HTML)
    parser.add_argument("--data-output", type=Path, default=OUT_DATA)
    args = parser.parse_args()
    data = build_data()
    args.data_output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    args.output.write_text(TEMPLATE.replace("__DATA__", payload))
    print(json.dumps({"ok": True, "output": str(args.output), "data": str(args.data_output),
                      "checkpoints": len(data["checkpoints"]), "bytes": args.output.stat().st_size}))


if __name__ == "__main__":
    raise SystemExit(main())
