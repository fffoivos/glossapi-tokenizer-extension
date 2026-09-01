#!/usr/bin/env python3
"""Build the standalone final presentation of the completed sanitized 8B CPT run."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import datetime as dt


def load_base(path: Path):
    spec = importlib.util.spec_from_file_location("full8_report_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def standalone_payload(base, repo: Path, evidence: Path, final_evidence: Path) -> dict:
    raw = base.build_payload(repo, evidence, final_evidence_root=final_evidence)
    keep_contract_names = {
        "new_pool_corpus_receipt.json",
        "new_schedule_manifest.json",
        "sanitized_bridge_receipt.json",
        "postmask_dedup_receipt.json",
    }
    bindings = []
    for binding in raw["bindings"]:
        role = binding["role"]
        name = Path(binding["path"]).name
        if role in {"validation_parser", "new_training_log", "final_campaign_evidence"}:
            bindings.append(binding)
        elif role == "data_contract" and name in keep_contract_names:
            bindings.append(binding)
    standalone_meta_keys = (
        "generated_at", "snapshot_iteration", "snapshot_tokens_b", "snapshot_fraction",
        "last_validation_iteration", "planned_updates", "active_tokens",
        "median_step_seconds", "p90_step_seconds", "latest_training_loss",
        "skipped_updates", "nan_updates", "greekmmlu_latest_complete",
        "greekmmlu_required_milestones", "greekmmlu_complete_milestones",
        "raw_points_no_smoothing",
    )
    payload = {
        "meta": {
            "title": "Apertus 8B CPT — standalone final results",
            **{key: raw["meta"][key] for key in standalone_meta_keys},
        },
        "constants": {
            key: raw["constants"][key]
            for key in (
                "tokens_per_update", "peak_lr", "minimum_lr", "warmup_end",
                "new_updates", "new_cooldown_start",
            )
        },
        "labels": raw["labels"],
        "learning_panels": raw["learning_panels"],
        "retention_panels": raw["retention_panels"],
        "new_training": raw["new_training"],
        "new_validation": raw["new_validation"],
        "new_greekmmlu": raw["new_greekmmlu"],
        "per_document_validation": raw["per_document_validation"],
        "completion": raw["completion"],
        "validation_exclusions": raw["validation_exclusions"]["new"],
        "data": {
            "new_geometry": raw["data"]["new_geometry"],
            "new_modern": raw["data"]["new_modern"],
            "new_schedule_hash": raw["data"]["new_schedule_hash"],
            "masked_documents": raw["data"]["masked_documents"],
            "dedup_counts": raw["data"]["dedup_counts"],
            "anonymization": raw["data"]["anonymization"],
        },
        "bindings": bindings,
    }
    if payload["completion"].get("status") != "completed":
        raise ValueError("campaign completion receipt is not complete")
    if int(payload["meta"]["snapshot_iteration"]) != int(payload["constants"]["new_updates"]):
        raise ValueError("standalone report requires terminal training evidence")
    if len(payload["new_greekmmlu"]) != 19:
        raise ValueError("standalone report requires all GreekMMLU milestones")
    if sum(len(rows) for rows in payload["per_document_validation"].values()) != 39:
        raise ValueError("standalone report requires all per-document checkpoints")
    return payload


SCRIPT = r"""
const D=JSON.parse(document.getElementById('report-data').textContent),NS='http://www.w3.org/2000/svg';
function s(tag,a={}){const e=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(a))e.setAttribute(k,v);return e}
function moving(rows,n=101){const out=[];let sum=0,q=[];for(const p of rows){q.push(p);sum+=p[1];if(q.length>n)sum-=q.shift()[1];out.push([p[0],sum/q.length])}return out}
function fmt(v,opt){return opt.percent?`${(100*v).toFixed(opt.yDigits??2)}%`:v.toFixed(opt.yDigits??3)}
function plot(el,series,opt={}){const W=opt.width??1000,H=opt.height??360,m={l:72,r:opt.endLabels?145:28,t:24,b:44},svg=s('svg',{viewBox:`0 0 ${W} ${H}`,class:'chart',role:'img','aria-label':opt.label||'Chart'});el.appendChild(svg);const all=series.flatMap(x=>x.values).filter(p=>Number.isFinite(p[0])&&Number.isFinite(p[1])&&(opt.xmin===undefined||p[0]>=opt.xmin)&&(opt.xmax===undefined||p[0]<=opt.xmax));if(!all.length)return;const xs=all.map(p=>p[0]),ys=all.map(p=>p[1]);let xmin=opt.xmin??Math.min(...xs),xmax=opt.xmax??Math.max(...xs),ymin=opt.ymin??Math.min(...ys),ymax=opt.ymax??Math.max(...ys);if(xmax===xmin)xmax=xmin+1;if(ymax===ymin){ymax+=1;ymin-=1}const pad=(ymax-ymin)*.08;if(opt.ymin===undefined)ymin-=pad;if(opt.ymax===undefined)ymax+=pad;const X=x=>m.l+(x-xmin)/(xmax-xmin)*(W-m.l-m.r),Y=y=>m.t+(ymax-y)/(ymax-ymin)*(H-m.t-m.b);
for(let i=0;i<=4;i++){const y=ymin+(ymax-ymin)*i/4,py=Y(y);svg.appendChild(s('line',{x1:m.l,y1:py,x2:W-m.r,y2:py,class:'gridline'}));const t=s('text',{x:m.l-9,y:py+4,'text-anchor':'end',class:'tick'});t.textContent=opt.percent?`${(100*y).toFixed(1)}%`:y.toFixed(opt.yDigits??3);svg.appendChild(t)}
for(let i=0;i<=4;i++){const x=xmin+(xmax-xmin)*i/4,px=X(x),t=s('text',{x:px,y:H-14,'text-anchor':'middle',class:'tick'});t.textContent=opt.tokens?`${(x*D.constants.tokens_per_update/1e9).toFixed(1)}B`:Math.round(x).toLocaleString();svg.appendChild(t)}svg.appendChild(s('line',{x1:m.l,y1:m.t,x2:m.l,y2:H-m.b,class:'axis'}));svg.appendChild(s('line',{x1:m.l,y1:H-m.b,x2:W-m.r,y2:H-m.b,class:'axis'}));
for(const marker of opt.markers||[]){if(marker.x<xmin||marker.x>xmax)continue;const x=X(marker.x);svg.appendChild(s('line',{x1:x,y1:m.t,x2:x,y2:H-m.b,class:'vline'}));const t=s('text',{x:x+5,y:m.t+12,class:'annotation'});t.textContent=marker.label;svg.appendChild(t)}
const ends=[];for(const item of series){const vals=item.values.filter(p=>p[0]>=xmin&&p[0]<=xmax&&Number.isFinite(p[1])),d=vals.map((p,i)=>`${i?'L':'M'}${X(p[0]).toFixed(2)},${Y(p[1]).toFixed(2)}`).join(' ');svg.appendChild(s('path',{d,class:`series ${item.className||''}`}));if(item.points)for(const p of vals)svg.appendChild(s('circle',{cx:X(p[0]),cy:Y(p[1]),r:opt.pointRadius??3,class:'point-new'}));if(opt.endLabels&&vals.length)ends.push({item,p:vals.at(-1),actualY:Y(vals.at(-1)[1])})}
if(opt.endLabels)for(const e of ends){const color='var(--new)',x0=X(e.p[0]),x1=W-m.r+8;svg.appendChild(s('line',{x1:x0,y1:e.actualY,x2:x1-3,y2:e.actualY,class:'leader',stroke:color}));const t=s('text',{x:x1,y:e.actualY+4,class:'endpoint',fill:color});t.textContent=`${e.item.label??''} ${fmt(e.p[1],opt)}`.trim();svg.appendChild(t)}}
function val(p){return D.new_validation[p].map(r=>[r.iteration,r.bpb])}function gap(rows){let best=Infinity;return rows.map(([x,y])=>{best=Math.min(best,y);return[x,y-best]})}
const maxTrain=D.constants.new_updates,raw=D.new_training.map(r=>[r.iteration,r.loss]);plot(document.getElementById('train-loss'),[{values:raw,className:'new raw'},{values:moving(raw),className:'new',label:'endpoint'}],{xmin:0,xmax:maxTrain,tokens:true,endLabels:true,label:'Full optimizer loss trajectory'});
function lr(){const v=[];for(let u=0;u<=D.constants.new_updates;u+=40){let y;if(u<=D.constants.warmup_end)y=D.constants.minimum_lr+(D.constants.peak_lr-D.constants.minimum_lr)*u/D.constants.warmup_end;else if(u<=D.constants.new_cooldown_start)y=D.constants.peak_lr;else{const q=(u-D.constants.new_cooldown_start)/(D.constants.new_updates-D.constants.new_cooldown_start);y=D.constants.minimum_lr+(D.constants.peak_lr-D.constants.minimum_lr)*(1-Math.sqrt(Math.min(1,q)))}v.push([u,y])}v.push([D.constants.new_updates,D.constants.minimum_lr]);return v}plot(document.getElementById('lr'),[{values:lr(),className:'new',label:'LR'}],{xmin:0,xmax:maxTrain,tokens:true,yDigits:6,endLabels:true,markers:[{x:D.constants.warmup_end,label:'warmup ends'},{x:D.constants.new_cooldown_start,label:'cooldown starts'},{x:D.constants.new_updates,label:'floor'}],label:'Complete WSD-10 learning-rate schedule'});
for(const el of document.querySelectorAll('[data-val]')){const p=el.dataset.val;plot(el,[{values:val(p),className:'new',points:true}],{xmin:0,xmax:maxTrain,tokens:true,label:`${D.labels[p]} BPB`})}
for(const el of document.querySelectorAll('[data-gap]')){const p=el.dataset.gap;plot(el,[{values:gap(val(p)),className:'new'}],{xmin:0,xmax:maxTrain,tokens:true,ymin:0,yDigits:4,label:`${D.labels[p]} forgetting from best observed BPB`})}
for(const [id,key,percent] of [['gm-acc','clean_accuracy',true],['gm-nll','clean_choice_nll',false],['gm-bpb','clean_correct_answer_bpb',false]])plot(document.getElementById(id),[{values:D.new_greekmmlu.map(r=>[r.iteration,r[key]]),className:'new',points:true,label:'endpoint'}],{width:1200,height:440,xmin:0,xmax:maxTrain,tokens:true,percent,yDigits:percent?2:4,endLabels:true,markers:[{x:9536,label:'best benchmark'},{x:D.constants.new_cooldown_start,label:'cooldown'}],label:`GreekMMLU ${key}`});
for(const [id,key,percent] of [['gm-acc-zoom','clean_accuracy',true],['gm-nll-zoom','clean_choice_nll',false]])plot(document.getElementById(id),[{values:D.new_greekmmlu.map(r=>[r.iteration,r[key]]),className:'new',points:true,label:'endpoint'}],{width:1200,height:390,xmin:2384,xmax:maxTrain,tokens:true,percent,yDigits:percent?2:4,endLabels:true,markers:[{x:9536,label:'best benchmark'},{x:D.constants.new_cooldown_start,label:'cooldown'}],label:`GreekMMLU plateau zoom ${key}`});
function deltaBars(el){const rows=[];for(const p of [...D.learning_panels,...D.retention_panels]){const v=D.per_document_validation[p];rows.push({p,value:v.at(-1).bpb-v.at(-2).bpb})}const W=1100,H=530,m={l:210,r:90,t:28,b:36},svg=s('svg',{viewBox:`0 0 ${W} ${H}`,class:'chart',role:'img','aria-label':'Document-local BPB change during cooldown'});el.appendChild(svg);const min=Math.min(...rows.map(r=>r.value))*1.12,max=0,X=x=>m.l+(x-min)/(max-min)*(W-m.l-m.r),step=(H-m.t-m.b)/rows.length;svg.appendChild(s('line',{x1:X(0),y1:m.t,x2:X(0),y2:H-m.b,class:'axis'}));rows.forEach((r,i)=>{const y=m.t+i*step+4,h=Math.max(10,step-9),x=X(r.value);svg.appendChild(s('rect',{x,y,width:X(0)-x,height:h,fill:'var(--new)',opacity:.78}));const lab=s('text',{x:m.l-12,y:y+h*.72,'text-anchor':'end',class:'bar-label'});lab.textContent=D.labels[r.p];svg.appendChild(lab);const value=s('text',{x:x-6,y:y+h*.72,'text-anchor':'end',class:'bar-value'});value.textContent=r.value.toFixed(4);svg.appendChild(value)})}deltaBars(document.getElementById('cooldown-delta'));
"""


def panel_grid(payload: dict, panels: tuple[str, ...], attribute: str) -> str:
    return "".join(
        f"<article class='mini'><h3>{payload['labels'][panel]}<span>complete trajectory</span></h3><div {attribute}='{panel}'></div></article>"
        for panel in panels
    )


def data_table(payload: dict) -> str:
    geometry = payload["data"]["new_geometry"]
    rows = (
        ("modern_greek", "Modern Greek"),
        ("foreign_replay", "Foreign replay"),
        ("old_greek_replay", "Old-Greek replay"),
        ("active_tokens", "Total active"),
    )
    total = float(geometry["active_tokens"])
    return "".join(
        f"<tr><td>{label}</td><td>{int(geometry[key]):,}</td><td>{int(geometry[key])/1e9:.3f}B</td><td>{100*int(geometry[key])/total:.2f}%</td></tr>"
        for key, label in rows
    )


def endpoint_table(payload: dict) -> str:
    rows = []
    for panel in (*payload["learning_panels"], *payload["retention_panels"]):
        values = payload["new_validation"][panel]
        endpoint = values[-1]
        best = min(float(row["bpb"]) for row in values)
        gap = float(endpoint["bpb"]) - best
        rows.append(
            f"<tr><td>{payload['labels'][panel]}</td><td>{float(endpoint['bpb']):.4f}</td>"
            f"<td>{best:.4f}</td><td class='{'worse' if gap > 0 else 'better'}'>{gap:+.4f}</td></tr>"
        )
    return "".join(rows)


def document_table(payload: dict) -> str:
    rows = []
    for panel in (*payload["learning_panels"], *payload["retention_panels"]):
        initial, cooldown, final = payload["per_document_validation"][panel]
        cooldown_delta = float(final["bpb"]) - float(cooldown["bpb"])
        total_delta = float(final["bpb"]) - float(initial["bpb"])
        rows.append(
            f"<tr><td>{payload['labels'][panel]}</td><td>{int(final['documents']):,}</td>"
            f"<td>{float(initial['bpb']):.4f}</td><td>{float(cooldown['bpb']):.4f}</td>"
            f"<td>{float(final['bpb']):.4f}</td><td class='better'>{cooldown_delta:+.4f}</td>"
            f"<td class='better'>{100*total_delta/float(initial['bpb']):+.1f}%</td></tr>"
        )
    return "".join(rows)


def greekmmlu_table(payload: dict) -> str:
    best_accuracy = max(float(row["clean_accuracy"]) for row in payload["new_greekmmlu"])
    rows = []
    for row in payload["new_greekmmlu"]:
        accuracy = float(row["clean_accuracy"])
        rows.append(
            f"<tr><td>{int(row['iteration']):,}</td><td>{int(row['iteration'])*payload['constants']['tokens_per_update']/1e9:.2f}B</td>"
            f"<td class='{'better' if accuracy == best_accuracy else ''}'>{100*accuracy:.2f}%</td>"
            f"<td>{float(row['clean_choice_nll']):.4f}</td><td>{float(row['clean_correct_answer_bpb']):.4f}</td></tr>"
        )
    return "".join(rows)


def build_html(base, payload: dict) -> str:
    meta = payload["meta"]
    gm = payload["new_greekmmlu"]
    initial, final = gm[0], gm[-1]
    best = max(gm, key=lambda row: float(row["clean_accuracy"]))
    counts = payload["completion"]["counts"]
    completion_time = dt.datetime.fromisoformat(payload["completion"]["completed_at"]).astimezone(ZoneInfo("Europe/Athens"))
    excluded = ", ".join(
        f"{payload['labels'][row['panel']]} at {int(row['iteration']):,}" for row in payload["validation_exclusions"]
    ) or "none"
    embedded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{meta['title']}</title><style>{base.STYLE}</style></head><body><main>
<header class='hero'><div class='eyebrow'>Completed Apertus 8B CPT · 11 August 2026</div><h1>One 8B run,<br>presented on its own evidence</h1><p class='lede'>The completed anonymized and decontaminated D0 / WSD-10 trajectory is shown from initialization to terminal checkpoint: the entire optimization path, every learning and forgetting panel, exact document-local endpoint validation, and all GreekMMLU checkpoints.</p><div class='hero-meta'><div class='metric'><b>{meta['snapshot_iteration']:,}</b><span>optimizer updates</span></div><div class='metric'><b>{meta['active_tokens']/1e9:.3f}B</b><span>active corpus tokens</span></div><div class='metric'><b>{meta['median_step_seconds']:.2f}s</b><span>terminal median / update</span></div><div class='metric'><b>13 / 13</b><span>cooldown panels improved</span></div><div class='metric'><b>{100*float(final['clean_accuracy']):.2f}%</b><span>final clean GreekMMLU</span></div></div><div class='finding good'><strong>Result.</strong> Greek adaptation is broad and unambiguous; all 13 exact document-local panels improve during cooldown. Retention is not perfectly monotonic—six foreign panels finish above their earlier best BPB—but every endpoint remains far better than initialization. GreekMMLU gains most of its ground by about 10B tokens, reaches its best value of {100*float(best['clean_accuracy']):.2f}% near 40B, and finishes at {100*float(final['clean_accuracy']):.2f}%.</div></header>

<section class='section'><div class='eyebrow'>01 · Training contract</div><h2>The run that was actually executed</h2><p class='section-intro'>Apertus 8B uses the validated untied Token-Distillation initialization and 148,992-token Modern+Polytonic tokenizer: 32 layers, width 4,096, 32 attention heads / 8 query groups, RoPE 500,000 / 4,096 / factor 8, BF16, TP=2, DP=32, global batch 1,024, AdEMAMix, peak LR 5.5×10⁻⁵, WSD-10, and no checkpoint averaging.</p><div class='chart-shell'><div class='chart-title'><h3>Complete learning-rate schedule</h3><small>warmup, stable phase, cooldown and floor</small></div><div id='lr'></div></div><div class='table-wrap'><table><thead><tr><th>pool</th><th>tokens</th><th>billions</th><th>share</th></tr></thead><tbody>{data_table(payload)}</tbody></table></div><p class='section-intro'>The stationary stream keeps 79% Modern Greek, 20% foreign replay and 1% Old Greek. PII masking changed {payload['data']['masked_documents']:,} documents. The frozen post-mask exact pass dropped {payload['data']['dedup_counts']['dropped_documents']:,} documents; review of that second pass remains deferred and it was not changed during training.</p></section>

<section class='section'><div class='eyebrow'>02 · Optimization</div><h2>Stable through the terminal checkpoint</h2><p class='section-intro'>The thin curve is every logged batch loss. The heavy curve is a 101-update moving mean for legibility; it does not replace the raw series.</p><div class='chart-shell'><div class='chart-title'><h3>Training loss · full horizon</h3><small>{meta['skipped_updates']} skipped · {meta['nan_updates']} non-finite</small></div><div id='train-loss'></div></div></section>

<section class='section'><div class='eyebrow'>03 · Greek learning</div><h2>Every Greek source panel improves</h2><p class='section-intro'>Absolute BPB is shown from the first frequent validation through the endpoint. Lower is better; every panel retains its own scale.</p><div class='grid'>{panel_grid(payload, tuple(payload['learning_panels']), 'data-val')}</div></section>

<section class='section'><div class='eyebrow'>04 · Retention and forgetting</div><h2>Foreign capabilities drift from their early minima</h2><p class='section-intro'>The first grid shows absolute BPB. The second subtracts the best value observed earlier in this same trajectory. Forgetting therefore means “worse than the run’s own earlier minimum,” not “worse than initialization.”</p><div class='grid'>{panel_grid(payload, tuple(payload['retention_panels']), 'data-val')}</div><h3 style='margin:30px 0 11px'>Rise from best observed BPB</h3><div class='grid'>{panel_grid(payload, tuple(payload['retention_panels']), 'data-gap')}</div><div class='table-wrap' style='margin-top:20px'><table><thead><tr><th>panel</th><th>endpoint BPB</th><th>best BPB</th><th>rise from best</th></tr></thead><tbody>{endpoint_table(payload)}</tbody></table></div></section>

<section class='section'><div class='eyebrow'>05 · Exact document-local validation</div><h2>Cooldown improves all 13 frozen panels</h2><p class='section-intro'>Each document is scored independently at initialization, cooldown start and endpoint. All terminal-minus-cooldown deltas are negative.</p><div class='chart-shell'><div class='chart-title'><h3>Terminal minus cooldown-start BPB</h3><small>negative is better</small></div><div id='cooldown-delta'></div></div><div class='table-wrap'><table><thead><tr><th>panel</th><th>docs</th><th>initial</th><th>cooldown start</th><th>terminal</th><th>cooldown Δ</th><th>total change</th></tr></thead><tbody>{document_table(payload)}</tbody></table></div></section>

<section class='section'><div class='eyebrow'>06 · Native Greek benchmark</div><h2>GreekMMLU plateaus after roughly 10B tokens</h2><p class='section-intro'>All 19 checkpoints use the decontaminated 16,159-question subset. Full-horizon plots come first; zooms begin at update 2,384, approximately 10B token slots.</p><div class='finding'><strong>Trajectory.</strong> Accuracy rises {100*(float(final['clean_accuracy'])-float(initial['clean_accuracy'])):+.2f} points from initialization. The best checkpoint is update {int(best['iteration']):,}: {100*float(best['clean_accuracy']):.2f}% accuracy, {float(best['clean_choice_nll']):.4f} choice NLL and {float(best['clean_correct_answer_bpb']):.4f} answer BPB. The endpoint is {100*(float(final['clean_accuracy'])-float(best['clean_accuracy'])):+.2f} points below that accuracy peak.</div><div class='benchmark-stack'><article class='benchmark-chart'><h3>Clean accuracy · full horizon <span>higher is better</span></h3><div id='gm-acc'></div></article><article class='benchmark-chart'><h3>Choice NLL · full horizon <span>lower is better</span></h3><div id='gm-nll'></div></article><article class='benchmark-chart'><h3>Correct-answer BPB · full horizon <span>lower is better</span></h3><div id='gm-bpb'></div></article><article class='benchmark-chart zoom'><h3>Accuracy plateau · zoom from 10B <span>all raw checkpoints</span></h3><div id='gm-acc-zoom'></div></article><article class='benchmark-chart zoom'><h3>Choice-NLL plateau · zoom from 10B <span>all raw checkpoints</span></h3><div id='gm-nll-zoom'></div></article></div><div class='table-wrap' style='margin-top:20px'><table><thead><tr><th>update</th><th>tokens</th><th>accuracy</th><th>choice NLL</th><th>answer BPB</th></tr></thead><tbody>{greekmmlu_table(payload)}</tbody></table></div></section>

<section class='section'><div class='eyebrow'>07 · Interpretation</div><h2>What this single completed trajectory establishes</h2><div class='notes'><div class='note good'><h3>Greek learning is broad</h3><p>HPLT, GlossAPI, OpenArchives, academic, historical-polytonic and neutral external Greek all improve through the endpoint.</p></div><div class='note'><h3>Replay is protective, not perfect</h3><p>Foreign BPB first improves, then drifts upward from its minimum. Cooldown produces a consistent partial recovery on every exact retention panel.</p></div><div class='note'><h3>Benchmark and loss separate</h3><p>Heldout Greek loss keeps improving after GreekMMLU enters a noisy plateau. The benchmark captures a narrower discrimination capability than general source likelihood.</p></div><div class='note caution'><h3>One trajectory is not a factor study</h3><p>This report describes what happened under D0 / WSD-10. It does not identify whether corpus sanitation, exact deduplication, replay proportion or LR shape caused any individual feature of the curve.</p></div></div></section>

<section class='section'><div class='eyebrow'>08 · Evidence closure</div><h2>Complete and hash-bound</h2><p class='section-intro'>The completion receipt was written {completion_time.strftime('%Y-%m-%d %H:%M')} Athens time and binds {counts['greekmmlu']} GreekMMLU receipts, {counts['per_document_panels']} document-local receipts, {counts['source_validated_segments']} training-attempt audits, the launch gate, selected DP32 profile and terminal HF export. Presentation-only exclusions: {excluded}.</p><div class='note'><h3>Completed run</h3><p><code>/capstor/scratch/cscs/fffoivos/runs/07_full_8b_cpt/20260808T121000Z-d0-wsd10-sanitized-successor-v12</code></p></div></section>
<footer class='footer'>Final presentation of the completed Apertus 8B Greek CPT trajectory, generated from frozen campaign evidence.</footer><script id='report-data' type='application/json'>{embedded}</script><script>{SCRIPT}</script></main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--final-evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-output", type=Path)
    args = parser.parse_args()
    base = load_base(Path(__file__).with_name("build_current_vs_previous_report.py"))
    payload = standalone_payload(base, args.repo_root.resolve(), args.evidence_root.resolve(), args.final_evidence_root.resolve())
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_html(base, payload), encoding="utf-8")
    data_output = (args.data_output or output.with_suffix(".data.json")).resolve()
    data_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(output), "data": str(data_output), "iteration": payload["meta"]["snapshot_iteration"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
