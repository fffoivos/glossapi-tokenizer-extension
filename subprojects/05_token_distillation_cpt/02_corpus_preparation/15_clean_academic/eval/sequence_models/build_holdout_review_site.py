#!/usr/bin/env python3
"""Build a self-contained dual-review site for source-matched ToC/BIB cases."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contract import canonical_json_sha256, sha256_file


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected object")
            rows.append(row)
    return rows


def _json_for_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def build_site(
    *,
    requests: str | Path,
    key: str | Path,
    responses: str | Path | None,
    output: str | Path,
    title: str = "ToC & bibliography holdout review",
) -> dict[str, Any]:
    request_rows = _read_jsonl(requests)
    key_rows = _read_jsonl(key)
    response_rows = _read_jsonl(responses) if responses else []
    request_by_id = {str(row.get("request_id", "")): row for row in request_rows}
    key_by_id = {str(row.get("request_id", "")): row for row in key_rows}
    response_by_id = {str(row.get("request_id", "")): row for row in response_rows}
    if "" in request_by_id or len(request_by_id) != len(request_rows):
        raise ValueError("requests have empty/duplicate IDs")
    if set(request_by_id) != set(key_by_id):
        raise ValueError("request/key inventories differ")
    if response_rows and set(request_by_id) != set(response_by_id):
        raise ValueError("request/response inventories differ")
    cases: list[dict[str, Any]] = []
    for request_id in sorted(request_by_id):
        request = request_by_id[request_id]
        private = key_by_id[request_id]
        cases.append(
            {
                "request_id": request_id,
                "request_sha256": request["request_sha256"],
                "source": private["source"],
                "document_id": private["document_id"],
                "source_doc_id": private.get("source_doc_id"),
                "work_id": private.get("work_id"),
                "n_physical_lines": private.get("n_physical_lines"),
                "stratum": private["stratum"],
                "model_prediction": private["candidate_prediction"],
                "model_spans": private.get("candidate_spans", []),
                "target_abs_idx": request["target_abs_idx"],
                "lines": request["lines"],
                "codex_review": response_by_id.get(request_id),
            }
        )
    legacy_cases = [
        {key: value for key, value in case.items() if key != "n_physical_lines"}
        for case in cases
    ]
    legacy_packet_sha = canonical_json_sha256(legacy_cases)
    packet_sha = canonical_json_sha256(cases)
    payload = {
        "packet_sha256": packet_sha,
        "legacy_packet_sha256": legacy_packet_sha,
        "title": title,
        "cases": cases,
    }
    page = _feed_page(title=title, payload=payload)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(page)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return {
        "schema_version": "academic-structure-holdout-review-site-receipt-v1",
        "status": "passed",
        "case_count": len(cases),
        "source_counts": dict(
            sorted(
                {
                    source: sum(case["source"] == source for case in cases)
                    for source in {case["source"] for case in cases}
                }.items()
            )
        ),
        "packet_sha256": packet_sha,
        "inputs": {
            "requests_sha256": sha256_file(requests),
            "key_sha256": sha256_file(key),
            "responses_sha256": sha256_file(responses) if responses else None,
        },
        "output": {"path": str(destination.resolve()), "sha256": sha256_file(destination)},
    }


def _feed_page(*, title: str, payload: Mapping[str, Any]) -> str:
    safe_title = html.escape(title)
    data = _json_for_script(payload)
    template = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--ink:#17201c;--muted:#6c756f;--paper:#eee9df;--card:#fffdf8;--line:#d8d1c3;--keep:#59625d;--bib:#a33c36;--toc:#28628f;--unknown:#ad751d;--agree:#24714b;--warn:#a86b14;--bad:#9a3434;--shadow:0 14px 38px rgba(38,45,40,.10)}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:linear-gradient(145deg,#e9e3d7,#f8f5ed 48%,#e4ece7);color:var(--ink);font:15px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
button,input,select{font:inherit}.top{position:sticky;top:0;z-index:20;background:rgba(249,246,239,.96);backdrop-filter:blur(16px);border-bottom:1px solid var(--line)}.topin{max-width:1220px;margin:auto;padding:14px 22px 12px}.titleline{display:flex;align-items:end;justify-content:space-between;gap:16px}.titleline h1{font:700 23px/1.1 Georgia,serif;margin:0}.status{font-weight:700}.sub{font-size:12px;color:var(--muted)}
.progress{height:7px;background:#ddd7cb;border-radius:9px;overflow:hidden;margin-top:11px}.progress i{display:block;height:100%;width:0;background:linear-gradient(90deg,#246245,#6f9b7c);transition:.2s}.toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:10px}button{border:0;border-radius:9px;padding:8px 11px;background:#e8e3d8;color:var(--ink);cursor:pointer}button:hover{filter:brightness(.97)}button.ghost{background:#fff;border:1px solid var(--line)}
.feed{max-width:1220px;margin:20px 250px 80px auto;padding:0 22px;display:flex;flex-direction:column;gap:12px}.case{display:grid;grid-template-columns:218px minmax(0,1fr);background:var(--card);border:2px solid transparent;border-radius:15px;box-shadow:var(--shadow);overflow:hidden;min-height:150px;scroll-margin:150px;transition:border-color .18s,transform .18s,opacity .18s}.case.active{border-color:#182b22;transform:translateY(-1px)}.case.reviewed.choice-O{border-color:#9ba39e}.case.reviewed.choice-BIB{border-color:#d58f89}.case.reviewed.choice-TOC{border-color:#83afd0}.case.reviewed.choice-UNKNOWN{border-color:#d7b46f}
.machine{padding:14px;background:#eee9df;border-right:1px solid var(--line);display:flex;flex-direction:column;gap:8px}.pred{border-radius:10px;padding:9px 10px;color:white;font-weight:750;min-height:44px}.pred small{display:block;opacity:.78;font-size:10px;text-transform:uppercase;letter-spacing:.09em}.pred.hiddenpred{background:#cbc5b9;color:#706a61}.lab-O{background:var(--keep)!important}.lab-BIB{background:var(--bib)!important}.lab-TOC{background:var(--toc)!important}.lab-UNKNOWN{background:var(--unknown)!important}.agreement{margin-top:auto;border-radius:9px;padding:7px 9px;font-size:11px;font-weight:800;text-align:center;letter-spacing:.035em;background:#d5d0c5;color:#6b665e}.agreement.all{background:#d7eddf;color:#155c38}.agreement.models{background:#f4e2bd;color:#80500b}.agreement.partial{background:#dce9f4;color:#20577f}.agreement.none{background:#f2d6d3;color:#842d2b}
.content{padding:17px 20px;display:flex;flex-direction:column;justify-content:center}.meta{display:flex;gap:8px;align-items:center;flex-wrap:wrap;color:var(--muted);font-size:12px}.position{font-weight:750;color:#405148}.posbar{width:110px;height:5px;background:#e1ddd3;border-radius:8px;overflow:hidden}.posbar i{display:block;height:100%;background:#527963}.target{font:18px/1.5 Georgia,"Times New Roman",serif;margin:14px 0;overflow-wrap:anywhere}.your{font-size:12px;font-weight:800;color:var(--muted)}.your strong{color:var(--ink)}
.controls{position:fixed;z-index:22;right:14px;top:50%;transform:translateY(-50%);display:grid;grid-template-columns:72px 72px 72px;grid-template-rows:72px 72px 72px;grid-template-areas:". up ." "left . right" ". down .";gap:5px;padding:9px;background:rgba(255,253,248,.96);border:1px solid var(--line);border-radius:20px;box-shadow:0 18px 55px rgba(25,35,29,.24);backdrop-filter:blur(16px)}.choice{color:white;font-weight:800;padding:7px 4px;border-radius:13px}.choice span{display:block;font-size:25px;line-height:1}.choice small{display:block;font-size:10px;line-height:1.1;margin-top:5px}.choice.keep{grid-area:left;background:var(--keep)}.choice.toc{grid-area:up;background:var(--toc)}.choice.bib{grid-area:down;background:var(--bib)}.choice.unknown{grid-area:right;background:var(--unknown)}
.modal{position:fixed;inset:0;z-index:40;background:rgba(20,28,24,.62);padding:28px;display:flex;align-items:center;justify-content:center}.hidden{display:none!important}.panel{width:min(1400px,100%);height:min(900px,94vh);background:var(--card);border-radius:17px;display:flex;flex-direction:column;overflow:hidden;box-shadow:0 30px 90px rgba(0,0,0,.3)}.panelhead{padding:15px 18px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center;gap:12px}.panelhead h2{font:700 21px Georgia,serif;margin:0}.tablewrap{overflow:auto;flex:1}table{border-collapse:collapse;width:100%;font-size:12px}th{position:sticky;top:0;background:#ece7dc;text-align:left;padding:8px;z-index:1}td{padding:8px;border-bottom:1px solid #e8e2d7;vertical-align:top}.excerpt{max-width:560px;font-family:Georgia,"Times New Roman",serif}.result{font-weight:800}.result.all{color:var(--agree)}.result.models{color:var(--warn)}.result.none{color:var(--bad)}
@media(max-width:1000px){.feed{margin:16px 176px 70px 8px;padding:0}.controls{right:7px;grid-template-columns:52px 52px 52px;grid-template-rows:60px 60px 60px}.choice span{font-size:21px}.choice small{font-size:8px}.case{grid-template-columns:124px 1fr}.machine{padding:9px}.content{padding:14px}.target{font-size:16px}.posbar{display:none}}
</style></head><body>
<header class="top"><div class="topin"><div class="titleline"><div><h1>__TITLE__</h1><div class="sub">One highlighted target line per case · machine judgments appear only after yours</div></div><div id="status" class="status"></div></div><div class="progress"><i id="bar"></i></div><div class="toolbar"><button id="resume" class="ghost">Resume undecided</button><button id="summary" class="ghost">Choices table</button><button id="export" class="ghost">Export review</button><button id="importButton" class="ghost">Import review</button><input id="importFile" type="file" accept="application/json" class="hidden"><span class="sub">Keyboard: ← Keep · ↑ ToC · ↓ Bibliography · → Uncertain</span></div></div></header>
<main id="feed" class="feed"></main>
<nav class="controls" aria-label="Arrow-key decisions"><button class="choice keep" data-choice="O"><span>←</span><small>Keep</small></button><button class="choice toc" data-choice="TOC"><span>↑</span><small>ToC</small></button><button class="choice bib" data-choice="BIB"><span>↓</span><small>Bibliography</small></button><button class="choice unknown" data-choice="UNKNOWN"><span>→</span><small>Uncertain</small></button></nav>
<div id="modal" class="modal hidden"><section class="panel"><div class="panelhead"><div><h2>All choices</h2><div id="tableStatus" class="sub"></div></div><button id="close" class="ghost">Close</button></div><div class="tablewrap"><table><thead><tr><th>#</th><th>Position</th><th>Line</th><th>Yours</th><th>C2</th><th>Codex</th><th>Agreement</th><th></th></tr></thead><tbody id="tbody"></tbody></table></div></section></div>
<script>const PACKET=__DATA__;
const $=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const storageKey='tocbib-review:'+PACKET.packet_sha256,legacyKey='tocbib-review:'+PACKET.legacy_packet_sha256;
let saved=localStorage.getItem(storageKey)||localStorage.getItem(legacyKey)||'{}',reviews=JSON.parse(saved),active=0,timer=null;
localStorage.setItem(storageKey,JSON.stringify(reviews));
const label=x=>x==='O'||x==='OTHER'?'KEEP':x||'—',codex=c=>c.codex_review?.label==='OTHER'?'O':c.codex_review?.label;
const target=c=>c.lines.find(l=>l.abs_idx===c.target_abs_idx)?.text||'';
function percent(c){const n=Number(c.n_physical_lines);return Number.isFinite(n)&&n>1?Math.max(0,Math.min(100,100*c.target_abs_idx/(n-1))):null}
function agreement(c,r){if(!r)return{key:'',text:'Predictions hidden'};const a=r.label,b=c.model_prediction,d=codex(c);if(a===b&&a===d)return{key:'all',text:'ALL 3 AGREE'};if(b===d)return{key:'models',text:'C2 + CODEX AGREE'};if(a===b)return{key:'partial',text:'YOU + C2 AGREE'};if(a===d)return{key:'partial',text:'YOU + CODEX AGREE'};return{key:'none',text:'ALL DIFFER'}}
function prediction(who,value,visible){return visible?`<div class="pred lab-${esc(value)}"><small>${who}</small>${esc(label(value))}</div>`:`<div class="pred hiddenpred"><small>${who}</small>hidden</div>`}
function caseHtml(c,i){const r=reviews[c.request_id],p=percent(c),a=agreement(c,r);return`<article id="case-${c.request_id}" class="case ${i===active?'active':''} ${r?'reviewed choice-'+r.label:''}" data-index="${i}"><aside class="machine">${prediction('C2',c.model_prediction,!!r)}${prediction('Codex',codex(c),!!r)}<div class="agreement ${a.key}">${a.text}</div></aside><section class="content"><div class="meta"><b>${i+1} / ${PACKET.cases.length}</b><span>${esc(c.source)}</span><span>line ${c.target_abs_idx}</span><span class="position">${p===null?'position unavailable':p.toFixed(1)+'% through document'}</span><span class="posbar"><i style="width:${p??0}%"></i></span></div><div class="target">${esc(target(c))}</div><div class="your">${r?'YOUR DECISION: <strong>'+esc(label(r.label))+'</strong>':'Use an arrow to decide'}</div></section></article>`}
function updateStatus(){const done=PACKET.cases.filter(c=>reviews[c.request_id]).length;$('status').textContent=done+' / '+PACKET.cases.length;$('bar').style.width=100*done/PACKET.cases.length+'%'}
function renderAll(){$('feed').innerHTML=PACKET.cases.map(caseHtml).join('');updateStatus()}
function refresh(i){const c=PACKET.cases[i],node=$('case-'+c.request_id);if(node)node.outerHTML=caseHtml(c,i);updateStatus()}
function setActive(i,scroll=true){if(i<0||i>=PACKET.cases.length)return;document.querySelector('.case.active')?.classList.remove('active');active=i;const node=$('case-'+PACKET.cases[i].request_id);node?.classList.add('active');if(scroll)node?.scrollIntoView({behavior:'smooth',block:'center'})}
function nextUndecided(after=-1){for(let i=after+1;i<PACKET.cases.length;i++)if(!reviews[PACKET.cases[i].request_id])return i;for(let i=0;i<=after;i++)if(!reviews[PACKET.cases[i].request_id])return i;return-1}
function decide(choice){const c=PACKET.cases[active],action=choice==='BIB'||choice==='TOC';reviews[c.request_id]={schema_version:'academic-structure-user-review-v1',request_id:c.request_id,request_sha256:c.request_sha256,reviewer:'foivos',label:choice,start_abs_idx:action?c.target_abs_idx:null,end_abs_idx:action?c.target_abs_idx:null,should_remove:action,confidence:null,notes:'rapid arrow review',reviewed_at:new Date().toISOString()};localStorage.setItem(storageKey,JSON.stringify(reviews));refresh(active);const next=nextUndecided(active);clearTimeout(timer);if(next>=0)timer=setTimeout(()=>setActive(next,true),180);else timer=setTimeout(openSummary,300)}
function resume(){const next=nextUndecided(-1);setActive(next>=0?next:PACKET.cases.length-1,true)}
function tableHtml(c,i){const r=reviews[c.request_id],a=agreement(c,r),p=percent(c),visible=!!r;return`<tr><td>${i+1}</td><td>${p===null?'—':p.toFixed(1)+'%'}</td><td class="excerpt">${esc(target(c))}</td><td>${esc(label(r?.label))}</td><td>${visible?esc(label(c.model_prediction)):'hidden'}</td><td>${visible?esc(label(codex(c))):'hidden'}</td><td class="result ${a.key}">${esc(a.text)}</td><td><button data-open="${i}">Open</button></td></tr>`}
function openSummary(){$('tbody').innerHTML=PACKET.cases.map(tableHtml).join('');$('tableStatus').textContent=PACKET.cases.filter(c=>reviews[c.request_id]).length+' / '+PACKET.cases.length+' reviewed';$('modal').classList.remove('hidden')}
$('feed').onclick=e=>{const card=e.target.closest('.case');if(card)setActive(Number(card.dataset.index),false)};document.querySelectorAll('[data-choice]').forEach(b=>b.onclick=()=>decide(b.dataset.choice));$('resume').onclick=resume;$('summary').onclick=openSummary;$('close').onclick=()=>$('modal').classList.add('hidden');$('modal').onclick=e=>{if(e.target===$('modal'))$('modal').classList.add('hidden')};$('tbody').onclick=e=>{const b=e.target.closest('[data-open]');if(b){$('modal').classList.add('hidden');setActive(Number(b.dataset.open),true)}};
$('export').onclick=()=>{const out={schema_version:'academic-structure-user-review-export-v1',packet_sha256:PACKET.packet_sha256,legacy_packet_sha256:PACKET.legacy_packet_sha256,exported_at:new Date().toISOString(),review_count:Object.keys(reviews).length,reviews:Object.values(reviews).sort((a,b)=>a.request_id.localeCompare(b.request_id))};const blob=new Blob([JSON.stringify(out,null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='foivos-toc-bib-review-'+PACKET.packet_sha256.slice(0,12)+'.json';a.click();URL.revokeObjectURL(a.href)};
$('importButton').onclick=()=>$('importFile').click();$('importFile').onchange=e=>{const f=e.target.files[0];if(!f)return;const reader=new FileReader();reader.onload=()=>{try{const x=JSON.parse(reader.result);if(![PACKET.packet_sha256,PACKET.legacy_packet_sha256].includes(x.packet_sha256))throw Error('packet hash differs');reviews=Object.fromEntries((x.reviews||[]).map(r=>[r.request_id,r]));localStorage.setItem(storageKey,JSON.stringify(reviews));renderAll();resume()}catch(err){alert(err.message)}};reader.readAsText(f)};
document.onkeydown=e=>{if(e.key==='Escape'){$('modal').classList.add('hidden');return}if(['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName))return;const map={ArrowLeft:'O',ArrowUp:'TOC',ArrowDown:'BIB',ArrowRight:'UNKNOWN'};if(map[e.key]){e.preventDefault();decide(map[e.key])}};
renderAll();active=Math.max(0,nextUndecided(-1));setTimeout(()=>setActive(active,true),80);
</script></body></html>"""
    return template.replace("__TITLE__", safe_title).replace("__DATA__", data)


def _page(*, title: str, payload: Mapping[str, Any]) -> str:
    safe_title = html.escape(title)
    data = _json_for_script(payload)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{safe_title}</title>
<style>
:root{{--ink:#17201c;--muted:#67716c;--paper:#f4f0e7;--card:#fffdf7;--green:#174f3b;--green2:#dbe9df;--red:#922f2f;--blue:#1d4f78;--gold:#b87716;--line:#d8d2c5;--shadow:0 16px 45px rgba(40,47,42,.10)}}
*{{box-sizing:border-box}} body{{margin:0;background:linear-gradient(135deg,#ebe5d8 0,#f8f5ed 48%,#e7eee9 100%);color:var(--ink);font:15px/1.48 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;min-height:100vh}}
button,input,select,textarea{{font:inherit}} .top{{position:sticky;top:0;z-index:10;background:rgba(248,245,237,.94);backdrop-filter:blur(16px);border-bottom:1px solid var(--line)}}
.topin{{max-width:1480px;margin:auto;padding:17px 24px 13px}} h1{{font:700 24px/1.1 Georgia,serif;margin:0 0 4px}} .sub{{color:var(--muted);font-size:13px}}
.progress{{height:7px;background:#ded9cd;border-radius:9px;overflow:hidden;margin-top:13px}} .progress i{{display:block;height:100%;background:linear-gradient(90deg,var(--green),#5d8f72);width:0;transition:.25s}}
.toolbar{{display:flex;gap:9px;flex-wrap:wrap;margin-top:12px;align-items:center}} select,.search{{background:#fff;border:1px solid var(--line);border-radius:9px;padding:8px 10px;color:var(--ink)}} .search{{min-width:220px}}
.layout{{max-width:1480px;margin:22px auto;padding:0 24px;display:grid;grid-template-columns:minmax(0,1fr) 355px;gap:20px}} .card{{background:var(--card);border:1px solid rgba(132,125,108,.26);border-radius:16px;box-shadow:var(--shadow)}}
.casehead{{padding:18px 20px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:12px;align-items:flex-start}} .eyebrow{{font-size:12px;text-transform:uppercase;letter-spacing:.09em;color:var(--muted)}} .casehead h2{{margin:3px 0 0;font:700 22px/1.2 Georgia,serif}} .badges{{display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end}} .badge{{padding:4px 8px;border-radius:999px;font-size:12px;background:#ede8dc}} .badge.BIB{{background:#f6dedb;color:#7b2424}} .badge.TOC{{background:#dce9f5;color:#173f61}} .badge.O{{background:#e6e8e5;color:#49524e}}
.context{{padding:12px 0;max-height:62vh;overflow:auto}} .line{{display:grid;grid-template-columns:74px 1fr;gap:12px;padding:6px 18px;border-left:4px solid transparent;white-space:pre-wrap;overflow-wrap:anywhere}} .line:hover{{background:#f4f0e7}} .line.target{{background:#fff0bd;border-left-color:var(--gold)}} .ln{{font:12px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;color:#8a8479;text-align:right;user-select:none}} .txt{{font-family:Georgia,"Times New Roman",serif;font-size:15px}}
.nav{{padding:14px 18px;border-top:1px solid var(--line);display:flex;justify-content:space-between;align-items:center}} button{{border:0;border-radius:9px;padding:9px 12px;cursor:pointer;background:#ebe7dc;color:var(--ink)}} button:hover{{filter:brightness(.97)}} button.primary{{background:var(--green);color:white}} button.ghost{{background:transparent;border:1px solid var(--line)}}
.side{{padding:18px;position:sticky;top:150px;align-self:start}} .side h3{{font:700 18px Georgia,serif;margin:0 0 5px}} .prompt{{color:var(--muted);font-size:13px;margin-bottom:15px}} .decisions{{display:grid;grid-template-columns:1fr 1fr;gap:8px}} .decision{{border:1px solid var(--line);background:#fff;padding:11px}} .decision.selected{{border-color:var(--green);box-shadow:0 0 0 2px #b8d3c2;background:#eef6f0}} .decision.keep.selected{{border-color:#505b55;box-shadow:0 0 0 2px #cbd0cc}} .decision.bib.selected{{border-color:var(--red);box-shadow:0 0 0 2px #e6b7b2}} .decision.toc.selected{{border-color:var(--blue);box-shadow:0 0 0 2px #b9d0e3}}
.fields{{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px}} label{{font-size:12px;color:var(--muted)}} label input,label textarea{{display:block;width:100%;margin-top:4px;border:1px solid var(--line);border-radius:8px;background:#fff;padding:8px}} label textarea{{min-height:80px;resize:vertical}} .wide{{grid-column:1/-1}} .save{{width:100%;margin-top:12px}}
.codex{{margin-top:16px;border-top:1px solid var(--line);padding-top:14px}} .codexbody{{margin-top:8px;padding:11px;border-radius:10px;background:#eeeae0;font-size:13px}} .hidden{{display:none!important}} .summary{{display:flex;justify-content:space-between;font-size:13px;color:var(--muted);margin:10px 0}} .kbd{{font:11px ui-monospace,SFMono-Regular,monospace;border:1px solid #c9c3b8;background:#fff;border-radius:4px;padding:1px 4px}}
.modal{{position:fixed;inset:0;z-index:30;background:rgba(20,28,24,.58);padding:30px;display:flex;align-items:center;justify-content:center}} .modalpanel{{width:min(1450px,100%);height:min(900px,94vh);background:var(--card);border-radius:18px;box-shadow:0 30px 90px rgba(0,0,0,.28);display:flex;flex-direction:column;overflow:hidden}} .modalhead{{padding:17px 20px;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center;justify-content:space-between}} .modalhead h2{{font:700 22px Georgia,serif;margin:0}} .tabletools{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}} .tablewrap{{overflow:auto;flex:1}} table{{border-collapse:collapse;width:100%;font-size:13px}} th{{position:sticky;top:0;background:#ece7dc;z-index:1;text-align:left;padding:9px;border-bottom:1px solid var(--line)}} td{{padding:8px 9px;border-bottom:1px solid #e8e3d8;vertical-align:top}} tr:hover td{{background:#f7f2e8}} .excerpt{{font-family:Georgia,"Times New Roman",serif;max-width:520px}} .result{{font-weight:650}} .result.both{{color:var(--green)}} .result.neither{{color:var(--red)}}
@media(max-width:900px){{.layout{{grid-template-columns:1fr}}.side{{position:static}}.context{{max-height:none}}}}
</style></head>
<body><div class="top"><div class="topin"><h1>{safe_title}</h1><div class="sub">Unseen works from the same canonical Greek academic sources · C2, Codex and the selection stratum stay hidden until you save your own judgment.</div><div class="progress"><i id="bar"></i></div>
<div class="toolbar"><select id="source"><option value="">All sources</option></select><select id="status"><option value="">All cases</option><option value="unreviewed">Unreviewed</option><option value="reviewed">Reviewed</option><option value="disagreement">You disagree with C2</option></select><input id="search" class="search" placeholder="Document ID or text"><button id="tableButton" class="ghost">Review table</button><button id="export" class="ghost">Export my review</button><button id="importButton" class="ghost">Import review</button><input id="importFile" type="file" accept="application/json" class="hidden"></div></div></div>
<main class="layout"><section class="card"><div class="casehead"><div><div class="eyebrow" id="counter"></div><h2 id="caseTitle"></h2><div class="sub" id="identity"></div></div><div class="badges"><span id="modelBadge" class="badge"></span><span id="stratumBadge" class="badge"></span></div></div><div id="context" class="context"></div><div class="nav"><button id="prev">← Previous</button><span class="sub"><span class="kbd">←</span> <span class="kbd">→</span> navigate · <span class="kbd">1–4</span> classify</span><button id="next" class="primary">Next →</button></div></section>
<aside class="card side"><h3>Your independent judgment</h3><div class="prompt">What does the highlighted line belong to? Judge the whole visible block, including whether it is safe to remove.</div><div class="decisions"><button class="decision keep" data-label="O">1 · Keep / other</button><button class="decision bib" data-label="BIB">2 · Bibliography</button><button class="decision toc" data-label="TOC">3 · Table of contents</button><button class="decision" data-label="UNKNOWN">4 · Uncertain</button></div><div class="fields"><label>Start line<input id="start" type="number" min="0"></label><label>End line<input id="end" type="number" min="0"></label><label>Confidence<input id="confidence" type="number" min="0" max="1" step=".05" value="0.9"></label><label class="wide">Notes<textarea id="notes" placeholder="Boundary correction, false-positive pattern, ambiguity…"></textarea></label></div><button id="save" class="primary save">Save &amp; reveal</button><div class="summary"><span id="reviewed"></span><span id="agreement"></span></div><div class="codex"><button id="reveal" class="ghost">Reveal comparisons</button><div id="codexBody" class="codexbody hidden"></div></div></aside></main>
<div id="reviewModal" class="modal hidden"><section class="modalpanel"><div class="modalhead"><div><h2>Review choices</h2><div id="tableCount" class="sub"></div></div><div class="tabletools"><select id="tableFilter"><option value="">All cases</option><option value="reviewed">Reviewed</option><option value="unreviewed">Unreviewed</option><option value="disagreement">Any disagreement</option></select><input id="tableSearch" class="search" placeholder="Search source, line, or excerpt"><button id="closeTable" class="ghost">Close</button></div></div><div class="tablewrap"><table><thead><tr><th>#</th><th>Source</th><th>Line</th><th>Target excerpt</th><th>Yours</th><th>C2</th><th>Codex</th><th>Agreement</th><th></th></tr></thead><tbody id="tableBody"></tbody></table></div></section></div>
<script>const PACKET={data};
const storageKey='tocbib-review:'+PACKET.packet_sha256;let reviews=JSON.parse(localStorage.getItem(storageKey)||'{{}}');let filtered=[];let index=0;let chosen=null;
const $=id=>document.getElementById(id), esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function options(id,values){{for(const v of [...new Set(values)].sort()){{const o=document.createElement('option');o.value=v;o.textContent=v;$(id).appendChild(o)}}}}
options('source',PACKET.cases.map(x=>x.source));
function applyFilters(){{const s=$('source').value,st=$('status').value,q=$('search').value.toLowerCase();filtered=PACKET.cases.filter(c=>{{const r=reviews[c.request_id];const disagreement=r&&r.label!=='UNKNOWN'&&r.label!==c.model_prediction;return(!s||c.source===s)&&(!st||(st==='reviewed'&&r)||(st==='unreviewed'&&!r)||(st==='disagreement'&&disagreement))&&(!q||JSON.stringify([c.document_id,c.source_doc_id,c.lines]).toLowerCase().includes(q))}});index=Math.min(index,Math.max(0,filtered.length-1));render()}}
function current(){{return filtered[index]}}
function render(){{const c=current();const done=Object.keys(reviews).length;$('bar').style.width=(100*done/PACKET.cases.length)+'%';$('reviewed').textContent=done+' / '+PACKET.cases.length+' reviewed';$('tableButton').textContent='Review table · '+done+'/'+PACKET.cases.length;if(!c){{$('caseTitle').textContent='No cases match';$('context').innerHTML='';return}};$('counter').textContent='Case '+(index+1)+' of '+filtered.length;$('caseTitle').textContent=c.source.replace('_',' ')+' · line '+c.target_abs_idx;$('identity').textContent=(c.source_doc_id||c.work_id||c.document_id).slice(0,90);$('modelBadge').className='badge';$('modelBadge').textContent='C2 hidden';$('stratumBadge').textContent='stratum hidden';const context=$('context');context.innerHTML=c.lines.map(l=>`<div class="line ${{l.abs_idx===c.target_abs_idx?'target':''}}"><span class="ln">L${{l.abs_idx}}</span><span class="txt">${{esc(l.text)}}</span></div>`).join('');requestAnimationFrame(()=>{{const target=context.querySelector('.target');if(target){{const top=target.getBoundingClientRect().top-context.getBoundingClientRect().top+context.scrollTop;context.scrollTop=Math.max(0,top-(context.clientHeight-target.offsetHeight)/2)}}}});const r=reviews[c.request_id]||{{}};chosen=r.label||null;document.querySelectorAll('.decision').forEach(b=>b.classList.toggle('selected',b.dataset.label===chosen));$('start').value=r.start_abs_idx??'';$('end').value=r.end_abs_idx??'';$('confidence').value=r.confidence??.9;$('notes').value=r.notes??'';$('codexBody').classList.add('hidden');$('reveal').textContent='Reveal comparisons';const agree=Object.entries(reviews).filter(([id,r])=>{{const x=PACKET.cases.find(c=>c.request_id===id);return x&&r.label===x.model_prediction}}).length;$('agreement').textContent=done?Math.round(100*agree/done)+'% agree with C2':'';$('prev').disabled=index===0;$('next').disabled=index>=filtered.length-1}}
function choose(label){{chosen=label;document.querySelectorAll('.decision').forEach(b=>b.classList.toggle('selected',b.dataset.label===label));const c=current();if(label==='BIB'||label==='TOC'){{$('start').value=c.target_abs_idx;$('end').value=c.target_abs_idx}}else{{$('start').value='';$('end').value=''}}}}
function save(){{const c=current();if(!c||!chosen){{alert('Choose a judgment first.');return}};const action=chosen==='BIB'||chosen==='TOC';const a=$('start').value,b=$('end').value;if(action&&(!a||!b)){{alert('Set inclusive span bounds for BIB/TOC.');return}};reviews[c.request_id]={{schema_version:'academic-structure-user-review-v1',request_id:c.request_id,request_sha256:c.request_sha256,reviewer:'foivos',label:chosen,start_abs_idx:action?Number(a):null,end_abs_idx:action?Number(b):null,should_remove:action,confidence:Number($('confidence').value),notes:$('notes').value,reviewed_at:new Date().toISOString()}};localStorage.setItem(storageKey,JSON.stringify(reviews));render();$('reveal').click()}}
document.querySelectorAll('.decision').forEach(b=>b.onclick=()=>choose(b.dataset.label));$('save').onclick=save;$('prev').onclick=()=>{{if(index>0)index--;render()}};$('next').onclick=()=>{{if(index<filtered.length-1)index++;render()}};
$('reveal').onclick=()=>{{const c=current(),mine=reviews[c.request_id];if(!mine){{alert('Save your own judgment before revealing the comparisons.');return}};const box=$('codexBody'),r=c.codex_review;box.classList.toggle('hidden');const hidden=box.classList.contains('hidden');$('reveal').textContent=hidden?'Reveal comparisons':'Hide comparisons';$('modelBadge').className=hidden?'badge':'badge '+c.model_prediction;$('modelBadge').textContent=hidden?'C2 hidden':'C2: '+(c.model_prediction==='O'?'KEEP':c.model_prediction);$('stratumBadge').textContent=hidden?'stratum hidden':c.stratum.replaceAll('_',' ');box.innerHTML=`<b>C2:</b> ${{esc(c.model_prediction)}}<br><b>Codex:</b> ${{r?esc(r.label)+' · confidence '+r.confidence:'not run yet'}}${{r?'<br>'+esc((r.structural_cues||[]).join(' · '))+'<br><span class="sub">span '+(r.start_abs_idx??'—')+'–'+(r.end_abs_idx??'—')+'</span>':''}}`}};
const showLabel=x=>x==='O'||x==='OTHER'?'KEEP':x||'—',codexLabel=c=>c.codex_review?.label==='OTHER'?'O':c.codex_review?.label;
function tableResult(c,r){{if(!r)return'—';const hits=[];if(r.label===c.model_prediction)hits.push('C2');if(r.label===codexLabel(c))hits.push('Codex');return hits.length===2?'Both':hits.length?hits[0]:'Neither'}}
function renderTable(){{const q=$('tableSearch').value.toLowerCase(),filter=$('tableFilter').value;let shown=0;$('tableBody').innerHTML=PACKET.cases.map((c,i)=>{{const r=reviews[c.request_id],result=tableResult(c,r),target=c.lines.find(l=>l.abs_idx===c.target_abs_idx),matches=!filter||(filter==='reviewed'&&r)||(filter==='unreviewed'&&!r)||(filter==='disagreement'&&r&&result!=='Both');if(!matches||q&&!JSON.stringify([c.source,c.target_abs_idx,target?.text,r?.label]).toLowerCase().includes(q))return'';shown++;const machine=r?showLabel(c.model_prediction):'hidden',codex=r?showLabel(codexLabel(c)):'hidden';return`<tr><td>${{i+1}}</td><td>${{esc(c.source)}}</td><td>L${{c.target_abs_idx}}</td><td class="excerpt">${{esc(target?.text||'')}}</td><td>${{esc(showLabel(r?.label))}}</td><td>${{esc(machine)}}</td><td>${{esc(codex)}}</td><td class="result ${{result==='Both'?'both':result==='Neither'?'neither':''}}">${{esc(result)}}</td><td><button data-open-case="${{c.request_id}}">Open case</button></td></tr>`}}).join('');$('tableCount').textContent=shown+' shown · '+Object.keys(reviews).length+' / '+PACKET.cases.length+' reviewed';document.querySelectorAll('[data-open-case]').forEach(b=>b.onclick=()=>openCase(b.dataset.openCase))}}
function openCase(requestId){{$('source').value='';$('status').value='';$('search').value='';applyFilters();index=filtered.findIndex(c=>c.request_id===requestId);$('reviewModal').classList.add('hidden');render();window.scrollTo({{top:0,behavior:'smooth'}})}}
$('tableButton').onclick=()=>{{renderTable();$('reviewModal').classList.remove('hidden')}};$('closeTable').onclick=()=>$('reviewModal').classList.add('hidden');$('tableFilter').onchange=renderTable;$('tableSearch').oninput=renderTable;$('reviewModal').onclick=e=>{{if(e.target===$('reviewModal'))$('reviewModal').classList.add('hidden')}};
['source','status'].forEach(id=>$(id).onchange=applyFilters);$('search').oninput=applyFilters;
$('export').onclick=()=>{{const out={{schema_version:'academic-structure-user-review-export-v1',packet_sha256:PACKET.packet_sha256,exported_at:new Date().toISOString(),review_count:Object.keys(reviews).length,reviews:Object.values(reviews).sort((a,b)=>a.request_id.localeCompare(b.request_id))}};const blob=new Blob([JSON.stringify(out,null,2)],{{type:'application/json'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='foivos-toc-bib-review-'+PACKET.packet_sha256.slice(0,12)+'.json';a.click();URL.revokeObjectURL(a.href)}};
$('importButton').onclick=()=>$('importFile').click();$('importFile').onchange=e=>{{const f=e.target.files[0];if(!f)return;const reader=new FileReader();reader.onload=()=>{{try{{const x=JSON.parse(reader.result);if(x.packet_sha256!==PACKET.packet_sha256)throw Error('packet hash differs');reviews=Object.fromEntries((x.reviews||[]).map(r=>[r.request_id,r]));localStorage.setItem(storageKey,JSON.stringify(reviews));render()}}catch(err){{alert(err.message)}}}};reader.readAsText(f)}};
document.onkeydown=e=>{{if(e.key==='Escape')$('reviewModal').classList.add('hidden');if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName))return;if(e.key==='ArrowLeft')$('prev').click();if(e.key==='ArrowRight')$('next').click();if('1234'.includes(e.key))choose(['O','BIB','TOC','UNKNOWN'][Number(e.key)-1])}};
applyFilters();</script></body></html>"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--responses")
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="ToC & bibliography holdout review")
    args = parser.parse_args(argv)
    receipt = build_site(
        requests=args.requests,
        key=args.key,
        responses=args.responses,
        output=args.output,
        title=args.title,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
