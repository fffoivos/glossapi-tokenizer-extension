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
                "stratum": private["stratum"],
                "model_prediction": private["candidate_prediction"],
                "model_spans": private.get("candidate_spans", []),
                "target_abs_idx": request["target_abs_idx"],
                "lines": request["lines"],
                "codex_review": response_by_id.get(request_id),
            }
        )
    packet_sha = canonical_json_sha256(cases)
    payload = {"packet_sha256": packet_sha, "title": title, "cases": cases}
    page = _page(title=title, payload=payload)
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
