#!/usr/bin/env python3
"""Build and serve the private Agent 1 v4 raw-review site on CSCS."""

from __future__ import annotations

import argparse
import functools
import hashlib
import hmac
import json
import os
import shutil
import stat
import tempfile
from collections import Counter, defaultdict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SCRIPT_DIR))

from agent1_v4_raw_review import (  # noqa: E402
    PACKET_SCHEMA,
    file_binding,
    read_json_object,
    sha256_file,
    validate_packet,
)
from run_agent1_v4_terra_reviews import validate_response  # noqa: E402


SITE_SCHEMA = "agent1_v4_raw_review_site_manifest_v1"
SITE_INDEX_SCHEMA = "agent1_v4_raw_review_site_index_v1"
HEX_SHA256_RE = __import__("re").compile(r"^[0-9a-f]{64}$")
DEFAULT_MAX_PORTABLE_ASSETS_BYTES = 16 * 1024 * 1024


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_file(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        os.chmod(path, mode)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: object) -> None:
    _write_file(path, (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _relative_document(root: Path, request: Mapping[str, object]) -> Path:
    relative = Path(str(request["document_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("review document path escapes packet root")
    document = root / relative
    if document.is_symlink() or not document.is_file():
        raise FileNotFoundError(f"review document missing/symlinked: {relative}")
    return document


def _opaque_id(secret: bytes, request_id: str) -> str:
    return hmac.new(secret, request_id.encode("ascii"), hashlib.sha256).hexdigest()


def _validate_inputs(
    packet_root: Path,
    packet_manifest: Path,
    requests_path: Path,
    responses_path: Path,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, dict[str, object]]]:
    manifest = validate_packet(packet_root, packet_manifest)
    declared_requests = manifest.get("requests")
    actual_requests = file_binding(requests_path)
    if not isinstance(declared_requests, Mapping) or actual_requests["sha256"] != declared_requests.get("sha256") or actual_requests["bytes"] != declared_requests.get("bytes"):
        raise ValueError("site request input differs from packet manifest")
    requests = _read_jsonl(requests_path)
    responses = _read_jsonl(responses_path)
    by_request = {str(request["request_id"]): request for request in requests}
    expected_count = manifest.get("logical_review_count")
    if not isinstance(expected_count, int) or len(by_request) != expected_count or len(responses) != expected_count:
        raise ValueError("site request/response count differs from the packet manifest")
    by_response: dict[str, dict[str, object]] = {}
    for response in responses:
        request_id = str(response.get("request_id") or "")
        request = by_request.get(request_id)
        if request is None or request_id in by_response:
            raise ValueError("response does not close one unique packet request")
        validate_response(response, request, _relative_document(packet_root, request))
        by_response[request_id] = response
    if set(by_request) != set(by_response):
        raise ValueError("request/response closure failed")
    return manifest, requests, by_response


def _index_card(request: Mapping[str, object], response: Mapping[str, object], opaque_id: str) -> dict[str, object]:
    locator = request["origin_locator"]
    assert isinstance(locator, Mapping)
    return {
        "request_id": request["request_id"],
        "opaque_id": opaque_id,
        "source_id": request["source_id"],
        "source_dataset": request["source_dataset"],
        "source_doc_id": request["source_doc_id"],
        "source_revision": request["source_revision"],
        "source_route": request["source_route"],
        "extraction_route": request["extraction_route"],
        "document_path": request["document_path"],
        "document_bytes": request["document_bytes"],
        "document_line_count": request["document_line_count"],
        "origin_locator": dict(locator),
        "cleanliness_score": response["cleanliness_score"],
        "text_quality_score": response["text_quality_score"],
        "confidence": response["confidence"],
        "coverage_mode": response["coverage_mode"],
        "summary": response["summary"],
        "extraction_artifacts": response["extraction_artifacts"],
    }


def _artifact_count(card: Mapping[str, object]) -> Counter[str]:
    artifacts = card["extraction_artifacts"]
    assert isinstance(artifacts, list)
    return Counter(str(artifact["type"]) for artifact in artifacts if isinstance(artifact, Mapping))


def _html() -> str:
    return """<!doctype html>
<html lang="el">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; connect-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; base-uri 'none'; form-action 'none'">
  <title>Apertus raw-document review</title>
  <link rel="stylesheet" href="assets/site.css">
</head>
<body>
  <header><h1>Apertus raw-document review</h1><p><a class="header-link" href="detections.html">VLM repetition detections</a> · <a class="header-link" href="normalization.html">HTML → GitHub Markdown normalization</a></p><p id="coverage"></p><div id="overview"></div></header>
  <main>
    <section class="controls" aria-label="Document filters">
      <label>Source <select id="source-filter"></select></label>
      <label>Cleanliness ≥ <select id="cleanliness-filter"><option value="1">1</option><option value="2">2</option><option value="3">3</option><option value="4">4</option><option value="5">5</option></select></label>
      <label>Text quality ≥ <select id="quality-filter"><option value="1">1</option><option value="2">2</option><option value="3">3</option><option value="4">4</option><option value="5">5</option></select></label>
      <label>Confidence <select id="confidence-filter"></select></label>
      <label>Artifact <select id="artifact-filter"></select></label>
    </section>
    <section class="layout">
      <nav aria-label="Documents"><ol id="document-list"></ol></nav>
      <article id="document" aria-live="polite"><p>Select a document.</p></article>
    </section>
    <section class="decisions">
      <h2>Human decision export</h2>
      <label>Source status <select id="source-status"><option value="">not decided</option><option value="admit">admit</option><option value="hold">hold</option><option value="exclude">exclude</option></select></label>
      <label>Source cleaning observation <textarea id="source-observation" rows="3"></textarea></label>
      <label>Text/title/author mapping question <textarea id="mapping-question" rows="3"></textarea></label>
      <label>Document disposition <select id="document-disposition"><option value="unreviewed">not decided</option><option value="agree">agree</option><option value="override">override scores</option><option value="flag">flag for follow-up</option><option value="hold">hold</option></select></label>
      <label>Override cleanliness <select id="cleanliness-override"><option value="">use Terra score</option><option value="1">1</option><option value="2">2</option><option value="3">3</option><option value="4">4</option><option value="5">5</option></select></label>
      <label>Override text quality <select id="quality-override"><option value="">use Terra score</option><option value="1">1</option><option value="2">2</option><option value="3">3</option><option value="4">4</option><option value="5">5</option></select></label>
      <label>Document note <textarea id="note" rows="3"></textarea></label>
      <label><input id="field-discovery-approval" type="checkbox"> I approve field discovery for explicitly admitted sources.</label>
      <button id="agree" type="button">Agree with Terra scores</button>
      <button id="export" type="button">Download decision bundle</button>
    </section>
  </main>
  <script src="assets/site.js"></script>
  <script src="assets/html-render.js"></script>
  <script src="assets/vlm-repetition.js"></script>
</body>
</html>
"""


def _css() -> str:
    return """*{box-sizing:border-box}body{margin:0;background:#f8fafc;color:#152238;font:16px/1.45 system-ui,-apple-system,sans-serif}header,main{max-width:1500px;margin:auto;padding:1rem 1.25rem}header{background:#102a43;color:#fff;max-width:none;padding-left:max(1.25rem,calc((100% - 1500px)/2 + 1.25rem))}.header-link{color:#dbeafe}.controls{display:flex;gap:1rem;flex-wrap:wrap;background:#e8f1fa;padding:1rem}.controls label,.decisions label{display:grid;gap:.3rem}.layout{display:grid;grid-template-columns:minmax(260px,28%) 1fr;gap:1rem;margin-top:1rem}.layout nav{max-height:74vh;overflow:auto;background:#fff;border:1px solid #cbd5e1}.layout ol{margin:0;padding:0;list-style:none}.layout button{width:100%;text-align:left;border:0;border-bottom:1px solid #e2e8f0;background:#fff;padding:.7rem;cursor:pointer}.layout button:hover,.layout button:focus{background:#dceeff}.card{background:#fff;border:1px solid #cbd5e1;padding:1rem;min-width:0}.meta{display:grid;grid-template-columns:max-content 1fr;gap:.25rem .75rem}.score{display:inline-block;border-radius:.25rem;padding:.15rem .45rem;background:#dbeafe;margin-right:.5rem}.artifact,.vlm-audit{border-left:4px solid #c2410c;padding:.5rem .75rem;margin:.7rem 0;background:#fff7ed}.vlm-audit.clear{border-left-color:#15803d;background:#f0fdf4}.artifact pre,.raw{white-space:pre-wrap;overflow-wrap:anywhere;background:#0b1220;color:#e5edf7;padding:1rem}.document-view-controls{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap;margin:.75rem 0}.document-view-controls button{width:auto;text-align:center;border:1px solid #94a3b8;border-radius:.25rem;background:#fff;padding:.35rem .6rem}.document-view-controls button[aria-pressed=true]{background:#1f4d78;color:#fff}.html-render-frame{width:100%;height:48rem;border:1px solid #94a3b8;background:#fff}.decisions{margin-top:1rem;background:#fff;border:1px solid #cbd5e1;padding:1rem;display:grid;gap:.7rem}.decisions button{width:max-content;padding:.5rem .8rem}#overview{display:flex;gap:.5rem;flex-wrap:wrap;font-size:.85rem}#overview span{background:#1f4d78;padding:.2rem .45rem;border-radius:.2rem}@media(max-width:850px){.layout{grid-template-columns:1fr}.layout nav{max-height:30vh}.html-render-frame{height:36rem}}"""


def _js() -> str:
    return """'use strict';
const state={index:null,cards:[],visible:[],source_counts:{},decisions:{schema_version:'agent1_v4_human_decision_bundle_v1',packet_manifest_sha256:'',approval_to_begin_field_discovery:false,source_status:{},source_observations:{},mapping_questions:{},documents:{}}};
const $=id=>document.getElementById(id);
const text=(tag,value)=>{const node=document.createElement(tag);node.textContent=String(value??'');return node;};
function select(id,values,label){const el=$(id);el.replaceChildren();el.append(new Option(label,''));for(const value of values)el.append(new Option(value,value));}
function sources(){return [...new Set(state.cards.map(card=>card.source_id))].sort();}
function initializeDecisions(){for(const source of sources()){state.decisions.source_status[source]='';state.decisions.source_observations[source]='';state.decisions.mapping_questions[source]='';}for(const card of state.cards){state.decisions.documents[card.request_id]={source_id:card.source_id,source_doc_id:card.source_doc_id,disposition:'unreviewed',cleanliness_score_override:null,text_quality_score_override:null,note:''};}}
function sourceSummary(){const ids=sources(),artifacts=[...new Set(state.cards.flatMap(card=>card.extraction_artifacts.map(artifact=>artifact.type)))].sort();select('source-filter',ids,'all sources');select('artifact-filter',artifacts,'all artifacts');select('confidence-filter',['low','medium','high'],'all confidence');$('coverage').textContent=`${ids.length} sources · ${state.cards.length} reviewed raw documents`;const overview=$('overview');overview.replaceChildren();for(const source of ids){const cards=state.cards.filter(card=>card.source_id===source),artifactCount=cards.reduce((n,card)=>n+card.extraction_artifacts.length,0);overview.append(text('span',`${source}: ${cards.length}/${state.source_counts[source]??cards.length} · ${artifactCount} artifacts`));}}
function filter(){const source=$('source-filter').value,artifact=$('artifact-filter').value,confidence=$('confidence-filter').value,clean=Number($('cleanliness-filter').value),quality=Number($('quality-filter').value);state.visible=state.cards.filter(card=>(!source||card.source_id===source)&&(!artifact||card.extraction_artifacts.some(item=>item.type===artifact))&&(!confidence||card.confidence===confidence)&&card.cleanliness_score>=clean&&card.text_quality_score>=quality);renderList();if(state.visible.length)show(state.visible[0]);else $('document').replaceChildren(text('p','No documents match these filters.'));}
function renderList(){const list=$('document-list');list.replaceChildren();for(const card of state.visible){const button=document.createElement('button');button.type='button';button.append(text('strong',card.source_id),document.createElement('br'),text('span',`clean ${card.cleanliness_score}/5 · quality ${card.text_quality_score}/5`));button.addEventListener('click',()=>show(card));const item=document.createElement('li');item.append(button);list.append(item);}}
function labeled(label,value){return [text('dt',label),text('dd',value)];}
function scoreValue(id){const value=$(id).value;return value===''?null:Number(value);}
function setDecisionControls(card){const decision=state.decisions.documents[card.request_id];$('source-status').value=state.decisions.source_status[card.source_id];$('source-observation').value=state.decisions.source_observations[card.source_id];$('mapping-question').value=state.decisions.mapping_questions[card.source_id];$('document-disposition').value=decision.disposition;$('cleanliness-override').value=decision.cleanliness_score_override??'';$('quality-override').value=decision.text_quality_score_override??'';$('note').value=decision.note;$('field-discovery-approval').checked=state.decisions.approval_to_begin_field_discovery;}
async function show(card){state.index=card;const article=$('document');article.replaceChildren();article.append(text('h2',card.source_id+' · '+card.source_doc_id));const scores=document.createElement('p');for(const value of [`Cleanliness ${card.cleanliness_score}/5`,`Text quality ${card.text_quality_score}/5`,card.confidence,card.coverage_mode]){const badge=text('span',value);badge.className='score';scores.append(badge);}article.append(scores);const meta=document.createElement('dl');for(const [label,value] of [['Repository',card.origin_locator.repo_id],['Dataset',card.source_dataset],['Revision',card.source_revision],['Logical route',card.source_route],['Observed route',card.extraction_route],['Packet path',card.document_path],['Origin',`${card.origin_locator.artifact_path}:${card.origin_locator.row_index} (${card.origin_locator.text_field})`]])meta.append(...labeled(label,value));meta.className='meta';article.append(meta);article.append(text('p',card.summary));article.append(text('h3','Extraction artifacts'));if(!card.extraction_artifacts.length)article.append(text('p','No material extraction artifact reported.'));for(const artifact of card.extraction_artifacts){const box=document.createElement('section');box.className='artifact';box.append(text('strong',`${artifact.type} · ${artifact.severity} · lines ${artifact.line_start}-${artifact.line_end}`),text('p',artifact.explanation),text('pre',artifact.evidence_excerpt),text('p',`Cleaning: ${artifact.deterministic_cleaning_possible?'deterministic':'manual/uncertain'} — ${artifact.suggested_cleaning_action}`));article.append(box);}article.append(text('h3','Exact raw reviewed document'));setDecisionControls(card);const response=await fetch(`data/documents/${card.opaque_id}.json`,{cache:'no-store'});if(!response.ok)throw new Error('Raw document could not be loaded.');const payload=await response.json();if(state.index!==card)return;const pre=text('pre',payload.text);pre.className='raw';article.append(pre);}
function saveDecision(){const card=state.index;if(!card)return;const disposition=$('document-disposition').value;const cleanliness=disposition==='override'?scoreValue('cleanliness-override'):null;const quality=disposition==='override'?scoreValue('quality-override'):null;state.decisions.source_status[card.source_id]=$('source-status').value;state.decisions.source_observations[card.source_id]=$('source-observation').value;state.decisions.mapping_questions[card.source_id]=$('mapping-question').value;state.decisions.documents[card.request_id]={source_id:card.source_id,source_doc_id:card.source_doc_id,disposition,cleanliness_score_override:cleanliness,text_quality_score_override:quality,note:$('note').value};}
for(const id of ['source-filter','artifact-filter','confidence-filter','cleanliness-filter','quality-filter'])$(id).addEventListener('change',filter);for(const id of ['source-status','source-observation','mapping-question','document-disposition','cleanliness-override','quality-override','note'])$(id).addEventListener(id==='note'||id.includes('observation')||id.includes('question')?'input':'change',saveDecision);$('field-discovery-approval').addEventListener('change',()=>{state.decisions.approval_to_begin_field_discovery=$('field-discovery-approval').checked;});$('agree').addEventListener('click',()=>{if(!state.index)return;$('document-disposition').value='agree';$('cleanliness-override').value='';$('quality-override').value='';saveDecision();$('agree').textContent='Scores marked agreed';});$('export').addEventListener('click',()=>{saveDecision();const blob=new Blob([JSON.stringify(state.decisions,null,2)+'\\n'],{type:'application/json'});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='agent1_v4_human_decisions.json';link.click();URL.revokeObjectURL(link.href);});
fetch('data/index.json',{cache:'no-store'}).then(response=>{if(!response.ok)throw new Error('index could not be loaded');return response.json();}).then(index=>{state.cards=index.cards;state.source_counts=index.source_counts;state.decisions.packet_manifest_sha256=index.packet_manifest_sha256;initializeDecisions();sourceSummary();filter();}).catch(error=>{$('document').replaceChildren(text('p','Site data failed validation: '+error.message));});
"""


def _lazy_js() -> str:
    """Client that fetches only the selected source's review cards."""

    return r"""'use strict';
var state={index:null,cards:[],visible:[],source_ids:[],source_counts:{},source_files:{},artifact_counts:{},active_source:'',decisions:{schema_version:'agent1_v4_human_decision_bundle_v1',packet_manifest_sha256:'',approval_to_begin_field_discovery:false,source_status:{},source_observations:{},mapping_questions:{},documents:{}}};
var $=function(id){return document.getElementById(id);};
var text=function(tag,value){var node=document.createElement(tag);node.textContent=String(value==null?'':value);return node;};
var clear=function(node){while(node.firstChild)node.removeChild(node.firstChild);};
function select(id,values,label){var el=$(id);clear(el);el.appendChild(new Option(label,''));values.forEach(function(value){el.appendChild(new Option(value,value));});}
function setMessage(value){var article=$('document');clear(article);article.appendChild(text('p',value));}
function initializeDecisions(documents){state.source_ids.forEach(function(source){state.decisions.source_status[source]='';state.decisions.source_observations[source]='';state.decisions.mapping_questions[source]='';});documents.forEach(function(card){state.decisions.documents[card.request_id]={source_id:card.source_id,source_doc_id:card.source_doc_id,disposition:'unreviewed',cleanliness_score_override:null,text_quality_score_override:null,note:''};});}
function sourceSummary(){var ids=state.source_ids;select('source-filter',ids,'choose a source');select('confidence-filter',['low','medium','high'],'all confidence');$('coverage').textContent=ids.length+' sources · '+state.index.document_count+' reviewed raw documents · cards load one source at a time';var overview=$('overview');clear(overview);ids.forEach(function(source){var artifacts=Object.keys(state.artifact_counts[source]||{}).reduce(function(total,key){return total+state.artifact_counts[source][key];},0);overview.appendChild(text('span',source+': '+state.source_counts[source]+' · '+artifacts+' artifacts'));});}
function artifactOptions(){var seen={};state.cards.forEach(function(card){(card.extraction_artifacts||[]).forEach(function(artifact){seen[artifact.type]=true;});});select('artifact-filter',Object.keys(seen).sort(),'all artifacts');}
function loadSource(source){if(!source){state.cards=[];state.visible=[];renderList();setMessage('Choose a source to load its review cards.');return;}state.active_source=source;state.cards=[];state.visible=[];renderList();setMessage('Loading '+source+' review cards…');fetch(state.source_files[source],{cache:'no-store'}).then(function(response){if(!response.ok)throw new Error('source cards could not be loaded');return response.json();}).then(function(payload){if(state.active_source!==source)return;if(payload.source_id!==source||!Array.isArray(payload.cards))throw new Error('source-card payload failed validation');state.cards=payload.cards;artifactOptions();filter();}).catch(function(error){if(state.active_source===source)setMessage('Source cards failed validation: '+error.message);});}
function filter(){var artifact=$('artifact-filter').value,confidence=$('confidence-filter').value,clean=Number($('cleanliness-filter').value),quality=Number($('quality-filter').value);state.visible=state.cards.filter(function(card){return (!artifact||(card.extraction_artifacts||[]).some(function(item){return item.type===artifact;}))&&(!confidence||card.confidence===confidence)&&card.cleanliness_score>=clean&&card.text_quality_score>=quality;});renderList();if(state.visible.length)show(state.visible[0]);else setMessage('No documents match these filters.');}
function renderList(){var list=$('document-list');clear(list);state.visible.forEach(function(card){var button=document.createElement('button');button.type='button';button.appendChild(text('strong',card.source_id));button.appendChild(document.createElement('br'));button.appendChild(text('span','clean '+card.cleanliness_score+'/5 · quality '+card.text_quality_score+'/5'));button.addEventListener('click',function(){show(card);});var item=document.createElement('li');item.appendChild(button);list.appendChild(item);});}
function labeled(label,value){return [text('dt',label),text('dd',value)];}
function scoreValue(id){var value=$(id).value;return value===''?null:Number(value);}
function setDecisionControls(card){var decision=state.decisions.documents[card.request_id];$('source-status').value=state.decisions.source_status[card.source_id];$('source-observation').value=state.decisions.source_observations[card.source_id];$('mapping-question').value=state.decisions.mapping_questions[card.source_id];$('document-disposition').value=decision.disposition;$('cleanliness-override').value=decision.cleanliness_score_override==null?'':decision.cleanliness_score_override;$('quality-override').value=decision.text_quality_score_override==null?'':decision.text_quality_score_override;$('note').value=decision.note;$('field-discovery-approval').checked=state.decisions.approval_to_begin_field_discovery;}
function show(card){state.index=card;var article=$('document');clear(article);article.appendChild(text('h2',card.source_id+' · '+card.source_doc_id));var scores=document.createElement('p');['Cleanliness '+card.cleanliness_score+'/5','Text quality '+card.text_quality_score+'/5',card.confidence,card.coverage_mode].forEach(function(value){var badge=text('span',value);badge.className='score';scores.appendChild(badge);});article.appendChild(scores);var meta=document.createElement('dl');[['Repository',card.origin_locator.repo_id],['Dataset',card.source_dataset],['Revision',card.source_revision],['Logical route',card.source_route],['Observed route',card.extraction_route],['Packet path',card.document_path],['Origin',card.origin_locator.artifact_path+':'+card.origin_locator.row_index+' ('+card.origin_locator.text_field+')']].forEach(function(pair){labeled(pair[0],pair[1]).forEach(function(node){meta.appendChild(node);});});meta.className='meta';article.appendChild(meta);article.appendChild(text('p',card.summary));article.appendChild(text('h3','Extraction artifacts'));if(!(card.extraction_artifacts||[]).length)article.appendChild(text('p','No material extraction artifact reported.'));(card.extraction_artifacts||[]).forEach(function(artifact){var box=document.createElement('section');box.className='artifact';box.appendChild(text('strong',artifact.type+' · '+artifact.severity+' · lines '+artifact.line_start+'-'+artifact.line_end));box.appendChild(text('p',artifact.explanation));box.appendChild(text('pre',artifact.evidence_excerpt));box.appendChild(text('p','Cleaning: '+(artifact.deterministic_cleaning_possible?'deterministic':'manual/uncertain')+' — '+artifact.suggested_cleaning_action));article.appendChild(box);});article.appendChild(text('h3','Exact raw reviewed document'));setDecisionControls(card);fetch('data/documents/'+card.opaque_id+'.json',{cache:'no-store'}).then(function(response){if(!response.ok)throw new Error('raw document could not be loaded');return response.json();}).then(function(payload){if(state.index!==card)return;var pre=text('pre',payload.text);pre.className='raw';article.appendChild(pre);}).catch(function(error){if(state.index===card)article.appendChild(text('p','Raw document failed to load: '+error.message));});}
function saveDecision(){var card=state.index;if(!card)return;var disposition=$('document-disposition').value;var cleanliness=disposition==='override'?scoreValue('cleanliness-override'):null;var quality=disposition==='override'?scoreValue('quality-override'):null;state.decisions.source_status[card.source_id]=$('source-status').value;state.decisions.source_observations[card.source_id]=$('source-observation').value;state.decisions.mapping_questions[card.source_id]=$('mapping-question').value;state.decisions.documents[card.request_id]={source_id:card.source_id,source_doc_id:card.source_doc_id,disposition:disposition,cleanliness_score_override:cleanliness,text_quality_score_override:quality,note:$('note').value};}
$('source-filter').addEventListener('change',function(){loadSource(this.value);});['artifact-filter','confidence-filter','cleanliness-filter','quality-filter'].forEach(function(id){$(id).addEventListener('change',filter);});['source-status','source-observation','mapping-question','document-disposition','cleanliness-override','quality-override','note'].forEach(function(id){$(id).addEventListener(id==='note'||id.indexOf('observation')>=0||id.indexOf('question')>=0?'input':'change',saveDecision);});$('field-discovery-approval').addEventListener('change',function(){state.decisions.approval_to_begin_field_discovery=this.checked;});$('agree').addEventListener('click',function(){if(!state.index)return;$('document-disposition').value='agree';$('cleanliness-override').value='';$('quality-override').value='';saveDecision();$('agree').textContent='Scores marked agreed';});$('export').addEventListener('click',function(){saveDecision();var blob=new Blob([JSON.stringify(state.decisions,null,2)+'\n'],{type:'application/json'});var link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='agent1_v4_human_decisions.json';link.click();URL.revokeObjectURL(link.href);});
fetch('data/index.json',{cache:'no-store'}).then(function(response){if(!response.ok)throw new Error('index could not be loaded');return response.json();}).then(function(index){if(!index||!Array.isArray(index.documents)||!index.source_counts||!index.source_files)throw new Error('index failed validation');state.index=index;state.source_counts=index.source_counts;state.source_files=index.source_files;state.artifact_counts=index.artifact_counts||{};state.source_ids=Object.keys(index.source_counts).sort();state.index.document_count=index.documents.length;initializeDecisions(index.documents);sourceSummary();var first=state.source_ids[0]||'';$('source-filter').value=first;loadSource(first);}).catch(function(error){setMessage('Site data failed validation: '+error.message);});
"""


def _html_renderer_js() -> str:
    """Add safe raw-HTML and hybrid Markdown/HTML views for OCR output."""

    return r"""'use strict';
(function(){
  var root=document.getElementById('document');
  if(!root)return;
  var TOKEN_OPEN='\uE000agent1-html-';
  var TOKEN_CLOSE='\uE001';
  var TOKEN_RE=/\uE000agent1-html-(\d+)\uE001/g;
  var TOKEN_PART_RE=/^\uE000agent1-html-\d+\uE001$/;
  var IMAGE_RE=/!\[([^\]]*)\]\(([^)\n]+)\)/g;
  function escapeHtml(value){return String(value||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');}
  function reserveHtml(raw){
    var saved=[];
    function save(value){saved.push(value);return TOKEN_OPEN+(saved.length-1)+TOKEN_CLOSE;}
    var text=raw.replace(/<table\b[^>]*>[\s\S]*?<\/table\s*>/gi,save);
    text=text.replace(/<\/?[A-Za-z][A-Za-z0-9:-]*(?:\s[^<>]*)?>/g,save);
    return {text:text,saved:saved};
  }
  function escapeTokenized(value){
    return String(value||'').split(/(\uE000agent1-html-\d+\uE001)/g).map(function(part){return TOKEN_PART_RE.test(part)?part:escapeHtml(part);}).join('');
  }
  function restoreHtml(value,saved){return value.replace(TOKEN_RE,function(_,index){return saved[Number(index)]||'';});}
  function inline(value,saved){
    var out=escapeTokenized(value);
    out=out.replace(/`([^`]+)`/g,'<code>$1</code>');
    out=out.replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>');
    out=out.replace(/__([^_]+)__/g,'<strong>$1</strong>');
    out=out.replace(/\*([^*\n]+)\*/g,'<em>$1</em>');
    out=out.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+|mailto:[^)\s]+)\)/g,'<a href="$2" rel="noreferrer noopener">$1</a>');
    return restoreHtml(out,saved);
  }
  function tokenOnly(value,saved){
    var match=String(value||'').trim().match(/^\uE000agent1-html-(\d+)\uE001$/);
    return match?saved[Number(match[1])]:null;
  }
  function comment(value){return '<!-- '+String(value||'').replace(/--/g,'—')+' -->';}
  function figure(alt){
    return comment(alt);
  }
  function mixedMarkup(raw){
    var protectedHtml=reserveHtml(String(raw||'').replace(/\r\n?/g,'\n'));
    var out=[];
    var previousWasFigure=false;
    var paragraph=[];
    var list=null;
    function blockInline(lines){return inline(lines.join('\n'),protectedHtml.saved).replace(/\n/g,'<br>');}
    function flushList(){
      if(!list)return;
      out.push('<'+list.type+'>'+list.items.map(function(item){return '<li>'+blockInline(item)+'</li>';}).join('')+'</'+list.type+'>');
      list=null;
    }
    function flushParagraph(){
      if(!paragraph.length)return;
      var value=paragraph.join('\n').trim();paragraph=[];
      if(previousWasFigure&&/^this image(?:\s|,|\.)/i.test(value))out.push(comment(value));
      else out.push('<p>'+blockInline([value])+'</p>');
      previousWasFigure=false;
    }
    function addTextLine(value){
      var text=String(value||'').trim();
      if(!text)return;
      if(previousWasFigure&&!paragraph.length&&/^this image(?:\s|,|\.)/i.test(text)){out.push(comment(text));previousWasFigure=false;return;}
      paragraph.push(text);previousWasFigure=false;
    }
    function addImage(alt){flushList();flushParagraph();out.push(figure(alt));previousWasFigure=true;}
    protectedHtml.text.split('\n').forEach(function(line){
      var rawHtml=tokenOnly(line,protectedHtml.saved);
      if(rawHtml&&/^<table\b/i.test(rawHtml)){flushList();flushParagraph();out.push(rawHtml);previousWasFigure=false;return;}
      if(!line.trim()){flushList();flushParagraph();return;}
      var heading=line.match(/^\s*(#{1,6})\s+(.+?)\s*$/);
      if(heading){flushList();flushParagraph();var level=heading[1].length;out.push('<h'+level+'>'+inline(heading[2],protectedHtml.saved)+'</h'+level+'>');previousWasFigure=false;return;}
      if(/^\s*(?:---+|\*\*\*+|___+)\s*$/.test(line)){flushList();flushParagraph();out.push('<hr>');previousWasFigure=false;return;}
      var unordered=line.match(/^\s*[-*+]\s+(.+)$/);
      var ordered=line.match(/^\s*\d+[.)]\s+(.+)$/);
      var continuation=line.match(/^\s{2,}(.+)$/);
      if(unordered||ordered){flushParagraph();var type=unordered?'ul':'ol';if(!list||list.type!==type){flushList();list={type:type,items:[]};}list.items.push([unordered?unordered[1]:ordered[1]]);previousWasFigure=false;return;}
      if(continuation&&list){list.items[list.items.length-1].push(continuation[1]);previousWasFigure=false;return;}
      if(list)flushList();
      var cursor=0,match;
      IMAGE_RE.lastIndex=0;
      while((match=IMAGE_RE.exec(line))!==null){addTextLine(line.slice(cursor,match.index));addImage(match[1]);cursor=IMAGE_RE.lastIndex;}
      addTextLine(line.slice(cursor));
    });
    flushList();flushParagraph();
    return out.join('');
  }
  function frameDocument(raw,view){
    var csp="default-src 'none'; img-src data:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'";
    var css="body{margin:1rem;color:#172033;font:16px/1.5 system-ui,-apple-system,sans-serif;overflow-wrap:anywhere}p{margin:.65rem 0}h1,h2,h3,h4,h5,h6{line-height:1.2}table{border-collapse:collapse;max-width:100%;margin:1rem 0}th,td{border:1px solid #94a3b8;padding:.35rem .5rem;text-align:left;vertical-align:top}img{max-width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}code{font:12px ui-monospace,monospace}";
    var body=view==='mixed'?mixedMarkup(raw):raw;
    return '<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="'+csp+'"><style>'+css+'</style></head><body>'+body+'</body></html>';
  }
  function addViewer(pre){
    if(pre.dataset.htmlViewer==='ready')return;
    pre.dataset.htmlViewer='ready';
    var controls=document.createElement('div');
    controls.className='document-view-controls';
    controls.appendChild((function(){var label=document.createElement('span');label.textContent='View OCR output:';return label;})());
    var rawButton=document.createElement('button');
    rawButton.type='button';rawButton.textContent='Raw text';rawButton.setAttribute('aria-pressed','true');
    var htmlButton=document.createElement('button');
    htmlButton.type='button';htmlButton.textContent='HTML only';htmlButton.setAttribute('aria-pressed','false');
    var mixedButton=document.createElement('button');
    mixedButton.type='button';mixedButton.textContent='Mixed Markdown + HTML';mixedButton.setAttribute('aria-pressed','false');
    var frame=document.createElement('iframe');
    frame.className='html-render-frame';frame.hidden=true;frame.setAttribute('sandbox','');frame.setAttribute('referrerpolicy','no-referrer');frame.title='Sandboxed rendering of OCR output';
    function showRaw(){pre.hidden=false;frame.hidden=true;rawButton.setAttribute('aria-pressed','true');htmlButton.setAttribute('aria-pressed','false');mixedButton.setAttribute('aria-pressed','false');}
    function showFrame(view){if(frame.dataset.view!==view){frame.srcdoc=frameDocument(pre.textContent||'',view);frame.dataset.view=view;}pre.hidden=true;frame.hidden=false;rawButton.setAttribute('aria-pressed','false');htmlButton.setAttribute('aria-pressed',view==='html'?'true':'false');mixedButton.setAttribute('aria-pressed',view==='mixed'?'true':'false');}
    rawButton.addEventListener('click',showRaw);htmlButton.addEventListener('click',function(){showFrame('html');});mixedButton.addEventListener('click',function(){showFrame('mixed');});
    controls.appendChild(rawButton);controls.appendChild(htmlButton);controls.appendChild(mixedButton);pre.parentNode.insertBefore(controls,pre);pre.parentNode.insertBefore(frame,pre.nextSibling);
  }
  function enhance(){root.querySelectorAll('pre.raw').forEach(addViewer);}
  new MutationObserver(enhance).observe(root,{childList:true,subtree:true});
  enhance();
})();
"""


def _vlm_repetition_viewer_js() -> str:
    """Show a post-build GlossAPI VLM-loop audit when one is attached to the site."""

    return r"""'use strict';
(function(){
  var root=document.getElementById('document');
  if(!root)return;
  var findings={};
  function key(source,doc){return source+'\u0000'+doc;}
  function addFinding(){
    var heading=root.querySelector('h2');
    if(!heading||root.querySelector('[data-vlm-repetition-audit]'))return;
    var parts=heading.textContent.split(' · ');
    if(parts.length<2)return;
    var finding=findings[key(parts.shift(),parts.join(' · '))];
    if(!finding)return;
    var box=document.createElement('section');
    box.dataset.vlmRepetitionAudit='true';
    box.className='vlm-audit'+((finding.rules||[]).length||finding.streaming_reason?'':' clear');
    var title=document.createElement('strong');
    title.textContent='VLM-output guard (diagnostic; no automatic deletion)';box.appendChild(title);
    var detail=document.createElement('p');
    if(!(finding.rules||[]).length&&!finding.streaming_reason){detail.textContent='No finalized-text detector trigger.';}
    else{
      var rules=(finding.rules||[]).map(function(item){return item.rule+' at character '+item.cut_index;});
      if(finding.streaming_reason)rules.push('streaming guard: '+finding.streaming_reason);
      detail.textContent=rules.join(' · ');
    }
    box.appendChild(detail);heading.parentNode.insertBefore(box,heading.nextSibling);
  }
  new MutationObserver(addFinding).observe(root,{childList:true,subtree:true});
  fetch('data/vlm_repetition_audit.json',{cache:'no-store'}).then(function(response){if(!response.ok)return null;return response.json();}).then(function(audit){if(!audit||!Array.isArray(audit.documents))return;audit.documents.forEach(function(row){if(row&&typeof row.source_id==='string'&&typeof row.source_doc_id==='string')findings[key(row.source_id,row.source_doc_id)]=row;});addFinding();}).catch(function(){});
})();
"""


def _detections_html() -> str:
    """Standalone presentation of finalized-text VLM repetition detector findings."""

    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; connect-src 'self'; script-src 'self'; style-src 'self'; base-uri 'none'; form-action 'none'">
  <title>Apertus VLM repetition detections</title>
  <link rel="stylesheet" href="assets/detections.css">
</head>
<body>
  <header>
    <p><a href="index.html">← Raw-document review</a> · <a href="normalization.html">HTML → GitHub Markdown normalization</a></p>
    <h1>VLM repetition detections</h1>
    <p>Evidence presentation from the finalized-text guards used by the GlossAPI corpus OCR path.</p>
  </header>
  <main>
    <section class="scope" aria-labelledby="scope-title">
      <h2 id="scope-title">What this page shows</h2>
      <p>Each card shows both the raw detector evidence and a dry-run cleaning preview. Runaway complex repetitions are replaced by exactly <code>&lt;!-- repeating-text-removed --&gt;</code>; no source document is changed by this presentation.</p>
      <p>Repeated structural table headers are preserved rather than cleaned and are listed explicitly so that the distinction remains reviewable.</p>
      <p>Signals from the older early-stop guards—such as single-character runs, repeated blank lines, or repeated empty table cells—remain visible but are not automatically removed by this complex-pattern cleaner.</p>
      <p>The exact VLLM token-triplet loop detector is not included because these archived review documents retain text, not VLLM token IDs.</p>
    </section>
    <section id="audit-summary" aria-live="polite"><p>Loading detector audit…</p></section>
    <section aria-labelledby="findings-title">
      <h2 id="findings-title">Flagged documents</h2>
      <div id="findings"><p>Loading raw evidence…</p></div>
    </section>
  </main>
  <script src="assets/detections.js"></script>
</body>
</html>
"""


def _normalization_html() -> str:
    """Presentation of dry-run HTML-to-GFM decisions and lazy document previews."""

    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; connect-src 'self'; script-src 'self'; style-src 'self'; base-uri 'none'; form-action 'none'">
  <title>Apertus HTML to GitHub Markdown normalization</title>
  <link rel="stylesheet" href="assets/normalization.css">
</head>
<body>
  <header>
    <p><a href="index.html">← Raw-document review</a> · <a href="detections.html">VLM repetition detections</a></p>
    <h1>HTML → GitHub Markdown normalization</h1>
    <p>Dry-run structures, transformation decisions, and rendered results for the reviewed sample.</p>
  </header>
  <main>
    <section class="scope" aria-labelledby="scope-title">
      <h2 id="scope-title">Scope and safety</h2>
      <p>The immutable raw documents are untouched. Each preview first applies the current GlossAPI complex-repetition cleaner, inserting exactly <code>&lt;!-- repeating-text-removed --&gt;</code>, then converts recognized HTML structure to HTML-free GitHub-Flavored Markdown.</p>
      <p>Existing Markdown is preserved. Where GFM has no non-HTML styling equivalent—such as superscript, subscript, underline, or arbitrary spans—the element and attributes are removed but their textual payload is retained. Embedded or executable elements are removed with their content.</p>
      <p>The rendered pane is a local Markdown approximation for review, isolated in a script-free sandbox. It is not new corpus text and image fetching is disabled.</p>
    </section>
    <section id="normalization-summary" aria-live="polite"><p>Loading normalization audit…</p></section>
    <section aria-labelledby="decisions-title">
      <h2 id="decisions-title">Transformation decisions</h2>
      <div class="table-scroll">
        <table class="decisions-table">
          <thead><tr><th>HTML element(s)</th><th>Observed starts</th><th>GitHub Markdown result</th><th>Content decision</th></tr></thead>
          <tbody id="decision-rows"></tbody>
        </table>
      </div>
    </section>
    <section aria-labelledby="documents-title">
      <h2 id="documents-title">Document previews</h2>
      <div class="controls">
        <label>Source <select id="normalization-source"><option value="">all sources</option></select></label>
        <label>HTML element <select id="normalization-tag"><option value="">all elements</option></select></label>
        <label class="checkbox"><input id="normalization-unchanged" type="checkbox"> Include byte-identical documents</label>
      </div>
      <p id="normalization-results"></p>
      <div id="normalization-documents"><p>Loading normalized-document index…</p></div>
    </section>
  </main>
  <script src="assets/normalization.js"></script>
</body>
</html>
"""


def _normalization_css() -> str:
    return """*{box-sizing:border-box}body{margin:0;background:#f8fafc;color:#172033;font:16px/1.5 system-ui,-apple-system,sans-serif}header,main{max-width:1500px;margin:auto;padding:1.25rem}header{background:#102a43;color:#fff;max-width:none;padding-left:max(1.25rem,calc((100% - 1500px)/2 + 1.25rem));padding-right:max(1.25rem,calc((100% - 1500px)/2 + 1.25rem))}header h1{margin:.15rem 0}header p{margin:.25rem 0}header a{color:#dbeafe}.scope,#normalization-summary,#normalization-documents,.table-scroll{background:#fff;border:1px solid #cbd5e1;padding:1rem;margin:1rem 0}.scope{border-left:4px solid #1d4ed8}.scope code{background:#e2e8f0;color:#0f172a;padding:.1rem .3rem}.summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:.75rem}.metric{background:#e8f1fa;border-radius:.35rem;padding:.7rem}.metric strong{display:block;font-size:1.5rem}.reuse{border-left:4px solid #15803d;background:#f0fdf4;padding:.7rem 1rem;margin-top:1rem}.table-scroll{overflow-x:auto;padding:0}.decisions-table{width:100%;border-collapse:collapse}.decisions-table th,.decisions-table td{border:1px solid #cbd5e1;padding:.55rem .7rem;text-align:left;vertical-align:top}.decisions-table th{background:#e8f1fa}.decisions-table code{white-space:nowrap}.controls{display:flex;gap:1rem;align-items:end;flex-wrap:wrap;background:#e8f1fa;padding:1rem}.controls label{display:grid;gap:.3rem}.controls .checkbox{display:flex;align-items:center;padding-bottom:.25rem}.normalization-card{border:1px solid #cbd5e1;border-left:4px solid #7c3aed;padding:1rem;margin:1rem 0;min-width:0}.normalization-card h3{margin:.1rem 0;overflow-wrap:anywhere}.metadata{display:grid;grid-template-columns:max-content minmax(0,1fr);gap:.2rem .75rem}.metadata dt{font-weight:700}.metadata dd{margin:0;overflow-wrap:anywhere}.badges{display:flex;gap:.4rem;flex-wrap:wrap;margin:.65rem 0}.badge{background:#ede9fe;border:1px solid #c4b5fd;border-radius:999px;padding:.1rem .55rem;font:600 .82rem/1.4 ui-monospace,SFMono-Regular,monospace}.load-preview{border:0;border-radius:.35rem;background:#5b21b6;color:#fff;cursor:pointer;font:600 1rem/1.2 system-ui,-apple-system,sans-serif;padding:.65rem .9rem}.load-preview:disabled{background:#64748b;cursor:wait}.comparison{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:1rem}.source-pane{min-width:0}.source-pane h4{margin:.25rem 0}.source-text{white-space:pre-wrap;overflow-wrap:anywhere;background:#0b1220;color:#e5edf7;padding:1rem;max-height:42rem;overflow:auto}.render-pane{grid-column:1/-1}.render-frame{width:100%;height:46rem;border:1px solid #94a3b8;background:#fff}.unchanged-note{background:#f1f5f9;padding:.7rem}.error{background:#fef2f2;border-left:4px solid #b91c1c;padding:.75rem 1rem}@media(max-width:850px){header,main{padding:1rem}.comparison{grid-template-columns:1fr}.render-frame{height:36rem}.metadata{grid-template-columns:1fr}.metadata dt{margin-top:.4rem}}"""


def _normalization_js() -> str:
    """Lazy, text-safe renderer for the normalization decision presentation."""

    return r"""'use strict';
(function(){
  var audit=null;
  var summaryRoot=document.getElementById('normalization-summary');
  var decisionsRoot=document.getElementById('decision-rows');
  var documentsRoot=document.getElementById('normalization-documents');
  var resultsRoot=document.getElementById('normalization-results');
  var sourceFilter=document.getElementById('normalization-source');
  var tagFilter=document.getElementById('normalization-tag');
  var unchangedFilter=document.getElementById('normalization-unchanged');
  function text(tag,value){var node=document.createElement(tag);node.textContent=String(value==null?'':value);return node;}
  function clear(node){while(node.firstChild)node.removeChild(node.firstChild);}
  function formatted(value){var number=Number(value);return Number.isFinite(number)?number.toLocaleString('en-US'):String(value==null?0:value);}
  function metadata(root,label,value){root.appendChild(text('dt',label));root.appendChild(text('dd',value));}
  function option(select,value,label){select.appendChild(new Option(label,value));}
  function failure(root,message){clear(root);var node=text('p',message);node.className='error';root.appendChild(node);}
  function renderSummary(){
    clear(summaryRoot);var summary=audit.summary||{},grid=document.createElement('div');grid.className='summary-grid';
    [[summary.document_count,'documents audited'],[summary.documents_changed,'documents changed'],[summary.documents_with_recognized_html,'documents with HTML'],[summary.html_start_tag_count,'HTML start tags'],[(summary.transformation_counts||{}).html_tables_to_gfm||0,'tables converted'],[(summary.transformation_counts||{}).html_table_cells_preserved||0,'table cells kept structurally'],[(summary.transformation_counts||{}).nested_table_cells_flattened||0,'nested table cells flattened'],[(summary.transformation_counts||{}).orphan_table_cells_flattened||0,'orphan table cells flattened'],[summary.repetition_replacement_count,'repetition comments'],[summary.repetition_characters_removed,'repeating characters removed'],[summary.content_characters_removed,'other content characters removed'],[summary.residual_recognized_html_tags,'recognized HTML left']].forEach(function(metric){var box=document.createElement('div');box.className='metric';box.appendChild(text('strong',formatted(metric[0])));box.appendChild(text('span',metric[1]));grid.appendChild(box);});summaryRoot.appendChild(grid);
    var before=summary.markdown_structures_before||{},after=summary.markdown_structures_after||{};summaryRoot.appendChild(text('p','Existing Markdown checks: headings '+formatted(before.atx_headings)+' → '+formatted(after.atx_headings)+', fence lines '+formatted(before.fence_lines)+' → '+formatted(after.fence_lines)+', links '+formatted(before.markdown_links)+' → '+formatted(after.markdown_links)+', images '+formatted(before.markdown_images)+' → '+formatted(after.markdown_images)+', and table delimiters '+formatted(before.gfm_table_delimiters)+' → '+formatted(after.gfm_table_delimiters)+'. Increases come only from converted HTML structures. Parser-token gates also require non-loss of emphasis, strikethrough, inline and block code, lists, blockquotes, links, images, headings, and existing tables.'));
    var reuse=audit.glossapi_reuse_review||{},box=document.createElement('div');box.className='reuse';box.appendChild(text('strong','GlossAPI integration decision'));box.appendChild(text('p',String(reuse.must_precede||'')));box.appendChild(text('p','Reused in this dry run: '+(reuse.reused_now||[]).join('; ')+'. Reuse after structural conversion: '+(reuse.reuse_when_ported||[]).join('; ')+'.'));summaryRoot.appendChild(box);
  }
  function renderDecisions(){
    clear(decisionsRoot);(audit.transformation_decisions||[]).forEach(function(row){var tr=document.createElement('tr'),tags=document.createElement('td');(row.tags||[]).forEach(function(tag,index){if(index)tags.appendChild(document.createTextNode(', '));tags.appendChild(text('code',tag));});tr.appendChild(tags);tr.appendChild(text('td',formatted(row.observed_start_tag_count)));tr.appendChild(text('td',row.target));tr.appendChild(text('td',row.content_policy));decisionsRoot.appendChild(tr);});
  }
  function initializeFilters(){
    var sources=new Set(),tags=new Set();(audit.documents||[]).forEach(function(row){sources.add(String(row.source_id));Object.keys(row.tag_counts||{}).forEach(function(tag){tags.add(tag);});});Array.from(sources).sort().forEach(function(source){option(sourceFilter,source,source);});Array.from(tags).sort().forEach(function(tag){option(tagFilter,tag,tag+' ('+formatted((audit.summary.html_tag_counts||{})[tag]||0)+')');});
  }
  function badges(row){var root=document.createElement('div');root.className='badges';Object.keys(row.transformations||{}).sort().forEach(function(key){var badge=text('span',key+': '+row.transformations[key]);badge.className='badge';root.appendChild(badge);});if(row.repetition_replacements){var repetition=text('span','repetition replacements: '+row.repetition_replacements);repetition.className='badge';root.appendChild(repetition);}return root;}
  function previewDocument(rendered){return '<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; img-src \'none\'; base-uri \'none\'; form-action \'none\'"><style>body{max-width:1000px;margin:0 auto;padding:1.25rem;color:#172033;font:16px/1.55 system-ui,-apple-system,sans-serif;overflow-wrap:anywhere}table{border-collapse:collapse;display:block;max-width:100%;overflow:auto}th,td{border:1px solid #94a3b8;padding:.35rem .55rem;text-align:left}th{background:#e8f1fa}pre{white-space:pre-wrap;background:#f1f5f9;padding:.8rem}code{background:#f1f5f9}blockquote{border-left:4px solid #94a3b8;margin-left:0;padding-left:1rem;color:#475569}img{max-width:100%}</style></head><body>'+rendered+'</body></html>';}
  function sourcePane(titleValue,bodyValue){var pane=document.createElement('section');pane.className='source-pane';pane.appendChild(text('h4',titleValue));var pre=text('pre',bodyValue);pre.className='source-text';pane.appendChild(pre);return pane;}
  function loadPreview(row,card,button){
    button.disabled=true;button.textContent='Loading preview…';var rawRequest=fetch('data/documents/'+encodeURIComponent(row.opaque_id)+'.json',{cache:'no-store'}).then(function(response){if(!response.ok)throw new Error('raw HTTP '+response.status);return response.json();});
    var normalizedRequest=row.changed?fetch('data/gfm/documents/'+encodeURIComponent(row.opaque_id)+'.json',{cache:'no-store'}).then(function(response){if(!response.ok)throw new Error('normalized HTTP '+response.status);return response.json();}):Promise.resolve(null);
    Promise.all([rawRequest,normalizedRequest]).then(function(payloads){var raw=payloads[0],normalized=payloads[1];if(!raw||typeof raw.text!=='string')throw new Error('invalid raw payload');var comparison=document.createElement('div');comparison.className='comparison';comparison.appendChild(sourcePane('Exact raw reviewed text',raw.text));if(normalized){if(typeof normalized.normalized_markdown!=='string'||typeof normalized.rendered_html!=='string')throw new Error('invalid normalized payload');comparison.appendChild(sourcePane('Normalized GitHub Markdown',normalized.normalized_markdown));var rendered=document.createElement('section');rendered.className='render-pane';rendered.appendChild(text('h4','Rendered normalized Markdown'));var frame=document.createElement('iframe');frame.className='render-frame';frame.title='Rendered normalized Markdown';frame.setAttribute('sandbox','');frame.referrerPolicy='no-referrer';frame.srcdoc=previewDocument(normalized.rendered_html);rendered.appendChild(frame);comparison.appendChild(rendered);}else{var unchanged=text('p','This document is byte-identical after normalization, so no duplicate normalized artifact was written.');unchanged.className='unchanged-note';comparison.appendChild(unchanged);}button.remove();card.appendChild(comparison);}).catch(function(error){button.disabled=false;button.textContent='Retry preview';var existing=card.querySelector('.error');if(existing)existing.remove();var node=text('p','Preview could not be loaded: '+error.message);node.className='error';card.appendChild(node);});
  }
  function renderCard(row){var card=document.createElement('article');card.className='normalization-card';card.appendChild(text('h3',row.source_id+' · '+row.source_doc_id));var meta=document.createElement('dl');meta.className='metadata';metadata(meta,'Opaque review document',row.opaque_id);metadata(meta,'Changed',row.changed?'yes':'no');metadata(meta,'Recognized HTML',row.has_recognized_html?'yes':'no');metadata(meta,'Characters',formatted(row.original_char_count)+' raw → '+formatted(row.normalized_char_count)+' normalized');metadata(meta,'Repetition removal',formatted(row.repetition_replacements)+' replacement(s), '+formatted(row.repetition_characters_removed)+' characters');metadata(meta,'Non-repetition content removed',formatted(row.content_characters_removed)+' characters');metadata(meta,'Escaped OCR angle-text spans',formatted(row.pseudo_tags_escaped));card.appendChild(meta);card.appendChild(badges(row));var button=text('button','Load raw, Markdown, and rendering');button.type='button';button.className='load-preview';button.addEventListener('click',function(){loadPreview(row,card,button);},{once:false});card.appendChild(button);return card;}
  function filter(){var source=sourceFilter.value,tag=tagFilter.value,includeUnchanged=unchangedFilter.checked,rows=(audit.documents||[]).filter(function(row){return (includeUnchanged||row.changed)&&(!source||row.source_id===source)&&(!tag||Number((row.tag_counts||{})[tag]||0)>0);});rows.sort(function(a,b){return String(a.source_id).localeCompare(String(b.source_id))||String(a.source_doc_id).localeCompare(String(b.source_doc_id));});resultsRoot.textContent=formatted(rows.length)+' documents shown; source and normalized text are fetched only when a preview is opened.';clear(documentsRoot);if(!rows.length){documentsRoot.appendChild(text('p','No documents match these filters.'));return;}rows.forEach(function(row){documentsRoot.appendChild(renderCard(row));});}
  [sourceFilter,tagFilter,unchangedFilter].forEach(function(control){control.addEventListener('change',filter);});
  fetch('data/gfm_normalization_audit.json',{cache:'no-store'}).then(function(response){if(!response.ok)throw new Error('HTTP '+response.status);return response.json();}).then(function(payload){if(!payload||payload.status!=='passed'||!Array.isArray(payload.documents))throw new Error('invalid audit payload');audit=payload;renderSummary();renderDecisions();initializeFilters();filter();}).catch(function(error){failure(summaryRoot,'Normalization audit failed to load: '+error.message);failure(documentsRoot,'No normalization previews can be presented until the audit is available.');});
})();
"""


def _detections_css() -> str:
    return """*{box-sizing:border-box}body{margin:0;background:#f8fafc;color:#172033;font:16px/1.5 system-ui,-apple-system,sans-serif}header,main{max-width:1450px;margin:auto;padding:1.25rem}header{background:#102a43;color:#fff;max-width:none;padding-left:max(1.25rem,calc((100% - 1450px)/2 + 1.25rem));padding-right:max(1.25rem,calc((100% - 1450px)/2 + 1.25rem))}header h1{margin:.15rem 0}header p{margin:.25rem 0}header a{color:#dbeafe}.scope,#audit-summary,.finding{background:#fff;border:1px solid #cbd5e1;padding:1rem;margin:1rem 0}.scope{border-left:4px solid #1d4ed8}.scope code{background:#e2e8f0;color:#0f172a;padding:.1rem .3rem}.summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:.75rem}.metric{background:#e8f1fa;border-radius:.35rem;padding:.7rem}.metric strong{display:block;font-size:1.55rem}.rule-list{display:flex;gap:.45rem;flex-wrap:wrap;margin:.75rem 0}.badge{display:inline-block;border-radius:999px;background:#fff1c2;border:1px solid #d39b00;padding:.1rem .55rem;font:600 .86rem/1.4 ui-monospace,SFMono-Regular,monospace}.finding{border-left:4px solid #c2410c;min-width:0}.finding h3{margin-top:0;overflow-wrap:anywhere}.metadata{display:grid;grid-template-columns:max-content minmax(0,1fr);gap:.25rem .8rem}.metadata dt{font-weight:700}.metadata dd{margin:0;overflow-wrap:anywhere}.load-evidence{border:0;border-radius:.35rem;background:#1d4ed8;color:#fff;cursor:pointer;font:600 1rem/1.2 system-ui,-apple-system,sans-serif;padding:.65rem .9rem}.load-evidence:disabled{background:#64748b;cursor:wait}.signal{border-top:1px solid #dbe3ee;margin-top:1rem;padding-top:1rem}.signal h4{margin:0 0 .3rem}.cleaning{border-left:4px solid #15803d;padding-left:1rem}.preserved{background:#eff6ff;border-left:4px solid #2563eb;padding:.65rem .85rem}.context-note{margin:.25rem 0;color:#475569;font-size:.9rem}.context{margin:.5rem 0 0;white-space:pre-wrap;overflow-wrap:anywhere;background:#0b1220;color:#e5edf7;padding:1rem;max-height:38rem;overflow:auto}.context mark{background:#ffdc5d;color:#1d2939;padding:0}.cleaned-context mark{background:#86efac}.streaming{background:#fff7ed;border-left:4px solid #c2410c;padding:.6rem .8rem;margin-top:.75rem}.error{background:#fef2f2;border-left:4px solid #b91c1c;padding:.75rem 1rem}@media(max-width:650px){header,main{padding:1rem}.metadata{grid-template-columns:1fr}.metadata dt{margin-top:.5rem}}"""


def _detections_js() -> str:
    """Browser-only renderer for the small, lazily loaded detector audit presentation."""

    return r"""'use strict';
(function(){
  var summaryRoot=document.getElementById('audit-summary');
  var findingsRoot=document.getElementById('findings');
  var BEFORE=420, HIGHLIGHT=180, AFTER=720;
  function text(tag,value){var node=document.createElement(tag);node.textContent=String(value==null?'':value);return node;}
  function clear(node){while(node.firstChild)node.removeChild(node.firstChild);}
  function failure(root,message){clear(root);var box=text('p',message);box.className='error';root.appendChild(box);}
  function isFinding(row){var preview=row&&row.cleaning_preview;return row&&((Array.isArray(row.rules)&&row.rules.length>0)||row.streaming_reason||(preview&&(preview.changed||(Array.isArray(preview.preserved_findings)&&preview.preserved_findings.length>0))));}
  function number(value){return typeof value==='number'&&Number.isFinite(value)?value:null;}
  function integer(value){var parsed=number(value);return parsed!==null&&Math.floor(parsed)===parsed?parsed:null;}
  function formatted(value){var parsed=number(value);return parsed===null?String(value==null?0:value):parsed.toLocaleString('en-US');}
  function metadata(label,value){return [text('dt',label),text('dd',value)];}
  function evidenceSummary(rule){
    var parts=[];
    if(rule.repeat_count!=null)parts.push(String(rule.repeat_count)+' adjacent repeats');
    if(rule.pattern_token_count!=null)parts.push(String(rule.pattern_token_count)+' tokens per pattern');
    if(rule.pattern_char_count!=null)parts.push(String(rule.pattern_char_count)+' characters per pattern');
    if(rule.sequence_term_count!=null)parts.push(String(rule.sequence_term_count)+' sequence terms');
    if(rule.sequence_step!=null)parts.push('step '+String(rule.sequence_step));
    if(rule.sequence_step_tolerance!=null&&String(rule.sequence_step_tolerance)!=='0')parts.push('display tolerance '+String(rule.sequence_step_tolerance));
    if(rule.evidence_end_index!=null)parts.push('detected span ends at character '+String(rule.evidence_end_index));
    if(rule.removal_end_index!=null&&rule.removal_end_index!==rule.evidence_end_index)parts.push('cleaning span ends at character '+String(rule.removal_end_index));
    if(rule.trailing_partial_token_count!=null)parts.push(String(rule.trailing_partial_token_count)+' tokens from a truncated final copy');
    if(rule.trailing_partial_fragment)parts.push('includes a truncated final token');
    if(rule.trailing_partial_number!=null)parts.push('includes truncated expected number '+String(rule.trailing_partial_number));
    return parts.join(' · ');
  }
  function renderSummary(audit,findings){
    clear(summaryRoot);
    var summary=audit.summary||{};
    var grid=document.createElement('div');grid.className='summary-grid';
    [[summary.document_count,'documents scanned'],[summary.documents_with_any_trigger,'documents flagged'],[summary.documents_changed_by_cleaner,'documents changed in preview'],[summary.cleaning_replacement_count,'replacement comments'],[summary.cleaning_characters_removed,'characters removed'],[summary.cleaning_preserved_finding_count,'structural findings preserved'],[(summary.rule_trigger_counts&&Object.values(summary.rule_trigger_counts).reduce(function(total,count){return total+Number(count||0);},0))||0,'rule detections']].forEach(function(metric){var box=document.createElement('div');box.className='metric';box.appendChild(text('strong',formatted(metric[0])));box.appendChild(text('span',metric[1]));grid.appendChild(box);});
    summaryRoot.appendChild(grid);
    var rules=summary.rule_trigger_counts||{};
    var ruleList=document.createElement('div');ruleList.className='rule-list';Object.keys(rules).sort().forEach(function(rule){ruleList.appendChild(text('span',rule+': '+rules[rule]));ruleList.lastChild.className='badge';});
    if(ruleList.childNodes.length)summaryRoot.appendChild(ruleList);
    var detector=audit.detector||{};
    var modules=[detector.implementation,detector.complex_repetition_implementation].filter(Boolean).join(' + ')||'not recorded';
    summaryRoot.appendChild(text('p','GlossAPI modules: '+modules+'. '+findings.length+' flagged document cards are shown. Cleaning is preview-only, and raw evidence loads only when requested.'));
  }
  function appendExcerpt(root,raw,rule){
    var cut=number(rule.cut_index);
    var section=document.createElement('section');section.className='signal';
    if(cut===null||cut<0||cut>raw.length){section.appendChild(text('h4',String(rule.rule||'unknown rule')));section.appendChild(text('p','The audit’s cut position is unavailable for this raw document.'));root.appendChild(section);return;}
    var start=Math.max(0,cut-BEFORE),highlightEnd=Math.min(raw.length,cut+HIGHLIGHT),end=Math.min(raw.length,highlightEnd+AFTER);
    section.appendChild(text('h4',String(rule.rule||'unknown rule')+' at character '+cut));
    var evidence=evidenceSummary(rule);if(evidence)section.appendChild(text('p',evidence));
    section.appendChild(text('p','Raw context: characters '+start+'–'+end+'; highlight begins at the recorded cut.'));
    var pre=document.createElement('pre');pre.className='context';
    pre.appendChild(document.createTextNode(raw.slice(start,cut)));
    var mark=document.createElement('mark');mark.textContent=raw.slice(cut,highlightEnd);pre.appendChild(mark);
    pre.appendChild(document.createTextNode(raw.slice(highlightEnd,end)));
    section.appendChild(pre);root.appendChild(section);
  }
  function applyCleaningPass(raw,passRecord,marker){
    var spans=Array.isArray(passRecord&&passRecord.spans)?passRecord.spans.slice():[];
    spans.sort(function(a,b){return Number(a.start_index)-Number(b.start_index)||Number(a.end_index)-Number(b.end_index);});
    var pieces=[],positions=[],cursor=0,outputLength=0;
    for(var index=0;index<spans.length;index+=1){
      var span=spans[index],start=integer(span.start_index),end=integer(span.end_index);
      if(start===null||end===null||start<cursor||end<=start||end>raw.length)return {ok:false,message:'invalid or overlapping cleaning span in pass '+String(passRecord&&passRecord.pass_index)};
      var unchanged=raw.slice(cursor,start);pieces.push(unchanged);outputLength+=unchanged.length;
      positions.push({position:outputLength,span:span});pieces.push(marker);outputLength+=marker.length;cursor=end;
    }
    pieces.push(raw.slice(cursor));
    return {ok:true,text:pieces.join(''),positions:positions};
  }
  function appendCleanedContext(root,cleaned,position,marker,span,passIndex){
    var start=Math.max(0,position-BEFORE),end=Math.min(cleaned.length,position+marker.length+AFTER);
    var section=document.createElement('section');section.className='signal cleaning';
    section.appendChild(text('h4','Proposed replacement · pass '+String(passIndex)+' · original characters '+String(span.start_index)+'–'+String(span.end_index)));
    var rules=Array.isArray(span.rules)?span.rules.join(', '):'complex repetition';
    section.appendChild(text('p','Removed '+formatted(span.removed_char_count)+' characters detected by '+rules+' and inserted exactly '+marker+'.'));
    section.appendChild(text('p','Cleaned context after this pass; the green highlight is the literal replacement comment.'));
    var pre=document.createElement('pre');pre.className='context cleaned-context';
    pre.appendChild(document.createTextNode(cleaned.slice(start,position)));
    var mark=document.createElement('mark');mark.textContent=cleaned.slice(position,position+marker.length);pre.appendChild(mark);
    pre.appendChild(document.createTextNode(cleaned.slice(position+marker.length,end)));
    section.appendChild(pre);root.appendChild(section);
  }
  function appendCleaningPreview(root,raw,preview){
    if(!preview)return;
    var preserved=Array.isArray(preview.preserved_findings)?preview.preserved_findings:[];
    if(preserved.length){
      var kept=document.createElement('section');kept.className='signal preserved';kept.appendChild(text('h4','Preserved structural repetitions'));
      preserved.forEach(function(finding){kept.appendChild(text('p',String(finding.preservation_reason||'preserved structure')+' · characters '+String(finding.cut_index)+'–'+String(finding.removal_end_index==null?finding.evidence_end_index:finding.removal_end_index)));});
      root.appendChild(kept);
    }
    if(!preview.changed)return;
    var marker=String(preview.replacement_marker||'<!-- repeating-text-removed -->');
    var passes=Array.isArray(preview.replacement_details)?preview.replacement_details.slice():[];
    passes.sort(function(a,b){return Number(a.pass_index)-Number(b.pass_index);});
    var current=raw;
    for(var passOffset=0;passOffset<passes.length;passOffset+=1){
      var passRecord=passes[passOffset],result=applyCleaningPass(current,passRecord,marker);
      if(!result.ok){var error=text('p','Cleaning preview could not be reconstructed: '+result.message);error.className='error';root.appendChild(error);return;}
      result.positions.forEach(function(item){appendCleanedContext(root,result.text,item.position,marker,item.span,passRecord.pass_index);});
      current=result.text;
    }
    if(integer(preview.cleaned_char_count)!==current.length){var mismatch=text('p','Cleaning preview length does not match the audited result.');mismatch.className='error';root.appendChild(mismatch);}
  }
  function renderFinding(row){
    var article=document.createElement('article');article.className='finding';
    article.appendChild(text('h3',row.source_id+' · '+row.source_doc_id));
    var meta=document.createElement('dl');meta.className='metadata';
    metadata('Opaque review document',row.opaque_id).forEach(function(node){meta.appendChild(node);});
    metadata('Text SHA-256',row.text_sha256).forEach(function(node){meta.appendChild(node);});
    metadata('Earliest reported cut',row.earliest_cut_index).forEach(function(node){meta.appendChild(node);});
    var preview=row.cleaning_preview||{};
    metadata('Proposed replacements',formatted(preview.replacement_count)).forEach(function(node){meta.appendChild(node);});
    metadata('Characters removed',formatted(preview.characters_removed)).forEach(function(node){meta.appendChild(node);});
    metadata('Structural findings preserved',formatted(Array.isArray(preview.preserved_findings)?preview.preserved_findings.length:0)).forEach(function(node){meta.appendChild(node);});
    article.appendChild(meta);
    var badges=document.createElement('div');badges.className='rule-list';(row.rules||[]).forEach(function(rule){var badge=text('span',String(rule.rule)+' @ '+String(rule.cut_index));badge.className='badge';badges.appendChild(badge);});if(badges.childNodes.length)article.appendChild(badges);
    if(row.streaming_reason){var streaming=text('p','Streaming guard reason: '+String(row.streaming_reason));streaming.className='streaming';article.appendChild(streaming);}
    var button=text('button','Load raw evidence');button.className='load-evidence';button.type='button';
    button.addEventListener('click',function(){
      button.disabled=true;button.textContent='Loading raw evidence…';
      loadRaw(row).then(function(result){
        if(!result.ok){var error=text('p','Raw document could not be loaded: '+result.message);error.className='error';article.appendChild(error);button.disabled=false;button.textContent='Retry raw evidence';return;}
        button.remove();
        appendCleaningPreview(article,result.text,row.cleaning_preview);
        (row.rules||[]).sort(function(a,b){return Number(a.cut_index)-Number(b.cut_index)||String(a.rule).localeCompare(String(b.rule));}).forEach(function(rule){appendExcerpt(article,result.text,rule);});
      });
    });
    article.appendChild(button);
    return article;
  }
  function loadRaw(row){return fetch('data/documents/'+encodeURIComponent(row.opaque_id)+'.json',{cache:'no-store'}).then(function(response){if(!response.ok)throw new Error('HTTP '+response.status);return response.json();}).then(function(payload){if(!payload||typeof payload.text!=='string')throw new Error('invalid document payload');return {ok:true,text:payload.text};}).catch(function(error){return {ok:false,message:error.message||'request failed'};});}
  fetch('data/vlm_repetition_audit.json',{cache:'no-store'}).then(function(response){if(!response.ok)throw new Error('HTTP '+response.status);return response.json();}).then(function(audit){
    if(!audit||!Array.isArray(audit.documents))throw new Error('invalid audit payload');
    var findings=audit.documents.filter(isFinding).sort(function(a,b){return String(a.source_id).localeCompare(String(b.source_id))||Number(a.earliest_cut_index)-Number(b.earliest_cut_index)||String(a.opaque_id).localeCompare(String(b.opaque_id));});
    renderSummary(audit,findings);clear(findingsRoot);
    findings.forEach(function(row){findingsRoot.appendChild(renderFinding(row));});
  }).catch(function(error){failure(summaryRoot,'Detector audit failed to load: '+error.message);failure(findingsRoot,'No detections can be presented until the audit payload is available.');});
})();
"""


def _tree_inventory(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"site output contains forbidden symlink: {path}")
        if path.is_file():
            records.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return records


def build_site(
    *,
    packet_root: Path,
    packet_manifest: Path,
    requests_path: Path,
    responses_path: Path,
    site_secret_hex: str,
    output_dir: Path,
    max_portable_assets_bytes: int = DEFAULT_MAX_PORTABLE_ASSETS_BYTES,
) -> dict[str, object]:
    if not HEX_SHA256_RE.fullmatch(site_secret_hex):
        raise ValueError("site-secret must be a 32-byte lowercase hexadecimal value")
    secret = bytes.fromhex(site_secret_hex)
    if len(secret) != 32:
        raise ValueError("site-secret must encode 32 bytes")
    if max_portable_assets_bytes < 1:
        raise ValueError("portable asset size limit must be positive")
    packet_root = packet_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite review site: {output_dir}")
    manifest, requests, responses = _validate_inputs(packet_root, packet_manifest, requests_path, responses_path)
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=parent))
    try:
        os.chmod(staging, 0o700)
        cards: list[dict[str, object]] = []
        cards_by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
        source_counts: Counter[str] = Counter()
        artifact_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for request in sorted(requests, key=lambda row: (str(row["source_id"]), str(row["sample_id"]))):
            response = responses[str(request["request_id"])]
            opaque_id = _opaque_id(secret, str(request["request_id"]))
            document = _relative_document(packet_root, request)
            text = document.read_text(encoding="utf-8")
            _write_json(
                staging / "data" / "documents" / f"{opaque_id}.json",
                {"schema_version": "agent1_v4_raw_review_site_document_v1", "opaque_id": opaque_id, "text": text},
            )
            card = _index_card(request, response, opaque_id)
            cards.append(card)
            source_id = str(request["source_id"])
            cards_by_source[source_id].append(card)
            source_counts[source_id] += 1
            artifact_counts[source_id].update(_artifact_count(card))
        if (
            len(cards) != manifest.get("logical_review_count")
            or len(source_counts) != 18
            or dict(sorted(source_counts.items())) != manifest.get("source_counts")
        ):
            raise ValueError("site source/document closure failed")
        source_files: dict[str, str] = {}
        for source_id, source_cards in sorted(cards_by_source.items()):
            source_file = f"{hashlib.sha256(source_id.encode('utf-8')).hexdigest()[:24]}.json"
            source_files[source_id] = f"data/sources/{source_file}"
            _write_json(
                staging / "data" / "sources" / source_file,
                {
                    "schema_version": "agent1_v4_raw_review_site_source_cards_v1",
                    "source_id": source_id,
                    "cards": source_cards,
                },
            )
        index = {
            "schema_version": SITE_INDEX_SCHEMA,
            "packet_manifest_sha256": file_binding(packet_manifest)["sha256"],
            "source_counts": dict(sorted(source_counts.items())),
            "artifact_counts": {source_id: dict(sorted(counts.items())) for source_id, counts in sorted(artifact_counts.items())},
            "source_files": source_files,
            "documents": [
                {
                    "request_id": card["request_id"],
                    "source_id": card["source_id"],
                    "source_doc_id": card["source_doc_id"],
                }
                for card in cards
            ],
        }
        _write_json(staging / "data" / "index.json", index)
        _write_file(staging / "index.html", _html().encode("utf-8"))
        _write_file(staging / "detections.html", _detections_html().encode("utf-8"))
        _write_file(staging / "normalization.html", _normalization_html().encode("utf-8"))
        _write_file(staging / "assets" / "site.css", _css().encode("utf-8"))
        _write_file(staging / "assets" / "site.js", _lazy_js().encode("utf-8"))
        _write_file(staging / "assets" / "html-render.js", _html_renderer_js().encode("utf-8"))
        _write_file(staging / "assets" / "vlm-repetition.js", _vlm_repetition_viewer_js().encode("utf-8"))
        _write_file(staging / "assets" / "detections.css", _detections_css().encode("utf-8"))
        _write_file(staging / "assets" / "detections.js", _detections_js().encode("utf-8"))
        _write_file(staging / "assets" / "normalization.css", _normalization_css().encode("utf-8"))
        _write_file(staging / "assets" / "normalization.js", _normalization_js().encode("utf-8"))
        inventory = _tree_inventory(staging)
        portable_assets = [
            item for item in inventory
            if not str(item["path"]).startswith(("data/documents/", "data/gfm/documents/"))
        ]
        portable_asset_bytes = sum(int(item["bytes"]) for item in portable_assets)
        if portable_asset_bytes > max_portable_assets_bytes:
            raise ValueError(
                f"portable site assets exceed frozen limit: {portable_asset_bytes} > {max_portable_assets_bytes}"
            )
        site_manifest = {
            "schema_version": SITE_SCHEMA,
            "status": "passed",
            "packet_manifest": file_binding(packet_manifest),
            "requests": file_binding(requests_path),
            "responses": file_binding(responses_path),
            "site_secret_sha256": hashlib.sha256(secret).hexdigest(),
            "source_count": len(source_counts),
            "document_count": len(cards),
            "portable_assets": portable_assets,
            "portable_asset_bytes": portable_asset_bytes,
            "max_portable_assets_bytes": max_portable_assets_bytes,
            "files": inventory,
        }
        _write_json(staging / "site_manifest.json", site_manifest)
        if output_dir.exists():
            raise FileExistsError(f"site appeared during build: {output_dir}")
        os.rename(staging, output_dir)
        return site_manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


class NoDirectoryListingHandler(SimpleHTTPRequestHandler):
    def list_directory(self, path: str):  # type: ignore[override]
        self.send_error(403, "Directory listing disabled")
        return None

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        super().end_headers()


def serve_site(site_dir: Path, *, port: int, bind: str) -> None:
    site_dir = site_dir.resolve()
    if bind != "127.0.0.1":
        raise ValueError("review site must bind only to 127.0.0.1")
    manifest = site_dir / "site_manifest.json"
    value = read_json_object(manifest)
    if value.get("schema_version") != SITE_SCHEMA or value.get("status") != "passed":
        raise ValueError("site manifest is not passed")
    handler = functools.partial(NoDirectoryListingHandler, directory=str(site_dir))
    with ThreadingHTTPServer((bind, port), handler) as server:
        print(f"serving {site_dir} on http://{bind}:{port}/")
        server.serve_forever()


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--packet-root", type=Path, required=True)
    build.add_argument("--packet-manifest", type=Path, required=True)
    build.add_argument("--requests", type=Path, required=True)
    build.add_argument("--responses", type=Path, required=True)
    build.add_argument("--site-secret", required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--max-portable-assets-bytes", type=int, default=DEFAULT_MAX_PORTABLE_ASSETS_BYTES)
    serve = commands.add_parser("serve")
    serve.add_argument("--site-dir", type=Path, required=True)
    serve.add_argument("--port", type=int, required=True)
    serve.add_argument("--bind", default="127.0.0.1")
    args = parser.parse_args(argv)
    if args.command == "build":
        site_manifest = build_site(
            packet_root=args.packet_root,
            packet_manifest=args.packet_manifest,
            requests_path=args.requests,
            responses_path=args.responses,
            site_secret_hex=args.site_secret,
            output_dir=args.output_dir,
            max_portable_assets_bytes=args.max_portable_assets_bytes,
        )
        print(json.dumps({"ok": True, "documents": site_manifest["document_count"]}, sort_keys=True))
    else:
        serve_site(args.site_dir, port=args.port, bind=args.bind)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
