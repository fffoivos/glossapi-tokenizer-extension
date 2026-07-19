#!/usr/bin/env python3
"""Build a static side-by-side reader for sealed bibliography passes A and B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contract import sha256_file


SITE_SCHEMA = "bibliography-sealed-ab-comparison-site-v3"
BIB_ROLES = frozenset(
    {"ENTRY", "CONTINUATION", "FILLER", "BIB_HEADER", "BIB_SUBHEADER"}
)
ROLE_NAMES = (
    "ENTRY",
    "CONTINUATION",
    "FILLER",
    "BIB_HEADER",
    "BIB_SUBHEADER",
    "NON_BIB_HEADER",
    "OTHER",
    "UNKNOWN",
)


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Sealed A/B annotation comparison</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header class="top">
    <div class="title">
      <h1>Sealed A/B annotation comparison</h1>
      <span id="doc-summary" class="muted">Loading…</span>
    </div>
    <div class="actions">
      <button id="previous-disagreement" type="button">↑ Previous disagreement</button>
      <button id="next-disagreement" type="button">↓ Next disagreement</button>
    </div>
    <div id="legend" class="legend" aria-label="Role colours"></div>
  </header>
  <main class="layout">
    <aside class="sidebar">
      <div id="dataset-agreement" class="dataset-agreement"></div>
      <div class="side-heading">
        <strong>Documents</strong>
        <span class="muted">Worst agreement first</span>
      </div>
      <nav id="documents" aria-label="Sealed documents"></nav>
    </aside>
    <section class="reader" aria-live="polite">
      <div id="document-task-summary" class="document-task-summary"></div>
      <div class="column-heads">
        <div class="pass-head"><strong>Pass A</strong><span id="pass-a-model">annotator</span></div>
        <span>line</span>
        <div class="pass-head"><strong>Pass B</strong><span id="pass-b-model">annotator</span></div>
      </div>
      <div id="lines" class="lines"></div>
    </section>
  </main>
  <script src="app.js"></script>
</body>
</html>
"""


STYLES_CSS = """:root{
  --paper:#f3efe6;--card:#fffdf8;--ink:#18201c;--muted:#68716c;--line:#d8d1c4;
  --entry:#a33c36;--entry-bg:#f8dfdc;--continuation:#c66b32;--continuation-bg:#f9e4d6;
  --filler:#9a7528;--filler-bg:#f6edcf;--bib-header:#28628f;--bib-header-bg:#dceaf5;
  --bib-subheader:#557da3;--bib-subheader-bg:#e4edf6;--non-bib-header:#58615c;
  --non-bib-header-bg:#e5e8e6;--other:#7a817d;--other-bg:#f5f3ed;
  --unknown:#7b5894;--unknown-bg:#ede2f4;--bad:#a33c36;--good:#24714b;
  --shadow:0 14px 40px #26352b17;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:linear-gradient(140deg,#ebe4d7,#f8f5ed 52%,#e4ece7);color:var(--ink);font:14px/1.45 Inter,system-ui,sans-serif}
button{font:inherit;cursor:pointer}
.top{position:sticky;top:0;z-index:20;background:#faf7f0f5;border-bottom:1px solid var(--line);backdrop-filter:blur(14px);padding:10px 16px;display:grid;grid-template-columns:minmax(260px,1fr) auto;gap:8px 16px;align-items:center}
.title{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}.title h1{font:700 22px Georgia,serif;margin:0}.muted{color:var(--muted);font-size:12px}
.actions{display:flex;gap:7px;justify-content:flex-end}.actions button{border:1px solid var(--line);background:var(--card);border-radius:8px;padding:7px 10px;color:var(--ink)}.actions button:hover{background:#eee9df}
.legend{grid-column:1/-1;display:flex;gap:6px 10px;flex-wrap:wrap}.legend-item{display:flex;gap:5px;align-items:center;font-size:11px;color:var(--muted)}.swatch{width:11px;height:11px;border-radius:3px;background:var(--role-bg);border-left:4px solid var(--role)}
.layout{max-width:1900px;margin:14px auto 70px;display:grid;grid-template-columns:390px minmax(0,1fr);gap:14px;padding:0 14px}
.sidebar{position:sticky;top:112px;max-height:calc(100vh - 128px);overflow:auto;background:#faf7f0;border:1px solid var(--line);border-radius:13px;padding:9px}.dataset-agreement{padding:4px 5px 10px;border-bottom:1px solid var(--line);margin-bottom:5px}.dataset-agreement strong{display:block;font:700 14px Georgia,serif;margin-bottom:6px}.dataset-row{padding:6px 0;border-top:1px solid #ded7ca}.dataset-row:first-of-type{border-top:0}.dataset-row b{display:block;font-size:11px}.task-values{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:3px;margin-top:3px}.task-values span{font-size:9px;line-height:1.2;text-align:center;color:var(--muted)}.task-values em{display:block;color:var(--ink);font-style:normal;font-weight:800}.side-heading{padding:5px 6px 9px;display:flex;justify-content:space-between;gap:8px;align-items:baseline}.doc{display:block;width:100%;text-align:left;margin:4px 0;border:0;background:#ebe6dc;padding:8px 9px;border-radius:8px;color:var(--ink)}.doc:hover{background:#dfd8ca}.doc.active{background:#24372d;color:#fff}.doc strong,.doc small{display:block}.doc strong{font-size:12px}.doc small{opacity:.75;font-size:10px;margin-top:2px}
.reader{background:var(--card);border:1px solid var(--line);border-radius:15px;box-shadow:var(--shadow);overflow:visible}.document-task-summary{padding:10px 12px 8px;border-bottom:1px solid var(--line)}.document-task-summary>strong{display:block;font:700 13px Georgia,serif}.document-task-summary .task-values{max-width:720px}.column-heads{position:sticky;top:80px;z-index:10;display:grid;grid-template-columns:minmax(0,1fr) 74px minmax(0,1fr);gap:8px;padding:10px 12px;background:#fffdf8f2;border-bottom:1px solid var(--line);backdrop-filter:blur(12px);text-align:center}.column-heads strong{font:700 17px Georgia,serif}.column-heads span{color:var(--muted);font-size:11px}.pass-head{display:flex;justify-content:center;align-items:baseline;gap:8px;flex-wrap:wrap}.pass-head span{background:#e7e1d5;color:var(--ink);border-radius:999px;padding:2px 7px;font-weight:700}
.lines{padding:9px}.compare-row{display:grid;grid-template-columns:minmax(0,1fr) 74px minmax(0,1fr);gap:8px;align-items:stretch;margin:2px 0;scroll-margin-top:170px}.annotation{position:relative;display:grid;grid-template-columns:110px minmax(0,1fr);gap:9px;padding:7px 9px;border-radius:8px;background:var(--role-bg);border-left:5px solid var(--role);min-width:0}.labels{align-self:start;display:grid;gap:4px}.role{border-radius:6px;background:var(--role);color:#fff;padding:4px 6px;font-size:10px;font-weight:800;letter-spacing:.025em;text-align:center;overflow-wrap:anywhere}.model,.repair{font-size:9px;font-weight:800;letter-spacing:.04em;text-align:center;text-transform:uppercase}.model{color:var(--muted)}.repair{color:var(--bad)}.text{font:15px/1.42 Georgia,"Times New Roman",serif;overflow-wrap:anywhere;white-space:pre-wrap}.line-marker{display:flex;flex-direction:column;align-items:center;justify-content:center;color:var(--muted);font:10px/1.25 ui-monospace,SFMono-Regular,Menlo,monospace;text-align:center}.line-marker b{font:800 16px/1 system-ui;color:var(--bad)}.compare-row.binary-agree .line-marker b{color:var(--good)}.compare-row.exact-agree .line-marker b{opacity:.28}.compare-row.binary-disagree{background:#f7e3e04d;border-radius:9px}.compare-row.binary-disagree .line-marker{background:#f8dfdc;border-radius:7px;color:#7e302c}
.role-ENTRY{--role:var(--entry);--role-bg:var(--entry-bg)}.role-CONTINUATION{--role:var(--continuation);--role-bg:var(--continuation-bg)}.role-FILLER{--role:var(--filler);--role-bg:var(--filler-bg)}.role-BIB_HEADER{--role:var(--bib-header);--role-bg:var(--bib-header-bg)}.role-BIB_SUBHEADER{--role:var(--bib-subheader);--role-bg:var(--bib-subheader-bg)}.role-NON_BIB_HEADER{--role:var(--non-bib-header);--role-bg:var(--non-bib-header-bg)}.role-OTHER{--role:var(--other);--role-bg:var(--other-bg)}.role-UNKNOWN{--role:var(--unknown);--role-bg:var(--unknown-bg)}
.loading,.empty{padding:40px;text-align:center;color:var(--muted)}
@media(max-width:1050px){.layout{grid-template-columns:310px minmax(0,1fr)}.annotation{grid-template-columns:90px minmax(0,1fr)}.compare-row,.column-heads{grid-template-columns:minmax(0,1fr) 55px minmax(0,1fr)}}
@media(max-width:760px){.top{grid-template-columns:1fr}.actions{justify-content:flex-start}.layout{grid-template-columns:1fr}.sidebar{position:static;max-height:220px}.column-heads{top:154px}.compare-row{grid-template-columns:1fr 44px 1fr;gap:4px}.annotation{grid-template-columns:1fr;padding:6px}.text{font-size:13px}.role{width:max-content;max-width:100%}}
"""


APP_JS = """const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[c]));
const roles=['ENTRY','CONTINUATION','FILLER','BIB_HEADER','BIB_SUBHEADER','NON_BIB_HEADER','OTHER','UNKNOWN'];
let manifest=null,current=null,currentIndex=0,disagreements=[],disagreementIndex=-1;
function roleClass(role){return 'role-'+String(role).replace(/[^A-Z_]/g,'')}
function shortId(doc){return doc.source_doc_id.length>22?doc.source_doc_id.slice(0,12)+'…':doc.source_doc_id}
function renderLegend(){$('legend').innerHTML=roles.map(r=>`<span class="legend-item ${roleClass(r)}"><i class="swatch"></i>${r.replaceAll('_',' ')}</span>`).join('')}
function pct(value){return `${(100*value).toFixed(1)}%`}
function compared(after,before){return before!==undefined&&Math.abs(after-before)>0.00005?`${pct(before)}→${pct(after)}`:pct(after)}
function taskValues(tasks,original){const h=tasks.heading_types,g=tasks.gap_line_types,oh=original?.heading_types,og=original?.gap_line_types;return `<div class="task-values"><span title="BIB versus non-BIB agreement; original to repaired"><em>${compared(tasks.bibliography_membership.exact_agreement,original?.bibliography_membership.exact_agreement)}</em>BIB</span><span title="Both found a heading among lines where either found one; original to repaired"><em>${compared(h.detection_agreement,oh?.detection_agreement)}</em>head found</span><span title="Heading subtype agreement when both found a heading; original to repaired"><em>${compared(h.both_identified_a_heading.exact_agreement,oh?.both_identified_a_heading.exact_agreement)}</em>head type</span><span title="Both found filler/continuation among lines where either found one; original to repaired"><em>${compared(g.detection_agreement,og?.detection_agreement)}</em>F/C found</span><span title="Filler versus continuation agreement when both found one; original to repaired"><em>${compared(g.both_identified_a_gap_line.exact_agreement,og?.both_identified_a_gap_line.exact_agreement)}</em>F/C type</span></div>`}
function renderDatasetAgreement(){$('dataset-agreement').innerHTML=`<strong>Agreement by dataset</strong>`+Object.entries(manifest.source_summary).map(([source,value])=>`<div class="dataset-row"><b>${esc(source.replaceAll('_',' '))} · ${value.document_count} docs</b>${taskValues(value.task_agreement,value.original_task_agreement)}</div>`).join('')}
function renderDocuments(){$('documents').innerHTML=manifest.documents.map((d,i)=>`<button type="button" class="doc ${i===currentIndex?'active':''}" data-index="${i}"><strong>${i+1}. ${esc(d.source)} · ${esc(shortId(d))}</strong><small>A: ${esc(d.pass_a_model)} · B: ${esc(d.pass_b_model)} · ${(100*d.binary_agreement).toFixed(2)}% repaired agreement · changed A/B ${d.pass_a_repaired_line_count}/${d.pass_b_repaired_line_count} · ${d.line_count.toLocaleString()} lines</small></button>`).join('')}
function annotation(role,original,text,model){const changed=role!==original;return `<div class="annotation ${roleClass(role)}"><span class="labels"><span class="role">${esc(role.replaceAll('_',' '))}</span><span class="model">${esc(model)}</span>${changed?`<span class="repair">was ${esc(original.replaceAll('_',' '))}</span>`:''}</span><span class="text">${esc(text)}</span></div>`}
function renderLine(line,index){const state=line.exact_agree?'exact-agree binary-agree':line.binary_agree?'binary-agree role-disagree':'binary-disagree';const mark=line.exact_agree?'=':line.binary_agree?'≈':'≠';const title=line.exact_agree?'Exact role agreement':line.binary_agree?'Same BIB/NON-BIB decision; detailed roles differ':'Binary BIB/NON-BIB disagreement';return `<div id="line-${index}" class="compare-row ${state}" data-line="${index}">${annotation(line.pass_a_role,line.pass_a_original_role,line.text,line.pass_a_model)}<span class="line-marker" title="${title}">L${line.abs_idx}<b>${mark}</b></span>${annotation(line.pass_b_role,line.pass_b_original_role,line.text,line.pass_b_model)}</div>`}
function renderCurrent(){renderDocuments();disagreements=[];current.lines.forEach((line,i)=>{if(!line.binary_agree)disagreements.push(i)});$('lines').innerHTML=current.lines.map(renderLine).join('');const d=manifest.documents[currentIndex];$('document-task-summary').innerHTML=`<strong>This document · ${esc(d.source)} · ${esc(shortId(d))} · original→repaired</strong>${taskValues(d.task_agreement,d.original_task_agreement)}`;$('pass-a-model').textContent=d.pass_a_model_details;$('pass-b-model').textContent=d.pass_b_model_details;$('doc-summary').textContent=`${d.source} · ${shortId(d)} · A: ${d.pass_a_model} · B: ${d.pass_b_model} · ${(100*d.binary_agreement).toFixed(2)}% repaired agreement · changed A/B ${d.pass_a_repaired_line_count}/${d.pass_b_repaired_line_count}`;document.title=`${(100*d.binary_agreement).toFixed(2)}% · ${shortId(d)} · A/B comparison`;disagreementIndex=-1;window.scrollTo({top:0,behavior:'instant'});history.replaceState(null,'','#'+d.document_id)}
async function loadDocument(index){currentIndex=Math.max(0,Math.min(manifest.documents.length-1,index));$('lines').innerHTML='<div class="loading">Loading document…</div>';renderDocuments();const d=manifest.documents[currentIndex];const response=await fetch('data/'+d.document_id+'.json');if(!response.ok)throw new Error('Could not load '+d.document_id);current=await response.json();renderCurrent()}
function jump(direction){if(!disagreements.length)return;const visible=disagreements.findIndex(i=>document.getElementById('line-'+i)?.getBoundingClientRect().top>170);if(direction>0){disagreementIndex=visible>=0?visible:0}else{const y=window.scrollY+170;let previous=-1;disagreements.forEach((line,i)=>{if(document.getElementById('line-'+line)?.offsetTop<y)previous=i});disagreementIndex=previous>0?previous-1:disagreements.length-1}document.getElementById('line-'+disagreements[disagreementIndex])?.scrollIntoView({behavior:'smooth',block:'center'})}
document.addEventListener('click',event=>{const button=event.target.closest('[data-index]');if(button)loadDocument(Number(button.dataset.index))});
$('previous-disagreement').onclick=()=>jump(-1);$('next-disagreement').onclick=()=>jump(1);
document.addEventListener('keydown',event=>{if(event.key.toLowerCase()==='n')jump(1);if(event.key.toLowerCase()==='p')jump(-1)});
async function start(){renderLegend();const response=await fetch('manifest.json');manifest=await response.json();renderDatasetAgreement();const wanted=location.hash.slice(1);const index=Math.max(0,manifest.documents.findIndex(d=>d.document_id===wanted));await loadDocument(index)}
start().catch(error=>{$('lines').innerHTML=`<div class="empty">${esc(error.message)}</div>`});
"""


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _display_model(model: str) -> str:
    lowered = model.casefold()
    if "sol" in lowered:
        return "Sol"
    if "terra" in lowered:
        return "Terra"
    raise ValueError(f"unsupported annotation model: {model!r}")


def load_annotation_provenance(
    packet_path: Path,
    response_dir: Path,
) -> dict[str, Any]:
    """Resolve each owned line to the model that actually annotated its chunk."""

    packets = _read_jsonl(packet_path)
    packet_by_chunk: dict[str, dict[str, Any]] = {}
    for packet in packets:
        chunk_id = str(packet.get("chunk_id") or "")
        if not chunk_id or chunk_id in packet_by_chunk:
            raise ValueError("packet has empty or duplicate chunk IDs")
        packet_by_chunk[chunk_id] = packet

    model_by_chunk: dict[str, str] = {}
    model_ids: dict[str, set[str]] = defaultdict(set)
    response_count = 0
    for path in sorted(response_dir.glob("*.json")):
        record = _read_json(path)
        runtime = record.get("annotation_runtime")
        review = record.get("review")
        if not isinstance(runtime, dict) or not isinstance(review, dict):
            raise ValueError(f"invalid annotation response: {path}")
        model_id = str(runtime.get("model") or "")
        model = _display_model(model_id)
        model_ids[model].add(model_id)
        chunks = review.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            raise ValueError(f"annotation response has no chunks: {path}")
        response_count += 1
        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id") or "")
            if not chunk_id or chunk_id in model_by_chunk:
                raise ValueError(f"response has empty or duplicate chunk: {chunk_id!r}")
            model_by_chunk[chunk_id] = model
    if set(packet_by_chunk) != set(model_by_chunk):
        missing = sorted(set(packet_by_chunk) - set(model_by_chunk))[:3]
        extra = sorted(set(model_by_chunk) - set(packet_by_chunk))[:3]
        raise ValueError(f"packet/response chunk mismatch; missing={missing}, extra={extra}")

    line_models: dict[str, str] = {}
    document_chunk_counts: dict[str, Counter[str]] = defaultdict(Counter)
    document_line_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for chunk_id, packet in packet_by_chunk.items():
        model = model_by_chunk[chunk_id]
        document_alias = str(packet.get("document_alias") or "")
        if not document_alias:
            raise ValueError(f"packet chunk has no document alias: {chunk_id}")
        document_chunk_counts[document_alias][model] += 1
        owned_start = int(packet.get("owned_start_offset", 0))
        owned_end = int(packet.get("owned_end_offset_exclusive", 0))
        for line in packet.get("lines", []):
            offset = int(line.get("offset", -1))
            if not owned_start <= offset < owned_end:
                continue
            line_alias = str(line.get("line_alias") or "")
            if not line_alias or line_alias in line_models:
                raise ValueError(f"owned line is empty or duplicated: {line_alias!r}")
            line_models[line_alias] = model
            document_line_counts[document_alias][model] += 1

    documents: dict[str, dict[str, Any]] = {}
    for document_alias, chunk_counts in document_chunk_counts.items():
        models = sorted(chunk_counts)
        documents[document_alias] = {
            "model": models[0] if len(models) == 1 else "Mixed",
            "models": models,
            "chunk_counts": dict(sorted(chunk_counts.items())),
            "owned_line_counts": dict(sorted(document_line_counts[document_alias].items())),
        }
    return {
        "line_models": line_models,
        "documents": documents,
        "response_count": response_count,
        "chunk_count": len(packet_by_chunk),
        "model_ids": {model: sorted(ids) for model, ids in sorted(model_ids.items())},
    }


def _agreement_metrics(
    pairs: Sequence[tuple[str, str]],
    labels: Sequence[str],
) -> dict[str, Any]:
    confusion = {left: {right: 0 for right in labels} for left in labels}
    for left, right in pairs:
        confusion[left][right] += 1
    count = len(pairs)
    exact = sum(confusion[label][label] for label in labels)
    a_counts = {label: sum(confusion[label].values()) for label in labels}
    b_counts = {
        label: sum(confusion[left][label] for left in labels) for label in labels
    }
    chance = (
        sum(a_counts[label] * b_counts[label] for label in labels) / (count * count)
        if count
        else 0.0
    )
    observed = exact / count if count else 0.0
    kappa = (observed - chance) / (1.0 - chance) if count and chance < 1.0 else 0.0
    return {
        "line_count": count,
        "exact_agreement": observed,
        "cohen_kappa": kappa,
        "confusion_a_to_b": confusion,
        "per_class": {
            label: {
                "pass_a_count": a_counts[label],
                "pass_b_count": b_counts[label],
                "agreed_count": confusion[label][label],
                "symmetric_f1": (
                    2.0 * confusion[label][label] / (a_counts[label] + b_counts[label])
                    if a_counts[label] + b_counts[label]
                    else 0.0
                ),
            }
            for label in labels
        },
    }


def build_task_agreement(lines: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Measure agreement after recoding roles for the three downstream tasks."""

    binary_pairs: list[tuple[str, str]] = []
    heading_union: list[tuple[str, str]] = []
    heading_both: list[tuple[str, str]] = []
    gap_union: list[tuple[str, str]] = []
    gap_both: list[tuple[str, str]] = []
    heading_labels = ("BIB_HEADER", "BIB_SUBHEADER", "NON_BIB_HEADER")
    gap_labels = ("FILLER", "CONTINUATION")
    for line in lines:
        role_a = str(line["pass_a_role"])
        role_b = str(line["pass_b_role"])
        if "UNKNOWN" not in (role_a, role_b):
            binary_pairs.append(
                ("BIB" if role_a in BIB_ROLES else "NON_BIB", "BIB" if role_b in BIB_ROLES else "NON_BIB")
            )
        heading_a = role_a if role_a in heading_labels else "NON_HEADER"
        heading_b = role_b if role_b in heading_labels else "NON_HEADER"
        if "NON_HEADER" not in (heading_a, heading_b):
            heading_both.append((heading_a, heading_b))
        if (heading_a, heading_b) != ("NON_HEADER", "NON_HEADER"):
            heading_union.append((heading_a, heading_b))
        gap_a = role_a if role_a in gap_labels else "OTHER"
        gap_b = role_b if role_b in gap_labels else "OTHER"
        if "OTHER" not in (gap_a, gap_b):
            gap_both.append((gap_a, gap_b))
        if (gap_a, gap_b) != ("OTHER", "OTHER"):
            gap_union.append((gap_a, gap_b))
    return {
        "bibliography_membership": _agreement_metrics(binary_pairs, ("BIB", "NON_BIB")),
        "heading_types": {
            "detection_agreement": (
                len(heading_both) / len(heading_union) if heading_union else 1.0
            ),
            "candidate_union_including_missed_headings": _agreement_metrics(
                heading_union, (*heading_labels, "NON_HEADER")
            ),
            "both_identified_a_heading": _agreement_metrics(heading_both, heading_labels),
        },
        "gap_line_types": {
            "detection_agreement": len(gap_both) / len(gap_union) if gap_union else 1.0,
            "candidate_union_including_missed_gap_lines": _agreement_metrics(
                gap_union, (*gap_labels, "OTHER")
            ),
            "both_identified_a_gap_line": _agreement_metrics(gap_both, gap_labels),
        },
    }


def _pass_lines(value: Mapping[str, Any], name: str) -> dict[str, dict[str, Any]]:
    rows = value.get("lines")
    if not isinstance(rows, list):
        raise ValueError(f"{name} has no line inventory")
    indexed = {str(row.get("line_alias") or ""): dict(row) for row in rows}
    if "" in indexed or len(indexed) != len(rows):
        raise ValueError(f"{name} has empty or duplicate line aliases")
    invalid = sorted({str(row.get("role")) for row in rows} - set(ROLE_NAMES))
    if invalid:
        raise ValueError(f"{name} has invalid roles: {invalid}")
    return indexed


def _binary(role: str) -> bool | None:
    if role == "UNKNOWN":
        return None
    return role in BIB_ROLES


def build_documents(
    documents: Sequence[Mapping[str, Any]],
    line_keys: Sequence[Mapping[str, Any]],
    pass_a: Mapping[str, Any],
    pass_b: Mapping[str, Any],
    pass_a_provenance: Mapping[str, Any] | None = None,
    pass_b_provenance: Mapping[str, Any] | None = None,
    original_pass_a: Mapping[str, Any] | None = None,
    original_pass_b: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return manifest rows and document payloads in worst-agreement order."""

    a_lines = _pass_lines(pass_a, "pass A")
    b_lines = _pass_lines(pass_b, "pass B")
    original_a_lines = _pass_lines(original_pass_a, "original pass A") if original_pass_a else a_lines
    original_b_lines = _pass_lines(original_pass_b, "original pass B") if original_pass_b else b_lines
    key_by_coordinate: dict[tuple[str, str], Mapping[str, Any]] = {}
    expected_aliases: set[str] = set()
    for row in line_keys:
        coordinate = (str(row.get("document_id") or ""), str(row.get("line_id") or ""))
        alias = str(row.get("line_alias") or "")
        if "" in coordinate or not alias or coordinate in key_by_coordinate:
            raise ValueError("line key has empty or duplicate coordinates")
        key_by_coordinate[coordinate] = row
        expected_aliases.add(alias)
    if any(
        set(lines) != expected_aliases
        for lines in (a_lines, b_lines, original_a_lines, original_b_lines)
    ):
        raise ValueError("current and original passes must cover exactly the line-key aliases")
    provenance_a_lines = dict((pass_a_provenance or {}).get("line_models") or {})
    provenance_b_lines = dict((pass_b_provenance or {}).get("line_models") or {})
    if pass_a_provenance is not None and set(provenance_a_lines) != expected_aliases:
        raise ValueError("pass A provenance must cover exactly the line-key aliases")
    if pass_b_provenance is not None and set(provenance_b_lines) != expected_aliases:
        raise ValueError("pass B provenance must cover exactly the line-key aliases")

    manifest_rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    seen_coordinates: set[tuple[str, str]] = set()
    for document in documents:
        document_id = str(document.get("document_id") or "")
        source = str(document.get("source") or "")
        output_lines: list[dict[str, Any]] = []
        document_aliases: set[str] = set()
        disagreement_pairs: Counter[str] = Counter()
        binary_disagreements = 0
        exact_agreements = 0
        for line in document.get("lines", []):
            line_id = str(line.get("line_id") or "")
            coordinate = (document_id, line_id)
            key = key_by_coordinate.get(coordinate)
            if key is None or coordinate in seen_coordinates:
                raise ValueError(f"document line is absent or duplicated in line key: {coordinate}")
            seen_coordinates.add(coordinate)
            alias = str(key["line_alias"])
            document_alias = str(key.get("document_alias") or "")
            if document_alias:
                document_aliases.add(document_alias)
            role_a, role_b = str(a_lines[alias]["role"]), str(b_lines[alias]["role"])
            original_role_a = str(original_a_lines[alias]["role"])
            original_role_b = str(original_b_lines[alias]["role"])
            model_a = str(provenance_a_lines.get(alias) or "Unknown")
            model_b = str(provenance_b_lines.get(alias) or "Unknown")
            binary_a, binary_b = _binary(role_a), _binary(role_b)
            binary_agree = binary_a is not None and binary_a == binary_b
            exact_agree = role_a == role_b and role_a != "UNKNOWN"
            binary_disagreements += int(not binary_agree)
            exact_agreements += int(exact_agree)
            if not binary_agree:
                disagreement_pairs[f"{role_a}->{role_b}"] += 1
            output_lines.append(
                {
                    "line_id": line_id,
                    "abs_idx": int(line["abs_idx"]),
                    "text": str(line["text"]),
                    "pass_a_role": role_a,
                    "pass_b_role": role_b,
                    "pass_a_original_role": original_role_a,
                    "pass_b_original_role": original_role_b,
                    "pass_a_repaired": role_a != original_role_a,
                    "pass_b_repaired": role_b != original_role_b,
                    "pass_a_model": model_a,
                    "pass_b_model": model_b,
                    "annotator_pair": f"{model_a}->{model_b}",
                    "binary_agree": binary_agree,
                    "exact_agree": exact_agree,
                }
            )
        line_count = len(output_lines)
        if not line_count:
            raise ValueError(f"sealed document has no present lines: {document_id}")
        agreement = 1.0 - binary_disagreements / line_count
        original_output_lines = [
            {
                **line,
                "pass_a_role": line["pass_a_original_role"],
                "pass_b_role": line["pass_b_original_role"],
            }
            for line in output_lines
        ]
        if len(document_aliases) > 1:
            raise ValueError(f"document has multiple aliases: {document_id}")
        document_alias = next(iter(document_aliases), "")
        provenance_a_document = dict(
            ((pass_a_provenance or {}).get("documents") or {}).get(document_alias) or {}
        )
        provenance_b_document = dict(
            ((pass_b_provenance or {}).get("documents") or {}).get(document_alias) or {}
        )

        def model_details(value: Mapping[str, Any]) -> str:
            model = str(value.get("model") or "Unknown")
            counts = dict(value.get("chunk_counts") or {})
            if not counts:
                return model
            if len(counts) == 1:
                return f"{model} · {next(iter(counts.values()))} chunks"
            chunks = ", ".join(f"{name} {count}" for name, count in sorted(counts.items()))
            return f"{model} · {chunks} chunks"

        summary = {
            "document_id": document_id,
            "source": source,
            "source_doc_id": str(document.get("source_doc_id") or ""),
            "source_repo_id": str(document.get("source_repo_id") or ""),
            "source_dataset": str(document.get("source_dataset") or ""),
            "source_row_id": str(document.get("source_row_id") or ""),
            "work_id": str(document.get("work_id") or ""),
            "document_alias": document_alias,
            "pass_a_model": str(provenance_a_document.get("model") or "Unknown"),
            "pass_b_model": str(provenance_b_document.get("model") or "Unknown"),
            "pass_a_model_details": model_details(provenance_a_document),
            "pass_b_model_details": model_details(provenance_b_document),
            "pass_a_model_chunk_counts": dict(provenance_a_document.get("chunk_counts") or {}),
            "pass_b_model_chunk_counts": dict(provenance_b_document.get("chunk_counts") or {}),
            "line_count": line_count,
            "binary_agreement": agreement,
            "binary_disagreements": binary_disagreements,
            "exact_role_agreement": exact_agreements / line_count,
            "top_disagreement_pairs": [
                {"pair": pair, "count": count}
                for pair, count in disagreement_pairs.most_common(5)
            ],
            "task_agreement": build_task_agreement(output_lines),
            "original_task_agreement": build_task_agreement(original_output_lines),
            "pass_a_repaired_line_count": sum(
                bool(line["pass_a_repaired"]) for line in output_lines
            ),
            "pass_b_repaired_line_count": sum(
                bool(line["pass_b_repaired"]) for line in output_lines
            ),
        }
        manifest_rows.append(summary)
        payloads.append({"schema_version": SITE_SCHEMA, "document": summary, "lines": output_lines})
    if seen_coordinates != set(key_by_coordinate):
        raise ValueError("documents do not cover every line-key coordinate")

    order = sorted(
        range(len(manifest_rows)),
        key=lambda index: (
            manifest_rows[index]["binary_agreement"],
            -manifest_rows[index]["binary_disagreements"],
            manifest_rows[index]["document_id"],
        ),
    )
    return [manifest_rows[index] for index in order], [payloads[index] for index in order]


def _write_text(path: Path, value: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def build_site(
    *,
    documents_path: Path,
    line_key_path: Path,
    pass_a_path: Path,
    pass_b_path: Path,
    pass_a_packet_path: Path | None = None,
    pass_b_packet_path: Path | None = None,
    pass_a_response_dir: Path | None = None,
    pass_b_response_dir: Path | None = None,
    original_pass_a_path: Path | None = None,
    original_pass_b_path: Path | None = None,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    documents = _read_jsonl(documents_path)
    line_keys = _read_jsonl(line_key_path)
    pass_a, pass_b = _read_json(pass_a_path), _read_json(pass_b_path)
    if (original_pass_a_path is None) != (original_pass_b_path is None):
        raise ValueError("original pass A and B paths must be supplied together")
    original_pass_a = _read_json(original_pass_a_path) if original_pass_a_path else None
    original_pass_b = _read_json(original_pass_b_path) if original_pass_b_path else None
    provenance_args = (
        pass_a_packet_path,
        pass_b_packet_path,
        pass_a_response_dir,
        pass_b_response_dir,
    )
    if any(value is not None for value in provenance_args) and not all(
        value is not None for value in provenance_args
    ):
        raise ValueError("all packet and response paths are required when provenance is enabled")
    pass_a_provenance = (
        load_annotation_provenance(pass_a_packet_path, pass_a_response_dir)
        if pass_a_packet_path is not None and pass_a_response_dir is not None
        else None
    )
    pass_b_provenance = (
        load_annotation_provenance(pass_b_packet_path, pass_b_response_dir)
        if pass_b_packet_path is not None and pass_b_response_dir is not None
        else None
    )
    manifest_rows, payloads = build_documents(
        documents,
        line_keys,
        pass_a,
        pass_b,
        pass_a_provenance,
        pass_b_provenance,
        original_pass_a,
        original_pass_b,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.partial-", dir=output_dir.parent))
    try:
        data_dir = partial / "data"
        data_dir.mkdir()
        _write_text(partial / "index.html", INDEX_HTML)
        _write_text(partial / "styles.css", STYLES_CSS)
        _write_text(partial / "app.js", APP_JS)
        for summary, payload in zip(manifest_rows, payloads, strict=True):
            _write_json(data_dir / f"{summary['document_id']}.json", payload)
        line_count = sum(row["line_count"] for row in manifest_rows)
        disagreement_count = sum(row["binary_disagreements"] for row in manifest_rows)
        all_lines = [line for payload in payloads for line in payload["lines"]]
        all_original_lines = [
            {
                **line,
                "pass_a_role": line["pass_a_original_role"],
                "pass_b_role": line["pass_b_original_role"],
            }
            for line in all_lines
        ]
        lines_by_annotator_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for line in all_lines:
            lines_by_annotator_pair[str(line["annotator_pair"])].append(line)
        source_summary: dict[str, dict[str, Any]] = {}
        for source in sorted({row["source"] for row in manifest_rows}):
            source_rows = [row for row in manifest_rows if row["source"] == source]
            source_lines = [
                line
                for payload in payloads
                if payload["document"]["source"] == source
                for line in payload["lines"]
            ]
            source_original_lines = [
                {
                    **line,
                    "pass_a_role": line["pass_a_original_role"],
                    "pass_b_role": line["pass_b_original_role"],
                }
                for line in source_lines
            ]
            pair_counts = Counter(str(line["annotator_pair"]) for line in source_lines)
            source_summary[source] = {
                "document_count": len(source_rows),
                "line_count": sum(row["line_count"] for row in source_rows),
                "binary_disagreements": sum(
                    row["binary_disagreements"] for row in source_rows
                ),
                "annotator_pair_line_counts": dict(sorted(pair_counts.items())),
                "task_agreement": build_task_agreement(source_lines),
                "original_task_agreement": build_task_agreement(source_original_lines),
                "pass_a_repaired_line_count": sum(
                    bool(line["pass_a_repaired"]) for line in source_lines
                ),
                "pass_b_repaired_line_count": sum(
                    bool(line["pass_b_repaired"]) for line in source_lines
                ),
            }
        manifest = {
            "schema_version": SITE_SCHEMA,
            "status": "passed",
            "sort_order": "binary_agreement_ascending_then_disagreement_count_descending",
            "document_count": len(manifest_rows),
            "line_count": line_count,
            "binary_disagreement_count": disagreement_count,
            "binary_agreement": 1.0 - disagreement_count / line_count,
            "pass_a_reviewer": str(pass_a.get("reviewer") or ""),
            "pass_b_reviewer": str(pass_b.get("reviewer") or ""),
            "task_agreement": build_task_agreement(all_lines),
            "original_task_agreement": build_task_agreement(all_original_lines),
            "task_agreement_by_annotator_pair": {
                pair: {
                    "line_count": len(lines),
                    "tasks": build_task_agreement(lines),
                }
                for pair, lines in sorted(lines_by_annotator_pair.items())
            },
            "annotation_provenance": {
                "pass_a": {
                    key: pass_a_provenance[key]
                    for key in ("response_count", "chunk_count", "model_ids")
                }
                if pass_a_provenance is not None
                else None,
                "pass_b": {
                    key: pass_b_provenance[key]
                    for key in ("response_count", "chunk_count", "model_ids")
                }
                if pass_b_provenance is not None
                else None,
            },
            "source_summary": source_summary,
            "documents": manifest_rows,
        }
        _write_json(partial / "manifest.json", manifest)
        site_sha256 = _tree_sha256(partial)
        receipt = {
            "schema_version": "bibliography-sealed-ab-comparison-site-receipt-v2",
            "status": "passed",
            "site_sha256_before_receipt": site_sha256,
            "document_count": len(manifest_rows),
            "line_count": line_count,
            "binary_disagreement_count": disagreement_count,
            "binary_agreement": manifest["binary_agreement"],
            "inputs": {
                "documents_sha256": sha256_file(documents_path),
                "line_key_sha256": sha256_file(line_key_path),
                "pass_a_sha256": sha256_file(pass_a_path),
                "pass_b_sha256": sha256_file(pass_b_path),
                "pass_a_packet_sha256": (
                    sha256_file(pass_a_packet_path) if pass_a_packet_path is not None else None
                ),
                "pass_b_packet_sha256": (
                    sha256_file(pass_b_packet_path) if pass_b_packet_path is not None else None
                ),
                "original_pass_a_sha256": (
                    sha256_file(original_pass_a_path) if original_pass_a_path else None
                ),
                "original_pass_b_sha256": (
                    sha256_file(original_pass_b_path) if original_pass_b_path else None
                ),
            },
            "output": {"path": str(output_dir)},
            "selection_or_model_use": (
                "annotation QA only; no model predictions or candidate selection"
            ),
        }
        _write_json(partial / "receipt.json", receipt)
        os.replace(partial, output_dir)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--line-key", type=Path, required=True)
    parser.add_argument("--pass-a", type=Path, required=True)
    parser.add_argument("--pass-b", type=Path, required=True)
    parser.add_argument("--pass-a-packet", type=Path)
    parser.add_argument("--pass-b-packet", type=Path)
    parser.add_argument("--pass-a-response-dir", type=Path)
    parser.add_argument("--pass-b-response-dir", type=Path)
    parser.add_argument("--original-pass-a", type=Path)
    parser.add_argument("--original-pass-b", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = build_site(
        documents_path=args.documents.resolve(),
        line_key_path=args.line_key.resolve(),
        pass_a_path=args.pass_a.resolve(),
        pass_b_path=args.pass_b.resolve(),
        pass_a_packet_path=args.pass_a_packet.resolve() if args.pass_a_packet else None,
        pass_b_packet_path=args.pass_b_packet.resolve() if args.pass_b_packet else None,
        pass_a_response_dir=(
            args.pass_a_response_dir.resolve() if args.pass_a_response_dir else None
        ),
        pass_b_response_dir=(
            args.pass_b_response_dir.resolve() if args.pass_b_response_dir else None
        ),
        original_pass_a_path=(
            args.original_pass_a.resolve() if args.original_pass_a else None
        ),
        original_pass_b_path=(
            args.original_pass_b.resolve() if args.original_pass_b else None
        ),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
