#!/usr/bin/env python3
"""Build a full-document reader for high-impact bibliography recall failures."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .bibliography_entry_blocks import blocks_from_mask
from .bibliography_entry_dataset import LABEL_TO_ID
from .bibliography_entry_models import load_table
from .bibliography_feature_explorer import FEATURE_SPECS
from .bibliography_v2 import extract_bibliography_feature_review


SCHEMA_VERSION = "bibliography-entry-failure-reader-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _iter_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"input row {row_number} is not an object")
            yield row


def line_status(*, predicted: bool, silver_bib: bool) -> str:
    if predicted and silver_bib:
        return "agreement_bib"
    if predicted:
        return "classifier_only"
    if silver_bib:
        return "silver_only"
    return "agreement_non_bib"


def _block_rows(mask: np.ndarray, abs_indices: np.ndarray) -> list[dict[str, int]]:
    return [
        {
            "local_start": start,
            "local_end": end,
            "abs_start": int(abs_indices[start]),
            "abs_end": int(abs_indices[end]),
        }
        for start, end in blocks_from_mask(mask, abs_indices)
    ]


def _line_row(
    raw: Mapping[str, Any],
    *,
    local_index: int,
    probability: float,
    predicted: bool,
) -> dict[str, Any]:
    text, abs_idx = raw.get("text"), raw.get("abs_idx")
    if not isinstance(text, str) or not isinstance(abs_idx, int):
        raise ValueError("malformed input line")
    silver_bib = raw.get("label") == "BIB"
    normalized = unicodedata.normalize("NFKC", text)
    features: dict[str, int] = {}
    matches: dict[str, list[list[int]]] = {}
    token_count = int(raw.get("token_count", 0))
    if predicted:
        review = extract_bibliography_feature_review(text)
        normalized = review.normalized_text
        values = review.features.as_dict()
        token_count = int(raw.get("token_count", values["token_count"]))
        features = {
            spec.key: int(values[spec.key])
            for spec in FEATURE_SPECS
            if int(values[spec.key]) > 0
        }
        for match in review.matches:
            matches.setdefault(match.feature, []).append([match.start, match.end])
    return {
        "local_index": local_index,
        "abs_idx": abs_idx,
        "text": normalized,
        "char_length": len(normalized),
        "token_count": token_count,
        "entry_probability": round(probability, 7),
        "predicted_bib": predicted,
        "silver_bib": silver_bib,
        "status": line_status(predicted=predicted, silver_bib=silver_bib),
        "features": features,
        "matches": matches,
    }


def build_packet(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input).resolve()
    validation_root = Path(args.validation_dir).resolve()
    table = load_table(validation_root / "validation_table", expected_split="validation")
    probability = np.load(
        validation_root / "validation_line_probability.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    prediction = np.load(
        validation_root / "validation_block_prediction.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    requested = tuple(dict.fromkeys(str(value) for value in args.document_id))
    if not requested:
        raise ValueError("at least one document ID is required")
    table_documents = {str(row["document_id"]): row for row in table.documents}
    missing = set(requested) - table_documents.keys()
    if missing:
        raise ValueError(f"requested documents are absent from validation: {sorted(missing)}")
    raw_documents = {
        str(row.get("document_id")): row
        for row in _iter_rows(input_path)
        if str(row.get("document_id")) in requested
    }
    if set(requested) != raw_documents.keys():
        raise ValueError("requested document is absent from the pinned input")

    documents = []
    total_missed_tokens = 0
    for document_id in requested:
        meta = table_documents[document_id]
        raw = raw_documents[document_id]
        raw_lines = raw.get("lines")
        if not isinstance(raw_lines, list):
            raise ValueError(f"{document_id}: invalid line inventory")
        start, end = int(meta["line_start"]), int(meta["line_end"])
        if len(raw_lines) != end - start:
            raise ValueError(f"{document_id}: source/table alignment failure")
        pred = np.asarray(prediction[start:end], dtype=bool)
        gold = np.asarray(
            table.original_labels[start:end] == LABEL_TO_ID["BIB"], dtype=bool
        )
        tokens = np.asarray(table.token_counts[start:end], dtype=np.int64)
        tp, fp, fn = gold & pred, ~gold & pred, gold & ~pred
        gold_lines, predicted_lines = int(gold.sum()), int(pred.sum())
        gold_tokens, predicted_gold_tokens = int(tokens[gold].sum()), int(tokens[tp].sum())
        missed_tokens = int(tokens[fn].sum())
        total_missed_tokens += missed_tokens
        lines = [
            _line_row(
                line,
                local_index=index,
                probability=float(probability[start + index]),
                predicted=bool(pred[index]),
            )
            for index, line in enumerate(raw_lines)
        ]
        documents.append(
            {
                "document_id": document_id,
                "work_id": str(meta["work_id"]),
                "source": str(meta["source"]),
                "coverage": str(meta["coverage"]),
                "line_count": len(lines),
                "n_physical_lines": int(meta["n_physical_lines"]),
                "gold_lines": gold_lines,
                "predicted_lines": predicted_lines,
                "true_positive_lines": int(tp.sum()),
                "false_positive_lines": int(fp.sum()),
                "missed_lines": int(fn.sum()),
                "line_precision": int(tp.sum()) / max(predicted_lines, 1),
                "line_recall": int(tp.sum()) / max(gold_lines, 1),
                "gold_tokens": gold_tokens,
                "recovered_gold_tokens": predicted_gold_tokens,
                "missed_tokens": missed_tokens,
                "token_recall": predicted_gold_tokens / max(gold_tokens, 1),
                "silver_blocks": _block_rows(gold, table.abs_indices[start:end]),
                "predicted_blocks": _block_rows(pred, table.abs_indices[start:end]),
                "lines": lines,
            }
        )
    features = [
        {"key": spec.key, "label": spec.label, "group": spec.group, "color": spec.color}
        for spec in FEATURE_SPECS
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "title": "Bibliography classifier: high-impact recall failures",
        "reference_label": "LLM-silver BIB region (not human ground truth)",
        "classifier": "frozen B1 / D1 / no-header, followed by H0",
        "validation_report_sha256": _sha256(validation_root / "validation_report.json"),
        "input_sha256": _sha256(input_path),
        "document_count": len(documents),
        "total_missed_tokens_in_selection": total_missed_tokens,
        "features": features,
        "documents": documents,
    }


HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Bibliography recall failures</title>
<style>
:root{--ink:#252b27;--muted:#667069;--paper:#f7f4ec;--panel:#fffdf8;--rule:#dcd5c7;--green:#d9f3df;--green-edge:#278552;--red:#fde1df;--red-edge:#c43e37;--blue:#deedf9;--blue-edge:#2876a5;--shadow:0 9px 28px #26392d17}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.45 system-ui,-apple-system,sans-serif}.top{position:sticky;top:0;z-index:7;background:#fffdf8f4;border-bottom:1px solid var(--rule);backdrop-filter:blur(10px)}.topin{max-width:1550px;margin:auto;padding:11px 17px;display:flex;gap:16px;align-items:center}.top h1{margin:0;font:700 21px/1.1 Georgia,serif}.sub{color:var(--muted);font-size:11px}.layout{max-width:1550px;margin:16px auto;display:grid;grid-template-columns:285px minmax(0,1fr) 250px;gap:14px;padding:0 14px}.panel{background:var(--panel);border:1px solid var(--rule);border-radius:13px;box-shadow:var(--shadow)}.side{position:sticky;top:79px;max-height:calc(100vh - 96px);overflow:auto;padding:14px}.side h2{font:700 17px Georgia,serif;margin:0 0 8px}label{display:block;color:var(--muted);font-size:11px;margin:9px 0 4px}select,button{border:1px solid var(--rule);border-radius:8px;background:white;color:var(--ink);padding:8px}select{width:100%}button{cursor:pointer}button:hover{background:#f0ece3}.buttons{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:9px}.stats{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin:12px 0}.stat{background:#f2eee5;border-radius:8px;padding:7px}.stat b{font-size:16px;display:block}.legend{display:grid;gap:6px;margin:11px 0}.legend div{padding:6px 8px;border-radius:7px;font-size:11px}.agree{background:var(--green);border-left:5px solid var(--green-edge)}.missed{background:var(--red);border-left:5px solid var(--red-edge)}.extra{background:var(--blue);border-left:5px solid var(--blue-edge)}.neither{background:#f5f2eb;border-left:5px solid #c9c3b7}.reader{overflow:hidden}.dochead{position:sticky;top:68px;z-index:5;padding:10px 13px;background:#fffdf8f2;border-bottom:1px solid var(--rule);display:flex;gap:10px;align-items:center;flex-wrap:wrap}.lines{padding:9px}.line{display:grid;grid-template-columns:58px 62px minmax(0,1fr);gap:9px;padding:5px 8px;margin:2px 0;border-radius:7px;scroll-margin-top:125px;content-visibility:auto;contain-intrinsic-size:34px}.line.agreement_bib{background:var(--green);box-shadow:inset 5px 0 var(--green-edge)}.line.silver_only{background:var(--red);box-shadow:inset 5px 0 var(--red-edge)}.line.classifier_only{background:var(--blue);box-shadow:inset 5px 0 var(--blue-edge)}.line.agreement_non_bib{border-left:5px solid transparent}.ln,.prob{font:10px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--muted);text-align:right}.prob b{display:block;color:var(--ink);font-size:11px}.text{font:15px/1.47 Georgia,"Times New Roman",serif;overflow-wrap:anywhere}.charhit{border-radius:3px;padding:1px 0;box-shadow:inset 0 -2px #0005;box-decoration-break:clone;-webkit-box-decoration-break:clone}.tags{display:block;font:9px/1.4 system-ui;margin-top:2px;color:var(--muted)}.feature-list{display:grid;gap:4px}.feature{display:grid;grid-template-columns:10px 1fr auto;gap:7px;padding:4px;border-radius:6px;font-size:11px}.feature:hover{background:#f1ede4}.dot{width:9px;height:9px;border-radius:3px;margin-top:3px}.count{font:9px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--muted)}@media(max-width:1050px){.layout{grid-template-columns:240px 1fr}.featureside{display:none}}@media(max-width:720px){.top,.side,.dochead{position:relative;top:auto}.layout{display:block;padding:0 6px}.side{max-height:none;margin-bottom:9px}.line{grid-template-columns:38px 48px 1fr}}
</style></head><body><header class="top"><div class="topin"><div><h1 id="title"></h1><div id="subtitle" class="sub"></div></div></div></header><main class="layout"><aside class="panel side"><h2>Document</h2><label>Recall-failure document</label><select id="docSelect"></select><label>Lines shown</label><select id="view"><option value="all">All document lines</option><option value="decisions">BIB decisions ± 2 lines</option><option value="errors">Disagreements ± 2 lines</option></select><div id="stats" class="stats"></div><div class="legend"><div class="agree">Green: classifier and silver agree on BIB</div><div class="missed">Red: silver BIB missed by classifier</div><div class="extra">Blue: classifier-only BIB</div><div class="neither">Neutral: both non-BIB</div></div><div class="buttons"><button id="prevError">Previous error</button><button id="nextError">Next error</button></div><div class="buttons"><button id="firstSilver">First silver block</button><button id="firstPred">First prediction</button></div><p class="sub">“Silver” is the GPT-generated reference annotation, not human ground truth. Feature boxes are shown only on lines the final classifier decided were BIB.</p></aside><section class="panel reader"><div id="docHead" class="dochead"></div><div id="lines" class="lines"></div></section><aside class="panel side featureside"><h2>Prediction features</h2><p class="sub">Hover a label to isolate its exact character boxes on classifier-positive lines.</p><div id="features" class="feature-list"></div></aside></main>
<script>
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let PACKET,docs,featureByKey,current=0,errorCursor=-1;
function doc(){return docs[current]}function shownLines(){const d=doc(),mode=document.getElementById('view').value;if(mode==='all')return d.lines;const keep=new Set;d.lines.forEach((line,i)=>{const hit=mode==='errors'?(line.status==='silver_only'||line.status==='classifier_only'):(line.silver_bib||line.predicted_bib);if(hit)for(let j=Math.max(0,i-2);j<=Math.min(d.lines.length-1,i+2);j++)keep.add(j)});return d.lines.filter(x=>keep.has(x.local_index))}
function spans(line,only=null){const out=[];for(const [key,ranges]of Object.entries(line.matches))if(!only||key===only)for(const [a,b]of ranges)out.push({key,a,b});return out}function highlight(line,only=null){const hits=spans(line,only);if(!hits.length)return esc(line.text);const chars=Array.from(line.text),bounds=[0,chars.length];for(const hit of hits)bounds.push(hit.a,hit.b);const points=[...new Set(bounds)].sort((a,b)=>a-b);let out='';for(let i=0;i<points.length-1;i++){const a=points[i],b=points[i+1],text=esc(chars.slice(a,b).join('')),owners=hits.filter(h=>h.a<=a&&h.b>=b);if(!owners.length){out+=text;continue}const colors=[...new Set(owners.map(h=>featureByKey[h.key].color+'77'))],background=colors.length===1?colors[0]:`linear-gradient(${colors.join(',')})`;out+=`<span class="charhit" style="background:${background}" title="${esc(owners.map(h=>featureByKey[h.key].label).join(' · '))}">${text}</span>`}return out}
function lineHtml(line){const tags=[line.silver_bib?'silver BIB':'silver non-BIB',line.predicted_bib?'classifier BIB':'classifier non-BIB'];return`<div id="line-${line.local_index}" class="line ${line.status}"><span class="ln">L${line.abs_idx}</span><span class="prob"><b>${line.entry_probability.toFixed(3)}</b>${line.char_length}ch</span><span class="text" data-line="${line.local_index}">${highlight(line)}<small class="tags">${tags.join(' · ')}</small></span></div>`}
function renderFeatures(){const d=doc(),counts={};for(const line of d.lines)for(const [key,n]of Object.entries(line.features))counts[key]=(counts[key]||0)+n;document.getElementById('features').innerHTML=PACKET.features.filter(f=>counts[f.key]).map(f=>`<div class="feature" data-feature="${esc(f.key)}"><span class="dot" style="background:${f.color}"></span><span>${esc(f.label)}</span><span class="count">${counts[f.key]}</span></div>`).join('');for(const node of document.querySelectorAll('[data-feature]')){node.onmouseenter=()=>{const key=node.dataset.feature;for(const text of document.querySelectorAll('[data-line]'))text.innerHTML=highlight(d.lines[Number(text.dataset.line)],key)};node.onmouseleave=()=>{for(const text of document.querySelectorAll('[data-line]')){const line=d.lines[Number(text.dataset.line)];text.innerHTML=highlight(line)+`<small class="tags">${line.silver_bib?'silver BIB':'silver non-BIB'} · ${line.predicted_bib?'classifier BIB':'classifier non-BIB'}</small>`}}}}
function render(){const d=doc();document.getElementById('docSelect').value=String(current);document.getElementById('stats').innerHTML=`<div class="stat"><b>${(100*d.line_recall).toFixed(1)}%</b>line recall</div><div class="stat"><b>${(100*d.token_recall).toFixed(1)}%</b>token recall</div><div class="stat"><b>${d.missed_lines.toLocaleString()}</b>missed lines</div><div class="stat"><b>${d.missed_tokens.toLocaleString()}</b>missed tokens</div>`;document.getElementById('docHead').innerHTML=`<b>${esc(d.source)}</b><span>${esc(d.document_id.slice(0,12))}</span><span class="sub">${d.line_count.toLocaleString()} lines · silver blocks ${d.silver_blocks.length} · predicted blocks ${d.predicted_blocks.length} · ${esc(d.coverage.replaceAll('_',' '))}</span>`;document.getElementById('lines').innerHTML=shownLines().map(lineHtml).join('');errorCursor=-1;renderFeatures()}
function jump(index){const line=doc().lines[index];if(!line)return;if(document.getElementById('view').value!=='all'&&!document.getElementById(`line-${index}`)){document.getElementById('view').value='all';render()}requestAnimationFrame(()=>document.getElementById(`line-${index}`)?.scrollIntoView({block:'center'}))}function errors(){return doc().lines.filter(l=>l.status==='silver_only'||l.status==='classifier_only').map(l=>l.local_index)}function moveError(step){const list=errors();if(!list.length)return;errorCursor=(errorCursor+step+list.length)%list.length;jump(list[errorCursor])}
fetch('packet.json').then(r=>{if(!r.ok)throw Error(r.status);return r.json()}).then(packet=>{PACKET=packet;docs=packet.documents;featureByKey=Object.fromEntries(packet.features.map(f=>[f.key,f]));document.getElementById('title').textContent=packet.title;document.getElementById('subtitle').textContent=`${packet.document_count} high-impact documents · ${packet.classifier} · ${packet.reference_label}`;document.getElementById('docSelect').innerHTML=docs.map((d,i)=>`<option value="${i}">${i+1}. ${esc(d.source)} · missed ${d.missed_tokens.toLocaleString()} tok · ${esc(d.document_id.slice(0,10))}</option>`).join('');document.getElementById('docSelect').onchange=e=>{current=Number(e.target.value);render();const first=doc().lines.findIndex(l=>l.status==='silver_only');if(first>=0)jump(first)};document.getElementById('view').onchange=render;document.getElementById('prevError').onclick=()=>moveError(-1);document.getElementById('nextError').onclick=()=>moveError(1);document.getElementById('firstSilver').onclick=()=>jump(doc().lines.findIndex(l=>l.silver_bib));document.getElementById('firstPred').onclick=()=>jump(doc().lines.findIndex(l=>l.predicted_bib));render();const first=doc().lines.findIndex(l=>l.status==='silver_only');if(first>=0)jump(first)}).catch(error=>document.body.innerHTML=`<pre>${esc(error.stack||error)}</pre>`);
</script></body></html>'''


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    packet = build_packet(args)
    _write_json(output_dir / "packet.json", packet)
    with (output_dir / "index.html").open("x", encoding="utf-8") as handle:
        handle.write(HTML)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_failure_presentation_built",
        "document_count": packet["document_count"],
        "total_missed_tokens_in_selection": packet["total_missed_tokens_in_selection"],
        "validation_report_sha256": packet["validation_report_sha256"],
        "code_commit": str(args.code_commit),
        "slurm_job_id": str(args.slurm_job_id),
        "production_eligible": False,
    }
    _write_json(output_dir / "report.json", report)
    outputs = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    receipt = {**report, "outputs": outputs}
    _write_json(output_dir / "receipt.json", receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--validation-dir", required=True)
    parser.add_argument("--document-id", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
