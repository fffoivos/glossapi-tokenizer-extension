#!/usr/bin/env python3
"""Build a full-document reader for the worst sealed-test bibliography errors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from .bibliography_feature_explorer import FEATURE_SPECS
    from .bibliography_v2 import extract_bibliography_feature_review
except ImportError:  # Standalone Clariden runner with the frozen bundle on PYTHONPATH.
    from sequence_models.bibliography_feature_explorer import FEATURE_SPECS
    from sequence_models.bibliography_v2 import extract_bibliography_feature_review


SCHEMA_VERSION = "bibliography-nextgen-worst-documents-review-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected object")
            yield row


def _write_json(path: Path, value: Any) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")


def line_status(*, predicted: bool, truth: bool, trusted: bool) -> str:
    if not trusted:
        return "untrusted"
    if predicted and truth:
        return "true_positive"
    if predicted:
        return "false_positive"
    if truth:
        return "false_negative"
    return "true_negative"


def _runs(mask: np.ndarray, lines: Sequence[Mapping[str, Any]]) -> list[dict[str, int]]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    runs: list[dict[str, int]] = []
    start = previous = int(indices[0])
    for value in indices[1:]:
        current = int(value)
        if current != previous + 1:
            runs.append({
                "local_start": start,
                "local_end": previous,
                "abs_start": int(lines[start]["abs_idx"]),
                "abs_end": int(lines[previous]["abs_idx"]),
                "line_count": previous - start + 1,
            })
            start = current
        previous = current
    runs.append({
        "local_start": start,
        "local_end": previous,
        "abs_start": int(lines[start]["abs_idx"]),
        "abs_end": int(lines[previous]["abs_idx"]),
        "line_count": previous - start + 1,
    })
    return runs


def _metrics(prediction: np.ndarray, truth: np.ndarray, trusted: np.ndarray) -> dict[str, Any]:
    tp = int(np.count_nonzero(prediction & truth & trusted))
    fp = int(np.count_nonzero(prediction & ~truth & trusted))
    fn = int(np.count_nonzero(~prediction & truth & trusted))
    tn = int(np.count_nonzero(~prediction & ~truth & trusted))
    return {
        "trusted_lines": tp + fp + fn + tn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "errors": fp + fn,
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "accuracy": (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn else 1.0,
    }


def _review_matches(text: str) -> tuple[str, dict[str, int], dict[str, list[list[int]]]]:
    review = extract_bibliography_feature_review(text)
    values = review.features.as_dict()
    features = {
        spec.key: int(values[spec.key])
        for spec in FEATURE_SPECS
        if int(values[spec.key]) > 0
    }
    matches: dict[str, list[list[int]]] = defaultdict(list)
    for match in review.matches:
        matches[match.feature].append([int(match.start), int(match.end)])
    return review.normalized_text, features, dict(matches)


def build_packet(args: argparse.Namespace) -> dict[str, Any]:
    documents_path = Path(args.documents).resolve()
    line_key_path = Path(args.line_key).resolve()
    labels_path = Path(args.labels).resolve()
    feature_root = Path(args.feature_dir).resolve()
    prediction_path = Path(args.prediction).resolve()
    probability_path = Path(args.probability).resolve()
    analysis_path = Path(args.analysis).resolve()

    source_documents = list(_iter_jsonl(documents_path))
    feature_documents = list(_iter_jsonl(feature_root / "documents.jsonl"))
    feature_lines = list(_iter_jsonl(feature_root / "line_ids.jsonl"))
    line_keys = list(_iter_jsonl(line_key_path))
    labels = list(_iter_jsonl(labels_path))
    prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False).astype(bool)
    probability = np.load(probability_path, mmap_mode="r", allow_pickle=False)
    if not (
        len(source_documents) == len(feature_documents)
        and len(feature_lines) == len(line_keys) == len(labels) == len(prediction) == len(probability)
    ):
        raise ValueError("sealed artifacts do not align")

    decisions: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for key, label in zip(line_keys, labels, strict=True):
        identity = (str(key["document_id"]), str(key["line_id"]), int(key["abs_idx"]))
        if identity != (str(label["document_id"]), str(label["line_id"]), int(label["abs_idx"])):
            raise ValueError("sealed line-key/label alignment failure")
        decisions[identity] = label["tasks"]["bibliography_membership"]
    truth = np.zeros(len(feature_lines), dtype=bool)
    trusted = np.zeros(len(feature_lines), dtype=bool)
    for index, row in enumerate(feature_lines):
        identity = (str(row["document_id"]), str(row["line_id"]), int(row["abs_idx"]))
        decision = decisions.pop(identity, None)
        if decision is None:
            raise ValueError(f"feature line {index} lacks sealed label")
        truth[index] = decision["label"] == "BIB"
        trusted[index] = bool(decision["trusted"])
    if decisions:
        raise ValueError("sealed labels contain lines absent from features")

    ranked: dict[str, list[tuple[int, str, int, int, dict[str, Any]]]] = defaultdict(list)
    for index, (source_document, metadata) in enumerate(zip(source_documents, feature_documents, strict=True)):
        if source_document["document_id"] != metadata["document_id"]:
            raise ValueError(f"document alignment failure at {index}")
        start, end = int(metadata["line_start"]), int(metadata["line_end"])
        stats = _metrics(prediction[start:end], truth[start:end], trusted[start:end])
        ranked[str(metadata["source"])].append(
            (-stats["errors"], str(metadata["document_id"]), start, end, stats)
        )
    selected = {
        document_id: (source, rank, start, end, stats)
        for source, rows in ranked.items()
        for rank, (_, document_id, start, end, stats) in enumerate(sorted(rows)[: args.per_source], 1)
    }
    all_document_stats = [row[4] for rows in ranked.values() for row in rows]
    source_quality = {
        source: {
            "document_count": len(rows),
            "precision_recall_at_least_0_95": sum(
                row[4]["precision"] >= 0.95 and row[4]["recall"] >= 0.95
                for row in rows
            ),
            "precision_recall_at_least_0_90": sum(
                row[4]["precision"] >= 0.90 and row[4]["recall"] >= 0.90
                for row in rows
            ),
            "zero_bib_document_count": sum(row[4]["tp"] + row[4]["fn"] == 0 for row in rows),
            "clean_zero_bib_document_count": sum(
                row[4]["tp"] + row[4]["fn"] == 0 and row[4]["fp"] == 0
                for row in rows
            ),
            "very_good_including_clean_zero_bib": sum(
                (
                    row[4]["precision"] >= 0.95
                    and row[4]["recall"] >= 0.95
                    and row[4]["tp"] + row[4]["fn"] > 0
                )
                or (row[4]["tp"] + row[4]["fn"] == 0 and row[4]["fp"] == 0)
                for row in rows
            ),
            "zero_error": sum(row[4]["errors"] == 0 for row in rows),
        }
        for source, rows in sorted(ranked.items())
    }

    analysis_payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    analyses = analysis_payload.get("documents", {})
    documents = []
    for source_document, metadata in zip(source_documents, feature_documents, strict=True):
        document_id = str(metadata["document_id"])
        selected_row = selected.get(document_id)
        if selected_row is None:
            continue
        source, rank, start, end, stats = selected_row
        raw_lines = source_document.get("lines")
        if not isinstance(raw_lines, list) or len(raw_lines) != end - start:
            raise ValueError(f"{document_id}: source/feature line mismatch")
        local_pred = np.asarray(prediction[start:end], dtype=bool)
        local_truth = np.asarray(truth[start:end], dtype=bool)
        local_trusted = np.asarray(trusted[start:end], dtype=bool)
        rows = []
        for offset, raw in enumerate(raw_lines):
            text = unicodedata.normalize("NFKC", str(raw.get("text", "")))
            features: dict[str, int] = {}
            matches: dict[str, list[list[int]]] = {}
            if local_pred[offset]:
                text, features, matches = _review_matches(text)
            rows.append({
                "local_index": offset,
                "abs_idx": int(raw["abs_idx"]),
                "line_id": str(raw["line_id"]),
                "text": text,
                "probability": round(float(probability[start + offset]), 6),
                "predicted_bib": bool(local_pred[offset]),
                "silver_bib": bool(local_truth[offset]),
                "trusted": bool(local_trusted[offset]),
                "status": line_status(
                    predicted=bool(local_pred[offset]),
                    truth=bool(local_truth[offset]),
                    trusted=bool(local_trusted[offset]),
                ),
                "features": features,
                "matches": matches,
            })
        analysis = analyses.get(document_id, {})
        documents.append({
            "source_rank": rank,
            "document_id": document_id,
            "work_id": str(metadata["work_id"]),
            "source": source,
            "source_dataset": source_document.get("source_dataset"),
            "source_doc_id": source_document.get("source_doc_id"),
            "source_row_id": source_document.get("source_row_id"),
            "n_physical_lines": int(source_document.get("n_physical_lines", len(rows))),
            **stats,
            "error_runs": _runs((local_pred != local_truth) & local_trusted, raw_lines),
            "false_positive_runs": _runs(local_pred & ~local_truth & local_trusted, raw_lines),
            "false_negative_runs": _runs(~local_pred & local_truth & local_trusted, raw_lines),
            "analysis": {
                "summary": str(analysis.get("summary", "Review pending.")),
                "false_positives": list(analysis.get("false_positives", [])),
                "false_negatives": list(analysis.get("false_negatives", [])),
                "interpretation": str(analysis.get("interpretation", "")),
                "tags": list(analysis.get("tags", [])),
            },
            "lines": rows,
        })
    documents.sort(key=lambda row: (row["source"], row["source_rank"]))
    features = [
        {"key": spec.key, "label": spec.label, "group": spec.group, "color": spec.color}
        for spec in FEATURE_SPECS
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "title": "Worst sealed-test bibliography documents",
        "subtitle": "position_hist_component_scope · 10 highest-error documents per source",
        "reference_label": "dual-annotator consensus silver, not human ground truth",
        "selection": "ranked within source by FP + FN trusted lines; ties by document ID",
        "documents_sha256": _sha256(documents_path),
        "labels_sha256": _sha256(labels_path),
        "prediction_sha256": _sha256(prediction_path),
        "analysis_sha256": _sha256(analysis_path),
        "document_count": len(documents),
        "whole_test_quality": {
            "document_count": len(all_document_stats),
            "precision_recall_at_least_0_95": sum(
                row["precision"] >= 0.95 and row["recall"] >= 0.95
                for row in all_document_stats
            ),
            "precision_recall_at_least_0_90": sum(
                row["precision"] >= 0.90 and row["recall"] >= 0.90
                for row in all_document_stats
            ),
            "zero_bib_document_count": sum(
                row["tp"] + row["fn"] == 0 for row in all_document_stats
            ),
            "clean_zero_bib_document_count": sum(
                row["tp"] + row["fn"] == 0 and row["fp"] == 0
                for row in all_document_stats
            ),
            "very_good_including_clean_zero_bib": sum(
                (
                    row["precision"] >= 0.95
                    and row["recall"] >= 0.95
                    and row["tp"] + row["fn"] > 0
                )
                or (row["tp"] + row["fn"] == 0 and row["fp"] == 0)
                for row in all_document_stats
            ),
            "zero_error": sum(row["errors"] == 0 for row in all_document_stats),
            "by_source": source_quality,
        },
        "features": features,
        "documents": documents,
    }


HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Worst sealed-test bibliography documents</title>
<style>
:root{--ink:#252b27;--muted:#667069;--paper:#f7f4ec;--panel:#fffdf8;--rule:#dcd5c7;--tp:#d9f3df;--tp-edge:#278552;--fp:#fde1df;--fp-edge:#c43e37;--fn:#fff0bd;--fn-edge:#b67b16;--unknown:#e7e4dd;--shadow:0 9px 28px #26392d17}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.45 system-ui,-apple-system,sans-serif}.top{position:sticky;top:0;z-index:9;background:#fffdf8f4;border-bottom:1px solid var(--rule);backdrop-filter:blur(10px)}.topin{max-width:1680px;margin:auto;padding:11px 17px;display:flex;gap:16px;align-items:center}.top h1{margin:0;font:700 21px/1.1 Georgia,serif}.sub{color:var(--muted);font-size:11px}.layout{max-width:1680px;margin:14px auto;display:grid;grid-template-columns:300px minmax(0,1fr) 270px;gap:12px;padding:0 12px}.panel{background:var(--panel);border:1px solid var(--rule);border-radius:13px;box-shadow:var(--shadow)}.side{position:sticky;top:71px;max-height:calc(100vh - 86px);overflow:auto;padding:12px}.side h2,.analysis h2{font:700 17px Georgia,serif;margin:0 0 8px}.source-title{margin:12px 0 5px;font-weight:800;font-size:11px;text-transform:uppercase;color:var(--muted);letter-spacing:.06em}.doc{display:block;width:100%;text-align:left;border:0;border-radius:8px;background:#eeeae1;color:var(--ink);padding:7px 8px;margin:3px 0;cursor:pointer}.doc.active{background:#293b31;color:white}.doc small{display:flex;justify-content:space-between;gap:5px;opacity:.72}.controls label{display:block;color:var(--muted);font-size:11px;margin:9px 0 4px}select,button{border:1px solid var(--rule);border-radius:8px;background:white;color:var(--ink);padding:7px}select{width:100%}button{cursor:pointer}.buttons{display:grid;grid-template-columns:1fr 1fr;gap:5px;margin-top:7px}.legend{display:grid;gap:5px;margin:10px 0}.legend div{padding:5px 7px;border-radius:6px;font-size:11px}.tp-key{background:var(--tp);border-left:5px solid var(--tp-edge)}.fp-key{background:var(--fp);border-left:5px solid var(--fp-edge)}.fn-key{background:var(--fn);border-left:5px solid var(--fn-edge)}.reader{overflow:hidden}.dochead{position:sticky;top:66px;z-index:7;padding:9px 12px;background:#fffdf8f5;border-bottom:1px solid var(--rule)}.identity{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.metrics{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}.metric{background:#eeeae1;border-radius:6px;padding:3px 7px;font-size:11px}.analysis{margin-top:9px;padding:9px 11px;background:#f4f0e7;border-radius:8px}.analysis p{margin:4px 0}.analysis ul{margin:5px 0;padding-left:19px}.tag{display:inline-block;border-radius:999px;background:#e1ddd2;padding:2px 7px;margin:2px;font-size:10px}.lines{padding:8px}.line{display:grid;grid-template-columns:58px 68px 68px 58px minmax(0,1fr);gap:7px;align-items:start;padding:5px 7px;margin:2px 0;border:2px solid transparent;border-radius:7px;scroll-margin-top:220px;content-visibility:auto;contain-intrinsic-size:38px}.line.true_positive{background:var(--tp);border-color:var(--tp-edge)}.line.false_positive{background:var(--fp);border-color:var(--fp-edge)}.line.false_negative{background:var(--fn);border:2px dashed var(--fn-edge)}.line.untrusted{background:var(--unknown);opacity:.72}.ln{font:10px/2.2 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--muted);text-align:right}.decision{border:1px solid var(--rule);border-radius:6px;background:#f2eee5;color:var(--muted);padding:3px;text-align:center;font:800 9px/1.25 system-ui}.decision.model.on{background:#456e88;border-color:#456e88;color:white}.decision.silver.on{background:#a64640;border-color:#a64640;color:white}.prob{font:10px/2.2 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--muted);text-align:right}.text{font:15px/1.47 Georgia,"Times New Roman",serif;overflow-wrap:anywhere}.charhit{border-radius:3px;padding:1px 0;box-shadow:inset 0 -2px #0005;box-decoration-break:clone;-webkit-box-decoration-break:clone}.feature-list{display:grid;gap:3px}.feature{display:grid;grid-template-columns:10px 1fr;gap:7px;padding:4px;border-radius:6px;font-size:11px}.feature:hover{background:#f1ede4}.dot{width:9px;height:9px;border-radius:3px;margin-top:3px}@media(max-width:1120px){.layout{grid-template-columns:270px 1fr}.featureside{display:none}}@media(max-width:760px){.top,.side,.dochead{position:relative;top:auto}.layout{display:block;padding:0 6px}.side{max-height:none;margin-bottom:8px}.line{grid-template-columns:42px 52px 52px 42px 1fr}}
</style></head><body><header class="top"><div class="topin"><div><h1 id="title"></h1><div id="subtitle" class="sub"></div></div></div></header><main class="layout"><aside class="panel side"><h2>Documents</h2><div id="docs"></div><section class="controls"><label>Lines shown</label><select id="view"><option value="errors">Errors ± 3 lines</option><option value="decisions">All BIB decisions ± 3 lines</option><option value="all">Full document</option></select><label>Error type</label><select id="errorType"><option value="all">All errors</option><option value="false_positive">False positives</option><option value="false_negative">False negatives</option></select><div class="buttons"><button id="prevError">Previous error</button><button id="nextError">Next error</button></div><div class="buttons"><button id="prevRun">Previous run</button><button id="nextRun">Next run</button></div></section><div class="legend"><div class="tp-key">Green: model and silver agree on BIB</div><div class="fp-key">Red: model-only BIB (FP)</div><div class="fn-key">Gold: silver-only BIB (FN)</div></div><p class="sub">Silver labels are consensus annotations, not human ground truth. Exact deterministic feature boxes appear on model-positive lines.</p></aside><section class="panel reader"><div id="docHead" class="dochead"></div><div id="lines" class="lines"></div></section><aside class="panel side featureside"><h2>Detected features</h2><p class="sub">Hover a label to isolate its exact character boxes.</p><div id="features" class="feature-list"></div></aside></main>
<script>
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let PACKET,docs,featureByKey,current=0,errorCursor=-1,runCursor=-1;function doc(){return docs[current]}function desiredStatus(line){const t=document.getElementById('errorType').value;return t==='all'?(line.status==='false_positive'||line.status==='false_negative'):line.status===t}function shownLines(){const d=doc(),mode=document.getElementById('view').value;if(mode==='all')return d.lines;const keep=new Set;d.lines.forEach((line,i)=>{const hit=mode==='errors'?desiredStatus(line):(line.silver_bib||line.predicted_bib);if(hit)for(let j=Math.max(0,i-3);j<=Math.min(d.lines.length-1,i+3);j++)keep.add(j)});return d.lines.filter(x=>keep.has(x.local_index))}function spans(line,only=null){const out=[];for(const[key,ranges]of Object.entries(line.matches))if(!only||key===only)for(const[a,b]of ranges)out.push({key,a,b});return out}function highlight(line,only=null){const hits=spans(line,only);if(!hits.length)return esc(line.text);const chars=Array.from(line.text),bounds=[0,chars.length,...hits.flatMap(x=>[x.a,x.b])],points=[...new Set(bounds)].sort((a,b)=>a-b);let out='';for(let i=0;i<points.length-1;i++){const a=points[i],b=points[i+1],owners=hits.filter(h=>h.a<=a&&h.b>=b),text=esc(chars.slice(a,b).join(''));if(!owners.length){out+=text;continue}const colors=[...new Set(owners.map(h=>featureByKey[h.key].color+'77'))],background=colors.length===1?colors[0]:`linear-gradient(${colors.join(',')})`;out+=`<span class="charhit" style="background:${background}">${text}</span>`}return out}function lineHtml(line){return`<div id="line-${line.local_index}" class="line ${line.status}"><span class="ln">L${line.abs_idx}</span><span class="decision model ${line.predicted_bib?'on':''}">MODEL<br>${line.predicted_bib?'BIB':'—'}</span><span class="decision silver ${line.silver_bib?'on':''}">SILVER<br>${line.silver_bib?'BIB':'—'}</span><span class="prob">p ${line.probability.toFixed(3)}</span><span class="text" data-line="${line.local_index}">${highlight(line)}</span></div>`}function list(title,values){return values?.length?`<b>${title}</b><ul>${values.map(x=>`<li>${esc(x)}</li>`).join('')}</ul>`:''}function docsHtml(){let source='';return docs.map((d,i)=>{const heading=d.source!==source?(source=d.source,`<div class="source-title">${esc(source.replace('_',' '))}</div>`):'';return heading+`<button class="doc ${i===current?'active':''}" data-doc="${i}"><b>#${d.source_rank} · ${esc(d.document_id.slice(0,12))}</b><small><span>${d.errors} errors</span><span>${d.fp} FP · ${d.fn} FN</span></small></button>`}).join('')}function renderFeatures(){const d=doc(),active=new Set;for(const line of d.lines)for(const key of Object.keys(line.features))active.add(key);document.getElementById('features').innerHTML=PACKET.features.filter(f=>active.has(f.key)).map(f=>`<div class="feature" data-feature="${esc(f.key)}"><span class="dot" style="background:${f.color}"></span><span>${esc(f.label)}</span></div>`).join('');for(const node of document.querySelectorAll('[data-feature]')){node.onmouseenter=()=>{const key=node.dataset.feature;for(const text of document.querySelectorAll('[data-line]'))text.innerHTML=highlight(d.lines[Number(text.dataset.line)],key)};node.onmouseleave=()=>{for(const text of document.querySelectorAll('[data-line]'))text.innerHTML=highlight(d.lines[Number(text.dataset.line)])}}}function render(){const d=doc(),a=d.analysis;document.getElementById('docs').innerHTML=docsHtml();for(const b of document.querySelectorAll('[data-doc]'))b.onclick=()=>{current=Number(b.dataset.doc);render();jump(errors()[0])};document.getElementById('docHead').innerHTML=`<div class="identity"><b>${esc(d.source.replace('_',' '))} #${d.source_rank}</b><code>${esc(d.document_id)}</code><span class="sub">${esc(String(d.source_doc_id??d.source_row_id??''))}</span></div><div class="metrics"><span class="metric"><b>${d.errors}</b> errors</span><span class="metric"><b>${d.fp}</b> FP</span><span class="metric"><b>${d.fn}</b> FN</span><span class="metric"><b>${(100*d.precision).toFixed(2)}%</b> precision</span><span class="metric"><b>${(100*d.recall).toFixed(2)}%</b> recall</span><span class="metric"><b>${(100*d.accuracy).toFixed(2)}%</b> accuracy</span></div><div class="analysis"><h2>${esc(a.summary)}</h2>${list('False positives',a.false_positives)}${list('False negatives',a.false_negatives)}${a.interpretation?`<p><b>Interpretation:</b> ${esc(a.interpretation)}</p>`:''}<div>${a.tags.map(t=>`<span class="tag">${esc(t)}</span>`).join('')}</div></div>`;document.getElementById('lines').innerHTML=shownLines().map(lineHtml).join('');errorCursor=runCursor=-1;renderFeatures()}function ensureVisible(index){if(index===undefined||index<0)return;if(!document.getElementById(`line-${index}`)){document.getElementById('view').value='all';render()}requestAnimationFrame(()=>document.getElementById(`line-${index}`)?.scrollIntoView({block:'center'}))}function errors(){return doc().lines.filter(desiredStatus).map(l=>l.local_index)}function moveError(step){const list=errors();if(!list.length)return;errorCursor=(errorCursor+step+list.length)%list.length;ensureVisible(list[errorCursor])}function runs(){const type=document.getElementById('errorType').value;return type==='false_positive'?doc().false_positive_runs:type==='false_negative'?doc().false_negative_runs:doc().error_runs}function moveRun(step){const list=runs();if(!list.length)return;runCursor=(runCursor+step+list.length)%list.length;ensureVisible(list[runCursor].local_start)}fetch('packet.json').then(r=>{if(!r.ok)throw Error(r.status);return r.json()}).then(packet=>{PACKET=packet;docs=packet.documents;featureByKey=Object.fromEntries(packet.features.map(f=>[f.key,f]));document.getElementById('title').textContent=packet.title;document.getElementById('subtitle').textContent=packet.subtitle+' · '+packet.reference_label;document.getElementById('view').onchange=render;document.getElementById('errorType').onchange=render;document.getElementById('prevError').onclick=()=>moveError(-1);document.getElementById('nextError').onclick=()=>moveError(1);document.getElementById('prevRun').onclick=()=>moveRun(-1);document.getElementById('nextRun').onclick=()=>moveRun(1);render();ensureVisible(errors()[0])}).catch(error=>document.body.innerHTML=`<pre>${esc(error.stack||error)}</pre>`);
</script></body></html>'''


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    packet = build_packet(args)
    _write_json(output_dir / "packet.json", packet)
    (output_dir / "index.html").write_text(HTML, encoding="utf-8")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "document_count": packet["document_count"],
        "source_counts": {
            source: sum(row["source"] == source for row in packet["documents"])
            for source in sorted({row["source"] for row in packet["documents"]})
        },
        "inputs": {
            "documents_sha256": packet["documents_sha256"],
            "labels_sha256": packet["labels_sha256"],
            "prediction_sha256": packet["prediction_sha256"],
            "analysis_sha256": packet["analysis_sha256"],
        },
    }
    _write_json(output_dir / "receipt.json", receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", required=True)
    parser.add_argument("--line-key", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--probability", required=True)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-source", type=int, default=10)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
