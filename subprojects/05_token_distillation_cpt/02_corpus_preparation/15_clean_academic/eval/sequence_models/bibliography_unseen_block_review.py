#!/usr/bin/env python3
"""Run the frozen bibliography classifier on unseen works and build a reader.

The input is the already-audited source-matched holdout.  A deterministic,
source-balanced candidate pool is chosen without looking at model output.  The
frozen D1 line model, five train-fold signal TCNs, and frozen anchored decoder
are then applied without fitting or tuning.  The final reader is enriched for
documents with predicted bibliography blocks, so it is a qualitative
precision/boundary review rather than a prevalence or recall estimate.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
import pickle
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .bibliography_auxiliary_scope_veto import materialize_auxiliary_headings
from .bibliography_deterministic_roles import ROLE_NAMES, _analyze_document
from .bibliography_entry_blocks import BlockConfig, blocks_from_mask
from .bibliography_entry_dataset import materialize_document
from .bibliography_entry_models import Table
from .bibliography_feature_explorer import FEATURE_SPECS
from .bibliography_signal_block_decode import decode_signal_blocks
from .bibliography_signal_tcn import build_signal_features
from .bibliography_signal_validation import (
    _ensemble_probability,
    select_train_recall_candidate,
)
from .bibliography_v2 import extract_bibliography_feature_review
from .contract import sha256_file


SCHEMA_VERSION = "bibliography-unseen-block-review-v1"
SOURCES = ("greek_phd", "kallipos", "openarchives")


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: expected JSON objects")
    return rows


def _rank(seed: str, row: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        f"{seed}\0{row.get('source')}\0{row.get('work_id')}\0{row.get('document_id')}".encode()
    ).hexdigest()


def select_candidate_pool(
    rows: Sequence[Mapping[str, Any]],
    *,
    per_source: int,
    seed: str,
    excluded_document_ids: Iterable[str] = (),
    excluded_work_ids: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Select an outcome-blind deterministic pool from each canonical source."""

    excluded_documents = set(map(str, excluded_document_ids))
    excluded_works = set(map(str, excluded_work_ids))
    grouped: dict[str, list[Mapping[str, Any]]] = {source: [] for source in SOURCES}
    seen: set[str] = set()
    for row in rows:
        document_id = str(row.get("document_id", ""))
        source = str(row.get("source", ""))
        if not document_id or document_id in seen:
            raise ValueError("holdout has an empty or duplicate document ID")
        seen.add(document_id)
        if document_id in excluded_documents or str(row.get("work_id", "")) in excluded_works:
            continue
        if source in grouped:
            grouped[source].append(row)
    selected: list[dict[str, Any]] = []
    for source in SOURCES:
        ordered = sorted(grouped[source], key=lambda row: (_rank(seed, row), row["document_id"]))
        if len(ordered) < per_source:
            raise ValueError(f"{source}: only {len(ordered)} holdout documents")
        selected.extend(dict(row) for row in ordered[:per_source])
    return selected


def _classifier_row(row: Mapping[str, Any]) -> dict[str, Any]:
    lines = row.get("lines")
    if not isinstance(lines, list) or not lines:
        raise ValueError(f"{row.get('document_id')}: no line inventory")
    return {
        "document_id": str(row["document_id"]),
        "work_id": str(row["work_id"]),
        "source": str(row["source"]),
        "split": "validation",
        "coverage": "full_document",
        "n_physical_lines": int(row["n_physical_lines"]),
        "lines": [
            {
                "abs_idx": int(line["abs_idx"]),
                "text": str(line["text"]),
                "label": "O",
            }
            for line in lines
        ],
    }


def materialize_inference_table(
    rows: Sequence[Mapping[str, Any]], *, workers: int
) -> tuple[Table, list[dict[str, Any]]]:
    source_rows = [_classifier_row(row) for row in rows]
    if workers == 1:
        materialized = [materialize_document(row) for row in source_rows]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            materialized = list(executor.map(materialize_document, source_rows, chunksize=1))
    documents: list[dict[str, Any]] = []
    cursor = 0
    for result in materialized:
        length = len(result["targets"])
        documents.append(
            {
                "document_id": result["document_id"],
                "work_id": result["work_id"],
                "source": result["source"],
                "split": "validation",
                "coverage": "full_document",
                "n_physical_lines": result["n_physical_lines"],
                "line_start": cursor,
                "line_end": cursor + length,
                "line_count": length,
                "fold": 0,
            }
        )
        cursor += length
    arrays = {
        key: np.concatenate([result[key] for result in materialized])
        for key in (
            "counts",
            "targets",
            "original_labels",
            "header_kinds",
            "abs_indices",
            "token_counts",
            "char_lengths",
            "block_indices",
        )
    }
    arrays["document_indices"] = np.concatenate(
        [np.full(row["line_count"], index, dtype=np.uint32) for index, row in enumerate(documents)]
    )
    arrays["folds"] = np.zeros(cursor, dtype=np.uint8)
    table = Table(
        root=Path("."),
        manifest={"schema_version": "in-memory-unseen-inference-v1", "line_count": cursor},
        documents=tuple(documents),
        **arrays,
    )
    return table, source_rows


def _roles(
    table: Table, source_rows: Sequence[Mapping[str, Any]], *, workers: int
) -> tuple[np.ndarray, dict[str, int]]:
    expected = {
        str(document["document_id"]): (int(document["line_start"]), int(document["line_end"]))
        for document in table.documents
    }
    tasks = [(str(row["document_id"]), list(row["lines"])) for row in source_rows]
    if workers == 1:
        results = map(_analyze_document, tasks)
    else:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
        results = executor.map(_analyze_document, tasks, chunksize=1)
    roles = np.zeros((len(table.targets), len(ROLE_NAMES)), dtype=np.uint8)
    counts = {name: 0 for name in ROLE_NAMES}
    try:
        for document_id, values, local_counts in results:
            start, end = expected[document_id]
            if values.shape != (end - start, len(ROLE_NAMES)):
                raise ValueError(f"{document_id}: role alignment failed")
            roles[start:end] = values
            for name, count in local_counts.items():
                counts[name] += int(count)
    finally:
        if workers != 1:
            executor.shutdown()  # type: ignore[possibly-undefined]
    return roles, counts


def _line_probability(table: Table, model_path: Path) -> np.ndarray:
    with model_path.open("rb") as handle:
        model, transform = pickle.load(handle)
    if getattr(transform, "arm", None) != "D1":
        raise ValueError("frozen line model is not D1")
    probability = model.predict_proba(transform.apply(table.counts))[:, 1].astype(np.float32)
    if len(probability) != len(table.targets) or not np.isfinite(probability).all():
        raise ValueError("line probability is invalid")
    return probability


def _feature_payload(text: str) -> tuple[str, dict[str, int], dict[str, list[list[int]]]]:
    review = extract_bibliography_feature_review(text)
    values = review.features.as_dict()
    features = {
        spec.key: int(values[spec.key])
        for spec in FEATURE_SPECS
        if int(values[spec.key]) > 0
    }
    matches: dict[str, list[list[int]]] = collections.defaultdict(list)
    for match in review.matches:
        matches[match.feature].append([int(match.start), int(match.end)])
    return review.normalized_text, features, dict(matches)


def choose_review_documents(
    rows: Sequence[Mapping[str, Any]],
    table: Table,
    prediction: np.ndarray,
    *,
    per_source: int,
) -> tuple[list[int], dict[str, Any]]:
    """Prefer detected-block documents within the independently chosen pool."""

    chosen: list[int] = []
    summary: dict[str, Any] = {}
    for source in SOURCES:
        indices = [i for i, document in enumerate(table.documents) if document["source"] == source]
        positive, negative = [], []
        for index in indices:
            document = table.documents[index]
            start, end = int(document["line_start"]), int(document["line_end"])
            target = positive if np.any(prediction[start:end]) else negative
            target.append(index)
        selected = (positive + negative)[:per_source]
        if len(selected) != per_source:
            raise ValueError(f"{source}: could not choose {per_source} review documents")
        chosen.extend(selected)
        summary[source] = {
            "candidate_pool": len(indices),
            "pool_with_predicted_blocks": len(positive),
            "selected": len(selected),
            "selected_with_predicted_blocks": sum(index in positive for index in selected),
        }
    return chosen, summary


def _packet(
    original_rows: Sequence[Mapping[str, Any]],
    table: Table,
    selected_indices: Sequence[int],
    line_probability: np.ndarray,
    signal_probability: np.ndarray,
    prediction: np.ndarray,
    *,
    selection_summary: Mapping[str, Any],
    classifier: Mapping[str, Any],
) -> dict[str, Any]:
    by_id = {str(row["document_id"]): row for row in original_rows}
    documents = []
    for document_index in selected_indices:
        meta = table.documents[document_index]
        raw = by_id[str(meta["document_id"])]
        start, end = int(meta["line_start"]), int(meta["line_end"])
        local_prediction = np.asarray(prediction[start:end], dtype=bool)
        blocks = []
        for block_index, (local_start, local_end) in enumerate(
            blocks_from_mask(local_prediction, table.abs_indices[start:end]), 1
        ):
            signal = signal_probability[start + local_start : start + local_end + 1]
            blocks.append(
                {
                    "block_index": block_index,
                    "local_start": local_start,
                    "local_end": local_end,
                    "abs_start": int(table.abs_indices[start + local_start]),
                    "abs_end": int(table.abs_indices[start + local_end]),
                    "present_line_count": local_end - local_start + 1,
                    "maximum_signal_probability": round(float(signal.max()), 6),
                }
            )
        lines = []
        raw_lines = raw["lines"]
        if len(raw_lines) != end - start:
            raise ValueError(f"{meta['document_id']}: packet line alignment failed")
        for local_index, line in enumerate(raw_lines):
            predicted = bool(local_prediction[local_index])
            if predicted:
                text, features, matches = _feature_payload(str(line["text"]))
            else:
                text = unicodedata.normalize("NFKC", str(line["text"]))
                features, matches = {}, {}
            lines.append(
                {
                    "local_index": local_index,
                    "abs_idx": int(line["abs_idx"]),
                    "text": text,
                    "predicted_bib": predicted,
                    "entry_probability": round(float(line_probability[start + local_index]), 6),
                    "signal_probability": round(float(signal_probability[start + local_index]), 6),
                    "features": features,
                    "matches": matches,
                }
            )
        documents.append(
            {
                "document_id": str(meta["document_id"]),
                "source": str(meta["source"]),
                "work_id": str(raw.get("work_id", "")),
                "source_doc_id": str(raw.get("source_doc_id", "")),
                "n_physical_lines": int(raw["n_physical_lines"]),
                "n_present_lines": len(lines),
                "predicted_line_count": int(local_prediction.sum()),
                "blocks": blocks,
                "lines": lines,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "title": "Frozen bibliography classifier · unseen documents",
        "interpretation": (
            "Outcome-enriched qualitative review of predicted blocks; not a recall or prevalence estimate."
        ),
        "source_counts": dict(collections.Counter(row["source"] for row in documents)),
        "selection_summary": selection_summary,
        "classifier": dict(classifier),
        "features": [
            {"key": spec.key, "label": spec.label, "group": spec.group, "color": spec.color}
            for spec in FEATURE_SPECS
        ],
        "documents": documents,
    }


HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Unseen bibliography blocks</title>
<style>
:root{--ink:#252b27;--muted:#667069;--paper:#f7f4ec;--panel:#fffdf8;--rule:#dcd5c7;--bib:#e9f3e7;--bib-edge:#4b7954;--active:#fff0bd;--active-edge:#b47b19;--wrong:#fde1df;--wrong-edge:#b63d37;--missed:#e2ebfb;--missed-edge:#416aa6;--weird:#eee3f8;--weird-edge:#755092;--shadow:0 9px 28px #26392d17}*{box-sizing:border-box}html{scroll-behavior:auto}body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.45 system-ui,-apple-system,sans-serif}.top{position:sticky;top:0;z-index:7;background:#fffdf8f4;border-bottom:1px solid var(--rule);backdrop-filter:blur(10px)}.topin{max-width:1580px;margin:auto;padding:11px 17px;display:flex;gap:16px;align-items:center;justify-content:space-between}.top h1{margin:0;font:700 21px/1.1 Georgia,serif}.sub{color:var(--muted);font-size:11px}.layout{max-width:1580px;margin:16px auto;display:grid;grid-template-columns:300px minmax(0,1fr) 245px;gap:14px;padding:0 14px}.panel{background:var(--panel);border:1px solid var(--rule);border-radius:13px;box-shadow:var(--shadow)}.side{position:sticky;top:79px;max-height:calc(100vh - 96px);overflow:auto;padding:14px}.side h2{font:700 17px Georgia,serif;margin:0 0 8px}label{display:block;color:var(--muted);font-size:11px;margin:9px 0 4px}select,button{border:1px solid var(--rule);border-radius:8px;background:white;color:var(--ink);padding:8px}select{width:100%}button{cursor:pointer}button:hover:not(:disabled){background:#f0ece3}button:disabled{opacity:.45;cursor:default}.docweird,.exportreview{width:100%;margin-top:8px;font-weight:700}.docweird.is-weird{background:var(--weird);border-color:var(--weird-edge);color:#4f2f67}.blocknav{display:grid;gap:7px;margin-top:12px}.blocknav button{width:100%;font-weight:700}.counter{text-align:center;padding:7px;color:var(--muted);font-size:12px}.legend{display:grid;gap:6px;margin:13px 0}.legend div{padding:7px 8px;border-radius:7px;font-size:11px}.predicted{background:var(--bib);border-left:5px solid var(--bib-edge)}.current{background:var(--active);border-left:5px solid var(--active-edge)}.wrong-key{background:var(--wrong);border-left:5px solid var(--wrong-edge)}.missed-key{background:var(--missed);border-left:5px solid var(--missed-edge)}.reader{overflow:hidden}.dochead{position:sticky;top:68px;z-index:5;padding:10px 13px;background:#fffdf8f2;border-bottom:1px solid var(--rule);display:flex;gap:10px;align-items:center;flex-wrap:wrap}.dochead.weird{background:var(--weird);box-shadow:inset 5px 0 var(--weird-edge)}.lines{padding:9px 9px 50vh}.line{display:grid;grid-template-columns:62px 78px minmax(0,1fr);gap:9px;align-items:start;width:100%;padding:5px 8px;margin:2px 0;border:2px solid transparent;border-radius:7px;background:transparent;text-align:left;font:inherit;color:inherit}.line.bib{background:var(--bib);border-color:var(--bib-edge)}.line.bib:hover{background:#dfeedd}.line.active-block{background:var(--active);border-color:var(--active-edge)}.line.wrong{background:var(--wrong);border-color:var(--wrong-edge)}.line.wrong:hover{background:#f8d2cf}.line.wrong.active-block{box-shadow:0 0 0 2px var(--active-edge)}.line.should-bib{background:var(--missed);border-color:var(--missed-edge)}.line.should-bib:hover{background:#d5e2f7}.ln{font:10px/2.2 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--muted);text-align:right}.decision{border:1px solid var(--rule);border-radius:6px;background:#f2eee5;color:var(--muted);padding:4px;text-align:center;font:800 9px/1.25 system-ui}.decision.on{background:var(--bib-edge);border-color:var(--bib-edge);color:white}.decision.wrong{background:var(--wrong-edge);border-color:var(--wrong-edge);color:white}.decision.missed{background:var(--missed-edge);border-color:var(--missed-edge);color:white}.text{font:15px/1.47 Georgia,"Times New Roman",serif;overflow-wrap:anywhere}.charhit{border-radius:3px;padding:1px 0;box-shadow:inset 0 -2px #0005;box-decoration-break:clone;-webkit-box-decoration-break:clone}.feature-list{display:grid;gap:4px}.feature{display:grid;grid-template-columns:10px 1fr;gap:7px;padding:4px;border-radius:6px;font-size:11px}.feature:hover{background:#f1ede4}.dot{width:9px;height:9px;border-radius:3px;margin-top:3px}@media(max-width:1050px){.layout{grid-template-columns:260px 1fr}.featureside{display:none}}@media(max-width:720px){.top,.side,.dochead{position:relative;top:auto}.layout{display:block;padding:0 6px}.side{max-height:none;margin-bottom:9px}.line{grid-template-columns:42px 72px 1fr}}
</style></head><body><header class="top"><div class="topin"><div><h1 id="title"></h1><div id="subtitle" class="sub"></div></div><div class="sub">↑ / ↓ moves between predicted blocks</div></div></header><main class="layout"><aside class="panel side"><h2>Document</h2><label>10 unseen works per source</label><select id="docSelect"></select><button id="toggleDocWeird" class="docweird" aria-pressed="false">Mark document WEIRD</button><div id="weirdCount" class="counter"></div><button id="exportReview" class="exportreview">Export review JSON</button><div class="blocknav"><button id="prevBlock">↑ Previous BIB block</button><div id="blockCounter" class="counter"></div><button id="nextBlock">↓ Next BIB block</button><button id="toggleBlockWrong">Mark whole block WRONG</button></div><div id="wrongCount" class="counter"></div><div class="legend"><div class="predicted">Predicted bibliography block</div><div class="current">Current predicted block</div><div class="wrong-key">Click a predicted line to toggle WRONG</div><div class="missed-key">Click any other line to toggle SHOULD BE BIB</div></div><p class="sub">The 30 works are absent from STRUCT-2K and were drawn from its canonical sources. This view is enriched for classifier-positive documents, so use it to judge precision and boundaries—not recall.</p></aside><section class="panel reader"><div id="docHead" class="dochead"></div><div id="lines" class="lines"></div></section><aside class="panel side featureside"><h2>Detected features</h2><p class="sub">Hover a label to isolate its exact character boxes inside predicted lines.</p><div id="features" class="feature-list"></div></aside></main>
<script>
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const wrongStorageKey='bib-unseen-wrong-lines-v1',shouldBibStorageKey='bib-unseen-should-bib-lines-v1',weirdStorageKey='bib-unseen-weird-docs-v1';let PACKET,docs,featureByKey,current=0,currentBlock=0,navigationEpoch=0,wrong=JSON.parse(localStorage.getItem(wrongStorageKey)||'{}'),shouldBib=JSON.parse(localStorage.getItem(shouldBibStorageKey)||'{}'),weird=JSON.parse(localStorage.getItem(weirdStorageKey)||'{}');
function doc(){return docs[current]}function spans(line,only=null){const out=[];for(const[key,ranges]of Object.entries(line.matches))if(!only||key===only)for(const[a,b]of ranges)out.push({key,a,b});return out}function highlight(line,only=null){const hits=spans(line,only);if(!hits.length)return esc(line.text);const chars=Array.from(line.text),bounds=[0,chars.length];for(const hit of hits)bounds.push(hit.a,hit.b);const points=[...new Set(bounds)].sort((a,b)=>a-b);let out='';for(let i=0;i<points.length-1;i++){const a=points[i],b=points[i+1],text=esc(chars.slice(a,b).join('')),owners=hits.filter(h=>h.a<=a&&h.b>=b);if(!owners.length){out+=text;continue}const colors=[...new Set(owners.map(h=>featureByKey[h.key].color+'77'))],background=colors.length===1?colors[0]:`linear-gradient(${colors.join(',')})`;out+=`<span class="charhit" style="background:${background}" title="${esc(owners.map(h=>featureByKey[h.key].label).join(' · '))}">${text}</span>`}return out}
function blockForLine(i){return doc().blocks.findIndex(b=>b.local_start<=i&&i<=b.local_end)}function reviewKey(line){return`${doc().document_id}:${line.abs_idx}`}function isWrong(line){return!!wrong[reviewKey(line)]}function isShouldBib(line){return!!shouldBib[reviewKey(line)]}function lineHtml(line){const block=blockForLine(line.local_index),active=block===currentBlock&&block>=0,bad=isWrong(line),missed=isShouldBib(line),pressed=bad||missed,label=line.predicted_bib?`Toggle wrong for line L${line.abs_idx}`:`Toggle should be bibliography for line L${line.abs_idx}`,decision=line.predicted_bib?(bad?'WRONG':'BIB'):(missed?'SHOULD BE BIB':'—');return`<button type="button" aria-pressed="${pressed}" aria-label="${label}" id="line-${line.local_index}" data-line-index="${line.local_index}" class="line ${line.predicted_bib?'bib':''} ${active?'active-block':''} ${bad?'wrong':''} ${missed?'should-bib':''}"><span class="ln">L${line.abs_idx}</span><span class="decision ${line.predicted_bib?'on':''} ${bad?'wrong':''} ${missed?'missed':''}">${decision}</span><span class="text" data-line="${line.local_index}">${highlight(line)}</span></button>`}
function renderFeatures(){const active=new Set;for(const line of doc().lines)for(const key of Object.keys(line.features))active.add(key);document.getElementById('features').innerHTML=PACKET.features.filter(f=>active.has(f.key)).map(f=>`<div class="feature" data-feature="${esc(f.key)}"><span class="dot" style="background:${f.color}"></span><span>${esc(f.label)}</span></div>`).join('');for(const node of document.querySelectorAll('[data-feature]')){node.onmouseenter=()=>{const key=node.dataset.feature;for(const text of document.querySelectorAll('[data-line]'))text.innerHTML=highlight(doc().lines[Number(text.dataset.line)],key)};node.onmouseleave=()=>{for(const text of document.querySelectorAll('[data-line]'))text.innerHTML=highlight(doc().lines[Number(text.dataset.line)])}}}
function currentBlockLines(){const block=doc().blocks[currentBlock];return block?doc().lines.slice(block.local_start,block.local_end+1):[]}function updateCounter(){const n=doc().blocks.length,has=n>0,allWrong=has&&currentBlockLines().every(isWrong);document.getElementById('blockCounter').textContent=has?`BIB block ${currentBlock+1} of ${n}`:'No BIB blocks predicted';document.getElementById('prevBlock').disabled=!has;document.getElementById('nextBlock').disabled=!has;const toggle=document.getElementById('toggleBlockWrong');toggle.disabled=!has;toggle.textContent=allWrong?'Unmark whole block':'Mark whole block WRONG'}
function updateReviewCount(){const hereWrong=doc().lines.filter(isWrong).length,hereMissed=doc().lines.filter(isShouldBib).length,total=Object.keys(wrong).length+Object.keys(shouldBib).length;document.getElementById('wrongCount').textContent=`${hereWrong} wrong · ${hereMissed} should be BIB in this document · ${total} total marks`}function paintWrong(line,bad){const node=document.getElementById(`line-${line.local_index}`);if(!node)return;node.classList.toggle('wrong',bad);node.setAttribute('aria-pressed',String(bad));const badge=node.querySelector('.decision');badge.classList.toggle('wrong',bad);badge.textContent=bad?'WRONG':'BIB'}function saveWrong(){localStorage.setItem(wrongStorageKey,JSON.stringify(wrong));updateReviewCount();updateCounter()}function toggleWrong(localIndex){const line=doc().lines[localIndex];if(!line?.predicted_bib)return;const key=reviewKey(line),bad=!wrong[key];if(bad)wrong[key]=true;else delete wrong[key];paintWrong(line,bad);saveWrong()}function toggleBlockWrong(){const lines=currentBlockLines();if(!lines.length)return;const mark=!lines.every(isWrong);for(const line of lines){const key=reviewKey(line);if(mark)wrong[key]=true;else delete wrong[key];paintWrong(line,mark)}saveWrong()}function toggleShouldBib(localIndex){const line=doc().lines[localIndex];if(!line||line.predicted_bib)return;const key=reviewKey(line),marked=!shouldBib[key];if(marked)shouldBib[key]=true;else delete shouldBib[key];localStorage.setItem(shouldBibStorageKey,JSON.stringify(shouldBib));const node=document.getElementById(`line-${line.local_index}`),badge=node?.querySelector('.decision');node?.classList.toggle('should-bib',marked);node?.setAttribute('aria-pressed',String(marked));if(badge){badge.classList.toggle('missed',marked);badge.textContent=marked?'SHOULD BE BIB':'—'}updateReviewCount()}function toggleLineReview(localIndex){const line=doc().lines[localIndex];if(line?.predicted_bib)toggleWrong(localIndex);else toggleShouldBib(localIndex)}
function renderLines(){document.getElementById('lines').innerHTML=doc().lines.map(lineHtml).join('');for(const node of document.querySelectorAll('button.line'))node.onclick=()=>toggleLineReview(Number(node.dataset.lineIndex));updateCounter();updateReviewCount();renderFeatures()}
function markActiveBlock(){for(const node of document.querySelectorAll('.line.active-block'))node.classList.remove('active-block');const block=doc().blocks[currentBlock];if(block)for(let i=block.local_start;i<=block.local_end;i++)document.getElementById(`line-${i}`)?.classList.add('active-block');updateCounter()}
function blockStartNode(index){const block=doc().blocks[index];return block?document.getElementById(`line-${block.local_start}`):null}function directionalBlock(step){const blocks=doc().blocks,anchor=document.getElementById('docHead').getBoundingClientRect().bottom+8,positions=blocks.map((_,index)=>({index,top:blockStartNode(index)?.getBoundingClientRect().top??Infinity}));if(step>0)return positions.find(item=>item.top>anchor+3)?.index??0;for(let i=positions.length-1;i>=0;i--)if(positions[i].top<anchor-3)return positions[i].index;return blocks.length-1}function alignCurrentBlock(epoch,attempt=0){if(epoch!==navigationEpoch)return;const node=blockStartNode(currentBlock);if(!node)return;const anchor=document.getElementById('docHead').getBoundingClientRect().bottom+8,delta=node.getBoundingClientRect().top-anchor;if(Math.abs(delta)>1)window.scrollBy(0,delta);if(attempt<2)requestAnimationFrame(()=>alignCurrentBlock(epoch,attempt+1))}function jumpBlock(step=0){const blocks=doc().blocks;if(!blocks.length)return;if(step)currentBlock=directionalBlock(step);currentBlock=(currentBlock+blocks.length)%blocks.length;markActiveBlock();alignCurrentBlock(++navigationEpoch)}
function updateDocWeird(){const marked=!!weird[doc().document_id],button=document.getElementById('toggleDocWeird');button.classList.toggle('is-weird',marked);button.setAttribute('aria-pressed',String(marked));button.textContent=marked?'Unmark document WEIRD':'Mark document WEIRD';document.getElementById('docHead').classList.toggle('weird',marked);const total=Object.keys(weird).length;document.getElementById('weirdCount').textContent=`${total} weird document${total===1?'':'s'}`}function toggleDocWeird(){const key=doc().document_id;if(weird[key])delete weird[key];else weird[key]=true;localStorage.setItem(weirdStorageKey,JSON.stringify(weird));updateDocWeird()}
function exportReview(){const payload={schema_version:'bibliography-unseen-review-v1',exported_at:new Date().toISOString(),packet_title:PACKET.title,wrong_predicted_lines:Object.keys(wrong).sort(),should_be_bib_lines:Object.keys(shouldBib).sort(),weird_documents:Object.keys(weird).sort()},blob=new Blob([JSON.stringify(payload,null,2)+'\n'],{type:'application/json'}),link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='bibliography-unseen-review.json';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000)}
function render(autoJump=true){const d=doc();document.getElementById('docSelect').value=String(current);document.getElementById('docHead').innerHTML=`<b>${esc(d.source)}</b><span>${esc(d.document_id.slice(0,12))}</span><span>${d.blocks.length} predicted block${d.blocks.length===1?'':'s'}</span><span>${d.n_present_lines.toLocaleString()} displayed lines</span>`;currentBlock=0;renderLines();updateDocWeird();localStorage.setItem('bib-unseen-review-doc',d.document_id);if(autoJump&&d.blocks.length)requestAnimationFrame(()=>jumpBlock(0,false))}
fetch('packet.json').then(r=>{if(!r.ok)throw Error(r.status);return r.json()}).then(packet=>{PACKET=packet;docs=packet.documents;featureByKey=Object.fromEntries(packet.features.map(f=>[f.key,f]));document.getElementById('title').textContent=packet.title;document.getElementById('subtitle').textContent=packet.interpretation;const select=document.getElementById('docSelect'),groups=new Map;docs.forEach((d,i)=>{if(!groups.has(d.source))groups.set(d.source,[]);groups.get(d.source).push([d,i])});for(const[source,items]of groups){const group=document.createElement('optgroup');group.label=`${source} (${items.length})`;for(const[d,i]of items){const option=document.createElement('option');option.value=i;option.textContent=`${d.document_id.slice(0,12)} · ${d.blocks.length} BIB block${d.blocks.length===1?'':'s'}`;group.appendChild(option)}select.appendChild(group)}const stored=localStorage.getItem('bib-unseen-review-doc'),found=docs.findIndex(d=>d.document_id===stored);current=found>=0?found:0;select.onchange=e=>{current=Number(e.target.value);render()};document.getElementById('toggleDocWeird').onclick=toggleDocWeird;document.getElementById('exportReview').onclick=exportReview;document.getElementById('prevBlock').onclick=()=>jumpBlock(-1);document.getElementById('nextBlock').onclick=()=>jumpBlock(1);document.getElementById('toggleBlockWrong').onclick=toggleBlockWrong;document.onkeydown=e=>{if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName))return;if(e.key==='ArrowUp'){e.preventDefault();jumpBlock(-1)}if(e.key==='ArrowDown'){e.preventDefault();jumpBlock(1)}};render()}).catch(error=>document.body.innerHTML=`<pre>${esc(error.stack||error)}</pre>`);
</script></body></html>'''


def run(args: argparse.Namespace) -> dict[str, Any]:
    documents_path = Path(args.documents).resolve()
    selection_path = Path(args.selection_manifest).resolve()
    selection_manifest = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection_manifest.get("schema_version") != "academic-structure-source-matched-holdout-manifest-v1":
        raise ValueError("unsupported source-matched selection manifest")
    if selection_manifest.get("outputs", {}).get("documents_sha256") != sha256_file(documents_path):
        raise ValueError("source-matched documents do not match the selection manifest")
    holdout_rows = _read_jsonl(documents_path)
    candidate_rows = select_candidate_pool(
        holdout_rows, per_source=int(args.candidate_per_source), seed=str(args.seed)
    )
    table, source_rows = materialize_inference_table(candidate_rows, workers=int(args.workers))

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    source_path = output_dir / "candidate_pool.jsonl"
    _write_jsonl(source_path, source_rows)

    line_model_path = Path(args.line_model).resolve()
    line_probability = _line_probability(table, line_model_path)
    roles, role_counts = _roles(table, source_rows, workers=int(args.workers))
    signal_features = build_signal_features(line_probability, roles, table.header_kinds)

    signal_root = Path(args.signal_tcn_dir).resolve()
    signal_report_path = signal_root / "signal_tcn_oof_report.json"
    signal_report = json.loads(signal_report_path.read_text(encoding="utf-8"))
    architecture = dict(signal_report["architecture"])
    checkpoint_architecture = {
        "hidden_dim": int(architecture["hidden_dim"]),
        "dilations": [int(value) for value in architecture["dilations"]],
        "dropout": float(architecture["dropout"]),
    }
    model_paths = sorted((signal_root / "models").glob("fold*.pt"))
    if len(model_paths) != 5:
        raise ValueError("expected five frozen signal-TCN fold models")
    signal_probability = _ensemble_probability(
        table,
        signal_features,
        model_paths,
        checkpoint_architecture,
        central_width=int(architecture["central_width"]),
        context=int(architecture["context_lines"]),
        batch_size=int(args.batch_size),
    )
    _headings, auxiliary_scope = materialize_auxiliary_headings(
        table, source_path, expected_split="validation"
    )
    recall_report_path = Path(args.recall_block_dir).resolve() / "signal_block_decode_oof_report.json"
    recall_report = json.loads(recall_report_path.read_text(encoding="utf-8"))
    decoder_row = select_train_recall_candidate(recall_report)
    prediction, barrier_count = decode_signal_blocks(
        table,
        signal_probability,
        line_probability,
        auxiliary_scope,
        BlockConfig(**decoder_row["config"]),
        qualified_documents=set(range(len(table.documents))),
        apply_veto=True,
    )
    selected_indices, selection_summary = choose_review_documents(
        candidate_rows,
        table,
        prediction,
        per_source=int(args.documents_per_source),
    )
    classifier = {
        "line_model": "D1 fitted on STRUCT-2K train",
        "signal_model": "five-fold train ensemble",
        "decoder": "frozen recall-first anchored decoder",
        "decoder_config": decoder_row["config"],
        "model_fitting_performed": False,
        "threshold_tuning_performed": False,
    }
    packet = _packet(
        candidate_rows,
        table,
        selected_indices,
        line_probability,
        signal_probability,
        prediction,
        selection_summary=selection_summary,
        classifier=classifier,
    )
    packet_path = output_dir / "packet.json"
    _write_json(packet_path, packet)
    with (output_dir / "index.html").open("x", encoding="utf-8") as handle:
        handle.write(HTML)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_frozen_classifier_unseen_block_review",
        "document_count": len(packet["documents"]),
        "source_counts": packet["source_counts"],
        "predicted_block_count": sum(len(row["blocks"]) for row in packet["documents"]),
        "predicted_line_count": sum(row["predicted_line_count"] for row in packet["documents"]),
        "candidate_selection": {
            "seed": str(args.seed),
            "candidate_per_source": int(args.candidate_per_source),
            "documents_per_source": int(args.documents_per_source),
            "policy": "outcome-blind hash pool, then classifier-positive documents first",
            "summary": selection_summary,
        },
        "role_counts": role_counts,
        "scope_barrier_interval_count": barrier_count,
        "classifier": classifier,
        "inputs": {
            "documents_sha256": sha256_file(documents_path),
            "selection_manifest_sha256": sha256_file(selection_path),
            "line_model_sha256": sha256_file(line_model_path),
            "signal_report_sha256": sha256_file(signal_report_path),
            "signal_model_sha256": {path.name: sha256_file(path) for path in model_paths},
            "recall_block_report_sha256": sha256_file(recall_report_path),
        },
        "execution": {
            "code_commit": str(args.code_commit),
            "slurm_job_id": str(args.slurm_job_id),
            "workers": int(args.workers),
            "model_fitting_performed": False,
            "threshold_tuning_performed": False,
            "corpus_mutation_performed": False,
        },
        "production_eligible": False,
    }
    _write_json(output_dir / "report.json", report)
    outputs = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    _write_json(output_dir / "receipt.json", {**report, "outputs": outputs})
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--line-model", required=True)
    parser.add_argument("--signal-tcn-dir", required=True)
    parser.add_argument("--recall-block-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-per-source", type=int, default=30)
    parser.add_argument("--documents-per-source", type=int, default=10)
    parser.add_argument("--seed", default="unseen-bibliography-review-20260714")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
