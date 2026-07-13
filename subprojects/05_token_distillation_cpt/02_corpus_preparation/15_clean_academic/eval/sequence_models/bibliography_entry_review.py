#!/usr/bin/env python3
"""Build the post-freeze bibliography-block joint-review site.

The packet contains complete emitted-document context for a source-balanced
sample of high-risk proposed removals.  It is a review artifact only: decisions
are stored in the reviewer's browser and never feed back into the frozen model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .bibliography_entry_blocks import BlockConfig, _iou, blocks_from_mask
from .bibliography_entry_dataset import LABEL_TO_ID
from .bibliography_entry_models import load_table
from .bibliography_feature_explorer import FEATURE_SPECS
from .bibliography_v2 import extract_bibliography_feature_review


SCHEMA_VERSION = "bibliography-entry-high-risk-review-v1"
DEFAULT_CASE_COUNT = 120


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


def _case_rows(
    table: Any,
    prediction: np.ndarray,
    probability: np.ndarray,
    config: BlockConfig,
) -> list[dict[str, Any]]:
    gold_all = table.original_labels == LABEL_TO_ID["BIB"]
    cases: list[dict[str, Any]] = []
    for document_index, document in enumerate(table.documents):
        start, end = int(document["line_start"]), int(document["line_end"])
        predicted = prediction[start:end]
        gold = gold_all[start:end]
        abs_indices = table.abs_indices[start:end]
        predicted_blocks = blocks_from_mask(predicted, abs_indices)
        gold_blocks = blocks_from_mask(gold, abs_indices)
        for block_index, (block_start, block_end) in enumerate(predicted_blocks):
            region = slice(block_start, block_end + 1)
            block_gold = gold[region]
            false_lines = int(np.count_nonzero(~block_gold))
            line_count = block_end - block_start + 1
            overlaps = [_iou((block_start, block_end), item) for item in gold_blocks]
            best_iou = max(overlaps, default=0.0)
            long = table.char_lengths[start:end][region] > config.seed_length_limit
            false_long = int(np.count_nonzero(long & ~block_gold))
            minimum_probability = float(np.min(probability[start:end][region]))
            mean_probability = float(np.mean(probability[start:end][region]))
            zero_silver = not gold_blocks
            risk_score = (
                100.0 * int(zero_silver)
                + 60.0 * false_lines / max(line_count, 1)
                + 25.0 * (1.0 - best_iou)
                + 8.0 * false_long
                + 5.0 * (1.0 - minimum_probability)
            )
            reasons = []
            if zero_silver:
                reasons.append("proposal_in_silver_zero_document")
            if false_lines:
                reasons.append("contains_silver_non_bib_lines")
            if best_iou < 0.5:
                reasons.append("no_silver_block_iou_at_least_0_5")
            elif best_iou < 1.0:
                reasons.append("boundary_disagreement")
            if false_long:
                reasons.append("long_silver_non_bib_line_absorbed")
            if minimum_probability < config.inside_probability:
                reasons.append("contains_very_low_probability_line")
            if not reasons:
                reasons.append("lower_risk_source_balancing_case")
            cases.append(
                {
                    "case_id": f"{document['document_id']}:{block_index}",
                    "document_id": str(document["document_id"]),
                    "document_index": document_index,
                    "source": str(document["source"]),
                    "coverage": str(document["coverage"]),
                    "predicted_block_index": block_index,
                    "line_start": block_start,
                    "line_end": block_end,
                    "abs_start": int(abs_indices[block_start]),
                    "abs_end": int(abs_indices[block_end]),
                    "line_count": line_count,
                    "false_positive_line_count": false_lines,
                    "false_positive_long_line_count": false_long,
                    "best_silver_iou": best_iou,
                    "minimum_entry_probability": minimum_probability,
                    "mean_entry_probability": mean_probability,
                    "risk_score": risk_score,
                    "risk_reasons": reasons,
                    "silver_zero_document": zero_silver,
                }
            )
    return cases


def select_source_balanced_cases(
    cases: Sequence[Mapping[str, Any]], count: int
) -> list[dict[str, Any]]:
    if count < 100:
        raise ValueError("joint review requires at least 100 requested cases")
    sources = sorted({str(case["source"]) for case in cases})
    if not sources:
        raise ValueError("no predicted bibliography blocks are available for review")
    ranked = sorted(
        (dict(case) for case in cases),
        key=lambda case: (-float(case["risk_score"]), str(case["case_id"])),
    )
    base, remainder = divmod(count, len(sources))
    quota = {source: base + int(index < remainder) for index, source in enumerate(sources)}
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for source in sources:
        source_rows = [case for case in ranked if str(case["source"]) == source]
        for case in source_rows[: quota[source]]:
            selected.append(case)
            used.add(str(case["case_id"]))
    for case in ranked:
        if len(selected) >= count:
            break
        if str(case["case_id"]) not in used:
            selected.append(case)
            used.add(str(case["case_id"]))
    if len(selected) < 100:
        raise ValueError(f"only {len(selected)} proposed blocks exist; 100 are required")
    selected.sort(key=lambda case: (-float(case["risk_score"]), str(case["case_id"])))
    for index, case in enumerate(selected):
        case["review_order"] = index
    return selected


def _feature_line(
    raw_line: Mapping[str, Any],
    *,
    local_index: int,
    probability: float,
    predicted: bool,
    seed_length_limit: int,
    anchor_probability: float,
) -> dict[str, Any]:
    text = raw_line.get("text")
    abs_idx = raw_line.get("abs_idx")
    label = str(raw_line.get("label", "UNKNOWN"))
    if not isinstance(text, str) or not isinstance(abs_idx, int):
        raise ValueError("malformed review line")
    review = extract_bibliography_feature_review(text)
    counts = review.features.as_dict()
    matches: dict[str, list[list[int]]] = {}
    for match in review.matches:
        matches.setdefault(match.feature, []).append([match.start, match.end])
    char_length = len(review.normalized_text)
    if predicted:
        if char_length > seed_length_limit:
            decision_reason = "long_line_absorbed_inside_confirmed_block"
        elif probability >= anchor_probability:
            decision_reason = "high_entry_probability_inside_proposed_block"
        else:
            decision_reason = "weak_line_absorbed_inside_confirmed_block"
    elif char_length > seed_length_limit:
        decision_reason = "long_line_did_not_seed_a_block"
    elif probability >= anchor_probability:
        decision_reason = "high_score_without_confirmed_block_context"
    else:
        decision_reason = "below_block_entry_evidence"
    return {
        "local_index": local_index,
        "abs_idx": abs_idx,
        "text": review.normalized_text,
        "silver_label": label,
        "entry_probability": round(probability, 7),
        "predicted_bib": predicted,
        "char_length": char_length,
        "token_count": int(raw_line.get("token_count", counts["token_count"])),
        "features": {
            spec.key: int(counts[spec.key])
            for spec in FEATURE_SPECS
            if int(counts[spec.key]) > 0
        },
        "matches": matches,
        "decision_reason": decision_reason,
    }


def build_packet(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input).resolve()
    validation_root = Path(args.validation_dir).resolve()
    frozen = json.loads(Path(args.frozen_config).resolve().read_text(encoding="utf-8"))
    table = load_table(validation_root / "validation_table", expected_split="validation")
    probability = np.load(
        validation_root / "validation_line_probability.npy", mmap_mode="r", allow_pickle=False
    )
    prediction = np.load(
        validation_root / "validation_block_prediction.npy", mmap_mode="r", allow_pickle=False
    )
    if len(probability) != len(table.targets) or len(prediction) != len(table.targets):
        raise ValueError("validation predictions do not align with the feature table")
    config = BlockConfig(**frozen["b0_h0_config"])
    selected = select_source_balanced_cases(
        _case_rows(table, prediction, probability, config), int(args.case_count)
    )

    rows = {
        str(row.get("document_id")): row
        for row in _iter_rows(input_path)
        if row.get("split") == "validation"
    }
    selected_ids = {str(case["document_id"]) for case in selected}
    if selected_ids - rows.keys():
        raise ValueError("selected validation documents are absent from the pinned input")
    document_by_id = {str(row["document_id"]): row for row in table.documents}
    documents = []
    for document_id in sorted(selected_ids):
        meta = document_by_id[document_id]
        raw = rows[document_id]
        raw_lines = raw.get("lines")
        if not isinstance(raw_lines, list):
            raise ValueError(f"{document_id}: missing line inventory")
        start, end = int(meta["line_start"]), int(meta["line_end"])
        if len(raw_lines) != end - start:
            raise ValueError(f"{document_id}: source/table line-count mismatch")
        lines = [
            _feature_line(
                line,
                local_index=index,
                probability=float(probability[start + index]),
                predicted=bool(prediction[start + index]),
                seed_length_limit=config.seed_length_limit,
                anchor_probability=config.anchor_probability,
            )
            for index, line in enumerate(raw_lines)
        ]
        documents.append(
            {
                "document_id": document_id,
                "work_id": str(meta["work_id"]),
                "source": str(meta["source"]),
                "coverage": str(meta["coverage"]),
                "n_physical_lines": int(meta["n_physical_lines"]),
                "lines": lines,
            }
        )
    features = [
        {"key": spec.key, "label": spec.label, "group": spec.group, "color": spec.color}
        for spec in FEATURE_SPECS
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "title": "Bibliography block high-risk joint review",
        "evidence_scope": "LLM-silver retrospective validation; not human gold",
        "selection_rule": "source-balanced highest-risk proposed blocks; no tuning on review decisions",
        "selected_architecture": frozen["selected"]["architecture"],
        "selected_line_arm": frozen["line_model"]["arm"],
        "block_config": frozen["b0_h0_config"],
        "validation_report_sha256": _sha256(validation_root / "validation_report.json"),
        "frozen_config_sha256": _sha256(Path(args.frozen_config).resolve()),
        "input_sha256": _sha256(input_path),
        "case_count": len(selected),
        "source_counts": {
            source: sum(str(case["source"]) == source for case in selected)
            for source in sorted({str(case["source"]) for case in selected})
        },
        "features": features,
        "cases": selected,
        "documents": documents,
        "review_contract": {
            "reviewers": ["foivos", "codex"],
            "decisions": ["approve_removal", "keep_false_removal", "boundary_issue", "unsure"],
            "storage": "browser_local_storage_only",
            "tuning_from_this_sample": False,
        },
    }


HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Bibliography block review</title>
<style>
:root{--ink:#272c29;--muted:#69716c;--paper:#fbf8f0;--panel:#fffdf8;--line:#ded8ca;--proposal:#dceef7;--silver:#b73232;--green:#27835a;--orange:#c77b1e;--shadow:0 10px 30px #24372a16}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.45 system-ui,-apple-system,sans-serif}.top{position:sticky;top:0;z-index:5;background:#fffdf8f5;border-bottom:1px solid var(--line);backdrop-filter:blur(10px)}.topin{max-width:1500px;margin:auto;padding:12px 18px;display:flex;align-items:center;gap:14px}.top h1{font:700 21px/1.1 Georgia,serif;margin:0}.sub{color:var(--muted);font-size:11px}.progress{margin-left:auto;text-align:right}.progress b{font-size:18px;color:var(--green)}.layout{max-width:1500px;margin:18px auto;display:grid;grid-template-columns:300px minmax(0,1fr) 260px;gap:15px;padding:0 16px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:13px;box-shadow:var(--shadow)}.side{position:sticky;top:82px;max-height:calc(100vh - 100px);overflow:auto;padding:14px}.side h2{font:700 17px Georgia,serif;margin:0 0 9px}.side label{display:block;color:var(--muted);font-size:11px;margin:10px 0 4px}select{width:100%;padding:8px;border:1px solid var(--line);border-radius:8px;background:white}.case-meta{margin-top:12px;display:grid;gap:7px}.pill{display:inline-block;padding:3px 7px;border-radius:999px;background:#eee8dc;font-size:10px;margin:2px}.risk{font-size:28px;font-weight:800;color:#a95528}.reason{font-size:11px;padding:5px 7px;background:#f4efe5;border-radius:7px}.reader{overflow:hidden}.dochead{position:sticky;top:70px;z-index:3;padding:11px 14px;background:#fffdf8f0;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center}.doclines{padding:11px}.line{display:grid;grid-template-columns:54px 62px minmax(0,1fr);gap:9px;padding:5px 7px;border-radius:7px;scroll-margin-top:135px}.line.proposed{background:var(--proposal)}.line.silver .text{text-decoration:underline 2px var(--silver);text-underline-offset:3px}.line.case-start{box-shadow:inset 0 3px #237aa1}.line.case-end{box-shadow:inset 0 -3px #237aa1}.ln,.prob{font:10px/1.8 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--muted);text-align:right}.prob strong{display:block;font-size:11px}.text{font:15px/1.48 Georgia,"Times New Roman",serif;overflow-wrap:anywhere}.charhit{border-radius:3px;padding:1px 0;box-shadow:inset 0 -2px #0004}.why{display:block;color:var(--muted);font:10px/1.35 system-ui;margin-top:3px}.feature-list{display:flex;flex-direction:column;gap:5px}.feature{display:grid;grid-template-columns:10px 1fr;gap:7px;font-size:11px;padding:4px;border-radius:6px}.dot{width:9px;height:9px;border-radius:3px;margin-top:3px}.legend{font-size:11px;color:var(--muted);margin-bottom:12px}.decision-grid{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:12px}.decision-grid button{border:1px solid var(--line);border-radius:9px;background:#f5f0e7;padding:9px 6px;cursor:pointer;font-size:11px}.decision-grid button:hover{background:#e9e2d5}.decision-grid button.chosen{outline:3px solid #27835a55;background:#e3f2e9}.kbd{font-size:18px;font-weight:800;display:block}.nav{display:flex;gap:7px;margin-top:9px}.nav button{flex:1;padding:7px;border:1px solid var(--line);border-radius:8px;background:white;cursor:pointer}@media(max-width:1050px){.layout{grid-template-columns:240px minmax(0,1fr)}.featureside{display:none}}@media(max-width:720px){.layout{display:block;padding:0 7px}.side{position:relative;top:auto;max-height:none;margin-bottom:10px}.top{position:relative}.dochead{top:0}.line{grid-template-columns:38px 50px 1fr}.progress{display:none}}
</style></head><body><header class="top"><div class="topin"><div><h1 id="title"></h1><div id="subtitle" class="sub"></div></div><div class="progress"><b id="progress"></b><div class="sub">reviewed by selected reviewer</div></div></div></header><main class="layout"><aside class="panel side"><h2>Review case</h2><label>Reviewer</label><select id="reviewer"><option value="foivos">Foivos</option><option value="codex">Codex</option></select><label>Case</label><select id="caseSelect"></select><div id="caseMeta" class="case-meta"></div><div class="decision-grid"><button data-decision="keep_false_removal"><span class="kbd">←</span>Keep / false removal</button><button data-decision="boundary_issue"><span class="kbd">↑</span>Boundary issue</button><button data-decision="approve_removal"><span class="kbd">→</span>Approve removal</button><button data-decision="unsure"><span class="kbd">↓</span>Unsure</button></div><div class="nav"><button id="prev">Previous</button><button id="next">Next undecided</button></div><div class="legend">Blue background = model proposal. Red underline = LLM-silver BIB. Blue top/bottom rules = current review block.</div></aside><section class="panel reader"><div id="docHead" class="dochead"></div><div id="docLines" class="doclines"></div></section><aside class="panel side featureside"><h2>Detected features</h2><div class="legend">Hover a feature to isolate its character boxes. These spans are diagnostic and did not change after validation.</div><div id="features" class="feature-list"></div></aside></main>
<script>
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let PACKET,cases,docs,featureByKey,current=0,reviewer='foivos';
function key(){return'bib-entry-review:'+PACKET.validation_report_sha256+':'+reviewer}function decisions(){try{return JSON.parse(localStorage.getItem(key())||'{}')}catch{return{}}}function save(d){localStorage.setItem(key(),JSON.stringify(d))}
function activeCase(){return cases[current]}function activeDoc(){return docs[activeCase().document_id]}
function highlight(line,only=null){const spans=[];for(const [k,ranges]of Object.entries(line.matches))if(!only||k===only)for(const [a,b]of ranges)spans.push({k,a,b});if(!spans.length)return esc(line.text);const chars=Array.from(line.text),bounds=[0,chars.length];for(const s of spans)bounds.push(s.a,s.b);const ordered=[...new Set(bounds)].sort((a,b)=>a-b);let out='';for(let i=0;i<ordered.length-1;i++){const a=ordered[i],b=ordered[i+1],txt=esc(chars.slice(a,b).join('')),hits=spans.filter(s=>s.a<=a&&s.b>=b);if(!hits.length){out+=txt;continue}const colors=[...new Set(hits.map(s=>featureByKey[s.k].color+'66'))],bg=colors.length===1?colors[0]:`linear-gradient(${colors.join(',')})`;out+=`<span class="charhit" style="background:${bg}" title="${esc(hits.map(s=>featureByKey[s.k].label).join(' · '))}">${txt}</span>`}return out}
function renderProgress(){const d=decisions(),n=cases.filter(c=>d[c.case_id]).length;document.getElementById('progress').textContent=`${n} / ${cases.length}`;for(const b of document.querySelectorAll('[data-decision]'))b.classList.toggle('chosen',d[activeCase().case_id]===b.dataset.decision)}
function renderCase(){const c=activeCase(),doc=activeDoc();document.getElementById('caseSelect').value=String(current);document.getElementById('caseMeta').innerHTML=`<div><span class="risk">${c.risk_score.toFixed(1)}</span><div class="sub">risk score · ${esc(c.source)}</div></div><div>${c.risk_reasons.map(x=>`<div class="reason">${esc(x.replaceAll('_',' '))}</div>`).join('')}</div><div class="sub">proposal L${c.abs_start}–L${c.abs_end} · ${c.line_count} emitted lines<br>best silver IoU ${c.best_silver_iou.toFixed(3)} · mean p ${c.mean_entry_probability.toFixed(3)}<br>${c.false_positive_line_count} silver-non-BIB lines · ${c.false_positive_long_line_count} long</div>`;document.getElementById('docHead').innerHTML=`<b>${esc(c.source)}</b><span>doc ${esc(doc.document_id.slice(0,12))}</span><span class="sub">${doc.lines.length.toLocaleString()} emitted lines · ${esc(PACKET.selected_architecture)} / ${esc(PACKET.selected_line_arm)}</span>`;document.getElementById('docLines').innerHTML=doc.lines.map(line=>`<div id="line-${line.local_index}" class="line ${line.predicted_bib?'proposed':''} ${line.silver_label==='BIB'?'silver':''} ${line.local_index===c.line_start?'case-start':''} ${line.local_index===c.line_end?'case-end':''}"><span class="ln">L${line.abs_idx}</span><span class="prob"><strong>${line.entry_probability.toFixed(3)}</strong>${line.char_length}ch</span><span class="text" data-line="${line.local_index}">${highlight(line)}<small class="why">${esc(line.decision_reason.replaceAll('_',' '))}</small></span></div>`).join('');renderProgress();requestAnimationFrame(()=>document.getElementById(`line-${c.line_start}`)?.scrollIntoView({block:'center'}))}
function renderFeatures(){document.getElementById('features').innerHTML=PACKET.features.map(f=>`<div class="feature" data-feature="${esc(f.key)}"><span class="dot" style="background:${f.color}"></span><span>${esc(f.label)}</span></div>`).join('');for(const node of document.querySelectorAll('[data-feature]')){node.onmouseenter=()=>{const k=node.dataset.feature;for(const text of document.querySelectorAll('[data-line]'))text.innerHTML=highlight(activeDoc().lines[Number(text.dataset.line)],k)};node.onmouseleave=()=>{for(const text of document.querySelectorAll('[data-line]')){const line=activeDoc().lines[Number(text.dataset.line)];text.innerHTML=highlight(line)+`<small class="why">${esc(line.decision_reason.replaceAll('_',' '))}</small>`}}}}
function move(delta){current=Math.max(0,Math.min(cases.length-1,current+delta));renderCase()}function nextUndecided(){const d=decisions();for(let step=1;step<=cases.length;step++){const i=(current+step)%cases.length;if(!d[cases[i].case_id]){current=i;renderCase();return}}move(1)}
fetch('packet.json').then(r=>{if(!r.ok)throw Error(r.status);return r.json()}).then(p=>{PACKET=p;cases=p.cases;docs=Object.fromEntries(p.documents.map(d=>[d.document_id,d]));featureByKey=Object.fromEntries(p.features.map(f=>[f.key,f]));document.getElementById('title').textContent=p.title;document.getElementById('subtitle').textContent=`${p.case_count} source-balanced high-risk proposals · ${p.evidence_scope}`;document.getElementById('caseSelect').innerHTML=cases.map((c,i)=>`<option value="${i}">${i+1}. ${esc(c.source)} · risk ${c.risk_score.toFixed(1)} · ${esc(c.document_id.slice(0,10))}</option>`).join('');document.getElementById('caseSelect').onchange=e=>{current=Number(e.target.value);renderCase()};document.getElementById('reviewer').onchange=e=>{reviewer=e.target.value;const d=decisions();current=Math.max(0,cases.findIndex(c=>!d[c.case_id]));renderCase()};for(const b of document.querySelectorAll('[data-decision]'))b.onclick=()=>{const d=decisions();d[activeCase().case_id]=b.dataset.decision;save(d);nextUndecided()};document.getElementById('prev').onclick=()=>move(-1);document.getElementById('next').onclick=nextUndecided;document.onkeydown=e=>{if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName))return;const map={ArrowLeft:'keep_false_removal',ArrowUp:'boundary_issue',ArrowRight:'approve_removal',ArrowDown:'unsure'};if(map[e.key]){e.preventDefault();document.querySelector(`[data-decision="${map[e.key]}"]`).click()}};const d=decisions();const first=cases.findIndex(c=>!d[c.case_id]);current=first<0?0:first;renderFeatures();renderCase()}).catch(e=>document.body.innerHTML=`<pre>${esc(e.stack||e)}</pre>`);
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
        "status": "passed_review_site_built_no_decisions_collected",
        "case_count": packet["case_count"],
        "source_counts": packet["source_counts"],
        "validation_report_sha256": packet["validation_report_sha256"],
        "code_commit": str(args.code_commit),
        "slurm_job_id": str(args.slurm_job_id),
        "review_decisions_used_for_tuning": False,
        "production_eligible": False,
    }
    _write_json(output_dir / "review_report.json", report)
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
    parser.add_argument("--frozen-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-count", type=int, default=DEFAULT_CASE_COUNT)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
