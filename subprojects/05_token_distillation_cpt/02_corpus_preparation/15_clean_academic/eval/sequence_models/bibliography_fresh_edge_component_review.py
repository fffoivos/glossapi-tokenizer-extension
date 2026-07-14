#!/usr/bin/env python3
"""Build independent fresh-edge and bibliography-component review packets.

The source-matched holdout already excludes all STRUCT-2K identities and
near-duplicates.  This job additionally excludes every document and work in
one or more earlier review packets before selecting a new outcome-blind pool.
It fits or tunes nothing.  The frozen bibliography stack is run once and used
to produce two separate blinded review datasets:

* every line removed by the frozen asymmetric edge rule, plus retained
  boundary controls; and
* source-balanced predicted components enriched separately for
  citation-dense narrative risk and bibliography-like structure.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .bibliography_auxiliary_scope_veto import materialize_auxiliary_headings
from .bibliography_entry_blocks import BlockConfig, blocks_from_mask
from .bibliography_feature_explorer import FEATURE_SPECS
from .bibliography_signal_block_decode import decode_signal_blocks
from .bibliography_signal_refinement import SCHEMA_VERSION as REFINEMENT_SCHEMA
from .bibliography_signal_refinement_unseen import (
    _core_prediction,
    _edge_prediction,
    _header_kinds,
)
from .bibliography_signal_tcn import build_signal_features
from .bibliography_signal_validation import _ensemble_probability, select_train_recall_candidate
from .bibliography_unseen_block_review import (
    SOURCES,
    _feature_payload,
    _line_probability,
    _read_jsonl,
    _roles,
    materialize_inference_table,
    select_candidate_pool,
)
from .contract import sha256_file


SCHEMA_VERSION = "bibliography-fresh-edge-component-review-v1"
EDGE_PACKET_SCHEMA = "bibliography-fresh-edge-review-v1"
COMPONENT_PACKET_SCHEMA = "bibliography-component-label-review-v1"


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_prior_exclusions(paths: Sequence[str | Path]) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    document_ids: set[str] = set()
    work_ids: set[str] = set()
    receipts: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        packet = json.loads(path.read_text(encoding="utf-8"))
        documents = packet.get("documents")
        if not isinstance(documents, list) or not documents:
            raise ValueError(f"{path}: prior packet lacks documents")
        before = len(document_ids)
        for row in documents:
            document_id = str(row.get("document_id", ""))
            work_id = str(row.get("work_id", ""))
            if not document_id:
                raise ValueError(f"{path}: prior packet has an empty document ID")
            document_ids.add(document_id)
            if work_id:
                work_ids.add(work_id)
        receipts.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "document_count": len(documents),
                "new_document_exclusions": len(document_ids) - before,
            }
        )
    return document_ids, work_ids, receipts


def load_frozen_edge(refinement_dir: Path) -> tuple[dict[str, Any], BlockConfig, dict[str, Any]]:
    path = refinement_dir / "signal_refinement_oof_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != REFINEMENT_SCHEMA:
        raise ValueError("unsupported refinement report")
    if report.get("validation_opened") is not False:
        raise ValueError("refinement was not train-OOF isolated")
    selected = report.get("edge_experiment", {}).get("selected")
    if not isinstance(selected, dict):
        raise ValueError("refinement has no frozen edge selection")
    arms = report["edge_experiment"]["role_arms"]
    selected = {
        **selected,
        "left_role_names": list(arms[selected["left_role_arm"]]),
        "right_role_names": list(arms[selected["right_role_arm"]]),
    }
    provenance = {
        "report": str(path.resolve()),
        "report_sha256": sha256_file(path),
        "code_commit": report.get("code_commit"),
        "slurm_job_id": report.get("slurm_job_id"),
        "development_review_informed_experiment_design": bool(
            report.get("development_review_informed_experiment_design")
        ),
        "independent_fresh_unseen_evaluation_required": bool(
            report.get("independent_fresh_unseen_evaluation_required")
        ),
    }
    return selected, BlockConfig(**report["frozen_decoder_config"]), provenance


def choose_edge_documents(
    table: Any,
    base: np.ndarray,
    edge: np.ndarray,
    *,
    per_source: int,
) -> tuple[list[int], dict[str, Any]]:
    """Select edge-changed documents first inside an outcome-blind pool."""

    selected: list[int] = []
    summary: dict[str, Any] = {}
    for source in SOURCES:
        rows = []
        for index, document in enumerate(table.documents):
            if document["source"] != source:
                continue
            start, end = int(document["line_start"]), int(document["line_end"])
            removed = int(np.count_nonzero(base[start:end] & ~edge[start:end]))
            predicted = int(np.count_nonzero(base[start:end]))
            rows.append((index, removed, predicted))
        rows.sort(key=lambda row: (row[1] > 0, row[2] > 0), reverse=True)
        chosen = rows[:per_source]
        if len(chosen) != per_source:
            raise ValueError(f"{source}: cannot select {per_source} fresh edge documents")
        selected.extend(row[0] for row in chosen)
        summary[source] = {
            "candidate_pool": len(rows),
            "candidate_documents_with_edge_removals": sum(row[1] > 0 for row in rows),
            "selected": len(chosen),
            "selected_with_edge_removals": sum(row[1] > 0 for row in chosen),
            "selected_removed_line_count": sum(row[1] for row in chosen),
        }
    return selected, summary


def _boundary_controls(base: np.ndarray, edge: np.ndarray, start: int, end: int) -> set[int]:
    """Return at most two kept controls inside each side of an edge transition."""

    kept = set(int(value) for value in np.flatnonzero(edge[start : end + 1]) + start)
    removed = set(range(start, end + 1)) - kept
    controls: set[int] = set()
    if not removed:
        controls.update(range(start, min(end + 1, start + 2)))
        controls.update(range(max(start, end - 1), end + 1))
        return controls
    for index in range(start, end):
        if index in removed and index + 1 in kept:
            controls.update(value for value in (index + 1, index + 2) if value in kept)
        if index in kept and index + 1 in removed:
            controls.update(value for value in (index - 1, index) if value in kept)
    return controls


def _line_payload(line: Mapping[str, Any], *, target: bool) -> dict[str, Any]:
    text = unicodedata.normalize("NFKC", str(line["text"]))
    features: dict[str, int] = {}
    matches: dict[str, list[list[int]]] = {}
    if target:
        text, features, matches = _feature_payload(text)
    return {
        "abs_idx": int(line["abs_idx"]),
        "text": text,
        "target": target,
        "features": features,
        "matches": matches,
    }


def balance_edge_controls(
    cases: Sequence[Mapping[str, Any]], *, minimum_controls_per_source: int = 20
) -> list[dict[str, Any]]:
    """Keep every removal and a bounded deterministic retained-control sample."""

    selected: list[dict[str, Any]] = []
    for source in SOURCES:
        local = [dict(row) for row in cases if row["source"] == source]
        removed = [row for row in local if row["frozen_action"] == "remove"]
        controls = [row for row in local if row["frozen_action"] == "keep"]
        control_limit = min(
            len(controls), max(minimum_controls_per_source, len(removed))
        )
        controls.sort(
            key=lambda row: hashlib.sha256(
                f"fresh-edge-control-v1\0{row['case_id']}".encode()
            ).hexdigest()
        )
        selected.extend(removed)
        selected.extend(controls[:control_limit])
    selected.sort(
        key=lambda row: (
            SOURCES.index(str(row["source"])),
            str(row["document_id"]),
            int(row["abs_idx"]),
        )
    )
    return selected


def build_edge_packet(
    original_rows: Sequence[Mapping[str, Any]],
    table: Any,
    base: np.ndarray,
    edge: np.ndarray,
    selected_documents: Sequence[int],
    *,
    selection_summary: Mapping[str, Any],
    frozen_edge: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    by_id = {str(row["document_id"]): row for row in original_rows}
    cases: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    for document_index in selected_documents:
        meta = table.documents[document_index]
        raw = by_id[str(meta["document_id"])]
        lines = raw["lines"]
        global_start, global_end = int(meta["line_start"]), int(meta["line_end"])
        local_base = base[global_start:global_end]
        local_edge = edge[global_start:global_end]
        case_positions: set[int] = set()
        removed_positions: set[int] = set()
        for block_start, block_end in blocks_from_mask(local_base, table.abs_indices[global_start:global_end]):
            removed = set(
                int(value)
                for value in np.flatnonzero(local_base[block_start : block_end + 1] & ~local_edge[block_start : block_end + 1])
                + block_start
            )
            controls = _boundary_controls(local_base, local_edge, block_start, block_end)
            removed_positions.update(removed)
            case_positions.update(removed | controls)
        documents.append(
            {
                "document_id": str(meta["document_id"]),
                "work_id": str(raw.get("work_id", "")),
                "source": str(meta["source"]),
                "case_count": len(case_positions),
                "removed_case_count": len(removed_positions),
            }
        )
        for local_index in sorted(case_positions):
            line = lines[local_index]
            context_start = max(0, local_index - 2)
            context_end = min(len(lines), local_index + 3)
            action = "remove" if local_index in removed_positions else "keep"
            case_id = f"{meta['document_id']}:{int(line['abs_idx'])}"
            cases.append(
                {
                    "case_id": case_id,
                    "document_id": str(meta["document_id"]),
                    "work_id": str(raw.get("work_id", "")),
                    "source": str(meta["source"]),
                    "abs_idx": int(line["abs_idx"]),
                    "document_position_percent": round(
                        100 * int(line["abs_idx"]) / max(1, int(raw["n_physical_lines"])), 2
                    ),
                    "frozen_action": action,
                    "context": [
                        _line_payload(item, target=index == local_index)
                        for index, item in enumerate(lines[context_start:context_end], context_start)
                    ],
                }
            )
    cases = balance_edge_controls(cases)
    case_counts = collections.Counter(row["document_id"] for row in cases)
    removed_counts = collections.Counter(
        row["document_id"] for row in cases if row["frozen_action"] == "remove"
    )
    for document in documents:
        document["case_count"] = case_counts[document["document_id"]]
        document["removed_case_count"] = removed_counts[document["document_id"]]
    return {
        "schema_version": EDGE_PACKET_SCHEMA,
        "mode": "edge",
        "title": "Fresh frozen-edge review",
        "interpretation": "Label each line before the frozen KEEP/REMOVE decision is revealed.",
        "features": [
            {"key": spec.key, "label": spec.label, "color": spec.color}
            for spec in FEATURE_SPECS
        ],
        "documents": documents,
        "cases": cases,
        "source_counts": dict(collections.Counter(row["source"] for row in documents)),
        "case_source_counts": dict(collections.Counter(row["source"] for row in cases)),
        "retained_control_sampling": (
            "all removed lines; per source, deterministic retained boundary controls "
            "equal to the removed count or 20, whichever is larger"
        ),
        "selection_summary": dict(selection_summary),
        "frozen_edge": dict(frozen_edge),
        "refinement_provenance": dict(provenance),
    }


def _component_rows(
    original_rows: Sequence[Mapping[str, Any]],
    table: Any,
    base: np.ndarray,
    line_probability: np.ndarray,
    signal_probability: np.ndarray,
    roles: np.ndarray,
) -> list[dict[str, Any]]:
    by_id = {str(row["document_id"]): row for row in original_rows}
    rows: list[dict[str, Any]] = []
    for document_index, meta in enumerate(table.documents):
        raw = by_id[str(meta["document_id"])]
        lines = raw["lines"]
        doc_start, doc_end = int(meta["line_start"]), int(meta["line_end"])
        for block_index, (local_start, local_end) in enumerate(
            blocks_from_mask(base[doc_start:doc_end], table.abs_indices[doc_start:doc_end]), 1
        ):
            if local_end - local_start + 1 < 2:
                continue
            start, end = doc_start + local_start, doc_start + local_end
            local_roles = roles[start : end + 1].astype(bool)
            hard_fraction = float(np.mean(np.any(local_roles, axis=1)))
            citation_lines = 0
            feature_sum = 0
            char_lengths = []
            for line in lines[local_start : local_end + 1]:
                review_text, features, _matches = _feature_payload(str(line["text"]))
                active = sum(int(value) > 0 for value in features.values())
                citation_lines += active >= 2
                feature_sum += active
                char_lengths.append(len(review_text))
            citation_fraction = citation_lines / (local_end - local_start + 1)
            entry_fraction = float(np.mean(line_probability[start : end + 1] >= 0.5))
            signal_median = float(np.median(signal_probability[start : end + 1]))
            mean_chars = float(np.mean(char_lengths))
            narrative_rank = (
                2.5 * hard_fraction
                + 1.5 * citation_fraction
                + min(mean_chars / 240.0, 1.5)
                + (1.0 - signal_median)
            )
            bibliography_rank = (
                1.5 * citation_fraction
                + 1.5 * entry_fraction
                + signal_median
                + min(feature_sum / max(1, len(char_lengths) * 6), 1.0)
                - 1.5 * hard_fraction
            )
            rows.append(
                {
                    "document_index": document_index,
                    "document_id": str(meta["document_id"]),
                    "work_id": str(raw.get("work_id", "")),
                    "source": str(meta["source"]),
                    "block_index": block_index,
                    "local_start": local_start,
                    "local_end": local_end,
                    "abs_start": int(lines[local_start]["abs_idx"]),
                    "abs_end": int(lines[local_end]["abs_idx"]),
                    "line_count": local_end - local_start + 1,
                    "citation_line_fraction": citation_fraction,
                    "hard_negative_role_fraction": hard_fraction,
                    "entry_positive_fraction": entry_fraction,
                    "signal_median": signal_median,
                    "mean_characters": mean_chars,
                    "narrative_rank": narrative_rank,
                    "bibliography_rank": bibliography_rank,
                }
            )
    return rows


def select_component_cases(
    rows: Sequence[Mapping[str, Any]], *, per_stratum_per_source: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for source in SOURCES:
        pool = [dict(row) for row in rows if row["source"] == source]
        narrative = sorted(
            pool,
            key=lambda row: (
                float(row["narrative_rank"]),
                float(row["citation_line_fraction"]),
                str(row["document_id"]),
                int(row["local_start"]),
            ),
            reverse=True,
        )[:per_stratum_per_source]
        used = {(row["document_id"], row["local_start"], row["local_end"]) for row in narrative}
        bibliography = [
            row
            for row in sorted(
                pool,
                key=lambda row: (
                    float(row["bibliography_rank"]),
                    float(row["citation_line_fraction"]),
                    str(row["document_id"]),
                    int(row["local_start"]),
                ),
                reverse=True,
            )
            if (row["document_id"], row["local_start"], row["local_end"]) not in used
        ][:per_stratum_per_source]
        if len(narrative) != per_stratum_per_source or len(bibliography) != per_stratum_per_source:
            raise ValueError(f"{source}: insufficient distinct component candidates")
        selected.extend({**row, "selection_stratum": "citation_dense_narrative_risk"} for row in narrative)
        selected.extend({**row, "selection_stratum": "bibliography_like"} for row in bibliography)
    selected.sort(
        key=lambda row: (
            SOURCES.index(str(row["source"])),
            str(row["selection_stratum"]),
            str(row["document_id"]),
            int(row["local_start"]),
        )
    )
    return selected


def balanced_component_quota(
    rows: Sequence[Mapping[str, Any]], *, requested_per_stratum_per_source: int
) -> tuple[int, dict[str, int]]:
    """Return the largest exact two-stratum quota supported by every source."""

    source_counts = {
        source: sum(row["source"] == source for row in rows) for source in SOURCES
    }
    effective = min(
        requested_per_stratum_per_source,
        *(count // 2 for count in source_counts.values()),
    )
    if effective < 5:
        raise ValueError(
            f"too few source-balanced component candidates: {source_counts!r}"
        )
    return effective, source_counts


def build_component_packet(
    original_rows: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_id = {str(row["document_id"]): row for row in original_rows}
    cases = []
    for row in selected:
        raw = by_id[str(row["document_id"])]
        lines = raw["lines"]
        start, end = int(row["local_start"]), int(row["local_end"])
        context_start, context_end = max(0, start - 3), min(len(lines), end + 4)
        context = []
        for index, line in enumerate(lines[context_start:context_end], context_start):
            context.append(_line_payload(line, target=start <= index <= end))
        cases.append(
            {
                "case_id": f"{row['document_id']}:{row['abs_start']}-{row['abs_end']}",
                "document_id": str(row["document_id"]),
                "work_id": str(row["work_id"]),
                "source": str(row["source"]),
                "abs_start": int(row["abs_start"]),
                "abs_end": int(row["abs_end"]),
                "line_count": int(row["line_count"]),
                "selection_stratum": str(row["selection_stratum"]),
                "selection_metrics": {
                    key: round(float(row[key]), 6)
                    for key in (
                        "citation_line_fraction",
                        "hard_negative_role_fraction",
                        "entry_positive_fraction",
                        "signal_median",
                        "mean_characters",
                    )
                },
                "context": context,
            }
        )
    return {
        "schema_version": COMPONENT_PACKET_SCHEMA,
        "mode": "component",
        "title": "Bibliography component labeling",
        "interpretation": "Classify the highlighted component; the selection stratum is hidden until you decide.",
        "features": [
            {"key": spec.key, "label": spec.label, "color": spec.color}
            for spec in FEATURE_SPECS
        ],
        "cases": cases,
        "source_counts": dict(collections.Counter(row["source"] for row in cases)),
        "stratum_counts": dict(collections.Counter(row["selection_stratum"] for row in cases)),
        "label_contract": {
            "BIBLIOGRAPHY": "The highlighted component is a bibliography/reference list.",
            "CITATION_PROSE": "Narrative prose with citations, not a bibliography list.",
            "OTHER": "Neither bibliography nor citation-dense narrative prose.",
            "WEIRD": "Extraction/layout is unusable for this decision.",
        },
        "human_gold": False,
    }


REVIEW_HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Bibliography review</title><style>
:root{--ink:#242825;--muted:#687069;--paper:#f5f2e9;--panel:#fffdf8;--rule:#d9d2c3;--target:#fff0bd;--target-edge:#aa741b;--bib:#dcefdc;--non:#f5dddd;--other:#dfe8f5;--weird:#eee2f5}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.45 system-ui,-apple-system,sans-serif}.top{position:sticky;top:0;z-index:4;background:#fffdf8f2;border-bottom:1px solid var(--rule);padding:10px 18px}.top h1{font:700 20px Georgia,serif;margin:0}.sub{color:var(--muted);font-size:12px}.layout{max-width:1280px;margin:15px auto;display:grid;grid-template-columns:245px minmax(0,1fr);gap:14px;padding:0 12px}.side,.reader{background:var(--panel);border:1px solid var(--rule);border-radius:12px}.side{position:sticky;top:75px;align-self:start;padding:13px}.side input,.side button{width:100%;margin-top:7px;padding:8px;border:1px solid var(--rule);border-radius:7px;background:white}.reader{padding:14px}.casehead{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px}.line{display:grid;grid-template-columns:70px minmax(0,1fr);gap:10px;padding:6px 8px;border-left:4px solid transparent}.line.target{background:var(--target);border-left-color:var(--target-edge)}.ln{color:var(--muted);font:11px/1.7 ui-monospace,monospace;text-align:right}.text{font:16px/1.52 Georgia,serif;overflow-wrap:anywhere}.choices{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:15px}.choices button{padding:12px 5px;border:1px solid var(--rule);border-radius:8px;font-weight:700;cursor:pointer}.choices button[data-label=BIBLIOGRAPHY]{background:var(--bib)}.choices button[data-label=CITATION_PROSE],.choices button[data-label=NOT_BIB]{background:var(--non)}.choices button[data-label=OTHER],.choices button[data-label=UNSURE]{background:var(--other)}.choices button[data-label=WEIRD]{background:var(--weird)}.choices button.selected{outline:3px solid var(--ink)}.reveal{margin-top:12px;padding:10px;border-left:4px solid var(--rule);background:#f4f0e7}.hidden{display:none}.charhit{border-radius:2px;box-shadow:inset 0 -3px #0004;box-decoration-break:clone;-webkit-box-decoration-break:clone}.legend{display:flex;gap:7px;flex-wrap:wrap;margin-top:12px}.feature{font-size:11px;padding:3px 6px;border-radius:5px;background:#f1ede4}.dot{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:4px}@media(max-width:720px){.layout{display:block}.side{position:relative;top:auto;margin-bottom:10px}.choices{grid-template-columns:1fr 1fr}}
</style></head><body><header class="top"><h1 id="title"></h1><div id="subtitle" class="sub"></div></header><main class="layout"><aside class="side"><label>Reviewer<input id="reviewer" value="foivos"></label><div id="progress" class="sub"></div><button id="previous">Previous case</button><button id="next">Next undecided</button><button id="export">Export review JSON</button><p class="sub">Decisions auto-save per reviewer. The next undecided case resumes automatically.</p></aside><section class="reader"><div id="casehead" class="casehead"></div><div id="lines"></div><div id="legend" class="legend"></div><div id="choices" class="choices"></div><div id="reveal" class="reveal hidden"></div></section></main><script>
let packet,cases,index=0,answers={},reviewer='foivos',features={};const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));function storageKey(){return`bib-review-${packet.schema_version}-${reviewer}`}function load(){answers=JSON.parse(localStorage.getItem(storageKey())||'{}')}function save(){localStorage.setItem(storageKey(),JSON.stringify(answers))}function labels(){return packet.mode==='edge'?[['← BIB','BIBLIOGRAPHY'],['→ NOT BIB','NOT_BIB'],['↑ UNSURE','UNSURE'],['↓ WEIRD','WEIRD']]:[['← BIB','BIBLIOGRAPHY'],['→ CITATION PROSE','CITATION_PROSE'],['↑ OTHER','OTHER'],['↓ WEIRD','WEIRD']]}function spans(line){const out=[];for(const[key,ranges]of Object.entries(line.matches||{}))for(const[a,b]of ranges)out.push({key,a,b});return out}function highlighted(line){const hits=spans(line);if(!hits.length)return esc(line.text);const chars=Array.from(line.text),bounds=[0,chars.length,...hits.flatMap(x=>[x.a,x.b])],points=[...new Set(bounds)].sort((a,b)=>a-b);let out='';for(let i=0;i<points.length-1;i++){const a=points[i],b=points[i+1],owners=hits.filter(h=>h.a<=a&&h.b>=b),text=esc(chars.slice(a,b).join(''));if(!owners.length){out+=text;continue}const color=features[owners[0].key]?.color||'#aaa';out+=`<span class="charhit" style="background:${color}77">${text}</span>`}return out}function activeFeatures(c){const keys=new Set;for(const line of c.context)for(const key of Object.keys(line.matches||{}))keys.add(key);return[...keys]}function resultText(c,answer){if(packet.mode==='edge'){const expected=c.frozen_action==='keep'?'BIBLIOGRAPHY':'NOT_BIB',agreement=answer===expected?'agrees':'disagrees';return`Frozen edge decision: <b>${c.frozen_action.toUpperCase()}</b> · your label <b>${agreement}</b>.`}return`Selection stratum: <b>${esc(c.selection_stratum.replaceAll('_',' '))}</b>. This is selection metadata, not ground truth.`}function render(){const c=cases[index],answer=answers[c.case_id];document.getElementById('casehead').innerHTML=`<b>${esc(c.source)}</b><span>${esc(c.document_id.slice(0,12))}</span><span>${packet.mode==='edge'?`L${c.abs_idx} · ${c.document_position_percent}%`:`L${c.abs_start}–L${c.abs_end} · ${c.line_count} lines`}</span><span>case ${index+1} / ${cases.length}</span>`;document.getElementById('lines').innerHTML=c.context.map(line=>`<div class="line ${line.target?'target':''}"><span class="ln">L${line.abs_idx}</span><span class="text">${highlighted(line)}</span></div>`).join('');document.getElementById('legend').innerHTML=activeFeatures(c).map(key=>`<span class="feature"><span class="dot" style="background:${features[key].color}"></span>${esc(features[key].label)}</span>`).join('');document.getElementById('choices').innerHTML=labels().map(([label,value])=>`<button data-label="${value}" class="${answer===value?'selected':''}">${label}</button>`).join('');for(const b of document.querySelectorAll('#choices button'))b.onclick=()=>decide(b.dataset.label);const reveal=document.getElementById('reveal');reveal.classList.toggle('hidden',!answer);reveal.innerHTML=answer?resultText(c,answer):'';document.getElementById('progress').textContent=`${Object.keys(answers).length} / ${cases.length} decided`;history.replaceState(null,'',`#${encodeURIComponent(c.case_id)}`)}function nextUndecided(step=1){for(let n=1;n<=cases.length;n++){const j=(index+step*n+cases.length)%cases.length;if(!answers[cases[j].case_id]){index=j;render();return}}index=(index+step+cases.length)%cases.length;render()}function decide(label){answers[cases[index].case_id]=label;save();render();setTimeout(()=>nextUndecided(1),220)}function exportReview(){const payload={schema_version:`${packet.schema_version}-responses-v1`,packet_sha256:packet.packet_sha256,reviewer,exported_at:new Date().toISOString(),responses:Object.fromEntries(Object.entries(answers).sort())},blob=new Blob([JSON.stringify(payload,null,2)+'\n'],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`${packet.mode}-${reviewer}-review.json`;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}fetch('packet.json').then(r=>r.json()).then(p=>{packet=p;cases=p.cases;features=Object.fromEntries(p.features.map(x=>[x.key,x]));document.getElementById('title').textContent=p.title;document.getElementById('subtitle').textContent=p.interpretation;const hash=decodeURIComponent(location.hash.slice(1)),found=cases.findIndex(c=>c.case_id===hash);reviewer=new URLSearchParams(location.search).get('reviewer')||'foivos';document.getElementById('reviewer').value=reviewer;load();index=found>=0?found:Math.max(0,cases.findIndex(c=>!answers[c.case_id]));document.getElementById('reviewer').onchange=e=>{reviewer=e.target.value.trim()||'foivos';load();index=Math.max(0,cases.findIndex(c=>!answers[c.case_id]));render()};document.getElementById('previous').onclick=()=>{index=(index-1+cases.length)%cases.length;render()};document.getElementById('next').onclick=()=>nextUndecided(1);document.getElementById('export').onclick=exportReview;document.onkeydown=e=>{if(['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName))return;const map={ArrowLeft:labels()[0][1],ArrowRight:labels()[1][1],ArrowUp:labels()[2][1],ArrowDown:labels()[3][1]};if(map[e.key]){e.preventDefault();decide(map[e.key])}};render()}).catch(e=>document.body.innerHTML=`<pre>${esc(e.stack||e)}</pre>`);
</script></body></html>'''


def _with_packet_hash(packet: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {**packet, "packet_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    documents_path = Path(args.documents).resolve()
    selection_path = Path(args.selection_manifest).resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("schema_version") != "academic-structure-source-matched-holdout-manifest-v1":
        raise ValueError("unsupported source-matched selection manifest")
    if selection.get("historical", {}).get("document_count") != 2000:
        raise ValueError("holdout does not bind the complete STRUCT-2K exclusion")
    if selection.get("outputs", {}).get("documents_sha256") != sha256_file(documents_path):
        raise ValueError("holdout documents are not bound to the manifest")
    excluded_documents, excluded_works, exclusion_receipts = load_prior_exclusions(args.exclude_packet)
    holdout_rows = _read_jsonl(documents_path)
    candidate_rows = select_candidate_pool(
        holdout_rows,
        per_source=int(args.candidate_per_source),
        seed=str(args.seed),
        excluded_document_ids=excluded_documents,
        excluded_work_ids=excluded_works,
    )
    if any(
        row["document_id"] in excluded_documents or row.get("work_id") in excluded_works
        for row in candidate_rows
    ):
        raise AssertionError("fresh candidate pool overlaps a prior review packet")

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    edge_dir, component_dir = output_dir / "edge", output_dir / "components"
    edge_dir.mkdir()
    component_dir.mkdir()

    table, source_rows = materialize_inference_table(candidate_rows, workers=int(args.workers))
    source_path = output_dir / "candidate_pool.jsonl"
    _write_jsonl(source_path, source_rows)
    line_model_path = Path(args.line_model).resolve()
    line_probability = _line_probability(table, line_model_path)
    roles, role_counts = _roles(table, source_rows, workers=int(args.workers))
    signal_features = build_signal_features(line_probability, roles, table.header_kinds)

    signal_root = Path(args.signal_tcn_dir).resolve()
    signal_report_path = signal_root / "signal_tcn_oof_report.json"
    signal_report = json.loads(signal_report_path.read_text(encoding="utf-8"))
    architecture = signal_report["architecture"]
    checkpoint_architecture = {
        "hidden_dim": int(architecture["hidden_dim"]),
        "dilations": [int(value) for value in architecture["dilations"]],
        "dropout": float(architecture["dropout"]),
    }
    model_paths = sorted((signal_root / "models").glob("fold*.pt"))
    if len(model_paths) != 5:
        raise ValueError("expected five frozen signal-TCN models")
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
    base, barrier_count = decode_signal_blocks(
        table,
        signal_probability,
        line_probability,
        auxiliary_scope,
        BlockConfig(**decoder_row["config"]),
        qualified_documents=set(range(len(table.documents))),
        apply_veto=True,
    )

    frozen_edge, edge_config, refinement_provenance = load_frozen_edge(
        Path(args.refinement_dir).resolve()
    )
    edge = np.zeros(len(base), dtype=bool)
    for document_index, document in enumerate(table.documents):
        start, end = int(document["line_start"]), int(document["line_end"])
        lines = source_rows[document_index]["lines"]
        absolute = table.abs_indices[start:end]
        headers = _header_kinds(lines)
        core = _core_prediction(
            base[start:end],
            signal_probability[start:end],
            line_probability[start:end],
            headers,
            absolute,
            edge_config,
        )
        edge[start:end] = _edge_prediction(
            base[start:end], core, roles[start:end], absolute, frozen_edge
        )
    if np.any(edge & ~base):
        raise AssertionError("edge refinement is not reject-only")

    selected_documents, edge_selection = choose_edge_documents(
        table, base, edge, per_source=int(args.documents_per_source)
    )
    edge_packet = _with_packet_hash(
        build_edge_packet(
            candidate_rows,
            table,
            base,
            edge,
            selected_documents,
            selection_summary=edge_selection,
            frozen_edge=frozen_edge,
            provenance=refinement_provenance,
        )
    )
    _write_json(edge_dir / "packet.json", edge_packet)
    (edge_dir / "index.html").write_text(REVIEW_HTML, encoding="utf-8")

    all_components = _component_rows(
        candidate_rows, table, base, line_probability, signal_probability, roles
    )
    component_quota, component_pool_counts = balanced_component_quota(
        all_components,
        requested_per_stratum_per_source=int(args.components_per_stratum_per_source),
    )
    selected_components = select_component_cases(
        all_components, per_stratum_per_source=component_quota
    )
    component_packet = _with_packet_hash(
        build_component_packet(candidate_rows, selected_components)
    )
    _write_json(component_dir / "packet.json", component_packet)
    (component_dir / "index.html").write_text(REVIEW_HTML, encoding="utf-8")

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_fresh_blinded_review_packet_build",
        "freshness": {
            "struct2k_document_exclusions": 2000,
            "prior_review_document_exclusions": len(excluded_documents),
            "prior_review_work_exclusions": len(excluded_works),
            "prior_packets": exclusion_receipts,
            "overlap_with_prior_reviews": 0,
        },
        "candidate_pool": {
            "seed": str(args.seed),
            "documents": len(candidate_rows),
            "source_counts": dict(collections.Counter(row["source"] for row in candidate_rows)),
            "selection_before_model_access": True,
        },
        "edge_review": {
            "documents": len(edge_packet["documents"]),
            "cases": len(edge_packet["cases"]),
            "removed_cases": sum(row["frozen_action"] == "remove" for row in edge_packet["cases"]),
            "retained_boundary_controls": sum(row["frozen_action"] == "keep" for row in edge_packet["cases"]),
            "source_counts": edge_packet["source_counts"],
            "case_source_counts": edge_packet["case_source_counts"],
            "selection": edge_selection,
        },
        "component_review": {
            "candidate_components": len(all_components),
            "candidate_source_counts": component_pool_counts,
            "requested_per_stratum_per_source": int(
                args.components_per_stratum_per_source
            ),
            "effective_per_stratum_per_source": component_quota,
            "selected_components": len(component_packet["cases"]),
            "source_counts": component_packet["source_counts"],
            "stratum_counts": component_packet["stratum_counts"],
            "selection_strata_are_ground_truth": False,
        },
        "classifier": {
            "decoder_config": decoder_row["config"],
            "frozen_edge": frozen_edge,
            "refinement_provenance": refinement_provenance,
            "model_fitting_performed": False,
            "threshold_tuning_performed": False,
        },
        "role_counts": role_counts,
        "scope_barrier_interval_count": barrier_count,
        "inputs": {
            "documents_sha256": sha256_file(documents_path),
            "selection_manifest_sha256": sha256_file(selection_path),
            "line_model_sha256": sha256_file(line_model_path),
            "signal_report_sha256": sha256_file(signal_report_path),
            "recall_report_sha256": sha256_file(recall_report_path),
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
    artifacts = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "receipt.json":
            artifacts.append(
                {
                    "path": str(path.relative_to(output_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    _write_json(output_dir / "receipt.json", {**report, "artifacts": artifacts})
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--exclude-packet", action="append", required=True)
    parser.add_argument("--line-model", required=True)
    parser.add_argument("--signal-tcn-dir", required=True)
    parser.add_argument("--recall-block-dir", required=True)
    parser.add_argument("--refinement-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-per-source", type=int, default=30)
    parser.add_argument("--documents-per-source", type=int, default=10)
    parser.add_argument("--components-per-stratum-per-source", type=int, default=20)
    parser.add_argument("--seed", default="fresh-edge-component-review-20260714-v1")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))
