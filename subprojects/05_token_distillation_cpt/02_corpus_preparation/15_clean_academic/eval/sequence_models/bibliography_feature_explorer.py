#!/usr/bin/env python3
"""Build a self-contained, unweighted bibliography line-feature explorer.

This is intentionally separate from the weighted bibliography-v2 scorer and
from its document-level block decoder. A feature contributes exactly one point
when its ownership-resolved count is non-zero. The browser can disable any
feature and immediately rerank the complete line inventory before displaying
it in descending, infinitely scrolling batches.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import html
import json
import os
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping, Sequence

from .bibliography_v2 import (
    BibliographyFeatures,
    extract_bibliography_feature_review,
)


SCHEMA_VERSION = "bibliography-line-feature-explorer-v5"
DEFAULT_SOURCES = ("greek_phd", "kallipos", "openarchives")
DEFAULT_SEED = "bibliography-feature-explorer-20260713"


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    label: str
    group: str
    color: str


# token_count is useful context, but it is not a detected bibliography event.
# It is shown on every card and deliberately excluded from the unit-point score.
FEATURE_SPECS = (
    FeatureSpec("year_count", "Standalone year", "Dates", "#3659a7"),
    FeatureSpec("no_date_count", "No-date marker", "Dates", "#5378c4"),
    FeatureSpec("numeric_date_count", "Numeric date", "Dates", "#6b91d4"),
    FeatureSpec("month_date_count", "Named-month date", "Dates", "#2f75a6"),
    FeatureSpec("access_date_count", "Access date", "Dates", "#3e8bb8"),
    FeatureSpec("url_count", "Non-DOI URL", "Identifiers", "#087f5b"),
    FeatureSpec("doi_count", "DOI", "Identifiers", "#0b6e4f"),
    FeatureSpec("isbn_count", "ISBN", "Identifiers", "#16896b"),
    FeatureSpec("issn_count", "ISSN", "Identifiers", "#2a9d78"),
    FeatureSpec("initial_count", "Initial", "Authors & names", "#7d3c98"),
    FeatureSpec("proper_name_word_count", "Residual proper-name word", "Authors & names", "#ae62bd"),
    FeatureSpec("inverted_author_count", "Inverted-order author", "Authors & names", "#6f3592"),
    FeatureSpec("name_initial_pair_count", "Surname + initial pair", "Authors & names", "#a155b3"),
    FeatureSpec("direct_author_count", "Direct-order author", "Authors & names", "#ba6ac3"),
    FeatureSpec("ampersand_count", "Ampersand", "Authors & names", "#7b5aa6"),
    FeatureSpec("numbered_entry_count", "Numbered entry", "Citation structure", "#b85c16"),
    FeatureSpec("quoted_span_count", "Quoted span", "Citation structure", "#cc7023"),
    FeatureSpec("editor_term_count", "Editor / translator term", "Citation structure", "#d1842f"),
    FeatureSpec("thesis_term_count", "Thesis term", "Citation structure", "#a84f20"),
    FeatureSpec("in_container_count", "Container “in”", "Citation structure", "#bd682b"),
    FeatureSpec("edition_term_count", "Edition term", "Citation structure", "#d07b37"),
    FeatureSpec("dotted_word_count", "Isolated dotted word", "Citation structure", "#a56336"),
    FeatureSpec("dotted_sequence_count", "Dotted-word sequence", "Citation structure", "#bc7848"),
    FeatureSpec("volume_marker_count", "Volume marker", "Publication details", "#a33c36"),
    FeatureSpec("volume_shape_count", "Standalone volume / issue shape", "Publication details", "#b54b43"),
    FeatureSpec("journal_year_volume_count", "Journal–year–volume", "Publication details", "#c25d51"),
    FeatureSpec("page_marker_count", "Page marker", "Publication details", "#8f3532"),
    FeatureSpec(
        "article_page_range_count",
        "Volume/article page range",
        "Publication details",
        "#9a3e39",
    ),
    FeatureSpec("page_range_count", "Page range", "Publication details", "#a94740"),
    FeatureSpec("publisher_term_count", "Residual publisher term", "Publication details", "#bc594e"),
    FeatureSpec("place_name_count", "Standalone place name", "Publication details", "#8b4a3f"),
    FeatureSpec("place_publisher_shape_count", "Place : publisher", "Publication details", "#a15e50"),
    FeatureSpec("punctuation_count", "Unclaimed punctuation", "General & counter-signals", "#59625d"),
    FeatureSpec("prose_lead_count", "Prose lead", "General & counter-signals", "#737b76"),
    FeatureSpec("table_row_count", "Markdown table row", "General & counter-signals", "#8a908c"),
)


def _validate_feature_inventory() -> None:
    raw = {field.name for field in fields(BibliographyFeatures)}
    configured = {spec.key for spec in FEATURE_SPECS}
    expected = raw - {"token_count"}
    if configured != expected:
        raise RuntimeError(
            "feature explorer inventory drift: "
            f"missing={sorted(expected - configured)}, extra={sorted(configured - expected)}"
        )


def _quota(total: int, sources: Sequence[str]) -> dict[str, int]:
    if total < 1:
        raise ValueError("document_count must be positive")
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("sources must be a non-empty unique sequence")
    base, remainder = divmod(total, len(sources))
    return {
        source: base + int(index < remainder)
        for index, source in enumerate(sources)
    }


def _rank(seed: str, source: str, document_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}\0{source}\0{document_id}".encode("utf-8")).digest(),
        "big",
    )


def _iter_jsonl(handle: BinaryIO, digest: "hashlib._Hash") -> Iterable[Mapping[str, Any]]:
    for row_number, raw_line in enumerate(handle, 1):
        digest.update(raw_line)
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid JSONL row {row_number}: {error}") from error
        if not isinstance(row, Mapping):
            raise ValueError(f"JSONL row {row_number} is not an object")
        yield row


def select_documents(
    path: str | Path,
    *,
    document_count: int,
    sources: Sequence[str],
    split: str,
    coverage: str,
    seed: str,
) -> tuple[list[Mapping[str, Any]], str, dict[str, int]]:
    """Select a deterministic source-balanced set without retaining the corpus."""

    targets = _quota(document_count, sources)
    selected: dict[str, list[tuple[int, str, Mapping[str, Any]]]] = {
        source: [] for source in sources
    }
    eligible = {source: 0 for source in sources}
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for row in _iter_jsonl(handle, digest):
            source = str(row.get("source", ""))
            if source not in selected:
                continue
            if row.get("split") != split or row.get("coverage") != coverage:
                continue
            document_id = str(row.get("document_id", ""))
            lines = row.get("lines")
            if not document_id or not isinstance(lines, list) or not lines:
                continue
            eligible[source] += 1
            rank = _rank(seed, source, document_id)
            # The negative rank makes heap[0] the worst (largest) selected rank.
            item = (-rank, document_id, row)
            heap = selected[source]
            if len(heap) < targets[source]:
                heapq.heappush(heap, item)
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, item)
    shortages = {
        source: targets[source] - len(selected[source])
        for source in sources
        if len(selected[source]) < targets[source]
    }
    if shortages:
        raise ValueError(
            f"not enough eligible {split}/{coverage} documents: {shortages}; eligible={eligible}"
        )
    documents: list[Mapping[str, Any]] = []
    for source in sources:
        ranked = sorted(selected[source], key=lambda item: (-item[0], item[1]))
        documents.extend(item[2] for item in ranked)
    return documents, digest.hexdigest(), eligible


def build_payload(
    documents: Sequence[Mapping[str, Any]],
    *,
    input_path: str,
    input_sha256: str,
    eligible_counts: Mapping[str, int],
    split: str,
    coverage: str,
    seed: str,
) -> dict[str, Any]:
    """Extract ownership-resolved counts without annotation/model labels."""

    _validate_feature_inventory()
    feature_stats = {
        spec.key: {"line_count": 0, "occurrence_count": 0} for spec in FEATURE_SPECS
    }
    line_rows: list[dict[str, Any]] = []
    doc_rows: list[dict[str, Any]] = []
    seen_documents: set[str] = set()
    ordinal = 0
    for document in documents:
        document_id = str(document.get("document_id", ""))
        source = str(document.get("source", ""))
        if not document_id or document_id in seen_documents:
            raise ValueError(f"empty or duplicate document_id {document_id!r}")
        seen_documents.add(document_id)
        n_physical = int(document.get("n_physical_lines", 0))
        raw_lines = document.get("lines")
        if n_physical < 1 or not isinstance(raw_lines, list) or not raw_lines:
            raise ValueError(f"document {document_id}: invalid line inventory")
        present = 0
        for raw_line in raw_lines:
            if not isinstance(raw_line, Mapping):
                raise ValueError(f"document {document_id}: line is not an object")
            text = raw_line.get("text")
            abs_idx = raw_line.get("abs_idx")
            if not isinstance(text, str) or not text.strip() or not isinstance(abs_idx, int):
                continue
            review = extract_bibliography_feature_review(text)
            extracted = review.features
            raw_features = extracted.as_dict()
            detected = {
                spec.key: int(raw_features[spec.key])
                for spec in FEATURE_SPECS
                if int(raw_features[spec.key]) > 0
            }
            for key, count in detected.items():
                feature_stats[key]["line_count"] += 1
                feature_stats[key]["occurrence_count"] += count
            match_offsets: dict[str, list[list[int]]] = {}
            for match in review.matches:
                match_offsets.setdefault(match.feature, []).append(
                    [match.start, match.end]
                )
            if set(match_offsets) != set(detected):
                raise RuntimeError(
                    f"document {document_id}, line {abs_idx}: "
                    "review matches do not cover the detected feature inventory"
                )
            line_rows.append(
                {
                    "ordinal": ordinal,
                    "document_id": document_id,
                    "source": source,
                    "abs_idx": abs_idx,
                    "position_percent": round(100 * abs_idx / max(1, n_physical - 1), 2),
                    "text": review.normalized_text,
                    "char_length": len(review.normalized_text),
                    "token_count": extracted.token_count,
                    "features": detected,
                    "matches": match_offsets,
                }
            )
            ordinal += 1
            present += 1
        doc_rows.append(
            {
                "document_id": document_id,
                "document_id_short": document_id[:12],
                "work_id": str(document.get("work_id", "")),
                "source": source,
                "n_physical_lines": n_physical,
                "n_present_lines": present,
            }
        )
    if len(line_rows) < 100:
        raise ValueError(f"only {len(line_rows)} usable lines; at least 100 are required")
    features = []
    for spec in FEATURE_SPECS:
        features.append(
            {
                "key": spec.key,
                "label": spec.label,
                "group": spec.group,
                "color": spec.color,
                **feature_stats[spec.key],
            }
        )
    payload_core = {
        "schema_version": SCHEMA_VERSION,
        "title": "Bibliography line feature explorer",
        "selection": {
            "input_path": input_path,
            "input_sha256": input_sha256,
            "split": split,
            "coverage": coverage,
            "seed": seed,
            "document_count": len(doc_rows),
            "source_counts": {
                source: sum(row["source"] == source for row in doc_rows)
                for source in sorted({row["source"] for row in doc_rows})
            },
            "eligible_counts": dict(sorted(eligible_counts.items())),
        },
        "scoring": {
            "rule": "one point for each enabled feature with a resolved occurrence count greater than zero",
            "ownership": "specific lexical and numeric spans suppress broader fallback detectors",
            "count_magnitude_used": False,
            "weighted_score_used": False,
            "block_decoder_used": False,
            "token_count_scored": False,
            "review_offsets": "Unicode character offsets into the displayed NFKC text",
            "diagnostics": [
                "matched_character_coverage",
                "matches_per_100_characters",
                "feature_points_per_100_characters",
            ],
        },
        "features": features,
        "documents": doc_rows,
        "lines": line_rows,
    }
    payload_sha = hashlib.sha256(
        json.dumps(
            payload_core, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {**payload_core, "payload_sha256": payload_sha}


def _json_for_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )


def build_page(payload: Mapping[str, Any]) -> str:
    title = html.escape(str(payload["title"]))
    data = _json_for_script(payload)
    template = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--ink:#17201c;--muted:#6c756f;--paper:#eee9df;--card:#fffdf8;--line:#d8d1c3;--accent:#294f3e;--shadow:0 14px 38px rgba(38,45,40,.10)}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:linear-gradient(145deg,#e9e3d7,#f8f5ed 48%,#e4ece7);color:var(--ink);font:15px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}button,input{font:inherit}
.top{position:sticky;top:0;z-index:30;background:rgba(249,246,239,.96);backdrop-filter:blur(16px);border-bottom:1px solid var(--line)}.topin{max-width:1500px;margin:auto;padding:15px 24px}.titleline{display:flex;align-items:end;justify-content:space-between;gap:18px}.titleline h1{font:700 25px/1.1 Georgia,serif;margin:0}.sub{font-size:12px;color:var(--muted)}.headline-score{text-align:right}.headline-score b{display:block;font:700 24px/1 Georgia,serif}.headline-score span{font-size:11px;color:var(--muted)}
.layout{max-width:1500px;margin:20px auto 80px;padding:0 24px;display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:18px;align-items:start}.feed{display:flex;flex-direction:column;gap:12px}.card{display:grid;grid-template-columns:96px minmax(0,1fr);background:var(--card);border:1px solid #ddd6ca;border-radius:15px;box-shadow:var(--shadow);overflow:hidden;scroll-margin:105px}.scorebox{background:#eee9df;border-right:1px solid var(--line);padding:15px 8px;text-align:center}.rank{font-size:11px;color:var(--muted);font-weight:750}.score{font:700 34px/1 Georgia,serif;margin-top:13px;color:var(--accent)}.scorelabel{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;overflow-wrap:anywhere}.content{padding:15px 18px 17px}.meta{display:flex;gap:9px;align-items:center;flex-wrap:wrap;color:var(--muted);font-size:12px}.meta b{color:#405148}.metric{background:#ece7dc;border-radius:6px;padding:2px 6px;color:#405148}.posbar{width:95px;height:5px;background:#e1ddd3;border-radius:8px;overflow:hidden}.posbar i{display:block;height:100%;background:#527963}.context{margin:11px 0 13px;display:flex;flex-direction:column;gap:3px}.contextline{display:grid;grid-template-columns:57px minmax(0,1fr);gap:10px;padding:6px 10px;border-radius:8px;font:14px/1.4 Georgia,"Times New Roman",serif;overflow-wrap:anywhere;color:#68706b}.contextline .ln{font:10px/1.9 ui-monospace,SFMono-Regular,Menlo,monospace;color:#979187;text-align:right}.contextline.focus{background:#fff8df;color:var(--ink);box-shadow:inset 4px 0 #b47b19;font-size:17px}.charhit{border-radius:3px;padding:1px 0;box-decoration-break:clone;-webkit-box-decoration-break:clone;box-shadow:inset 0 -2px rgba(0,0,0,.24)}.badges{display:flex;gap:6px;flex-wrap:wrap}.badge{padding:5px 8px;border-radius:7px;color:white;font-size:11px;font-weight:750;box-shadow:inset 0 -1px rgba(0,0,0,.14);cursor:crosshair;transition:opacity .12s,transform .12s}.badge:hover,.spotlight-active{transform:translateY(-1px)}.badge small{display:block;opacity:.82;font:9px/1.2 ui-monospace,SFMono-Regular,Menlo,monospace;margin-top:2px}.spotlight-dim{opacity:.23!important}.empty{font-size:12px;color:var(--muted);font-style:italic}.loadsentinel{text-align:center;padding:22px;color:var(--muted);font-size:12px}
.filters{position:sticky;top:105px;background:rgba(255,253,248,.97);border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow);overflow:hidden;max-height:calc(100vh - 125px);display:flex;flex-direction:column}.filterhead{padding:15px;border-bottom:1px solid var(--line)}.filterhead h2{font:700 19px/1.1 Georgia,serif;margin:0 0 5px}.rankselect{display:block;width:100%;margin-top:10px;border:1px solid var(--line);border-radius:8px;background:#fffdf8;padding:7px 8px;color:var(--ink);font-size:12px}.actions{display:flex;gap:6px;margin-top:9px}.actions button{border:1px solid var(--line);background:#f4f0e7;border-radius:8px;padding:6px 9px;cursor:pointer;font-size:11px}.actions button:hover{background:#e9e3d8}.filterlist{padding:9px 12px 15px;overflow:auto}.group h3{margin:13px 3px 6px;font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}.filter{display:grid;grid-template-columns:17px 11px minmax(0,1fr);gap:7px;align-items:start;padding:5px 3px;border-radius:7px;cursor:pointer}.filter:hover{background:#f2eee5}.filter input{margin:2px 0}.dot{width:10px;height:10px;border-radius:3px;margin-top:3px}.filtertext{font-size:12px;line-height:1.25}.filtertext small{display:block;color:var(--muted);font-size:9px;margin-top:2px}.footer-note{padding:10px 15px;border-top:1px solid var(--line);font-size:10px;color:var(--muted);background:#f4f0e7}
@media(max-width:1050px){.layout{grid-template-columns:minmax(0,1fr) 250px;padding:0 10px}.filters{top:116px}.contextline{grid-template-columns:43px 1fr}.titleline{align-items:start}.topin{padding:12px}.headline-score b{font-size:19px}}
@media(max-width:760px){.layout{display:flex;flex-direction:column-reverse}.filters{position:relative;top:auto;width:100%;max-height:none}.filterlist{max-height:360px}.card{grid-template-columns:62px 1fr}.scorebox{padding:12px 5px}.score{font-size:30px}.content{padding:12px}.headline-score{display:none}}
</style></head><body>
<header class="top"><div class="topin"><div class="titleline"><div><h1>__TITLE__</h1><div id="subtitle" class="sub"></div></div><div class="headline-score"><b id="activeCount"></b><span>active detector features</span></div></div></div></header>
<main class="layout"><section id="feed" class="feed"></section><aside class="filters"><div class="filterhead"><h2>Feature menu</h2><div id="filterSummary" class="sub"></div><select id="ranking" class="rankselect" aria-label="Ranking metric"><option value="points">Rank: feature points</option><option value="match_density">Rank: matches / 100 chars</option><option value="coverage">Rank: matched-character coverage</option><option value="point_density">Rank: points / 100 chars</option></select><div class="actions"><button id="all">Enable all</button><button id="none">Clear all</button><button id="reset">Reset view</button></div></div><div id="filterList" class="filterlist"></div><div class="footer-note">Specific lexical and numeric detectors own their spans before broad fallback detectors. Results remain in descending order and load automatically in batches of 100. Hover a coloured badge or sidebar feature label to isolate it. Each checked, non-zero feature adds one point.</div></aside></main>
<script>const PACKET=__DATA__;
const $=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const featureByKey=Object.fromEntries(PACKET.features.map(f=>[f.key,f])),docById=Object.fromEntries(PACKET.documents.map(d=>[d.document_id,d])),lineByOrdinal=Object.fromEntries(PACKET.lines.map(line=>[line.ordinal,line]));
const linesByDoc={};for(const line of PACKET.lines)(linesByDoc[line.document_id]??=[]).push(line);for(const lines of Object.values(linesByDoc))lines.sort((a,b)=>a.abs_idx-b.abs_idx);
const lineIndex=new Map;for(const lines of Object.values(linesByDoc))lines.forEach((line,i)=>lineIndex.set(line.document_id+'|'+line.abs_idx,i));
const storageKey='bib-feature-explorer:'+PACKET.payload_sha256;let stored={};try{stored=JSON.parse(localStorage.getItem(storageKey)||'{}')}catch{};
let enabled=Object.fromEntries(PACKET.features.map(f=>[f.key,stored.features?.[f.key]!==false])),ranking=stored.ranking||'points',rankedLines=[],renderedCount=0;
const batchSize=100,scrollObserver=new IntersectionObserver(entries=>{if(entries.some(entry=>entry.isIntersecting))appendNextBatch()},{rootMargin:'1200px 0px'});
function activeKeys(){return new Set(PACKET.features.filter(f=>enabled[f.key]).map(f=>f.key))}
function score(line,active){let n=0;for(const key of Object.keys(line.features))if(active.has(key))n++;return n}
function activeSpans(line,active){const spans=[];for(const [key,ranges] of Object.entries(line.matches))if(active.has(key))for(const [start,end] of ranges)spans.push({key,start,end});return spans}
function metrics(line,active){const spans=activeSpans(line,active),points=score(line,active),length=Math.max(1,line.char_length);let covered=0,last=-1;for(const span of [...spans].sort((a,b)=>a.start-b.start||a.end-b.end)){const start=Math.max(last,span.start);if(span.end>start)covered+=span.end-start;last=Math.max(last,span.end)}return{points,match_count:spans.length,coverage:covered/length,match_density:100*spans.length/length,point_density:100*points/length}}
function context(line){const lines=linesByDoc[line.document_id],i=lineIndex.get(line.document_id+'|'+line.abs_idx);let start=Math.max(0,i-2),end=Math.min(lines.length,start+5);start=Math.max(0,end-5);return lines.slice(start,end)}
function highlight(line,active,only=null){const shown=only&&active.has(only)?new Set([only]):active,chars=Array.from(line.text),spans=activeSpans(line,shown);if(!spans.length)return esc(line.text);const bounds=[0,chars.length];for(const span of spans)bounds.push(span.start,span.end);const ordered=[...new Set(bounds)].sort((a,b)=>a-b);let out='';for(let i=0;i<ordered.length-1;i++){const start=ordered[i],end=ordered[i+1],text=esc(chars.slice(start,end).join('')),hits=spans.filter(span=>span.start<=start&&span.end>=end);if(!hits.length){out+=text;continue}const keys=[...new Set(hits.map(hit=>hit.key))],colors=keys.map(key=>featureByKey[key].color+'55'),background=colors.length===1?colors[0]:`linear-gradient(to bottom,${colors.map((color,j)=>`${color} ${100*j/colors.length}% ${100*(j+1)/colors.length}%`).join(',')})`,title=keys.map(key=>`${featureByKey[key].label} [${start}:${end}]`).join(' · ');out+=`<span class="charhit" style="background:${background}" title="${esc(title)}">${text}</span>`}return out}
function contextHtml(line,active){return context(line).map(other=>{const focus=other.abs_idx===line.abs_idx;return`<div class="contextline ${focus?'focus':''}"><span class="ln">L${other.abs_idx}</span><span ${focus?`class="focuscopy" data-focus-line="${line.ordinal}"`:''}>${focus?highlight(line,active):esc(other.text)}</span></div>`}).join('')}
function offsets(ranges){const visible=ranges.slice(0,6).map(([a,b])=>`${a}:${b}`).join(' · ');return ranges.length>6?visible+` · +${ranges.length-6} more`:visible}
function badges(line,active){const keys=Object.keys(line.features).filter(key=>active.has(key));if(!keys.length)return'<span class="empty">No enabled feature fires on this line.</span>';return keys.map(key=>{const f=featureByKey[key],ranges=line.matches[key],all=ranges.map(([a,b])=>`${a}:${b}`).join(' · ');return`<span class="badge" data-spotlight="${esc(key)}" tabindex="0" style="background:${f.color}" title="Hover to isolate · ${esc(f.key+' @ '+all)}">${esc(f.label)}<small>${esc(offsets(ranges))}</small></span>`}).join('')}
function metricValue(m){return ranking==='points'?m.points:ranking==='coverage'?100*m.coverage:ranking==='match_density'?m.match_density:m.point_density}
function metricDisplay(m){if(ranking==='points')return[String(m.points),'feature points'];if(ranking==='coverage')return[(100*m.coverage).toFixed(1)+'%','char coverage'];if(ranking==='match_density')return[m.match_density.toFixed(1),'matches / 100 chars'];return[m.point_density.toFixed(1),'points / 100 chars']}
function card(line,rank,active,m){const doc=docById[line.document_id],[value,label]=metricDisplay(m);return`<article class="card"><aside class="scorebox"><div class="rank">#${rank}</div><div class="score">${value}</div><div class="scorelabel">${label}</div></aside><section class="content"><div class="meta"><b>${esc(line.source)}</b><span>doc ${esc(doc.document_id_short)}</span><span>line ${line.abs_idx}</span><span>${line.position_percent.toFixed(1)}% through document</span><span class="posbar"><i style="width:${line.position_percent}%"></i></span><span>${line.token_count} tokens · ${line.char_length} chars</span><span class="metric">${m.match_count} matches</span><span class="metric">${(100*m.coverage).toFixed(1)}% matched chars</span><span class="metric">${m.match_density.toFixed(1)} matches / 100 chars</span></div><div class="context">${contextHtml(line,active)}</div><div class="badges">${badges(line,active)}</div></section></article>`}
function spotlight(key){const active=activeKeys();if(!active.has(key))return;for(const node of document.querySelectorAll('[data-focus-line]'))node.innerHTML=highlight(lineByOrdinal[node.dataset.focusLine],active,key);for(const node of document.querySelectorAll('[data-spotlight]')){node.classList.remove('spotlight-active','spotlight-dim');node.classList.add(node.dataset.spotlight===key?'spotlight-active':'spotlight-dim')}}
function clearSpotlight(){const active=activeKeys();for(const node of document.querySelectorAll('[data-focus-line]'))node.innerHTML=highlight(lineByOrdinal[node.dataset.focusLine],active);for(const node of document.querySelectorAll('[data-spotlight]'))node.classList.remove('spotlight-active','spotlight-dim')}
function bindSpotlights(root=document){for(const node of root.querySelectorAll('[data-spotlight]:not([data-spotlight-bound])')){node.dataset.spotlightBound='1';node.onmouseenter=()=>spotlight(node.dataset.spotlight);node.onmouseleave=clearSpotlight;node.onfocus=()=>spotlight(node.dataset.spotlight);node.onblur=clearSpotlight}}
function updateFeedSummary(){const active=activeKeys();$('filterSummary').textContent=`${active.size} enabled · showing ${renderedCount.toLocaleString()} / ${rankedLines.length.toLocaleString()} · descending`}
function appendNextBatch(){const feed=$('feed'),old=$('scrollSentinel');if(old){scrollObserver.unobserve(old);old.remove()}if(renderedCount>=rankedLines.length){updateFeedSummary();return}const active=activeKeys(),end=Math.min(renderedCount+batchSize,rankedLines.length),batch=rankedLines.slice(renderedCount,end);feed.insertAdjacentHTML('beforeend',batch.map((x,i)=>card(x.line,renderedCount+i+1,active,x.m)).join(''));renderedCount=end;bindSpotlights(feed);if(renderedCount<rankedLines.length){const sentinel=document.createElement('div');sentinel.id='scrollSentinel';sentinel.className='loadsentinel';sentinel.textContent=`Loading the next ${Math.min(batchSize,rankedLines.length-renderedCount).toLocaleString()} lines…`;feed.appendChild(sentinel);scrollObserver.observe(sentinel)}updateFeedSummary()}
function renderFeed(){scrollObserver.disconnect();const active=activeKeys();rankedLines=PACKET.lines.map(line=>{const m=metrics(line,active);return{line,m,value:metricValue(m)}}).sort((a,b)=>b.value-a.value||b.m.points-a.m.points||a.line.ordinal-b.line.ordinal);renderedCount=0;$('feed').innerHTML='';$('activeCount').textContent=active.size+' / '+PACKET.features.length;appendNextBatch()}
function renderFilters(){const groups=[];for(const f of PACKET.features){let group=groups.find(x=>x.name===f.group);if(!group){group={name:f.group,features:[]};groups.push(group)}group.features.push(f)}$('filterList').innerHTML=groups.map(group=>`<section class="group"><h3>${esc(group.name)}</h3>${group.features.map(f=>`<label class="filter" data-spotlight="${esc(f.key)}"><input type="checkbox" data-feature="${esc(f.key)}" ${enabled[f.key]?'checked':''}><span class="dot" style="background:${f.color}"></span><span class="filtertext">${esc(f.label)}<small>${f.line_count.toLocaleString()} lines · ${f.occurrence_count.toLocaleString()} occurrences</small></span></label>`).join('')}</section>`).join('');bindSpotlights()}
function persist(){localStorage.setItem(storageKey,JSON.stringify({features:enabled,ranking}))}
function setAll(value){for(const f of PACKET.features)enabled[f.key]=value;persist();renderFilters();renderFeed()}
$('ranking').value=ranking;$('ranking').onchange=e=>{ranking=e.target.value;persist();renderFeed()};$('filterList').onchange=e=>{const box=e.target.closest('[data-feature]');if(!box)return;enabled[box.dataset.feature]=box.checked;persist();renderFeed()};$('all').onclick=()=>setAll(true);$('none').onclick=()=>setAll(false);$('reset').onclick=()=>{localStorage.removeItem(storageKey);enabled=Object.fromEntries(PACKET.features.map(f=>[f.key,true]));ranking='points';$('ranking').value=ranking;renderFilters();renderFeed();window.scrollTo({top:0,behavior:'smooth'})};
const counts=Object.entries(PACKET.selection.source_counts).map(([k,v])=>`${k}: ${v}`).join(' · ');$('subtitle').textContent=`${PACKET.selection.document_count} real ${PACKET.selection.split} / ${PACKET.selection.coverage.replace('_',' ')} documents · ${PACKET.lines.length.toLocaleString()} nonblank lines · ${counts}`;renderFilters();renderFeed();
</script></body></html>'''
    return template.replace("__TITLE__", title).replace("__DATA__", data)


def documents_from_site(
    path: str | Path,
) -> tuple[list[dict[str, Any]], Mapping[str, Any], str]:
    """Recover the exact review sample from a previously built explorer.

    This supports cheap local UI/feature iterations without rescanning the
    original corpus. The prior site already contains every displayed document
    line and the source selection provenance, but never annotation labels.
    """

    source = Path(path)
    raw = source.read_bytes()
    page = raw.decode("utf-8")
    match = re.search(r"<script>const PACKET=(.*?);\nconst \$=", page, re.S)
    if match is None:
        raise ValueError(f"{source}: embedded explorer packet not found")
    packet = json.loads(match.group(1))
    if not isinstance(packet, Mapping) or not str(packet.get("schema_version", "")).startswith(
        "bibliography-line-feature-explorer-v"
    ):
        raise ValueError(f"{source}: unsupported explorer packet")
    selection = packet.get("selection")
    document_rows = packet.get("documents")
    line_rows = packet.get("lines")
    if (
        not isinstance(selection, Mapping)
        or not isinstance(document_rows, list)
        or not isinstance(line_rows, list)
    ):
        raise ValueError(f"{source}: incomplete explorer packet")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for line in line_rows:
        if not isinstance(line, Mapping):
            raise ValueError(f"{source}: invalid line row")
        document_id = str(line.get("document_id", ""))
        text = line.get("text")
        abs_idx = line.get("abs_idx")
        if not document_id or not isinstance(text, str) or not isinstance(abs_idx, int):
            raise ValueError(f"{source}: invalid line identity")
        grouped.setdefault(document_id, []).append(
            {"abs_idx": abs_idx, "text": text}
        )
    documents: list[dict[str, Any]] = []
    for row in document_rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{source}: invalid document row")
        document_id = str(row.get("document_id", ""))
        lines = sorted(grouped.pop(document_id, []), key=lambda line: line["abs_idx"])
        if not document_id or not lines:
            raise ValueError(f"{source}: missing lines for document {document_id!r}")
        documents.append(
            {
                "document_id": document_id,
                "work_id": str(row.get("work_id", "")),
                "source": str(row.get("source", "")),
                "split": str(selection.get("split", "")),
                "coverage": str(selection.get("coverage", "")),
                "n_physical_lines": int(row.get("n_physical_lines", 0)),
                "lines": lines,
            }
        )
    if grouped:
        raise ValueError(f"{source}: lines refer to unknown documents")
    return documents, selection, hashlib.sha256(raw).hexdigest()


def _exclusive_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def run_build(args: argparse.Namespace) -> dict[str, Any]:
    rebuild_source: dict[str, Any] | None = None
    if args.input_site:
        documents, previous_selection, site_sha = documents_from_site(args.input_site)
        input_path = str(previous_selection["input_path"])
        input_sha = str(previous_selection["input_sha256"])
        eligible = dict(previous_selection["eligible_counts"])
        split = str(previous_selection["split"])
        coverage = str(previous_selection["coverage"])
        seed = str(previous_selection["seed"])
        rebuild_source = {
            "path": str(Path(args.input_site).resolve()),
            "sha256": site_sha,
        }
    else:
        sources = tuple(part.strip() for part in args.sources.split(",") if part.strip())
        documents, input_sha, eligible = select_documents(
            args.input,
            document_count=args.document_count,
            sources=sources,
            split=args.split,
            coverage=args.coverage,
            seed=args.seed,
        )
        input_path = str(Path(args.input).resolve())
        split, coverage, seed = args.split, args.coverage, args.seed
    payload = build_payload(
        documents,
        input_path=input_path,
        input_sha256=input_sha,
        eligible_counts=eligible,
        split=split,
        coverage=coverage,
        seed=seed,
    )
    output = Path(args.output)
    receipt = Path(args.receipt) if args.receipt else output.with_suffix(".receipt.json")
    _exclusive_write(output, build_page(payload))
    receipt_data = {
        "schema_version": "bibliography-line-feature-explorer-receipt-v5",
        "status": "passed",
        "payload_sha256": payload["payload_sha256"],
        "input": {
            "path": payload["selection"]["input_path"],
            "sha256": input_sha,
        },
        "selection": payload["selection"],
        "feature_count": len(payload["features"]),
        "line_count": len(payload["lines"]),
        "output": {
            "path": str(output.resolve()),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "bytes": output.stat().st_size,
        },
    }
    if rebuild_source is not None:
        receipt_data["rebuild_source_site"] = rebuild_source
    try:
        _exclusive_write(receipt, json.dumps(receipt_data, ensure_ascii=False, indent=2) + "\n")
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return receipt_data


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", help="STRUCT-2K JSONL document inventory")
    inputs.add_argument(
        "--input-site",
        help="existing explorer whose exact label-blind sample should be reused",
    )
    parser.add_argument("--output", required=True, help="new self-contained HTML output")
    parser.add_argument("--receipt", help="new receipt path (default: OUTPUT.receipt.json)")
    parser.add_argument("--document-count", type=int, default=20)
    parser.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    parser.add_argument("--split", default="train")
    parser.add_argument("--coverage", default="full_document")
    parser.add_argument("--seed", default=DEFAULT_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = run_build(args)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
