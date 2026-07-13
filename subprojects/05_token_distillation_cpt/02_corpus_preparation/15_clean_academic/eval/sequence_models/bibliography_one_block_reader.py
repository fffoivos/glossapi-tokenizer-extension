#!/usr/bin/env python3
"""Build a document reader for silver documents containing one BIB block.

The builder scans the physically test-stripped STRUCT-2K silver JSONL, retains
documents with exactly one continuous silver BIB block, and scores every
present line with the unweighted deterministic bibliography feature inventory.
Each document is stored as a deterministic gzip packet and loaded on demand by
the reader, so selecting among hundreds of documents does not load the corpus
into the browser at once.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .bibliography_block_audit import audit_document
from .bibliography_feature_explorer import FEATURE_SPECS
from .bibliography_v2 import extract_bibliography_feature_review


SCHEMA_VERSION = "bibliography-one-block-reader-v1"
DEFAULT_MAX_PHYSICAL_GAP = 64


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL row {row_number}: {error}") from error
            if not isinstance(row, Mapping):
                raise ValueError(f"JSONL row {row_number} is not an object")
            yield row


def _build_document_packet(
    row: Mapping[str, Any], max_physical_gap: int
) -> tuple[dict[str, Any], bytes, dict[str, int]] | None:
    audit = audit_document(row, max_physical_gap=max_physical_gap)
    if audit["bib_block_count"] != 1:
        return None
    document_id = str(audit["document_id"])
    raw_lines = row.get("lines")
    if not isinstance(raw_lines, list):
        raise ValueError(f"{document_id}: missing line inventory")

    feature_index = {spec.key: index for index, spec in enumerate(FEATURE_SPECS)}
    packet_lines: list[list[Any]] = []
    score_counts: collections.Counter[int] = collections.Counter()
    silver_bib_lines = 0
    for raw_line in raw_lines:
        if not isinstance(raw_line, Mapping):
            raise ValueError(f"{document_id}: reader requires materialized line objects")
        text = raw_line.get("text")
        abs_idx = raw_line.get("abs_idx")
        label = raw_line.get("label")
        if not isinstance(text, str) or not isinstance(abs_idx, int):
            raise ValueError(f"{document_id}: invalid reader line")
        if label not in {"O", "BIB", "TOC"}:
            raise ValueError(f"{document_id}: invalid line label {label!r}")
        review = extract_bibliography_feature_review(text)
        raw_features = review.features.as_dict()
        matches_by_feature: dict[str, list[list[int]]] = collections.defaultdict(list)
        for match in review.matches:
            matches_by_feature[match.feature].append([match.start, match.end])
        detected: list[list[Any]] = []
        for spec in FEATURE_SPECS:
            count = int(raw_features[spec.key])
            if count <= 0:
                continue
            ranges = matches_by_feature.get(spec.key, [])
            if not ranges:
                raise RuntimeError(
                    f"{document_id}, line {abs_idx}: {spec.key} has no review offsets"
                )
            detected.append([feature_index[spec.key], count, ranges])
        if {FEATURE_SPECS[item[0]].key for item in detected} != set(matches_by_feature):
            raise RuntimeError(f"{document_id}, line {abs_idx}: feature/match drift")
        score = len(detected)
        score_counts[score] += 1
        is_bib = int(label == "BIB")
        silver_bib_lines += is_bib
        packet_lines.append(
            [abs_idx, review.normalized_text, score, is_bib, detected]
        )

    block = audit["blocks"][0]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "document_id": document_id,
        "work_id": str(audit["work_id"]),
        "source": str(audit["source"]),
        "split": str(audit["split"]),
        "coverage": str(audit["coverage"]),
        "historical_mode": str(audit["historical_mode"]),
        "n_physical_lines": int(audit["n_physical_lines"]),
        "n_present_lines": int(audit["n_present_lines"]),
        "bib_block": block,
        "lines": packet_lines,
    }
    raw_payload = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    compressed = gzip.compress(raw_payload, compresslevel=6, mtime=0)
    filename = f"documents/{document_id}.json.gz"
    metadata = {
        "document_id": document_id,
        "document_id_short": document_id[:12],
        "work_id": str(audit["work_id"]),
        "source": str(audit["source"]),
        "split": str(audit["split"]),
        "coverage": str(audit["coverage"]),
        "n_physical_lines": int(audit["n_physical_lines"]),
        "n_present_lines": int(audit["n_present_lines"]),
        "bib_line_count": silver_bib_lines,
        "bib_start_abs_idx": int(block["start_abs_idx"]),
        "bib_end_abs_idx": int(block["end_abs_idx"]),
        "packet_path": filename,
        "packet_sha256": _sha256_bytes(compressed),
        "packet_bytes": len(compressed),
    }
    return metadata, compressed, {str(key): value for key, value in score_counts.items()}


def _write_exclusive(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _reader_page() -> str:
    return r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>One-block bibliography document reader</title>
<style>
:root{--ink:#20241f;--muted:#6f756d;--paper:#f4f0e7;--sheet:#fffdfa;--line:#d8d1c3;--silver:#c43d4b;--score:#315e4b;--shadow:0 12px 32px rgba(38,45,40,.12)}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;color:var(--ink);background:linear-gradient(135deg,#e9e4da,#f6f3ec 55%,#e3ebe5);font:14px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
button,select{font:inherit}.toolbar{position:sticky;top:0;z-index:50;background:rgba(247,244,237,.97);backdrop-filter:blur(14px);border-bottom:1px solid var(--line)}.toolbar-inner{max-width:1540px;margin:auto;padding:12px 22px;display:grid;grid-template-columns:minmax(300px,1fr) auto;gap:14px;align-items:end}.picker label{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px}.picker select{width:100%;border:1px solid var(--line);border-radius:8px;background:var(--sheet);padding:8px 10px;color:var(--ink)}.toolbar-actions{display:flex;gap:8px;align-items:center}.toolbar button{border:1px solid var(--line);border-radius:8px;background:var(--sheet);padding:8px 11px;color:var(--ink);cursor:pointer}.toolbar button:hover{background:#ece7dd}.docmeta{max-width:1540px;margin:auto;padding:7px 22px 10px;display:flex;gap:12px;flex-wrap:wrap;color:var(--muted);font-size:12px}.silver-key{display:inline-flex;gap:6px;align-items:center}.silver-key i{width:28px;border-bottom:3px solid var(--silver)}
.reader-shell{max-width:1540px;margin:20px auto 80px;padding:0 22px}.reader{background:var(--sheet);border:1px solid #ded8cd;border-radius:10px;box-shadow:var(--shadow);padding:30px 22px 50px;min-height:70vh}.reader-status{text-align:center;color:var(--muted);padding:60px 20px}.line-row{display:grid;grid-template-columns:minmax(180px,250px) minmax(0,1fr) 62px;gap:14px;align-items:start;padding:2px 0;scroll-margin-top:132px}.line-row:hover{background:rgba(233,228,218,.48)}.feature-rail{min-height:24px;display:flex;gap:4px 6px;flex-wrap:wrap;justify-content:flex-end;align-items:center;padding-top:2px}.line-no{color:#9b958b;font:10px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}.feature-label{border-left:4px solid var(--feature-color);background:color-mix(in srgb,var(--feature-color) 13%,transparent);border-radius:4px;padding:1px 5px;color:#50564f;font-size:10px;cursor:crosshair}.feature-label:hover{background:color-mix(in srgb,var(--feature-color) 23%,transparent)}.line-text{margin:0;white-space:pre-wrap;overflow-wrap:anywhere;font:16px/1.6 Georgia,"Times New Roman",serif;min-height:25px}.silver-bib .line-text{text-decoration-line:underline;text-decoration-color:var(--silver);text-decoration-thickness:2px;text-underline-offset:4px;text-decoration-skip-ink:auto}.match-box{border-radius:3px;padding:1px 0;box-decoration-break:clone;-webkit-box-decoration-break:clone;box-shadow:inset 0 -2px rgba(0,0,0,.18)}.score{color:var(--score);font:600 18px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;text-align:right;padding-top:1px}.score small{display:block;color:var(--muted);font:9px/1.2 ui-sans-serif,system-ui;text-transform:uppercase;letter-spacing:.05em}.render-progress{height:3px;position:sticky;top:111px;z-index:40;background:transparent}.render-progress i{display:block;height:100%;width:0;background:var(--score);transition:width .12s}.spotlight-dim{opacity:.16}.empty-label{color:#aaa49a;font-size:10px}.error{color:#a62f3b}.bib-start{border-top:1px dashed color-mix(in srgb,var(--silver) 55%,transparent);padding-top:8px;margin-top:8px}.bib-end{border-bottom:1px dashed color-mix(in srgb,var(--silver) 55%,transparent);padding-bottom:8px;margin-bottom:8px}
@media(max-width:980px){.line-row{grid-template-columns:150px minmax(0,1fr) 48px;gap:8px}.reader{padding-left:10px;padding-right:10px}.line-text{font-size:15px}}
@media(max-width:700px){.toolbar-inner{grid-template-columns:1fr}.docmeta{padding-top:0}.reader-shell{padding:0 6px}.reader{padding:18px 8px}.line-row{grid-template-columns:minmax(0,1fr) 42px}.feature-rail{grid-column:1/-1;justify-content:flex-start;padding-left:4px}.line-text{grid-column:1}.score{grid-column:2;grid-row:2}.line-no{order:-1;width:100%}}
@media(prefers-color-scheme:dark){:root{--ink:#ecefe9;--muted:#a8b0aa;--paper:#161a17;--sheet:#1c211d;--line:#3f4842;--silver:#ff7580;--score:#84c6a4;--shadow:0 12px 32px rgba(0,0,0,.28)}body{background:linear-gradient(135deg,#111411,#1a1f1b 55%,#151d18)}.toolbar{background:rgba(24,29,25,.97)}.picker select,.toolbar button{background:#1c211d;color:var(--ink)}.toolbar button:hover{background:#293029}.reader{border-color:#3f4842}.line-row:hover{background:rgba(255,255,255,.035)}.feature-label{color:#d7ddd8}.empty-label{color:#777f79}}
</style></head><body>
<header class="toolbar"><div class="toolbar-inner"><div class="picker"><label for="documentSelect">Document</label><select id="documentSelect" disabled><option>Loading document index…</option></select></div><div class="toolbar-actions"><button id="jumpBib" type="button" disabled>Jump to silver BIB</button><button id="jumpTop" type="button" disabled>Top</button></div></div><div id="docMeta" class="docmeta"><span>Loading manifest…</span><span class="silver-key"><i></i>red underline = existing silver BIB label</span></div><div class="render-progress"><i id="renderProgress"></i></div></header>
<main class="reader-shell"><article id="reader" class="reader"><div class="reader-status">Loading reader…</div></article></main>
<script>
const $=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let manifest=null,currentDoc=null,renderToken=0,lineNodes=new Map(),featureByIndex=[];
function formatOption(doc){return`${doc.source} · ${doc.document_id_short} · ${doc.n_present_lines.toLocaleString()} lines · BIB L${doc.bib_start_abs_idx}–${doc.bib_end_abs_idx}`}
function populateDocuments(){const select=$('documentSelect'),groups=new Map;for(const doc of manifest.documents){if(!groups.has(doc.source))groups.set(doc.source,[]);groups.get(doc.source).push(doc)}select.innerHTML='';for(const[source,docs]of groups){const group=document.createElement('optgroup');group.label=`${source} (${docs.length})`;for(const doc of docs){const option=document.createElement('option');option.value=doc.document_id;option.textContent=formatOption(doc);group.appendChild(option)}select.appendChild(group)}const params=new URLSearchParams(location.search),requested=params.get('doc'),stored=localStorage.getItem('bib-one-block-reader-doc'),chosen=manifest.documents.find(doc=>doc.document_id===requested)||manifest.documents.find(doc=>doc.document_id===stored)||manifest.documents[0];select.value=chosen.document_id;select.disabled=false;select.onchange=()=>loadDocument(select.value,true);$('jumpBib').onclick=()=>jumpToBib();$('jumpTop').onclick=()=>scrollTo({top:0,behavior:'smooth'});loadDocument(chosen.document_id,false)}
async function gunzipJson(response){if(!response.ok)throw new Error(`HTTP ${response.status}`);if(typeof DecompressionStream==='undefined')throw new Error('This browser lacks gzip stream support.');const stream=response.body.pipeThrough(new DecompressionStream('gzip'));return new Response(stream).json()}
async function loadDocument(documentId,pushHistory){const token=++renderToken,meta=manifest.documents.find(doc=>doc.document_id===documentId);if(!meta)return;$('reader').innerHTML='<div class="reader-status">Loading document…</div>';$('jumpBib').disabled=true;$('jumpTop').disabled=true;$('renderProgress').style.width='0';lineNodes.clear();try{const payload=await gunzipJson(await fetch(meta.packet_path));if(token!==renderToken)return;currentDoc=payload;localStorage.setItem('bib-one-block-reader-doc',documentId);if(pushHistory)history.replaceState(null,'',`?doc=${encodeURIComponent(documentId)}`);$('docMeta').innerHTML=`<b>${esc(meta.source)}</b><span>doc ${esc(meta.document_id_short)}</span><span>${esc(meta.split)} · ${esc(meta.coverage.replace('_',' '))}</span><span>${meta.n_present_lines.toLocaleString()} nonblank / ${meta.n_physical_lines.toLocaleString()} physical lines</span><span>silver BIB L${meta.bib_start_abs_idx}–${meta.bib_end_abs_idx} · ${meta.bib_line_count.toLocaleString()} labelled lines</span><span class="silver-key"><i></i>red underline = existing silver BIB label</span>`;$('reader').innerHTML='';$('jumpBib').disabled=false;$('jumpTop').disabled=false;renderDocument(payload,token,true)}catch(error){if(token!==renderToken)return;$('reader').innerHTML=`<div class="reader-status error">Could not load document: ${esc(error.message)}</div>`}}
function entries(line){return line[4]}
function activeSpans(line,onlyFeature=null){const spans=[];for(const[index,count,ranges]of entries(line)){if(onlyFeature!==null&&index!==onlyFeature)continue;for(const[start,end]of ranges)spans.push({index,start,end})}return spans}
function highlight(line,onlyFeature=null){const text=line[1],chars=Array.from(text),spans=activeSpans(line,onlyFeature);if(!spans.length)return esc(text);const bounds=[0,chars.length];for(const span of spans)bounds.push(span.start,span.end);const ordered=[...new Set(bounds)].sort((a,b)=>a-b);let out='';for(let i=0;i<ordered.length-1;i++){const start=ordered[i],end=ordered[i+1],piece=esc(chars.slice(start,end).join('')),hits=spans.filter(span=>span.start<=start&&span.end>=end);if(!hits.length){out+=piece;continue}const indexes=[...new Set(hits.map(hit=>hit.index))],colors=indexes.map(index=>featureByIndex[index].color+'55'),background=colors.length===1?colors[0]:`linear-gradient(to bottom,${colors.map((color,j)=>`${color} ${100*j/colors.length}% ${100*(j+1)/colors.length}%`).join(',')})`,label=indexes.map(index=>featureByIndex[index].label).join(' · ');out+=`<span class="match-box" style="background:${background}" aria-label="${esc(label)}">${piece}</span>`}return out}
function labels(line){if(!entries(line).length)return`<span class="line-no">L${line[0]}</span>`;return`<span class="line-no">L${line[0]}</span>`+entries(line).map(([index,count])=>{const feature=featureByIndex[index];return`<span class="feature-label" data-feature="${index}" style="--feature-color:${feature.color}">${esc(feature.label)}${count>1?` ×${count}`:''}</span>`}).join('')}
function lineHtml(line){const classes=['line-row'];if(line[3])classes.push('silver-bib');if(line[0]===currentDoc.bib_block.start_abs_idx)classes.push('bib-start');if(line[0]===currentDoc.bib_block.end_abs_idx)classes.push('bib-end');return`<section class="${classes.join(' ')}" id="line-${line[0]}" data-abs-idx="${line[0]}"><aside class="feature-rail">${labels(line)}</aside><p class="line-text">${highlight(line)}</p><aside class="score">${line[2]}<small>points</small></aside></section>`}
function bindLine(row,line){lineNodes.set(line[0],{row,line});for(const label of row.querySelectorAll('[data-feature]')){const index=Number(label.dataset.feature);label.onmouseenter=()=>spotlight(row,line,index);label.onmouseleave=()=>clearSpotlight(row,line);label.onfocus=()=>spotlight(row,line,index);label.onblur=()=>clearSpotlight(row,line)}}
function spotlight(row,line,index){row.querySelector('.line-text').innerHTML=highlight(line,index);for(const label of row.querySelectorAll('[data-feature]'))label.classList.toggle('spotlight-dim',Number(label.dataset.feature)!==index)}
function clearSpotlight(row,line){row.querySelector('.line-text').innerHTML=highlight(line);for(const label of row.querySelectorAll('[data-feature]'))label.classList.remove('spotlight-dim')}
function renderDocument(doc,token,autoJump){const reader=$('reader'),batch=250;let offset=0,jumped=false;const targetIndex=Math.max(0,doc.lines.findIndex(line=>line[0]===doc.bib_block.start_abs_idx));function next(){if(token!==renderToken)return;const end=Math.min(doc.lines.length,offset+batch),fragment=document.createDocumentFragment();for(let i=offset;i<end;i++){const holder=document.createElement('div');holder.innerHTML=lineHtml(doc.lines[i]);const row=holder.firstElementChild;bindLine(row,doc.lines[i]);fragment.appendChild(row)}reader.appendChild(fragment);offset=end;$('renderProgress').style.width=`${100*offset/doc.lines.length}%`;if(autoJump&&!jumped&&offset>targetIndex){jumped=true;requestAnimationFrame(()=>jumpToBib(false))}if(offset<doc.lines.length)setTimeout(next,0);else $('renderProgress').style.width='100%'}next()}
function jumpToBib(smooth=true){if(!currentDoc)return;const target=$(`line-${currentDoc.bib_block.start_abs_idx}`);if(target)target.scrollIntoView({block:'center',behavior:smooth?'smooth':'auto'})}
fetch('manifest.json').then(response=>{if(!response.ok)throw new Error(`HTTP ${response.status}`);return response.json()}).then(packet=>{manifest=packet;featureByIndex=packet.features;populateDocuments()}).catch(error=>{$('reader').innerHTML=`<div class="reader-status error">Could not load manifest: ${esc(error.message)}</div>`;$('docMeta').textContent='Reader unavailable'});
</script></body></html>'''


def run_build(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not input_path.is_file() or input_path.is_symlink():
        raise ValueError(f"input must be a regular file: {input_path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "documents").mkdir()

    workers = max(1, int(args.workers))
    metadata_rows: list[dict[str, Any]] = []
    score_counts: collections.Counter[int] = collections.Counter()
    total_input_documents = 0
    pending: set[concurrent.futures.Future[Any]] = set()

    def consume(future: concurrent.futures.Future[Any]) -> None:
        result = future.result()
        if result is None:
            return
        metadata, compressed, counts = result
        destination = output_dir / metadata["packet_path"]
        _write_exclusive(destination, compressed)
        metadata_rows.append(metadata)
        score_counts.update({int(key): value for key, value in counts.items()})

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for row in _iter_jsonl(input_path):
            total_input_documents += 1
            pending.add(
                executor.submit(_build_document_packet, row, args.max_physical_gap)
            )
            if len(pending) >= workers * 2:
                done, pending = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    consume(future)
        for future in concurrent.futures.as_completed(pending):
            consume(future)

    metadata_rows.sort(key=lambda row: (row["source"], row["document_id"]))
    if not metadata_rows:
        raise ValueError("no one-block documents found")
    feature_rows = [
        {
            "index": index,
            "key": spec.key,
            "label": spec.label,
            "group": spec.group,
            "color": spec.color,
        }
        for index, spec in enumerate(FEATURE_SPECS)
    ]
    packet_inventory = [
        [row["packet_path"], row["packet_sha256"], row["packet_bytes"]]
        for row in metadata_rows
    ]
    packet_inventory_sha = _sha256_bytes(
        json.dumps(packet_inventory, separators=(",", ":")).encode("utf-8")
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "title": "One-block bibliography document reader",
        "selection": {
            "input_path": str(input_path),
            "input_sha256": _sha256_file(input_path),
            "input_document_count": total_input_documents,
            "rule": "exactly one continuous silver BIB block",
            "selected_document_count": len(metadata_rows),
            "source_counts": dict(
                sorted(collections.Counter(row["source"] for row in metadata_rows).items())
            ),
            "coverage_counts": dict(
                sorted(collections.Counter(row["coverage"] for row in metadata_rows).items())
            ),
        },
        "scoring": {
            "rule": "one point per deterministic feature with at least one resolved match",
            "weighted_score_used": False,
            "block_decoder_used": False,
            "silver_label_display": "red underline",
            "score_histogram": [
                {"score": score, "line_count": score_counts[score]}
                for score in range(max(score_counts) + 1)
            ],
        },
        "features": feature_rows,
        "documents": metadata_rows,
        "packet_inventory_sha256": packet_inventory_sha,
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    index_bytes = _reader_page().encode("utf-8")
    _write_exclusive(output_dir / "manifest.json", manifest_bytes)
    _write_exclusive(output_dir / "index.html", index_bytes)

    receipt = {
        "schema_version": "bibliography-one-block-reader-receipt-v1",
        "status": "passed",
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "input": manifest["selection"],
        "feature_count": len(feature_rows),
        "selected_document_count": len(metadata_rows),
        "selected_present_line_count": sum(row["n_present_lines"] for row in metadata_rows),
        "selected_silver_bib_line_count": sum(row["bib_line_count"] for row in metadata_rows),
        "packet_inventory_sha256": packet_inventory_sha,
        "outputs": {
            "index": {
                "path": str((output_dir / "index.html").resolve()),
                "sha256": _sha256_file(output_dir / "index.html"),
                "bytes": len(index_bytes),
            },
            "manifest": {
                "path": str((output_dir / "manifest.json").resolve()),
                "sha256": _sha256_file(output_dir / "manifest.json"),
                "bytes": len(manifest_bytes),
            },
            "document_packets": {
                "directory": str((output_dir / "documents").resolve()),
                "count": len(metadata_rows),
                "compressed_bytes": sum(row["packet_bytes"] for row in metadata_rows),
            },
        },
    }
    _write_exclusive(
        output_dir / "receipt.json",
        (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-physical-gap", type=int, default=DEFAULT_MAX_PHYSICAL_GAP)
    parser.add_argument("--code-commit", default="")
    parser.add_argument("--slurm-job-id", default=os.environ.get("SLURM_JOB_ID", ""))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    receipt = run_build(_parser().parse_args(argv))
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

