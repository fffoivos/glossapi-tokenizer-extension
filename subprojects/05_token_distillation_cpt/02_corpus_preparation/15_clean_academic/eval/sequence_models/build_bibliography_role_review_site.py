#!/usr/bin/env python3
"""Build a self-contained, prediction-blind-first bibliography role audit site."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .bibliography_role_adjudication import collect_votes, load_review
from .bibliography_role_review_runner import load_packet
from .contract import canonical_json_sha256, sha256_file


SITE_SCHEMA = "bibliography-role-human-audit-site-v1"


def _consensus(values: Sequence[str]) -> str:
    unique = set(values)
    return next(iter(unique)) if len(unique) == 1 else "UNRESOLVED"


def select_blocks(
    packet: Mapping[str, Any], review_a: Mapping[str, Any], review_b: Mapping[str, Any],
    *, per_source: int, seed: str,
) -> list[str]:
    if per_source <= 0:
        raise ValueError("per_source must be positive")
    votes_a, votes_b = collect_votes(review_a, packet), collect_votes(review_b, packet)
    case_by_id = {str(case["case_id"]): case for case in packet["cases"]}
    block_cases: dict[str, list[Mapping[str, Any]]] = {}
    for case in packet["cases"]:
        block_cases.setdefault(str(case["block_case_id"]), []).append(case)
    rankings: dict[str, list[tuple[tuple[Any, ...], str]]] = {}
    for block_id, cases in block_cases.items():
        source = str(cases[0]["source"])
        keys = {
            (str(case["document_id"]), str(line["line_id"]))
            for case in cases for line in case["lines"]
        }
        role_disagreements = boundary_disagreements = boundary_votes = rare_roles = 0
        minimum_confidence = 1.0
        for key in keys:
            a, b = votes_a[key], votes_b[key]
            role_a, role_b = _consensus(a["roles"]), _consensus(b["roles"])
            boundary_a, boundary_b = _consensus(a["boundaries"]), _consensus(b["boundaries"])
            role_disagreements += int(role_a != role_b)
            boundary_disagreements += int(boundary_a != boundary_b)
            boundary_votes += int(boundary_a != "NONE" or boundary_b != "NONE")
            rare_roles += int(role_a in {"CONTINUATION", "FILLER"} or role_b in {"CONTINUATION", "FILLER"})
            minimum_confidence = min(minimum_confidence, *(a["confidences"] + b["confidences"]))
        rank_hash = hashlib.sha256(f"{seed}\0{source}\0{block_id}".encode()).hexdigest()
        priority = (
            -int(role_disagreements > 0),
            -int(boundary_disagreements > 0),
            -int(rare_roles > 0),
            -int(boundary_votes > 0),
            -role_disagreements,
            -boundary_disagreements,
            minimum_confidence,
            rank_hash,
        )
        rankings.setdefault(source, []).append((priority, block_id))
    selected: list[str] = []
    for source in sorted(rankings):
        local = sorted(rankings[source])
        if len(local) < per_source:
            raise ValueError(f"{source}: only {len(local)} blocks for {per_source}-block audit")
        selected.extend(block_id for _, block_id in local[:per_source])
    return selected


def build_payload(
    packet: Mapping[str, Any], review_a: Mapping[str, Any], review_b: Mapping[str, Any],
    *, selected_blocks: Sequence[str], packet_sha256: str,
) -> dict[str, Any]:
    votes_a, votes_b = collect_votes(review_a, packet), collect_votes(review_b, packet)
    selected = set(selected_blocks)
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for case in packet["cases"]:
        block_id = str(case["block_case_id"])
        if block_id in selected:
            grouped.setdefault(block_id, []).append(case)
    blocks: list[dict[str, Any]] = []
    for block_id in selected_blocks:
        cases = grouped[block_id]
        first = cases[0]
        lines: dict[str, dict[str, Any]] = {}
        for case in cases:
            for line in case["lines"]:
                key = (str(case["document_id"]), str(line["line_id"]))
                row = {
                    "line_id": str(line["line_id"]),
                    "abs_idx": int(line["abs_idx"]),
                    "position_percent": float(line["document_position_percent"]),
                    "text": str(line["text"]),
                    "pass_a_role": _consensus(votes_a[key]["roles"]),
                    "pass_b_role": _consensus(votes_b[key]["roles"]),
                    "pass_a_boundary": _consensus(votes_a[key]["boundaries"]),
                    "pass_b_boundary": _consensus(votes_b[key]["boundaries"]),
                    "pass_a_confidence": min(votes_a[key]["confidences"]),
                    "pass_b_confidence": min(votes_b[key]["confidences"]),
                }
                previous = lines.get(row["line_id"])
                if previous is not None and previous != row:
                    raise ValueError(f"overlap payload conflict for {row['line_id']}")
                lines[row["line_id"]] = row
        blocks.append(
            {
                "block_case_id": block_id,
                "document_id": str(first["document_id"]),
                "work_id": str(first["work_id"]),
                "source": str(first["source"]),
                "n_physical_lines": int(first["n_physical_lines"]),
                "lines": sorted(lines.values(), key=lambda row: (row["abs_idx"], row["line_id"])),
            }
        )
    return {
        "schema_version": SITE_SCHEMA,
        "packet_sha256": packet_sha256,
        "reviewers": [str(review_a["reviewer"]), str(review_b["reviewer"])],
        "selection": {
            "block_count": len(blocks),
            "source_counts": {
                source: sum(block["source"] == source for block in blocks)
                for source in sorted({block["source"] for block in blocks})
            },
            "priority": "role disagreement, boundary disagreement, continuation/filler, boundaries, uncertainty, deterministic fill",
        },
        "blocks": blocks,
    }


def _script_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def _page(payload: Mapping[str, Any], title: str) -> str:
    data, safe_title = _script_json(payload), html.escape(title)
    return """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<style>
:root{--paper:#f3efe6;--card:#fffdf8;--ink:#18201c;--muted:#6a746e;--line:#d8d1c4;--entry:#a33c36;--cont:#c66b32;--fill:#9a7528;--head:#28628f;--sub:#4f78a0;--non:#58615c;--unk:#7b5894;--good:#24714b;--bad:#a33c36;--shadow:0 14px 40px #26352b17}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:linear-gradient(140deg,#ebe4d7,#f8f5ed 52%,#e4ece7);color:var(--ink);font:14px/1.45 Inter,system-ui,sans-serif}button{font:inherit;cursor:pointer}.top{position:sticky;top:0;z-index:20;background:#faf7f0f5;border-bottom:1px solid var(--line);backdrop-filter:blur(14px);padding:12px 18px}.topin{max-width:1500px;margin:auto;display:flex;align-items:center;gap:14px;flex-wrap:wrap}.top h1{font:700 22px Georgia,serif;margin:0 auto 0 0}.progress{font-weight:800}.bar{width:180px;height:7px;background:#ddd7ca;border-radius:10px;overflow:hidden}.bar i{display:block;height:100%;background:var(--good)}button{border:1px solid var(--line);background:#fffdf8;border-radius:8px;padding:7px 10px;color:var(--ink)}.layout{max-width:1500px;margin:16px auto 70px;display:grid;grid-template-columns:250px minmax(0,1fr);gap:16px;padding:0 16px}.side{position:sticky;top:82px;max-height:calc(100vh - 100px);overflow:auto;background:#faf7f0;border:1px solid var(--line);border-radius:13px;padding:10px}.doc{display:block;width:100%;text-align:left;margin:4px 0;border:0;background:#ebe6dc;padding:9px;border-radius:8px}.doc.active{background:#24372d;color:white}.doc small{display:block;opacity:.7}.reader{background:var(--card);border:1px solid var(--line);border-radius:15px;box-shadow:var(--shadow);overflow:hidden}.readerhead{padding:15px 18px;border-bottom:1px solid var(--line);display:flex;gap:10px;align-items:center;flex-wrap:wrap}.readerhead strong{font:700 18px Georgia,serif}.muted{color:var(--muted);font-size:12px}.lines{padding:12px}.line{display:grid;grid-template-columns:64px minmax(0,1fr) 252px;gap:10px;padding:9px 8px;border-radius:9px;scroll-margin:130px;border:1px solid transparent}.line.active{background:#fff0bd;border-color:#c79635}.line.decided{background:#f2f6f2}.ln{font:11px/1.7 ui-monospace,monospace;color:#918b81;text-align:right}.text{font:16px/1.42 Georgia,serif;overflow-wrap:anywhere}.machine{display:grid;grid-template-columns:1fr 1fr;gap:5px}.prediction{border-radius:7px;padding:5px 7px;color:white;font-size:10px;font-weight:800}.prediction small{display:block;opacity:.75}.hiddenpred{background:#c9c3b7;color:#6d675e}.role-ENTRY_ANCHOR{background:var(--entry)}.role-CONTINUATION{background:var(--cont)}.role-FILLER{background:var(--fill)}.role-HEADER{background:var(--head)}.role-SUBHEADER{background:var(--sub)}.role-NON_BIB{background:var(--non)}.role-UNKNOWN,.role-UNRESOLVED{background:var(--unk)}.agree{grid-column:1/-1;text-align:center;font-size:10px;font-weight:900;color:var(--good)}.disagree{color:var(--bad)}.controls{position:fixed;z-index:30;right:14px;bottom:14px;width:min(700px,calc(100vw - 28px));background:#fffcf7f5;border:1px solid var(--line);box-shadow:0 18px 55px #14201933;border-radius:14px;padding:10px;backdrop-filter:blur(14px)}.choices{display:grid;grid-template-columns:repeat(7,1fr);gap:5px}.choices button{color:white;border:0;padding:8px 4px;font-size:10px;font-weight:800}.choices button.selected{outline:3px solid #151915;outline-offset:1px;box-shadow:0 0 0 2px #fff}.choices b{display:block;font-size:15px}.bounds{display:flex;gap:5px;margin-top:6px;align-items:center}.bounds button{flex:1}.bounds button.selected{background:#263a2f;color:white}.bounds .clear{border-color:#b46860;color:#8c2f28}.summary{font-size:11px;color:var(--muted);margin-left:auto}.hidden{display:none!important}@media(max-width:950px){.layout{grid-template-columns:1fr}.side{position:static;max-height:180px}.line{grid-template-columns:45px minmax(0,1fr)}.machine{grid-column:2}.controls{right:7px;bottom:7px}.choices{grid-template-columns:repeat(4,1fr)}}
</style></head><body><header class="top"><div class="topin"><h1>__TITLE__</h1><span id="progress" class="progress"></span><span class="bar"><i id="bar"></i></span><button id="resume">Resume undecided</button><button id="export">Export review</button><span class="muted">Role: 1–7 · boundary: N/S/H · clear: X · next: ↓ or J</span></div></header><main class="layout"><aside id="docs" class="side"></aside><section class="reader"><div id="readerhead" class="readerhead"></div><div id="lines" class="lines"></div></section></main><nav class="controls"><div class="choices"><button class="role-ENTRY_ANCHOR" data-role="ENTRY_ANCHOR"><b>1</b>Entry</button><button class="role-CONTINUATION" data-role="CONTINUATION"><b>2</b>Continuation</button><button class="role-FILLER" data-role="FILLER"><b>3</b>Filler</button><button class="role-HEADER" data-role="HEADER"><b>4</b>Header</button><button class="role-SUBHEADER" data-role="SUBHEADER"><b>5</b>Subheader</button><button class="role-NON_BIB" data-role="NON_BIB"><b>6</b>Non-bib</button><button class="role-UNKNOWN" data-role="UNKNOWN"><b>7</b>Unknown</button></div><div class="bounds"><button data-boundary="NONE"><b>N</b> None</button><button data-boundary="SOFT_STOP"><b>S</b> Soft stop</button><button data-boundary="HARD_STOP"><b>H</b> Hard stop</button><button class="clear" data-clear><b>X</b> Clear line</button><span id="current" class="summary"></span></div></nav><script>const DATA=__DATA__,KEY='bib-role-human:'+DATA.packet_sha256,$=x=>document.getElementById(x),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let saved=JSON.parse(localStorage.getItem(KEY)||'{}'),bi=0,li=0;const roleShort={ENTRY_ANCHOR:'ENTRY',CONTINUATION:'CONT',FILLER:'FILLER',HEADER:'HEADER',SUBHEADER:'SUBHEAD',NON_BIB:'NON-BIB',UNKNOWN:'UNKNOWN',UNRESOLVED:'UNRESOLVED'};function id(b,l){return b.block_case_id+':'+l.line_id}function decision(b,l){return saved[id(b,l)]}function allLines(){return DATA.blocks.flatMap((b,i)=>b.lines.map((l,j)=>[b,l,i,j]))}function complete(){return allLines().filter(([b,l])=>decision(b,l)?.role).length}function docs(){return DATA.blocks.map((b,i)=>`<button class="doc ${i===bi?'active':''}" data-doc="${i}">${i+1}. ${esc(b.source)}<small>${b.lines.filter(l=>decision(b,l)?.role).length}/${b.lines.length} lines · ${b.block_case_id.slice(0,8)}</small></button>`).join('')}function pred(label,value,bound,visible){return visible?`<span class="prediction role-${value}"><small>${label} · ${esc(bound)}</small>${esc(roleShort[value]||value)}</span>`:`<span class="prediction hiddenpred"><small>${label}</small>hidden</span>`}function lineHtml(b,l,i){const d=decision(b,l),visible=!!d?.role,agree=l.pass_a_role===l.pass_b_role&&l.pass_a_boundary===l.pass_b_boundary,human=visible&&d.role===l.pass_a_role&&d.role===l.pass_b_role&&d.boundary===l.pass_a_boundary&&d.boundary===l.pass_b_boundary;return`<div id="line-${l.line_id}" class="line ${i===li?'active':''} ${visible?'decided':''}" data-line="${i}"><span class="ln">L${l.abs_idx}<br>${l.position_percent.toFixed(1)}%</span><span class="text">${esc(l.text)}</span><span class="machine">${pred('Pass A',l.pass_a_role,l.pass_a_boundary,visible)}${pred('Pass B',l.pass_b_role,l.pass_b_boundary,visible)}<span class="agree ${agree?'':'disagree'}">${visible?(human?'ALL AGREE':agree?'PASSES AGREE · YOU DIFFER':'PASSES DISAGREE'):'decide first to reveal'}</span></span></div>`}function render(){const b=DATA.blocks[bi];$('docs').innerHTML=docs();$('readerhead').innerHTML=`<strong>${bi+1}/${DATA.blocks.length} · ${esc(b.source)}</strong><span class="muted">document ${b.document_id.slice(0,12)} · work ${b.work_id.slice(0,12)} · ${b.lines.length} displayed lines</span>`;$('lines').innerHTML=b.lines.map((l,i)=>lineHtml(b,l,i)).join('');const d=decision(b,b.lines[li]);document.querySelectorAll('[data-role]').forEach(x=>x.classList.toggle('selected',x.dataset.role===d?.role));document.querySelectorAll('[data-boundary]').forEach(x=>x.classList.toggle('selected',x.dataset.boundary===(d?.boundary||'NONE')));$('current').textContent=`L${b.lines[li].abs_idx} · ${d?.role||'undecided'} / ${d?.boundary||'NONE'}`;const n=complete(),total=allLines().length;$('progress').textContent=`${n}/${total}`;$('bar').style.width=(100*n/total)+'%'}function activateDoc(i){bi=Math.max(0,Math.min(DATA.blocks.length-1,i));const b=DATA.blocks[bi],u=b.lines.findIndex(l=>!decision(b,l)?.role);li=u<0?0:u;render();setTimeout(()=>$('line-'+b.lines[li].line_id)?.scrollIntoView({behavior:'smooth',block:'center'}),30)}function activateLine(i,scroll=false){li=Math.max(0,Math.min(DATA.blocks[bi].lines.length-1,i));render();if(scroll)$('line-'+DATA.blocks[bi].lines[li].line_id)?.scrollIntoView({behavior:'smooth',block:'center'})}function save(role=null,boundary=null){const b=DATA.blocks[bi],l=b.lines[li],old=decision(b,l)||{};saved[id(b,l)]={schema_version:'bibliography-role-human-decision-v1',block_case_id:b.block_case_id,document_id:b.document_id,line_id:l.line_id,abs_idx:l.abs_idx,role:role||old.role||null,boundary:boundary||old.boundary||'NONE',reviewer:'foivos',updated_at:new Date().toISOString()};localStorage.setItem(KEY,JSON.stringify(saved));render()}function clearLine(){const b=DATA.blocks[bi],l=b.lines[li];delete saved[id(b,l)];localStorage.setItem(KEY,JSON.stringify(saved));render()}function next(){const rows=allLines(),here=rows.findIndex(([,l,i,j])=>i===bi&&j===li);for(let n=1;n<=rows.length;n++){const [b,l,i,j]=rows[(here+n)%rows.length];if(!decision(b,l)?.role){bi=i;li=j;render();setTimeout(()=>$('line-'+l.line_id)?.scrollIntoView({behavior:'smooth',block:'center'}),30);return}}}document.addEventListener('click',e=>{const d=e.target.closest('[data-doc]');if(d)return activateDoc(Number(d.dataset.doc));const l=e.target.closest('[data-line]');if(l)return activateLine(Number(l.dataset.line));const r=e.target.closest('[data-role]');if(r)return save(r.dataset.role,null);const b=e.target.closest('[data-boundary]');if(b)return save(null,b.dataset.boundary);if(e.target.closest('[data-clear]'))return clearLine()});$('resume').onclick=next;$('export').onclick=()=>{const out={schema_version:'bibliography-role-human-audit-export-v1',packet_sha256:DATA.packet_sha256,site_payload_sha256:'__PAYLOAD_SHA__',reviewer:'foivos',exported_at:new Date().toISOString(),decision_count:Object.values(saved).filter(x=>x.role).length,decisions:Object.values(saved).sort((a,b)=>a.document_id.localeCompare(b.document_id)||a.abs_idx-b.abs_idx)};const blob=new Blob([JSON.stringify(out,null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='foivos-bibliography-role-audit-'+DATA.packet_sha256.slice(0,12)+'.json';a.click();URL.revokeObjectURL(a.href)};document.onkeydown=e=>{if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName))return;const roles={1:'ENTRY_ANCHOR',2:'CONTINUATION',3:'FILLER',4:'HEADER',5:'SUBHEADER',6:'NON_BIB',7:'UNKNOWN'},bounds={n:'NONE',s:'SOFT_STOP',h:'HARD_STOP'};if(roles[e.key]){e.preventDefault();save(roles[e.key],null)}else if(bounds[e.key.toLowerCase()]){e.preventDefault();save(null,bounds[e.key.toLowerCase()])}else if(e.key.toLowerCase()==='x'){e.preventDefault();clearLine()}else if(e.key==='ArrowDown'||e.key.toLowerCase()==='j'){e.preventDefault();activateLine(li+1,true)}else if(e.key==='ArrowUp'||e.key.toLowerCase()==='k'){e.preventDefault();activateLine(li-1,true)}};render();next();</script></body></html>""".replace("__TITLE__", safe_title).replace("__DATA__", data).replace("__PAYLOAD_SHA__", canonical_json_sha256(payload))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--review-a", required=True)
    parser.add_argument("--review-b", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt-out", required=True)
    parser.add_argument("--per-source", type=int, default=10)
    parser.add_argument("--seed", default="bibliography-role-human-audit-v1")
    parser.add_argument("--title", default="Bibliography role audit")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    packet_path = Path(args.packet).resolve()
    review_a_path, review_b_path = Path(args.review_a).resolve(), Path(args.review_b).resolve()
    packet, review_a, review_b = load_packet(packet_path), load_review(review_a_path), load_review(review_b_path)
    selected = select_blocks(packet, review_a, review_b, per_source=args.per_source, seed=args.seed)
    payload = build_payload(
        packet, review_a, review_b, selected_blocks=selected,
        packet_sha256=sha256_file(packet_path),
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(_page(payload, args.title))
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    receipt = {
        "schema_version": "bibliography-role-human-audit-site-receipt-v1",
        "status": "passed",
        "site_payload_sha256": canonical_json_sha256(payload),
        "block_count": len(payload["blocks"]),
        "line_count": sum(len(block["lines"]) for block in payload["blocks"]),
        "source_counts": payload["selection"]["source_counts"],
        "inputs": {
            "packet_sha256": sha256_file(packet_path),
            "review_a_sha256": sha256_file(review_a_path),
            "review_b_sha256": sha256_file(review_b_path),
        },
        "output": {"path": str(output), "sha256": sha256_file(output)},
    }
    receipt_path = Path(args.receipt_out).resolve()
    _write = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(_write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
