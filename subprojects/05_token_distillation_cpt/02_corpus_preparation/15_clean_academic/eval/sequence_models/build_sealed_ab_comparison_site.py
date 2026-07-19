#!/usr/bin/env python3
"""Build a static side-by-side reader for sealed bibliography passes A and B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contract import sha256_file


SITE_SCHEMA = "bibliography-sealed-ab-comparison-site-v1"
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
      <div class="side-heading">
        <strong>Documents</strong>
        <span class="muted">Worst agreement first</span>
      </div>
      <nav id="documents" aria-label="Sealed documents"></nav>
    </aside>
    <section class="reader" aria-live="polite">
      <div class="column-heads">
        <strong>Pass A</strong>
        <span>line</span>
        <strong>Pass B</strong>
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
.layout{max-width:1900px;margin:14px auto 70px;display:grid;grid-template-columns:280px minmax(0,1fr);gap:14px;padding:0 14px}
.sidebar{position:sticky;top:112px;max-height:calc(100vh - 128px);overflow:auto;background:#faf7f0;border:1px solid var(--line);border-radius:13px;padding:9px}.side-heading{padding:5px 6px 9px;display:flex;justify-content:space-between;gap:8px;align-items:baseline}.doc{display:block;width:100%;text-align:left;margin:4px 0;border:0;background:#ebe6dc;padding:8px 9px;border-radius:8px;color:var(--ink)}.doc:hover{background:#dfd8ca}.doc.active{background:#24372d;color:#fff}.doc strong,.doc small{display:block}.doc strong{font-size:12px}.doc small{opacity:.75;font-size:10px;margin-top:2px}
.reader{background:var(--card);border:1px solid var(--line);border-radius:15px;box-shadow:var(--shadow);overflow:hidden}.column-heads{position:sticky;top:80px;z-index:10;display:grid;grid-template-columns:minmax(0,1fr) 74px minmax(0,1fr);gap:8px;padding:10px 12px;background:#fffdf8f2;border-bottom:1px solid var(--line);backdrop-filter:blur(12px);text-align:center}.column-heads strong{font:700 17px Georgia,serif}.column-heads span{color:var(--muted);font-size:11px}
.lines{padding:9px}.compare-row{display:grid;grid-template-columns:minmax(0,1fr) 74px minmax(0,1fr);gap:8px;align-items:stretch;margin:2px 0;scroll-margin-top:170px}.annotation{position:relative;display:grid;grid-template-columns:110px minmax(0,1fr);gap:9px;padding:7px 9px;border-radius:8px;background:var(--role-bg);border-left:5px solid var(--role);min-width:0}.role{align-self:start;border-radius:6px;background:var(--role);color:#fff;padding:4px 6px;font-size:10px;font-weight:800;letter-spacing:.025em;text-align:center;overflow-wrap:anywhere}.text{font:15px/1.42 Georgia,"Times New Roman",serif;overflow-wrap:anywhere;white-space:pre-wrap}.line-marker{display:flex;flex-direction:column;align-items:center;justify-content:center;color:var(--muted);font:10px/1.25 ui-monospace,SFMono-Regular,Menlo,monospace;text-align:center}.line-marker b{font:800 16px/1 system-ui;color:var(--bad)}.compare-row.binary-agree .line-marker b{color:var(--good)}.compare-row.exact-agree .line-marker b{opacity:.28}.compare-row.binary-disagree{background:#f7e3e04d;border-radius:9px}.compare-row.binary-disagree .line-marker{background:#f8dfdc;border-radius:7px;color:#7e302c}
.role-ENTRY{--role:var(--entry);--role-bg:var(--entry-bg)}.role-CONTINUATION{--role:var(--continuation);--role-bg:var(--continuation-bg)}.role-FILLER{--role:var(--filler);--role-bg:var(--filler-bg)}.role-BIB_HEADER{--role:var(--bib-header);--role-bg:var(--bib-header-bg)}.role-BIB_SUBHEADER{--role:var(--bib-subheader);--role-bg:var(--bib-subheader-bg)}.role-NON_BIB_HEADER{--role:var(--non-bib-header);--role-bg:var(--non-bib-header-bg)}.role-OTHER{--role:var(--other);--role-bg:var(--other-bg)}.role-UNKNOWN{--role:var(--unknown);--role-bg:var(--unknown-bg)}
.loading,.empty{padding:40px;text-align:center;color:var(--muted)}
@media(max-width:1050px){.layout{grid-template-columns:220px minmax(0,1fr)}.annotation{grid-template-columns:90px minmax(0,1fr)}.compare-row,.column-heads{grid-template-columns:minmax(0,1fr) 55px minmax(0,1fr)}}
@media(max-width:760px){.top{grid-template-columns:1fr}.actions{justify-content:flex-start}.layout{grid-template-columns:1fr}.sidebar{position:static;max-height:220px}.column-heads{top:154px}.compare-row{grid-template-columns:1fr 44px 1fr;gap:4px}.annotation{grid-template-columns:1fr;padding:6px}.text{font-size:13px}.role{width:max-content;max-width:100%}}
"""


APP_JS = """const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[c]));
const roles=['ENTRY','CONTINUATION','FILLER','BIB_HEADER','BIB_SUBHEADER','NON_BIB_HEADER','OTHER','UNKNOWN'];
let manifest=null,current=null,currentIndex=0,disagreements=[],disagreementIndex=-1;
function roleClass(role){return 'role-'+String(role).replace(/[^A-Z_]/g,'')}
function shortId(doc){return doc.source_doc_id.length>22?doc.source_doc_id.slice(0,12)+'…':doc.source_doc_id}
function renderLegend(){$('legend').innerHTML=roles.map(r=>`<span class="legend-item ${roleClass(r)}"><i class="swatch"></i>${r.replaceAll('_',' ')}</span>`).join('')}
function renderDocuments(){$('documents').innerHTML=manifest.documents.map((d,i)=>`<button type="button" class="doc ${i===currentIndex?'active':''}" data-index="${i}"><strong>${i+1}. ${esc(d.source)} · ${esc(shortId(d))}</strong><small>${(100*d.binary_agreement).toFixed(2)}% agreement · ${d.binary_disagreements.toLocaleString()} disagreements · ${d.line_count.toLocaleString()} lines</small></button>`).join('')}
function annotation(role,text){return `<div class="annotation ${roleClass(role)}"><span class="role">${esc(role.replaceAll('_',' '))}</span><span class="text">${esc(text)}</span></div>`}
function renderLine(line,index){const state=line.exact_agree?'exact-agree binary-agree':line.binary_agree?'binary-agree role-disagree':'binary-disagree';const mark=line.exact_agree?'=':line.binary_agree?'≈':'≠';const title=line.exact_agree?'Exact role agreement':line.binary_agree?'Same BIB/NON-BIB decision; detailed roles differ':'Binary BIB/NON-BIB disagreement';return `<div id="line-${index}" class="compare-row ${state}" data-line="${index}">${annotation(line.pass_a_role,line.text)}<span class="line-marker" title="${title}">L${line.abs_idx}<b>${mark}</b></span>${annotation(line.pass_b_role,line.text)}</div>`}
function renderCurrent(){renderDocuments();disagreements=[];current.lines.forEach((line,i)=>{if(!line.binary_agree)disagreements.push(i)});$('lines').innerHTML=current.lines.map(renderLine).join('');const d=manifest.documents[currentIndex];$('doc-summary').textContent=`${d.source} · ${shortId(d)} · ${(100*d.binary_agreement).toFixed(2)}% agreement · ${d.binary_disagreements.toLocaleString()} disagreements / ${d.line_count.toLocaleString()} lines`;document.title=`${(100*d.binary_agreement).toFixed(2)}% · ${shortId(d)} · A/B comparison`;disagreementIndex=-1;window.scrollTo({top:0,behavior:'instant'});history.replaceState(null,'','#'+d.document_id)}
async function loadDocument(index){currentIndex=Math.max(0,Math.min(manifest.documents.length-1,index));$('lines').innerHTML='<div class="loading">Loading document…</div>';renderDocuments();const d=manifest.documents[currentIndex];const response=await fetch('data/'+d.document_id+'.json');if(!response.ok)throw new Error('Could not load '+d.document_id);current=await response.json();renderCurrent()}
function jump(direction){if(!disagreements.length)return;const visible=disagreements.findIndex(i=>document.getElementById('line-'+i)?.getBoundingClientRect().top>170);if(direction>0){disagreementIndex=visible>=0?visible:0}else{const y=window.scrollY+170;let previous=-1;disagreements.forEach((line,i)=>{if(document.getElementById('line-'+line)?.offsetTop<y)previous=i});disagreementIndex=previous>0?previous-1:disagreements.length-1}document.getElementById('line-'+disagreements[disagreementIndex])?.scrollIntoView({behavior:'smooth',block:'center'})}
document.addEventListener('click',event=>{const button=event.target.closest('[data-index]');if(button)loadDocument(Number(button.dataset.index))});
$('previous-disagreement').onclick=()=>jump(-1);$('next-disagreement').onclick=()=>jump(1);
document.addEventListener('keydown',event=>{if(event.key.toLowerCase()==='n')jump(1);if(event.key.toLowerCase()==='p')jump(-1)});
async function start(){renderLegend();const response=await fetch('manifest.json');manifest=await response.json();const wanted=location.hash.slice(1);const index=Math.max(0,manifest.documents.findIndex(d=>d.document_id===wanted));await loadDocument(index)}
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return manifest rows and document payloads in worst-agreement order."""

    a_lines = _pass_lines(pass_a, "pass A")
    b_lines = _pass_lines(pass_b, "pass B")
    key_by_coordinate: dict[tuple[str, str], Mapping[str, Any]] = {}
    expected_aliases: set[str] = set()
    for row in line_keys:
        coordinate = (str(row.get("document_id") or ""), str(row.get("line_id") or ""))
        alias = str(row.get("line_alias") or "")
        if "" in coordinate or not alias or coordinate in key_by_coordinate:
            raise ValueError("line key has empty or duplicate coordinates")
        key_by_coordinate[coordinate] = row
        expected_aliases.add(alias)
    if set(a_lines) != expected_aliases or set(b_lines) != expected_aliases:
        raise ValueError("passes A and B must cover exactly the line-key aliases")

    manifest_rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    seen_coordinates: set[tuple[str, str]] = set()
    for document in documents:
        document_id = str(document.get("document_id") or "")
        source = str(document.get("source") or "")
        output_lines: list[dict[str, Any]] = []
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
            role_a, role_b = str(a_lines[alias]["role"]), str(b_lines[alias]["role"])
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
                    "binary_agree": binary_agree,
                    "exact_agree": exact_agree,
                }
            )
        line_count = len(output_lines)
        if not line_count:
            raise ValueError(f"sealed document has no present lines: {document_id}")
        agreement = 1.0 - binary_disagreements / line_count
        summary = {
            "document_id": document_id,
            "source": source,
            "source_doc_id": str(document.get("source_doc_id") or ""),
            "source_repo_id": str(document.get("source_repo_id") or ""),
            "source_dataset": str(document.get("source_dataset") or ""),
            "source_row_id": str(document.get("source_row_id") or ""),
            "work_id": str(document.get("work_id") or ""),
            "line_count": line_count,
            "binary_agreement": agreement,
            "binary_disagreements": binary_disagreements,
            "exact_role_agreement": exact_agreements / line_count,
            "top_disagreement_pairs": [
                {"pair": pair, "count": count}
                for pair, count in disagreement_pairs.most_common(5)
            ],
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
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    documents = _read_jsonl(documents_path)
    line_keys = _read_jsonl(line_key_path)
    pass_a, pass_b = _read_json(pass_a_path), _read_json(pass_b_path)
    manifest_rows, payloads = build_documents(documents, line_keys, pass_a, pass_b)
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
        source_summary = {
            source: {
                "document_count": sum(row["source"] == source for row in manifest_rows),
                "line_count": sum(
                    row["line_count"]
                    for row in manifest_rows
                    if row["source"] == source
                ),
                "binary_disagreements": sum(
                    row["binary_disagreements"] for row in manifest_rows if row["source"] == source
                ),
            }
            for source in sorted({row["source"] for row in manifest_rows})
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
            "source_summary": source_summary,
            "documents": manifest_rows,
        }
        _write_json(partial / "manifest.json", manifest)
        site_sha256 = _tree_sha256(partial)
        receipt = {
            "schema_version": "bibliography-sealed-ab-comparison-site-receipt-v1",
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
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = build_site(
        documents_path=args.documents.resolve(),
        line_key_path=args.line_key.resolve(),
        pass_a_path=args.pass_a.resolve(),
        pass_b_path=args.pass_b.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
