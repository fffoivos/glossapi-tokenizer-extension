#!/usr/bin/env python3
"""Build and serve the private Agent 1 v4 raw-review site on CSCS."""

from __future__ import annotations

import argparse
import functools
import hashlib
import hmac
import json
import os
import shutil
import stat
import tempfile
from collections import Counter, defaultdict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SCRIPT_DIR))

from agent1_v4_raw_review import (  # noqa: E402
    PACKET_SCHEMA,
    file_binding,
    read_json_object,
    sha256_file,
    validate_packet,
)
from run_agent1_v4_terra_reviews import validate_response  # noqa: E402


SITE_SCHEMA = "agent1_v4_raw_review_site_manifest_v1"
SITE_INDEX_SCHEMA = "agent1_v4_raw_review_site_index_v1"
HEX_SHA256_RE = __import__("re").compile(r"^[0-9a-f]{64}$")
DEFAULT_MAX_PORTABLE_ASSETS_BYTES = 16 * 1024 * 1024


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_file(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        os.chmod(path, mode)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: object) -> None:
    _write_file(path, (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _relative_document(root: Path, request: Mapping[str, object]) -> Path:
    relative = Path(str(request["document_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("review document path escapes packet root")
    document = root / relative
    if document.is_symlink() or not document.is_file():
        raise FileNotFoundError(f"review document missing/symlinked: {relative}")
    return document


def _opaque_id(secret: bytes, request_id: str) -> str:
    return hmac.new(secret, request_id.encode("ascii"), hashlib.sha256).hexdigest()


def _validate_inputs(
    packet_root: Path,
    packet_manifest: Path,
    requests_path: Path,
    responses_path: Path,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, dict[str, object]]]:
    manifest = validate_packet(packet_root, packet_manifest)
    declared_requests = manifest.get("requests")
    actual_requests = file_binding(requests_path)
    if not isinstance(declared_requests, Mapping) or actual_requests["sha256"] != declared_requests.get("sha256") or actual_requests["bytes"] != declared_requests.get("bytes"):
        raise ValueError("site request input differs from packet manifest")
    requests = _read_jsonl(requests_path)
    responses = _read_jsonl(responses_path)
    by_request = {str(request["request_id"]): request for request in requests}
    expected_count = manifest.get("logical_review_count")
    if not isinstance(expected_count, int) or len(by_request) != expected_count or len(responses) != expected_count:
        raise ValueError("site request/response count differs from the packet manifest")
    by_response: dict[str, dict[str, object]] = {}
    for response in responses:
        request_id = str(response.get("request_id") or "")
        request = by_request.get(request_id)
        if request is None or request_id in by_response:
            raise ValueError("response does not close one unique packet request")
        validate_response(response, request, _relative_document(packet_root, request))
        by_response[request_id] = response
    if set(by_request) != set(by_response):
        raise ValueError("request/response closure failed")
    return manifest, requests, by_response


def _index_card(request: Mapping[str, object], response: Mapping[str, object], opaque_id: str) -> dict[str, object]:
    locator = request["origin_locator"]
    assert isinstance(locator, Mapping)
    return {
        "request_id": request["request_id"],
        "opaque_id": opaque_id,
        "source_id": request["source_id"],
        "source_dataset": request["source_dataset"],
        "source_doc_id": request["source_doc_id"],
        "source_revision": request["source_revision"],
        "source_route": request["source_route"],
        "extraction_route": request["extraction_route"],
        "document_path": request["document_path"],
        "document_bytes": request["document_bytes"],
        "document_line_count": request["document_line_count"],
        "origin_locator": dict(locator),
        "cleanliness_score": response["cleanliness_score"],
        "text_quality_score": response["text_quality_score"],
        "confidence": response["confidence"],
        "coverage_mode": response["coverage_mode"],
        "summary": response["summary"],
        "extraction_artifacts": response["extraction_artifacts"],
    }


def _artifact_count(card: Mapping[str, object]) -> Counter[str]:
    artifacts = card["extraction_artifacts"]
    assert isinstance(artifacts, list)
    return Counter(str(artifact["type"]) for artifact in artifacts if isinstance(artifact, Mapping))


def _html() -> str:
    return """<!doctype html>
<html lang="el">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; connect-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; base-uri 'none'; form-action 'none'">
  <title>Apertus raw-document review</title>
  <link rel="stylesheet" href="assets/site.css">
</head>
<body>
  <header><h1>Apertus raw-document review</h1><p id="coverage"></p><div id="overview"></div></header>
  <main>
    <section class="controls" aria-label="Document filters">
      <label>Source <select id="source-filter"></select></label>
      <label>Cleanliness ≥ <select id="cleanliness-filter"><option value="1">1</option><option value="2">2</option><option value="3">3</option><option value="4">4</option><option value="5">5</option></select></label>
      <label>Text quality ≥ <select id="quality-filter"><option value="1">1</option><option value="2">2</option><option value="3">3</option><option value="4">4</option><option value="5">5</option></select></label>
      <label>Confidence <select id="confidence-filter"></select></label>
      <label>Artifact <select id="artifact-filter"></select></label>
    </section>
    <section class="layout">
      <nav aria-label="Documents"><ol id="document-list"></ol></nav>
      <article id="document" aria-live="polite"><p>Select a document.</p></article>
    </section>
    <section class="decisions">
      <h2>Human decision export</h2>
      <label>Source status <select id="source-status"><option value="">not decided</option><option value="admit">admit</option><option value="hold">hold</option><option value="exclude">exclude</option></select></label>
      <label>Source cleaning observation <textarea id="source-observation" rows="3"></textarea></label>
      <label>Text/title/author mapping question <textarea id="mapping-question" rows="3"></textarea></label>
      <label>Document disposition <select id="document-disposition"><option value="unreviewed">not decided</option><option value="agree">agree</option><option value="override">override scores</option><option value="flag">flag for follow-up</option><option value="hold">hold</option></select></label>
      <label>Override cleanliness <select id="cleanliness-override"><option value="">use Terra score</option><option value="1">1</option><option value="2">2</option><option value="3">3</option><option value="4">4</option><option value="5">5</option></select></label>
      <label>Override text quality <select id="quality-override"><option value="">use Terra score</option><option value="1">1</option><option value="2">2</option><option value="3">3</option><option value="4">4</option><option value="5">5</option></select></label>
      <label>Document note <textarea id="note" rows="3"></textarea></label>
      <label><input id="field-discovery-approval" type="checkbox"> I approve field discovery for explicitly admitted sources.</label>
      <button id="agree" type="button">Agree with Terra scores</button>
      <button id="export" type="button">Download decision bundle</button>
    </section>
  </main>
  <script src="assets/site.js"></script>
</body>
</html>
"""


def _css() -> str:
    return """*{box-sizing:border-box}body{margin:0;background:#f8fafc;color:#152238;font:16px/1.45 system-ui,-apple-system,sans-serif}header,main{max-width:1500px;margin:auto;padding:1rem 1.25rem}header{background:#102a43;color:#fff;max-width:none;padding-left:max(1.25rem,calc((100% - 1500px)/2 + 1.25rem))}.controls{display:flex;gap:1rem;flex-wrap:wrap;background:#e8f1fa;padding:1rem}.controls label,.decisions label{display:grid;gap:.3rem}.layout{display:grid;grid-template-columns:minmax(260px,28%) 1fr;gap:1rem;margin-top:1rem}.layout nav{max-height:74vh;overflow:auto;background:#fff;border:1px solid #cbd5e1}.layout ol{margin:0;padding:0;list-style:none}.layout button{width:100%;text-align:left;border:0;border-bottom:1px solid #e2e8f0;background:#fff;padding:.7rem;cursor:pointer}.layout button:hover,.layout button:focus{background:#dceeff}.card{background:#fff;border:1px solid #cbd5e1;padding:1rem;min-width:0}.meta{display:grid;grid-template-columns:max-content 1fr;gap:.25rem .75rem}.score{display:inline-block;border-radius:.25rem;padding:.15rem .45rem;background:#dbeafe;margin-right:.5rem}.artifact{border-left:4px solid #c2410c;padding:.5rem .75rem;margin:.7rem 0;background:#fff7ed}.artifact pre,.raw{white-space:pre-wrap;overflow-wrap:anywhere;background:#0b1220;color:#e5edf7;padding:1rem}.decisions{margin-top:1rem;background:#fff;border:1px solid #cbd5e1;padding:1rem;display:grid;gap:.7rem}.decisions button{width:max-content;padding:.5rem .8rem}#overview{display:flex;gap:.5rem;flex-wrap:wrap;font-size:.85rem}#overview span{background:#1f4d78;padding:.2rem .45rem;border-radius:.2rem}@media(max-width:850px){.layout{grid-template-columns:1fr}.layout nav{max-height:30vh}}"""


def _js() -> str:
    return """'use strict';
const state={index:null,cards:[],visible:[],source_counts:{},decisions:{schema_version:'agent1_v4_human_decision_bundle_v1',packet_manifest_sha256:'',approval_to_begin_field_discovery:false,source_status:{},source_observations:{},mapping_questions:{},documents:{}}};
const $=id=>document.getElementById(id);
const text=(tag,value)=>{const node=document.createElement(tag);node.textContent=String(value??'');return node;};
function select(id,values,label){const el=$(id);el.replaceChildren();el.append(new Option(label,''));for(const value of values)el.append(new Option(value,value));}
function sources(){return [...new Set(state.cards.map(card=>card.source_id))].sort();}
function initializeDecisions(){for(const source of sources()){state.decisions.source_status[source]='';state.decisions.source_observations[source]='';state.decisions.mapping_questions[source]='';}for(const card of state.cards){state.decisions.documents[card.request_id]={source_id:card.source_id,source_doc_id:card.source_doc_id,disposition:'unreviewed',cleanliness_score_override:null,text_quality_score_override:null,note:''};}}
function sourceSummary(){const ids=sources(),artifacts=[...new Set(state.cards.flatMap(card=>card.extraction_artifacts.map(artifact=>artifact.type)))].sort();select('source-filter',ids,'all sources');select('artifact-filter',artifacts,'all artifacts');select('confidence-filter',['low','medium','high'],'all confidence');$('coverage').textContent=`${ids.length} sources · ${state.cards.length} reviewed raw documents`;const overview=$('overview');overview.replaceChildren();for(const source of ids){const cards=state.cards.filter(card=>card.source_id===source),artifactCount=cards.reduce((n,card)=>n+card.extraction_artifacts.length,0);overview.append(text('span',`${source}: ${cards.length}/${state.source_counts[source]??cards.length} · ${artifactCount} artifacts`));}}
function filter(){const source=$('source-filter').value,artifact=$('artifact-filter').value,confidence=$('confidence-filter').value,clean=Number($('cleanliness-filter').value),quality=Number($('quality-filter').value);state.visible=state.cards.filter(card=>(!source||card.source_id===source)&&(!artifact||card.extraction_artifacts.some(item=>item.type===artifact))&&(!confidence||card.confidence===confidence)&&card.cleanliness_score>=clean&&card.text_quality_score>=quality);renderList();if(state.visible.length)show(state.visible[0]);else $('document').replaceChildren(text('p','No documents match these filters.'));}
function renderList(){const list=$('document-list');list.replaceChildren();for(const card of state.visible){const button=document.createElement('button');button.type='button';button.append(text('strong',card.source_id),document.createElement('br'),text('span',`clean ${card.cleanliness_score}/5 · quality ${card.text_quality_score}/5`));button.addEventListener('click',()=>show(card));const item=document.createElement('li');item.append(button);list.append(item);}}
function labeled(label,value){return [text('dt',label),text('dd',value)];}
function scoreValue(id){const value=$(id).value;return value===''?null:Number(value);}
function setDecisionControls(card){const decision=state.decisions.documents[card.request_id];$('source-status').value=state.decisions.source_status[card.source_id];$('source-observation').value=state.decisions.source_observations[card.source_id];$('mapping-question').value=state.decisions.mapping_questions[card.source_id];$('document-disposition').value=decision.disposition;$('cleanliness-override').value=decision.cleanliness_score_override??'';$('quality-override').value=decision.text_quality_score_override??'';$('note').value=decision.note;$('field-discovery-approval').checked=state.decisions.approval_to_begin_field_discovery;}
async function show(card){state.index=card;const article=$('document');article.replaceChildren();article.append(text('h2',card.source_id+' · '+card.source_doc_id));const scores=document.createElement('p');for(const value of [`Cleanliness ${card.cleanliness_score}/5`,`Text quality ${card.text_quality_score}/5`,card.confidence,card.coverage_mode]){const badge=text('span',value);badge.className='score';scores.append(badge);}article.append(scores);const meta=document.createElement('dl');for(const [label,value] of [['Repository',card.origin_locator.repo_id],['Dataset',card.source_dataset],['Revision',card.source_revision],['Logical route',card.source_route],['Observed route',card.extraction_route],['Packet path',card.document_path],['Origin',`${card.origin_locator.artifact_path}:${card.origin_locator.row_index} (${card.origin_locator.text_field})`]])meta.append(...labeled(label,value));meta.className='meta';article.append(meta);article.append(text('p',card.summary));article.append(text('h3','Extraction artifacts'));if(!card.extraction_artifacts.length)article.append(text('p','No material extraction artifact reported.'));for(const artifact of card.extraction_artifacts){const box=document.createElement('section');box.className='artifact';box.append(text('strong',`${artifact.type} · ${artifact.severity} · lines ${artifact.line_start}-${artifact.line_end}`),text('p',artifact.explanation),text('pre',artifact.evidence_excerpt),text('p',`Cleaning: ${artifact.deterministic_cleaning_possible?'deterministic':'manual/uncertain'} — ${artifact.suggested_cleaning_action}`));article.append(box);}article.append(text('h3','Exact raw reviewed document'));setDecisionControls(card);const response=await fetch(`data/documents/${card.opaque_id}.json`,{cache:'no-store'});if(!response.ok)throw new Error('Raw document could not be loaded.');const payload=await response.json();if(state.index!==card)return;const pre=text('pre',payload.text);pre.className='raw';article.append(pre);}
function saveDecision(){const card=state.index;if(!card)return;const disposition=$('document-disposition').value;const cleanliness=disposition==='override'?scoreValue('cleanliness-override'):null;const quality=disposition==='override'?scoreValue('quality-override'):null;state.decisions.source_status[card.source_id]=$('source-status').value;state.decisions.source_observations[card.source_id]=$('source-observation').value;state.decisions.mapping_questions[card.source_id]=$('mapping-question').value;state.decisions.documents[card.request_id]={source_id:card.source_id,source_doc_id:card.source_doc_id,disposition,cleanliness_score_override:cleanliness,text_quality_score_override:quality,note:$('note').value};}
for(const id of ['source-filter','artifact-filter','confidence-filter','cleanliness-filter','quality-filter'])$(id).addEventListener('change',filter);for(const id of ['source-status','source-observation','mapping-question','document-disposition','cleanliness-override','quality-override','note'])$(id).addEventListener(id==='note'||id.includes('observation')||id.includes('question')?'input':'change',saveDecision);$('field-discovery-approval').addEventListener('change',()=>{state.decisions.approval_to_begin_field_discovery=$('field-discovery-approval').checked;});$('agree').addEventListener('click',()=>{if(!state.index)return;$('document-disposition').value='agree';$('cleanliness-override').value='';$('quality-override').value='';saveDecision();$('agree').textContent='Scores marked agreed';});$('export').addEventListener('click',()=>{saveDecision();const blob=new Blob([JSON.stringify(state.decisions,null,2)+'\\n'],{type:'application/json'});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='agent1_v4_human_decisions.json';link.click();URL.revokeObjectURL(link.href);});
fetch('data/index.json',{cache:'no-store'}).then(response=>{if(!response.ok)throw new Error('index could not be loaded');return response.json();}).then(index=>{state.cards=index.cards;state.source_counts=index.source_counts;state.decisions.packet_manifest_sha256=index.packet_manifest_sha256;initializeDecisions();sourceSummary();filter();}).catch(error=>{$('document').replaceChildren(text('p','Site data failed validation: '+error.message));});
"""


def _tree_inventory(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"site output contains forbidden symlink: {path}")
        if path.is_file():
            records.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return records


def build_site(
    *,
    packet_root: Path,
    packet_manifest: Path,
    requests_path: Path,
    responses_path: Path,
    site_secret_hex: str,
    output_dir: Path,
    max_portable_assets_bytes: int = DEFAULT_MAX_PORTABLE_ASSETS_BYTES,
) -> dict[str, object]:
    if not HEX_SHA256_RE.fullmatch(site_secret_hex):
        raise ValueError("site-secret must be a 32-byte lowercase hexadecimal value")
    secret = bytes.fromhex(site_secret_hex)
    if len(secret) != 32:
        raise ValueError("site-secret must encode 32 bytes")
    if max_portable_assets_bytes < 1:
        raise ValueError("portable asset size limit must be positive")
    packet_root = packet_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite review site: {output_dir}")
    manifest, requests, responses = _validate_inputs(packet_root, packet_manifest, requests_path, responses_path)
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=parent))
    try:
        os.chmod(staging, 0o700)
        cards: list[dict[str, object]] = []
        source_counts: Counter[str] = Counter()
        artifact_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for request in sorted(requests, key=lambda row: (str(row["source_id"]), str(row["sample_id"]))):
            response = responses[str(request["request_id"])]
            opaque_id = _opaque_id(secret, str(request["request_id"]))
            document = _relative_document(packet_root, request)
            text = document.read_text(encoding="utf-8")
            _write_json(
                staging / "data" / "documents" / f"{opaque_id}.json",
                {"schema_version": "agent1_v4_raw_review_site_document_v1", "opaque_id": opaque_id, "text": text},
            )
            card = _index_card(request, response, opaque_id)
            cards.append(card)
            source_counts[str(request["source_id"])] += 1
            artifact_counts[str(request["source_id"])].update(_artifact_count(card))
        if (
            len(cards) != manifest.get("logical_review_count")
            or len(source_counts) != 18
            or dict(sorted(source_counts.items())) != manifest.get("source_counts")
        ):
            raise ValueError("site source/document closure failed")
        index = {
            "schema_version": SITE_INDEX_SCHEMA,
            "packet_manifest_sha256": file_binding(packet_manifest)["sha256"],
            "source_counts": dict(sorted(source_counts.items())),
            "artifact_counts": {source_id: dict(sorted(counts.items())) for source_id, counts in sorted(artifact_counts.items())},
            "cards": cards,
        }
        _write_json(staging / "data" / "index.json", index)
        _write_file(staging / "index.html", _html().encode("utf-8"))
        _write_file(staging / "assets" / "site.css", _css().encode("utf-8"))
        _write_file(staging / "assets" / "site.js", _js().encode("utf-8"))
        inventory = _tree_inventory(staging)
        portable_assets = [
            item for item in inventory
            if not str(item["path"]).startswith("data/documents/")
        ]
        portable_asset_bytes = sum(int(item["bytes"]) for item in portable_assets)
        if portable_asset_bytes > max_portable_assets_bytes:
            raise ValueError(
                f"portable site assets exceed frozen limit: {portable_asset_bytes} > {max_portable_assets_bytes}"
            )
        site_manifest = {
            "schema_version": SITE_SCHEMA,
            "status": "passed",
            "packet_manifest": file_binding(packet_manifest),
            "requests": file_binding(requests_path),
            "responses": file_binding(responses_path),
            "site_secret_sha256": hashlib.sha256(secret).hexdigest(),
            "source_count": len(source_counts),
            "document_count": len(cards),
            "portable_assets": portable_assets,
            "portable_asset_bytes": portable_asset_bytes,
            "max_portable_assets_bytes": max_portable_assets_bytes,
            "files": inventory,
        }
        _write_json(staging / "site_manifest.json", site_manifest)
        if output_dir.exists():
            raise FileExistsError(f"site appeared during build: {output_dir}")
        os.rename(staging, output_dir)
        return site_manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


class NoDirectoryListingHandler(SimpleHTTPRequestHandler):
    def list_directory(self, path: str):  # type: ignore[override]
        self.send_error(403, "Directory listing disabled")
        return None

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        super().end_headers()


def serve_site(site_dir: Path, *, port: int, bind: str) -> None:
    site_dir = site_dir.resolve()
    if bind != "127.0.0.1":
        raise ValueError("review site must bind only to 127.0.0.1")
    manifest = site_dir / "site_manifest.json"
    value = read_json_object(manifest)
    if value.get("schema_version") != SITE_SCHEMA or value.get("status") != "passed":
        raise ValueError("site manifest is not passed")
    handler = functools.partial(NoDirectoryListingHandler, directory=str(site_dir))
    with ThreadingHTTPServer((bind, port), handler) as server:
        print(f"serving {site_dir} on http://{bind}:{port}/")
        server.serve_forever()


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--packet-root", type=Path, required=True)
    build.add_argument("--packet-manifest", type=Path, required=True)
    build.add_argument("--requests", type=Path, required=True)
    build.add_argument("--responses", type=Path, required=True)
    build.add_argument("--site-secret", required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--max-portable-assets-bytes", type=int, default=DEFAULT_MAX_PORTABLE_ASSETS_BYTES)
    serve = commands.add_parser("serve")
    serve.add_argument("--site-dir", type=Path, required=True)
    serve.add_argument("--port", type=int, required=True)
    serve.add_argument("--bind", default="127.0.0.1")
    args = parser.parse_args(argv)
    if args.command == "build":
        site_manifest = build_site(
            packet_root=args.packet_root,
            packet_manifest=args.packet_manifest,
            requests_path=args.requests,
            responses_path=args.responses,
            site_secret_hex=args.site_secret,
            output_dir=args.output_dir,
            max_portable_assets_bytes=args.max_portable_assets_bytes,
        )
        print(json.dumps({"ok": True, "documents": site_manifest["document_count"]}, sort_keys=True))
    else:
        serve_site(args.site_dir, port=args.port, bind=args.bind)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
