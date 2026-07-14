#!/usr/bin/env python3
"""Audit a built Agent 1 review site with GlossAPI's OCR VLM-loop guards.

The raw-review site intentionally retains exact, uncleaned OCR text.  This
command applies the finalized-text detectors from ``glossapi.ocr.utils.cleaning``
to every presented document, writes a compact audit beside the site data, and
refreshes the site's integrity manifest.  It does not alter document text.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


AUDIT_SCHEMA = "agent1_v4_vlm_repetition_audit_v1"
SITE_SCHEMA = "agent1_v4_raw_review_site_manifest_v1"
AUDIT_RELATIVE_PATH = Path("data/vlm_repetition_audit.json")
DEFAULT_GLOSSAPI_ROOT = Path.home() / "Projects/glossapi-development"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: object) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _load_cleaning_module(glossapi_root: Path) -> tuple[object, Path]:
    module_path = (glossapi_root / "src/glossapi/ocr/utils/cleaning.py").resolve()
    if not module_path.is_file():
        raise FileNotFoundError(f"GlossAPI OCR cleaning module not found: {module_path}")
    specification = importlib.util.spec_from_file_location("agent1_v4_glossapi_cleaning", module_path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not import GlossAPI OCR cleaning module: {module_path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    for name in ("detect_early_stop_rule_details", "detect_early_stop_index", "StreamingGarbageDetector"):
        if not callable(getattr(module, name, None)):
            raise RuntimeError(f"GlossAPI OCR cleaning module lacks {name}")
    return module, module_path


def _cards_by_opaque_id(site_dir: Path) -> dict[str, Mapping[str, object]]:
    cards: dict[str, Mapping[str, object]] = {}
    source_paths = sorted((site_dir / "data/sources").glob("*.json"))
    if not source_paths:
        raise FileNotFoundError("review site has no per-source card files")
    for source_path in source_paths:
        payload = _read_json(source_path)
        source_cards = payload.get("cards")
        if not isinstance(source_cards, list):
            raise ValueError(f"{source_path}: cards must be a list")
        for card in source_cards:
            if not isinstance(card, Mapping):
                raise ValueError(f"{source_path}: card must be an object")
            opaque_id = card.get("opaque_id")
            if not isinstance(opaque_id, str) or len(opaque_id) != 64 or opaque_id in cards:
                raise ValueError(f"{source_path}: invalid or duplicate opaque_id")
            cards[opaque_id] = card
    return cards


def _streaming_reason(cleaning: object, text: str) -> str | None:
    detector = cleaning.StreamingGarbageDetector()
    for offset in range(0, len(text), 257):
        if detector.feed(text[offset : offset + 257]):
            break
    reason = detector.triggered_reason
    return str(reason) if reason is not None else None


def audit_site(*, site_dir: Path, glossapi_root: Path) -> dict[str, object]:
    site_dir = site_dir.resolve()
    manifest_path = site_dir / "site_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != SITE_SCHEMA or manifest.get("status") != "passed":
        raise ValueError("review site manifest is not passed")

    cleaning, cleaning_path = _load_cleaning_module(glossapi_root)
    cards = _cards_by_opaque_id(site_dir)
    documents_dir = site_dir / "data/documents"
    document_paths = sorted(documents_dir.glob("*.json"))
    if len(document_paths) != len(cards):
        raise ValueError("review site source-card/document count mismatch")

    rows: list[dict[str, object]] = []
    rule_counts: Counter[str] = Counter()
    source_hit_counts: Counter[str] = Counter()
    for document_path in document_paths:
        payload = _read_json(document_path)
        opaque_id = payload.get("opaque_id")
        text = payload.get("text")
        if not isinstance(opaque_id, str) or document_path.stem != opaque_id or opaque_id not in cards:
            raise ValueError(f"{document_path}: opaque_id does not close with source cards")
        if not isinstance(text, str):
            raise ValueError(f"{document_path}: text must be a string")
        details = cleaning.detect_early_stop_rule_details(text)
        if not isinstance(details, list):
            raise RuntimeError("GlossAPI detector returned invalid rule details")
        normalized_details: list[dict[str, object]] = []
        for detail in details:
            if not isinstance(detail, Mapping) or not isinstance(detail.get("rule"), str) or not isinstance(detail.get("cut_index"), int):
                raise RuntimeError("GlossAPI detector emitted invalid rule detail")
            normalized_details.append({"rule": detail["rule"], "cut_index": detail["cut_index"]})
            rule_counts[str(detail["rule"])] += 1
        streaming_reason = _streaming_reason(cleaning, text)
        earliest_cut_index = min((int(detail["cut_index"]) for detail in normalized_details), default=None)
        if cleaning.detect_early_stop_index(text) != earliest_cut_index:
            raise RuntimeError("GlossAPI early-stop detector returned inconsistent cut indices")
        card = cards[opaque_id]
        source_id = str(card["source_id"])
        if normalized_details or streaming_reason is not None:
            source_hit_counts[source_id] += 1
        rows.append(
            {
                "opaque_id": opaque_id,
                "request_id": str(card["request_id"]),
                "source_id": source_id,
                "source_doc_id": str(card["source_doc_id"]),
                "text_sha256": sha256_bytes(text.encode("utf-8")),
                "rules": normalized_details,
                "earliest_cut_index": earliest_cut_index,
                "streaming_reason": streaming_reason,
            }
        )

    if len(rows) != int(manifest.get("document_count", -1)):
        raise ValueError("review site manifest/document count mismatch")
    hit_count = sum(1 for row in rows if row["rules"] or row["streaming_reason"] is not None)
    return {
        "schema_version": AUDIT_SCHEMA,
        "status": "passed",
        "detector": {
            "implementation": "glossapi.ocr.utils.cleaning",
            "cleaning_module_sha256": sha256_bytes(cleaning_path.read_bytes()),
            "final_text_rules": [
                "repeated_char_run",
                "repeated_line_run",
                "symbol_garbage",
                "numeric_list_garbage",
                "descending_dotted_numeric_run",
            ],
            "stream_chunk_chars": 257,
            "note": "The token-ID triplet detector is not run: review samples retain text, not VLLM token IDs.",
        },
        "summary": {
            "document_count": len(rows),
            "documents_with_any_trigger": hit_count,
            "rule_trigger_counts": dict(sorted(rule_counts.items())),
            "documents_with_any_trigger_by_source": dict(sorted(source_hit_counts.items())),
        },
        "documents": rows,
    }


def _inventory_without_manifest(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative == "site_manifest.json":
            continue
        if path.is_symlink():
            raise ValueError(f"review site contains a forbidden symlink: {relative}")
        if path.is_file():
            payload = path.read_bytes()
            records.append({"path": relative, "bytes": len(payload), "sha256": sha256_bytes(payload)})
    return records


def write_audit(*, site_dir: Path, glossapi_root: Path) -> dict[str, object]:
    site_dir = site_dir.resolve()
    audit = audit_site(site_dir=site_dir, glossapi_root=glossapi_root)
    audit_path = site_dir / AUDIT_RELATIVE_PATH
    _write_json(audit_path, audit)

    manifest_path = site_dir / "site_manifest.json"
    manifest = _read_json(manifest_path)
    inventory = _inventory_without_manifest(site_dir)
    portable_assets = [item for item in inventory if not str(item["path"]).startswith("data/documents/")]
    portable_asset_bytes = sum(int(item["bytes"]) for item in portable_assets)
    max_bytes = int(manifest["max_portable_assets_bytes"])
    if portable_asset_bytes > max_bytes:
        raise ValueError(f"portable site assets exceed frozen limit: {portable_asset_bytes} > {max_bytes}")
    audit_payload = audit_path.read_bytes()
    manifest["vlm_repetition_audit"] = {
        "path": AUDIT_RELATIVE_PATH.as_posix(),
        "sha256": sha256_bytes(audit_payload),
        "bytes": len(audit_payload),
    }
    manifest["portable_assets"] = portable_assets
    manifest["portable_asset_bytes"] = portable_asset_bytes
    manifest["files"] = inventory
    _write_json(manifest_path, manifest)
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, required=True)
    parser.add_argument("--glossapi-root", type=Path, default=DEFAULT_GLOSSAPI_ROOT)
    args = parser.parse_args(argv)
    audit = write_audit(site_dir=args.site_dir, glossapi_root=args.glossapi_root)
    print(json.dumps(audit["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
