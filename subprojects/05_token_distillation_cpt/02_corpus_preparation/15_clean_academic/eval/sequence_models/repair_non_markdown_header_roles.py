#!/usr/bin/env python3
"""Drop inferred non-Markdown header roles without mutating source passes."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contract import canonical_json_sha256, sha256_file
from .deterministic_structure import _ATX_HEADING


SCHEMA_VERSION = "bibliography-markdown-header-repair-v1"
HEADER_ROLES = frozenset({"BIB_HEADER", "BIB_SUBHEADER", "NON_BIB_HEADER"})


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_text_new(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


def _write_json_new(path: Path, value: Any) -> None:
    _write_text_new(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl_new(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    _write_text_new(
        path,
        "".join(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
            for value in values
        ),
    )


def _text_by_line_id(documents: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for document in documents:
        document_id = str(document.get("document_id") or "")
        lines = document.get("lines")
        if not document_id or not isinstance(lines, list):
            raise ValueError("document inventory is incomplete")
        for line in lines:
            line_id = str(line.get("line_id") or "")
            text = line.get("text")
            if not line_id or not isinstance(text, str) or line_id in result:
                raise ValueError("document lines have missing or duplicate identifiers")
            result[line_id] = text
    return result


def repair_pass(
    role_pass: Mapping[str, Any],
    line_keys: Sequence[Mapping[str, Any]],
    documents: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Change header roles without an ATX Markdown marker to OTHER."""

    raw_lines = role_pass.get("lines")
    if not isinstance(raw_lines, list):
        raise ValueError("role pass has no line inventory")
    lines_by_alias = {str(row.get("line_alias") or ""): dict(row) for row in raw_lines}
    if "" in lines_by_alias or len(lines_by_alias) != len(raw_lines):
        raise ValueError("role pass has empty or duplicate line aliases")

    keys_by_alias = {str(key.get("line_alias") or ""): dict(key) for key in line_keys}
    if "" in keys_by_alias or len(keys_by_alias) != len(line_keys):
        raise ValueError("line key has empty or duplicate line aliases")
    if set(keys_by_alias) != set(lines_by_alias):
        raise ValueError("line key must cover exactly the role-pass lines")

    text_by_line_id = _text_by_line_id(documents)
    keyed_line_ids = {str(key.get("line_id") or "") for key in line_keys}
    if "" in keyed_line_ids or not keyed_line_ids.issubset(text_by_line_id):
        raise ValueError("documents do not cover every keyed line")

    corrected_by_alias = {alias: dict(row) for alias, row in lines_by_alias.items()}
    audit: list[dict[str, Any]] = []
    changed_by_source_role: Counter[tuple[str, str]] = Counter()
    retained_by_source_role: Counter[tuple[str, str]] = Counter()
    changed_by_document: Counter[str] = Counter()

    for alias, row in lines_by_alias.items():
        old_role = str(row.get("role") or "")
        if old_role not in HEADER_ROLES:
            continue
        key = keys_by_alias[alias]
        line_id = str(key["line_id"])
        text = text_by_line_id[line_id]
        source = str(row.get("source") or key.get("source") or "")
        if _ATX_HEADING.match(text):
            retained_by_source_role[(source, old_role)] += 1
            continue
        corrected_by_alias[alias]["role"] = "OTHER"
        corrected_by_alias[alias]["confidence"] = 1.0
        document_id = str(key.get("document_id") or "")
        changed_by_source_role[(source, old_role)] += 1
        changed_by_document[document_id] += 1
        audit.append(
            {
                "schema_version": SCHEMA_VERSION,
                "pass_id": str(role_pass.get("pass_id") or ""),
                "source": source,
                "document_alias": str(key.get("document_alias") or ""),
                "document_id": document_id,
                "line_alias": alias,
                "line_id": line_id,
                "abs_idx": int(key["abs_idx"]),
                "old_role": old_role,
                "new_role": "OTHER",
                "original_confidence": float(row.get("confidence") or 0.0),
                "text": text,
                "reason": "header_role_without_atx_markdown_heading_syntax",
            }
        )

    corrected_lines = [corrected_by_alias[str(row["line_alias"])] for row in raw_lines]
    derived = dict(role_pass)
    derived.update(
        {
            "schema_version": SCHEMA_VERSION,
            "reviewer": f"{role_pass.get('reviewer', '')}+markdown-header-repair-v1",
            "lines": corrected_lines,
            "markdown_header_repair": {
                "rule": "BIB_HEADER/BIB_SUBHEADER/NON_BIB_HEADER -> OTHER unless the source line matches ATX Markdown heading syntax",
                "atx_pattern": _ATX_HEADING.pattern,
                "changed_line_count": len(audit),
                "original_pass_sha256": canonical_json_sha256(role_pass),
            },
        }
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "pass_id": str(role_pass.get("pass_id") or ""),
        "line_count": len(raw_lines),
        "atx_pattern": _ATX_HEADING.pattern,
        "changed_line_count": len(audit),
        "changed_by_source_and_old_role": {
            f"{source}:{role}": count
            for (source, role), count in sorted(changed_by_source_role.items())
        },
        "retained_markdown_headers_by_source_and_role": {
            f"{source}:{role}": count
            for (source, role), count in sorted(retained_by_source_role.items())
        },
        "changed_document_count": len(changed_by_document),
        "top_changed_documents": [
            {"document_id": document_id, "changed_line_count": count}
            for document_id, count in changed_by_document.most_common(20)
        ],
    }
    return derived, audit, summary


def run(
    *,
    pass_path: Path,
    line_key_path: Path,
    documents_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    role_pass = _read_json(pass_path)
    line_keys = _read_jsonl(line_key_path)
    documents = _read_jsonl(documents_path)
    derived, audit, summary = repair_pass(role_pass, line_keys, documents)
    output_dir.mkdir(parents=True)
    corrected_path = output_dir / "pass.markdown-header-repaired.json"
    audit_path = output_dir / "changes.audit.jsonl"
    _write_json_new(corrected_path, derived)
    _write_jsonl_new(audit_path, audit)
    receipt = {
        **summary,
        "inputs": {
            "pass_path": str(pass_path),
            "pass_sha256": sha256_file(pass_path),
            "line_key_path": str(line_key_path),
            "line_key_sha256": sha256_file(line_key_path),
            "documents_path": str(documents_path),
            "documents_sha256": sha256_file(documents_path),
        },
        "outputs": {
            "corrected_pass_path": str(corrected_path),
            "corrected_pass_sha256": sha256_file(corrected_path),
            "audit_path": str(audit_path),
            "audit_sha256": sha256_file(audit_path),
        },
        "original_data_mutated": False,
        "evaluation_use": "post-hoc annotation repair audit; not the original sealed gate",
    }
    _write_json_new(output_dir / "receipt.json", receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass", dest="pass_path", type=Path, required=True)
    parser.add_argument("--line-key", type=Path, required=True)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = run(
        pass_path=args.pass_path.resolve(),
        line_key_path=args.line_key.resolve(),
        documents_path=args.documents.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
