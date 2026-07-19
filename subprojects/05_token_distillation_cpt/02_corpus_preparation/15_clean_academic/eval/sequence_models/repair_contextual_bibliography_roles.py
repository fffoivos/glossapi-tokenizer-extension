#!/usr/bin/env python3
"""Repair impossible FILLER/CONTINUATION labels without mutating raw passes."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contract import canonical_json_sha256, sha256_file


SCHEMA_VERSION = "bibliography-contextual-role-repair-v1"
COMPONENT_ROLES = frozenset(
    {"ENTRY", "CONTINUATION", "FILLER", "BIB_HEADER", "BIB_SUBHEADER"}
)
CONTEXT_ONLY_ROLES = frozenset({"CONTINUATION", "FILLER"})
TAIL_THRESHOLDS = (10, 20, 30, 50, 100)


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


def repair_pass(
    role_pass: Mapping[str, Any],
    line_keys: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Return a derived pass, line audit, and summary.

    Components are maximal contiguous runs of bibliography-region roles in
    present-line order. FILLER and CONTINUATION are impossible in a component
    with no ENTRY anchor and are deterministically changed to OTHER.
    """

    raw_lines = role_pass.get("lines")
    if not isinstance(raw_lines, list):
        raise ValueError("role pass has no line inventory")
    lines_by_alias = {str(row.get("line_alias") or ""): dict(row) for row in raw_lines}
    if "" in lines_by_alias or len(lines_by_alias) != len(raw_lines):
        raise ValueError("role pass has empty or duplicate line aliases")

    keys_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key in line_keys:
        alias = str(key.get("line_alias") or "")
        document_alias = str(key.get("document_alias") or "")
        if not alias or not document_alias or alias not in lines_by_alias:
            raise ValueError("line key is incomplete or differs from the role pass")
        keys_by_document[document_alias].append(dict(key))
    key_aliases = {str(key["line_alias"]) for key in line_keys}
    if key_aliases != set(lines_by_alias):
        raise ValueError("line key must cover exactly the role-pass lines")
    for keys in keys_by_document.values():
        keys.sort(key=lambda row: (int(row["abs_idx"]), str(row["line_alias"])))

    corrected_by_alias = {alias: dict(row) for alias, row in lines_by_alias.items()}
    audit: list[dict[str, Any]] = []
    changed_by_source_role: Counter[tuple[str, str]] = Counter()
    changed_by_document: Counter[str] = Counter()
    component_counts: Counter[str] = Counter()
    anchored_context_distances: dict[str, list[int]] = defaultdict(list)

    for document_alias, keys in sorted(keys_by_document.items()):
        index = 0
        component_index = 0
        while index < len(keys):
            role = str(lines_by_alias[str(keys[index]["line_alias"])]["role"])
            if role not in COMPONENT_ROLES:
                index += 1
                continue
            end = index + 1
            while end < len(keys):
                next_role = str(lines_by_alias[str(keys[end]["line_alias"])]["role"])
                if next_role not in COMPONENT_ROLES:
                    break
                end += 1
            component = keys[index:end]
            entry_positions = [
                position
                for position, key in enumerate(component)
                if lines_by_alias[str(key["line_alias"])]["role"] == "ENTRY"
            ]
            component_counts["anchored" if entry_positions else "unanchored"] += 1
            role_counts = Counter(
                str(lines_by_alias[str(key["line_alias"])]["role"])
                for key in component
            )
            for position, key in enumerate(component):
                alias = str(key["line_alias"])
                row = lines_by_alias[alias]
                old_role = str(row["role"])
                if old_role not in CONTEXT_ONLY_ROLES:
                    continue
                if entry_positions:
                    anchored_context_distances[old_role].append(
                        min(abs(position - anchor) for anchor in entry_positions)
                    )
                    continue
                corrected_by_alias[alias]["role"] = "OTHER"
                corrected_by_alias[alias]["confidence"] = 1.0
                source = str(row.get("source") or key.get("source") or "")
                document_id = str(key.get("document_id") or "")
                changed_by_source_role[(source, old_role)] += 1
                changed_by_document[document_id] += 1
                audit.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "pass_id": str(role_pass.get("pass_id") or ""),
                        "source": source,
                        "document_alias": document_alias,
                        "document_id": document_id,
                        "line_alias": alias,
                        "line_id": str(key.get("line_id") or ""),
                        "abs_idx": int(key["abs_idx"]),
                        "old_role": old_role,
                        "new_role": "OTHER",
                        "original_confidence": float(row.get("confidence") or 0.0),
                        "component_index": component_index,
                        "component_start_abs_idx": int(component[0]["abs_idx"]),
                        "component_end_abs_idx": int(component[-1]["abs_idx"]),
                        "component_present_line_count": len(component),
                        "component_role_counts": dict(sorted(role_counts.items())),
                        "entry_anchor_count": 0,
                        "reason": "context_only_role_in_component_without_entry_anchor",
                    }
                )
            component_index += 1
            index = end

    corrected_lines = [corrected_by_alias[str(row["line_alias"])] for row in raw_lines]
    derived = dict(role_pass)
    derived.update(
        {
            "schema_version": SCHEMA_VERSION,
            "reviewer": f"{role_pass.get('reviewer', '')}+context-repair-v1",
            "lines": corrected_lines,
            "contextual_role_repair": {
                "rule": "FILLER/CONTINUATION -> OTHER when their contiguous bibliography-role component contains no ENTRY",
                "changed_line_count": len(audit),
                "original_pass_sha256": canonical_json_sha256(role_pass),
            },
        }
    )
    tail_profile = {
        role: {
            "anchored_line_count": len(distances),
            "maximum_nearest_entry_distance": max(distances, default=None),
            "counts_beyond_distance": {
                str(threshold): sum(distance > threshold for distance in distances)
                for threshold in TAIL_THRESHOLDS
            },
        }
        for role, distances in sorted(anchored_context_distances.items())
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "pass_id": str(role_pass.get("pass_id") or ""),
        "line_count": len(raw_lines),
        "changed_line_count": len(audit),
        "changed_by_source_and_old_role": {
            f"{source}:{role}": count
            for (source, role), count in sorted(changed_by_source_role.items())
        },
        "changed_document_count": len(changed_by_document),
        "top_changed_documents": [
            {"document_id": document_id, "changed_line_count": count}
            for document_id, count in changed_by_document.most_common(20)
        ],
        "component_counts": dict(sorted(component_counts.items())),
        "anchored_context_tail_profile": tail_profile,
    }
    return derived, audit, summary


def run(
    *,
    pass_path: Path,
    line_key_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    role_pass = _read_json(pass_path)
    line_keys = _read_jsonl(line_key_path)
    derived, audit, summary = repair_pass(role_pass, line_keys)
    output_dir.mkdir(parents=True)
    corrected_path = output_dir / "pass.context-repaired.json"
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
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = run(
        pass_path=args.pass_path.resolve(),
        line_key_path=args.line_key.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
