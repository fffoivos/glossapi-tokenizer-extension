#!/usr/bin/env python3
"""Apply provenance-bound human role decisions to a dual-review overlay."""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .bibliography_role_review_runner import BOUNDARIES, ROLES
from .contract import sha256_file


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected object")
            yield row


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def apply_human(
    overlay: Sequence[Mapping[str, Any]], export: Mapping[str, Any], *, expected_packet_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if export.get("schema_version") != "bibliography-role-human-audit-export-v1":
        raise ValueError("unsupported human audit export")
    if export.get("packet_sha256") != expected_packet_sha256:
        raise ValueError("human audit packet hash mismatch")
    reviewer = export.get("reviewer")
    decisions = export.get("decisions")
    if not isinstance(reviewer, str) or not reviewer or not isinstance(decisions, list):
        raise ValueError("human audit lacks reviewer/decisions")
    decision_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for decision in decisions:
        if not isinstance(decision, Mapping):
            raise ValueError("human decision is not an object")
        key = (str(decision.get("document_id", "")), str(decision.get("line_id", "")))
        if not all(key) or key in decision_by_key:
            raise ValueError("human decision identity is empty/repeated")
        if decision.get("role") not in ROLES or decision.get("boundary") not in BOUNDARIES:
            raise ValueError(f"invalid human role/boundary for {key}")
        decision_by_key[key] = decision
    output, seen = [], set()
    role_changes = boundary_changes = all_three_role_agree = 0
    for original in overlay:
        row = dict(original)
        key = (str(row.get("document_id", "")), str(row.get("line_id", "")))
        decision = decision_by_key.get(key)
        if decision is not None:
            if int(decision.get("abs_idx", -1)) != row.get("abs_idx"):
                raise ValueError(f"human line coordinate mismatch: {key}")
            seen.add(key)
            human_role, human_boundary = str(decision["role"]), str(decision["boundary"])
            raw_roles = {name: list(values) for name, values in row["raw_role_votes"].items()}
            raw_boundaries = {
                name: list(values) for name, values in row["raw_boundary_votes"].items()
            }
            automatic_roles = {value for values in raw_roles.values() for value in values}
            automatic_boundaries = {value for values in raw_boundaries.values() for value in values}
            all_three_role_agree += int(automatic_roles == {human_role})
            role_changes += int(row["role"] != human_role or row["role_status"] == "UNRESOLVED")
            boundary_changes += int(
                row["boundary_flag"] != human_boundary or row["boundary_status"] == "UNRESOLVED"
            )
            raw_roles[reviewer] = [human_role]
            raw_boundaries[reviewer] = [human_boundary]
            row.update(
                role=human_role,
                role_status="ADJUDICATED",
                boundary_flag=human_boundary,
                boundary_status="ADJUDICATED",
                role_confidence=1.0,
                boundary_confidence=1.0,
                label_origin="human_adjudicated_after_dual_codex",
                reviewers=list(dict.fromkeys([*row["reviewers"], reviewer])),
                raw_role_votes=raw_roles,
                raw_boundary_votes=raw_boundaries,
            )
        output.append(row)
    if seen != set(decision_by_key):
        raise ValueError(f"{len(set(decision_by_key) - seen)} human decisions are absent from overlay")
    report = {
        "schema_version": "bibliography-role-human-adjudication-report-v1",
        "status": "passed",
        "reviewer": reviewer,
        "human_decision_count": len(decision_by_key),
        "role_change_or_resolution_count": role_changes,
        "boundary_change_or_resolution_count": boundary_changes,
        "all_three_exact_role_agreement_count": all_three_role_agree,
        "human_role_counts": dict(collections.Counter(str(row["role"]) for row in output if row["role_status"] == "ADJUDICATED")),
    }
    return output, report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--automatic-overlay", required=True)
    parser.add_argument("--human-export", required=True)
    parser.add_argument("--expected-packet-sha256", required=True)
    parser.add_argument("--overlay-out", required=True)
    parser.add_argument("--report-out", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    automatic_path, human_path = Path(args.automatic_overlay).resolve(), Path(args.human_export).resolve()
    rows, report = apply_human(
        list(_iter_jsonl(automatic_path)), json.loads(human_path.read_text(encoding="utf-8")),
        expected_packet_sha256=args.expected_packet_sha256,
    )
    overlay_path = Path(args.overlay_out).resolve()
    _write_jsonl(overlay_path, rows)
    report["inputs"] = {
        "automatic_overlay_sha256": sha256_file(automatic_path),
        "human_export_sha256": sha256_file(human_path),
        "packet_sha256": args.expected_packet_sha256,
    }
    report["overlay_sha256"] = sha256_file(overlay_path)
    _write_json(Path(args.report_out).resolve(), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
