#!/usr/bin/env python3
"""Versioned operational bibliography roles and lossless v1-to-v2 migration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contract import canonical_json_sha256, sha256_file


CONTRACT_SCHEMA = "bibliography-role-contract-v2"
OVERLAY_SCHEMA = "bibliography-role-overlay-v3"
MIGRATION_SCHEMA = "bibliography-role-v1-to-v2-migration-v1"
OPERATIONAL_ROLES = (
    "ENTRY",
    "CONTINUATION",
    "FILLER",
    "BIB_HEADER",
    "BIB_SUBHEADER",
    "NON_BIB_HEADER",
    "OTHER",
)
ANNOTATION_ROLES = (*OPERATIONAL_ROLES, "UNKNOWN")
TRUSTED_STATUSES = frozenset({"AGREED_REVIEW", "ADJUDICATED"})
V1_TO_V2_ROLE = {
    "ENTRY_ANCHOR": "ENTRY",
    "CONTINUATION": "CONTINUATION",
    "FILLER": "FILLER",
    "HEADER": "BIB_HEADER",
    "SUBHEADER": "BIB_SUBHEADER",
    "NON_BIB": "OTHER",
    "UNKNOWN": "UNKNOWN",
}
ROLE_TO_ID = {role: index for index, role in enumerate(ANNOTATION_ROLES)}
ID_TO_ROLE = {index: role for role, index in ROLE_TO_ID.items()}


def _map_vote(value: Any) -> Any:
    if isinstance(value, str):
        return V1_TO_V2_ROLE.get(value, value)
    if isinstance(value, list):
        return [_map_vote(item) for item in value]
    return value


def migrate_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return a derived v2-role row without mutating the v1/v2 source row."""

    role = str(row.get("role", ""))
    if role not in V1_TO_V2_ROLE:
        raise ValueError(f"unsupported source role: {role!r}")
    migrated = dict(row)
    migrated["schema_version"] = OVERLAY_SCHEMA
    migrated["role"] = V1_TO_V2_ROLE[role]
    migrated["role_contract_schema"] = CONTRACT_SCHEMA
    migrated["migrated_from_role"] = role
    migrated["migrated_from_schema"] = str(row.get("schema_version", ""))
    if "raw_role_votes" in migrated:
        votes = migrated["raw_role_votes"]
        if not isinstance(votes, Mapping):
            raise ValueError("raw_role_votes must be an object")
        migrated["raw_role_votes"] = {
            str(reviewer): _map_vote(values) for reviewer, values in votes.items()
        }
    return migrated


def apply_heading_overrides(
    rows: Iterable[Mapping[str, Any]],
    overrides: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Apply only trusted typed-heading decisions to an already migrated overlay."""

    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in rows:
        row = dict(raw)
        key = (str(row.get("document_id", "")), str(row.get("line_id", "")))
        override = overrides.get(key)
        if override is not None:
            role = str(override.get("role", ""))
            status = str(override.get("role_status", ""))
            if role not in ANNOTATION_ROLES:
                raise ValueError(f"invalid heading override role for {key}: {role}")
            if status in TRUSTED_STATUSES or role == "UNKNOWN":
                for field in (
                    "role", "role_status", "role_confidence", "reviewers",
                    "review_case_ids", "raw_role_votes", "label_origin",
                ):
                    if field in override:
                        row[field] = override[field]
                row["heading_override_applied"] = True
            seen.add(key)
        output.append(row)
    missing = set(overrides) - seen
    if missing:
        raise ValueError(f"heading overrides contain {len(missing)} absent lines")
    return output


def merge_heading_overrides(
    rows: Iterable[Mapping[str, Any]],
    overrides: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Apply heading decisions and append reviewed lines absent from the old overlay."""

    existing = [dict(row) for row in rows]
    present = {
        (str(row.get("document_id", "")), str(row.get("line_id", ""))) for row in existing
    }
    applied = apply_heading_overrides(
        existing, {key: value for key, value in overrides.items() if key in present}
    )
    for key in sorted(set(overrides) - present):
        row = dict(overrides[key])
        if row.get("schema_version") != OVERLAY_SCHEMA:
            raise ValueError(f"new heading override has wrong schema for {key}")
        if (str(row.get("document_id", "")), str(row.get("line_id", ""))) != key:
            raise ValueError(f"new heading override identity mismatch for {key}")
        applied.append(row)
    applied.sort(key=lambda row: (str(row["document_id"]), int(row["abs_idx"]), str(row["line_id"])))
    return applied


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected object")
            yield value


def _write_jsonl_new(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def run_migration(input_path: Path, output_path: Path, receipt_path: Path) -> dict[str, Any]:
    source_rows = list(_iter_jsonl(input_path))
    migrated = [migrate_row(row) for row in source_rows]
    identities = [(row.get("document_id"), row.get("line_id")) for row in migrated]
    if len(identities) != len(set(identities)):
        raise ValueError("migration produced repeated line identities")
    _write_jsonl_new(output_path, migrated)
    receipt = {
        "schema_version": MIGRATION_SCHEMA,
        "status": "passed_lossless_role_migration",
        "input": {"path": str(input_path), "sha256": sha256_file(input_path)},
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "line_count": len(migrated),
        "input_content_sha256": canonical_json_sha256(source_rows),
        "output_content_sha256": canonical_json_sha256(migrated),
        "role_mapping": V1_TO_V2_ROLE,
    }
    _write_json_new(receipt_path, receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_migration(args.input.resolve(), args.output.resolve(), args.receipt.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
