#!/usr/bin/env python3
"""Immutable user-confirmed source admission for the Agent 1 v3 lane."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


PACKET_SCHEMA = "agent1_full_corpus_v3_source_admission_packet_v1"
CONFIRMATION_SCHEMA = "agent1_full_corpus_v3_source_admission_confirmation_v1"
DECISIONS = {"include", "include_after_cleaning", "low_weight", "exclude", "quarantine"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def binding(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_decisions(value: Any) -> list[dict[str, str]]:
    rows = value.get("sources") if isinstance(value, dict) else value
    if not isinstance(rows, list) or not rows:
        raise ValueError("source decisions must be a non-empty list")
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("source decision must be an object")
        source_id = str(row.get("source_id") or row.get("source_dataset") or "")
        decision = str(row.get("decision") or "")
        if not source_id or source_id in seen or decision not in DECISIONS:
            raise ValueError(f"invalid/duplicate source decision: {source_id!r} {decision!r}")
        seen.add(source_id)
        result.append({"source_id": source_id, "decision": decision})
    return sorted(result, key=lambda row: row["source_id"])


def cmd_build_packet(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite admission packet: {args.output}")
    resolution = read_object(args.review_resolution)
    if int(resolution.get("pending_count", resolution.get("pending_adjudications", -1))) != 0:
        raise ValueError("review resolution has pending adjudications")
    decisions = validate_decisions(read_object(args.proposed_decisions))
    payload = {
        "schema_version": PACKET_SCHEMA,
        "status": "pending_user_confirmation",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": args.run_id,
        "inputs": {
            "quality_summary": binding(args.quality_summary),
            "review_selection": binding(args.review_selection),
            "review_execution": binding(args.review_execution),
            "review_resolution": binding(args.review_resolution),
            "lineage_summary": binding(args.lineage_summary),
            "license_adjudication": binding(args.license_adjudication),
        },
        "sources": decisions,
    }
    write_exclusive(args.output, payload)
    print(json.dumps({"ok": True, "packet": str(args.output)}))


def cmd_confirm(args: argparse.Namespace) -> None:
    packet = read_object(args.packet)
    if packet.get("schema_version") != PACKET_SCHEMA or packet.get("status") != "pending_user_confirmation":
        raise ValueError("admission packet is not a pending v3 packet")
    actual = sha256(args.packet)
    if actual != args.packet_sha256:
        raise ValueError(f"admission packet sha256 mismatch: {actual} != {args.packet_sha256}")
    approved = validate_decisions(read_object(args.decisions))
    proposed = validate_decisions(packet)
    proposed_ids = {row["source_id"] for row in proposed}
    if {row["source_id"] for row in approved} != proposed_ids:
        raise ValueError("confirmation must decide exactly the packet's sources")
    payload = {
        "schema_version": CONFIRMATION_SCHEMA,
        "status": "approved",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": packet.get("run_id"),
        "packet": binding(args.packet),
        "sources": approved,
    }
    write_exclusive(args.output, payload)
    print(json.dumps({"ok": True, "confirmation": str(args.output)}))


def cmd_validate(args: argparse.Namespace) -> None:
    confirmation = read_object(args.confirmation)
    if confirmation.get("schema_version") != CONFIRMATION_SCHEMA or confirmation.get("status") != "approved":
        raise ValueError("not an approved v3 admission confirmation")
    packet = confirmation.get("packet")
    if not isinstance(packet, dict):
        raise ValueError("confirmation lacks packet binding")
    source = Path(str(packet.get("path", "")))
    if not source.is_file() or source.stat().st_size != packet.get("bytes") or sha256(source) != packet.get("sha256"):
        raise ValueError("confirmed admission packet drift")
    validate_decisions(confirmation)
    print(json.dumps({"ok": True, "sources": len(confirmation["sources"])}))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-packet")
    build.add_argument("--run-id", required=True)
    build.add_argument("--quality-summary", type=Path, required=True)
    build.add_argument("--review-selection", type=Path, required=True)
    build.add_argument("--review-execution", type=Path, required=True)
    build.add_argument("--review-resolution", type=Path, required=True)
    build.add_argument("--lineage-summary", type=Path, required=True)
    build.add_argument("--license-adjudication", type=Path, required=True)
    build.add_argument("--proposed-decisions", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.set_defaults(func=cmd_build_packet)
    confirm = sub.add_parser("confirm")
    confirm.add_argument("--packet", type=Path, required=True)
    confirm.add_argument("--packet-sha256", required=True)
    confirm.add_argument("--decisions", type=Path, required=True)
    confirm.add_argument("--output", type=Path, required=True)
    confirm.set_defaults(func=cmd_confirm)
    validate = sub.add_parser("validate")
    validate.add_argument("--confirmation", type=Path, required=True)
    validate.set_defaults(func=cmd_validate)
    return result


def main() -> int:
    args = parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
