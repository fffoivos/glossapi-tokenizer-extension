#!/usr/bin/env python3
"""Immutable prestructural and Agent-2 handoff gates for Agent 1 v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PRESTRUCTURAL_SCHEMA = "agent1_full_corpus_v3_prestructural_manifest_v1"
STRUCTURAL_GATE_SCHEMA = "agent1_full_corpus_v3_structural_gate_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    if not path.is_file() or not path.stat().st_size:
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def cmd_freeze(args: argparse.Namespace) -> None:
    payload = {
        "schema_version": PRESTRUCTURAL_SCHEMA,
        "status": "prestructural_frozen",
        "run_id": args.run_id,
        "publish_permitted": False,
        "structural_state": "awaiting_agent2_handoff",
        "inputs": {
            "dedup_manifest": binding(args.dedup_manifest),
            "decontamination_manifest": binding(args.decontamination_manifest),
            "anonymization_manifest": binding(args.anonymization_manifest),
            "anonymization_ledger": binding(args.anonymization_ledger),
            "postmask_duplicate_report": binding(args.postmask_duplicate_report),
        },
        "corpus_root": str(args.corpus_root.resolve()),
    }
    duplicate = read_object(args.postmask_duplicate_report)
    if int(duplicate.get("material_new_duplicate_count", duplicate.get("new_duplicate_count", 0))) != 0:
        raise ValueError("post-mask duplicate report requires an explicit user decision before freeze")
    write_once(args.output, payload)
    print(json.dumps({"ok": True, "manifest": str(args.output), "publish_permitted": False}))


def cmd_gate(args: argparse.Namespace) -> None:
    prestructural = read_object(args.prestructural_manifest)
    if prestructural.get("schema_version") != PRESTRUCTURAL_SCHEMA:
        raise ValueError("v3 prestructural manifest required")
    payload: dict[str, Any] = {
        "schema_version": STRUCTURAL_GATE_SCHEMA,
        "prestructural_manifest": binding(args.prestructural_manifest),
        "source_allowlist": ["academic_ocr", "academic_sectioned"],
    }
    if args.model_handoff is None:
        payload.update({
            "mode": "no_op",
            "ready_for_application": False,
            "reason": "agent2_immutable_handoff_absent",
            "publish_permitted": False,
        })
    else:
        handoff = read_object(args.model_handoff)
        required = {
            "ready_for_corpus_application": True,
            "python_rust_probability_parity_passed": True,
            "python_rust_decoded_span_parity_passed": True,
            "source_balanced_safety_metrics_passed": True,
            "false_deletion_audit_passed": True,
        }
        for field, expected in required.items():
            if handoff.get(field) is not expected:
                raise ValueError(f"Agent 2 handoff does not pass required gate: {field}")
        payload.update({
            "mode": "eligible_child_run_only",
            "ready_for_application": True,
            "publish_permitted": False,
            "model_handoff": binding(args.model_handoff),
        })
    write_once(args.output, payload)
    print(json.dumps({"ok": True, "gate": str(args.output), "mode": payload["mode"]}))


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--run-id", required=True)
    freeze.add_argument("--dedup-manifest", type=Path, required=True)
    freeze.add_argument("--decontamination-manifest", type=Path, required=True)
    freeze.add_argument("--anonymization-manifest", type=Path, required=True)
    freeze.add_argument("--anonymization-ledger", type=Path, required=True)
    freeze.add_argument("--postmask-duplicate-report", type=Path, required=True)
    freeze.add_argument("--corpus-root", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.set_defaults(func=cmd_freeze)
    gate = sub.add_parser("gate")
    gate.add_argument("--prestructural-manifest", type=Path, required=True)
    gate.add_argument("--model-handoff", type=Path)
    gate.add_argument("--output", type=Path, required=True)
    gate.set_defaults(func=cmd_gate)
    return parser


def main() -> int:
    args = parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
