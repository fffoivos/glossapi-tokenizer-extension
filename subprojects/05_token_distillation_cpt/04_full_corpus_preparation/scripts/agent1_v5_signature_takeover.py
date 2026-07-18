#!/usr/bin/env python3
"""Small dependency-free verifier for the one-time serial-chain handoff.

This module is copied beside the live legacy chain helper immediately before
the handoff.  It deliberately uses only the Python standard library: it is
safe to run in the existing debug job after the signature computation, without
depending on the accelerated pipeline environment.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


REQUEST_SCHEMA = "agent1_v5_dedup_acceleration_takeover_request_v1"
ARM_SCHEMA = "agent1_v5_dedup_acceleration_takeover_arm_v1"
STOP_SCHEMA = "agent1_v5_dedup_acceleration_sentinel_stop_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_path(path: Path) -> str:
    return str(path.resolve())


def _read(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.partial-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        os.link(temporary, path)
    finally:
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass


def _binding(path: Path) -> dict[str, str]:
    return {"path": canonical_path(path), "sha256": sha256_file(path)}


def _validate_request(
    request_path: Path,
    *,
    run_root: Path,
    coord_root: Path,
    legacy_pipeline_root: Path,
    active_helper: Path,
    takeover_tool: Path,
    task_index: int,
) -> Mapping[str, Any]:
    request = _read(request_path)
    if request.get("schema_version") != REQUEST_SCHEMA or request.get("status") != "passed":
        raise ValueError("takeover request is not a passed request")
    expected_paths = {
        "run_root": run_root,
        "coord_root": coord_root,
        "legacy_pipeline_root": legacy_pipeline_root,
        "active_helper": active_helper,
        "takeover_tool": takeover_tool,
    }
    for field, path in expected_paths.items():
        if request.get(field) != canonical_path(path):
            raise ValueError(f"takeover request path drift: {field}")
    if int(request.get("stop_after_rank", -1)) != task_index:
        raise ValueError("takeover request stop rank does not match this serial task")
    expected_hashes = {
        "guarded_helper_sha256": active_helper,
        "takeover_tool_sha256": takeover_tool,
        "run_contract_sha256": run_root / "run_contract.json",
        "combined_manifest_sha256": run_root / "release-pre-dedup" / "manifests" / "combined_manifest.json",
        "runtime_receipt_sha256": run_root / "datatrove_runtime.json",
        "full_input_audit_sha256": run_root / "dedup_full_input_audit.json",
    }
    for field, path in expected_hashes.items():
        if not path.is_file() or request.get(field) != sha256_file(path):
            raise ValueError(f"takeover request checksum drift: {field}")
    audit = _read(run_root / "dedup_full_input_audit.json")
    if audit.get("schema_version") != "agent1_v5_dedup_full_input_audit_v1" or audit.get("status") != "passed":
        raise ValueError("full input audit is not passed")
    if audit.get("combined_manifest_sha256") != request["combined_manifest_sha256"]:
        raise ValueError("full input audit is not bound to the requested manifest")
    return request


def create_request(args: argparse.Namespace) -> int:
    run = args.run_root.resolve()
    coord = args.coord_root.resolve()
    legacy_pipeline = args.legacy_pipeline_root.resolve()
    active = args.active_helper.resolve()
    guarded = args.guarded_helper.resolve()
    tool = args.takeover_tool.resolve()
    original = args.original_helper.resolve()
    if int(args.stop_after_rank) < 0:
        raise ValueError("stop rank must be non-negative")
    for path in (active, guarded, tool, original):
        if not path.is_file():
            raise FileNotFoundError(path)
    if active != original:
        raise ValueError("active helper must still be the verified original before arming")
    audit = _read(run / "dedup_full_input_audit.json")
    manifest = run / "release-pre-dedup" / "manifests" / "combined_manifest.json"
    if audit.get("schema_version") != "agent1_v5_dedup_full_input_audit_v1" or audit.get("status") != "passed":
        raise ValueError("full input audit is not passed")
    if audit.get("combined_manifest_sha256") != sha256_file(manifest):
        raise ValueError("full input audit does not bind the combined manifest")
    value = {
        "schema_version": REQUEST_SCHEMA,
        "status": "passed",
        "created_at": _now(),
        "run_root": canonical_path(run),
        "coord_root": canonical_path(coord),
        "legacy_pipeline_root": canonical_path(legacy_pipeline),
        "active_helper": canonical_path(active),
        "takeover_tool": canonical_path(active.parent / tool.name),
        "stop_after_rank": int(args.stop_after_rank),
        "original_helper_sha256": sha256_file(original),
        "guarded_helper_sha256": sha256_file(guarded),
        "takeover_tool_sha256": sha256_file(tool),
        "run_contract_sha256": sha256_file(run / "run_contract.json"),
        "combined_manifest_sha256": sha256_file(manifest),
        "runtime_receipt_sha256": sha256_file(run / "datatrove_runtime.json"),
        "full_input_audit_sha256": sha256_file(run / "dedup_full_input_audit.json"),
    }
    _write_immutable(args.output, value)
    print(json.dumps({"ok": True, "stop_after_rank": value["stop_after_rank"]}, sort_keys=True))
    return 0


def validate_request(args: argparse.Namespace) -> int:
    if not args.request.exists():
        print(json.dumps({"state": "absent"}, sort_keys=True))
        return 0
    value = _validate_request(
        args.request,
        run_root=args.run_root.resolve(),
        coord_root=args.coord_root.resolve(),
        legacy_pipeline_root=args.legacy_pipeline_root.resolve(),
        active_helper=args.active_helper.resolve(),
        takeover_tool=args.takeover_tool.resolve(),
        task_index=int(args.task_index),
    )
    print(json.dumps({"state": "valid", "stop_after_rank": value["stop_after_rank"]}, sort_keys=True))
    return 0


def write_arm(args: argparse.Namespace) -> int:
    request = _validate_request(
        args.request,
        run_root=args.run_root.resolve(),
        coord_root=args.coord_root.resolve(),
        legacy_pipeline_root=args.legacy_pipeline_root.resolve(),
        active_helper=args.active_helper.resolve(),
        takeover_tool=args.takeover_tool.resolve(),
        task_index=int(args.stop_after_rank),
    )
    value = {
        "schema_version": ARM_SCHEMA,
        "status": "passed",
        "created_at": _now(),
        "request_sha256": sha256_file(args.request),
        "active_helper_sha256": sha256_file(args.active_helper),
        "takeover_tool_sha256": sha256_file(args.takeover_tool),
        "stop_after_rank": int(args.stop_after_rank),
        "run_root": request["run_root"],
        "coord_root": request["coord_root"],
    }
    _write_immutable(args.output, value)
    print(json.dumps({"ok": True, "armed": value["stop_after_rank"]}, sort_keys=True))
    return 0


def write_stop(args: argparse.Namespace) -> int:
    request = _validate_request(
        args.request,
        run_root=args.run_root.resolve(),
        coord_root=args.coord_root.resolve(),
        legacy_pipeline_root=args.legacy_pipeline_root.resolve(),
        active_helper=args.active_helper.resolve(),
        takeover_tool=args.takeover_tool.resolve(),
        task_index=int(args.task_index),
    )
    receipt = _read(args.signature_receipt)
    outputs = receipt.get("outputs")
    if receipt.get("status") != "passed" or int(receipt.get("task_index", -1)) != int(args.task_index):
        raise ValueError("signature receipt is not a passed receipt for the stop rank")
    if not isinstance(outputs, list) or len(outputs) != 32:
        raise ValueError("signature receipt lacks 32 outputs")
    value = {
        "schema_version": STOP_SCHEMA,
        "status": "passed",
        "created_at": _now(),
        "request_sha256": sha256_file(args.request),
        "signature_receipt": _binding(args.signature_receipt),
        "active_helper_sha256": sha256_file(args.active_helper),
        "takeover_tool_sha256": sha256_file(args.takeover_tool),
        "stopped_after_rank": int(args.task_index),
        "first_missing_rank": int(args.task_index) + 1,
        "run_root": request["run_root"],
        "coord_root": request["coord_root"],
        "successor_submitted": False,
    }
    _write_immutable(args.output, value)
    print(json.dumps({"ok": True, "stopped_after_rank": value["stopped_after_rank"]}, sort_keys=True))
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("create-request")
    command.add_argument("--run-root", type=Path, required=True)
    command.add_argument("--coord-root", type=Path, required=True)
    command.add_argument("--legacy-pipeline-root", type=Path, required=True)
    command.add_argument("--active-helper", type=Path, required=True)
    command.add_argument("--original-helper", type=Path, required=True)
    command.add_argument("--guarded-helper", type=Path, required=True)
    command.add_argument("--takeover-tool", type=Path, required=True)
    command.add_argument("--stop-after-rank", type=int, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=create_request)

    for name, function in (("validate-request", validate_request), ("write-arm", write_arm), ("write-stop", write_stop)):
        command = commands.add_parser(name)
        command.add_argument("--request", type=Path, required=True)
        command.add_argument("--run-root", type=Path, required=True)
        command.add_argument("--coord-root", type=Path, required=True)
        command.add_argument("--legacy-pipeline-root", type=Path, required=True)
        command.add_argument("--active-helper", type=Path, required=True)
        command.add_argument("--takeover-tool", type=Path, required=True)
        command.add_argument("--task-index", type=int, required=name != "write-arm")
        if name == "write-arm":
            command.add_argument("--stop-after-rank", type=int, required=True)
        if name == "write-stop":
            command.add_argument("--signature-receipt", type=Path, required=True)
        command.add_argument("--output", type=Path, required=name != "validate-request")
        command.set_defaults(func=function)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
