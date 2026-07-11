#!/usr/bin/env python3
"""Small stdlib-only receipt contract for the Phase-04 Clariden DAG."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


RUN_SCHEMA = "full_cpt_pipeline_run_v1"
STAGE_SCHEMA = "full_cpt_pipeline_stage_receipt_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def write_json_atomic(path: Path, value: dict[str, Any], *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(path, flags, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def cmd_init_run(args: argparse.Namespace) -> None:
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    expected = {
        "schema_version": RUN_SCHEMA,
        "run_id": args.run_id,
        "run_root": str(root),
        "code_commit": args.code_commit,
        "sources": str(args.sources.resolve()),
        "sources_sha256": sha256(args.sources),
        "cleaning_policy": str(args.cleaning_policy.resolve()),
        "cleaning_policy_sha256": sha256(args.cleaning_policy),
        "tokenizer_sha256": args.tokenizer_sha256,
    }
    manifest = root / "run_manifest.json"
    if manifest.exists():
        current = read_json(manifest)
        for key, value in expected.items():
            if current.get(key) != value:
                raise ValueError(f"immutable run identity drift for {key}: {current.get(key)!r} != {value!r}")
        return
    value = {**expected, "created_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        write_json_atomic(manifest, value, exclusive=True)
    except FileExistsError:
        cmd_init_run(args)


def stage_identity(args: argparse.Namespace) -> dict[str, str]:
    return {
        "stage": args.stage,
        "run_id": args.run_id,
        "code_commit": args.code_commit,
    }


def cmd_begin_stage(args: argparse.Namespace) -> None:
    args.stage_dir.mkdir(parents=True, exist_ok=True)
    path = args.stage_dir / "stage_attempt.json"
    attempt = {
        "schema_version": "full_cpt_pipeline_stage_attempt_v1",
        **stage_identity(args),
        "job_id": args.job_id,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "resume": path.exists(),
    }
    write_json_atomic(path, attempt)


def cmd_add_input(args: argparse.Namespace) -> None:
    inputs_path = args.stage_dir / "stage_inputs.json"
    current = read_json(inputs_path) if inputs_path.exists() else {
        "schema_version": "full_cpt_pipeline_stage_inputs_v1",
        "inputs": {},
    }
    path = args.path.resolve()
    value = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
    previous = current["inputs"].get(args.name)
    if previous is not None and previous != value:
        raise ValueError(f"resume input drift for {args.name}: {previous!r} != {value!r}")
    current["inputs"][args.name] = value
    write_json_atomic(inputs_path, current)


def output_record(stage_dir: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(stage_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"declared output is outside its immutable stage directory: {resolved}") from exc
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise ValueError(f"required non-empty stage output missing: {resolved}")
    return {"path": relative.as_posix(), "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def cmd_finish_stage(args: argparse.Namespace) -> None:
    receipt_path = args.stage_dir / "stage_receipt.json"
    if receipt_path.exists() or (args.stage_dir / "COMPLETED").exists():
        raise FileExistsError(f"refusing to overwrite completed stage: {args.stage_dir}")
    inputs_path = args.stage_dir / "stage_inputs.json"
    inputs = read_json(inputs_path) if inputs_path.exists() else {"inputs": {}}
    outputs = [output_record(args.stage_dir, path) for path in args.required_output]
    receipt = {
        "schema_version": STAGE_SCHEMA,
        "status": "passed",
        **stage_identity(args),
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inputs": inputs.get("inputs", {}),
        "outputs": outputs,
    }
    write_json_atomic(receipt_path, receipt, exclusive=True)
    completed = args.stage_dir / "COMPLETED"
    descriptor = os.open(completed, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(sha256(receipt_path) + "  stage_receipt.json\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_receipt_outputs(args: argparse.Namespace) -> Path:
    receipt_path = args.stage_dir / "stage_receipt.json"
    if not receipt_path.is_file():
        raise ValueError(f"stage receipt is missing: {args.stage_dir}")
    receipt = read_json(receipt_path)
    expected = {"schema_version": STAGE_SCHEMA, "status": "passed", **stage_identity(args)}
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"{receipt_path}: {key} mismatch")
    for item in receipt.get("outputs", []):
        path = args.stage_dir / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"completed output drift: {path}")
    return receipt_path


def cmd_validate_stage(args: argparse.Namespace) -> None:
    receipt_path = validate_receipt_outputs(args)
    completed = args.stage_dir / "COMPLETED"
    if not completed.is_file():
        raise ValueError(f"stage completion marker is missing: {args.stage_dir}")
    marker = completed.read_text(encoding="utf-8").split()[0]
    if marker != sha256(receipt_path):
        raise ValueError(f"{completed}: receipt hash mismatch")


def cmd_repair_stage_marker(args: argparse.Namespace) -> None:
    receipt_path = validate_receipt_outputs(args)
    completed = args.stage_dir / "COMPLETED"
    if completed.exists():
        raise FileExistsError(f"completion marker already exists: {completed}")
    descriptor = os.open(completed, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(sha256(receipt_path) + "  stage_receipt.json\n")
        handle.flush()
        os.fsync(handle.fileno())


def cmd_validate_admission(args: argparse.Namespace) -> None:
    value = read_json(args.path)
    actual_sha256 = sha256(args.path)
    if actual_sha256 != args.expected_sha256:
        raise ValueError(
            f"manual admission confirmation hash mismatch: {actual_sha256} != {args.expected_sha256}"
        )
    schema = value.get("schema_version")
    rows = value.get("sources")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source admission has no source decisions")
    if schema == "source_quality_review_admission_v1":
        if value.get("pending_adjudications") != 0:
            raise ValueError("source reviews still require adjudication")
        key = "source_dataset"
    elif schema == "full_cpt_source_admission_v1":
        if value.get("status") != "approved":
            raise ValueError("legacy source admission is not explicitly approved")
        key = "source_id"
    else:
        raise ValueError("unsupported source admission schema")
    seen: set[str] = set()
    for row in rows:
        identity = row.get(key) if isinstance(row, dict) else None
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError(f"source admission has invalid/duplicate {key}")
        seen.add(identity)
        if row.get("decision") not in {"include", "include_after_cleaning", "quarantine", "exclude"}:
            raise ValueError(f"invalid source decision for {identity}")


def cmd_validate_cleaning_replay(args: argparse.Namespace) -> None:
    reference = read_json(args.reference_receipt).get("inputs", {})
    current = read_json(args.current_inputs).get("inputs", {})

    def relevant(values: dict[str, Any]) -> dict[str, Any]:
        fixed = {"tokenizer", "eligibility_policy"}
        prefixes = ("document_actions:", "structural_")
        return {
            key: value
            for key, value in values.items()
            if key in fixed or key.startswith(prefixes)
        }

    reference_relevant = relevant(reference)
    current_relevant = relevant(current)
    if reference_relevant != current_relevant:
        raise ValueError(
            "final cleaning inputs differ from the reviewed cleaning pass: "
            f"reference={sorted(reference_relevant)} current={sorted(current_relevant)}"
        )


def cmd_get_input_path(args: argparse.Namespace) -> None:
    receipt = read_json(args.receipt)
    value = receipt.get("inputs", {}).get(args.name)
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise ValueError(f"{args.receipt}: missing recorded input {args.name!r}")
    path = Path(value["path"])
    if not path.is_file() or sha256(path) != value.get("sha256"):
        raise ValueError(f"recorded input drift: {path}")
    print(path)


def cmd_get_json_field(args: argparse.Namespace) -> None:
    value: Any = read_json(args.path)
    for component in args.field.split("."):
        if not isinstance(value, dict) or component not in value:
            raise ValueError(f"{args.path}: missing JSON field {args.field!r}")
        value = value[component]
    if not isinstance(value, (str, int, float, bool)):
        raise ValueError(f"{args.path}: JSON field {args.field!r} is not scalar")
    print(str(value))


def inventory_entries(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= set(value):
            found.append(value)
        for nested in value.values():
            found.extend(inventory_entries(nested))
    elif isinstance(value, list):
        for nested in value:
            found.extend(inventory_entries(nested))
    return found


def manifest_inventory_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    schema = manifest.get("schema_version")
    if schema == "full_cpt_greekmmlu_decontamination_v1":
        result = []
        for row in manifest.get("files", []):
            for key in ("output", "dropped", "ledger"):
                receipt = dict(row[key])
                path = Path(str(receipt["path"]))
                if not path.is_absolute():
                    path = Path(str(manifest[key])) / path
                receipt["path"] = str(path)
                result.append(receipt)
        return result
    if schema == "full_cpt_release_manifest_v1":
        result = []
        root = Path(str(manifest["output"]))
        for row in manifest.get("files", []):
            for key in ("training", "redistribution"):
                receipt = dict(row[key])
                path = Path(str(receipt["path"]))
                receipt["path"] = str(path if path.is_absolute() else root / path)
                result.append(receipt)
        return result
    if schema == "full_cpt_dedup_wrapper_manifest_v1":
        result = []
        staged_root = Path(str(manifest["staged_input"]))
        for row in manifest.get("files", []):
            receipt = dict(row["staged"])
            path = Path(str(receipt["path"]))
            receipt["path"] = str(path if path.is_absolute() else staged_root / path)
            result.append(receipt)
        output = manifest.get("dedup_output")
        if isinstance(output, dict) and isinstance(output.get("decisions"), dict):
            result.append(dict(output["decisions"]))
        return result
    return inventory_entries(manifest)


def cmd_validate_inventory(args: argparse.Namespace) -> None:
    manifest = read_json(args.manifest)
    entries = manifest_inventory_entries(manifest)
    if not entries:
        raise ValueError(f"{args.manifest}: no path/bytes/sha256 inventory entries")
    validated: list[dict[str, Any]] = []
    seen: dict[str, tuple[int, str]] = {}
    for entry in entries:
        path = Path(str(entry["path"])).resolve()
        expected = (int(entry["bytes"]), str(entry["sha256"]))
        previous = seen.get(str(path))
        if previous is not None:
            if previous != expected:
                raise ValueError(f"conflicting inventory entries for {path}")
            continue
        if not path.is_file() or path.stat().st_size != expected[0] or sha256(path) != expected[1]:
            raise ValueError(f"inventory verification failed: {path}")
        seen[str(path)] = expected
        validated.append({"path": str(path), "bytes": expected[0], "sha256": expected[1]})
    validated.sort(key=lambda row: row["path"])
    aggregate = hashlib.sha256(
        json.dumps(validated, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    result = {
        "schema_version": "full_cpt_shard_inventory_validation_v1",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "files": len(validated),
        "bytes": sum(row["bytes"] for row in validated),
        "inventory_sha256": aggregate,
        "validated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if args.output.exists():
        current = read_json(args.output)
        stable_keys = {key: value for key, value in result.items() if key != "validated_at"}
        current_stable = {key: value for key, value in current.items() if key != "validated_at"}
        if current_stable != stable_keys:
            raise ValueError(f"existing inventory validation drift: {args.output}")
        return
    write_json_atomic(args.output, result, exclusive=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init-run")
    init.add_argument("--run-root", type=Path, required=True)
    init.add_argument("--run-id", required=True)
    init.add_argument("--code-commit", required=True)
    init.add_argument("--sources", type=Path, required=True)
    init.add_argument("--cleaning-policy", type=Path, required=True)
    init.add_argument("--tokenizer-sha256", required=True)
    init.set_defaults(func=cmd_init_run)

    for name, function in (
        ("begin-stage", cmd_begin_stage),
        ("finish-stage", cmd_finish_stage),
        ("validate-stage", cmd_validate_stage),
        ("repair-stage-marker", cmd_repair_stage_marker),
    ):
        command = commands.add_parser(name)
        command.add_argument("--stage-dir", type=Path, required=True)
        command.add_argument("--stage", required=True)
        command.add_argument("--run-id", required=True)
        command.add_argument("--code-commit", required=True)
        if name == "begin-stage":
            command.add_argument("--job-id", required=True)
        elif name == "finish-stage":
            command.add_argument("--required-output", action="append", type=Path, default=[])
        command.set_defaults(func=function)

    add_input = commands.add_parser("add-input")
    add_input.add_argument("--stage-dir", type=Path, required=True)
    add_input.add_argument("--name", required=True)
    add_input.add_argument("--path", type=Path, required=True)
    add_input.set_defaults(func=cmd_add_input)

    admission = commands.add_parser("validate-admission")
    admission.add_argument("--path", type=Path, required=True)
    admission.add_argument("--expected-sha256", required=True)
    admission.set_defaults(func=cmd_validate_admission)

    replay = commands.add_parser("validate-cleaning-replay")
    replay.add_argument("--reference-receipt", type=Path, required=True)
    replay.add_argument("--current-inputs", type=Path, required=True)
    replay.set_defaults(func=cmd_validate_cleaning_replay)

    get_input = commands.add_parser("get-input-path")
    get_input.add_argument("--receipt", type=Path, required=True)
    get_input.add_argument("--name", required=True)
    get_input.set_defaults(func=cmd_get_input_path)

    get_field = commands.add_parser("get-json-field")
    get_field.add_argument("--path", type=Path, required=True)
    get_field.add_argument("--field", required=True)
    get_field.set_defaults(func=cmd_get_json_field)

    inventory = commands.add_parser("validate-inventory")
    inventory.add_argument("--manifest", type=Path, required=True)
    inventory.add_argument("--output", type=Path, required=True)
    inventory.set_defaults(func=cmd_validate_inventory)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
