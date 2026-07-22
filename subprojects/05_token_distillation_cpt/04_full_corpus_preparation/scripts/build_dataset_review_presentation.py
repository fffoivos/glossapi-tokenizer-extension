#!/usr/bin/env python3
"""Accept an immutable Agent-1 handoff and build the dataset-review site.

This is deliberately a narrow presentation boundary.  It receives compact
review artefacts only; it never opens raw corpus shards, samples documents, or
changes Agent-1 output.  The underlying static-site builder remains responsible
for site-file integrity and loopback serving.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from build_dataset_review_site import (
    DEFAULT_OUTPUT,
    build_site,
    safe_json,
    validate_site_directory,
)


HANDOFF_NAME = "dataset_review_site_handoff.json"
HANDOFF_SCHEMA = "dataset_review_presentation_handoff_v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
FORBIDDEN_SUFFIXES = {".parquet", ".arrow", ".safetensors", ".pt", ".bin", ".ckpt"}
FORBIDDEN_ROLES = {"raw_parquet", "canonical_shard", "checkpoint", "model", "corpus_export"}
ROLE_TO_ARGUMENT = {
    "inventory": "inventory",
    "evaluations": "evaluations",
    "sources_config": "sources_config",
    "quality_summary": "quality_summary",
    "quality_handoff_receipt": "quality_handoff_receipt",
    "review_requests": "review_requests",
    "review_responses": "review_responses",
    "admission": "admission",
    "novelty": "novelty",
    "complete_samples": "complete_samples",
    "complete_samples_receipt": "complete_samples_receipt",
    "complete_samples_attestation": "complete_samples_attestation",
    "public_preview": "public_previews",
    "pipeline_waterfall": "pipeline_waterfall",
}
REQUIRED_FINAL_ROLES = {
    "inventory",
    "evaluations",
    "sources_config",
    "quality_summary",
    "quality_handoff_receipt",
    "review_requests",
    "review_responses",
    "admission",
    "complete_samples",
    "complete_samples_receipt",
    "complete_samples_attestation",
}


@dataclass(frozen=True)
class AcceptedFile:
    role: str
    path: Path
    relative_path: str
    bytes: int
    sha256: str
    schema_version: str


@dataclass(frozen=True)
class AcceptedHandoff:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    run_id: str
    producer_commit: str
    created_at: str
    kind: str
    files: dict[str, AcceptedFile]

    def provenance(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "producer_commit": self.producer_commit,
            "handoff_sha256": self.manifest_sha256,
            "site_verification": "passed before publication",
        }


def _safe_member(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if (
        not relative_path
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != relative_path
    ):
        raise ValueError(f"unsafe handoff member path: {relative_path!r}")
    candidate = root / relative
    current = root
    for part in relative.parts:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError as exc:
            raise ValueError(f"missing handoff member: {relative_path}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"handoff member is a symlink: {relative_path}")
    if not candidate.is_file():
        raise ValueError(f"handoff member is not a regular file: {relative_path}")
    return candidate


def _read_regular(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(_read_regular(path))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _actual_files(root: Path) -> set[str]:
    result: set[str] = set()
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise ValueError(f"handoff contains a symlink: {candidate}")
        if candidate.is_file():
            result.add(candidate.relative_to(root).as_posix())
        elif not candidate.is_dir():
            raise ValueError(f"handoff contains a special filesystem entry: {candidate}")
    return result


def _validate_declared_schema(member: Path, content: bytes, declared: str) -> None:
    """Check the declared JSON/JSONL schema marker before role-specific parsing."""
    suffix = member.suffix.casefold()
    if suffix == ".json":
        try:
            value = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSON handoff member: {member}") from exc
        if not isinstance(value, dict) or value.get("schema_version") != declared:
            raise ValueError(f"handoff schema marker drift: {member}")
    elif suffix == ".jsonl":
        try:
            lines = content.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise ValueError(f"invalid UTF-8 JSONL handoff member: {member}") from exc
        if not lines:
            raise ValueError(f"empty JSONL handoff member: {member}")
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL handoff row: {member}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL handoff row: {member}:{line_number}") from exc
            if not isinstance(row, dict) or row.get("schema_version") != declared:
                raise ValueError(f"handoff schema marker drift: {member}:{line_number}")
    else:
        raise ValueError(f"handoff member must be JSON or JSONL: {member}")


def accept_handoff(root: Path, *, require_complete: bool) -> AcceptedHandoff:
    root = root.expanduser().absolute()
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"handoff root must be a non-symlink directory: {root}")
    manifest_path = _safe_member(root, HANDOFF_NAME)
    manifest_bytes = _read_regular(manifest_path)
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid handoff manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("handoff manifest must be an object")
    required = {
        "schema_version", "status", "run_id", "producer_commit", "created_at",
        "handoff_kind", "files",
    }
    if set(manifest) != required:
        raise ValueError("handoff manifest keys differ from the v1 contract")
    if (
        manifest["schema_version"] != HANDOFF_SCHEMA
        or manifest["status"] not in {"passed", "READY"}
        or not isinstance(manifest["run_id"], str)
        or not RUN_ID_RE.fullmatch(manifest["run_id"])
        or not isinstance(manifest["producer_commit"], str)
        or not COMMIT_RE.fullmatch(manifest["producer_commit"])
        or not isinstance(manifest["created_at"], str)
        or not isinstance(manifest["handoff_kind"], str)
        or manifest["handoff_kind"] not in {"fixture", "agent1"}
        or not isinstance(manifest["files"], list)
    ):
        raise ValueError("handoff manifest status, identity, or file list is invalid")
    try:
        datetime.fromisoformat(manifest["created_at"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("handoff created_at must be RFC 3339") from exc
    files: dict[str, AcceptedFile] = {}
    declared_paths: set[str] = set()
    for index, row in enumerate(manifest["files"]):
        if not isinstance(row, Mapping) or set(row) != {
            "role", "path", "bytes", "sha256", "schema_version"
        }:
            raise ValueError(f"handoff file entry {index} violates the v1 contract")
        role = row["role"]
        relative_path = row["path"]
        expected_bytes = row["bytes"]
        expected_sha256 = row["sha256"]
        schema_version = row["schema_version"]
        if (
            not isinstance(role, str)
            or role not in ROLE_TO_ARGUMENT
            or role in FORBIDDEN_ROLES
            or role in files
            or not isinstance(relative_path, str)
            or relative_path in declared_paths
            or Path(relative_path).suffix.casefold() in FORBIDDEN_SUFFIXES
            or not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes < 1
            or not isinstance(expected_sha256, str)
            or not SHA256_RE.fullmatch(expected_sha256)
            or not isinstance(schema_version, str)
            or not schema_version
        ):
            raise ValueError(f"invalid handoff file entry {index}")
        member = _safe_member(root, relative_path)
        content = _read_regular(member)
        if len(content) != expected_bytes or hashlib.sha256(content).hexdigest() != expected_sha256:
            raise ValueError(f"handoff receipt/hash drift: {relative_path}")
        _validate_declared_schema(member, content, schema_version)
        files[role] = AcceptedFile(
            role=role,
            path=member,
            relative_path=relative_path,
            bytes=expected_bytes,
            sha256=expected_sha256,
            schema_version=schema_version,
        )
        declared_paths.add(relative_path)
    actual = _actual_files(root) - {HANDOFF_NAME}
    if actual != declared_paths:
        raise ValueError(
            "handoff directory closure failed; "
            f"missing={sorted(declared_paths - actual)}, extra={sorted(actual - declared_paths)}"
        )
    if not {"inventory", "evaluations", "sources_config"}.issubset(files):
        raise ValueError("handoff lacks the frozen inventory/source-assessment closure")
    if require_complete:
        if manifest["handoff_kind"] != "agent1":
            raise ValueError("only an Agent-1 handoff can be published")
        missing = sorted(REQUIRED_FINAL_ROLES - set(files))
        if missing:
            raise ValueError(f"handoff is not complete for publication: {missing}")
    return AcceptedHandoff(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        run_id=manifest["run_id"],
        producer_commit=manifest["producer_commit"],
        created_at=manifest["created_at"],
        kind=manifest["handoff_kind"],
        files=files,
    )


def _acceptance_report(accepted: AcceptedHandoff) -> str:
    rows = [
        "# Dataset-review site acceptance report",
        "",
        f"- Run ID: `{accepted.run_id}`",
        f"- Producer commit: `{accepted.producer_commit}`",
        f"- Handoff SHA-256: `{accepted.manifest_sha256}`",
        f"- Handoff kind: `{accepted.kind}`",
        "- Validation: passed; every declared compact input was rehashed and directory closure was verified.",
        "",
        "## Accepted inputs",
        "",
        "| Role | File | Bytes | SHA-256 | Schema |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for item in sorted(accepted.files.values(), key=lambda value: value.role):
        rows.append(
            f"| {item.role} | `{item.relative_path}` | {item.bytes} | `{item.sha256}` | `{item.schema_version}` |"
        )
    rows.extend(
        [
            "",
            "## Evidence state",
            "",
            "Missing roles remain visible in the site as unavailable evidence. A fixture handoff is UI/test-only and cannot be published.",
            "",
            "## Verification checklist",
            "",
            "- Schema, byte, SHA-256, role, run-ID, and producer-commit closure passed.",
            "- No raw corpus artifact, undeclared file, symlink, or path escape was accepted.",
            "- Generated site manifest verifies every emitted file before serving.",
            "- Browser checklist: overview, filters, every source page, document navigation, keyboard N/Shift+N, public/masked variant toggle, desktop/mobile layout, and no console errors.",
            "- Screenshot checklist: desktop overview, a populated source page, document browser, and a 390 px mobile source page.",
            "",
        ]
    )
    return "\n".join(rows)


def _builder_args(accepted: AcceptedHandoff, output: Path) -> SimpleNamespace:
    values: dict[str, Any] = {name: None for name in set(ROLE_TO_ARGUMENT.values())}
    for role, item in accepted.files.items():
        values[ROLE_TO_ARGUMENT[role]] = item.path
    values.update(
        {
            "presentation_handoff": accepted.manifest_path,
            "output_dir": output,
            "replace": False,
            "site_key": hashlib.sha256(
                b"dataset-review-site-v3\0" + bytes.fromhex(accepted.manifest_sha256)
            ).digest(),
            "generated_at": accepted.created_at,
            "manifest_output_root": ".",
            "presentation_provenance": accepted.provenance(),
            "acceptance_report": _acceptance_report(accepted),
        }
    )
    return SimpleNamespace(**values)


def build(args: argparse.Namespace) -> int:
    accepted = accept_handoff(args.handoff_dir, require_complete=False)
    output = args.output_dir.expanduser().absolute()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite a staging site: {output}")
    return int(build_site(_builder_args(accepted, output)))


def publish(args: argparse.Namespace) -> int:
    accepted = accept_handoff(args.handoff_dir, require_complete=True)
    staging = args.staging_dir.expanduser().absolute()
    output = args.output_dir.expanduser().absolute()
    manifest = validate_site_directory(staging)
    inputs = manifest.get("inputs", {})
    handoff = inputs.get("presentation_handoff") if isinstance(inputs, dict) else None
    if not isinstance(handoff, dict) or handoff.get("sha256") != accepted.manifest_sha256:
        raise ValueError("staging site was not built from the accepted handoff")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite final site: {output}")
    if staging.parent != output.parent:
        raise ValueError("staging and final directories must share a parent for atomic publication")
    # accept_handoff above rehashes every input immediately before this rename.
    os.replace(staging, output)
    validate_site_directory(output)
    print(safe_json({"ok": True, "published": str(output), "run_id": accepted.run_id}))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build_command = commands.add_parser("build")
    build_command.add_argument("--handoff-dir", type=Path, required=True)
    build_command.add_argument("--output-dir", type=Path, required=True)
    build_command.set_defaults(function=build)
    publish_command = commands.add_parser("publish")
    publish_command.add_argument("--handoff-dir", type=Path, required=True)
    publish_command.add_argument("--staging-dir", type=Path, required=True)
    publish_command.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    publish_command.set_defaults(function=publish)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
