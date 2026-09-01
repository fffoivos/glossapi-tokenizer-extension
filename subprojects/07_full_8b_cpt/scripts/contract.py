"""Receipt helpers for the full Apertus-8B CPT campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


PASS_STATUSES = {"accepted", "completed", "frozen", "passed", "promoted"}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def verify_code_bundle_receipt(path: Path, root: Path, kind: str = "scientific") -> dict[str, Any]:
    receipt = require_passing(path, "apertus_mini_immutable_code_bundle_v1")
    resolved_root = root.resolve()
    rows = receipt.get("files", [])
    if (
        receipt.get("kind") != kind
        or Path(receipt.get("root", "")).resolve() != resolved_root
        or int(receipt.get("file_count", -1)) != len(rows)
        or not rows
    ):
        raise ValueError("code bundle receipt/root drift")
    seen = set()
    for row in rows:
        relative = str(row.get("relative_path", ""))
        unresolved = resolved_root / relative
        candidate = unresolved.resolve()
        inside = candidate.parent == resolved_root or resolved_root in candidate.parents
        if (
            not relative
            or relative in seen
            or unresolved.is_symlink()
            or not inside
            or not candidate.is_file()
            or candidate.stat().st_size != int(row.get("bytes", -1))
            or sha256_file(candidate) != row.get("sha256")
        ):
            raise ValueError(f"code bundle file drift: {relative}")
        seen.add(relative)
    exclusions = receipt.get("exclusions", {})
    ignored_parts = set(exclusions.get("directory_parts", []))
    ignored_suffixes = tuple(exclusions.get("file_suffixes", []))
    observed = set()
    for candidate in resolved_root.rglob("*"):
        relative = candidate.relative_to(resolved_root)
        if any(part in ignored_parts for part in relative.parts):
            continue
        if candidate.is_symlink():
            raise ValueError(f"code bundle contains a symlink: {relative.as_posix()}")
        if candidate.is_file() and not candidate.name.endswith(ignored_suffixes):
            observed.add(relative.as_posix())
    if observed != seen:
        raise ValueError(
            f"code bundle inventory drift: missing={sorted(seen - observed)}, "
            f"extra={sorted(observed - seen)}"
        )
    canonical = json.dumps(rows, separators=(",", ":"), sort_keys=True).encode()
    if hashlib.sha256(canonical).hexdigest() != receipt.get("tree_sha256"):
        raise ValueError("code bundle tree hash drift")
    return receipt


def atomic_write_json(path: Path, value: Any, *, exclusive: bool = True) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise FileExistsError(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def require_passing(path: Path, schema: str) -> dict[str, Any]:
    value = read_json(path)
    if value.get("schema_version") != schema:
        raise ValueError(f"{path}: schema drift")
    if str(value.get("status", "")).lower() not in PASS_STATUSES:
        raise ValueError(f"{path}: non-passing status")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser(
        "verify-code-bundle", description="Verify a frozen code bundle receipt."
    )
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--receipt", type=Path, required=True)
    verify.add_argument("--kind", choices=("scientific", "efficiency"), required=True)
    args = parser.parse_args()
    if args.command == "verify-code-bundle":
        value = verify_code_bundle_receipt(args.receipt, args.root, args.kind)
        print(json.dumps({"ok": True, "kind": args.kind, "tree_sha256": value["tree_sha256"]}, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())


def scientific_digest(recipe: dict[str, Any]) -> str:
    """Hash only experiment semantics, excluding parallelism and segmentation."""

    payload = {
        "recipe_id": recipe["recipe_id"],
        "data": recipe["data"],
        "tokenizer": recipe["tokenizer"],
        "initialization": recipe["initialization"],
        "model": recipe["model"],
        "optimization": recipe["optimization"],
        "batch": {
            key: recipe["batch_and_parallelism"][key]
            for key in (
                "global_batch_sequences",
                "global_batch_tokens",
                "micro_batch_sequences",
                "training_updates",
                "training_samples",
                "tensor_parallel",
                "pipeline_parallel",
                "context_parallel",
            )
        },
        "evaluation": recipe["evaluation"],
        "software": recipe["software"],
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()
