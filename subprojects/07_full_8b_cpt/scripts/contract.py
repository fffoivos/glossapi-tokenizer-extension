"""Receipt helpers for the full Apertus-8B CPT campaign."""

from __future__ import annotations

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
