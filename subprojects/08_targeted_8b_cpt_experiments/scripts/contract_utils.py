from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


GLOBAL_BATCH_SEQUENCES = 1024
SEQUENCE_LENGTH = 4096
GLOBAL_BATCH_TOKENS = GLOBAL_BATCH_SEQUENCES * SEQUENCE_LENGTH
TOKENIZER_SHA256 = "bbb08e71929b519c5c2362338b0fc6a0e99955cb8fdbf0729ae1311117e6561b"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def require_file_binding(
    binding: Mapping[str, Any],
    *,
    expected_path: Path | None = None,
    verify_sha256: bool = True,
) -> Path:
    """Verify a receipt-style absolute file binding and return its path."""
    require(isinstance(binding, Mapping), "file binding is not an object")
    path = Path(str(binding.get("path", ""))).resolve()
    if expected_path is not None:
        require(path == expected_path.resolve(), f"file binding path drift: {path} != {expected_path.resolve()}")
    require(path.is_file() and not path.is_symlink(), f"bound file missing or symlinked: {path}")
    require(path.stat().st_size == int(binding.get("bytes", -1)), f"bound file size drift: {path}")
    if verify_sha256:
        require(sha256_file(path) == binding.get("sha256"), f"bound file SHA-256 drift: {path}")
    return path


def require_receipt(
    path: Path,
    *,
    schemas: Iterable[str],
    statuses: Iterable[str] = ("passed", "frozen", "completed", "accepted", "promoted"),
) -> dict[str, Any]:
    value = read_json(path)
    require(value.get("schema_version") in set(schemas), f"receipt schema drift: {path}")
    require(str(value.get("status", "")).lower() in set(statuses), f"receipt status is not passing: {path}")
    return value


def require_relative_inventory(
    *,
    root: Path,
    rows: object,
    relative_key: str = "path",
    require_exact_file_set: bool = True,
) -> list[dict[str, Any]]:
    """Verify a full relative-path file inventory under an artifact root."""
    resolved = root.resolve()
    require(resolved.is_dir(), f"artifact root missing: {resolved}")
    require(isinstance(rows, list) and bool(rows), f"artifact inventory missing: {resolved}")
    seen: set[str] = set()
    verified: list[dict[str, Any]] = []
    for row in rows:
        require(isinstance(row, dict), f"artifact inventory row malformed: {resolved}")
        relative = str(row.get(relative_key, ""))
        candidate = (resolved / relative).resolve()
        require(
            relative
            and relative not in seen
            and (candidate.parent == resolved or resolved in candidate.parents),
            f"unsafe or duplicate artifact path: {relative}",
        )
        require_file_binding(
            {"path": str(candidate), "bytes": row.get("bytes"), "sha256": row.get("sha256")},
            expected_path=candidate,
        )
        seen.add(relative)
        verified.append({"path": relative, "bytes": candidate.stat().st_size, "sha256": str(row.get("sha256"))})
    if require_exact_file_set:
        actual = {str(path.relative_to(resolved)) for path in resolved.rglob("*") if path.is_file()}
        require(actual == seen, f"artifact inventory file-set drift: {resolved}")
    return verified


def executing_code_bundle() -> dict[str, Any]:
    """Bind a receipt to the immutable code tree its Slurm wrapper verified."""
    root_value = os.environ.get("H2G_CODE_ROOT")
    receipt_value = os.environ.get("H2G_CODE_RECEIPT")
    require(bool(root_value) and bool(receipt_value), "H2G code-bundle environment is missing")
    root = Path(str(root_value)).resolve()
    receipt_path = Path(str(receipt_value)).resolve()
    require(root.is_dir(), f"executing code root missing: {root}")
    require(receipt_path.is_file(), f"executing code receipt missing: {receipt_path}")
    receipt = read_json(receipt_path)
    require(receipt.get("schema_version") == "apertus_mini_immutable_code_bundle_v1", "code receipt schema drift")
    require(receipt.get("status") == "frozen" and receipt.get("kind") == "scientific", "code receipt is not frozen scientific code")
    require(Path(str(receipt.get("root", ""))).resolve() == root, "code receipt root drift")
    tree_sha256 = str(receipt.get("tree_sha256", ""))
    require(len(tree_sha256) == 64, "code receipt tree SHA-256 missing")
    return {
        "root": str(root),
        "tree_sha256": tree_sha256,
        "receipt": file_binding(receipt_path),
    }


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable JSON output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def copy_file_atomic(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(f"immutable copied output exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output_handle:
            for chunk in iter(lambda: input_handle.read(8 * 1024 * 1024), b""):
                output_handle.write(chunk)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.link(temporary, destination)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def nearest_replay_targets(modern_tokens: int) -> tuple[int, int]:
    return round(modern_tokens * 20 / 79), round(modern_tokens / 79)


def geometry(modern_tokens: int, foreign_tokens: int, old_tokens: int) -> dict[str, int]:
    total = modern_tokens + foreign_tokens + old_tokens
    updates = (total + GLOBAL_BATCH_TOKENS - 1) // GLOBAL_BATCH_TOKENS
    slots = updates * GLOBAL_BATCH_TOKENS
    return {
        "modern_active_tokens": modern_tokens,
        "foreign_active_tokens": foreign_tokens,
        "old_greek_active_tokens": old_tokens,
        "total_active_tokens": total,
        "global_batch_token_slots": GLOBAL_BATCH_TOKENS,
        "updates": updates,
        "training_slot_tokens": slots,
        "loss_inactive_tail_slots": slots - total,
    }


def token_milestones(
    updates: int,
    *,
    cadence_tokens: int,
    warmup_updates: int | None = None,
    cooldown_start: int | None = None,
    boundaries: list[int] | None = None,
) -> list[int]:
    result = {0, updates}
    if warmup_updates is not None and 0 < warmup_updates < updates:
        result.add(warmup_updates)
    if cooldown_start is not None and 0 < cooldown_start < updates:
        result.add(cooldown_start)
    result.update(value for value in boundaries or [] if 0 <= value <= updates)
    token_mark = cadence_tokens
    total = updates * GLOBAL_BATCH_TOKENS
    while token_mark < total:
        result.add(round(token_mark / GLOBAL_BATCH_TOKENS))
        token_mark += cadence_tokens
    return sorted(result)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
