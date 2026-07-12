"""Dependency-light integrity helpers for the full-corpus training bridge."""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


INDEX_HEADER = b"MMIDIDX\x00\x00"
INDEX_VERSION = 1
DTYPE_CODE_INT32 = 4
DTYPE_SIZE_INT32 = 4
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
HEX_COMMIT = re.compile(r"^[0-9a-f]{40}$")
READ_CHUNK = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(READ_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(payload)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return value


def write_json_atomic(
    path: Path, value: Mapping[str, Any], *, replace: bool = False
) -> None:
    if path.exists() and not replace:
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
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


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._")
    if not normalized:
        raise ValueError(f"cannot derive a safe name from {value!r}")
    return normalized


def resolve_relative(config_path: Path, value: str) -> Path:
    path = Path(value)
    return (
        path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()
    )


def relative_file_receipt(
    path: Path, root: Path, *, rows: int | None = None
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        receipt["rows"] = int(rows)
    return receipt


def validate_file_receipt(
    receipt: Mapping[str, Any], root: Path, *, hash_file: bool = True
) -> Path:
    relative = Path(str(receipt.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe receipt path: {relative}")
    path = (root / relative).resolve()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(receipt.get("bytes", -1)):
        raise ValueError(f"file-size drift: {path}")
    expected = str(receipt.get("sha256", ""))
    if not HEX_SHA256.fullmatch(expected):
        raise ValueError(f"invalid receipt SHA-256: {path}")
    if hash_file and sha256_file(path) != expected:
        raise ValueError(f"file checksum drift: {path}")
    return path


def tokenizer_tree_receipt(root: Path) -> dict[str, Any]:
    if root.is_symlink():
        raise ValueError(f"tokenizer root is a symlink: {root}")
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"tokenizer tree contains a symlink: {path}")
        if path.is_file():
            files.append(path)
    files.sort(key=lambda path: path.relative_to(root).as_posix())
    if not files:
        raise ValueError(f"tokenizer directory is empty: {root}")
    receipts = [relative_file_receipt(path, root) for path in files]
    return {
        "root": str(root.resolve()),
        "files": receipts,
        "tree_sha256": canonical_sha256(receipts),
    }


def validate_tokenizer_tree_receipt(receipt: Mapping[str, Any]) -> Path:
    """Re-hash the complete tokenizer tree using the historical receipt shape."""

    raw_root = Path(str(receipt.get("root", "")))
    expected_files = receipt.get("files")
    if not isinstance(expected_files, list) or not expected_files:
        raise ValueError("tokenizer tree receipt has no files")
    expected_paths = [str(row.get("path", "")) for row in expected_files]
    if len(set(expected_paths)) != len(expected_paths):
        raise ValueError("tokenizer tree receipt contains duplicate paths")
    actual = tokenizer_tree_receipt(raw_root)
    root = Path(actual["root"])
    if actual["files"] != expected_files:
        raise ValueError(f"tokenizer tree content drift: {root}")
    if receipt.get("tree_sha256") != actual["tree_sha256"]:
        raise ValueError(f"tokenizer tree digest drift: {root}")
    return root


def file_tree_receipt(
    root: Path,
    *,
    exclude_top_level: Sequence[str] = (),
) -> dict[str, Any]:
    """Hash a complete, symlink-free file tree in deterministic path order."""

    root = root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(root)
    excluded = set(exclude_top_level)
    files: list[Path] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in excluded:
            continue
        if path.is_symlink():
            raise ValueError(f"file tree contains a symlink: {path}")
        if path.is_file():
            files.append(path)
    files.sort(key=lambda path: path.relative_to(root).as_posix())
    if not files:
        raise ValueError(f"file tree is empty: {root}")
    receipts = [relative_file_receipt(path, root) for path in files]
    return {
        "root": str(root),
        "exclude_top_level": sorted(excluded),
        "files": receipts,
        "file_count": len(receipts),
        "total_bytes": sum(int(row["bytes"]) for row in receipts),
        "tree_sha256": canonical_sha256(
            {"exclude_top_level": sorted(excluded), "files": receipts}
        ),
    }


def validate_file_tree_receipt(receipt: Mapping[str, Any]) -> Path:
    """Re-hash a tree and reject missing, changed, extra, or linked files."""

    root = Path(str(receipt.get("root", ""))).resolve()
    expected_files = receipt.get("files")
    if not isinstance(expected_files, list) or not expected_files:
        raise ValueError("tree receipt has no files")
    expected_paths = {str(row.get("path", "")) for row in expected_files}
    if len(expected_paths) != len(expected_files):
        raise ValueError("tree receipt contains duplicate paths")
    actual_paths: set[str] = set()
    excluded = set(str(value) for value in receipt.get("exclude_top_level", []))
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in excluded:
            continue
        if path.is_symlink():
            raise ValueError(f"file tree contains a symlink: {path}")
        if path.is_file():
            actual_paths.add(relative.as_posix())
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)[:10]
        extra = sorted(actual_paths - expected_paths)[:10]
        raise ValueError(f"file tree inventory drift: missing={missing}, extra={extra}")
    for row in expected_files:
        validate_file_receipt(row, root)
    if int(receipt.get("file_count", -1)) != len(expected_files):
        raise ValueError("tree file-count receipt drift")
    if int(receipt.get("total_bytes", -1)) != sum(
        int(row["bytes"]) for row in expected_files
    ):
        raise ValueError("tree byte-count receipt drift")
    if receipt.get("tree_sha256") != canonical_sha256(
        {"exclude_top_level": sorted(excluded), "files": expected_files}
    ):
        raise ValueError("tree receipt digest drift")
    return root


def bound_code_sha(receipt: Mapping[str, Any], script: Path) -> str:
    """Verify that the executing script is the exact file frozen upstream."""

    script = script.resolve()
    rows = receipt.get("repository", {}).get("code_files", [])
    matches = [
        row for row in rows if Path(str(row.get("path", ""))).resolve() == script
    ]
    if len(matches) != 1:
        raise ValueError(
            f"input receipt does not uniquely bind executing code: {script}"
        )
    expected = str(matches[0].get("sha256", ""))
    if not HEX_SHA256.fullmatch(expected) or sha256_file(script) != expected:
        raise ValueError(f"executing code differs from frozen receipt: {script}")
    return expected


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def validate_frozen_repository(
    input_receipt: Mapping[str, Any], repo_root: Path
) -> dict[str, Any]:
    """Require the exact clean checkout recorded by the bridge input freeze."""

    frozen = input_receipt.get("repository")
    if not isinstance(frozen, Mapping):
        raise ValueError("bridge input receipt has no repository binding")
    expected_root = Path(str(frozen.get("root", ""))).resolve()
    expected_commit = str(frozen.get("commit", ""))
    if not HEX_COMMIT.fullmatch(expected_commit):
        raise ValueError("bridge input receipt has an invalid repository commit")
    requested_root = repo_root.resolve()
    try:
        actual_root = Path(_git(requested_root, "rev-parse", "--show-toplevel")).resolve()
    except subprocess.CalledProcessError as error:
        raise ValueError(f"training repository is not a Git checkout: {requested_root}") from error
    if requested_root != actual_root or actual_root != expected_root:
        raise ValueError(
            "training repository root differs from the bridge freeze: "
            f"requested={requested_root}, actual={actual_root}, frozen={expected_root}"
        )
    actual_commit = _git(actual_root, "rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise ValueError(
            f"training repository commit drift: {actual_commit} != {expected_commit}"
        )
    dirty = _git(actual_root, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise ValueError("training repository is dirty or has untracked files")
    return {"root": str(actual_root), "commit": actual_commit, "clean": True}


def absolute_file_receipt(path: Path) -> dict[str, Any]:
    selected = path
    path = selected.resolve()
    if not path.is_file() or selected.is_symlink():
        raise FileNotFoundError(f"launch dependency is absent or linked: {selected}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_launch_dependency_receipts(
    dependencies: Mapping[str, Path],
) -> dict[str, dict[str, Any]]:
    if not dependencies or any(not str(name) for name in dependencies):
        raise ValueError("launch dependencies must have non-empty names")
    return {
        str(name): absolute_file_receipt(path)
        for name, path in sorted(dependencies.items())
    }


def validate_launch_dependency_receipts(
    receipts: object,
    dependencies: Mapping[str, Path],
) -> dict[str, Path]:
    if not isinstance(receipts, Mapping):
        raise ValueError("training-assets receipt has no launch dependency map")
    expected_names = {str(name) for name in dependencies}
    if set(receipts) != expected_names:
        raise ValueError(
            "launch dependency identity drift: "
            f"missing={sorted(expected_names - set(receipts))}, "
            f"unexpected={sorted(set(receipts) - expected_names)}"
        )
    resolved: dict[str, Path] = {}
    for name, selected in dependencies.items():
        receipt = receipts[name]
        if not isinstance(receipt, Mapping):
            raise ValueError(f"invalid launch dependency receipt: {name}")
        selected_path = selected
        path = selected_path.resolve()
        if Path(str(receipt.get("path", ""))).resolve() != path:
            raise ValueError(f"launcher selected a different {name} path")
        if (
            not path.is_file()
            or selected_path.is_symlink()
            or path.stat().st_size != int(receipt.get("bytes", -1))
            or sha256_file(path) != receipt.get("sha256")
        ):
            raise ValueError(f"frozen launch dependency drift: {name}: {path}")
        resolved[name] = path
    return resolved


def document_key(
    source_name: str,
    input_relative: str,
    row_index: int,
    identity_values: Mapping[str, object] | object | None,
    *,
    identity_scope: str = "file",
) -> str:
    """Return a domain-separated composite training-document identity.

    ``file`` scope prevents shard-local upstream ids from colliding. ``global``
    scope is allowed only when the configured field tuple is globally unique;
    old Greek uses the global ``(source_dataset, source_doc_id)`` tuple.
    """

    if identity_scope not in {"file", "global"}:
        raise ValueError(f"unsupported document identity scope: {identity_scope}")
    if isinstance(identity_values, Mapping):
        components = [
            [str(key), str(value)]
            for key, value in identity_values.items()
            if value is not None and str(value)
        ]
    elif identity_values is not None and str(identity_values):
        components = [["id", str(identity_values)]]
    else:
        components = []
    payload: dict[str, Any] = {
        "contract": "full-cpt-document-identity-v2",
        "source_name": source_name,
        "identity_scope": identity_scope,
        "components": components,
    }
    if identity_scope == "file" or not components:
        payload["input_relative"] = input_relative
    if not components:
        payload["row_index"] = int(row_index)
    return "docv2:" + canonical_sha256(payload)


def heldout_hash(seed: int, set_name: str, source_name: str, doc_id: str) -> int:
    payload = (
        f"full-cpt-heldout-v1\0{seed}\0{set_name}\0{source_name}\0{doc_id}".encode(
            "utf-8"
        )
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def selected_by_threshold(
    *,
    seed: int,
    set_name: str,
    source_name: str,
    doc_id: str,
    numerator: int,
    denominator: int,
) -> bool:
    if numerator <= 0:
        return False
    if denominator <= 0 or numerator > denominator:
        raise ValueError("invalid deterministic selection threshold")
    return heldout_hash(
        seed, set_name, source_name, doc_id
    ) * denominator < numerator * (1 << 64)


def iter_index_lengths(path: Path) -> tuple[int, int, int]:
    """Return (sequences, document-index entries, exact tokens) from a Megatron v1 index."""

    with path.open("rb") as handle:
        if handle.read(len(INDEX_HEADER)) != INDEX_HEADER:
            raise ValueError(f"invalid Megatron index header: {path}")
        version = struct.unpack("<Q", handle.read(8))[0]
        dtype_code = struct.unpack("<B", handle.read(1))[0]
        sequences = struct.unpack("<Q", handle.read(8))[0]
        documents = struct.unpack("<Q", handle.read(8))[0]
        if version != INDEX_VERSION or dtype_code != DTYPE_CODE_INT32:
            raise ValueError(f"unsupported Megatron index format: {path}")
        lengths_raw = handle.read(sequences * 4)
        if len(lengths_raw) != sequences * 4:
            raise ValueError(f"truncated Megatron sequence lengths: {path}")
        lengths = struct.unpack(f"<{sequences}i", lengths_raw) if sequences else ()
        pointer_bytes = handle.read(sequences * 8)
        document_bytes = handle.read(documents * 8)
        if len(pointer_bytes) != sequences * 8 or len(document_bytes) != documents * 8:
            raise ValueError(f"truncated Megatron index arrays: {path}")
        if handle.read(1):
            raise ValueError(f"unexpected trailing Megatron index bytes: {path}")
    return int(sequences), int(documents), int(sum(lengths))


def write_index(path: Path, lengths: Iterable[int]) -> tuple[int, int, int]:
    import numpy as np

    values = np.asarray(list(lengths), dtype=np.int32)
    pointers = np.empty(len(values), dtype=np.int64)
    offset = 0
    for index, length in enumerate(values):
        pointers[index] = offset
        offset += int(length) * DTYPE_SIZE_INT32
    documents = np.arange(len(values) + 1, dtype=np.int64)
    with path.open("wb") as handle:
        handle.write(INDEX_HEADER)
        handle.write(struct.pack("<Q", INDEX_VERSION))
        handle.write(struct.pack("<B", DTYPE_CODE_INT32))
        handle.write(struct.pack("<Q", len(values)))
        handle.write(struct.pack("<Q", len(documents)))
        handle.write(values.tobytes(order="C"))
        handle.write(pointers.tobytes(order="C"))
        handle.write(documents.tobytes(order="C"))
    return len(values), len(documents), int(values.astype(np.int64).sum())


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            yield value


def load_exclusion_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {str(row["doc_id"]) for row in iter_jsonl(path)}


def task_output_prefix(stage_root: Path, task: Mapping[str, Any]) -> Path:
    return stage_root / "megatron" / str(task["output_prefix"])


def receipt_sha(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def require_exact_keys(
    value: Mapping[str, Any], required: set[str], *, label: str
) -> None:
    missing = required - set(value)
    if missing:
        raise ValueError(f"{label} misses required keys: {sorted(missing)}")
