#!/usr/bin/env python3
"""Resolve the tracked HF source registry to an immutable selected-file lock."""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import hashlib
import json
import os
import tempfile
from pathlib import Path


HEX_SHA256 = frozenset("0123456789abcdef")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_atomic(path: Path, value: dict) -> None:
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
        os.link(temporary, path)
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def lfs_value(value: object, name: str) -> object | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def exact_lfs_sha256(value: object | None, *, repo_id: str, path: str) -> str | None:
    """Require an exact LFS content identifier when the API declares LFS metadata."""

    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 64 or set(value) - HEX_SHA256:
        raise ValueError(
            f"{repo_id}:{path}: Hugging Face did not expose an exact LFS SHA-256; "
            "resolve with authenticated metadata (HF_TOKEN)"
        )
    return value


def matches(path: str, includes: list[str], excludes: list[str]) -> bool:
    included = not includes or any(
        fnmatch.fnmatchcase(path, pattern) for pattern in includes
    )
    excluded = any(fnmatch.fnmatchcase(path, pattern) for pattern in excludes)
    return included and not excluded


def entries(config: dict) -> list[dict]:
    base = {"source_id": "nanochat_base", **config["base"]}
    overlap = {
        "source_id": "apertus_overlap_overlay",
        **config["apertus_overlap_overlay"],
    }
    tokenizer = {
        "source_id": "modern_greek_148k_tokenizer",
        **config["tokenizer"],
    }
    # External registries (currently Mozilla Data Collective) have their own
    # authenticated, checksum-bound acquisition path.  Never ask the HF
    # resolver to interpret their metadata-only pointer repositories as empty
    # corpus artifacts.
    hf_sources = [
        source
        for source in config["sources"]
        if source.get("acquisition_kind", "huggingface") == "huggingface"
    ]
    return [base, overlap, tokenizer, *hf_sources]


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources", type=Path, default=here / "configs" / "sources.json"
    )
    parser.add_argument(
        "--output", type=Path, default=here / "configs" / "sources.lock.json"
    )
    parser.add_argument(
        "--source", action="append", help="resolve only selected source_id values"
    )
    parser.add_argument(
        "--anonymous",
        action="store_true",
        help="resolve public metadata without env or cached Hugging Face credentials",
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite immutable source lock: {args.output}"
        )

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.hf_api import RepoFile
    except ImportError as exc:  # pragma: no cover - exercised on Clariden
        raise RuntimeError("install huggingface_hub in the Phase-04 runtime") from exc

    config = json.loads(args.sources.read_text(encoding="utf-8"))
    selected_ids = set(args.source or [])
    registry = [
        entry
        for entry in entries(config)
        if not selected_ids or entry["source_id"] in selected_ids
    ]
    if selected_ids - {entry["source_id"] for entry in registry}:
        raise ValueError(
            f"unknown source ids: {sorted(selected_ids - {entry['source_id'] for entry in registry})}"
        )

    api = HfApi(token=False if args.anonymous else os.environ.get("HF_TOKEN"))
    locked: list[dict] = []
    for entry in registry:
        repo_id = entry["repo_id"]
        repo_type = entry.get("repo_type", "dataset")
        revision = entry["revision"]
        info = api.repo_info(
            repo_id=repo_id, repo_type=repo_type, revision=revision, files_metadata=True
        )
        if info.sha != revision:
            raise ValueError(
                f"{repo_id}: requested {revision}, API resolved {info.sha}"
            )
        includes = list(entry.get("include_globs", []))
        excludes = list(entry.get("exclude_globs", []))
        files = []
        for item in api.list_repo_tree(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            recursive=True,
            expand=True,
        ):
            if not isinstance(item, RepoFile) or not matches(
                item.path, includes, excludes
            ):
                continue
            lfs = getattr(item, "lfs", None)
            lfs_sha256 = exact_lfs_sha256(
                lfs_value(lfs, "sha256"), repo_id=repo_id, path=item.path
            )
            files.append(
                {
                    "path": item.path,
                    "size": int(item.size or 0),
                    "blob_id": getattr(item, "blob_id", None),
                    "lfs_sha256": lfs_sha256,
                    "lfs_size": lfs_value(lfs, "size"),
                }
            )
        files.sort(key=lambda row: row["path"])
        if not files:
            raise ValueError(
                f"{entry['source_id']} ({repo_id}@{revision}): selected no files; includes={includes}, excludes={excludes}"
            )
        locked.append(
            {
                "source_id": entry["source_id"],
                "repo_id": repo_id,
                "repo_type": repo_type,
                "revision": revision,
                "role": entry.get("role", "artifact"),
                "selected_files": files,
                "selected_file_count": len(files),
                "selected_bytes": sum(row["size"] for row in files),
            }
        )
        print(
            f"resolved {entry['source_id']}: {len(files)} files, "
            f"{sum(row['size'] for row in files):,} bytes",
            flush=True,
        )

    lock = {
        "schema_version": "full_cpt_sources_lock_v1",
        "resolved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sources_config": str(args.sources.resolve()),
        "sources_config_sha256": sha256_file(args.sources),
        "sources": locked,
    }
    write_json_atomic(args.output, lock)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
