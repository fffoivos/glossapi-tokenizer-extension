#!/usr/bin/env python3
"""Download and freeze the minimal vetted FineWeb2-HQ replay supplement."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def acquire(spec: dict[str, Any], root: Path, repo_id: str, revision: str) -> dict[str, Any]:
    target = root / "raw" / spec["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = int(spec["bytes"])
    expected_sha = str(spec["sha256"])
    if target.exists():
        if not target.is_file() or target.stat().st_size != expected_bytes or sha(target) != expected_sha:
            raise ValueError(f"existing supplement drift: {target}")
    else:
        url = (
            f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/"
            f"{spec['path']}?download=true"
        )
        temporary = target.with_name(f".{target.name}.{os.getpid()}.partial")
        request = urllib.request.Request(url, headers={"User-Agent": "glossapi-lr13-replay-freezer/1"})
        digest = hashlib.sha256()
        size = 0
        try:
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("xb") as handle:
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            if size != expected_bytes or digest.hexdigest() != expected_sha:
                raise ValueError(f"downloaded supplement hash/size mismatch: {spec['path']}")
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
    metadata = pq.ParquetFile(target).metadata
    return {
        **spec,
        "local_path": str(target.resolve()),
        "rows": int(metadata.num_rows),
        "verified_bytes": target.stat().st_size,
        "verified_sha256": sha(target),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    recipe_path = args.recipe.resolve()
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    config = recipe["replay_supplements"]
    root = args.output_root.resolve()
    if root != Path(config["root"]).resolve():
        raise ValueError("supplement output root differs from frozen recipe")
    specs = list(config["files"])
    if len(specs) != int(config["expected_files"]):
        raise ValueError("supplement file-count drift")
    with ThreadPoolExecutor(max_workers=len(specs)) as pool:
        files = list(
            pool.map(
                lambda spec: acquire(spec, root, config["repo_id"], config["revision"]),
                specs,
            )
        )
    receipt = {
        "schema_version": "lr13_replay_supplement_download_v1",
        "status": "completed",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "recipe": {"path": str(recipe_path), "sha256": sha(recipe_path)},
        "repo_id": config["repo_id"],
        "revision": config["revision"],
        "files": files,
    }
    output = root / "download_receipt.json"
    write(output, receipt)
    print(json.dumps({"ok": True, "output": str(output), "files": len(files)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
