#!/usr/bin/env python3
"""Search CSCS-owned storage for the already-existing poly_train artifact.

This is deliberately discovery-only.  It never materializes, reconstructs, or
rewrites a corpus.  Exact candidates are hashed so a later source audit can
bind one immutably before Experiment A proceeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


NAME_PATTERNS = (
    "poly_train.parquet",
    "*poly*train*.parquet",
    "*polytonic*training*.parquet",
    "*scholarios*.parquet",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_candidates(root: Path) -> list[Path]:
    expression: list[str] = []
    for index, pattern in enumerate(NAME_PATTERNS):
        if index:
            expression.append("-o")
        expression.extend(("-iname", pattern))
    command = [
        "find",
        str(root),
        "-xdev",
        "(",
        "-type",
        "f",
        "-o",
        "-type",
        "l",
        ")",
        "(",
        *expression,
        ")",
        "-print0",
    ]
    completed = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    return [Path(raw.decode()) for raw in completed.stdout.split(b"\0") if raw]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable receipt: {args.output}")

    candidates: dict[str, dict[str, object]] = {}
    searched_roots: list[dict[str, object]] = []
    for root in args.root:
        resolved = root.resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(resolved)
        found = find_candidates(resolved)
        searched_roots.append({"path": str(resolved), "matches": len(found)})
        for path in found:
            resolved_path = path.resolve()
            if resolved_path.is_file():
                key = str(resolved_path)
                record = candidates.setdefault(
                    key,
                    {
                        "path": resolved_path,
                        "discovered_paths": set(),
                        "exact_expected_name": False,
                    },
                )
                discovered_paths = record["discovered_paths"]
                assert isinstance(discovered_paths, set)
                discovered_paths.add(str(path.absolute()))
                record["exact_expected_name"] = bool(record["exact_expected_name"]) or path.name == "poly_train.parquet"

    records: list[dict[str, object]] = []
    for candidate in sorted(candidates.values(), key=lambda item: str(item["path"])):
        path = candidate["path"]
        assert isinstance(path, Path)
        stat = path.stat()
        records.append(
            {
                "path": str(path),
                "discovered_paths": sorted(candidate["discovered_paths"]),
                "basename": path.name,
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256_file(path),
                "exact_expected_name": bool(candidate["exact_expected_name"]),
            }
        )

    payload = {
        "schema_version": "targeted_8b_existing_poly_artifact_search_v1",
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "discovery_only": True,
            "reconstruction_performed": False,
            "substitution_performed": False,
            "artifact_authorities": ["repository", "huggingface", "cscs"],
        },
        "name_patterns": list(NAME_PATTERNS),
        "searched_roots": searched_roots,
        "candidate_count": len(records),
        "exact_name_candidate_count": sum(bool(item["exact_expected_name"]) for item in records),
        "candidates": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=args.output.parent, delete=False) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        temp = Path(stream.name)
    os.replace(temp, args.output)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
