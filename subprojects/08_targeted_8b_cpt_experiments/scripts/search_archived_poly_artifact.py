#!/usr/bin/env python3
"""Find existing archive members that may contain the frozen poly_train file.

This is a discovery-only archive-index scan.  It lists members but never
extracts, materializes, rewrites, reconstructs, or substitutes corpus data.
Any reported member remains only a candidate until a separate exact-content
audit proves all frozen split facts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


ARCHIVE_SUFFIXES = (
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz2",
    ".tar.xz",
    ".txz",
    ".tar.zst",
    ".tzst",
    ".zip",
)
EXACT_BASENAME = "poly_train.parquet"
PATH_HINTS = ("c3p_polytonic_20260518t_impl", "polytonic_extension")


def discover_archive_paths(root: Path) -> list[Path]:
    """Return unique recognized archive files without following directories."""

    command = ["find", str(root), "-xdev", "(", "-type", "f", "-o", "-type", "l", ")", "-print0"]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    paths: dict[tuple[int, int], Path] = {}
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        discovered = Path(os.fsdecode(raw))
        if not discovered.name.lower().endswith(ARCHIVE_SUFFIXES):
            continue
        try:
            resolved = discovered.resolve(strict=True)
            stat = resolved.stat()
        except (FileNotFoundError, OSError, RuntimeError):
            continue
        if not resolved.is_file():
            continue
        paths.setdefault((stat.st_dev, stat.st_ino), resolved)
    return sorted(paths.values(), key=str)


def listing_command(path: Path) -> list[str]:
    if path.name.lower().endswith(".zip"):
        return ["unzip", "-Z1", str(path)]
    return ["tar", "-tf", str(path)]


def is_candidate_member(member: str) -> bool:
    normalized = member.replace("\\", "/").strip("/")
    lowered = normalized.lower()
    basename = lowered.rsplit("/", 1)[-1]
    return basename == EXACT_BASENAME or (
        basename.endswith(".parquet")
        and "poly" in basename
        and "train" in basename
        and any(hint in lowered for hint in PATH_HINTS)
    )


def inspect_archive(path: Path, timeout_seconds: int) -> dict[str, Any]:
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    try:
        completed = subprocess.run(
            listing_command(path),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        result.update(classification="listing_timeout", timeout_seconds=timeout_seconds)
        return result
    if completed.returncode != 0:
        result.update(
            classification="unreadable_archive",
            returncode=completed.returncode,
            stderr=completed.stderr.decode("utf-8", "replace")[-2_000:],
        )
        return result
    members = completed.stdout.decode("utf-8", "replace").splitlines()
    candidates = sorted({member for member in members if is_candidate_member(member)})
    result.update(
        classification="candidate_members_found" if candidates else "no_candidate_members",
        member_count=len(members),
        candidate_members=candidates,
    )
    return result


def write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-bytes", type=int, default=1_000_000)
    parser.add_argument("--maximum-bytes", type=int, default=20_000_000_000)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable receipt: {args.output}")
    if not 0 < args.minimum_bytes <= args.maximum_bytes or args.timeout_seconds <= 0:
        raise ValueError("invalid archive scan bounds")

    unique: dict[str, Path] = {}
    roots: list[dict[str, Any]] = []
    for root in args.root:
        resolved = root.resolve(strict=True)
        if not resolved.is_dir():
            raise NotADirectoryError(resolved)
        discovered = discover_archive_paths(resolved)
        archives = [
            path for path in discovered
            if args.minimum_bytes <= path.stat().st_size <= args.maximum_bytes
        ]
        roots.append({
            "path": str(resolved),
            "recognized_archives": len(discovered),
            "archives_within_scan_bounds": len(archives),
        })
        for archive in archives:
            unique.setdefault(str(archive), archive)

    all_unique: dict[str, Path] = {}
    for root in args.root:
        resolved = root.resolve(strict=True)
        for archive in discover_archive_paths(resolved):
            all_unique.setdefault(str(archive), archive)
    below = [
        {"path": str(path), "bytes": path.stat().st_size}
        for path in all_unique.values() if path.stat().st_size < args.minimum_bytes
    ]
    above = [
        {"path": str(path), "bytes": path.stat().st_size}
        for path in all_unique.values() if path.stat().st_size > args.maximum_bytes
    ]

    inspected = [inspect_archive(path, args.timeout_seconds) for path in unique.values()]
    matches = [row for row in inspected if row["classification"] == "candidate_members_found"]
    timeouts = [row for row in inspected if row["classification"] == "listing_timeout"]
    unreadable = [row for row in inspected if row["classification"] == "unreadable_archive"]
    payload = {
        "schema_version": "targeted_8b_archived_poly_artifact_search_v1",
        "status": "completed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "contract": {
            "discovery_only": True,
            "archive_members_extracted": 0,
            "corpus_files_written": 0,
            "reconstruction_performed": False,
            "substitution_performed": False,
            "candidate_is_not_authority_until_exact_content_audit": True,
            "artifact_authority": "cscs",
        },
        "archive_suffixes": list(ARCHIVE_SUFFIXES),
        "candidate_member_rule": {
            "exact_basename": EXACT_BASENAME,
            "path_hints": list(PATH_HINTS),
        },
        "archive_byte_bounds": [args.minimum_bytes, args.maximum_bytes],
        "per_archive_timeout_seconds": args.timeout_seconds,
        "searched_roots": roots,
        "plausible_unique_archives": len(inspected),
        "recognized_unique_archives": len(all_unique),
        "excluded_below_minimum_count": len(below),
        "excluded_below_minimum": sorted(below, key=lambda row: str(row["path"])),
        "excluded_above_maximum_count": len(above),
        "excluded_above_maximum": sorted(above, key=lambda row: str(row["path"])),
        "candidate_archive_count": len(matches),
        "candidate_archives": matches,
        "listing_timeout_count": len(timeouts),
        "listing_timeouts": timeouts,
        "unreadable_archive_count": len(unreadable),
        "unreadable_archives": unreadable,
    }
    write_atomic(args.output, payload)
    print(json.dumps({
        "ok": True,
        "archives": len(inspected),
        "candidate_archives": len(matches),
        "timeouts": len(timeouts),
        "unreadable": len(unreadable),
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
