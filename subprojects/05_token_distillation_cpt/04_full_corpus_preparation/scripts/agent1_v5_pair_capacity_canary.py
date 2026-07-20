#!/usr/bin/env python3
"""Probe node-local capacity and durable I/O before the LSH pair merge."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Sequence


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def filesystem_receipt(path: Path) -> dict[str, int]:
    value = os.statvfs(path)
    fragment = int(value.f_frsize or value.f_bsize)
    return {
        "block_size": fragment,
        "total_bytes": int(value.f_blocks) * fragment,
        "free_bytes": int(value.f_bfree) * fragment,
        "available_bytes": int(value.f_bavail) * fragment,
        "total_inodes": int(value.f_files),
        "free_inodes": int(value.f_ffree),
    }


def write_exclusive(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def run_canary(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    work_directory = args.work_directory.resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.probe_bytes < 1024 * 1024 or args.probe_bytes > 1024**3:
        raise ValueError("probe bytes must be in [1 MiB, 1 GiB]")
    if args.sqlite_rows < 1_000 or args.sqlite_rows > 1_000_000:
        raise ValueError("SQLite rows must be in [1,000, 1,000,000]")
    work_directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    before = filesystem_receipt(work_directory)
    if before["available_bytes"] < args.required_free_bytes:
        write_exclusive(
            output,
            {
                "schema_version": "agent1_v5_pair_merge_capacity_canary_v1",
                "status": "blocked",
                "created_at": utc_now(),
                "reason": "insufficient_node_local_capacity",
                "work_directory": str(work_directory),
                "required_free_bytes": args.required_free_bytes,
                "filesystem_before": before,
            },
        )
        return 1

    probe = work_directory / "durable-io-probe.bin"
    chunk = bytes(range(256)) * (4 * 1024 * 1024 // 256)
    digest = hashlib.sha256()
    remaining = args.probe_bytes
    started = time.monotonic()
    with probe.open("xb") as handle:
        while remaining:
            block = chunk[: min(len(chunk), remaining)]
            handle.write(block)
            digest.update(block)
            remaining -= len(block)
        handle.flush()
        os.fsync(handle.fileno())
    write_seconds = max(time.monotonic() - started, 1e-9)
    started = time.monotonic()
    read_sha256 = sha256_file(probe)
    read_seconds = max(time.monotonic() - started, 1e-9)
    if read_sha256 != digest.hexdigest() or probe.stat().st_size != args.probe_bytes:
        raise ValueError("node-local durable I/O probe checksum drift")

    database = work_directory / "sqlite-capacity-canary.sqlite"
    started = time.monotonic()
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("CREATE TABLE probe(key INTEGER PRIMARY KEY, value BLOB NOT NULL)")
        rows = ((index, hashlib.sha256(str(index).encode()).digest()) for index in range(args.sqlite_rows))
        connection.executemany("INSERT INTO probe(key, value) VALUES (?, ?)", rows)
        connection.execute("CREATE INDEX probe_value ON probe(value)")
        connection.commit()
        count = int(connection.execute("SELECT COUNT(*) FROM probe").fetchone()[0])
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
    finally:
        connection.close()
    sqlite_seconds = max(time.monotonic() - started, 1e-9)
    if count != args.sqlite_rows or integrity != "ok":
        raise ValueError("node-local SQLite canary closure failed")
    sqlite_bytes = database.stat().st_size
    sqlite_sha256 = sha256_file(database)
    after = filesystem_receipt(work_directory)

    probe.unlink()
    database.unlink()
    work_directory.rmdir()
    payload: dict[str, object] = {
        "schema_version": "agent1_v5_pair_merge_capacity_canary_v1",
        "status": "passed",
        "created_at": utc_now(),
        "work_directory": str(work_directory),
        "work_device": os.stat(work_directory.parent).st_dev,
        "required_free_bytes": args.required_free_bytes,
        "filesystem_before": before,
        "filesystem_after_probe": after,
        "durable_probe": {
            "bytes": args.probe_bytes,
            "sha256": read_sha256,
            "write_seconds": write_seconds,
            "read_seconds": read_seconds,
            "write_bytes_per_second": args.probe_bytes / write_seconds,
            "read_bytes_per_second": args.probe_bytes / read_seconds,
        },
        "sqlite_probe": {
            "rows": count,
            "bytes": sqlite_bytes,
            "sha256": sqlite_sha256,
            "elapsed_seconds": sqlite_seconds,
            "rows_per_second": count / sqlite_seconds,
            "integrity_check": integrity,
        },
        "cleaned_up": True,
    }
    write_exclusive(output, payload)
    print(json.dumps({"ok": True, "output": str(output)}, sort_keys=True))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-free-bytes", type=int, default=64 * 1024**3)
    parser.add_argument("--probe-bytes", type=int, default=256 * 1024**2)
    parser.add_argument("--sqlite-rows", type=int, default=200_000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    return run_canary(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
