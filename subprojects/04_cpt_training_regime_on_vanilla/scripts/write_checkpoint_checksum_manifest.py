#!/usr/bin/env python3
"""Write a checksum manifest for one Megatron checkpoint and HF conversion.

This is intended to run on Clariden/xfer near the artifacts. It streams each
file once and writes a JSON manifest with sizes, mtimes, and SHA256 digests.
"""

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import tempfile


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--token-mark", type=int, required=True)
    parser.add_argument("--megatron-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--hf-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--chunk-mib", type=int, default=16)
    return parser.parse_args()


def utc_now():
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256_file(path, chunk_bytes):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_record(root, path, chunk_bytes):
    stat = path.stat()
    return {
        "path": str(path),
        "relative_path": str(path.relative_to(root)),
        "size_bytes": stat.st_size,
        "mtime_utc": dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "sha256": sha256_file(path, chunk_bytes),
    }


def collect(root, patterns, chunk_bytes):
    records = []
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                records.append(file_record(root, path, chunk_bytes))
    return records


def main():
    args = parse_args()
    chunk_bytes = args.chunk_mib * 1024 * 1024
    megatron_dir = args.megatron_checkpoint_dir
    hf_dir = args.hf_checkpoint_dir
    if not megatron_dir.is_dir():
        raise SystemExit("missing Megatron checkpoint dir: {}".format(megatron_dir))
    if not hf_dir.is_dir():
        raise SystemExit("missing HF checkpoint dir: {}".format(hf_dir))

    megatron = collect(
        megatron_dir,
        ["*.distcp", ".metadata", "*.json", "*.txt", "*.pt"],
        chunk_bytes,
    )
    hf = collect(
        hf_dir,
        ["*.safetensors", "*.json", "*.model", "*.txt", "*.py"],
        chunk_bytes,
    )
    manifest = {
        "schema": "04-vanilla-checkpoint-checksum-manifest-v1",
        "generated_utc": utc_now(),
        "checkpoint_label": args.checkpoint_label,
        "iteration": args.iteration,
        "token_mark": args.token_mark,
        "megatron_checkpoint_dir": str(megatron_dir),
        "hf_checkpoint_dir": str(hf_dir),
        "chunk_mib": args.chunk_mib,
        "megatron": {
            "file_count": len(megatron),
            "total_size_bytes": sum(item["size_bytes"] for item in megatron),
            "files": megatron,
        },
        "hf": {
            "file_count": len(hf),
            "total_size_bytes": sum(item["size_bytes"] for item in hf),
            "files": hf,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(args.output_json.parent),
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(args.output_json)
    print("wrote {}".format(args.output_json))
    print(
        "megatron_files={} hf_files={} total_bytes={}".format(
            len(megatron),
            len(hf),
            manifest["megatron"]["total_size_bytes"]
            + manifest["hf"]["total_size_bytes"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
