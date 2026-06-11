#!/usr/bin/env python3
"""Verify that Megatron GPTDataset cache indices preserve physical order."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def find_cache_dirs(data_prefix: Path, cache_dir: Path | None) -> list[Path]:
    if cache_dir is not None:
        return [cache_dir]
    return [
        p
        for p in (
            data_prefix / "cache" / "GPTDataset_indices",
            data_prefix.parent / "cache" / "GPTDataset_indices",
        )
        if p.is_dir()
    ]


def groups(cache: Path) -> list[tuple[Path, Path, Path]]:
    out = []
    for doc in sorted(cache.glob("*GPTDataset-train-document_index.npy")):
        stem = doc.name[: -len("document_index.npy")]
        sample = cache / f"{stem}sample_index.npy"
        shuffle = cache / f"{stem}shuffle_index.npy"
        if sample.exists() and shuffle.exists():
            out.append((doc, sample, shuffle))
    return out


def verify_one(doc_path: Path, sample_path: Path, shuffle_path: Path, probe_samples: int) -> dict:
    document_index = np.load(doc_path, mmap_mode="r")
    sample_index = np.load(sample_path, mmap_mode="r")
    shuffle_index = np.load(shuffle_path, mmap_mode="r")

    n = int(shuffle_index.shape[0])
    probe = min(n, probe_samples)
    shuffle_identity = bool(
        np.array_equal(shuffle_index[:probe], np.arange(probe, dtype=shuffle_index.dtype))
    )
    full_shuffle_identity = None
    if n <= probe_samples:
        full_shuffle_identity = bool(
            np.array_equal(shuffle_index[:], np.arange(n, dtype=shuffle_index.dtype))
        )

    doc_probe_n = min(int(document_index.shape[0]), max(probe, 2))
    doc_probe = np.asarray(document_index[:doc_probe_n])
    descents = np.where(doc_probe[1:] < doc_probe[:-1])[0]
    document_monotonic_probe = bool(descents.size == 0)

    sample_probe = min(probe, int(sample_index.shape[0]) - 1)
    sample_doc_positions = np.asarray(sample_index[:sample_probe, 0])
    sample_doc_non_decreasing = bool(
        sample_doc_positions.size < 2
        or np.all(sample_doc_positions[1:] >= sample_doc_positions[:-1])
    )

    ok = shuffle_identity and document_monotonic_probe and sample_doc_non_decreasing
    return {
        "ok": ok,
        "document_index": str(doc_path),
        "sample_index": str(sample_path),
        "shuffle_index": str(shuffle_path),
        "num_samples": n,
        "document_index_len": int(document_index.shape[0]),
        "sample_index_len": int(sample_index.shape[0]),
        "probe_samples": probe,
        "shuffle_identity_probe": shuffle_identity,
        "shuffle_identity_full": full_shuffle_identity,
        "document_index_monotonic_probe": document_monotonic_probe,
        "document_index_first_descents": [int(x) for x in descents[:10]],
        "sample_doc_positions_non_decreasing_probe": sample_doc_non_decreasing,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-prefix", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--probe-samples", type=int, default=100000)
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.cache_dir is None and args.data_prefix is None:
        raise SystemExit("pass --cache-dir or --data-prefix")
    cache_dirs = find_cache_dirs(args.data_prefix or Path("."), args.cache_dir)
    all_groups = []
    for cache in cache_dirs:
        all_groups.extend(groups(cache))
    if not all_groups:
        result = {
            "ok": bool(args.allow_missing),
            "reason": "no GPTDataset train index cache found",
            "cache_dirs": [str(p) for p in cache_dirs],
        }
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(("OK" if result["ok"] else "BAD"), result["reason"], result["cache_dirs"])
        return 0 if result["ok"] else 1

    results = [verify_one(*group, probe_samples=args.probe_samples) for group in all_groups]
    ok = all(r["ok"] for r in results)
    payload = {"ok": ok, "results": results}
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for r in results:
            print(("OK   " if r["ok"] else "BAD  ") + r["shuffle_index"])
            print(
                "      shuffle_identity_probe={shuffle_identity_probe} "
                "document_monotonic_probe={document_index_monotonic_probe} "
                "sample_doc_non_decreasing={sample_doc_positions_non_decreasing_probe}".format(**r)
            )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
