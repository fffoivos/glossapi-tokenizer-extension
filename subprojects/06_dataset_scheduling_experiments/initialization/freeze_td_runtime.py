#!/usr/bin/env python3
"""Freeze the external canonical TD and Megatron conversion runtime."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path


MEGATRON_COMMIT = "c92402e39ef3c8e69ea378a59e79059dc14541f4"
EXPECTED = {
    "canonical_adapter": "c9d417bf8f28aaa9dc079c05641bdeae84b86df039b0ce3066f9e96955104462",
    "canonical_train_loop": "aa7128f8025ddde091460bdda2b65e927c98d5683b29d5bb2c573c7717263430",
    "canonical_utils": "82f96caa6e2e527e7d3881ee72e65c13a257e42b6a20fdc67d673dab06641672",
    "fair_metrics": "96c3b0e048753a3414e2ab9abf3b78824375fc0994793fe116d2dbffbb09b3ff",
    "megatron_loader": "5b0215969ef05257be22e50fd60f47498542a99637179fbcade73a61384b1b77",
    "megatron_saver": "4a113dbf49bb53afaaf138314c9270edc05aa5798bf905d314cd310a5291c57d",
    "patch_apertus_extras": "4263bc2e2604dd61fc1a492b8e140aa8f8c4c80da52569f347e9e5284046c9ac",
    "verify_hf_roundtrip": "3b45279fdd09f4e37032ae959a3619dc315f849f94470da12904ce92803951af",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def receipt(path: Path, expected: str) -> dict[str, object]:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"runtime dependency drift: {path}: {observed}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": observed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.runtime_root.resolve()
    adapter = root / "token_distillation" / "train_retok_td.py"
    vendor = adapter.parent / "external" / "token-distillation" / "token_distillation"
    metrics = root / "eval" / "compute_tokenizer_fair_metrics.py"
    megatron = root / "megatron"
    patches = root / "megatron_patches"
    commit = subprocess.check_output(
        ["git", "-C", str(megatron), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != MEGATRON_COMMIT:
        raise ValueError(f"Megatron commit drift: {commit}")
    tracked_diff = subprocess.run(
        ["git", "-C", str(megatron), "diff", "--quiet", "HEAD", "--"], check=False
    )
    if tracked_diff.returncode != 0:
        raise ValueError("Megatron worktree has tracked changes")
    files = {
        "canonical_adapter": receipt(adapter, EXPECTED["canonical_adapter"]),
        "canonical_train_loop": receipt(
            vendor / "train_loop.py", EXPECTED["canonical_train_loop"]
        ),
        "canonical_utils": receipt(vendor / "utils.py", EXPECTED["canonical_utils"]),
        "fair_metrics": receipt(metrics, EXPECTED["fair_metrics"]),
        "megatron_loader": receipt(
            megatron / "tools" / "checkpoint" / "loader_apertus_hf.py",
            EXPECTED["megatron_loader"],
        ),
        "megatron_saver": receipt(
            megatron / "tools" / "checkpoint" / "saver_swissai_hf.py",
            EXPECTED["megatron_saver"],
        ),
        "patch_apertus_extras": receipt(
            patches / "patch_apertus_extras.py", EXPECTED["patch_apertus_extras"]
        ),
        "verify_hf_roundtrip": receipt(
            patches / "verify_hf_roundtrip.py", EXPECTED["verify_hf_roundtrip"]
        ),
    }
    payload = {
        "schema_version": "apertus_mini_td_external_runtime_v1",
        "status": "frozen",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runtime_root": str(root),
        "megatron_commit": commit,
        "canonical_token_distillation_commit": "35702b5809599ecd68b7845eca27a0d7b7cec0da",
        "tracked_megatron_diff": False,
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output) + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"ok": True, "runtime_root": str(root), "megatron_commit": commit}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
