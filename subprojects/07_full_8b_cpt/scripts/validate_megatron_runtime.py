#!/usr/bin/env python3
"""Revalidate the frozen patched Megatron runtime before a full-8B job."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


UPSTREAM = "c92402e39ef3c8e69ea378a59e79059dc14541f4"
RECEIPT_SHA = "99b9ecbd49bec162941d1b9bd11996b39ab421f4c43eca962c09ca37fdc7b36a"
DIFF_SHA = "550a9f570a99f1ca20773bd2d97795210ce153ace1be794cf9f11aeb4b2238e6"
PATCHES = {
    "megatron_extra_valid_c92402e.patch": "2e6810fa8b6c25597ccb3bcb9dc1ff5bf843ead2337e3edde0344605a23ec4c6",
    "megatron_exact_eval_iterations_c92402e.patch": "6d9392cfb0dd08e62089d0a98e2817b222bb9a25ee5cefa8f3cdf29a8ce16bea",
}
FILES = {
    "megatron/training/arguments.py",
    "megatron/training/training.py",
    "pretrain_gpt.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--megatron-root", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.receipt.resolve()
    root = args.megatron_root.resolve()
    if sha256(receipt_path) != RECEIPT_SHA:
        raise ValueError("Megatron runtime receipt hash drift")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema_version") != "apertus_mini_patched_megatron_runtime_v1"
        or receipt.get("status") != "frozen"
        or Path(receipt.get("output_root", "")).resolve() != root
        or receipt.get("upstream_commit") != UPSTREAM
        or receipt.get("git_diff_sha256") != DIFF_SHA
        or not receipt.get("checks")
        or not all(receipt["checks"].values())
    ):
        raise ValueError("Megatron runtime receipt contract drift")
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != UPSTREAM:
        raise ValueError(f"Megatron HEAD drift: {head}")
    diff = subprocess.check_output(["git", "-C", str(root), "diff", "--binary"])
    if hashlib.sha256(diff).hexdigest() != DIFF_SHA:
        raise ValueError("Megatron live git diff drift")
    observed_files = set()
    for row in receipt.get("patched_files", []):
        path = Path(row["path"]).resolve()
        try:
            relative = str(path.relative_to(root))
        except ValueError as error:
            raise ValueError(f"patched file outside runtime root: {path}") from error
        observed_files.add(relative)
        if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            raise ValueError(f"patched runtime file drift: {path}")
    if observed_files != FILES:
        raise ValueError("patched runtime file inventory drift")
    observed_patches = {Path(row["path"]).name: row["sha256"] for row in receipt.get("patches", [])}
    if observed_patches != PATCHES:
        raise ValueError("runtime patch receipt drift")
    print(json.dumps({"ok": True, "root": str(root), "upstream_commit": head, "git_diff_sha256": DIFF_SHA}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
