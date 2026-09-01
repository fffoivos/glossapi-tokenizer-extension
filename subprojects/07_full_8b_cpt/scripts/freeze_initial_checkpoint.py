#!/usr/bin/env python3
"""Hash and freeze the exact legacy TP2 initialization tree used by training."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

from contract import atomic_write_json, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if (root / "latest_checkpointed_iteration.txt").read_text().strip() != "release":
        raise ValueError("initial checkpoint tracker is not release")
    expected = {
        "release/mp_rank_00/model_optim_rng.pt",
        "release/mp_rank_01/model_optim_rng.pt",
        "latest_checkpointed_iteration.txt",
    }
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"initial checkpoint contains a symlink: {path}")
        if path.is_file():
            files.append({
                "relative_path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    if {row["relative_path"] for row in files} != expected:
        raise ValueError("initial checkpoint file inventory drift")
    canonical = json.dumps(files, separators=(",", ":"), sort_keys=True).encode()
    payload = {
        "schema_version": "apertus_full_8b_initial_checkpoint_tree_v1",
        "status": "frozen",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "root": str(root),
        "tensor_parallel": 2,
        "pipeline_parallel": 1,
        "tracker": "release",
        "file_count": len(files),
        "tree_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": files,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"ok": True, "files": len(files), "tree_sha256": payload["tree_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
