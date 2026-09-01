#!/usr/bin/env python3
"""Hash and receipt one exact full-8B segment checkpoint."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--schedule-manifest", type=Path, required=True)
    parser.add_argument("--selected-profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    boundaries = recipe["segments"]["boundaries"]
    selected_profile_sha256 = None
    if args.selected_profile:
        selected = json.loads(args.selected_profile.read_text())
        if selected.get("schema_version") != "apertus_full_8b_selected_execution_profile_v1" or selected.get("status") != "frozen":
            raise ValueError("selected execution profile is not frozen")
        boundaries = selected["selection"]["segment_boundaries"]
        selected_profile_sha256 = sha256(args.selected_profile)
    if args.iteration not in boundaries[1:]:
        raise ValueError("iteration is not an exact segment boundary")
    marker = args.checkpoint_root / "latest_checkpointed_iteration.txt"
    if marker.read_text().strip() != str(args.iteration):
        raise ValueError("latest-checkpoint marker does not match the segment boundary")
    directory = args.checkpoint_root / f"iter_{args.iteration:07d}"
    if not (directory / ".metadata").is_file():
        raise ValueError("torch_dist metadata is absent")
    files = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        files.append(
            {
                "path": str(path.relative_to(directory)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    canonical = json.dumps(files, separators=(",", ":"), sort_keys=True)
    payload = {
        "schema_version": "apertus_full_8b_checkpoint_receipt_v1",
        "status": "frozen",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "iteration": args.iteration,
        "tokens": args.iteration * 4_194_304,
        "terminal": args.iteration == recipe["batch_and_parallelism"]["training_updates"],
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "checkpoint_directory": str(directory.resolve()),
        "files": len(files),
        "bytes": sum(row["bytes"] for row in files),
        "tree_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "recipe_sha256": sha256(args.recipe),
        "schedule_manifest_sha256": sha256(args.schedule_manifest),
        "selected_execution_profile_sha256": selected_profile_sha256,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        existing = json.loads(args.output.read_text())
        if existing.get("tree_sha256") != payload["tree_sha256"]:
            raise ValueError("existing receipt binds different checkpoint bytes")
        return 0
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".partial", dir=args.output.parent
    )
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    print(json.dumps({"ok": True, "iteration": args.iteration, "files": len(files)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
