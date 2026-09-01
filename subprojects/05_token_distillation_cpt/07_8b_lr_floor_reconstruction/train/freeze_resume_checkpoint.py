#!/usr/bin/env python3
"""Freeze one shared-prefix or terminal LR-floor checkpoint."""

import argparse
import datetime as dt
import hashlib
import json
import os
import tempfile
from pathlib import Path


ALLOWED = {2253, 2574, 3218}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree(root: Path) -> dict:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and not item.is_symlink()):
        rows.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha(path)})
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return {"root": str(root.resolve()), "files": rows, "tree_sha256": hashlib.sha256(canonical.encode()).hexdigest()}


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--training-assets-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iteration not in ALLOWED:
        raise ValueError("iteration is not a frozen LR-floor boundary")
    assets = json.loads(args.training_assets_receipt.read_text(encoding="utf-8"))
    if assets.get("schema_version") != "apertus8b_lr_floor_training_assets_v1" or assets.get("status") != "frozen":
        raise ValueError("training assets receipt drift")
    checkpoint_dir = args.checkpoint_dir.resolve()
    marker = checkpoint_dir / "latest_checkpointed_iteration.txt"
    if marker.read_text().strip() != str(args.iteration):
        raise ValueError("checkpoint marker does not name the requested iteration")
    iteration_dir = checkpoint_dir / f"iter_{args.iteration:07d}"
    if not (iteration_dir / ".metadata").is_file():
        raise ValueError("complete torch_dist checkpoint metadata is absent")
    payload = {
        "schema_version": "apertus8b_lr_floor_checkpoint_v1",
        "status": "frozen",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "iteration": args.iteration,
        "terminal": args.iteration == 3218,
        "training_assets_receipt": {"path": str(args.training_assets_receipt.resolve()), "sha256": sha(args.training_assets_receipt)},
        "checkpoint_root": str(checkpoint_dir),
        "checkpoint_tree": tree(iteration_dir),
        "marker": {"path": str(marker), "sha256": sha(marker), "value": str(args.iteration)},
    }
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing["checkpoint_tree"]["tree_sha256"] != payload["checkpoint_tree"]["tree_sha256"]:
            raise ValueError("existing checkpoint receipt binds different bytes")
    else:
        write(args.output.resolve(), payload)
    print(json.dumps({"ok": True, "output": str(args.output.resolve()), "iteration": args.iteration}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
