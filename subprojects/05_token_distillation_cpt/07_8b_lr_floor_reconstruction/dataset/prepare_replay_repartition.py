#!/usr/bin/env python3
"""Derive a receipt-bound replay-only input at the actual 2253/965 split."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    source_root = Path(recipe["source_stage"]["root"]).resolve()
    source_input_path = source_root / "input_receipt.json"
    source_heldout_path = source_root / "heldouts" / "heldout_manifest.json"
    if sha(source_input_path) != recipe["replay_repartition"]["source_input_receipt_sha256"]:
        raise ValueError("source input receipt drift")
    source = json.loads(source_input_path.read_text(encoding="utf-8"))
    heldouts = json.loads(source_heldout_path.read_text(encoding="utf-8"))
    phase_path = (args.code_root / "dataset" / "replay_phase_partition.py").resolve()
    phase_binding = {"path": str(phase_path), "sha256": sha(phase_path)}
    unique_sources = {}
    for task in source["tasks"]:
        if task["pool"] not in {"foreign_replay", "old_greek_replay"}:
            continue
        key = (task["pool"], task["source_name"], task["input_path"])
        base = dict(task)
        base.pop("phase_partition", None)
        base.pop("output_prefix", None)
        base.pop("task_index", None)
        base.pop("task_id", None)
        if key in unique_sources:
            comparable = dict(unique_sources[key]); comparable.pop("source_phase", None)
            if base != comparable:
                raise ValueError(f"source task pair drift: {key}")
        else:
            unique_sources[key] = base
    tasks = []
    for key in sorted(unique_sources):
        base = unique_sources[key]
        pool, source_name, _ = key
        for phase in (1, 2):
            task = dict(base)
            task["task_index"] = len(tasks)
            task["task_id"] = f"lr13-replay-{len(tasks):05d}-{source_name}"
            task["phase_partition"] = {"implementation": phase_binding, "seed": int(recipe["seed"]), "corpus": "replay", "phase": phase, "logical_pool": pool}
            stem = Path(str(task["input_relative"])).stem
            task["output_prefix"] = f"phase_{phase}/{pool}/{source_name}/{stem}_text_document"
            tasks.append(task)
    if len(tasks) != int(recipe["replay_repartition"]["expected_tasks"]):
        raise ValueError(f"expected 164 replay tasks, found {len(tasks)}")
    output_root = args.output_root.resolve()
    input_path = output_root / "input_receipt.json"
    new_input = dict(source)
    new_input.update({
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "recipe_id": recipe["recipe_id"],
        "config": {"path": str(args.recipe.resolve()), "sha256": sha(args.recipe)},
        "tasks": tasks,
        "tasks_sha256": canonical(tasks),
        "heldout_tasks": [],
        "heldout_tasks_sha256": canonical([]),
        "parent_input_receipt": {"path": str(source_input_path), "sha256": sha(source_input_path)},
    })
    write(input_path, new_input)
    input_sha = sha(input_path)
    exclusions = {}
    for name, receipt in heldouts["exclusions"].items():
        source_path = Path(receipt["path"]).resolve()
        target = output_root / "heldouts" / "exclusions" / source_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha(target) != receipt["sha256"]:
                raise ValueError(f"existing exclusion drift: {target}")
        else:
            temporary = target.with_suffix(target.suffix + ".partial")
            shutil.copyfile(source_path, temporary)
            os.replace(temporary, target)
        exclusions[name] = {**receipt, "path": str(target), "bytes": target.stat().st_size, "sha256": sha(target)}
    new_heldouts = dict(heldouts)
    new_heldouts.update({
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_receipt": str(input_path),
        "input_receipt_sha256": input_sha,
        "exclusions": exclusions,
        "parent_heldout_manifest": {"path": str(source_heldout_path), "sha256": sha(source_heldout_path)},
    })
    heldout_path = output_root / "heldouts" / "heldout_manifest.json"
    write(heldout_path, new_heldouts)
    print(json.dumps({"ok": True, "input_receipt": str(input_path), "input_sha256": input_sha, "heldout_manifest": str(heldout_path), "tasks": len(tasks)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
