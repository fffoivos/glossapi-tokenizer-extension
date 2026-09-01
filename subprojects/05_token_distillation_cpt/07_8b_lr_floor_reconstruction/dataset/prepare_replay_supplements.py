#!/usr/bin/env python3
"""Create an eight-task, receipt-bound overlay for replay capacity deficits."""

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
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    recipe_path = args.recipe.resolve()
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    base_config = recipe["replay_repartition"]
    supplement_config = recipe["replay_supplements"]
    output_root = args.output_root.resolve()
    if output_root != Path(supplement_config["root"]).resolve():
        raise ValueError("supplement output root differs from frozen recipe")
    base_root = Path(base_config["root"]).resolve()
    base_input_path = base_root / "input_receipt.json"
    base_heldout_path = base_root / "heldouts" / "heldout_manifest.json"
    if sha(base_input_path) != base_config["input_receipt_sha256"]:
        raise ValueError("base replay input receipt drift")
    if sha(base_heldout_path) != base_config["heldout_manifest_sha256"]:
        raise ValueError("base replay heldout manifest drift")
    base_input = json.loads(base_input_path.read_text(encoding="utf-8"))
    base_heldouts = json.loads(base_heldout_path.read_text(encoding="utf-8"))
    download_path = output_root / "download_receipt.json"
    download = json.loads(download_path.read_text(encoding="utf-8"))
    if (
        download.get("schema_version") != "lr13_replay_supplement_download_v1"
        or download.get("status") != "completed"
        or download.get("recipe", {}).get("sha256") != sha(recipe_path)
        or download.get("repo_id") != supplement_config["repo_id"]
        or download.get("revision") != supplement_config["revision"]
    ):
        raise ValueError("supplement download receipt drift")
    templates = {}
    for task in base_input["tasks"]:
        if task["pool"] == "foreign_replay":
            templates.setdefault(task["source_name"], task)
    phase_path = (args.code_root / "dataset" / "replay_phase_partition.py").resolve()
    phase_binding = {"path": str(phase_path), "sha256": sha(phase_path)}
    tasks = []
    for item in sorted(download["files"], key=lambda row: row["source_name"]):
        source_name = item["source_name"]
        if source_name not in templates:
            raise ValueError(f"no vetted base replay template for {source_name}")
        local = Path(item["local_path"]).resolve()
        if (
            not local.is_file()
            or local.stat().st_size != int(item["bytes"])
            or sha(local) != item["sha256"]
        ):
            raise ValueError(f"supplement payload drift: {local}")
        base = dict(templates[source_name])
        for key in ("phase_partition", "output_prefix", "task_index", "task_id"):
            base.pop(key, None)
        base.update(
            {
                "input_path": str(local),
                "input_relative": f"supplements/{item['path']}",
                "input_bytes": int(item["bytes"]),
                "input_rows": int(item["rows"]),
                "input_sha256": item["sha256"],
            }
        )
        for phase in (1, 2):
            task = dict(base)
            task["task_index"] = len(tasks)
            task["task_id"] = f"lr13-supplement-{len(tasks):05d}-{source_name}"
            task["phase_partition"] = {
                "implementation": phase_binding,
                "seed": int(recipe["seed"]),
                "corpus": "replay",
                "phase": phase,
                "logical_pool": "foreign_replay",
            }
            task["output_prefix"] = (
                f"phase_{phase}/foreign_replay/{source_name}/"
                f"supplement_{Path(item['path']).stem}_text_document"
            )
            tasks.append(task)
    if len(tasks) != int(supplement_config["expected_tasks"]):
        raise ValueError(f"supplement task-count drift: {len(tasks)}")
    input_path = output_root / "input_receipt.json"
    new_input = dict(base_input)
    new_input.update(
        {
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "recipe_id": recipe["recipe_id"],
            "config": {"path": str(recipe_path), "sha256": sha(recipe_path)},
            "tasks": tasks,
            "tasks_sha256": canonical(tasks),
            "heldout_tasks": [],
            "heldout_tasks_sha256": canonical([]),
            "parent_input_receipt": {
                "path": str(base_input_path),
                "sha256": sha(base_input_path),
            },
            "supplement_download_receipt": {
                "path": str(download_path),
                "sha256": sha(download_path),
            },
        }
    )
    write(input_path, new_input)
    input_sha = sha(input_path)
    exclusions = {}
    for name, receipt in base_heldouts["exclusions"].items():
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
        exclusions[name] = {
            **receipt,
            "path": str(target),
            "bytes": target.stat().st_size,
            "sha256": sha(target),
        }
    new_heldouts = dict(base_heldouts)
    new_heldouts.update(
        {
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "input_receipt": str(input_path),
            "input_receipt_sha256": input_sha,
            "exclusions": exclusions,
            "parent_heldout_manifest": {
                "path": str(base_heldout_path),
                "sha256": sha(base_heldout_path),
            },
        }
    )
    heldout_path = output_root / "heldouts" / "heldout_manifest.json"
    write(heldout_path, new_heldouts)
    print(
        json.dumps(
            {
                "ok": True,
                "input_receipt": str(input_path),
                "input_sha256": input_sha,
                "heldout_manifest": str(heldout_path),
                "tasks": len(tasks),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
