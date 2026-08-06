#!/usr/bin/env python3
"""Promote only byte-identical masked inventories from a prior stopped stage."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from anonymization_common import (
    INVENTORY_SCHEMA,
    absolute_receipt,
    canonical_sha256,
    read_json,
    sha256_file,
    task_manifest_path,
    utc_now,
    validate_file_receipt,
    validate_overlay,
    write_json_atomic,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--prior-stage-root", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    overlay_path = args.overlay.resolve()
    overlay = validate_overlay(overlay_path, Path(__file__))
    prior_root = args.prior_stage_root.resolve()
    prior_overlay_path = prior_root / "anonymization_overlay.json"
    prior_overlay = read_json(prior_overlay_path)
    if (
        prior_overlay.get("schema_version") != overlay["schema_version"]
        or prior_overlay.get("status") != "frozen"
        or prior_overlay.get("anonymization", {}).get("implementation", {}).get("sha256")
        != overlay.get("anonymization", {}).get("implementation", {}).get("sha256")
        or prior_overlay.get("parent_input_receipt") != overlay.get("parent_input_receipt")
    ):
        raise ValueError("prior inventory stage is not compatible")
    prior_tasks = prior_overlay.get("tasks", [])
    tasks = overlay["tasks"]
    if len(prior_tasks) != len(tasks):
        raise ValueError("prior/new task inventory length drift")

    promoted: list[dict[str, Any]] = []
    already_built: list[int] = []
    changed: list[int] = []
    for task_index, (prior_task, task) in enumerate(zip(prior_tasks, tasks)):
        if canonical_sha256(prior_task) != canonical_sha256(task):
            changed.append(task_index)
            continue
        prior_manifest_path = task_manifest_path(prior_root, task_index, "inventory")
        prior_manifest = read_json(prior_manifest_path)
        if (
            prior_manifest.get("schema_version") != INVENTORY_SCHEMA
            or prior_manifest.get("status") != "completed"
            or prior_manifest.get("task_index") != task_index
            or prior_manifest.get("task_sha256") != canonical_sha256(task)
            or prior_manifest.get("overlay_sha256") != sha256_file(prior_overlay_path)
        ):
            raise ValueError(f"prior inventory manifest drift: {prior_manifest_path}")
        validate_file_receipt(prior_manifest["output"])
        target = task_manifest_path(args.stage_root.resolve(), task_index, "inventory")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            current = read_json(target)
            if (
                current.get("schema_version") != INVENTORY_SCHEMA
                or current.get("status") != "completed"
                or current.get("task_index") != task_index
                or current.get("task_sha256") != canonical_sha256(task)
                or current.get("overlay_sha256") != sha256_file(overlay_path)
            ):
                raise ValueError(f"existing current inventory manifest drift: {target}")
            validate_file_receipt(current["output"])
            already_built.append(task_index)
            continue
        value = copy.deepcopy(prior_manifest)
        value["completed_at"] = utc_now()
        value["overlay"] = absolute_receipt(overlay_path)
        value["overlay_sha256"] = sha256_file(overlay_path)
        value["reuse"] = {
            "policy": "promote_only_canonical_task_and_masker_identical_inventory_v1",
            "prior_overlay": absolute_receipt(prior_overlay_path),
            "prior_manifest": absolute_receipt(prior_manifest_path),
            "payload_rehashed": True,
            "scientific_task_changed": False,
        }
        write_json_atomic(target, value)
        promoted.append({
            "task_index": task_index,
            "task_sha256": canonical_sha256(task),
            "manifest": absolute_receipt(target),
            "payload": dict(prior_manifest["output"]),
        })

    if not promoted or not changed:
        raise ValueError("reuse gate expected both compatible and changed tasks")
    if any(tasks[index].get("source_name") != "cleaned_greek_v2" for index in changed):
        raise ValueError("a non-Modern-Greek task unexpectedly changed")
    payload = {
        "schema_version": "full_cpt_compatible_inventory_promotion_v1",
        "status": "completed",
        "completed_at": utc_now(),
        "overlay": absolute_receipt(overlay_path),
        "prior_overlay": absolute_receipt(prior_overlay_path),
        "counts": {
            "tasks": len(tasks),
            "promoted_identical_tasks": len(promoted),
            "compatible_tasks_already_built_by_smoke": len(already_built),
            "changed_tasks_requiring_rebuild": len(changed),
        },
        "promoted": promoted,
        "changed_task_indices_sha256": canonical_sha256(changed),
    }
    write_json_atomic(args.output.resolve(), payload)
    print(json.dumps({"ok": True, **payload["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
