#!/usr/bin/env python3
"""Prove that a successor inventory changes only receipt/path bindings."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def diff(left: Any, right: Any, path: str = "$") -> list[str]:
    if type(left) is not type(right):
        return [path]
    if isinstance(left, dict):
        if set(left) != set(right):
            return [path]
        rows: list[str] = []
        for key in sorted(left):
            rows.extend(diff(left[key], right[key], f"{path}.{key}"))
        return rows
    if isinstance(left, list):
        if len(left) != len(right):
            return [path]
        rows = []
        for index, (first, second) in enumerate(zip(left, right)):
            rows.extend(diff(first, second, f"{path}[{index}]"))
        return rows
    return [] if left == right else [path]


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def normalized_plan(value: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(value))
    result["pool_corpus_receipt"] = "<REBINDING>"
    for task in result["tasks"]:
        task["catalog"]["path"] = Path(task["catalog"]["path"]).name
        source = task["catalog"]["selection_source_catalog"]
        source["path"] = Path(source["path"]).name
    return result


def normalized_pool(value: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(value))
    result["sanitized_bridge"]["sha256"] = "<REBINDING>"
    for row in result["sorted_training_catalogs"].values():
        row["path"] = Path(row["path"]).name
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-stage", type=Path, required=True)
    parser.add_argument("--successor-stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent_plan_path = args.parent_stage / "inventory/packing_plan.json"
    successor_plan_path = args.successor_stage / "inventory/packing_plan.json"
    parent_pool_path = args.parent_stage / "inventory/pool_corpus_receipt.json"
    successor_pool_path = args.successor_stage / "inventory/pool_corpus_receipt.json"
    parent_plan = read(parent_plan_path); successor_plan = read(successor_plan_path)
    parent_pool = read(parent_pool_path); successor_pool = read(successor_pool_path)

    plan_diffs = diff(parent_plan, successor_plan)
    allowed_plan = {
        "$.pool_corpus_receipt.path", "$.pool_corpus_receipt.sha256",
        *{f"$.tasks[{index}].catalog.path" for index in range(512)},
        *{f"$.tasks[{index}].catalog.selection_source_catalog.path" for index in range(512)},
    }
    pool_diffs = diff(parent_pool, successor_pool)
    allowed_pool = {
        "$.sanitized_bridge.sha256",
        *{
            f"$.sorted_training_catalogs.{name}.path"
            for name in ("foreign_replay", "hplt_new_greek", "non_hplt_new_greek", "old_greek_replay")
        },
    }
    checks = {
        "packing_plan_has_512_tasks": len(parent_plan.get("tasks", [])) == len(successor_plan.get("tasks", [])) == 512,
        "packing_plan_differences_are_only_rebindings": set(plan_diffs) == allowed_plan,
        "pool_differences_are_only_rebindings": set(pool_diffs) == allowed_pool,
        "normalized_packing_plans_are_identical": normalized_plan(parent_plan) == normalized_plan(successor_plan),
        "normalized_pools_are_identical": normalized_pool(parent_pool) == normalized_pool(successor_pool),
        "geometry_is_identical": parent_plan.get("geometry") == successor_plan.get("geometry"),
        "pool_geometry_is_identical": parent_pool.get("integer_79_20_1_geometry") == successor_pool.get("integer_79_20_1_geometry"),
    }
    if not all(checks.values()):
        raise ValueError({"checks": checks, "plan_diffs": plan_diffs[:20], "pool_diffs": pool_diffs})
    payload = {
        "schema_version": "apertus_full_8b_successor_stage_identity_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "parent_stage": str(args.parent_stage.resolve()),
        "successor_stage": str(args.successor_stage.resolve()),
        "parent_plan": binding(parent_plan_path),
        "successor_plan": binding(successor_plan_path),
        "parent_pool": binding(parent_pool_path),
        "successor_pool": binding(successor_pool_path),
        "checks": checks,
        "task_count": 512,
        "active_tokens": int(successor_pool["integer_79_20_1_geometry"]["active_tokens"]),
        "normalized_plan_sha256": canonical_sha(normalized_plan(successor_plan)),
        "normalized_pool_sha256": canonical_sha(normalized_pool(successor_pool)),
        "allowed_difference_counts": {"packing_plan": len(plan_diffs), "pool": len(pool_diffs)},
    }
    atomic_write(args.output, payload)
    print(json.dumps({"ok": True, "tasks": 512, "active_tokens": payload["active_tokens"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
