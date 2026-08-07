#!/usr/bin/env python3
"""Globally deduplicate masked text and remove raw-validation collisions."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import IO, Any, Iterator

from anonymization_common import (
    DEDUP_SCHEMA,
    INVENTORY_SCHEMA,
    absolute_receipt,
    canonical_sha256,
    load_parent,
    read_json,
    sha256_file,
    task_manifest_path,
    utc_now,
    validate_file_receipt,
    validate_overlay,
    write_json_atomic,
)
from pii_masker import mask


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            yield value


def validation_hashes(
    manifest: dict[str, Any],
) -> tuple[set[str], list[dict[str, Any]], dict[str, int]]:
    hashes: set[str] = set()
    panels: list[dict[str, Any]] = []
    raw_hashes: set[str] = set()
    masked_hashes: set[str] = set()
    masked_documents = 0
    for panel in manifest["panels"]:
        receipt = panel["raw_jsonl"]
        path = validate_file_receipt(receipt)
        observed = 0
        for row in iter_jsonl(path):
            text = row.get("text")
            if not isinstance(text, str) or not text:
                raise ValueError(f"validation row lacks text: {path}")
            raw_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            masked_text, _ = mask(text)
            masked_digest = hashlib.sha256(masked_text.encode("utf-8")).hexdigest()
            raw_hashes.add(raw_digest)
            masked_hashes.add(masked_digest)
            hashes.add(raw_digest)
            hashes.add(masked_digest)
            masked_documents += int(masked_text != text)
            observed += 1
        if observed != int(receipt["rows"]):
            raise ValueError(f"validation row-count drift: {path}")
        panels.append({"name": panel["name"], "raw_jsonl": dict(receipt), "documents": observed})
    return hashes, panels, {
        "raw_unique_hashes": len(raw_hashes),
        "masked_unique_hashes": len(masked_hashes),
        "union_unique_hashes": len(hashes),
        "documents_changed_by_masker": masked_documents,
    }


class DropWriters:
    def __init__(self, root: Path, limit: int = 128):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=False)
        self.limit = limit
        self.handles: collections.OrderedDict[int, IO[str]] = collections.OrderedDict()
        self.counts: collections.Counter[int] = collections.Counter()

    def write(self, task_index: int, doc_id: str, reason: str) -> None:
        handle = self.handles.pop(task_index, None)
        if handle is None:
            path = self.root / f"task_{task_index:05d}.drops.tsv"
            handle = path.open("a", encoding="ascii")
        self.handles[task_index] = handle
        handle.write(f"{doc_id}\t{reason}\n")
        self.counts[task_index] += 1
        if len(self.handles) > self.limit:
            _, oldest = self.handles.popitem(last=False)
            oldest.close()

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


def parse_catalog_line(line: str) -> tuple[str, int, str]:
    fields = line.rstrip("\n").split("\t")
    if len(fields) != 3 or len(fields[0]) != 64:
        raise ValueError("malformed post-mask catalog row")
    int(fields[0], 16)
    task_index = int(fields[1])
    doc_id = fields[2]
    if not doc_id:
        raise ValueError("empty document identity in post-mask catalog")
    return fields[0], task_index, doc_id


def survivor_key(tasks: list[dict[str, Any]], task_index: int, doc_id: str) -> tuple[int, int, str]:
    """Prefer quota-limited Old-Greek replay, then preserve the legacy ordering.

    The original task-index rule accidentally made the two Old-Greek replay
    tasks (the final tasks in the frozen overlay) lose every cross-pool exact
    duplicate.  That destroyed the capacity required for the predeclared 1%
    replay stream.  The text still survives only once: ownership is transferred
    to Old Greek when a duplicate group contains an Old-Greek replay row.
    """
    task = tasks[task_index]
    old_greek_priority = 0 if task.get("pool") == "old_greek_replay" else 1
    return old_greek_priority, task_index, doc_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--sort-temp", type=Path, required=True)
    parser.add_argument("--sort-parallel", type=int, default=128)
    parser.add_argument("--sort-memory", default="320G")
    args = parser.parse_args()

    overlay_path = args.overlay.resolve()
    overlay = validate_overlay(overlay_path, Path(__file__))
    parent = load_parent(overlay)
    tasks = overlay["tasks"]
    task_count = len(tasks)
    stage = args.stage_root.resolve()
    output_path = stage / "postmask_dedup" / "dedup_receipt.json"
    if output_path.exists():
        value = read_json(output_path)
        if value.get("schema_version") != DEDUP_SCHEMA or value.get("overlay_sha256") != sha256_file(overlay_path):
            raise ValueError("existing post-mask dedup receipt binding drift")
        print(json.dumps({"ok": True, "resumed": True, "output": str(output_path)}, sort_keys=True))
        return 0

    manifests: list[dict[str, Any]] = []
    catalogs: list[Path] = []
    input_rows = 0
    for task_index, task in enumerate(tasks):
        path = task_manifest_path(stage, task_index, "inventory")
        value = read_json(path)
        if (
            value.get("schema_version") != INVENTORY_SCHEMA
            or value.get("status") != "completed"
            or value.get("task_index") != task_index
            or value.get("task_sha256") != canonical_sha256(task)
            or value.get("overlay_sha256") != sha256_file(overlay_path)
        ):
            raise ValueError(f"invalid inventory manifest: {path}")
        catalog = validate_file_receipt(value["output"])
        catalogs.append(catalog)
        input_rows += int(value["output"]["rows"])
        manifests.append(absolute_receipt(path))

    validation_manifest = read_json(validate_file_receipt(overlay["validation_manifest"]))
    validation_set, panel_receipts, validation_hash_counts = validation_hashes(
        validation_manifest
    )
    dedup_root = output_path.parent
    dedup_root.mkdir(parents=True, exist_ok=False)
    args.sort_temp.mkdir(parents=True, exist_ok=True)
    sorted_path = dedup_root / "all_tasks.sorted.postmask.tsv"
    command = [
        "sort", "-T", str(args.sort_temp.resolve()),
        f"--parallel={args.sort_parallel}", f"--buffer-size={args.sort_memory}",
        "-k1,1", "-k2,2n", "-k3,3", "-o", str(sorted_path),
        *[str(path) for path in catalogs],
    ]
    environment = dict(os.environ)
    environment["LC_ALL"] = "C"
    subprocess.run(command, check=True, env=environment)

    writers = DropWriters(dedup_root / "drops")
    duplicate_groups = 0
    duplicate_documents = 0
    validation_collision_groups = 0
    validation_collision_documents = 0
    kept_documents = 0
    observed_rows = 0
    old_greek_priority_overrides = 0
    current_hash: str | None = None
    group: list[tuple[int, str]] = []

    def close_group(digest: str | None, rows: list[tuple[int, str]]) -> None:
        nonlocal duplicate_groups, duplicate_documents
        nonlocal validation_collision_groups, validation_collision_documents, kept_documents
        nonlocal old_greek_priority_overrides
        if digest is None or not rows:
            return
        if digest in validation_set:
            validation_collision_groups += 1
            validation_collision_documents += len(rows)
            for task_index, doc_id in rows:
                writers.write(task_index, doc_id, "validation_content_collision")
            return
        kept_documents += 1
        if len(rows) > 1:
            duplicate_groups += 1
            duplicate_documents += len(rows) - 1
            prioritized = sorted(rows, key=lambda row: survivor_key(tasks, row[0], row[1]))
            if prioritized[0] != rows[0]:
                old_greek_priority_overrides += 1
            for task_index, doc_id in prioritized[1:]:
                writers.write(task_index, doc_id, "postmask_exact_duplicate")

    with sorted_path.open("r", encoding="ascii") as handle:
        for line in handle:
            digest, task_index, doc_id = parse_catalog_line(line)
            if task_index < 0 or task_index >= task_count:
                raise ValueError("catalog task index outside frozen inventory")
            if current_hash is not None and digest != current_hash:
                close_group(current_hash, group)
                group = []
            current_hash = digest
            group.append((task_index, doc_id))
            observed_rows += 1
    close_group(current_hash, group)
    writers.close()
    if observed_rows != input_rows:
        raise RuntimeError("sorted post-mask row accounting does not close")
    dropped_documents = duplicate_documents + validation_collision_documents
    if kept_documents + dropped_documents != input_rows:
        raise RuntimeError("post-mask deduplication accounting does not close")

    drop_receipts: list[dict[str, Any]] = []
    for task_index in range(task_count):
        path = dedup_root / "drops" / f"task_{task_index:05d}.drops.tsv"
        count = writers.counts[task_index]
        if count:
            drop_receipts.append({"task_index": task_index, **absolute_receipt(path, rows=count)})
    payload = {
        "schema_version": DEDUP_SCHEMA,
        "status": "completed",
        "completed_at": utc_now(),
        "overlay": absolute_receipt(overlay_path),
        "overlay_sha256": sha256_file(overlay_path),
        "policy": {
            "key": "sha256(masked_utf8_text)",
            "scope": "all_training_pools_and_phases",
            "survivor": "old_greek_replay_first_then_lowest_task_index_then_document_id",
            "survivor_reason": "preserve_exactly_once_global_dedup_and_the_predeclared_1pct_old_greek_capacity",
            "validation_collision_action": "drop_all_training_rows",
            "validation_representation": "union_of_raw_and_masked_frozen_utf8_text",
        },
        "counts": {
            "tasks": task_count,
            "input_documents": input_rows,
            "retained_documents": kept_documents,
            "dropped_documents": dropped_documents,
            "duplicate_groups": duplicate_groups,
            "duplicate_documents_dropped": duplicate_documents,
            "old_greek_priority_overrides": old_greek_priority_overrides,
            "validation_unique_hashes": len(validation_set),
            "validation_hash_representations": validation_hash_counts,
            "validation_collision_groups": validation_collision_groups,
            "validation_collision_documents_dropped": validation_collision_documents,
        },
        "inventory_manifests": manifests,
        "validation_panels": panel_receipts,
        "sorted_catalog": absolute_receipt(sorted_path, rows=observed_rows),
        "task_drop_files": drop_receipts,
    }
    write_json_atomic(output_path, payload)
    print(json.dumps({"ok": True, **payload["counts"], "output": str(output_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
