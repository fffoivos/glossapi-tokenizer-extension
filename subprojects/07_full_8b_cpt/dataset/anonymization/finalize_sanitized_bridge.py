#!/usr/bin/env python3
"""Validate every sanitized shard and publish a training-bridge receipt."""

from __future__ import annotations

import argparse
import collections
import json
import shutil
from pathlib import Path
from typing import Any

from anonymization_common import (
    DEDUP_SCHEMA,
    SHARD_SCHEMA,
    absolute_receipt,
    canonical_sha256,
    load_overlay_heldouts,
    load_parent,
    read_json,
    sha256_file,
    utc_now,
    validate_file_receipt,
    validate_overlay,
    write_json_atomic,
)
from bridge_common import iter_index_lengths, task_output_prefix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--dedup-receipt", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    args = parser.parse_args()

    overlay_path = args.overlay.resolve()
    overlay = validate_overlay(overlay_path, Path(__file__))
    parent = load_parent(overlay)
    heldouts = load_overlay_heldouts(overlay, overlay_path, args.heldout_manifest)
    dedup_path = args.dedup_receipt.resolve()
    dedup = read_json(dedup_path)
    if (
        dedup.get("schema_version") != DEDUP_SCHEMA
        or dedup.get("status") != "completed"
        or dedup.get("overlay_sha256") != sha256_file(overlay_path)
    ):
        raise ValueError("invalid post-mask deduplication receipt")
    stage = args.stage_root.resolve()
    output = stage / "sanitized_bridge_receipt.json"
    if output.exists():
        value = read_json(output)
        if value.get("overlay_sha256") != sha256_file(overlay_path):
            raise ValueError("existing sanitized bridge receipt drift")
        print(json.dumps({"ok": True, "resumed": True, "output": str(output)}, sort_keys=True))
        return 0

    pool_counts: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"tasks": 0, "documents": 0, "tokens": 0, "masked_documents": 0,
                 "email_matches": 0, "ip_matches": 0, "iban_matches": 0,
                 "greekmmlu_dropped": 0, "postmask_dropped": 0}
    )
    manifests: list[dict[str, Any]] = []
    for task_index, task in enumerate(parent["tasks"]):
        prefix = task_output_prefix(stage, task)
        path = Path(str(prefix) + ".manifest.json")
        value = read_json(path)
        if (
            value.get("schema_version") != SHARD_SCHEMA
            or value.get("status") != "completed"
            or value.get("kind") != "training"
            or value.get("task_index") != task_index
            or value.get("task_sha256") != canonical_sha256(task)
            or value.get("anonymization_overlay_sha256") != sha256_file(overlay_path)
            or value.get("postmask_dedup_receipt_sha256") != sha256_file(dedup_path)
        ):
            raise ValueError(f"invalid sanitized shard manifest: {path}")
        for key in ("bin", "idx", "dropped_ledger", "retained_ledger"):
            validate_file_receipt(value["outputs"][key])
        sequences, documents, tokens = iter_index_lengths(Path(value["outputs"]["idx"]["path"]))
        counts = value["counts"]
        if (sequences, documents, tokens) != (
            int(counts["documents"]), int(counts["document_index_entries"]), int(counts["tokens"])
        ):
            raise ValueError(f"sanitized shard index accounting drift: {path}")
        logical_pool = str(task["output_prefix"]).split("/", 2)[1]
        row = pool_counts[logical_pool]
        row["tasks"] += 1
        row["documents"] += documents
        row["tokens"] += tokens
        row["masked_documents"] += int(counts["masked_documents"])
        row["email_matches"] += int(counts["email_matches"])
        row["ip_matches"] += int(counts["ip_matches"])
        row["iban_matches"] += int(counts["iban_matches"])
        row["greekmmlu_dropped"] += int(counts["contaminated_rows"])
        row["postmask_dropped"] += int(counts["postmask_dropped_rows"])
        manifests.append({"task_index": task_index, **absolute_receipt(path)})

    # Reuse the frozen raw validation binaries without copying payloads. The
    # copied manifests remain byte-identical and point to the immutable parent.
    parent_stage = Path(overlay["parent_heldout_manifest"]["path"]).resolve().parents[1]
    source_heldout_dir = parent_stage / "megatron" / "heldout"
    target_heldout_dir = stage / "megatron" / "heldout"
    target_heldout_dir.mkdir(parents=True, exist_ok=False)
    heldout_manifests: list[dict[str, Any]] = []
    for source in sorted(source_heldout_dir.glob("*.manifest.json")):
        value = read_json(source)
        if value.get("status") != "completed" or value.get("kind") != "heldout":
            raise ValueError(f"invalid parent heldout binary: {source}")
        target = target_heldout_dir / source.name
        shutil.copyfile(source, target)
        if sha256_file(source) != sha256_file(target):
            raise RuntimeError("heldout manifest copy drift")
        heldout_manifests.append(absolute_receipt(target))
    if len(heldout_manifests) != len(heldouts["sets"]):
        raise ValueError("heldout binary inventory does not close")

    modern_tokens = pool_counts["hplt_new_greek"]["tokens"] + pool_counts["non_hplt_new_greek"]["tokens"]
    payload = {
        "schema_version": "full_cpt_sanitized_training_bridge_v1",
        "status": "completed",
        "completed_at": utc_now(),
        "stage_root": str(stage),
        "overlay": absolute_receipt(overlay_path),
        "overlay_sha256": sha256_file(overlay_path),
        "postmask_dedup": absolute_receipt(dedup_path),
        "task_manifests": manifests,
        "heldout_manifests": heldout_manifests,
        "pool_counts": dict(pool_counts),
        "expected_inventory_tokens": {
            "modern": modern_tokens,
            "foreign": pool_counts["foreign_replay"]["tokens"],
            "old_greek": pool_counts["old_greek_replay"]["tokens"],
        },
        "anonymization": overlay["anonymization"],
        "validation_is_raw_and_frozen": True,
    }
    write_json_atomic(output, payload)
    print(json.dumps({"ok": True, "tasks": len(manifests), "pool_counts": dict(pool_counts), "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
