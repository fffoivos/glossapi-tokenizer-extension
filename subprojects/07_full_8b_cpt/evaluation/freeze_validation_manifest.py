#!/usr/bin/env python3
"""Freeze the 12 inherited heldouts plus neutral external Greek for full 8B CPT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def checked(receipt: dict[str, Any]) -> Path:
    path = Path(receipt["path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(receipt["bytes"])
        or sha256_file(path) != receipt["sha256"]
    ):
        raise ValueError(f"validation input drift: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-receipt", type=Path, required=True)
    parser.add_argument("--neutral-manifest", type=Path, required=True)
    parser.add_argument("--neutral-corpus-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    pool = read_json(args.pool_receipt)
    neutral = read_json(args.neutral_manifest)
    neutral_corpus = read_json(args.neutral_corpus_receipt)
    if pool.get("schema_version") != "apertus_schedule_pool_corpus_v1":
        raise ValueError("unsupported pool receipt")
    if neutral.get("status") != "completed" or neutral.get("heldout_name") != "neutral_external_modern_greek":
        raise ValueError("neutral heldout is not complete")
    panels = []
    for row in pool["heldouts"]:
        raw = row["raw_jsonl"]
        checked(raw)
        panels.append(
            {
                **row,
                "raw_jsonl": raw,
                "cluster_field": None,
                "bootstrap_unit": "doc_id",
            }
        )
    for output in ("bin", "idx", "retained_ledger"):
        checked(neutral["outputs"][output])
    candidate = neutral_corpus["candidate_jsonl"]
    checked(candidate)
    panels.append(
        {
            "name": "neutral_external_modern_greek",
            "documents": int(neutral["counts"]["documents"]),
            "tokens": int(neutral["counts"]["tokens"]),
            "megatron_prefix": neutral["output_prefix"],
            "manifest_path": str(args.neutral_manifest.resolve()),
            "manifest_sha256": sha256_file(args.neutral_manifest),
            "raw_jsonl": candidate,
            "cluster_field": "cluster_id",
            "bootstrap_unit": "cluster_id",
        }
    )
    panels.sort(key=lambda row: row["name"])
    if len(panels) != 13 or len({row["name"] for row in panels}) != 13:
        raise ValueError("full validation manifest must contain 13 distinct panels")
    if any(int(row["tokens"]) < 4_194_304 for row in panels):
        raise ValueError("every fast panel must cover at least one full 8B global batch")
    payload = {
        "schema_version": "apertus_full_8b_validation_manifest_v1",
        "status": "frozen",
        "pool_corpus_receipt": {
            "path": str(args.pool_receipt.resolve()),
            "sha256": sha256_file(args.pool_receipt),
        },
        "fast_panel": {
            "fixed_loss_active_tokens_per_checkpoint": 4_194_304,
            "same_examples_every_checkpoint": True,
            "goldfish_masking": False,
        },
        "per_document": {
            "score_all_documents": True,
            "explicit_cluster_ids_available_for": ["neutral_external_modern_greek"],
            "other_panels_bootstrap_by_doc_id": True,
        },
        "panels": panels,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output) + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"ok": True, "panels": len(panels)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
