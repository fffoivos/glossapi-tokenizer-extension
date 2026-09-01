#!/usr/bin/env python3
"""Bridge the frozen Mini validation manifest to raw per-document inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_file(row: dict) -> Path:
    path = Path(row["path"])
    if not path.is_file() or path.stat().st_size != int(row["bytes"]):
        raise ValueError(f"missing or size-drifted input: {path}")
    if sha256_file(path) != row["sha256"]:
        raise ValueError(f"hash-drifted input: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mini-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    mini = json.loads(args.mini_manifest.read_text(encoding="utf-8"))
    if mini.get("schema_version") != "apertus_mini_validation_manifest_v1":
        raise ValueError("unsupported Mini validation manifest")
    panels = []
    for panel in mini["panels"]:
        source_manifest = Path(panel["manifest"])
        if sha256_file(source_manifest) != panel["manifest_sha256"]:
            raise ValueError(f"source manifest drift: {source_manifest}")
        source = json.loads(source_manifest.read_text(encoding="utf-8"))
        raw = source["input"]
        checked_file(raw)
        neutral = panel["name"] == "neutral_external_modern_greek"
        panels.append(
            {
                "name": panel["name"],
                "documents": int(panel["documents"]),
                "tokens": int(panel["tokens"]),
                "megatron_prefix": panel["megatron_prefix"],
                "source_manifest": {
                    "path": str(source_manifest.resolve()),
                    "sha256": panel["manifest_sha256"],
                },
                "raw_jsonl": raw,
                "cluster_field": "cluster_id" if neutral else None,
                "bootstrap_unit": "cluster_id" if neutral else "doc_id",
            }
        )
    panels.sort(key=lambda row: row["name"])
    if len(panels) != 13 or len({row["name"] for row in panels}) != 13:
        raise ValueError("expected exactly 13 distinct validation panels")
    payload = {
        "schema_version": "apertus_per_document_validation_manifest_v1",
        "status": "frozen",
        "source_mini_manifest": {
            "path": str(args.mini_manifest.resolve()),
            "sha256": sha256_file(args.mini_manifest),
        },
        "metric_contract": "document_local_bos_context_no_cross_document_context_text_targets_only",
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
