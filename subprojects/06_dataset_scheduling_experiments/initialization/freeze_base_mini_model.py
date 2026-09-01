#!/usr/bin/env python3
"""Download and freeze the pinned Apertus-v1.1-0.5B base checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_REPO = "swiss-ai/Apertus-v1.1-0.5B"
EXPECTED_REVISION = "1b7276176e564fc0cc7d7c3b991a8d653c8b8792"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_inventory(root: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(root))
        if relative.startswith(".cache/") or relative == "base_model_receipt.json":
            continue
        rows[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default=EXPECTED_REPO)
    parser.add_argument("--revision", default=EXPECTED_REVISION)
    args = parser.parse_args()
    if args.repo_id != EXPECTED_REPO or args.revision != EXPECTED_REVISION:
        raise SystemExit("the Mini base model identity is pinned")
    output = args.output_dir.resolve()
    receipt_path = output / "base_model_receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("status") == "frozen"
            and receipt.get("repo_id") == args.repo_id
            and receipt.get("resolved_revision") == args.revision
            and file_inventory(output) == receipt.get("files")
        ):
            print(json.dumps({"ok": True, "resumed": True, "root": str(output)}))
            return 0
        raise ValueError("existing Mini base-model receipt or files drifted")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing unreceipted model directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    resolved = HfApi().model_info(args.repo_id, revision=args.revision).sha
    if resolved != args.revision:
        raise ValueError(f"Hugging Face revision resolution drift: {resolved}")
    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=output,
    )
    config = AutoConfig.from_pretrained(output, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(output, local_files_only=True)
    expected = {
        "hidden_size": 1024,
        "intermediate_size": 6144,
        "num_hidden_layers": 20,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "hidden_act": "xielu",
        "qk_norm": True,
        "tie_word_embeddings": True,
        "rope_theta": 500000.0,
        "max_position_embeddings": 4096,
        "vocab_size": 131072,
    }
    observed = {key: getattr(config, key, None) for key in expected}
    if observed != expected:
        raise ValueError(f"pinned Mini config geometry drift: {observed}")
    if len(tokenizer) != expected["vocab_size"]:
        raise ValueError("pinned Mini tokenizer vocabulary drift")
    files = file_inventory(output)
    required = {"config.json", "tokenizer.json"}
    if not required.issubset(files) or not any(
        name.endswith(".safetensors") for name in files
    ):
        raise ValueError("downloaded Mini snapshot is incomplete")
    payload = {
        "schema_version": "apertus_v1_1_0p5b_base_model_receipt_v1",
        "status": "frozen",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "repo_id": args.repo_id,
        "requested_revision": args.revision,
        "resolved_revision": resolved,
        "root": str(output),
        "geometry": observed,
        "files": files,
    }
    temporary = Path(str(receipt_path) + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, receipt_path)
    print(json.dumps({"ok": True, "root": str(output), "files": len(files)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

