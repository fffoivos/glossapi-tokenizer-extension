#!/usr/bin/env python3
"""Materialize the pinned Mini tokenizer and build the Greek overlay on CSCS.

The model repository is public, but every downloaded file is bound to an exact
Hugging Face revision and the tokenizer payload is checked against the hash
recorded by the experiment authority.  Completed outputs are validated and
resumed; incomplete output directories are never silently reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
EXPECTED_MINI_SHA256 = (
    "be12f4375d655cc740864e3a9041bcddd8477942f209d9e7f27f6c8767162638"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = Path(str(path) + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_completed(output_dir: Path) -> dict[str, Any] | None:
    manifest_path = output_dir / "overlay_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema_version") != "apertus_mini_greek_tokenizer_overlay_v1"
        or manifest.get("status") != "completed"
    ):
        raise ValueError(f"invalid existing overlay manifest: {manifest_path}")
    tokenizer_path = output_dir / "tokenizer.json"
    expected = manifest.get("output", {}).get("tokenizer_json_sha256")
    if not tokenizer_path.is_file() or sha256_file(tokenizer_path) != expected:
        raise ValueError(f"existing overlay payload drift: {tokenizer_path}")
    if int(manifest.get("target_vocab_size", -1)) != 148_992:
        raise ValueError("existing overlay vocabulary size drift")
    if int(manifest.get("alignment", {}).get("remainder", -1)) != 0:
        raise ValueError("existing overlay is not aligned without padding")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="swiss-ai/Apertus-v1.1-0.5B")
    parser.add_argument(
        "--revision", default="1b7276176e564fc0cc7d7c3b991a8d653c8b8792"
    )
    parser.add_argument("--mini-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--production-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--overlay-builder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    completed = validate_completed(args.output_dir)
    if completed is not None:
        print(json.dumps({"ok": True, "resumed": True, "output": str(args.output_dir)}))
        return 0
    if args.output_dir.exists():
        raise SystemExit(f"refusing incomplete existing overlay directory: {args.output_dir}")

    from huggingface_hub import hf_hub_download

    source_dir = args.mini_tokenizer_dir
    if source_dir.exists():
        if not all((source_dir / name).is_file() for name in FILES):
            raise SystemExit(f"refusing incomplete Mini tokenizer directory: {source_dir}")
    else:
        temporary = Path(str(source_dir) + ".partial")
        if temporary.exists():
            raise SystemExit(f"refusing stale partial directory: {temporary}")
        temporary.mkdir(parents=True)
        receipts: dict[str, dict[str, Any]] = {}
        for name in FILES:
            cached = Path(
                hf_hub_download(
                    repo_id=args.repo_id,
                    revision=args.revision,
                    filename=name,
                    cache_dir=str(args.cache_dir) if args.cache_dir else None,
                )
            )
            destination = temporary / name
            shutil.copyfile(cached, destination)
            receipts[name] = {
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        if receipts["tokenizer.json"]["sha256"] != EXPECTED_MINI_SHA256:
            raise ValueError("downloaded Mini tokenizer hash drift")
        write_json_atomic(
            temporary / "source_manifest.json",
            {
                "schema_version": "apertus_mini_tokenizer_source_v1",
                "status": "completed",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "repo_id": args.repo_id,
                "revision": args.revision,
                "files": receipts,
            },
        )
        os.replace(temporary, source_dir)

    if sha256_file(source_dir / "tokenizer.json") != EXPECTED_MINI_SHA256:
        raise ValueError("staged Mini tokenizer hash drift")
    if not args.overlay_builder.is_file():
        raise FileNotFoundError(args.overlay_builder)
    subprocess.run(
        [
            sys.executable,
            str(args.overlay_builder),
            "--mini-tokenizer-dir",
            str(source_dir),
            "--production-tokenizer-dir",
            str(args.production_tokenizer_dir),
            "--output-dir",
            str(args.output_dir),
        ],
        check=True,
    )
    manifest = validate_completed(args.output_dir)
    if manifest is None:
        raise RuntimeError("overlay builder returned without a completed manifest")
    print(json.dumps({"ok": True, "resumed": False, "output": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
