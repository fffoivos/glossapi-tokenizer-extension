#!/usr/bin/env python3
"""Materialize the uncpt TokenDistil-Init subfolder at the tokenizer pin."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


REPO_ID = "fffoivos/apertus-tokenizer-extension"
REVISION = "fcd33ec09fb7d86bc072b3a4b3e890efa6473b66"
PREFIX = "experiment-checkpoints/TokenDistil-Init/"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    return parser.parse_args()


def _validate(root: Path) -> dict:
    receipt_path = root / "materialization_receipt.json"
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    if value.get("repo_id") != REPO_ID or value.get("revision") != REVISION:
        raise ValueError("TokenDistil-Init receipt identity drift")
    for row in value.get("files", []):
        path = root / row["path"]
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(row["bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            raise ValueError(f"TokenDistil-Init payload drift: {path}")
    return value


def main() -> int:
    args = parse_args()
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required")
    output = args.output_root.resolve()
    if output.exists():
        value = _validate(output)
        print(json.dumps({"ok": True, "resumed": True, "files": len(value["files"])}, sort_keys=True))
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=os.environ["HF_TOKEN"])
    resolved = api.model_info(REPO_ID, revision=REVISION, token=os.environ["HF_TOKEN"]).sha
    if resolved != REVISION:
        raise ValueError("tokenizer repository revision drift")
    selected = sorted(
        path
        for path in api.list_repo_files(REPO_ID, revision=REVISION, token=os.environ["HF_TOKEN"])
        if path.startswith(PREFIX) and path != PREFIX
    )
    required = {"config.json", "model.safetensors.index.json", "tokenizer.json", "manifest.json"}
    relative_names = {path.removeprefix(PREFIX) for path in selected}
    if not required <= relative_names or len([name for name in relative_names if name.endswith(".safetensors")]) != 4:
        raise ValueError("TokenDistil-Init inventory drift")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    receipts = []
    try:
        for remote in selected:
            relative = remote.removeprefix(PREFIX)
            cached = Path(
                hf_hub_download(
                    repo_id=REPO_ID,
                    revision=REVISION,
                    filename=remote,
                    cache_dir=args.cache_dir,
                    token=os.environ["HF_TOKEN"],
                )
            )
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, destination)
            receipts.append(
                {
                    "path": relative,
                    "bytes": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                }
            )
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(temporary / "tokenizer.json"))
        if tokenizer.get_vocab_size(with_added_tokens=True) != 148480:
            raise ValueError("TokenDistil-Init vocabulary drift")
        manifest = json.loads((temporary / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest.get("human_name") != "TokenDistil-Init"
            or manifest.get("training_point") != "Init"
            or int(manifest.get("target_layer", -1)) != 11
        ):
            raise ValueError("refusing a CPT-trained or wrong-layer initialization")
        receipt = {
            "schema_version": "token_distil_init_materialization_v1",
            "status": "completed",
            "repo_id": REPO_ID,
            "revision": REVISION,
            "subfolder": PREFIX.rstrip("/"),
            "vocab_size": 148480,
            "source_manifest": manifest,
            "files": receipts,
        }
        (temporary / "materialization_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({"ok": True, "output": str(output), "files": len(receipts)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
