#!/usr/bin/env python3
"""Materialize and receipt an exact Hugging Face model revision or subfolder."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from contract_utils import executing_code_bundle, file_binding, require, sha256_file, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--subfolder", default="")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--expected-vocab-size", type=int, required=True)
    parser.add_argument("--expected-hidden-size", type=int, required=True)
    parser.add_argument("--expected-hidden-layers", type=int, required=True)
    parser.add_argument("--expected-attention-heads", type=int, required=True)
    parser.add_argument("--expected-kv-heads", type=int, required=True)
    parser.add_argument("--expected-safetensors", type=int)
    parser.add_argument("--expected-safetensor-bytes", type=int)
    parser.add_argument("--expected-td-target-layer", type=int)
    return parser.parse_args()


def relative_remote(path: str, prefix: str) -> str:
    require(path.startswith(prefix), f"remote path escaped subfolder: {path}")
    relative = path[len(prefix) :]
    require(bool(relative) and not relative.startswith("/") and ".." not in Path(relative).parts, "unsafe remote path")
    return relative


def validate_config(config: dict[str, Any], args: argparse.Namespace) -> None:
    expected = {
        "vocab_size": args.expected_vocab_size,
        "hidden_size": args.expected_hidden_size,
        "num_hidden_layers": args.expected_hidden_layers,
        "num_attention_heads": args.expected_attention_heads,
        "num_key_value_heads": args.expected_kv_heads,
        "tie_word_embeddings": False,
    }
    observed = {key: config.get(key) for key in expected}
    require(observed == expected, f"HF model configuration drift: {observed!r} != {expected!r}")


def main() -> int:
    args = parse_args()
    require(bool(os.environ.get("HF_TOKEN")), "HF_TOKEN is required")
    require(len(args.revision) == 40, "revision must be a full immutable commit")
    require(not args.output_receipt.exists(), f"immutable receipt exists: {args.output_receipt}")
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=os.environ["HF_TOKEN"])
    resolved = api.model_info(args.repo_id, revision=args.revision, token=os.environ["HF_TOKEN"]).sha
    require(resolved == args.revision, "Hugging Face revision did not resolve exactly")
    subfolder = args.subfolder.strip("/")
    prefix = f"{subfolder}/" if subfolder else ""
    selected = sorted(
        path for path in api.list_repo_files(args.repo_id, revision=args.revision, token=os.environ["HF_TOKEN"])
        if path.startswith(prefix) and path != prefix
    )
    names = {relative_remote(path, prefix) for path in selected}
    require("config.json" in names and "tokenizer.json" in names, "model subfolder lacks config/tokenizer")
    safetensors = sorted(name for name in names if name.endswith(".safetensors"))
    require(bool(safetensors), "model subfolder has no safetensors")
    if args.expected_safetensors is not None:
        require(len(safetensors) == args.expected_safetensors, "safetensor file-count drift")

    created_now = False
    if args.output_root.exists():
        require(args.output_root.is_dir(), "materialized output is not a directory")
        local_names = {str(path.relative_to(args.output_root)) for path in args.output_root.rglob("*") if path.is_file()}
        require(local_names == names, "partial or drifted resumed HF materialization")
        rows = [
            {"path": name, "bytes": (args.output_root / name).stat().st_size, "sha256": sha256_file(args.output_root / name)}
            for name in sorted(names)
        ]
        materialized = args.output_root
    else:
        temporary = Path(tempfile.mkdtemp(prefix=f".{args.output_root.name}.", dir=args.output_root.parent))
        rows = []
        try:
            for remote in selected:
                relative = relative_remote(remote, prefix)
                cached = Path(hf_hub_download(
                    repo_id=args.repo_id,
                    revision=args.revision,
                    filename=remote,
                    cache_dir=args.cache_dir,
                    token=os.environ["HF_TOKEN"],
                ))
                destination = temporary / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cached, destination)
                rows.append({"path": relative, "bytes": destination.stat().st_size, "sha256": sha256_file(destination)})
            os.replace(temporary, args.output_root)
            created_now = True
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        materialized = args.output_root
    try:
        config = json.loads((materialized / "config.json").read_text(encoding="utf-8"))
        validate_config(config, args)
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(materialized / "tokenizer.json"))
        require(tokenizer.get_vocab_size(with_added_tokens=True) == args.expected_vocab_size, "tokenizer vocabulary drift")
        tensor_bytes = sum(row["bytes"] for row in rows if str(row["path"]).endswith(".safetensors"))
        if args.expected_safetensor_bytes is not None:
            require(tensor_bytes == args.expected_safetensor_bytes, "safetensor byte-size drift")
        source_manifest: dict[str, Any] | None = None
        if args.expected_td_target_layer is not None:
            manifest_path = materialized / "manifest.json"
            require(manifest_path.is_file(), "TD source manifest missing")
            source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            require(int(source_manifest.get("target_layer", -1)) == args.expected_td_target_layer, "TD target-layer drift")
            require(source_manifest.get("training_point") == "Init", "refusing a CPT-trained TD artifact")
    except BaseException:
        if created_now:
            shutil.rmtree(args.output_root, ignore_errors=True)
        raise

    receipt = {
        "schema_version": "apertus_pinned_hf_model_materialization_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo_id": args.repo_id,
        "revision": args.revision,
        "subfolder": subfolder,
        "output_root": str(args.output_root.resolve()),
        "model_config": {key: config[key] for key in (
            "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "tie_word_embeddings",
        )},
        "source_geometry": {
            "rope_theta": config.get("rope_theta"),
            "max_position_embeddings": config.get("max_position_embeddings"),
        },
        "safetensor_files": len(safetensors),
        "safetensor_bytes": tensor_bytes,
        "source_manifest": source_manifest,
        "files": rows,
        "config": file_binding(args.output_root / "config.json"),
        "tokenizer": file_binding(args.output_root / "tokenizer.json"),
        "executing_code_bundle": executing_code_bundle(),
    }
    write_json_atomic(args.output_receipt, receipt)
    print(args.output_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
