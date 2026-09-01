#!/usr/bin/env python3
"""Build a tied-embedding ReTok initialization for Apertus-v1.1-0.5B.

The target tokenizer must already be the Mini-compatible overlay produced by
build_mini_tokenizer_overlay.py.  Existing rows remain bitwise unchanged.  Each
new shared input/output row is initialized from the mean of its decomposition
under the pinned Mini tokenizer, matching the canonical Token Distillation
pre-initialization strategy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from tokenizer_geometry import derive_added_token_base_ids


BASE_VOCAB_SIZE = 131_072
TARGET_VOCAB_SIZE = 148_992
PAD_TOKEN_ID = 10
UPSTREAM_COMMIT = "35702b5809599ecd68b7845eca27a0d7b7cec0da"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_by_id(tokenizer) -> dict[int, str]:
    values = {int(token_id): str(token) for token, token_id in tokenizer.get_vocab().items()}
    if len(values) != len(tokenizer.get_vocab()):
        raise ValueError("tokenizer contains duplicate IDs")
    return values


def copy_exact_tokenizer_files(source: Path, destination: Path) -> None:
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "overlay_manifest.json",
    ):
        path = source / name
        if path.is_file():
            shutil.copy2(path, destination / name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--base-tokenizer", type=Path, required=True)
    parser.add_argument("--target-tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-vocab-size", type=int, default=BASE_VOCAB_SIZE)
    parser.add_argument("--target-vocab-size", type=int, default=TARGET_VOCAB_SIZE)
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float32"), default="bfloat16"
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output_dir}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    base_tokenizer = AutoTokenizer.from_pretrained(
        args.base_tokenizer, local_files_only=True
    )
    target_tokenizer = AutoTokenizer.from_pretrained(
        args.target_tokenizer, local_files_only=True
    )
    base_ids = tokenizer_by_id(base_tokenizer)
    target_ids = tokenizer_by_id(target_tokenizer)
    if sorted(base_ids) != list(range(args.base_vocab_size)):
        raise SystemExit("base tokenizer ID range drift")
    if sorted(target_ids) != list(range(args.target_vocab_size)):
        raise SystemExit("target tokenizer ID range drift")
    if any(target_ids[token_id] != token for token_id, token in base_ids.items()):
        raise SystemExit("target tokenizer does not preserve every Mini base ID")
    if target_tokenizer.pad_token != "<pad>" or target_tokenizer.pad_token_id != PAD_TOKEN_ID:
        raise SystemExit("target tokenizer must reconcile Mini padding to <pad> at ID 10")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    if not bool(getattr(model.config, "tie_word_embeddings", False)):
        raise SystemExit("expected tie_word_embeddings=true")
    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    if input_weight.data_ptr() != output_weight.data_ptr():
        raise SystemExit("config says tied but input/output weights do not share storage")
    if tuple(input_weight.shape) != (args.base_vocab_size, 1024):
        raise SystemExit(f"unexpected base embedding shape: {tuple(input_weight.shape)}")
    source_model_pad_token_id = model.config.pad_token_id
    if source_model_pad_token_id != 3 or base_ids.get(3) != "[INST]":
        raise SystemExit("pinned Mini pad-metadata mismatch changed; re-audit required")
    model.config.pad_token_id = PAD_TOKEN_ID
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = PAD_TOKEN_ID

    original_rows = input_weight.detach().cpu().clone()
    phrase_ids = derive_added_token_base_ids(
        args.target_tokenizer / "tokenizer.json",
        base_vocab_size=args.base_vocab_size,
        target_vocab_size=args.target_vocab_size,
    )
    new_rows: list[torch.Tensor] = []
    with torch.no_grad():
        source_weight = input_weight.detach()
        for token_id in range(args.base_vocab_size, args.target_vocab_size):
            ids = torch.tensor(phrase_ids[token_id], device=source_weight.device)
            new_rows.append(source_weight[ids].mean(dim=0).cpu())

    try:
        model.resize_token_embeddings(args.target_vocab_size, mean_resizing=False)
    except TypeError:
        model.resize_token_embeddings(args.target_vocab_size)
    model.tie_weights()
    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    if input_weight.data_ptr() != output_weight.data_ptr():
        raise SystemExit("resize broke tied input/output storage")
    with torch.no_grad():
        input_weight[args.base_vocab_size : args.target_vocab_size].copy_(
            torch.stack(new_rows).to(input_weight.device, dtype=input_weight.dtype)
        )
    if not torch.equal(input_weight[: args.base_vocab_size].cpu(), original_rows):
        raise SystemExit("base embedding rows changed during resize")
    if not torch.isfinite(input_weight).all():
        raise SystemExit("non-finite tied embeddings after ReTok initialization")

    args.output_dir.mkdir(parents=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    target_tokenizer.save_pretrained(args.output_dir)
    copy_exact_tokenizer_files(args.target_tokenizer, args.output_dir)
    norms = input_weight.detach().float().norm(dim=1).cpu()
    manifest = {
        "schema_version": "apertus_mini_tied_retok_init_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "base_model": str(args.base_model.resolve()),
        "base_tokenizer": str(args.base_tokenizer.resolve()),
        "target_tokenizer": str(args.target_tokenizer.resolve()),
        "target_tokenizer_sha256": sha256_file(args.target_tokenizer / "tokenizer.json"),
        "output_dir": str(args.output_dir.resolve()),
        "base_vocab_size": args.base_vocab_size,
        "target_vocab_size": args.target_vocab_size,
        "new_rows": args.target_vocab_size - args.base_vocab_size,
        "hidden_size": input_weight.shape[1],
        "tie_word_embeddings": True,
        "input_output_share_storage": True,
        "pad_metadata_reconciliation": {
            "source_model_config_pad_token_id": source_model_pad_token_id,
            "source_token_at_id_3": base_ids[3],
            "output_model_and_tokenizer_pad_token_id": PAD_TOKEN_ID,
            "pad_token": target_tokenizer.pad_token,
        },
        "base_rows_bitwise_preserved": True,
        "pre_init": "canonical_fvt_subtoken_mean",
        "canonical_token_distillation_commit": UPSTREAM_COMMIT,
        "base_decomposition_policy": "exact_dependency_ordered_appended_merge_dag_leaves",
        "base_phrase_ids": {str(key): value for key, value in phrase_ids.items()},
        "norms": {
            "base_min": norms[: args.base_vocab_size].min().item(),
            "base_max": norms[: args.base_vocab_size].max().item(),
            "new_min": norms[args.base_vocab_size :].min().item(),
            "new_max": norms[args.base_vocab_size :].max().item(),
        },
    }
    (args.output_dir / "tied_retok_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
