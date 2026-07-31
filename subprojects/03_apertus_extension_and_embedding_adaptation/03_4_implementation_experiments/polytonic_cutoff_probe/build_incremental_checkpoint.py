#!/usr/bin/env python3
"""Append a polytonic BPE cutoff to an existing ModernGreek-148480 model.

Only rows at and above the existing model vocabulary are initialized.  Each
new row is the norm-matched mean of the two rows which produced its BPE merge.
This merge-chain initialization also handles structural ByteLevel pieces whose
individual token bytes are not standalone UTF-8.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_merge(merge: Any) -> tuple[str, str] | None:
    if isinstance(merge, list) and len(merge) == 2:
        return str(merge[0]), str(merge[1])
    if isinstance(merge, str):
        parts = merge.split(" ", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
    return None


def norm_match(row, target: float):
    import torch

    norm = torch.linalg.vector_norm(row.float())
    if float(norm) < 1.0e-12:
        raise ValueError("cannot norm-match a zero row")
    return row * (target / norm.to(row.dtype))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--candidate-tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-base-vocab", type=int, default=148_480)
    parser.add_argument("--modern-id-start", type=int, default=131_072)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    base_tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True
    )
    candidate_tokenizer = AutoTokenizer.from_pretrained(
        args.candidate_tokenizer, trust_remote_code=True
    )
    if len(base_tokenizer) != args.expected_base_vocab:
        raise SystemExit(
            f"base tokenizer vocab {len(base_tokenizer)} != "
            f"{args.expected_base_vocab}"
        )
    if len(candidate_tokenizer) <= args.expected_base_vocab:
        raise SystemExit("candidate tokenizer does not append any tokens")

    base_json = json.loads(
        (args.base_model / "tokenizer.json").read_text(encoding="utf-8")
    )
    candidate_json = json.loads(
        (args.candidate_tokenizer / "tokenizer.json").read_text(encoding="utf-8")
    )
    base_vocab = base_json["model"]["vocab"]
    candidate_vocab = candidate_json["model"]["vocab"]
    base_merges = base_json["model"]["merges"]
    candidate_merges = candidate_json["model"]["merges"]
    if candidate_merges[: len(base_merges)] != base_merges:
        raise SystemExit("candidate does not preserve the base merge prefix")
    for token, token_id in base_vocab.items():
        if candidate_vocab.get(token) != token_id:
            raise SystemExit(f"base token id changed: {token!r}")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    if input_weight.shape[0] != args.expected_base_vocab:
        raise SystemExit(
            f"base model vocab {input_weight.shape[0]} != "
            f"{args.expected_base_vocab}"
        )
    if bool(getattr(model.config, "tie_word_embeddings", False)):
        raise SystemExit("expected untied input/output embeddings")

    old_input = input_weight.detach().cpu().clone()
    old_output = output_weight.detach().cpu().clone()
    modern_slice = slice(args.modern_id_start, args.expected_base_vocab)
    target_input_norm = float(
        torch.linalg.vector_norm(old_input[modern_slice].float(), dim=1).median()
    )
    target_output_norm = float(
        torch.linalg.vector_norm(old_output[modern_slice].float(), dim=1).median()
    )

    new_vocab_size = len(candidate_tokenizer)
    try:
        model.resize_token_embeddings(new_vocab_size, mean_resizing=False)
    except TypeError:
        model.resize_token_embeddings(new_vocab_size)
    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    token_by_id = {token_id: token for token, token_id in candidate_vocab.items()}

    added_merges = candidate_merges[len(base_merges) :]
    if len(added_merges) != new_vocab_size - args.expected_base_vocab:
        raise SystemExit("added merge count does not match appended vocab size")
    merge_rows: list[dict[str, object]] = []
    with torch.no_grad():
        for offset, merge in enumerate(added_merges):
            token_id = args.expected_base_vocab + offset
            parts = split_merge(merge)
            if parts is None:
                raise SystemExit(f"invalid merge at id {token_id}: {merge!r}")
            left, right = parts
            left_id = candidate_vocab.get(left)
            right_id = candidate_vocab.get(right)
            result = left + right
            if (
                left_id is None
                or right_id is None
                or left_id >= token_id
                or right_id >= token_id
                or token_by_id.get(token_id) != result
            ):
                raise SystemExit(
                    f"invalid merge dependency at id {token_id}: "
                    f"{left_id!r}, {right_id!r}, result={result!r}"
                )
            input_row = (input_weight[left_id] + input_weight[right_id]) / 2
            output_row = (output_weight[left_id] + output_weight[right_id]) / 2
            input_weight[token_id] = norm_match(input_row, target_input_norm)
            output_weight[token_id] = norm_match(output_row, target_output_norm)
            if offset < 8:
                merge_rows.append(
                    {
                        "token_id": token_id,
                        "left_id": left_id,
                        "right_id": right_id,
                    }
                )

    if not torch.equal(
        model.get_input_embeddings().weight[: args.expected_base_vocab].cpu(),
        old_input,
    ):
        raise SystemExit("resize/init changed an existing input-embedding row")
    if not torch.equal(
        model.get_output_embeddings().weight[: args.expected_base_vocab].cpu(),
        old_output,
    ):
        raise SystemExit("resize/init changed an existing output row")

    probe = "Ἐν ἀρχῇ ἦν ὁ Λόγος· οὓς εἷς Ἦχος."
    probe_ids = candidate_tokenizer.encode(probe, add_special_tokens=False)
    if candidate_tokenizer.decode(
        probe_ids, skip_special_tokens=False
    ) != probe:
        raise SystemExit("candidate tokenizer failed exact Greek probe roundtrip")
    model.to(args.device)
    with torch.no_grad():
        outputs = model(
            torch.tensor([probe_ids], dtype=torch.long, device=args.device)
        )
    if not torch.isfinite(outputs.logits).all():
        raise SystemExit("probe forward produced non-finite logits")
    model.to("cpu")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    candidate_tokenizer.save_pretrained(args.output_dir)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        source = args.candidate_tokenizer / name
        if source.is_file():
            shutil.copy2(source, args.output_dir / name)

    summary = {
        "schema_version": "incremental-polytonic-init-v1",
        "base_model": str(args.base_model),
        "base_tokenizer_sha256": sha256_path(args.base_model / "tokenizer.json"),
        "candidate_tokenizer": str(args.candidate_tokenizer),
        "candidate_tokenizer_sha256": sha256_path(
            args.candidate_tokenizer / "tokenizer.json"
        ),
        "base_vocab_size": args.expected_base_vocab,
        "new_vocab_size": new_vocab_size,
        "new_rows": new_vocab_size - args.expected_base_vocab,
        "initialization": "merge-chain mean with modern-row median norm match",
        "target_input_norm": target_input_norm,
        "target_output_norm": target_output_norm,
        "existing_input_rows_exact": True,
        "existing_output_rows_exact": True,
        "probe_roundtrip_exact": True,
        "probe_logits_finite": True,
        "first_merge_rows": merge_rows,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "incremental_init_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
