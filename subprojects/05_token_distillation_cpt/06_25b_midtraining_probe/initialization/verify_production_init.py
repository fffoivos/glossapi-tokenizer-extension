#!/usr/bin/env python3
"""Prove that only the appended 512 rows changed in the production HF init."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--final-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    final_tokenizer = AutoTokenizer.from_pretrained(args.final_model, trust_remote_code=True)
    if len(base_tokenizer) != 148480 or len(final_tokenizer) != 148992:
        raise ValueError("production init tokenizer vocabulary drift")
    if sha256_file(args.final_model / "tokenizer.json") != "bbb08e71929b519c5c2362338b0fc6a0e99955cb8fdbf0729ae1311117e6561b":
        raise ValueError("production init tokenizer JSON drift")

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    final = AutoModelForCausalLM.from_pretrained(
        args.final_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    base_input = base.get_input_embeddings().weight.detach().cpu()
    base_output = base.get_output_embeddings().weight.detach().cpu()
    final_input = final.get_input_embeddings().weight.detach().cpu()
    final_output = final.get_output_embeddings().weight.detach().cpu()
    if base_input.shape[0] != 148480 or final_input.shape[0] != 148992:
        raise ValueError("production init embedding shape drift")
    existing_input_exact = torch.equal(final_input[:148480], base_input)
    existing_output_exact = torch.equal(final_output[:148480], base_output)

    input_name = next(name for name, parameter in base.named_parameters() if parameter is base.get_input_embeddings().weight)
    output_name = next(name for name, parameter in base.named_parameters() if parameter is base.get_output_embeddings().weight)
    base_parameters = dict(base.named_parameters())
    final_parameters = dict(final.named_parameters())
    if set(base_parameters) != set(final_parameters):
        raise ValueError("production init parameter inventory drift")
    changed_non_embedding: list[str] = []
    for name in sorted(base_parameters):
        if name in {input_name, output_name}:
            continue
        if not torch.equal(base_parameters[name].detach().cpu(), final_parameters[name].detach().cpu()):
            changed_non_embedding.append(name)
    base_buffers = dict(base.named_buffers())
    final_buffers = dict(final.named_buffers())
    if set(base_buffers) != set(final_buffers):
        raise ValueError("production init buffer inventory drift")
    changed_buffers = [
        name for name in sorted(base_buffers)
        if not torch.equal(base_buffers[name].detach().cpu(), final_buffers[name].detach().cpu())
    ]
    new_input = final_input[148480:].float()
    new_output = final_output[148480:].float()
    new_rows_finite = bool(torch.isfinite(new_input).all() and torch.isfinite(new_output).all())
    new_rows_nonzero = bool(
        torch.linalg.vector_norm(new_input, dim=1).min() > 0
        and torch.linalg.vector_norm(new_output, dim=1).min() > 0
    )
    reports = {
        "incremental_init": args.final_model / "incremental_init_summary.json",
        "token_distillation": args.final_model / "retok_td_manifest.json",
        "output_calibration": args.final_model / "output_calibration_summary.json",
    }
    # The TD and calibration stages save into distinct directories. The build
    # driver copies their immutable reports into the final directory.
    for name, path in reports.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {name} report: {path}")
    passed = (
        existing_input_exact
        and existing_output_exact
        and not changed_non_embedding
        and not changed_buffers
        and new_rows_finite
        and new_rows_nonzero
    )
    payload = {
        "schema_version": "production_polytonic_td_init_verification_v1",
        "status": "passed" if passed else "failed",
        "base_model": str(args.base_model.resolve()),
        "final_model": str(args.final_model.resolve()),
        "base_vocab_size": 148480,
        "final_vocab_size": 148992,
        "target_layer": 11,
        "existing_input_rows_exact": existing_input_exact,
        "existing_output_rows_exact": existing_output_exact,
        "non_embedding_tensors_exact": not changed_non_embedding and not changed_buffers,
        "changed_non_embedding_parameters": changed_non_embedding,
        "changed_buffers": changed_buffers,
        "new_rows_finite": new_rows_finite,
        "new_rows_nonzero": new_rows_nonzero,
        "tokenizer_json_sha256": sha256_file(args.final_model / "tokenizer.json"),
        "reports": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in reports.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    if not passed:
        raise SystemExit("production initialization preservation verification failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
