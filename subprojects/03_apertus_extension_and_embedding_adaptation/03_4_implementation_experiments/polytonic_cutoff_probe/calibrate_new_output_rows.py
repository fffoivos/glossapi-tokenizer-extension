#!/usr/bin/env python3
"""Calibrate appended LM-head rows on balanced old/new-token contexts.

The token-distillation adapter trains only on positive ancient-token snippets.
For an already-trained model, that can make new output rows overconfident on
ordinary Modern Greek and inflate the expanded softmax denominator.  This
bounded pass freezes the complete model and manually Adam-updates only appended
output rows on disjoint, balanced ancient/modern blocks.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path


def load_rows(path: Path) -> dict[str, list[str]]:
    registers: dict[str, list[str]] = {
        "ancient_polytonic": [],
        "modern_greek": [],
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        register = row.get("register")
        text = row.get("text")
        if register in registers and isinstance(text, str) and text:
            registers[register].append(text)
    if not all(registers.values()):
        raise SystemExit(f"missing calibration register in {path}")
    return registers


def make_blocks(tokenizer, texts: list[str], seq_length: int) -> list[list[int]]:
    eos_id = tokenizer.eos_token_id
    stream: list[int] = []
    for text in texts:
        stream.extend(tokenizer.encode(text, add_special_tokens=False))
        if eos_id is not None:
            stream.append(eos_id)
    width = seq_length + 1
    return [
        stream[offset : offset + width]
        for offset in range(0, len(stream) - width + 1, width)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-vocab-size", type=int, default=148_480)
    parser.add_argument("--new-vocab-size", type=int, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1.0e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if len(tokenizer) != args.new_vocab_size:
        raise SystemExit(
            f"tokenizer vocab {len(tokenizer)} != {args.new_vocab_size}"
        )
    registers = load_rows(args.calibration_jsonl)
    blocks = {
        register: make_blocks(tokenizer, texts, args.seq_length)
        for register, texts in registers.items()
    }
    if not all(blocks.values()):
        raise SystemExit("calibration data did not form blocks for both registers")
    for values in blocks.values():
        rng.shuffle(values)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()
    if bool(getattr(model.config, "tie_word_embeddings", False)):
        raise SystemExit("expected untied embeddings")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    output_weight = model.get_output_embeddings().weight
    if output_weight.shape[0] != args.new_vocab_size:
        raise SystemExit("LM-head vocab mismatch")
    old_output = output_weight[: args.base_vocab_size].detach().cpu().clone()
    old_input = (
        model.get_input_embeddings()
        .weight[: args.base_vocab_size]
        .detach()
        .cpu()
        .clone()
    )
    output_weight.requires_grad_(True)
    width = args.new_vocab_size - args.base_vocab_size
    moment1 = torch.zeros(
        (width, output_weight.shape[1]), device=args.device, dtype=torch.float32
    )
    moment2 = torch.zeros_like(moment1)

    losses: list[dict[str, object]] = []
    register_names = ["ancient_polytonic", "modern_greek"]
    indices = {name: 0 for name in register_names}
    for step in range(1, args.steps + 1):
        register = register_names[(step - 1) % 2]
        values = blocks[register]
        block = values[indices[register] % len(values)]
        indices[register] += 1
        input_ids = torch.tensor(
            [block], dtype=torch.long, device=args.device
        )
        logits = model(input_ids=input_ids).logits
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, args.new_vocab_size).float(),
            input_ids[:, 1:].reshape(-1),
        )
        loss.backward()
        gradient = output_weight.grad[
            args.base_vocab_size : args.new_vocab_size
        ].float()
        gradient_norm = torch.linalg.vector_norm(gradient)
        if float(gradient_norm) > args.max_grad_norm:
            gradient.mul_(args.max_grad_norm / gradient_norm)
        moment1.mul_(args.beta1).add_(gradient, alpha=1.0 - args.beta1)
        moment2.mul_(args.beta2).addcmul_(
            gradient, gradient, value=1.0 - args.beta2
        )
        correction1 = 1.0 - args.beta1**step
        correction2 = 1.0 - args.beta2**step
        update = (moment1 / correction1) / (
            torch.sqrt(moment2 / correction2) + args.adam_eps
        )
        with torch.no_grad():
            new_rows = output_weight[
                args.base_vocab_size : args.new_vocab_size
            ]
            new_rows.add_(update.to(new_rows.dtype), alpha=-args.learning_rate)
        output_weight.grad = None
        if step <= 10 or step % 25 == 0 or step == args.steps:
            record = {
                "step": step,
                "register": register,
                "loss": float(loss),
                "new_row_gradient_norm": float(gradient_norm),
            }
            losses.append(record)
            print(json.dumps(record), flush=True)

    if not torch.equal(
        output_weight[: args.base_vocab_size].detach().cpu(), old_output
    ):
        raise SystemExit("calibration changed an existing output row")
    if not torch.equal(
        model.get_input_embeddings()
        .weight[: args.base_vocab_size]
        .detach()
        .cpu(),
        old_input,
    ):
        raise SystemExit("calibration changed an existing input row")
    new_norms = torch.linalg.vector_norm(
        output_weight[
            args.base_vocab_size : args.new_vocab_size
        ].detach().float(),
        dim=1,
    )
    if not torch.isfinite(new_norms).all():
        raise SystemExit("calibrated rows contain non-finite values")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        source = args.model / name
        if source.is_file():
            shutil.copy2(source, args.output_dir / name)
    report = {
        "schema_version": "polytonic-output-row-calibration-v1",
        "input_model": str(args.model),
        "output_dir": str(args.output_dir),
        "calibration_jsonl": str(args.calibration_jsonl),
        "base_vocab_size": args.base_vocab_size,
        "new_vocab_size": args.new_vocab_size,
        "updated_rows": width,
        "steps": args.steps,
        "seq_length": args.seq_length,
        "register_schedule": "strict alternating ancient_polytonic/modern_greek",
        "learning_rate": args.learning_rate,
        "existing_input_rows_exact": True,
        "existing_output_rows_exact": True,
        "new_output_norm": {
            "min": float(new_norms.min()),
            "median": float(new_norms.median()),
            "max": float(new_norms.max()),
        },
        "loss_records": losses,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "output_calibration_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
