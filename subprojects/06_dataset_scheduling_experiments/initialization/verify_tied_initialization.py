#!/usr/bin/env python3
"""Verify a Mini tied-TD artifact and reject the known collapse failure mode."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


BASE_VOCAB_SIZE = 131_072
TARGET_VOCAB_SIZE = 148_992
PAD_TOKEN_ID = 10


def iter_probe_texts(path: Path, text_key: str, limit: int):
    emitted = 0
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if emitted >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            text = row.get(text_key) if isinstance(row, dict) else None
            if isinstance(text, str) and text.strip():
                emitted += 1
                yield text
    if emitted == 0:
        raise SystemExit(f"no non-empty {text_key!r} values in {path}")


def norm_summary(values) -> dict[str, float]:
    values = values.detach().float().cpu()
    return {
        "min": values.min().item(),
        "p50": values.quantile(0.50).item(),
        "p95": values.quantile(0.95).item(),
        "p99": values.quantile(0.99).item(),
        "p999": values.quantile(0.999).item(),
        "max": values.max().item(),
    }


def dominant_added(counter: Counter[int]) -> tuple[int | None, int]:
    added = [(count, token_id) for token_id, count in counter.items() if token_id >= BASE_VOCAB_SIZE]
    if not added:
        return None, 0
    count, token_id = max(added)
    return token_id, count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--probe-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--max-probe-docs", type=int, default=256)
    parser.add_argument("--max-probe-tokens", type=int, default=512)
    parser.add_argument("--max-generation-prompts", type=int, default=64)
    parser.add_argument("--generation-prompt-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-new-to-base-p999-ratio", type=float, default=4.0)
    parser.add_argument("--max-dominant-added-top1-fraction", type=float, default=0.05)
    parser.add_argument("--max-dominant-added-generation-fraction", type=float, default=0.10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_json.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output_json}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    if tuple(base.get_input_embeddings().weight.shape) != (BASE_VOCAB_SIZE, 1024):
        raise SystemExit("base embedding geometry drift")
    original_rows = base.get_input_embeddings().weight.detach().cpu().clone()
    del base

    tokenizer = AutoTokenizer.from_pretrained(args.candidate_model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.candidate_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(args.device)
    weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    checks = {
        "tie_word_embeddings": bool(getattr(model.config, "tie_word_embeddings", False)),
        "input_output_share_storage": weight.data_ptr() == output_weight.data_ptr(),
        "embedding_shape_exact": tuple(weight.shape) == (TARGET_VOCAB_SIZE, 1024),
        "base_rows_bitwise_preserved": torch.equal(
            weight[:BASE_VOCAB_SIZE].detach().cpu(), original_rows
        ),
        "all_embeddings_finite": bool(torch.isfinite(weight).all().item()),
        "model_pad_token_id_is_10": model.config.pad_token_id == PAD_TOKEN_ID,
        "tokenizer_pad_token_id_is_10": tokenizer.pad_token_id == PAD_TOKEN_ID,
    }
    if not all(checks.values()):
        raise SystemExit(f"structural tied-initialization gate failed: {checks}")

    norms = weight.detach().float().norm(dim=1)
    base_norms = norm_summary(norms[:BASE_VOCAB_SIZE])
    new_norms = norm_summary(norms[BASE_VOCAB_SIZE:])
    norm_ratio = new_norms["max"] / base_norms["p999"]
    checks["norm_ratio_gate"] = norm_ratio <= args.max_new_to_base_p999_ratio

    texts = list(iter_probe_texts(args.probe_jsonl, args.text_key, args.max_probe_docs))
    top1_counts: Counter[int] = Counter()
    top1_positions = 0
    generation_counts: Counter[int] = Counter()
    generation_tokens = 0
    model.eval()
    with torch.inference_mode():
        for text in texts:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_probe_tokens,
                add_special_tokens=True,
            )
            input_ids = encoded["input_ids"].to(args.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(args.device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            predicted = logits[:, :-1].argmax(dim=-1).reshape(-1).cpu().tolist()
            top1_counts.update(int(value) for value in predicted)
            top1_positions += len(predicted)

        for text in texts[: args.max_generation_prompts]:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.generation_prompt_tokens,
                add_special_tokens=True,
            )
            input_ids = encoded["input_ids"].to(args.device)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=encoded.get("attention_mask", None).to(args.device)
                if encoded.get("attention_mask", None) is not None
                else None,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
            suffix = generated[0, input_ids.shape[1] :].cpu().tolist()
            generation_counts.update(int(value) for value in suffix)
            generation_tokens += len(suffix)

    dominant_top1_id, dominant_top1_count = dominant_added(top1_counts)
    dominant_generation_id, dominant_generation_count = dominant_added(generation_counts)
    dominant_top1_fraction = dominant_top1_count / max(top1_positions, 1)
    dominant_generation_fraction = dominant_generation_count / max(generation_tokens, 1)
    checks["dominant_added_top1_gate"] = (
        dominant_top1_fraction <= args.max_dominant_added_top1_fraction
    )
    checks["dominant_added_generation_gate"] = (
        dominant_generation_fraction <= args.max_dominant_added_generation_fraction
    )

    report = {
        "schema_version": "apertus_mini_tied_initialization_verification_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if all(checks.values()) else "fail",
        "base_model": str(args.base_model.resolve()),
        "candidate_model": str(args.candidate_model.resolve()),
        "probe_jsonl": str(args.probe_jsonl.resolve()),
        "checks": checks,
        "norms": {
            "base": base_norms,
            "new": new_norms,
            "max_new_over_base_p999": norm_ratio,
            "threshold": args.max_new_to_base_p999_ratio,
        },
        "teacher_forced_top1": {
            "positions": top1_positions,
            "dominant_added_token_id": dominant_top1_id,
            "dominant_added_token_string": tokenizer.convert_ids_to_tokens(dominant_top1_id)
            if dominant_top1_id is not None
            else None,
            "dominant_added_fraction_of_all_positions": dominant_top1_fraction,
            "threshold": args.max_dominant_added_top1_fraction,
        },
        "greedy_generation": {
            "tokens": generation_tokens,
            "dominant_added_token_id": dominant_generation_id,
            "dominant_added_token_string": tokenizer.convert_ids_to_tokens(dominant_generation_id)
            if dominant_generation_id is not None
            else None,
            "dominant_added_fraction_of_all_generated_tokens": dominant_generation_fraction,
            "threshold": args.max_dominant_added_generation_fraction,
        },
        "threshold_policy": "project rejection gates; no clipping or post-hoc mutation",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
