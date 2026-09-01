#!/usr/bin/env python3
"""Verify an Apertus HF -> Megatron -> HF roundtrip.

This checker is intentionally independent of Megatron. It compares two HF
safetensors checkpoints key by key and can also run a small prompt-logit check.
Use it after patch_apertus_extras.py and saver_swissai_hf have produced a
round-tripped HF directory.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from safetensors import safe_open


R17_PATTERN = re.compile(r"(act_fn\.(alpha_p|alpha_n|beta|eps)|self_attn\.(q_norm|k_norm)\.weight)")
QK_PATTERN = re.compile(r"self_attn\.(q_norm|k_norm)\.weight")
XIELU_PATTERN = re.compile(r"act_fn\.(alpha_p|alpha_n|beta|eps)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-hf-dir", required=True, type=Path)
    parser.add_argument("--roundtrip-hf-dir", required=True, type=Path)
    parser.add_argument("--tokenizer-dir", type=Path, help="Tokenizer dir for --logits. Defaults to --reference-hf-dir.")
    parser.add_argument("--standard-tol", type=float, default=1.0e-3)
    parser.add_argument("--r17-tol", type=float, default=1.0e-3)
    parser.add_argument("--require-r17-match", action="store_true")
    parser.add_argument("--logits", action="store_true", help="Also compare logits on a few fixed prompts.")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def read_weight_map(hf_dir: Path) -> Dict[str, str]:
    index = hf_dir / "model.safetensors.index.json"
    if index.exists():
        return json.loads(index.read_text())["weight_map"]
    single = hf_dir / "model.safetensors"
    if single.exists():
        with safe_open(single, framework="pt", device="cpu") as handle:
            return {key: single.name for key in handle.keys()}
    raise FileNotFoundError(f"found neither model.safetensors.index.json nor model.safetensors in {hf_dir}")


def load_tensor(hf_dir: Path, weight_map: Dict[str, str], key: str) -> torch.Tensor:
    shard = hf_dir / weight_map[key]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).detach().cpu()


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.numel() == 0:
        return 0.0
    return float((a.float() - b.float()).abs().max().item())


def compare_tensors(reference_hf_dir: Path, roundtrip_hf_dir: Path, standard_tol: float, r17_tol: float, require_r17_match: bool) -> Dict[str, object]:
    ref_map = read_weight_map(reference_hf_dir)
    trip_map = read_weight_map(roundtrip_hf_dir)
    ref_keys = set(ref_map)
    trip_keys = set(trip_map)
    orig_only = sorted(ref_keys - trip_keys)
    trip_only = sorted(trip_keys - ref_keys)

    summary: Dict[str, object] = {
        "orig_only": orig_only,
        "trip_only": trip_only,
        "standard_max_abs_diff": 0.0,
        "r17_max_abs_diff": 0.0,
        "xielu_max_abs_diff": 0.0,
        "qk_norm_max_abs_diff": 0.0,
        "standard_changed_over_tol": [],
        "r17_changed_over_tol": [],
        "shape_mismatches": [],
    }

    standard_changed: List[Tuple[str, float]] = []
    r17_changed: List[Tuple[str, float]] = []
    shape_mismatches: List[str] = []

    for key in sorted(ref_keys & trip_keys):
        ref = load_tensor(reference_hf_dir, ref_map, key)
        trip = load_tensor(roundtrip_hf_dir, trip_map, key)
        if ref.shape != trip.shape:
            shape_mismatches.append(f"{key}: {tuple(ref.shape)} vs {tuple(trip.shape)}")
            continue
        diff = max_abs_diff(ref, trip)
        if R17_PATTERN.search(key):
            summary["r17_max_abs_diff"] = max(float(summary["r17_max_abs_diff"]), diff)
            if XIELU_PATTERN.search(key):
                summary["xielu_max_abs_diff"] = max(float(summary["xielu_max_abs_diff"]), diff)
            if QK_PATTERN.search(key):
                summary["qk_norm_max_abs_diff"] = max(float(summary["qk_norm_max_abs_diff"]), diff)
            if diff > r17_tol:
                r17_changed.append((key, diff))
        else:
            summary["standard_max_abs_diff"] = max(float(summary["standard_max_abs_diff"]), diff)
            if diff > standard_tol:
                standard_changed.append((key, diff))

    summary["standard_changed_over_tol"] = standard_changed[:50]
    summary["r17_changed_over_tol"] = r17_changed[:50]
    summary["standard_changed_over_tol_count"] = len(standard_changed)
    summary["r17_changed_over_tol_count"] = len(r17_changed)
    summary["shape_mismatches"] = shape_mismatches

    if orig_only or trip_only:
        raise AssertionError(f"state_dict keys differ: orig_only={orig_only[:5]} trip_only={trip_only[:5]}")
    if shape_mismatches:
        raise AssertionError(f"shape mismatches: {shape_mismatches[:5]}")
    if standard_changed:
        raise AssertionError(f"standard tensors changed over tol: {standard_changed[:5]}")
    if require_r17_match and r17_changed:
        raise AssertionError(f"R17 tensors changed over tol: {r17_changed[:5]}")
    return summary


def default_prompts() -> List[str]:
    return [
        "Η Αθήνα είναι η πρωτεύουσα της",
        "Ο Όμηρος έγραψε",
        "The capital of Greece is",
    ]


def logits_for_dir(model_dir: Path, tokenizer_dir: Path, prompts: Iterable[str]) -> List[torch.Tensor]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.to(device)
    model.eval()

    outputs: List[torch.Tensor] = []
    with torch.no_grad():
        for prompt in prompts:
            batch = tokenizer(prompt, return_tensors="pt").to(device)
            logits = model(**batch).logits[:, -1, :].float().cpu()
            outputs.append(logits)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs


def compare_logits(reference_hf_dir: Path, roundtrip_hf_dir: Path, tokenizer_dir: Path) -> Dict[str, object]:
    prompts = default_prompts()
    ref_logits = logits_for_dir(reference_hf_dir, tokenizer_dir, prompts)
    trip_logits = logits_for_dir(roundtrip_hf_dir, tokenizer_dir, prompts)

    per_prompt = []
    max_abs = 0.0
    mean_abs_max = 0.0
    for prompt, ref, trip in zip(prompts, ref_logits, trip_logits):
        if ref.shape != trip.shape:
            raise AssertionError(f"logit shape mismatch for prompt {prompt!r}: {tuple(ref.shape)} vs {tuple(trip.shape)}")
        diff = (ref - trip).abs()
        prompt_max = float(diff.max().item())
        prompt_mean = float(diff.mean().item())
        ref_top = int(ref.argmax(dim=-1).item())
        trip_top = int(trip.argmax(dim=-1).item())
        max_abs = max(max_abs, prompt_max)
        mean_abs_max = max(mean_abs_max, prompt_mean)
        per_prompt.append(
            {
                "prompt": prompt,
                "max_abs": prompt_max,
                "mean_abs": prompt_mean,
                "reference_top_id": ref_top,
                "roundtrip_top_id": trip_top,
                "top_id_match": ref_top == trip_top,
            }
        )

    return {
        "logit_max_abs_diff": max_abs,
        "logit_mean_abs_diff_max": mean_abs_max,
        "per_prompt": per_prompt,
    }


def main() -> int:
    args = parse_args()
    tokenizer_dir = args.tokenizer_dir or args.reference_hf_dir

    summary = compare_tensors(
        reference_hf_dir=args.reference_hf_dir,
        roundtrip_hf_dir=args.roundtrip_hf_dir,
        standard_tol=args.standard_tol,
        r17_tol=args.r17_tol,
        require_r17_match=args.require_r17_match,
    )
    if args.logits:
        summary["logits"] = compare_logits(args.reference_hf_dir, args.roundtrip_hf_dir, tokenizer_dir)

    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + os.linesep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
