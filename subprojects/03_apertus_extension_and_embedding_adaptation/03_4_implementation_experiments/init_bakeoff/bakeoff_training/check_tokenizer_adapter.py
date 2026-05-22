#!/usr/bin/env python3
"""Check HF tokenizer IDs against Megatron's HuggingFaceTokenizer adapter."""

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-dir", required=True, type=Path)
    parser.add_argument("--label", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hf = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    from megatron.training.tokenizer.tokenizer import _HuggingFaceTokenizer

    meg = _HuggingFaceTokenizer(str(args.tokenizer_dir))

    attrs = (
        "bos_token_id",
        "eos_token_id",
        "unk_token_id",
        "pad_token_id",
        "sep_token_id",
        "cls_token_id",
        "mask_token_id",
    )
    mismatches = {}
    for attr in attrs:
        hf_value = getattr(hf, attr, None)
        meg_value = getattr(meg._tokenizer, attr, None)
        if hf_value != meg_value:
            mismatches[attr] = {"hf": hf_value, "megatron": meg_value}

    samples = [
        "Η Αθήνα είναι η πρωτεύουσα της Ελλάδας.",
        "Ο Όμηρος έγραψε την Ιλιάδα και την Οδύσσεια.",
        "The capital of Greece is Athens.",
    ]
    encode_mismatches = []
    for text in samples:
        hf_ids = hf.encode(text, add_special_tokens=False)
        meg_ids = meg.tokenize(text, add_special_tokens=False)
        if hf_ids != meg_ids:
            encode_mismatches.append({"text": text, "hf": hf_ids[:32], "megatron": meg_ids[:32]})

    summary = {
        "label": args.label,
        "tokenizer_dir": str(args.tokenizer_dir),
        "vocab_size_hf": len(hf),
        "vocab_size_megatron": meg.vocab_size,
        "eos_token_id": hf.eos_token_id,
        "megatron_eod": meg.eod,
        "all_special_ids_hf": list(hf.all_special_ids),
        "all_special_ids_megatron": list(meg._tokenizer.all_special_ids),
        "attr_mismatches": mismatches,
        "encode_mismatches": encode_mismatches,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if len(hf) != meg.vocab_size:
        raise AssertionError(f"vocab size mismatch: HF={len(hf)} Megatron={meg.vocab_size}")
    if hf.eos_token_id != meg.eod:
        raise AssertionError(f"EoD mismatch: HF eos={hf.eos_token_id} Megatron eod={meg.eod}")
    if list(hf.all_special_ids) != list(meg._tokenizer.all_special_ids):
        raise AssertionError("all_special_ids mismatch between HF and Megatron adapter")
    if mismatches:
        raise AssertionError(f"special token attr mismatches: {mismatches}")
    if encode_mismatches:
        raise AssertionError(f"encoding mismatches: {encode_mismatches[:1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
