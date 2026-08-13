#!/usr/bin/env python3
"""Summarize Megatron ``common.pt`` control state without tensor payloads."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import torch


INTERESTING_ARGUMENTS = (
    "iteration",
    "consumed_train_samples",
    "train_samples",
    "lr",
    "min_lr",
    "lr_decay_style",
    "lr_wsd_decay_style",
    "lr_wsd_decay_samples",
    "ademamix_alpha",
    "ademamix_beta3",
    "ademamix_alpha_warmup",
    "ademamix_beta3_warmup",
    "seed",
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "context_parallel_size",
    "global_batch_size",
    "micro_batch_size",
)


def summarize(value: Any, depth: int = 0) -> Any:
    if depth >= 5:
        return {"type": type(value).__name__}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "length": len(value)}
    if isinstance(value, torch.Tensor):
        return {
            "type": "Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, dict):
        return {str(key): summarize(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "length": len(value),
            "items": [summarize(item, depth + 1) for item in value[:12]],
        }
    if dataclasses.is_dataclass(value):
        return {
            "type": type(value).__name__,
            "fields": summarize(dataclasses.asdict(value), depth + 1),
        }
    if hasattr(value, "__dict__"):
        payload = vars(value)
        selected = {
            name: payload[name]
            for name in INTERESTING_ARGUMENTS
            if name in payload
        }
        return {
            "type": type(value).__name__,
            "attribute_count": len(payload),
            "selected_attributes": summarize(selected, depth + 1),
        }
    return {"type": type(value).__name__, "repr": repr(value)[:240]}


def inspect(root: Path) -> dict[str, Any]:
    path = root / "common.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "root": str(root.resolve()),
        "common_pt_bytes": path.stat().st_size,
        "payload": summarize(payload),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema_version": "apertus_checkpoint_common_inspection_v1",
        "checkpoints": [inspect(root) for root in args.roots],
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
