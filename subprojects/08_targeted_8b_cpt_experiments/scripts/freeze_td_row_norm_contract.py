#!/usr/bin/env python3
"""Freeze 8B-derived appended-row norm bands before the 1.5B TD run."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from contract_utils import executing_code_bundle, file_binding, require, write_json_atomic


EMBED_KEYS = ("model.embed_tokens.weight", "lm_head.weight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verified-8b-td-root", type=Path, required=True)
    parser.add_argument("--historical-td-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lower-percentile", type=float, default=0.5)
    parser.add_argument("--upper-percentile", type=float, default=99.5)
    parser.add_argument("--padding-fraction", type=float, default=0.2)
    return parser.parse_args()


def weight_map(root: Path) -> dict[str, str]:
    index = root / "model.safetensors.index.json"
    require(index.is_file(), "8B TD safetensor index missing")
    value = json.loads(index.read_text(encoding="utf-8"))
    return value["weight_map"]


def load_tensor(root: Path, mapping: dict[str, str], key: str):
    from safetensors import safe_open

    require(key in mapping, f"embedding key missing: {key}")
    with safe_open(root / mapping[key], framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def main() -> int:
    args = parse_args()
    require(not args.output.exists(), f"immutable norm contract exists: {args.output}")
    require(0 <= args.lower_percentile < args.upper_percentile <= 100, "invalid percentile interval")
    require(args.padding_fraction >= 0, "negative padding fraction")
    manifest = json.loads(args.historical_td_manifest.read_text(encoding="utf-8"))
    require(manifest.get("target_layer") == 11, "8B TD target layer drift")
    require(manifest.get("trained_token_count") == 17_377, "8B TD trained-token count drift")
    mapping = weight_map(args.verified_8b_td_root)
    shard_paths = sorted({args.verified_8b_td_root / relative for relative in mapping.values()})
    require(bool(shard_paths) and all(path.is_file() for path in shard_paths), "8B TD safetensor shard inventory incomplete")
    import torch

    matrices = {}
    for key in EMBED_KEYS:
        tensor = load_tensor(args.verified_8b_td_root, mapping, key).float()
        require(tuple(tensor.shape) == (148_480, 4096), f"8B embedding geometry drift: {key}")
        base_norms = torch.linalg.vector_norm(tensor[:131_072], dim=1)
        added_norms = torch.linalg.vector_norm(tensor[131_072:], dim=1)
        require(torch.isfinite(base_norms).all().item() and torch.isfinite(added_norms).all().item(), "non-finite 8B row norms")
        base_median = float(base_norms.median().item())
        ratios = added_norms / base_median
        quantiles = torch.quantile(
            ratios,
            torch.tensor([args.lower_percentile / 100.0, args.upper_percentile / 100.0]),
            interpolation="linear",
        )
        lower_q, upper_q = float(quantiles[0].item()), float(quantiles[1].item())
        width = upper_q - lower_q
        lower = max(0.0, lower_q - args.padding_fraction * width)
        upper = upper_q + args.padding_fraction * width
        require(lower < upper, "degenerate 8B norm band")
        matrices[key] = {
            "base_row_count": 131_072,
            "added_row_count": 17_408,
            "base_norm_median": base_median,
            "added_ratio_median": float(ratios.median().item()),
            "lower_percentile": args.lower_percentile,
            "upper_percentile": args.upper_percentile,
            "lower_quantile": lower_q,
            "upper_quantile": upper_q,
            "padding_fraction": args.padding_fraction,
            "padding_definition": "expand each percentile boundary outward by padding_fraction times the unpadded interval width",
            "accepted_ratio_lower": lower,
            "accepted_ratio_upper": upper,
        }
        del tensor, base_norms, added_norms, ratios
    receipt = {
        "schema_version": "apertus_td_row_norm_contract_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "frozen_before_1p5b_td": True,
        "minimum_fraction_inside": 0.99,
        "median_ratio_lower_multiplier": 0.8,
        "median_ratio_upper_multiplier": 1.2,
        "verified_8b_td_config": file_binding(args.verified_8b_td_root / "config.json"),
        "verified_8b_td_index": file_binding(args.verified_8b_td_root / "model.safetensors.index.json"),
        "verified_8b_td_shards": [file_binding(path) for path in shard_paths],
        "historical_td_manifest": file_binding(args.historical_td_manifest),
        "matrices": matrices,
        "executing_code_bundle": executing_code_bundle(),
    }
    write_json_atomic(args.output, receipt)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
