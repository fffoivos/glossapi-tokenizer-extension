#!/usr/bin/env python3
"""Measure why the frozen 1.5B Token-Distillation row-norm gate failed."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

import torch
from contract_utils import (
    executing_code_bundle,
    file_binding,
    require,
    write_json_atomic,
)
from safetensors import safe_open

MATRICES = ("model.embed_tokens.weight", "lm_head.weight")
QUANTILES = (0.0, 0.005, 0.01, 0.05, 0.5, 0.95, 0.99, 0.995, 1.0)
BASE_VOCAB = 131_072


def load_matrix(model_root: Path, key: str) -> torch.Tensor:
    shards = sorted(model_root.glob("*.safetensors"))
    require(bool(shards), f"no safetensor shards in {model_root}")
    matches: list[torch.Tensor] = []
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            if key in handle.keys():  # noqa: SIM118 - safe_open has no __contains__
                matches.append(handle.get_tensor(key))
    require(len(matches) == 1, f"expected one tensor for {key} in {model_root}")
    return matches[0]


def summarize(values: torch.Tensor) -> dict[str, Any]:
    values = values.float()
    quantiles = torch.quantile(values, torch.tensor(QUANTILES, dtype=torch.float32))
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "quantiles": {str(q): float(v.item()) for q, v in zip(QUANTILES, quantiles, strict=True)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("SLURM_JOB_PARTITION") == "debug", "diagnostic must run on debug")
    stage = args.stage_root.resolve()
    parent = stage / "assets/init/1p5b_parent_hf"
    reference = stage / "assets/init/1p5b_retok_reference"
    td = stage / "assets/init/1p5b_td_hf_raw"
    manifest_path = td / "retok_td_manifest.json"
    contract_path = stage / "receipts/td_row_norm_contract.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    trained = torch.tensor(manifest["trained_token_ids"], dtype=torch.long) - BASE_VOCAB
    skipped = torch.tensor(sorted(int(value) for value in manifest["skipped_tokens"]), dtype=torch.long) - BASE_VOCAB
    require(bool((trained >= 0).all()) and bool((skipped >= 0).all()), "added-token ledger drift")

    results: dict[str, Any] = {}
    for key in MATRICES:
        parent_weight = load_matrix(parent, key)
        reference_weight = load_matrix(reference, key)
        td_weight = load_matrix(td, key)
        require(tuple(parent_weight.shape) == (BASE_VOCAB, 2048), f"parent shape drift: {key}")
        require(tuple(reference_weight.shape) == tuple(td_weight.shape) == (148_480, 2048), f"extended shape drift: {key}")
        require(torch.equal(parent_weight, reference_weight[:BASE_VOCAB]), f"reference base rows drift: {key}")
        require(torch.equal(parent_weight, td_weight[:BASE_VOCAB]), f"TD base rows drift: {key}")
        base_median = torch.linalg.vector_norm(parent_weight.float(), dim=1).median()
        reference_ratios = torch.linalg.vector_norm(reference_weight[BASE_VOCAB:].float(), dim=1) / base_median
        td_ratios = torch.linalg.vector_norm(td_weight[BASE_VOCAB:].float(), dim=1) / base_median
        band = contract["matrices"][key]
        lower = float(band["accepted_ratio_lower"])
        upper = float(band["accepted_ratio_upper"])
        inside = (td_ratios >= lower) & (td_ratios <= upper)
        ref_inside = (reference_ratios >= lower) & (reference_ratios <= upper)
        results[key] = {
            "frozen_band": {
                "lower": lower,
                "upper": upper,
                "source_8b_median": float(band["added_ratio_median"]),
                "required_fraction_inside": 0.99,
            },
            "base_norm_median": float(base_median.item()),
            "reference_ratio_distribution": summarize(reference_ratios),
            "td_ratio_distribution": summarize(td_ratios),
            "td_minus_reference_ratio_distribution": summarize(td_ratios - reference_ratios),
            "reference_fraction_inside": float(ref_inside.float().mean().item()),
            "td_fraction_inside": float(inside.float().mean().item()),
            "td_fraction_below": float((td_ratios < lower).float().mean().item()),
            "td_fraction_above": float((td_ratios > upper).float().mean().item()),
            "trained_fraction_inside": float(inside[trained].float().mean().item()),
            "skipped_fraction_inside": float(inside[skipped].float().mean().item()),
            "trained_ratio_distribution": summarize(td_ratios[trained]),
            "skipped_ratio_distribution": summarize(td_ratios[skipped]),
            "median_gate_interval": [
                0.8 * float(band["added_ratio_median"]),
                1.2 * float(band["added_ratio_median"]),
            ],
            "median_gate_passed": (
                0.8 * float(band["added_ratio_median"])
                <= float(td_ratios.median().item())
                <= 1.2 * float(band["added_ratio_median"])
            ),
            "fraction_gate_passed": float(inside.float().mean().item()) >= 0.99,
        }
        del parent_weight, reference_weight, td_weight

    payload = {
        "schema_version": "apertus_1p5b_td_row_norm_failure_diagnostic_v1",
        "status": "diagnostic_completed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "executing_code_bundle": executing_code_bundle(),
        "trained_token_count": int(trained.numel()),
        "skipped_token_count": int(skipped.numel()),
        "gate_passed": all(
            row["fraction_gate_passed"] and row["median_gate_passed"]
            for row in results.values()
        ),
        "matrices": results,
        "td_manifest": file_binding(manifest_path),
        "row_norm_contract": file_binding(contract_path),
    }
    write_json_atomic(args.output, payload)
    print(json.dumps({"gate_passed": payload["gate_passed"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
