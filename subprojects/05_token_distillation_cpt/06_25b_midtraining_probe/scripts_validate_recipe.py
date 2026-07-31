#!/usr/bin/env python3
"""Fail closed when the frozen 25B two-phase recipe is internally inconsistent."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recipe",
        type=Path,
        nargs="?",
        default=Path(__file__).with_name("configs") / "recipe_25b_midtraining.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    if recipe.get("schema_version") != "greek_cpt_25b_midtraining_recipe_v1":
        raise ValueError("unsupported recipe schema")
    geometry = recipe["geometry"]
    if geometry["effective_iterations"] * geometry["global_batch_tokens"] != geometry["effective_tokens"]:
        raise ValueError("effective token arithmetic drift")
    if geometry["phase_1_iterations"] + geometry["phase_2_iterations"] != geometry["effective_iterations"]:
        raise ValueError("phase iteration arithmetic drift")
    if geometry["phase_1_tokens"] + geometry["phase_2_tokens"] != geometry["effective_tokens"]:
        raise ValueError("phase token arithmetic drift")
    if geometry["phase_boundary_iteration"] != geometry["phase_1_iterations"]:
        raise ValueError("phase boundary drift")
    capacity = recipe["capacity"]
    if (
        capacity["minimum_unique_capacity_ratio"] != "1.005"
        or capacity["physical_prefix_sample_capacity_ratio"] != "1.005"
        or capacity["physical_prefix_boundary_samples"] != 1
    ):
        raise ValueError("unique-capacity policy drift")
    if geometry["segment_boundaries"][0] != 0 or geometry["segment_boundaries"][-1] != geometry["effective_iterations"]:
        raise ValueError("segment endpoints drift")
    if geometry["segment_boundaries"] != sorted(set(geometry["segment_boundaries"])):
        raise ValueError("segment boundaries are not strictly increasing")
    resume_boundaries = geometry["segment_boundaries"][1:-1]
    if any(value % geometry["save_interval"] for value in resume_boundaries):
        raise ValueError("resume boundary is not checkpoint-aligned")

    tokenizer = recipe["tokenizer"]
    if tokenizer["padding_tokens"] != 0:
        raise ValueError("production tokenizer must not use padding tokens")
    if tokenizer["vocab_size"] % tokenizer["make_vocab_size_divisible_by"]:
        raise ValueError("tokenizer vocabulary violates divisibility contract")

    phase_1 = recipe["phases"]["phase_1"]["mix_exact"]
    phase_2 = recipe["phases"]["phase_2"]["mix_exact"]
    if sum(Fraction(value) for value in phase_1.values()) != 1:
        raise ValueError("phase-1 mix does not sum exactly to one")
    if sum(Fraction(value) for value in phase_2.values()) != 1:
        raise ValueError("phase-2 mix does not sum exactly to one")

    dataset_counts = recipe["dataset"]["token_count_receipt"]
    natural_hplt = Fraction(
        dataset_counts["hplt_training_tokens"], dataset_counts["training_tokens"]
    )
    global_greek = Fraction(79, 100)
    expected_phase_2_hplt = (
        Fraction(geometry["effective_tokens"]) * global_greek * natural_hplt
        - Fraction(geometry["phase_1_tokens"]) * global_greek
    ) / Fraction(geometry["phase_2_tokens"])
    if Fraction(phase_2["hplt_new_greek"]) != expected_phase_2_hplt:
        raise ValueError("phase-2 HPLT share no longer preserves global natural composition")
    if Fraction(phase_2["non_hplt_new_greek"]) != global_greek - expected_phase_2_hplt:
        raise ValueError("phase-2 non-HPLT share drift")

    print(
        json.dumps(
            {
                "ok": True,
                "recipe_id": recipe["recipe_id"],
                "effective_tokens": geometry["effective_tokens"],
                "phase_boundary_iteration": geometry["phase_boundary_iteration"],
                "phase_2_hplt_share": str(expected_phase_2_hplt),
                "tokenizer_vocab_size": tokenizer["vocab_size"],
                "padding_tokens": tokenizer["padding_tokens"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
