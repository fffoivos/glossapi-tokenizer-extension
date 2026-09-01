#!/usr/bin/env python3
"""Validate a 16- or 32-node execution profile without changing science."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from contract import atomic_write_json, file_binding, read_json, scientific_digest


def validate(recipe: dict, profiles: dict, profile_id: str) -> dict:
    if profiles.get("schema_version") != "apertus_full_8b_execution_profiles_v1":
        raise ValueError("execution profile schema drift")
    if profiles.get("scientific_recipe_id") != recipe.get("recipe_id"):
        raise ValueError("profile/recipe identity drift")
    profile = profiles.get("profiles", {}).get(profile_id)
    if not isinstance(profile, dict):
        raise ValueError(f"unknown execution profile: {profile_id}")
    batch = recipe["batch_and_parallelism"]
    invariants = profiles["scientific_invariants"]
    expected = {
        "sequence_length": recipe["data"]["sequence_length"],
        "global_batch_sequences": batch["global_batch_sequences"],
        "global_batch_tokens": batch["global_batch_tokens"],
        "micro_batch_sequences": batch["micro_batch_sequences"],
        "training_updates": batch["training_updates"],
        "tensor_parallel": batch["tensor_parallel"],
        "pipeline_parallel": batch["pipeline_parallel"],
        "context_parallel": batch["context_parallel"],
    }
    for key, value in expected.items():
        if invariants.get(key) != value:
            raise ValueError(f"scientific invariant drift: {key}")
    if profile["world_size"] != profile["nodes"] * profile["gpus_per_node"]:
        raise ValueError("profile world-size drift")
    denominator = batch["tensor_parallel"] * batch["pipeline_parallel"] * batch["context_parallel"]
    if profile["data_parallel"] != profile["world_size"] // denominator:
        raise ValueError("profile DP drift")
    micro_global = profile["data_parallel"] * batch["micro_batch_sequences"]
    if batch["global_batch_sequences"] % micro_global:
        raise ValueError("global batch is not divisible by DP times microbatch")
    if profile["gradient_accumulation_steps"] != batch["global_batch_sequences"] // micro_global:
        raise ValueError("gradient accumulation drift")
    boundaries = profile["segment_boundaries"]
    if boundaries[0] != 0 or boundaries[-1] != batch["training_updates"]:
        raise ValueError("profile boundaries do not cover the run")
    if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("profile boundaries are not strictly increasing")
    if not set(boundaries[1:]).issubset(set(recipe["evaluation"]["checkpoint_updates"])):
        raise ValueError("profile boundary lacks an exact saved checkpoint")
    return {**profile, "profile_id": profile_id, "scientific_digest": scientific_digest(recipe)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    recipe = read_json(args.recipe)
    profiles = read_json(args.profiles)
    selected = validate(recipe, profiles, args.profile_id)
    payload = {
        "schema_version": "apertus_full_8b_selected_execution_profile_v1",
        "status": "frozen",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "recipe": file_binding(args.recipe),
        "profiles": file_binding(args.profiles),
        "selection": selected,
    }
    if args.output:
        atomic_write_json(args.output, payload)
    print(json.dumps({"ok": True, **selected}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
