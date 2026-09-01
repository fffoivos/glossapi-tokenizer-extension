#!/usr/bin/env python3
"""Derive the exact full-run recipe and execution profiles from sanitized data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def segment_boundaries(updates: int) -> list[int]:
    # The proven DP32 rate is about 8.7 s/update. Four 4,000-update segments
    # remain below 10h, leaving more than two hours of the normal 12h limit.
    penultimate = updates - math.ceil(updates / 5)
    boundaries = [0, 4000, 8000, 12000, penultimate, updates]
    if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError(f"five-segment geometry is invalid for {updates} updates")
    return boundaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-recipe", type=Path, required=True)
    parser.add_argument("--base-profiles", type=Path, required=True)
    parser.add_argument("--pool-receipt", type=Path, required=True)
    parser.add_argument("--recipe-output", type=Path, required=True)
    parser.add_argument("--profiles-output", type=Path, required=True)
    args = parser.parse_args()

    pool_path = args.pool_receipt.resolve()
    pool = read(pool_path)
    if (
        pool.get("schema_version") != "apertus_schedule_pool_corpus_v1"
        or pool.get("status") != "completed"
        or not isinstance(pool.get("sanitized_bridge"), dict)
    ):
        raise ValueError("pool receipt is not a sanitized completed corpus")
    recipe = copy.deepcopy(read(args.base_recipe))
    profiles = copy.deepcopy(read(args.base_profiles))
    active_tokens = int(pool["integer_79_20_1_geometry"]["active_tokens"])
    global_batch_tokens = int(recipe["batch_and_parallelism"]["global_batch_tokens"])
    global_batch_sequences = int(recipe["batch_and_parallelism"]["global_batch_sequences"])
    updates = math.ceil(active_tokens / global_batch_tokens)
    slot_tokens = updates * global_batch_tokens
    stable = math.floor(0.8 * updates)
    cooldown = updates - stable
    boundaries = segment_boundaries(updates)
    dp64_boundaries = [0, updates // 3, (2 * updates) // 3, updates]
    greek_stride = round(5_000_000_000 / global_batch_tokens)
    greek_updates = {0, int(recipe["optimization"]["learning_rate"]["warmup_updates"]), stable, updates}
    greek_updates.update(range(greek_stride, updates + 1, greek_stride))
    greek = sorted(greek_updates)
    per_document = [0, stable, updates]
    checkpoints = sorted(
        set(greek[1:]) | set(boundaries[1:]) | set(dp64_boundaries[1:])
    )

    recipe["recipe_id"] = "full8b-mixed-79-20-1-wsd10-sanitized-v1"
    recipe["provenance_disclosures"]["openarchives_needs_ocr"] = (
        "The sanitized training derivative excludes exactly 6,648 "
        "openarchives.gr rows whose needs_ocr flag is true. The exclusion and "
        "zero retained matches are receipt-gated; no OCR-quality claim is made "
        "about other OpenArchives rows."
    )
    data = recipe["data"]
    data["source_binary_root"] = pool["source_root"]
    data["sanitized_source_receipt"] = {
        **pool["sanitized_bridge"],
        "pool_receipt": binding(pool_path),
    }
    data["eligibility_policy"] = {
        "openarchives_needs_ocr_true": "excluded",
        "proof": pool["sanitized_bridge"]["eligibility_audit"],
    }
    data["expected_pre_global_exact_dedup_capacity"] = {
        "modern_greek_tokens": int(pool["raw_counts"]["hplt_new_greek"]["tokens"])
        + int(pool["raw_counts"]["non_hplt_new_greek"]["tokens"]),
        "foreign_replay_tokens": int(pool["raw_counts"]["foreign_replay"]["tokens"]),
        "old_greek_replay_tokens": int(pool["raw_counts"]["old_greek_replay"]["tokens"]),
    }
    data["planning_post_dedup_active_tokens"] = active_tokens
    data["planning_training_slots_tokens"] = slot_tokens
    data["planning_loss_inactive_tail_tokens"] = slot_tokens - active_tokens
    optimization = recipe["optimization"]
    optimization["beta3_warmup_updates"] = updates
    optimization["alpha_warmup_updates"] = updates
    optimization["learning_rate"]["stable_until_update"] = stable
    optimization["learning_rate"]["cooldown_updates"] = cooldown
    batch = recipe["batch_and_parallelism"]
    batch["training_updates"] = updates
    batch["training_samples"] = updates * global_batch_sequences
    recipe["segments"]["boundaries"] = boundaries
    recipe["segments"]["count"] = len(boundaries) - 1
    evaluation = recipe["evaluation"]
    evaluation["source_conditioned"]["interval_updates"] = 238
    evaluation["source_conditioned"]["cadence_tokens_approx"] = 238 * global_batch_tokens
    evaluation["greekmmlu"]["checkpoint_updates"] = greek
    evaluation["greekmmlu"]["cadence_updates"] = greek_stride
    evaluation["per_document_validation"]["milestone_updates"] = per_document
    evaluation["checkpoint_updates"] = checkpoints
    recipe["initialization"]["token_distillation_dropout_context"] = {
        "model_mode_during_teacher_hidden_state_extraction": "train",
        "attention_dropout": 0.1,
        "hidden_dropout": 0.1,
        "existing_verified_initialization_preserved": True,
        "claim": "dropout_active_token_distillation_initialization",
    }
    recipe["derivation"] = {
        "policy": "full_eligible_sanitized_modern_pass_with_exact_79_20_1_replay_v1",
        "base_recipe": binding(args.base_recipe),
        "pool_receipt": binding(pool_path),
        "active_tokens": active_tokens,
        "global_batch_tokens": global_batch_tokens,
        "training_updates": updates,
        "loss_inactive_tail_tokens": slot_tokens - active_tokens,
    }

    profiles["scientific_recipe_id"] = recipe["recipe_id"]
    profiles["scientific_invariants"]["training_updates"] = updates
    profiles["profiles"]["dp32_16node"]["segment_boundaries"] = boundaries
    profiles["profiles"]["dp64_32node"]["segment_boundaries"] = dp64_boundaries
    profiles["derivation"] = {
        "base_profiles": binding(args.base_profiles),
        "recipe_output": str(args.recipe_output.resolve()),
        "sanitized_pool_receipt": binding(pool_path),
    }

    write_exclusive(args.recipe_output, recipe)
    write_exclusive(args.profiles_output, profiles)
    print(json.dumps({
        "ok": True,
        "active_tokens": active_tokens,
        "training_updates": updates,
        "loss_inactive_tail_tokens": slot_tokens - active_tokens,
        "segments": boundaries,
        "greekmmlu_checkpoints": len(greek),
        "source_validation_interval_updates": 238,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
