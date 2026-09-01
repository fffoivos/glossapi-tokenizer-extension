#!/usr/bin/env python3
"""Validate the frozen 13.5B LR-floor reconstruction recipe."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path


HERE = Path(__file__).resolve().parent


def validate(recipe: dict) -> None:
    assert recipe["schema_version"] == "apertus8b_lr_floor_reconstruction_v1"
    assert recipe["status"] == "frozen"
    geometry = recipe["geometry"]
    seq = int(geometry["sequence_length"])
    gbs = int(geometry["global_batch_sequences"])
    nominal = int(geometry["nominal_tokens"])
    samples = nominal // seq
    iterations = samples // gbs
    assert samples == int(geometry["train_samples"]) == 3_295_898
    assert iterations == int(geometry["train_iterations"]) == 3_218
    assert iterations * seq * gbs == int(geometry["effective_tokens"])
    assert nominal - int(geometry["effective_tokens"]) == int(
        geometry["floor_residual_tokens"]
    )
    cooldown = samples // 5
    threshold = samples - cooldown
    branch = threshold // gbs
    assert cooldown == int(recipe["training"]["lr_decay_samples"]) == 659_179
    assert threshold == int(geometry["cooldown_threshold_samples"])
    assert branch == int(geometry["cooldown_branch_iteration"]) == 2_574
    assert branch * gbs < threshold <= (branch + 1) * gbs
    assert int(geometry["phase_boundary_iteration"]) == 2_253
    assert 2_253 < branch < iterations
    checkpoints = list(map(int, geometry["tail_averaging_checkpoints"]))
    assert checkpoints == [2_675, 2_782, 2_889, 2_996, 3_103, 3_210]
    assert all(b - a == 107 for a, b in zip(checkpoints, checkpoints[1:]))
    assert int(geometry["terminal_iteration"]) == iterations
    assert len(checkpoints) == 6 and checkpoints[-1] < iterations
    for phase in recipe["phases"].values():
        assert sum(Decimal(str(value)) for value in phase["mix_exact"].values()) == Decimal(1)
        values = list(phase["mix_exact"].values())
        assert set(values) == {
            "0.740740740740740741",
            "0.222222222222222222",
            "0.037037037037037037",
        }
    assert recipe["training"]["lr_floors"] == {
        "T10": "5.5e-6",
        "T20": "1.10e-5",
        "T30": "1.65e-5",
    }
    assert Decimal(recipe["training"]["lr_warmup_init"]) == Decimal("5.5e-6")
    assert int(recipe["tokenizer"]["vocab_size"]) % int(
        recipe["tokenizer"]["make_vocab_size_divisible_by"]
    ) == 0
    assert int(recipe["tokenizer"]["padding_tokens"]) == 0
    replay = recipe["replay_repartition"]
    assert int(replay["phase_1_numerator"]) == 2_253
    assert int(replay["phase_denominator"]) == 3_218
    assert int(replay["expected_tasks"]) == 164
    assert len(replay["input_receipt_sha256"]) == 64
    assert len(replay["heldout_manifest_sha256"]) == 64
    supplements = recipe["replay_supplements"]
    assert supplements["repo_id"] == "epfml/FineWeb2-HQ"
    assert len(supplements["revision"]) == 40
    assert int(supplements["expected_files"]) == 4
    assert int(supplements["expected_tasks"]) == 8
    assert len(supplements["download_receipt_sha256"]) == 64
    assert len(supplements["input_receipt_sha256"]) == 64
    assert len(supplements["heldout_manifest_sha256"]) == 64
    files = supplements["files"]
    assert len(files) == 4
    assert len({row["source_name"] for row in files}) == 4
    assert len({row["path"] for row in files}) == 4
    for row in files:
        assert int(row["bytes"]) > 0
        assert len(row["sha256"]) == 64
    runtime = recipe["runtime"]
    assert runtime["megatron_commit"] == "f8d8a30ba22a807321ec5875abbd9692b9282940"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=HERE / "configs" / "recipe_13b_lr_floor.json",
    )
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    validate(recipe)
    print(json.dumps({"ok": True, "recipe": str(args.recipe.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
