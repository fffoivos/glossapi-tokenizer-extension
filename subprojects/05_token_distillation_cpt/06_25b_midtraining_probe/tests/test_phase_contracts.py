from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dataset"))
sys.path.insert(0, str(ROOT / "train" / "runtime_patches"))

from phase_partition import (  # noqa: E402
    UINT64_RANGE,
    assign_new_greek,
    assign_replay,
    composite_document_id,
    split_phase,
)
from finalize_phase_bridge import _write_env  # noqa: E402
from phase_relative_data_index import phase_relative_consumed_samples  # noqa: E402


def test_recipe_arithmetic_and_checkpoint_boundaries() -> None:
    recipe = json.loads((ROOT / "configs" / "recipe_25b_midtraining.json").read_text())
    geometry = recipe["geometry"]
    assert geometry["effective_iterations"] * geometry["global_batch_tokens"] == geometry["effective_tokens"]
    assert geometry["phase_1_iterations"] + geometry["phase_2_iterations"] == geometry["effective_iterations"]
    assert geometry["phase_1_tokens"] + geometry["phase_2_tokens"] == geometry["effective_tokens"]
    assert all(
        value % geometry["save_interval"] == 0
        for value in geometry["segment_boundaries"][1:-1]
    )
    assert geometry["effective_iterations"] % geometry["save_interval"] == 10
    phase_2 = recipe["phases"]["phase_2"]["mix_exact"]
    assert sum(Fraction(value) for value in phase_2.values()) == 1


def test_tokenizer_is_divisible_without_padding() -> None:
    recipe = json.loads((ROOT / "configs" / "recipe_25b_midtraining.json").read_text())
    tokenizer = recipe["tokenizer"]
    assert tokenizer["vocab_size"] % tokenizer["make_vocab_size_divisible_by"] == 0
    assert tokenizer["padding_tokens"] == 0


def test_new_greek_heldout_selectors_exist_in_published_schema() -> None:
    recipe = json.loads((ROOT / "configs" / "recipe_25b_midtraining.json").read_text())
    available = set(recipe["dataset"]["required_columns"]) | {
        "is_historical_or_polytonic"
    }
    specs = {row["name"]: row for row in recipe["heldouts"]["new_greek"]}
    assert all(row["selector_column"] in available for row in specs.values())
    assert specs["openarchives"]["selector_regex"] == r"^openarchives\.gr"
    assert specs["greek_phd"]["selector_regex"] == r"^greek_phd"


def test_partition_boundary_is_exact_and_deterministic() -> None:
    boundary = (3 * UINT64_RANGE + 4) // 5
    assert split_phase(boundary - 1) == 1
    assert split_phase(boundary) == 2
    first = assign_replay(seed=20260609, logical_pool="foreign_replay", document_id="x")
    assert first == assign_replay(seed=20260609, logical_pool="foreign_replay", document_id="x")


def test_new_greek_pool_and_phase_rules() -> None:
    hplt = assign_new_greek(seed=20260609, source_dataset="HPLT/ell_Grek", source_doc_id="17")
    other = assign_new_greek(seed=20260609, source_dataset="glossAPI/libduth", source_doc_id="17")
    assert hplt.logical_pool == "hplt_new_greek"
    assert hplt.phase in {1, 2}
    assert other.logical_pool == "non_hplt_new_greek"
    assert other.phase == 2
    assert other.score_u64 is None
    assert composite_document_id("a", "bc") != composite_document_id("ab", "c")


def test_composite_identity_rejects_missing_components() -> None:
    for source, doc_id in [("", "x"), ("x", ""), (None, "x")]:
        try:
            composite_document_id(source, doc_id)
        except ValueError:
            pass
        else:
            raise AssertionError("missing identity component was accepted")


def test_phase_relative_resume_every_segment() -> None:
    gbs = 1024
    phase_start = 3570
    assert phase_relative_consumed_samples(3570 * gbs, phase_start, gbs) == 0
    assert phase_relative_consumed_samples(4760 * gbs, phase_start, gbs) == 1190 * gbs
    assert phase_relative_consumed_samples(5950 * gbs, phase_start, gbs) == 2380 * gbs


def test_phase_relative_resume_rejects_preboundary_checkpoint() -> None:
    try:
        phase_relative_consumed_samples(3569 * 1024, 3570, 1024)
    except ValueError:
        pass
    else:
        raise AssertionError("pre-boundary checkpoint was accepted for phase 2")


def test_training_data_env_exports_logical_heldout_names(tmp_path: Path) -> None:
    output = tmp_path / "training_data.env"
    _write_env(
        output,
        input_receipt={"tokenizer": {"root": "/tokenizer"}},
        phase_blends={
            1: [{"weight_cli": "1", "prefix": "/phase1"}],
            2: [{"weight_cli": "1", "prefix": "/phase2"}],
        },
        heldouts={
            "sets": [
                {"pool": "new_greek", "name": "hplt"},
                {"pool": "foreign_replay", "name": "english"},
            ]
        },
        stage_root=tmp_path,
        input_receipt_path=tmp_path / "input.json",
        heldout_manifest_path=tmp_path / "heldouts.json",
        recipe_path=tmp_path / "recipe.json",
    )
    text = output.read_text(encoding="utf-8")
    assert 'EXTRA_VALID_SETS="english hplt"' in text
    assert "val_hplt" not in text
    assert "val_forget_english" not in text


def test_phase_config_restores_segment_exit_and_resume_after_shared_defaults(
    tmp_path: Path,
) -> None:
    bridge = tmp_path / "training_data.env"
    bridge.write_text(
        "\n".join(
            [
                'FULL_CPT_TOKENIZER_DIR="/tokenizer"',
                'PHASE1_CPT_DATA_PREFIX="1 /phase1"',
                'PHASE2_CPT_DATA_PREFIX="1 /phase2"',
                'VAL_DATA_DIR="/heldout"',
                'EXTRA_VALID_SETS="hplt english"',
                'FULL_CPT_BRIDGE_MANIFEST="/bridge.json"',
                'FULL_CPT_INPUT_RECEIPT="/input.json"',
                'FULL_CPT_HELDOUT_MANIFEST="/heldout.json"',
                'FULL_CPT_MIX_RECIPE="/recipe.json"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    config = ROOT / "train" / "phase_config.env"
    for start, end, phase, expected_resume in (
        (0, 1785, 1, 0),
        (1785, 3570, 1, 1),
        (3570, 5960, 2, 1),
    ):
        command = f"""
set -euo pipefail
BRIDGE_DATA_ENV={bridge!s}
CPT_PHASE={phase}
START_ITERATION={start}
END_ITERATION={end}
source {config!s}
printf '%s %s %s %s\n' "$EXIT_INTERVAL" "$RESUME_TRAINING" "$CPT_PHASE_START_ITERATION" "$FULL_CPT_DATA_PREFIX"
"""
        result = subprocess.run(
            ["bash", "-c", command], check=True, text=True, capture_output=True
        )
        phase_start = 0 if phase == 1 else 3570
        prefix = "/phase1" if phase == 1 else "/phase2"
        assert result.stdout.strip() == f"{end} {expected_resume} {phase_start} 1 {prefix}"
