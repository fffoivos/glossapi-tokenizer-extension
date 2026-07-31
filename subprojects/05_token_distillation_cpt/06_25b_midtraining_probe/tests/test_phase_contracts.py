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
from finalize_phase_bridge import (  # noqa: E402
    _capacity_report,
    _prove_disjoint_and_measure_uniqueness,
    _write_env,
)
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


def test_capacity_policy_is_frozen_at_1_005_plus_boundary() -> None:
    recipe = json.loads((ROOT / "configs" / "recipe_25b_midtraining.json").read_text())
    assert recipe["capacity"] == {
        "basis": "exact_unique_text_sha256_tokens",
        "minimum_unique_capacity_ratio": "1.005",
        "physical_prefix_sample_capacity_ratio": "1.005",
        "physical_prefix_boundary_samples": 1,
        "gate_scope": [
            "phase_logical_pool",
            "phase_source",
            "phase_physical_prefix",
        ],
    }


def test_after_freeze_is_receipt_gated_without_completed_job_dependency() -> None:
    launcher = (ROOT / "clariden" / "submit_data_pipeline.sh").read_text()
    after_freeze = launcher.split("  after-freeze)", 1)[1].split("  assets)", 1)[0]
    assert 'test -s "$INPUT_RECEIPT"' in after_freeze
    assert 'test -s "$HELDOUT_MANIFEST"' in after_freeze
    assert 'h.get("input_receipt_sha256") != sha(input_path)' in after_freeze
    assert 'h.get("config_sha256") != sha(recipe_path)' in after_freeze
    assert "heldout_job" not in after_freeze
    assert '--dependency="afterok:$dependency"' in after_freeze


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


def test_smoke_config_is_small_and_resets_phase2_index(tmp_path: Path) -> None:
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
    config = ROOT / "train" / "smoke_phase_config.env"
    for start, end, phase, expected_resume, phase_start, prefix in (
        (0, 1, 1, 0, 0, "/phase1"),
        (1, 2, 2, 1, 1, "/phase2"),
    ):
        command = f"""
set -euo pipefail
BRIDGE_DATA_ENV={bridge!s}
CPT_SMOKE=1
CPT_PHASE={phase}
START_ITERATION={start}
END_ITERATION={end}
source {config!s}
printf '%s %s %s %s %s %s %s %s\n' "$EXIT_INTERVAL" "$RESUME_TRAINING" "$CPT_PHASE_START_ITERATION" "$CPT_GLOBAL_BATCH_SIZE" "$GLOBAL_BATCH_SIZE" "$TRAIN_ITERS" "$SAVE_INTERVAL" "$FULL_CPT_DATA_PREFIX"
"""
        result = subprocess.run(
            ["bash", "-c", command], check=True, text=True, capture_output=True
        )
        assert result.stdout.strip() == (
            f"{end} {expected_resume} {phase_start} 8 8 2 1 1 {prefix}"
        )


def test_smoke_launcher_has_separate_confirmation_and_geometry() -> None:
    launcher = (ROOT / "train" / "submit_smoke.sh").read_text(encoding="utf-8")
    assert "CONFIRM_GPU_LAUNCH=GREEK_CPT25B_SMOKE" in launcher
    assert "GREEK_CPT25B_64GPU" not in launcher
    assert "--nodes=1" in launcher
    assert "--gpus-per-node=4" in launcher
    assert "START_ITERATION=0,END_ITERATION=1,CPT_PHASE=1" in launcher
    assert "START_ITERATION=1,END_ITERATION=2,CPT_PHASE=2" in launcher

    production = (ROOT / "train" / "submit_segment.sh").read_text(encoding="utf-8")
    assert 'SMOKE_VERIFICATION:?set SMOKE_VERIFICATION' in production
    assert "SMOKE_VERIFICATION=$SMOKE_VERIFICATION" in production
    assert "freeze_checkpoint.sbatch" in production
    assert "checkpoint_freeze_job_id" in production


def test_preflight_keeps_smoke_and_production_boundaries_disjoint() -> None:
    sys.path.insert(0, str(ROOT / "clariden"))
    from preflight_segment import BOUNDARIES, SMOKE_BOUNDARIES

    assert BOUNDARIES == {0: (1, 1785), 1785: (1, 3570), 3570: (2, 5960)}
    assert SMOKE_BOUNDARIES == {0: (1, 1), 1: (2, 2)}
    assert set(BOUNDARIES.items()).isdisjoint(SMOKE_BOUNDARIES.items())


def test_checkpoint_watcher_waits_for_completed_marker_and_uses_new_sidecar() -> None:
    watcher = (ROOT / "eval" / "watch_greekmmlu_checkpoints.sbatch").read_text(
        encoding="utf-8"
    )
    assert "submit_greekmmlu_checkpoint.sh" in watcher
    assert "submit_td_checkpoint_sidecars.sh" not in watcher
    assert "latest_checkpointed_iteration.txt" in watcher
    assert "latest >= iteration" in watcher
    assert 'range(119,5951,119)' in watcher
    assert "if end == 5960: values.append(5960)" in watcher


def test_greekmmlu_finalizer_freezes_full_16632_result(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "iter_0000119"
    checkpoint.mkdir(parents=True)
    (checkpoint / ".metadata").write_text("complete", encoding="utf-8")
    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    for name in ("config.json", "tokenizer.json", "model.safetensors.index.json"):
        (hf_dir / name).write_text("{}", encoding="utf-8")
    (hf_dir / "model-00001-of-00001.safetensors").write_bytes(b"model")
    eval_dir = tmp_path / "greekmmlu"
    eval_dir.mkdir()
    aggregate = {
        "schema": "native-greek-mcq-aggregate-v1",
        "headline": {
            "n_tasks": 1,
            "total_n": 16632,
            "micro_accuracy": 0.5,
        },
    }
    (eval_dir / "model_native_mcq_aggregate.json").write_text(
        json.dumps(aggregate), encoding="utf-8"
    )
    assets = tmp_path / "assets.json"
    assets.write_text(
        json.dumps(
            {
                "schema_version": "greek_cpt_training_assets_receipt_v1",
                "status": "frozen",
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "evaluation_receipt.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "eval" / "finalize_greekmmlu_checkpoint.py"),
            "--iteration",
            "119",
            "--tokens",
            str(119 * 4_194_304),
            "--checkpoint-dir",
            str(checkpoint),
            "--hf-dir",
            str(hf_dir),
            "--eval-dir",
            str(eval_dir),
            "--training-assets-receipt",
            str(assets),
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "passed"
    assert receipt["iteration"] == 119
    assert receipt["greekmmlu"]["total_n"] == 16632
    assert receipt["greekmmlu"]["micro_accuracy"] == 0.5


def test_production_resume_receipt_hashes_only_boundary_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    old = checkpoint_root / "iter_0000119"
    boundary = checkpoint_root / "iter_0001785"
    old.mkdir(parents=True)
    boundary.mkdir()
    (old / ".metadata").write_text("old", encoding="utf-8")
    (boundary / ".metadata").write_text("boundary", encoding="utf-8")
    (boundary / "state.pt").write_bytes(b"state")
    (checkpoint_root / "latest_checkpointed_iteration.txt").write_text(
        "1785\n", encoding="utf-8"
    )
    assets = tmp_path / "assets.json"
    assets.write_text(
        json.dumps(
            {
                "schema_version": "greek_cpt_training_assets_receipt_v1",
                "status": "frozen",
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "resume.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "train" / "freeze_resume_checkpoint.py"),
            "--checkpoint-dir",
            str(checkpoint_root),
            "--iteration",
            "1785",
            "--training-assets-receipt",
            str(assets),
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["checkpoint_root"] == str(checkpoint_root.resolve())
    assert receipt["checkpoint_tree"]["root"] == str(boundary.resolve())
    assert {row["path"] for row in receipt["checkpoint_tree"]["files"]} == {
        ".metadata",
        "state.pt",
    }


def test_two_phase_capacity_gate_checks_boundary_sample_exactly() -> None:
    recipe = {
        "capacity": {
            "minimum_unique_capacity_ratio": "1.005",
            "physical_prefix_sample_capacity_ratio": "1.005",
            "physical_prefix_boundary_samples": 1,
        },
        "geometry": {
            "sequence_length": 4,
            "global_batch_sequences": 10,
            "phase_1_iterations": 10,
            "phase_2_iterations": 10,
        },
        "phases": {
            "phase_1": {"mix_exact": {"pool": "1"}},
            "phase_2": {"mix_exact": {"pool": "1"}},
        },
    }
    blends = {
        phase: [
            {
                "task_id": f"task-{phase}",
                "logical_pool": "pool",
                "source_name": "source",
                "prefix": f"/phase-{phase}",
                "weight_exact": "1",
            }
        ]
        for phase in (1, 2)
    }
    uniqueness = {
        "tasks": [
            {
                "phase": phase,
                "task_id": f"task-{phase}",
                "unique_content_tokens": 409,
            }
            for phase in (1, 2)
        ],
        "pools": [
            {"phase": phase, "logical_pool": "pool", "unique_content_tokens": 409}
            for phase in (1, 2)
        ],
        "sources": [
            {
                "phase": phase,
                "logical_pool": "pool",
                "source_name": "source",
                "unique_content_tokens": 409,
            }
            for phase in (1, 2)
        ],
    }
    report, failures = _capacity_report(recipe, blends, uniqueness)
    assert failures == []
    prefix = report["phases"]["1"]["physical_prefixes"][0]
    assert prefix["planned_samples"] == 100
    assert prefix["required_samples"] == 102
    assert prefix["available_nonrepeating_samples"] == 102

    uniqueness["tasks"][0]["unique_content_tokens"] = 405
    _, failures = _capacity_report(recipe, blends, uniqueness)
    assert failures == [
        "phase-1/prefix/task-1: nonrepeating samples 101 < required 102"
    ]


def test_phase_disjoint_proof_also_discounts_duplicate_content(tmp_path: Path) -> None:
    shards = []
    for phase, suffix in ((1, "1"), (2, "2")):
        ledger = tmp_path / f"phase{phase}.jsonl"
        ledger.write_text(
            json.dumps(
                {
                    "doc_id": "docv2:" + suffix * 64,
                    "text_sha256": "a" * 64,
                    "tokens": 3,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        shards.append(
            {
                "task_id": f"task-{phase}",
                "source_name": "source",
                "counts": {"documents": 1, "tokens": 3},
                "task": {
                    "task_index": phase - 1,
                    "phase_partition": {
                        "phase": phase,
                        "logical_pool": "pool",
                    },
                },
                "outputs": {"retained_ledger": {"path": str(ledger)}},
            }
        )
    proof, uniqueness = _prove_disjoint_and_measure_uniqueness(
        shards, tmp_path / "proof.sqlite"
    )
    assert proof["unique_documents"] == 2
    assert proof["documents_by_phase"] == {"1": 1, "2": 1}
    assert uniqueness["global"]["identity_tokens"] == 6
    assert uniqueness["global"]["unique_content_documents"] == 1
    assert uniqueness["global"]["unique_content_tokens"] == 3
    assert [row["unique_content_tokens"] for row in uniqueness["tasks"]] == [3, 3]
