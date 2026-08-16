from __future__ import annotations

import json
import sys
from itertools import pairwise
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_canonical_campaign_contracts import (  # noqa: E402
    BOUNDARIES,
    LATE_BOUND_PHASE_CACHE,
    MILESTONES,
    NATIVE_MILESTONES,
    PROFILE_CHECKS,
    build_evaluation_plan,
    build_campaign,
    build_qualification_gates,
    build_runtime,
    canonical_digest,
    build_segments,
    late_phase3_asset,
    phase_for_start,
)

EXPERIMENT = json.loads(
    (ROOT / "configs/hard_h_to_g_replication_v1.json").read_text(encoding="utf-8")
)


def run_permit(scale: str, *, nodes: int) -> dict[str, object]:
    tensor = 2 if scale == "8b" else 1
    return {
        "profile": {
            "profile_id": "dp32_16node" if scale == "8b" else f"1p5b_tp1_{nodes}node",
            "nodes": nodes,
            "tensor_parallel": tensor,
            "pipeline_parallel": 1,
            "context_parallel": 1,
            "data_parallel": nodes * 4 // tensor,
            "microbatch": 2 if scale == "8b" else 8,
        },
        "learning_rate": {"peak": "5.5e-5", "floor": "5.5e-6"},
    }


def evaluation_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "stage_root": tmp_path / "stage",
        "code_root": tmp_path / "science",
        "canonical_runner_root": tmp_path / "runner",
        "code_receipt": tmp_path / "science.json",
        "initial_checkpoint_root": tmp_path / "init",
        "megatron_root": tmp_path / "megatron",
        "tokenizer_dir": tmp_path / "tokenizer",
        "experiment": tmp_path / "experiment.json",
        "eval_venv": tmp_path / "venv",
        "validation_manifest": tmp_path / "validation.json",
        "clean_examples": tmp_path / "clean.json",
        "sentinel_manifest": tmp_path / "sentinel.json",
        "native_eval_code_root": tmp_path / "native-code",
        "native_eval_code_receipt": tmp_path / "native-code.json",
        "native_source_contract": tmp_path / "native-contract.json",
        "native_source_manifest": tmp_path / "native-manifest.json",
        "native_source_gate": tmp_path / "native-gate.json",
        "native_exclusions": tmp_path / "native-exclusions.jsonl",
        "legacy_contract": tmp_path / "legacy.json",
    }


def timing_and_schedule(scale: str) -> tuple[dict[str, object], dict[str, object]]:
    rows = []
    for segment_id, (start, end) in enumerate(pairwise(BOUNDARIES)):
        rows.append(
            {
                "segment_id": segment_id,
                "start_update": start,
                "exit_update": end,
                "minimum_train_seconds": 1000 + segment_id,
                "conservative_wall_seconds": 10_000 + segment_id,
            }
        )
    successors = []
    for target in range(1, 6):
        target_runtime = rows[target]["conservative_wall_seconds"]
        maximum_hold = 43_200 - target_runtime - 1_200
        source_runtime = rows[target - 1]["conservative_wall_seconds"]
        source_trigger = max(60, source_runtime - maximum_hold)
        successors.append(
            {
                "scale": scale,
                "source_segment_id": target - 1,
                "target_segment_id": target,
                "source_trigger_seconds": source_trigger,
                "maximum_hold_seconds": maximum_hold,
                "target_runtime_seconds": target_runtime,
                "reserve_seconds": 1_200,
            }
        )
    return (
        {"scales": {scale: {"segments": rows}}},
        {
            "allocation_seconds": 43_200,
            "successors": successors,
        },
    )


def phase_assets(tmp_path: Path) -> dict[int, dict[str, object]]:
    return {
        phase: {
            "phase": phase,
            "spec": tmp_path / f"phase{phase}.json",
            "receipt": tmp_path / f"phase{phase}.receipt.json",
            "root": tmp_path / f"phase{phase}-cache",
            "data_path": f"1.0 /phase{phase}/modern 0.25 /phase{phase}/foreign",
            "cache_sha256": str(phase) * 64,
        }
        for phase in (1, 2, 3)
    }


def test_runtime_preserves_scale_geometry_and_allows_four_node_evaluation() -> None:
    runtime_8b = build_runtime(
        "8b",
        run_permit("8b", nodes=16),
        experiment=EXPERIMENT,
        megatron_revision="c92402e39ef3c8e69ea378a59e79059dc14541f4",
    )
    assert runtime_8b["parallelism"] == {
        "tensor": 2,
        "pipeline": 1,
        "context": 1,
        "data": 32,
    }
    assert runtime_8b["slurm"]["nodes"] == 16
    runtime_1p5b = build_runtime(
        "1p5b",
        run_permit("1p5b", nodes=1),
        experiment=EXPERIMENT,
        megatron_revision="c92402e39ef3c8e69ea378a59e79059dc14541f4",
    )
    assert runtime_1p5b["parallelism"]["data"] == 4
    assert runtime_1p5b["submission_policy"]["max_nodes"] == 4
    assert runtime_1p5b["qualification_scope"] == {
        "model_id": "swiss-ai/Apertus-v1.1-1.5B",
        "model_revision": "dbe8919b2f0389888bada6b3a19e81e0ef4286c1",
        "model_geometry_sha256": runtime_1p5b["qualification_scope"]["model_geometry_sha256"],
        "megatron_revision": "c92402e39ef3c8e69ea378a59e79059dc14541f4",
        "sequence_length": 4096,
        "micro_batch": 8,
        "global_batch_tokens": 4_194_304,
        "precision": "bf16",
        "checkpoint_format": "megatron-core-distributed-checkpoint-v1",
    }


def test_qualification_gates_consume_predeclared_profile_checks(tmp_path: Path) -> None:
    code = tmp_path / "code.json"
    cache = tmp_path / "cache.json"
    code.write_text("{}\n", encoding="utf-8")
    cache.write_text("{}\n", encoding="utf-8")
    gates = build_qualification_gates(code_receipt=code, phase1_cache_receipt=cache)
    by_id = {row["id"]: row for row in gates}
    assert set(by_id) == {
        "scientific_code_binding",
        "phase1_cache_binding",
        "profile_receipt_passed",
        "exact_slurm_profile",
        *PROFILE_CHECKS,
    }
    assert by_id["profile_receipt_passed"]["accepted"] == ["passed"]
    for check in PROFILE_CHECKS:
        assert by_id[check]["type"] == "json_field_equal"
        assert by_id[check]["left"] == {
            "evidence_id": "profile_measurement",
            "field": f"checks.{check}",
        }
        assert by_id[check]["right"] == {"literal": True}
    assert not {"restart_parity", "checkpoint_cursor", "training_metrics"} & {
        row["type"] for row in gates
    }


def test_1p5b_campaign_binds_first_allocation_qualification(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    code = tmp_path / "science"
    runner = tmp_path / "runner"
    canonical = tmp_path / "canonical"
    for root in (stage, code, runner, canonical):
        root.mkdir(parents=True)
    keys = {
        "code_receipt", "experiment", "allocation", "initialization_permit",
        "run_permit", "profile_promotion", "timing", "schedule",
        "pre_main_launch_gate", "megatron_receipt", "validation_receipt",
        "tokenizer_receipt", "validation_manifest", "clean_examples",
        "sentinel_manifest", "statistical_contract", "qualification_contract",
        "producer_compatibility", "native_eval_code_receipt",
        "native_source_contract", "native_source_manifest", "native_source_gate",
        "native_exclusions", "legacy_contract",
    }
    paths = {key: stage / f"{key}.json" for key in keys}
    for path in paths.values():
        path.write_text("{}\n", encoding="utf-8")
    paths.update(
        {
            "stage_root": stage,
            "code_root": code,
            "canonical_runner_root": runner,
            "canonical_data_root": canonical,
            "training_data_manifest": canonical / "training_data_manifest.json",
            "initial_checkpoint_root": stage / "init",
            "megatron_root": stage / "megatron",
            "validation_root": stage / "validation",
            "tokenizer_dir": stage / "tokenizer",
            "eval_venv": stage / "eval-venv",
            "native_eval_code_root": stage / "native-eval",
        }
    )
    paths["megatron_receipt"].write_text(
        json.dumps({"upstream_commit": "c92402e39ef3c8e69ea378a59e79059dc14541f4"}) + "\n",
        encoding="utf-8",
    )
    paths["training_data_manifest"].write_text("{}\n", encoding="utf-8")
    permit = run_permit("1p5b", nodes=1)
    runtime = build_runtime(
        "1p5b", permit, experiment=EXPERIMENT,
        megatron_revision="c92402e39ef3c8e69ea378a59e79059dc14541f4",
    )
    timing, schedule = timing_and_schedule("1p5b")
    assets = phase_assets(tmp_path)
    for asset in assets.values():
        asset["spec"].write_text("{}\n", encoding="utf-8")
        asset["receipt"].write_text("{}\n", encoding="utf-8")
    campaign = build_campaign(
        scale="1p5b", paths=paths, experiment=EXPERIMENT,
        run_permit=permit, data_manifest={"identity_sha256": "d" * 64},
        runtime=runtime, evaluation={"plan_id": "eval"},
        phase_assets=assets, timing=timing, schedule=schedule,
    )
    qualification = campaign["qualification"]["argv"]
    assert qualification[0] == "/usr/bin/env"
    assert qualification[1].startswith("PYTHONPATH=")
    assert str(code / "subprojects/08_targeted_8b_cpt_experiments/scripts") in qualification[1]
    assert str(runner / "src/_vendor/campaign_pydeps") in qualification[1]
    assert qualification[2] == "python3"
    assert "{qualification_context}" in qualification
    assert "{qualification_root}" in qualification
    assert any(value.endswith("run_in_allocation_profile_qualification.py") for value in qualification)
    assert campaign["science"]["runtime_requirements_sha256"] == canonical_digest(
        runtime["qualification_scope"]
    )
    train = campaign["science"]["train_argv"]
    assert train[:1] == ["/usr/bin/env"]
    assert train[1].startswith("PYTHONPATH=")
    assert train[2] == "/usr/bin/python3.11"


def test_evaluation_order_enforces_export_before_dependent_scorers(
    tmp_path: Path,
) -> None:
    plan = build_evaluation_plan("8b", evaluation_paths(tmp_path))
    evaluator_ids = [row["id"] for row in plan["evaluators"]]
    assert evaluator_ids[:5] == [
        "checkpoint_export",
        "offline_panels",
        "greekmmlu",
        "native_greek_suite",
        "legacy_public_greekmmlu",
    ]
    assert evaluator_ids[5] == "greekmmlu_calibration"
    assert evaluator_ids[6:15] == [
        f"greekmmlu_full_fallback_i{target:07d}"
        for target in (952, 1190, 1428, 1666, 1904, 2142, 2261, 2380, 3456)
    ]
    assert evaluator_ids[15] == "greekmmlu_plateau_confirmation"
    for evaluator in plan["evaluators"]:
        assert evaluator["argv"][0] == "/usr/bin/env"
        assert evaluator["argv"][1].startswith("PYTHONPATH=")
        assert evaluator["argv"][2] == "/usr/bin/python3.11"
    assert plan["evaluators"][0]["milestone_iterations"] == list(MILESTONES)
    assert plan["evaluators"][2]["milestone_iterations"] == list(MILESTONES)
    assert plan["evaluators"][3]["milestone_iterations"] == list(NATIVE_MILESTONES)
    assert plan["evaluators"][4]["milestone_iterations"] == [3218]
    assert plan["evaluators"][5]["milestone_iterations"] == [3218]
    assert plan["evaluators"][6]["milestone_iterations"] == [3218]
    assert plan["evaluators"][6]["release_gate"] == {
        "path": str(
            tmp_path
            / "stage/receipts/greekmmlu_sentinel_calibration_authority.json"
        ),
        "expected_schema": "apertus_greekmmlu_sentinel_calibration_authority_v1",
        "accepted_statuses": ["passed"],
        "field_equals": {"scope": "both_scales"},
    }
    assert plan["evaluators"][14]["milestone_iterations"] == [3456]
    assert plan["evaluators"][15]["milestone_iterations"] == [3694]
    plan_1p5b = build_evaluation_plan("1p5b", evaluation_paths(tmp_path))
    ids_1p5b = [row["id"] for row in plan_1p5b["evaluators"]]
    assert "legacy_public_greekmmlu" not in ids_1p5b
    assert ids_1p5b[4] == "greekmmlu_calibration"
    assert ids_1p5b[-1] == "greekmmlu_plateau_confirmation"


def test_segments_bind_exact_main_assets_and_late_phase3_receipt(
    tmp_path: Path,
) -> None:
    assets = phase_assets(tmp_path)
    assets[3]["data_path"] = LATE_BOUND_PHASE_CACHE
    assets[3]["cache_sha256"] = LATE_BOUND_PHASE_CACHE
    timing, schedule = timing_and_schedule("8b")
    pre_main_gate = tmp_path / "stage/receipts/launch_gate_pre_main_8b_v65.json"
    segments = build_segments(
        scale="8b",
        initialization_root=tmp_path / "init",
        initialization_permit=tmp_path / "init.json",
        phase_assets=assets,
        stage_root=tmp_path / "stage",
        timing=timing,
        schedule=schedule,
        pre_main_launch_gate=pre_main_gate,
    )
    assert [(row["start_iteration"], row["end_iteration"]) for row in segments] == list(
        pairwise(BOUNDARIES)
    )
    assert segments[0]["load_checkpoint"] == str(tmp_path / "init")
    assert all(
        row["load_checkpoint"] == "{previous_checkpoint}" for row in segments[1:]
    )
    assert "--initialization-permit" in segments[0]["argv_overrides"]
    assert all(
        "--initialization-permit" not in row["argv_overrides"] for row in segments[1:]
    )
    assert str(pre_main_gate) in segments[0]["argv_overrides"]
    assert LATE_BOUND_PHASE_CACHE in segments[4]["argv_overrides"]
    assert any(
        value.endswith("launch_gate_pre_extension.json")
        for value in segments[4]["argv_overrides"]
    )
    assert any(
        value.endswith("launch_gate_pre_second_extension.json")
        for value in segments[5]["argv_overrides"]
    )
    assert "release_gate" not in segments[3]
    assert segments[4]["release_gate"] == {
        "path": str(tmp_path / "stage/receipts/launch_gate_pre_extension.json"),
        "expected_schema": "apertus_hard_h_to_g_frozen_contract_v2",
        "accepted_statuses": ["launch_ready"],
        "field_equals": {"gate_stage": "pre_extension"},
    }
    assert segments[5]["release_gate"]["field_equals"] == {
        "gate_stage": "pre_second_extension"
    }
    for row in segments[1:]:
        handoff = row["handoff"]
        assert (
            handoff["conservative_target_runtime_seconds"]
            + handoff["maximum_hold_seconds"]
            + handoff["reserve_seconds"]
            == handoff["allocation_seconds"]
        )
        assert (
            row["minimum_train_seconds"]
            < handoff["conservative_target_runtime_seconds"]
        )


def test_phase_mapping_matches_scientific_boundaries() -> None:
    assert [phase_for_start(value) for value in BOUNDARIES[:-1]] == [1, 1, 1, 2, 3, 3]
    with pytest.raises(ValueError, match="invalid segment start"):
        phase_for_start(3694)


def test_future_phase3_paths_must_remain_inside_stage(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    value = late_phase3_asset(
        spec=stage / "phase3.json",
        receipt=stage / "phase3.receipt.json",
        root=stage / "cache",
        stage_root=stage,
    )
    assert value["data_path"] == LATE_BOUND_PHASE_CACHE
    with pytest.raises(ValueError, match="escapes the stage root"):
        late_phase3_asset(
            spec=tmp_path / "outside.json",
            receipt=stage / "phase3.receipt.json",
            root=stage / "cache",
            stage_root=stage,
        )


def test_campaign_compiler_wrapper_is_debug_only_and_verifies_both_bundles() -> None:
    wrapper = (
        ROOT / "clariden/build_canonical_campaign_contracts_debug.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --partition=debug" in wrapper
    assert '"${SLURM_NNODES:-0}" == 1' in wrapper
    assert '--root "$H2G_CODE_ROOT"' in wrapper
    assert '--root "$APERTUS_CANONICAL_ROOT"' in wrapper
    assert "--phase3-cache-receipt" in wrapper
    assert "H2G_MEGATRON_RECEIPT" in wrapper
    assert '--megatron-receipt "$H2G_MEGATRON_RECEIPT"' in wrapper
    assert "H2G_PRODUCER_COMPATIBILITY" in wrapper
    assert '--producer-compatibility "$H2G_PRODUCER_COMPATIBILITY"' in wrapper
    assert "--native-exclusions" in wrapper


def test_campaign_compiler_never_hardcodes_an_older_megatron_receipt() -> None:
    compiler = (ROOT / "scripts/build_canonical_campaign_contracts.py").read_text(
        encoding="utf-8"
    )
    assert 'parser.add_argument("--megatron-receipt"' in compiler
    assert 'receipts / "training_megatron_runtime.json"' not in compiler
    assert "training Megatron receipt escapes the stage receipts root" in compiler
