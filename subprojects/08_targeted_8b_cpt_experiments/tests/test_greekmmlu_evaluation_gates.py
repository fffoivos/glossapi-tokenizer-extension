from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "evaluation"))

from contract_utils import file_binding, read_json, write_json_atomic
from run_greekmmlu_calibration_evaluator import main as calibration_main
from run_greekmmlu_fallback_evaluator import main as fallback_main
from run_greekmmlu_plateau_evaluator import main as plateau_main
from validate_greekmmlu_sentinels import CALIBRATION_UPDATES


def write_text(path: Path, value: str = "{}\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def canonical_result(
    run_root: Path,
    *,
    evaluator_id: str,
    iteration: int,
    scale: str,
    schema: str,
    payload: dict[str, object],
) -> Path:
    attempt = (
        run_root
        / "evaluations"
        / evaluator_id
        / f"iter_{iteration:07d}"
        / "attempts/attempt_000001"
    )
    attempt.mkdir(parents=True)
    result_path = attempt / "result.json"
    result = {
        "schema_version": schema,
        "status": "completed",
        "campaign_id": "campaign",
        "evaluator_id": evaluator_id,
        "iteration": iteration,
        "attempt": 1,
        "contract_digest": "digest",
        "scale": scale,
        **payload,
    }
    write_json_atomic(result_path, result)
    write_json_atomic(
        attempt / "evaluation.json",
        {
            "schema_version": "apertus_campaign_evaluation_attempt_v2",
            "status": "completed",
            "result": file_binding(result_path),
            "result_checks": {
                "schema": True,
                "campaign_id": True,
                "evaluator_id": True,
                "iteration": True,
                "attempt": True,
                "contract_digest": True,
            },
        },
    )
    return result_path


def frozen_evaluation(
    root: Path,
    *,
    scale: str,
    iteration: int,
    mode: str,
    views: tuple[str, ...],
) -> Path:
    bindings = {}
    for view in views:
        bindings[view] = file_binding(write_text(root / f"{view}.jsonl", "{\"example_id\":\"q\"}\n"))
    path = root / "receipt.json"
    write_json_atomic(
        path,
        {
            "schema_version": "apertus_frozen_greekmmlu_evaluation_v1",
            "status": "completed",
            "scale": scale,
            "iteration": iteration,
            "mode": mode,
            "views": bindings,
        },
    )
    return path


def calibration_payload(
    path: Path,
    *,
    scale: str,
    full_required: bool,
) -> dict[str, object]:
    selected = None if full_required else 4096
    value = {
        "schema_version": "apertus_greekmmlu_sentinel_calibration_v1",
        "status": "passed",
        "scale": scale,
        "decision_state": "full_panel_required" if full_required else "4096_pass",
        "selected_size": selected,
        "selection_authorized": not full_required,
        "full_panel_required": full_required,
    }
    write_json_atomic(path, value)
    return value


def joint_authority(
    path: Path,
    *,
    scale: str,
    calibration_result: Path,
    full_required: bool,
) -> Path:
    write_json_atomic(
        path,
        {
            "schema_version": "apertus_greekmmlu_sentinel_calibration_authority_v1",
            "status": "passed",
            "scope": "both_scales",
            "calibrations": {
                scale: {"canonical_result": file_binding(calibration_result)}
            },
            "cross_scale_trajectory": {
                "mode": "full_clean" if full_required else "sentinel_pair",
                "selected_size": None if full_required else 4096,
            },
        },
    )
    return path


def common_argv(
    *,
    run_root: Path,
    output: Path,
    evaluator_id: str,
    iteration: int,
    scale: str = "8b",
) -> list[str]:
    output.mkdir(parents=True)
    return [
        "program",
        "--scale",
        scale,
        "--iteration",
        str(iteration),
        "--run-root",
        str(run_root),
        "--output",
        str(output),
        "--code-root",
        str(ROOT.parents[1]),
        "--code-receipt",
        str(write_text(output.parent / "code.json")),
        "--campaign-id",
        "campaign",
        "--evaluator-id",
        evaluator_id,
        "--attempt",
        "1",
        "--contract-digest",
        "digest",
    ]


def test_calibration_binds_all_full_panel_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    for update in CALIBRATION_UPDATES:
        receipt = frozen_evaluation(
            tmp_path / f"gmmlu-{update}",
            scale="8b",
            iteration=update,
            mode="full_clean",
            views=("full_clean",),
        )
        canonical_result(
            run_root,
            evaluator_id="greekmmlu",
            iteration=update,
            scale="8b",
            schema="apertus_hard_h_to_g_greekmmlu_evaluation_v1",
            payload={"mode": "full_clean", "evaluation": file_binding(receipt)},
        )
    output = tmp_path / "attempt"
    sentinel = write_text(tmp_path / "sentinel.json")
    eval_venv = tmp_path / "venv"

    def fake_run(argv: list[str], **_kwargs: object) -> SimpleNamespace:
        output_path = Path(argv[argv.index("--output") + 1])
        calibration_payload(output_path, scale="8b", full_required=False)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("run_greekmmlu_calibration_evaluator.subprocess.run", fake_run)
    argv = common_argv(
        run_root=run_root,
        output=output,
        evaluator_id="greekmmlu_calibration",
        iteration=3218,
    ) + ["--sentinel-manifest", str(sentinel), "--eval-venv", str(eval_venv)]
    monkeypatch.setattr(sys, "argv", argv)
    assert calibration_main() == 0
    result = read_json(output / "result.json")
    assert result["decision_state"] == "4096_pass"
    assert sorted(map(int, result["source_evaluations"])) == CALIBRATION_UPDATES


@pytest.mark.parametrize("full_required", [False, True])
def test_fallback_obeys_calibration_terminal_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    full_required: bool,
) -> None:
    run_root = tmp_path / "run"
    calibration = tmp_path / "calibration.json"
    calibration_payload(calibration, scale="8b", full_required=full_required)
    calibration_result = canonical_result(
        run_root,
        evaluator_id="greekmmlu_calibration",
        iteration=3218,
        scale="8b",
        schema="apertus_hard_h_to_g_greekmmlu_calibration_evaluation_v1",
        payload={
            "calibration": file_binding(calibration),
            "decision_state": "full_panel_required" if full_required else "4096_pass",
        },
    )
    joint = joint_authority(
        tmp_path / "joint.json",
        scale="8b",
        calibration_result=calibration_result,
        full_required=full_required,
    )
    source_receipt = frozen_evaluation(
        tmp_path / "sentinel-source",
        scale="8b",
        iteration=952,
        mode="sentinel_pair",
        views=("sentinel_4096",),
    )
    canonical_result(
        run_root,
        evaluator_id="greekmmlu",
        iteration=952,
        scale="8b",
        schema="apertus_hard_h_to_g_greekmmlu_evaluation_v1",
        payload={
            "mode": "sentinel_pair",
            "evaluation": file_binding(source_receipt),
            "checkpoint_export": file_binding(write_text(tmp_path / "export.json")),
            "summary": file_binding(write_text(tmp_path / "summary.json")),
        },
    )
    calls = []

    def fake_execute(**_kwargs: object) -> dict[str, object]:
        calls.append(True)
        return {
            "checkpoint_export": file_binding(write_text(tmp_path / "full-export.json")),
            "evaluation": file_binding(write_text(tmp_path / "full-evaluation.json")),
            "summary": file_binding(write_text(tmp_path / "full-summary.json")),
            "views": {"full_clean": file_binding(write_text(tmp_path / "full.jsonl"))},
        }

    monkeypatch.setattr("run_greekmmlu_fallback_evaluator.execute_frozen_greekmmlu", fake_execute)
    output = tmp_path / "attempt"
    argv = common_argv(
        run_root=run_root,
        output=output,
        evaluator_id="greekmmlu_full_fallback_i0000952",
        iteration=3218,
    ) + [
        "--target-iteration",
        "952",
        "--clean-examples",
        str(write_text(tmp_path / "clean.json")),
        "--sentinel-manifest",
        str(write_text(tmp_path / "sentinel.json")),
        "--eval-venv",
        str(tmp_path / "venv"),
        "--joint-calibration-authority",
        str(joint),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert fallback_main() == 0
    result = read_json(output / "result.json")
    assert bool(calls) is full_required
    assert result["action"] == (
        "full_panel_scored" if full_required else "sentinel_authorized_no_full_score_required"
    )


def test_plateau_uses_full_trajectory_when_calibration_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    calibration = tmp_path / "calibration.json"
    calibration_payload(calibration, scale="8b", full_required=True)
    calibration_result = canonical_result(
        run_root,
        evaluator_id="greekmmlu_calibration",
        iteration=3218,
        scale="8b",
        schema="apertus_hard_h_to_g_greekmmlu_calibration_evaluation_v1",
        payload={"calibration": file_binding(calibration)},
    )
    joint = joint_authority(
        tmp_path / "joint.json",
        scale="8b",
        calibration_result=calibration_result,
        full_required=True,
    )

    def fake_selected(
        _run_root: Path, *, iteration: int, **_kwargs: object
    ) -> tuple[Path, dict[str, float], dict[str, object]]:
        path = write_text(tmp_path / f"pred-{iteration}.jsonl")
        return path, {"a": float(3694 - iteration), "b": float(3695 - iteration)}, {"path": str(path)}

    monkeypatch.setattr("run_greekmmlu_plateau_evaluator.selected_predictions", fake_selected)
    monkeypatch.setattr(
        "run_greekmmlu_plateau_evaluator.bootstrap_mean",
        lambda values, _n, _seed: (0.1, min(values), max(values)),
    )
    output = tmp_path / "attempt"
    argv = common_argv(
        run_root=run_root,
        output=output,
        evaluator_id="greekmmlu_plateau_confirmation",
        iteration=3694,
    ) + [
        "--clean-examples",
        str(write_text(tmp_path / "clean.json")),
        "--sentinel-manifest",
        str(write_text(tmp_path / "sentinel.json")),
        "--eval-venv",
        str(tmp_path / "venv"),
        "--joint-calibration-authority",
        str(joint),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert plateau_main() == 0
    result = read_json(output / "result.json")
    assert result["trajectory_view"] == "full_clean"
    assert result["minimum_choice_nll_iteration"] == 3694
    assert result["confirmation_action"] == "full_panel_trajectory_already_required"


def test_trajectory_driver_isolates_checkpoint_table_stdin() -> None:
    driver = (
        ROOT / "operational_workarounds/run_greekmmlu_trajectory_in_allocation.sh"
    ).read_text(encoding="utf-8")
    assert 'bash "$export_script" </dev/null' in driver
    assert 'bash "$score_script" </dev/null' in driver
