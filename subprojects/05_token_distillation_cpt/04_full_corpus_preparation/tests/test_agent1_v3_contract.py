from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


PHASE = Path(__file__).resolve().parents[1]
SCRIPT = PHASE / "scripts" / "agent1_v3_contract.py"


def _write(path: Path, payload: str = "fixture\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def _freeze_args(tmp_path: Path) -> list[str]:
    files = {name: _write(tmp_path / "inputs" / f"{name}.json") for name in (
        "source_registry",
        "source_aliases",
        "candidate_roster",
        "acquisition_receipt",
        "tokenizer",
        "review_policy",
        "policy",
    )}
    return [
        sys.executable,
        str(SCRIPT),
        "freeze-run",
        "--run-root",
        str(tmp_path / "run"),
        "--run-id",
        "agent1-full-corpus-v3-20260712T123456Z-abcdef0",
        "--code-commit",
        "abcdef0123456789",
        "--source-registry",
        str(files["source_registry"]),
        "--source-aliases",
        str(files["source_aliases"]),
        "--candidate-roster",
        str(files["candidate_roster"]),
        "--acquisition-receipt",
        str(files["acquisition_receipt"]),
        "--tokenizer",
        str(files["tokenizer"]),
        "--review-policy",
        str(files["review_policy"]),
        "--dedup-policy",
        str(files["policy"]),
        "--greekmmlu-policy",
        str(files["policy"]),
        "--anonymization-policy",
        str(files["policy"]),
        "--structural-policy",
        str(files["policy"]),
        "--prestructural-only",
    ]


def _run_id() -> str:
    return "agent1-full-corpus-v3-20260712T123456Z-abcdef0"


def test_contract_allows_phase0_evidence_then_freezes_once(tmp_path: Path) -> None:
    phase0 = tmp_path / "run" / "phase0"
    _write(phase0 / "merged_acquisition_receipt.json", "merged\n")
    result = subprocess.run(_freeze_args(tmp_path), check=True, capture_output=True, text=True)
    contract = json.loads((tmp_path / "run" / "run_contract.json").read_text())

    assert json.loads(result.stdout)["ok"] is True
    assert contract["prestructural_only"] is True
    assert contract["stage_graph"][0] == "10-normalize"
    assert contract["contract_sha256"]

    second = subprocess.run(_freeze_args(tmp_path), capture_output=True, text=True)
    assert second.returncode != 0
    assert "already exists" in second.stderr


def test_contract_rejects_out_of_order_stage_and_completes_attempt(tmp_path: Path) -> None:
    subprocess.run(_freeze_args(tmp_path), check=True)
    root = tmp_path / "run"
    common = ["--run-root", str(root), "--run-id", _run_id()]

    early = subprocess.run(
        [sys.executable, str(SCRIPT), "begin-stage", *common, "--stage", "20-lineage", "--attempt-id", "1"],
        capture_output=True,
        text=True,
    )
    assert early.returncode != 0
    assert "stage receipt" in early.stderr

    subprocess.run(
        [sys.executable, str(SCRIPT), "begin-stage", *common, "--stage", "10-normalize", "--attempt-id", "1"],
        check=True,
    )
    output = _write(root / "stages" / "10-normalize" / "attempts" / "1" / "normalization_manifest.json")
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "finish-stage",
            *common,
            "--stage",
            "10-normalize",
            "--attempt-id",
            "1",
            "--output",
            str(output),
        ],
        check=True,
    )
    assert (root / "stages" / "10-normalize" / "COMPLETED").is_file()

    subprocess.run(
        [sys.executable, str(SCRIPT), "begin-stage", *common, "--stage", "20-lineage", "--attempt-id", "2"],
        check=True,
    )


def test_contract_detects_bound_input_drift(tmp_path: Path) -> None:
    subprocess.run(_freeze_args(tmp_path), check=True)
    source = tmp_path / "inputs" / "source_registry.json"
    source.write_text("drift\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "validate-run", "--run-root", str(tmp_path / "run"), "--run-id", _run_id()],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "bound input drift" in result.stderr


def test_outputs_must_be_attempt_local(tmp_path: Path) -> None:
    subprocess.run(_freeze_args(tmp_path), check=True)
    root = tmp_path / "run"
    common = ["--run-root", str(root), "--run-id", _run_id()]
    subprocess.run(
        [sys.executable, str(SCRIPT), "begin-stage", *common, "--stage", "10-normalize", "--attempt-id", "1"],
        check=True,
    )
    output = _write(tmp_path / "elsewhere.json")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "finish-stage",
            *common,
            "--stage",
            "10-normalize",
            "--attempt-id",
            "1",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "job-unique attempt" in result.stderr
