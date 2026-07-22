from __future__ import annotations

import json
from pathlib import Path

import pytest

from sequence_models.bibliography_evolution_prepare_g1_full import (
    EXPECTED_FAMILY_COUNTS,
    candidate_store_snapshot,
    select_full_g1_templates,
)


EVOLUTION_ROOT = Path(__file__).resolve().parents[1] / "evolution"


def test_selects_exact_complete_predeclared_g1_inventory() -> None:
    packet = json.loads(
        (EVOLUTION_ROOT / "experiment_templates.json").read_text(encoding="utf-8")
    )
    selected = select_full_g1_templates(packet["templates"])
    assert [row["parameter_family"] for row in selected] == list(
        EXPECTED_FAMILY_COUNTS
    )


def _candidate(root: Path, name: str, generation: str, family: str, job: str) -> None:
    path = root / name
    path.mkdir()
    (path / "spec.json").write_text(
        json.dumps({"generation": generation, "parameter_family": family}),
        encoding="utf-8",
    )
    (path / "execution.json").write_text(
        json.dumps({"slurm_job_id": job, "slurm_array_task_id": "0"}),
        encoding="utf-8",
    )


def test_candidate_snapshot_proves_no_g2_execution(tmp_path: Path) -> None:
    _candidate(tmp_path, "g0-a", "G0", "baseline", "1")
    for index in range(5):
        _candidate(tmp_path, f"g1-{index}", "G1", "anchor_threshold", str(2 + index))
    snapshot = candidate_store_snapshot(tmp_path)
    assert snapshot["generation_counts"] == {"G0": 1, "G1": 5}
    assert snapshot["g2_candidate_count"] == snapshot["g2_execution_count"] == 0

    _candidate(tmp_path, "g2-bad", "G2", "headers", "9")
    with pytest.raises(RuntimeError, match="candidate store"):
        candidate_store_snapshot(tmp_path)
