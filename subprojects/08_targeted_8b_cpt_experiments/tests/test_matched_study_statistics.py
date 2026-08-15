from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "evaluation"))

from finalize_matched_study_evidence import (  # noqa: E402
    SCALES,
    UPDATES,
    decision_same_sign,
    make_selection,
    metric_trajectory,
    spearman,
    spearman_gate,
)


def write_panel(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "doc_id": "a",
            "nll_numerator_nats": value,
            "utf8_bytes": 10,
        },
        {
            "doc_id": "b",
            "nll_numerator_nats": value * 1.2,
            "utf8_bytes": 12,
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_metric_bootstrap_is_paired_across_scales_and_updates(tmp_path: Path) -> None:
    documents = {scale: {} for scale in SCALES}
    for scale in SCALES:
        for index, update in enumerate(UPDATES):
            path = tmp_path / scale / f"{update}.jsonl"
            write_panel(path, 20.0 - index)
            documents[scale][update] = {"hplt": path}
    points, boot = metric_trajectory(
        documents, ("hplt",), replicates=128, seed=9
    )
    assert boot["8b"].shape == (128, len(UPDATES))
    assert np.array_equal(boot["8b"], boot["1p5b"])
    assert np.array_equal(points["8b"], points["1p5b"])
    assert np.all(np.diff(points["8b"]) < 0)


def test_same_sign_gate_distinguishes_agreement_and_opposition() -> None:
    negative = np.full(100, -1.0)
    assert decision_same_sign(negative, negative)["decision"] == "pass"
    assert decision_same_sign(negative, -negative)["decision"] == "fail"
    observed = decision_same_sign(
        negative,
        negative,
        point_left=-2.0,
        point_right=-3.0,
    )
    assert observed["8b"]["point"] == -2.0
    assert observed["1p5b"]["point"] == -3.0


def test_spearman_gate_uses_observed_point_not_bootstrap_mean() -> None:
    left = np.asarray([[0.0, 1.0, 3.0], [0.0, 1.0, 3.0]])
    right = np.asarray([[0.0, 1.0, 3.0], [0.0, 1.0, 3.0]])
    point_left = np.asarray([0.0, 1.0, 3.0])
    point_right = np.asarray([0.0, 2.0, 1.0])
    result = spearman_gate(
        left,
        right,
        point_left=point_left,
        point_right=point_right,
    )
    assert result["point"] == spearman(np.diff(point_left), np.diff(point_right))
    assert result["point"] != spearman(
        np.diff(left.mean(axis=0)), np.diff(right.mean(axis=0))
    )


def test_goal_b_passes_for_matching_nonstationary_trajectories() -> None:
    replicates = 200
    index = np.arange(len(UPDATES), dtype=np.float64)
    curve = 4.0 - 0.03 * index - 0.001 * index * index
    offsets = np.linspace(-0.01, 0.01, replicates)[:, None]
    matched_boot = {
        scale: np.repeat(curve[None, :], replicates, axis=0) + offsets
        for scale in SCALES
    }
    metrics = (
        "hplt_bpb",
        "openarchives_macro_bpb",
        "foreign_replay_macro_bpb",
        "old_greek_bpb",
        "neutral_external_modern_greek_bpb",
        "balanced_greek_bpb",
    )
    source_boot = {metric: matched_boot for metric in metrics}
    source_points = {
        metric: {scale: curve.copy() for scale in SCALES} for metric in metrics
    }
    result = make_selection(
        source_boot=source_boot,
        source_points=source_points,
        greek_boot=matched_boot,
        greek_points={scale: curve.copy() for scale in SCALES},
        plateaus={"8b": [3218, 3456], "1p5b": [3218]},
        legacy={"decision": "pass"},
    )
    assert result["goal_b"]["decision"] == "pass"
    assert result["goal_b"]["plateau"]["intersection"] == [3218]
