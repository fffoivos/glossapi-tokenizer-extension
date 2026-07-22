from __future__ import annotations

import argparse
import json
from pathlib import Path

from sequence_models.bibliography_entry_validation import freeze


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _metrics(recall: float, precision: float = 0.995) -> dict:
    return {
        "line_precision": precision,
        "token_recall": recall,
        "token_f0_5": 0.9,
        "spurious_blocks_per_zero_block_document": 0.0,
    }


def _inputs(root: Path, *, b1_recall: float) -> dict[str, Path]:
    paths = {
        "line": root / "line",
        "block": root / "block",
        "b1": root / "b1",
        "b2": root / "b2",
    }
    _write(
        paths["line"] / "line_oof_report.json",
        {
            "validation_opened": False,
            "arms": {"L1": {"selected": {"C": 1.0, "threshold": 0.4}}},
        },
    )
    _write(
        paths["block"] / "block_oof_report.json",
        {
            "validation_opened": False,
            "primary_arm": "L1",
            "arms": {
                "L1": {
                    "selected_config": {
                        "anchor_probability": 0.85,
                        "seed_length_limit": 330,
                        "anchors_required": 2,
                        "anchor_window": 3,
                        "maximum_bridge_gap": 2,
                        "inside_probability": 0.25,
                        "adjacent_expansion": 1,
                        "header_window": 2,
                    },
                    "selected_b0_plus_h0_metrics": _metrics(0.80),
                }
            },
        },
    )
    _write(
        paths["b1"] / "b1_oof_report.json",
        {
            "validation_opened": False,
            "selected_arm": "L1",
            "selected_key": "L1:no_header",
            "training": {
                "epochs": 3,
                "learning_rate": 0.08,
                "l2": 0.0001,
                "gradient_clip": 5.0,
            },
            "variants": {"L1:no_header": {"metrics": _metrics(b1_recall)}},
        },
    )
    _write(
        paths["b2"] / "b2_oof_report.json",
        {"validation_opened": False, "b2_required": False},
    )
    return paths


def test_freeze_prefers_simpler_model_within_recall_tolerance(tmp_path: Path) -> None:
    paths = _inputs(tmp_path, b1_recall=0.804)
    output = tmp_path / "frozen"
    result = freeze(
        argparse.Namespace(
            line_oof_dir=str(paths["line"]),
            block_oof_dir=str(paths["block"]),
            b1_oof_dir=str(paths["b1"]),
            b2_oof_dir=str(paths["b2"]),
            output_dir=str(output),
            code_commit="test",
            slurm_job_id="test",
        )
    )
    assert result["selected"]["architecture"] == "B0_H0"
    assert result["status"] == "passed_train_only_safety_gate"
    assert result["validation_opened"] is False


def test_freeze_selects_b1_for_material_recall_gain(tmp_path: Path) -> None:
    paths = _inputs(tmp_path, b1_recall=0.82)
    result = freeze(
        argparse.Namespace(
            line_oof_dir=str(paths["line"]),
            block_oof_dir=str(paths["block"]),
            b1_oof_dir=str(paths["b1"]),
            b2_oof_dir=str(paths["b2"]),
            output_dir=str(tmp_path / "frozen"),
            code_commit="test",
            slurm_job_id="test",
        )
    )
    assert result["selected"]["architecture"] == "B1"
