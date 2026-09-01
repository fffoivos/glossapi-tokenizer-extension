from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts/analyze_retention_snapshot.py"
    spec = importlib.util.spec_from_file_location("retention_snapshot", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recipe() -> dict:
    panels = [
        "hplt", "non_hplt", "openarchives", "greek_phd",
        "historical_polytonic", "english", "de", "ru", "zh", "code",
        "math", "old_greek", "neutral_external_modern_greek",
    ]
    return {
        "evaluation": {
            "source_conditioned": {"panels": panels},
            "retention_alerts": {
                "panels": ["english", "de", "ru", "zh", "code", "math", "old_greek"],
                "reference": {"start_update": 400},
                "warning": {"any_panel_increase_nats": 0.05},
                "critical": {
                    "any_panel_increase_nats": 0.08,
                    "macro_mean_increase_nats": 0.05,
                    "consecutive_observations": 2,
                },
                "action": {"automatic_training_stop": False},
            },
        }
    }


def rows(*, english=(1.0, 1.01, 1.02)) -> list[dict]:
    values = []
    for panel in recipe()["evaluation"]["source_conditioned"]["panels"]:
        losses = english if panel == "english" else (2.0, 1.9, 1.8)
        for iteration, loss in zip((400, 800, 1200), losses):
            values.append({
                "iteration": iteration,
                "panel": panel,
                "lm_loss": loss,
                "bpb": loss / 2,
            })
    return values


def test_no_alert_for_subthreshold_retention_drift() -> None:
    module = load_module()
    result = module.analyze_rows(rows(), recipe(), end_update=1200)
    assert result["status"] == "no_alert"
    assert result["complete_points"] == 3
    assert result["retention"]["warning_panels"] == []


def test_two_consecutive_large_increases_trigger_critical() -> None:
    module = load_module()
    result = module.analyze_rows(
        rows(english=(1.0, 1.09, 1.10)), recipe(), end_update=1200
    )
    assert result["status"] == "critical"
    assert result["retention"]["critical_panels"] == ["english"]


def test_incomplete_panel_point_is_rejected() -> None:
    module = load_module()
    incomplete = rows()[:-1]
    try:
        module.analyze_rows(incomplete, recipe(), end_update=1200)
    except ValueError as error:
        assert "incomplete" in str(error)
    else:
        raise AssertionError("incomplete validation point was accepted")


def test_conflicting_duplicate_metric_is_rejected() -> None:
    module = load_module()
    duplicate = dict(rows()[0])
    duplicate["lm_loss"] += 0.1
    try:
        module.analyze_rows(rows() + [duplicate], recipe(), end_update=1200)
    except ValueError as error:
        assert "conflicting duplicate" in str(error)
    else:
        raise AssertionError("conflicting duplicate validation was accepted")
