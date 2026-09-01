from pathlib import Path
import importlib.util

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "forecast_retention_slope.py"
SPEC = importlib.util.spec_from_file_location("retention_slope_forecast", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def rows_for_linear_gaps():
    rows = []
    # Each panel starts at loss 1.0 then rises by 0.01 per update after update 1.
    for iteration in range(1, 6):
        for offset, panel in enumerate(("english", "de")):
            rows.append({"iteration": iteration, "panel": panel, "lm_loss": 1.0 + offset + 0.01 * (iteration - 1)})
    return rows


def test_exact_linear_macro_forecast():
    result = MODULE.build_forecast(
        rows_for_linear_gaps(),
        panels=("english", "de"),
        reference_update=1,
        fit_start_update=1,
        fit_end_update=5,
        forecast_update=7,
        tokens_per_update=1_000_000_000,
    )
    assert result["fit"]["slope_nats_per_billion_tokens"] == pytest.approx(0.01)
    assert result["fit"]["r_squared"] == pytest.approx(1.0)
    assert result["forecast"]["macro_gap_nats"] == pytest.approx(0.06)
    assert result["forecast"]["observed_gap_nats"] is None


def test_incomplete_panel_grid_is_rejected():
    rows = rows_for_linear_gaps()
    rows.pop()
    with pytest.raises(ValueError, match="incomplete panel grid"):
        MODULE.build_forecast(
            rows,
            panels=("english", "de"),
            reference_update=1,
            fit_start_update=1,
            forecast_update=5,
            tokens_per_update=1_000_000_000,
        )
