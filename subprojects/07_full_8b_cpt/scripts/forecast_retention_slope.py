#!/usr/bin/env python3
"""Fit a read-only retention-drift forecast from fixed validation trajectories.

The forecast is deliberately descriptive: it estimates the pre-cooldown slope
of loss above each panel's running minimum.  It must only be compared within an
identical frozen validation manifest; it is not a model-selection gate and it
does not control training.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_PANELS = ("english", "de", "ru", "zh")
TOKENS_PER_UPDATE = 4_194_304


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def linear_fit(points: list[tuple[float, float]]) -> dict[str, float]:
    if len(points) < 2:
        raise ValueError("at least two complete validation points are required")
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator == 0:
        raise ValueError("forecast points have no token-range variation")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator
    intercept = mean_y - slope * mean_x
    total = sum((y - mean_y) ** 2 for _, y in points)
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in points)
    return {
        "intercept_nats": intercept,
        "slope_nats_per_billion_tokens": slope,
        "slope_nats_per_10b_tokens": slope * 10.0,
        "r_squared": 1.0 if total == 0 else 1.0 - residual / total,
    }


def select_rows(rows: list[dict[str, Any]], panels: tuple[str, ...]) -> dict[int, dict[str, float]]:
    """Select exact loss values and reject ambiguous duplicate log entries."""
    selected: dict[tuple[int, str], float] = {}
    for row in rows:
        panel = str(row["panel"])
        if panel not in panels:
            continue
        iteration = int(row["iteration"])
        loss = float(row["lm_loss"])
        if not math.isfinite(loss):
            raise ValueError(f"non-finite loss at {iteration}/{panel}")
        key = (iteration, panel)
        previous = selected.get(key)
        if previous is not None and abs(previous - loss) > 1.0e-9:
            raise ValueError(f"conflicting duplicate loss at {iteration}/{panel}")
        selected[key] = loss
    grouped: dict[int, dict[str, float]] = defaultdict(dict)
    for (iteration, panel), loss in selected.items():
        grouped[iteration][panel] = loss
    incomplete = {
        iteration: sorted(set(panels) - set(losses))
        for iteration, losses in grouped.items()
        if set(losses) != set(panels)
    }
    if incomplete:
        raise ValueError(f"incomplete panel grid: {incomplete}")
    if not grouped:
        raise ValueError("no requested panel rows were found")
    return dict(grouped)


def running_minimum_gaps(
    grouped: dict[int, dict[str, float]], panels: tuple[str, ...], reference_update: int
) -> dict[str, list[tuple[int, float]]]:
    result: dict[str, list[tuple[int, float]]] = {panel: [] for panel in panels}
    minima = {panel: math.inf for panel in panels}
    for iteration in sorted(grouped):
        if iteration < reference_update:
            continue
        for panel in panels:
            loss = grouped[iteration][panel]
            minima[panel] = min(minima[panel], loss)
            result[panel].append((iteration, loss - minima[panel]))
    if any(not values for values in result.values()):
        raise ValueError("no observations remain after the reference update")
    return result


def build_forecast(
    rows: list[dict[str, Any]],
    *,
    panels: tuple[str, ...] = DEFAULT_PANELS,
    reference_update: int = 400,
    fit_start_update: int = 400,
    fit_end_update: int | None = None,
    forecast_update: int | None = None,
    tokens_per_update: int = TOKENS_PER_UPDATE,
) -> dict[str, Any]:
    """Build the macro and per-panel linear drift forecast from parsed rows."""
    if reference_update < 0 or fit_start_update < reference_update:
        raise ValueError("fit-start must be at or after the running-minimum reference")
    if fit_end_update is not None and fit_end_update < fit_start_update:
        raise ValueError("fit-end must not precede fit-start")
    grouped = select_rows(rows, panels)
    gaps = running_minimum_gaps(grouped, panels, reference_update)
    grid = [iteration for iteration in sorted(grouped) if iteration >= reference_update]
    if any([iteration for iteration, _ in gaps[panel]] != grid for panel in panels):
        raise ValueError("panel checkpoint grids differ")
    macro = [
        (iteration, sum(dict(gaps[panel])[iteration] for panel in panels) / len(panels))
        for iteration in grid
    ]
    stop = fit_end_update if fit_end_update is not None else grid[-1]
    selected = [point for point in macro if fit_start_update <= point[0] <= stop]
    points_b = [(iteration * tokens_per_update / 1e9, gap) for iteration, gap in selected]
    fit = linear_fit(points_b)
    forecast_iteration = forecast_update if forecast_update is not None else stop
    forecast_tokens_b = forecast_iteration * tokens_per_update / 1e9
    forecast_gap = fit["intercept_nats"] + fit["slope_nats_per_billion_tokens"] * forecast_tokens_b
    per_panel = {}
    for panel in panels:
        values = [point for point in gaps[panel] if fit_start_update <= point[0] <= stop]
        per_panel[panel] = linear_fit(
            [(iteration * tokens_per_update / 1e9, gap) for iteration, gap in values]
        )
    return {
        "schema_version": "apertus_full_8b_retention_slope_forecast_v1",
        "metric": "lm_loss_nats_above_panel_running_minimum",
        "panels": list(panels),
        "reference_update": reference_update,
        "fit": {
            "start_update": fit_start_update,
            "end_update": stop,
            "points": len(selected),
            "start_tokens_b": points_b[0][0],
            "end_tokens_b": points_b[-1][0],
            **fit,
        },
        "forecast": {
            "update": forecast_iteration,
            "tokens_b": forecast_tokens_b,
            "macro_gap_nats": forecast_gap,
            "observed_gap_nats": dict(macro).get(forecast_iteration),
        },
        "per_panel_fit": per_panel,
        "macro_trajectory": [
            {"iteration": iteration, "tokens_b": iteration * tokens_per_update / 1e9, "gap_nats": gap}
            for iteration, gap in macro
        ],
        "interpretation": (
            "Descriptive pre-cooldown extrapolation only. Do not extend through an "
            "LR cooldown or compare across different validation manifests."
        ),
    }


def load_parser(path: Path):
    spec = importlib.util.spec_from_file_location("validation_trajectory_parser", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_log


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".partial")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, action="append", required=True)
    parser.add_argument("--trajectory-parser", type=Path, required=True)
    parser.add_argument("--panel", action="append", default=[])
    parser.add_argument("--reference-update", type=int, default=400)
    parser.add_argument("--fit-start-update", type=int, required=True)
    parser.add_argument("--fit-end-update", type=int)
    parser.add_argument("--forecast-update", type=int, required=True)
    parser.add_argument("--tokens-per-update", type=int, default=TOKENS_PER_UPDATE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logs = [path.resolve() for path in args.log]
    for path in (*logs, args.trajectory_parser):
        if not path.is_file():
            raise FileNotFoundError(path)
    panels = tuple(args.panel) if args.panel else DEFAULT_PANELS
    if len(set(panels)) != len(panels):
        raise ValueError("duplicate panel requested")
    parse_log = load_parser(args.trajectory_parser.resolve())
    rows = [row for log in logs for row in parse_log(log)]
    forecast = build_forecast(
        rows,
        panels=panels,
        reference_update=args.reference_update,
        fit_start_update=args.fit_start_update,
        fit_end_update=args.fit_end_update,
        forecast_update=args.forecast_update,
        tokens_per_update=args.tokens_per_update,
    )
    payload = {
        "status": "completed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "bindings": {
            "logs": [{"path": str(path), "sha256": sha256_file(path)} for path in logs],
            "trajectory_parser": {
                "path": str(args.trajectory_parser.resolve()),
                "sha256": sha256_file(args.trajectory_parser),
            },
        },
        "forecast": forecast,
    }
    atomic_write_json(args.output.resolve(), payload)
    print(json.dumps({"ok": True, "fit": forecast["fit"], "forecast": forecast["forecast"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
