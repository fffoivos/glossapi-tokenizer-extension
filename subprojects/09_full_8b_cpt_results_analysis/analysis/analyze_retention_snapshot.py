#!/usr/bin/env python3
"""Freeze source-conditioned learning and retention alerts from one train log."""

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def consecutive_at_least(values: list[float], threshold: float) -> int:
    current = maximum = 0
    for value in values:
        current = current + 1 if value >= threshold else 0
        maximum = max(maximum, current)
    return maximum


def analyze_rows(rows: list[dict], recipe: dict, *, end_update: int) -> dict:
    evaluation = recipe["evaluation"]
    expected_panels = tuple(evaluation["source_conditioned"]["panels"])
    alert = evaluation["retention_alerts"]
    retention_panels = tuple(alert["panels"])
    reference_start = int(alert["reference"]["start_update"])
    selected: dict[tuple[int, str], dict] = {}
    duplicate_rows = 0
    for row in rows:
        iteration = int(row["iteration"])
        panel = str(row["panel"])
        if panel not in expected_panels or iteration > end_update:
            continue
        if not math.isfinite(float(row["lm_loss"])) or not math.isfinite(
            float(row["bpb"])
        ):
            raise ValueError("non-finite source-conditioned validation metric")
        key = (iteration, panel)
        previous = selected.get(key)
        if previous is not None:
            duplicate_rows += 1
            if any(
                abs(float(previous[name]) - float(row[name])) > 1.0e-9
                for name in ("lm_loss", "bpb")
            ):
                raise ValueError(f"conflicting duplicate validation result: {key}")
            continue
        selected[key] = row
    by_panel: dict[str, list[dict]] = defaultdict(list)
    for row in selected.values():
        by_panel[row["panel"]].append(row)
    if set(by_panel) != set(expected_panels):
        raise ValueError("source-conditioned panel inventory drift")
    iterations = sorted({iteration for iteration, _panel in selected})
    incomplete = {
        iteration: sorted(
            set(expected_panels)
            - {panel for row_iteration, panel in selected if row_iteration == iteration}
        )
        for iteration in iterations
    }
    incomplete = {key: value for key, value in incomplete.items() if value}
    if incomplete:
        raise ValueError(f"incomplete source-conditioned points: {incomplete}")
    if not iterations or iterations[-1] != end_update:
        raise ValueError("source-conditioned trajectory lacks the segment endpoint")

    summaries: dict[str, dict] = {}
    deltas_by_iteration: dict[int, list[float]] = defaultdict(list)
    warning_limit = float(alert["warning"]["any_panel_increase_nats"])
    critical_limit = float(alert["critical"]["any_panel_increase_nats"])
    required_consecutive = int(alert["critical"]["consecutive_observations"])
    warning_panels: list[str] = []
    critical_panels: list[str] = []
    for panel in expected_panels:
        points = sorted(by_panel[panel], key=lambda row: int(row["iteration"]))
        eligible = [row for row in points if int(row["iteration"]) >= reference_start]
        if not eligible:
            raise ValueError(f"panel lacks post-reference observations: {panel}")
        running_minimum = math.inf
        deltas: list[float] = []
        for row in eligible:
            loss = float(row["lm_loss"])
            running_minimum = min(running_minimum, loss)
            delta = loss - running_minimum
            deltas.append(delta)
            if panel in retention_panels:
                deltas_by_iteration[int(row["iteration"])].append(delta)
        warning_run = consecutive_at_least(deltas, warning_limit)
        critical_run = consecutive_at_least(deltas, critical_limit)
        if panel in retention_panels and warning_run >= required_consecutive:
            warning_panels.append(panel)
        if panel in retention_panels and critical_run >= required_consecutive:
            critical_panels.append(panel)
        summaries[panel] = {
            "points": len(points),
            "first_iteration": int(points[0]["iteration"]),
            "last_iteration": int(points[-1]["iteration"]),
            "first_lm_loss": float(points[0]["lm_loss"]),
            "last_lm_loss": float(points[-1]["lm_loss"]),
            "last_bpb": float(points[-1]["bpb"]),
            "best_lm_loss_after_reference": min(
                float(row["lm_loss"]) for row in eligible
            ),
            "final_delta_from_running_minimum": deltas[-1],
            "maximum_delta_from_running_minimum": max(deltas),
            "maximum_consecutive_warning_observations": warning_run,
            "maximum_consecutive_critical_observations": critical_run,
        }

    macro = [
        {
            "iteration": iteration,
            "mean_delta_nats": sum(values) / len(values),
        }
        for iteration, values in sorted(deltas_by_iteration.items())
        if len(values) == len(retention_panels)
    ]
    macro_limit = float(alert["critical"]["macro_mean_increase_nats"])
    macro_critical_run = consecutive_at_least(
        [row["mean_delta_nats"] for row in macro], macro_limit
    )
    critical = bool(critical_panels) or macro_critical_run >= required_consecutive
    warning = bool(warning_panels)
    return {
        "status": "critical" if critical else "warning" if warning else "no_alert",
        "iterations": iterations,
        "complete_points": len(iterations),
        "row_count": len(selected),
        "identical_duplicate_rows_ignored": duplicate_rows,
        "panel_summaries": summaries,
        "retention": {
            "panels": list(retention_panels),
            "warning_panels": warning_panels,
            "critical_panels": critical_panels,
            "macro_trajectory": macro,
            "final_macro_delta_nats": macro[-1]["mean_delta_nats"],
            "maximum_macro_delta_nats": max(
                row["mean_delta_nats"] for row in macro
            ),
            "maximum_consecutive_macro_critical_observations": macro_critical_run,
            "automatic_training_stop": bool(alert["action"]["automatic_training_stop"]),
        },
    }


def load_parser(path: Path):
    spec = importlib.util.spec_from_file_location("validation_trajectory_parser", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_log


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--trajectory-parser", type=Path, required=True)
    parser.add_argument("--segment-id", type=int, required=True)
    parser.add_argument("--source-train-job", required=True)
    parser.add_argument("--end-update", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = (args.log, args.recipe, args.trajectory_parser)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    parse_log = load_parser(args.trajectory_parser)
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    analysis = analyze_rows(parse_log(args.log), recipe, end_update=args.end_update)
    payload = {
        "schema_version": "apertus_full_8b_retention_snapshot_v1",
        "status": "completed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "segment_id": args.segment_id,
        "source_train_job": args.source_train_job,
        "end_update": args.end_update,
        "bindings": {
            "log": {"path": str(args.log.resolve()), "sha256": sha256_file(args.log)},
            "recipe": {
                "path": str(args.recipe.resolve()),
                "sha256": sha256_file(args.recipe),
            },
            "trajectory_parser": {
                "path": str(args.trajectory_parser.resolve()),
                "sha256": sha256_file(args.trajectory_parser),
            },
        },
        "analysis": analysis,
    }
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    descriptor, temporary = tempfile.mkstemp(
        dir=args.output.parent, prefix=f".{args.output.name}.", suffix=".partial"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, args.output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(json.dumps({
        "ok": True,
        "status": analysis["status"],
        "complete_points": analysis["complete_points"],
        "final_macro_delta_nats": analysis["retention"]["final_macro_delta_nats"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
