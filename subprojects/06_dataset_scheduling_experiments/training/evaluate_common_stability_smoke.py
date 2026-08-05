#!/usr/bin/env python3
"""Fail closed on non-finite or divergent common 1,024-step LR smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
from pathlib import Path


ITERATION = re.compile(r"iteration\s+(\d+)\s*/")
METRIC = re.compile(r"(?:^|\|)\s*([^|:]+?)\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
VALIDATION = re.compile(r"validation loss at .*?\[([^\]]+)\]", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def median_window(values: list[float], *, first: bool) -> float:
    window = values[:64] if first else values[-64:]
    return statistics.median(window)


def normalize_metric(value: str) -> str:
    return " ".join(value.lower().split())


def parse_validation(path: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    lines = [
        re.sub(r"^\d+:\s?", "", line)
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
    ]
    for index, line in enumerate(lines):
        header = VALIDATION.search(line)
        if header is None:
            continue
        name = header.group(1)
        block = line
        for continuation in lines[index + 1 :]:
            if VALIDATION.search(continuation) or re.fullmatch(r"\s*-{10,}\s*", continuation):
                break
            block += continuation
        metrics = {normalize_metric(key): float(raw) for key, raw in METRIC.findall(block)}
        lm_keys = [key for key in metrics if key in {"lm loss", "lm loss value"}]
        if len(lm_keys) != 1:
            raise ValueError(f"{path}: validation panel {name!r} has no unique LM loss")
        row = {"lm_loss": metrics[lm_keys[0]]}
        for source, target in (
            ("base-token target loss value", "base_target_loss"),
            ("added-token target loss value", "added_target_loss"),
            ("base-token target count value", "base_target_count"),
            ("added-token target count value", "added_target_count"),
        ):
            if source in metrics:
                row[target] = metrics[source]
        if name in result:
            raise ValueError(f"{path}: duplicate terminal validation panel {name!r}")
        result[name] = row
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver-log", type=Path, required=True)
    parser.add_argument("--initial-validation-log", type=Path, required=True)
    parser.add_argument("--endpoint-validation-log", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--peak-lr", type=float, required=True)
    parser.add_argument("--process-exit-code", type=int, required=True)
    parser.add_argument("--max-retention-relative-regression", type=float, default=0.05)
    parser.add_argument("--max-any-panel-relative-regression", type=float, default=0.10)
    parser.add_argument("--max-added-target-relative-regression", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    losses = []
    grad_norms = []
    iterations = set()
    for line in args.driver_log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ITERATION.search(line)
        if match:
            iterations.add(int(match.group(1)))
            for key, raw in METRIC.findall(line):
                normalized = normalize_metric(key)
                value = float(raw)
                if normalized == "lm loss":
                    losses.append(value)
                elif normalized == "grad norm":
                    grad_norms.append(value)
    validation_manifest = json.loads(args.validation_manifest.read_text())
    expected_panels = [row["name"] for row in validation_manifest.get("panels", [])]
    if len(expected_panels) != 13 or len(set(expected_panels)) != 13:
        raise ValueError("validation manifest must contain 13 unique panels")
    initial_validation = parse_validation(args.initial_validation_log)
    final_validation = parse_validation(args.endpoint_validation_log)
    validation_complete = (
        set(initial_validation) == set(expected_panels)
        and set(final_validation) == set(expected_panels)
    )
    retention_names = {"code", "de", "english", "math", "ru", "zh", "old_greek", "historical_polytonic"}
    if not retention_names.issubset(expected_panels):
        raise ValueError("validation manifest lacks required retention panels")
    relative_changes = {}
    for name in expected_panels:
        before = initial_validation.get(name, {}).get("lm_loss", math.inf)
        after = final_validation.get(name, {}).get("lm_loss", math.inf)
        relative_changes[name] = (after - before) / before if before > 0 and math.isfinite(before) else math.inf
    validation_finite = all(
        math.isfinite(row.get("lm_loss", math.inf))
        for rows in (initial_validation, final_validation)
        for row in rows.values()
    )
    retention_ok = all(
        relative_changes[name] <= args.max_retention_relative_regression
        for name in retention_names
    )
    all_panels_stable = all(
        value <= args.max_any_panel_relative_regression
        for value in relative_changes.values()
    )
    added_probe_names = {"hplt", "non_hplt", "greek_phd", "historical_polytonic"}
    added_target_rows_complete = added_probe_names.issubset(initial_validation) and all(
        initial_validation[name].get("added_target_count", 0) > 0
        and final_validation[name].get("added_target_count", 0) > 0
        and math.isfinite(initial_validation[name].get("added_target_loss", math.inf))
        and math.isfinite(final_validation[name].get("added_target_loss", math.inf))
        for name in added_probe_names
    )
    added_target_stable = added_target_rows_complete and all(
        final_validation[name]["added_target_loss"]
        <= initial_validation[name]["added_target_loss"]
        * (1.0 + args.max_added_target_relative_regression)
        for name in added_probe_names
    )
    complete = args.checkpoint_root / "iter_0001024" / ".metadata"
    finite = all(math.isfinite(value) for value in losses + grad_norms)
    enough = len(losses) >= 900 and max(iterations or {0}) >= 1_024
    first_loss = median_window(losses, first=True) if losses else math.inf
    final_loss = median_window(losses, first=False) if losses else math.inf
    loss_ratio = final_loss / first_loss if first_loss > 0 and math.isfinite(first_loss) else math.inf
    max_grad = max(grad_norms) if grad_norms else math.inf
    checks = {
        "process_exit_zero": args.process_exit_code == 0,
        "iteration_1024_checkpoint_complete": complete.is_file(),
        "at_least_900_logged_losses_and_reached_1024": enough,
        "loss_and_gradient_metrics_finite": finite,
        "final_64_loss_median_at_most_1p5x_first_64": loss_ratio <= 1.5,
        "maximum_gradient_norm_below_1000": max_grad < 1_000.0,
        "all_13_baseline_and_endpoint_validation_panels_present": validation_complete,
        "all_validation_losses_finite": validation_finite,
        "retention_panels_within_predeclared_relative_margin": retention_ok,
        "every_panel_within_predeclared_catastrophic_regression_margin": all_panels_stable,
        "added_token_target_loss_present_and_stable_on_four_greek_probes": added_target_stable,
    }
    payload = {
        "schema_version": "apertus_mini_common_stability_smoke_v2",
        "status": "passed" if all(checks.values()) else "failed",
        "peak_lr": args.peak_lr,
        "optimizer_updates": 1_024,
        "active_schedule": "fixed balanced HPLT/non-HPLT prefix with 20% foreign and 1% Old-Greek replay",
        "checks": checks,
        "diagnostics": {
            "logged_loss_count": len(losses),
            "logged_gradient_norm_count": len(grad_norms),
            "first_64_loss_median": first_loss,
            "final_64_loss_median": final_loss,
            "final_over_first_loss_median": loss_ratio,
            "maximum_gradient_norm": max_grad,
            "maximum_logged_iteration": max(iterations or {0}),
            "validation_relative_lm_loss_changes": relative_changes,
        },
        "validation": {
            "expected_panels": expected_panels,
            "initial": initial_validation,
            "endpoint": final_validation,
            "retention_panels": sorted(retention_names),
            "added_token_probe_panels": sorted(added_probe_names),
            "thresholds": {
                "max_retention_relative_regression": args.max_retention_relative_regression,
                "max_any_panel_relative_regression": args.max_any_panel_relative_regression,
                "max_added_target_relative_regression": args.max_added_target_relative_regression,
            },
        },
        "driver_log": {"path": str(args.driver_log.resolve()), "sha256": sha256_file(args.driver_log), "bytes": args.driver_log.stat().st_size},
        "initial_validation_log": {"path": str(args.initial_validation_log.resolve()), "sha256": sha256_file(args.initial_validation_log), "bytes": args.initial_validation_log.stat().st_size},
        "endpoint_validation_log": {"path": str(args.endpoint_validation_log.resolve()), "sha256": sha256_file(args.endpoint_validation_log), "bytes": args.endpoint_validation_log.stat().st_size},
        "validation_manifest": {"path": str(args.validation_manifest.resolve()), "sha256": sha256_file(args.validation_manifest), "bytes": args.validation_manifest.stat().st_size},
        "checkpoint_root": str(args.checkpoint_root.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output) + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"ok": all(checks.values()), "peak_lr": args.peak_lr, "checks": checks}, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
