#!/usr/bin/env python3
"""Authorize a nested GreekMMLU sentinel or require full-panel scoring."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from contract_utils import executing_code_bundle, file_binding, read_json, require, write_json_atomic  # noqa: E402


CALIBRATION_WINDOWS = {
    "early": [0, 238, 476, 714],
    "late": [2618, 2856, 3094, 3218],
}
CALIBRATION_UPDATES = sorted({update for values in CALIBRATION_WINDOWS.values() for update in values})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("8b", "1p5b"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prediction", action="append", required=True, help="UPDATE=JSONL")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260814)
    parser.add_argument("--expected-full-count", type=int, default=16_159)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            require(isinstance(value, dict), f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def choice_nll(row: dict[str, Any]) -> float:
    scores = [float(item["avg_logprob"]) for item in row["choice_scores"]]
    answer = int(row["answer_index"])
    require(0 <= answer < len(scores), "answer index drift")
    maximum = max(scores)
    log_sum_exp = maximum + math.log(sum(math.exp(value - maximum) for value in scores))
    return log_sum_exp - scores[answer]


def prediction_map(path: Path, expected_count: int) -> dict[str, float]:
    rows = read_jsonl(path)
    result = {str(row["example_id"]): choice_nll(row) for row in rows}
    require(len(result) == len(rows) == expected_count, f"{path}: prediction count drift")
    return result


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_mean(values: list[float], replicates: int, seed: int) -> tuple[float, float, float]:
    require(values and replicates > 1, "invalid bootstrap input")
    try:
        import numpy as np  # type: ignore
    except ImportError:
        rng = random.Random(seed)
        means = [statistics.fmean(rng.choices(values, k=len(values))) for _ in range(replicates)]
    else:
        array = np.asarray(values, dtype=np.float64)
        rng = np.random.default_rng(seed)
        means_array = np.empty(replicates, dtype=np.float64)
        batch = max(1, min(128, 16_000_000 // len(array)))
        for start in range(0, replicates, batch):
            width = min(batch, replicates - start)
            indices = rng.integers(0, len(array), size=(width, len(array)), endpoint=False)
            means_array[start : start + width] = array[indices].mean(axis=1)
        means = means_array.tolist()
    return statistics.pstdev(means), percentile(means, 0.025), percentile(means, 0.975)


def evaluate_size(
    ids: set[str],
    predictions: dict[int, dict[str, float]],
    updates: list[int],
    tau: float,
    replicates: int,
    seed: int,
) -> tuple[bool, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    passed = True
    for pair_index, (left, right) in enumerate(zip(updates, updates[1:])):
        full_ids = sorted(predictions[left])
        full_values = [predictions[right][key] - predictions[left][key] for key in full_ids]
        subset_values = [predictions[right][key] - predictions[left][key] for key in sorted(ids)]
        full_delta = statistics.fmean(full_values)
        subset_delta = statistics.fmean(subset_values)
        full_se, full_low, full_high = bootstrap_mean(full_values, replicates, seed + pair_index * 10)
        subset_se, subset_low, subset_high = bootstrap_mean(subset_values, replicates, seed + pair_index * 10 + 1)
        full_sign_required = full_low > 0 or full_high < 0
        sign_match = (not full_sign_required) or (full_delta > 0) == (subset_delta > 0)
        resolution_pass = subset_se <= tau
        pair_pass = sign_match and resolution_pass
        passed = passed and pair_pass
        rows.append(
            {
                "updates": [left, right],
                "full_delta": full_delta,
                "full_bootstrap_se": full_se,
                "full_ci95": [full_low, full_high],
                "subset_delta": subset_delta,
                "subset_bootstrap_se": subset_se,
                "subset_ci95": [subset_low, subset_high],
                "full_sign_required": full_sign_required,
                "sign_match": sign_match,
                "resolution_target": tau,
                "resolution_pass": resolution_pass,
                "passed": pair_pass,
            }
        )
    return passed, rows


def main() -> int:
    args = parse_args()
    require(not args.output.exists(), f"immutable output exists: {args.output}")
    require(args.bootstrap_replicates == 10_000, "production bootstrap replicate drift")
    require(args.bootstrap_seed == 20260814, "production bootstrap seed drift")
    manifest = read_json(args.manifest)
    require(manifest.get("schema_version") == "apertus_greekmmlu_sentinel_manifest_v1", "sentinel manifest schema drift")
    require(manifest.get("status") == "frozen" and manifest.get("selection_authorized") is False, "sentinel freeze state drift")
    current_bundle = executing_code_bundle()
    manifest_bundle = manifest.get("executing_code_bundle")
    require(
        isinstance(manifest_bundle, dict)
        and manifest_bundle.get("root") == current_bundle["root"]
        and manifest_bundle.get("tree_sha256") == current_bundle["tree_sha256"],
        "sentinel manifest code-bundle drift",
    )
    predictions: dict[int, dict[str, float]] = {}
    bindings: dict[str, Any] = {}
    for raw in args.prediction:
        update_text, path_text = raw.split("=", 1)
        update = int(update_text)
        path = Path(path_text)
        require(update not in predictions, f"duplicate prediction update: {update}")
        predictions[update] = prediction_map(path, args.expected_full_count)
        bindings[str(update)] = file_binding(path)
    require(sorted(predictions) == CALIBRATION_UPDATES, "calibration update set drift")
    reference_ids = set(predictions[CALIBRATION_UPDATES[0]])
    require(all(set(value) == reference_ids for value in predictions.values()), "prediction id sets drift")
    panel_ids: dict[int, set[str]] = {}
    for size in manifest["sizes"]:
        panel_path = Path(manifest["panels"][str(size)]["path"])
        require(file_binding(panel_path) == manifest["panels"][str(size)], f"sentinel {size} binding drift")
        ids = {str(row["example_id"]) for row in read_jsonl(panel_path)}
        require(len(ids) == int(size) and ids <= reference_ids, f"sentinel {size} id drift")
        panel_ids[int(size)] = ids

    windows: dict[str, Any] = {}
    for window_index, (name, updates) in enumerate(CALIBRATION_WINDOWS.items()):
        full_deltas = [
            abs(statistics.fmean(predictions[right][key] - predictions[left][key] for key in reference_ids))
            for left, right in zip(updates, updates[1:])
        ]
        tau = 0.5 * statistics.median(full_deltas)
        require(tau > 0 and math.isfinite(tau), f"{name} trajectory resolution target is not positive/finite")
        evaluations: dict[str, Any] = {}
        for size, ids in panel_ids.items():
            size_passed, pairs = evaluate_size(
                ids,
                predictions,
                updates,
                tau,
                args.bootstrap_replicates,
                args.bootstrap_seed + window_index * 100_000 + size,
            )
            evaluations[str(size)] = {"passed": size_passed, "pairs": pairs}
        windows[name] = {
            "updates": updates,
            "resolution_target": tau,
            "resolution_formula": "0.5 * median(abs(adjacent_full_choice_nll_delta))",
            "evaluations": evaluations,
        }

    jointly_passing = [
        size for size in sorted(panel_ids)
        if all(windows[name]["evaluations"][str(size)]["passed"] for name in CALIBRATION_WINDOWS)
    ]
    selected_size = jointly_passing[0] if jointly_passing else None

    decision_state = f"{selected_size}_pass" if selected_size is not None else "full_panel_required"
    payload = {
        "schema_version": "apertus_greekmmlu_sentinel_calibration_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "executing_code_bundle": current_bundle,
        "scale": args.scale,
        "manifest": file_binding(args.manifest),
        "predictions": bindings,
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "seed": args.bootstrap_seed,
            "interval": "percentile_95",
            "paired_by_example_id": True,
        },
        "windows": windows,
        "authorization_requires_both_early_and_late_tests": True,
        "jointly_passing_sizes": jointly_passing,
        "decision_state": decision_state,
        "selected_size": selected_size,
        "selection_authorized": selected_size is not None,
        "full_panel_required": selected_size is None,
    }
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
