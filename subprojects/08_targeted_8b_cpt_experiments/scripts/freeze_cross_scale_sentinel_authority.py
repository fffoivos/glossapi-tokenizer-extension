#!/usr/bin/env python3
"""Freeze the joint 8B/1.5B early-and-late GreekMMLU sentinel decision."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

from contract_utils import (
    executing_code_bundle,
    file_binding,
    read_json,
    require,
    require_file_binding,
    write_json_atomic,
)

WINDOWS = {
    "early": [0, 238, 476, 714],
    "late": [2618, 2856, 3094, 3218],
}
UPDATES = sorted({update for values in WINDOWS.values() for update in values})
DECISIONS = {"4096_pass", "8192_pass", "full_panel_required"}


def select_cross_scale_trajectory(
    calibrations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    require(set(calibrations) == {"8b", "1p5b"}, "cross-scale calibration set drift")
    common_passing_sizes = sorted(
        set(calibrations["8b"].get("jointly_passing_sizes", []))
        & set(calibrations["1p5b"].get("jointly_passing_sizes", []))
    )
    if common_passing_sizes:
        return {
            "mode": "sentinel_pair",
            "selected_size": common_passing_sizes[-1],
            "common_passing_sizes": common_passing_sizes,
            "full_fallback_scales": [],
        }
    return {
        "mode": "full_clean",
        "selected_size": None,
        "common_passing_sizes": [],
        "full_fallback_scales": ["8b", "1p5b"],
    }


def validate_calibration(
    path: Path,
    *,
    scale: str,
    manifest_binding: dict[str, Any],
    current_bundle: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = read_json(path)
    source_binding = file_binding(path)
    if source.get("schema_version") == "apertus_hard_h_to_g_greekmmlu_calibration_evaluation_v1":
        require(
            source.get("status") == "completed"
            and source.get("scale") == scale
            and int(source.get("iteration", -1)) == 3218,
            f"{scale}: canonical calibration result drift",
        )
        calibration_path = require_file_binding(source["calibration"])
        value = read_json(calibration_path)
        require(
            source.get("decision_state") == value.get("decision_state")
            and source.get("selected_size") == value.get("selected_size")
            and source.get("full_panel_required") == value.get("full_panel_required"),
            f"{scale}: canonical calibration projection drift",
        )
        provenance = {
            "canonical_result": source_binding,
            "calibration": file_binding(calibration_path),
        }
    else:
        value = source
        provenance = {"calibration": source_binding}
    require(
        value.get("schema_version") == "apertus_greekmmlu_sentinel_calibration_v1",
        f"{scale}: sentinel calibration schema drift",
    )
    require(value.get("status") == "passed" and value.get("scale") == scale, f"{scale}: calibration identity drift")
    require(value.get("manifest") == manifest_binding, f"{scale}: sentinel manifest binding drift")
    require(value.get("authorization_requires_both_early_and_late_tests") is True, f"{scale}: joint-window authorization drift")
    require(value.get("decision_state") in DECISIONS, f"{scale}: invalid sentinel decision")
    require(
        value.get("selection_authorized") is (value.get("decision_state") != "full_panel_required")
        and value.get("full_panel_required") is (value.get("decision_state") == "full_panel_required"),
        f"{scale}: decision flags disagree",
    )
    jointly_passing = value.get("jointly_passing_sizes")
    require(
        isinstance(jointly_passing, list)
        and jointly_passing == sorted(set(jointly_passing))
        and set(jointly_passing) <= {4096, 8192}
        and value.get("selected_size")
        == (jointly_passing[0] if jointly_passing else None),
        f"{scale}: jointly passing sentinel inventory drift",
    )
    bundle = value.get("executing_code_bundle")
    require(
        isinstance(bundle, dict)
        and bundle.get("root") == current_bundle["root"]
        and bundle.get("tree_sha256") == current_bundle["tree_sha256"],
        f"{scale}: calibration code-bundle drift",
    )
    windows = value.get("windows")
    require(isinstance(windows, dict) and set(windows) == set(WINDOWS), f"{scale}: calibration windows drift")
    for name, updates in WINDOWS.items():
        row = windows[name]
        require(isinstance(row, dict) and row.get("updates") == updates, f"{scale}: {name} update window drift")
        evaluations = row.get("evaluations")
        require(isinstance(evaluations, dict) and set(evaluations) == {"4096", "8192"}, f"{scale}: {name} panel set drift")
        for size in (4096, 8192):
            evaluation = evaluations[str(size)]
            require(isinstance(evaluation, dict) and isinstance(evaluation.get("passed"), bool), f"{scale}: {name}/{size} result missing")
            pairs = evaluation.get("pairs")
            require(isinstance(pairs, list) and len(pairs) == 3, f"{scale}: {name}/{size} pair count drift")
            require(all(isinstance(pair, dict) and isinstance(pair.get("passed"), bool) for pair in pairs), f"{scale}: {name}/{size} pair result drift")
    predictions = value.get("predictions")
    require(isinstance(predictions, dict) and sorted(map(int, predictions)) == UPDATES, f"{scale}: full-panel prediction set drift")
    return value, provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--8b-calibration", type=Path, required=True)
    parser.add_argument("--1p5b-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable sentinel authority exists: {args.output}")
    manifest = read_json(args.manifest)
    require(manifest.get("schema_version") == "apertus_greekmmlu_sentinel_manifest_v1", "sentinel manifest schema drift")
    require(manifest.get("status") == "frozen" and manifest.get("sizes") == [4096, 8192], "sentinel manifest state drift")
    current_bundle = executing_code_bundle()
    manifest_bundle = manifest.get("executing_code_bundle")
    require(
        isinstance(manifest_bundle, dict)
        and manifest_bundle.get("root") == current_bundle["root"]
        and manifest_bundle.get("tree_sha256") == current_bundle["tree_sha256"],
        "sentinel manifest code-bundle drift",
    )
    manifest_binding = file_binding(args.manifest)
    validated = {
        "8b": validate_calibration(
            args.__dict__["8b_calibration"], scale="8b",
            manifest_binding=manifest_binding, current_bundle=current_bundle,
        ),
        "1p5b": validate_calibration(
            args.__dict__["1p5b_calibration"], scale="1p5b",
            manifest_binding=manifest_binding, current_bundle=current_bundle,
        ),
    }
    calibrations = {scale: row[0] for scale, row in validated.items()}
    cross_scale_trajectory = select_cross_scale_trajectory(calibrations)
    payload = {
        "schema_version": "apertus_greekmmlu_sentinel_calibration_authority_v1",
        "status": "passed",
        "scope": "both_scales",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "executing_code_bundle": current_bundle,
        "manifest": manifest_binding,
        "calibrations": {
            scale: {
                **validated[scale][1],
                "decision_state": value["decision_state"],
                "selected_size": value.get("selected_size"),
                "full_panel_required": value["full_panel_required"],
            }
            for scale, value in calibrations.items()
        },
        "calibration_windows": WINDOWS,
        "full_panel_updates_per_scale": UPDATES,
        "cross_scale_trajectory": cross_scale_trajectory,
        "invariants": {
            "both_scales_calibrated_independently": True,
            "both_early_and_late_windows_passed_through_decision_logic": True,
            "same_frozen_sentinel_manifest_used": True,
            "full_panel_fallback_is_an_authorized_terminal_state": True,
            "cross_scale_statistics_use_one_shared_question_panel": True,
            "absence_of_a_jointly_passing_sentinel_requires_full_panels_for_both_scales": True,
        },
    }
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
