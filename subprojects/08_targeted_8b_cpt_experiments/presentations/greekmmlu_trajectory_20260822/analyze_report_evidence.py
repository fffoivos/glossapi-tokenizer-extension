#!/usr/bin/env python3
"""Derive auditable report summaries from the frozen trajectory evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
UPDATES = [238, 476, 714, 952, 1190, 1428, 1666, 1904, 2142, 2261,
           2380, 2618, 2856, 3094, 3218, 3456, 3694]
LANDMARKS = {"first": 238, "hplt_boundary": 2261, "oa_endpoint": 3218, "final": 3694}


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phase(update: int) -> str:
    if update <= 2261:
        return "hplt"
    if update <= 3218:
        return "openarchives"
    return "extension"


def trajectory_analysis(aggregate: dict, training: dict) -> dict:
    rows = {
        scale: {int(row["update"]): row for row in aggregate["rows"] if row["scale"] == scale}
        for scale in ("1p5b", "8b")
    }
    if any(sorted(values) != UPDATES for values in rows.values()):
        raise ValueError("checkpoint grid drift")
    metrics = ("accuracy", "choice_nll", "correct_answer_bpb")
    scales = {}
    for scale, values in rows.items():
        metric_summary = {}
        for metric in metrics:
            landmarks = {name: float(values[update][metric]) for name, update in LANDMARKS.items()}
            best_row = (max if metric == "accuracy" else min)(values.values(), key=lambda row: float(row[metric]))
            metric_summary[metric] = {
                "landmarks": landmarks,
                "best": {"update": int(best_row["update"]), "value": float(best_row[metric])},
                "phase_deltas": {
                    "hplt_first_to_boundary": landmarks["hplt_boundary"] - landmarks["first"],
                    "openarchives_boundary_to_endpoint": landmarks["oa_endpoint"] - landmarks["hplt_boundary"],
                    "extension_endpoint_to_final": landmarks["final"] - landmarks["oa_endpoint"],
                    "overall_first_to_final": landmarks["final"] - landmarks["first"],
                    "final_minus_best": landmarks["final"] - float(best_row[metric]),
                },
            }

        levels = {}
        level_names = sorted(values[UPDATES[0]]["by_educational_level"])
        for name in level_names:
            series = [(update, float(values[update]["by_educational_level"][name]["accuracy"])) for update in UPDATES]
            peak_update, peak_value = max(series, key=lambda pair: pair[1])
            levels[name] = {
                "n": int(values[UPDATES[0]]["by_educational_level"][name]["n"]),
                "first": series[0][1],
                "peak_update": peak_update,
                "peak_phase": phase(peak_update),
                "peak": peak_value,
                "final": series[-1][1],
                "final_minus_first": series[-1][1] - series[0][1],
                "final_minus_peak": series[-1][1] - peak_value,
            }

        subject_names = sorted(values[UPDATES[0]]["by_subject"])
        subjects = {}
        peak_phase_counts = {"hplt": 0, "openarchives": 0, "extension": 0}
        for name in subject_names:
            series = [(update, float(values[update]["by_subject"][name]["accuracy"])) for update in UPDATES]
            peak_update, peak_value = max(series, key=lambda pair: pair[1])
            peak_phase_counts[phase(peak_update)] += 1
            subjects[name] = {
                "n": int(values[UPDATES[0]]["by_subject"][name]["n"]),
                "first": series[0][1],
                "peak_update": peak_update,
                "peak_phase": phase(peak_update),
                "peak": peak_value,
                "final": series[-1][1],
                "final_minus_first": series[-1][1] - series[0][1],
                "final_minus_peak": series[-1][1] - peak_value,
            }

        validation = {}
        for panel, series_raw in training["scales"][scale]["validation_loss"].items():
            series = [(int(update), float(value)) for update, value in series_raw]
            by_update = dict(series)
            required = (25, 2250, 3200, 3675)
            if any(update not in by_update for update in required):
                raise ValueError(f"missing validation landmark for {scale}/{panel}")
            minimum_update, minimum = min(series, key=lambda pair: pair[1])
            validation[panel] = {
                "first": by_update[25],
                "hplt_boundary": by_update[2250],
                "oa_endpoint": by_update[3200],
                "final": by_update[3675],
                "minimum": minimum,
                "minimum_update": minimum_update,
                "final_forgetting": by_update[3675] - minimum,
                "hplt_delta": by_update[2250] - by_update[25],
                "openarchives_delta": by_update[3200] - by_update[2250],
                "extension_delta": by_update[3675] - by_update[3200],
            }

        scales[scale] = {
            "metrics": metric_summary,
            "educational_levels": levels,
            "subjects": subjects,
            "subject_peak_phase_counts": peak_phase_counts,
            "subjects_final_above_first": sum(item["final"] > item["first"] for item in subjects.values()),
            "subjects_final_below_first": sum(item["final"] < item["first"] for item in subjects.values()),
            "validation_loss": validation,
        }

    gaps = {}
    for update in UPDATES:
        gaps[str(update)] = {
            metric: float(rows["8b"][update][metric]) - float(rows["1p5b"][update][metric])
            for metric in metrics
        }
    agreements = [float(row["answer_correctness_agreement"]) for row in aggregate["cross_scale_question_comparisons"]]
    return {
        "schema_version": "apertus_h2g_cross_scale_report_analysis_v1",
        "status": "completed",
        "source_aggregate_sha256": sha256(EVIDENCE / "trajectory_aggregate.json"),
        "source_training_trajectories_sha256": sha256(EVIDENCE / "training_trajectories.json"),
        "scales": scales,
        "cross_scale": {
            "shape": aggregate["cross_scale_shape"],
            "metric_gaps_8b_minus_1p5b": gaps,
            "correctness_agreement_min": min(agreements),
            "correctness_agreement_max": max(agreements),
            "correctness_agreement_mean": sum(agreements) / len(agreements),
        },
    }


def parity_audit() -> dict:
    paths = sorted(EVIDENCE.glob("export_receipts/**/checkpoint_export_receipt.json"))
    paths += sorted(EVIDENCE.glob("export_receipts_external/**/checkpoint_export_receipt.json"))
    rows = []
    for path in paths:
        receipt = load(path)
        conversion = receipt.get("conversion") or {}
        mapping_path = path.with_name("exact_weight_mapping_receipt.json")
        mapping = load(mapping_path)
        mapping_passed = (
            mapping.get("status") == "passed"
            and mapping.get("all_hf_tensors_accounted_for") is True
            and mapping.get("all_mapped_parameter_tensors_bit_exact") is True
            and mapping.get("all_source_parameters_covered") is True
        )
        rows.append({
            "scale": receipt["scale"],
            "iteration": int(receipt["iteration"]),
            "schema_version": receipt.get("schema_version"),
            "ready_for_frozen_evaluators": bool(receipt.get("ready_for_frozen_evaluators")),
            "ready_for_trajectory_evaluator": bool(receipt.get("ready_for_trajectory_evaluator")),
            "scope": receipt.get("scope"),
            "exact_weight_mapping_passed": mapping_passed,
            "exact_weight_mapping_receipt_sha256": sha256(mapping_path),
            "runtime_semantic_parity_passed": conversion.get("runtime_semantic_parity_passed"),
            "diagnostics_complete": conversion.get("diagnostics_complete"),
            "prediction_agreement_percent": conversion.get("prediction_agreement_percent"),
            "logits_close_percent": conversion.get("logits_close_percent"),
            "receipt_sha256": sha256(path),
            "local_path": str(path.relative_to(ROOT)),
        })
    identities = {(row["scale"], row["iteration"]) for row in rows}
    if len(rows) != 34 or identities != {(scale, update) for scale in ("1p5b", "8b") for update in UPDATES}:
        raise ValueError("export receipt grid drift")
    if not all(row["exact_weight_mapping_passed"] for row in rows):
        raise ValueError("exact weight mapping failure")
    trajectory_only = [row for row in rows if not row["ready_for_frozen_evaluators"]]
    prediction = [float(row["prediction_agreement_percent"]) for row in rows if row["prediction_agreement_percent"] is not None]
    return {
        "schema_version": "apertus_h2g_export_parity_audit_v1",
        "status": "completed",
        "receipt_count": len(rows),
        "exact_weight_mapping_pass_count": sum(row["exact_weight_mapping_passed"] for row in rows),
        "frozen_evaluator_ready_count": sum(row["ready_for_frozen_evaluators"] for row in rows),
        "trajectory_only_count": len(trajectory_only),
        "trajectory_only_checkpoints": [
            {"scale": row["scale"], "iteration": row["iteration"]} for row in trajectory_only
        ],
        "explicit_incomplete_diagnostics_count": sum(row["diagnostics_complete"] is False for row in rows),
        "prediction_agreement_percent_min": min(prediction),
        "prediction_agreement_percent_max": max(prediction),
        "rows": sorted(rows, key=lambda row: (row["iteration"], row["scale"])),
    }


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    aggregate = load(EVIDENCE / "trajectory_aggregate.json")
    training = load(EVIDENCE / "training_trajectories.json")
    write(EVIDENCE / "analysis_summary.json", trajectory_analysis(aggregate, training))
    write(EVIDENCE / "export_parity_audit.json", parity_audit())
    print(EVIDENCE / "analysis_summary.json")
    print(EVIDENCE / "export_parity_audit.json")


if __name__ == "__main__":
    main()
