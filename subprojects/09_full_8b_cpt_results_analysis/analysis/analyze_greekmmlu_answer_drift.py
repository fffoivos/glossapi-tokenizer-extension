#!/usr/bin/env python3
"""Analyze clean GreekMMLU answer-set drift across exact 8B checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_receipt(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def load_predictions(path: Path, clean_ids: set[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            example_id = str(row["example_id"])
            if example_id in clean_ids:
                if example_id in rows:
                    raise ValueError(f"duplicate example id in {path}: {example_id}")
                rows[example_id] = row
    if set(rows) != clean_ids:
        raise ValueError(
            f"clean example set drift in {path}: missing={len(clean_ids-set(rows))} "
            f"unexpected={len(set(rows)-clean_ids)}"
        )
    return rows


def pct(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def checkpoint_record(iteration: int, rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    correct = sum(bool(row["correct"]) for row in rows.values())
    return {"iteration": iteration, "n": len(rows), "correct": correct, "accuracy": correct / len(rows)}


def transitions(
    current: dict[str, dict[str, Any]], previous: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    keys = current.keys()
    gained = sum(not bool(previous[k]["correct"]) and bool(current[k]["correct"]) for k in keys)
    lost = sum(bool(previous[k]["correct"]) and not bool(current[k]["correct"]) for k in keys)
    retained = sum(bool(previous[k]["correct"]) and bool(current[k]["correct"]) for k in keys)
    stable_wrong = len(current) - gained - lost - retained
    flipped = sum(int(previous[k]["pred_index"]) != int(current[k]["pred_index"]) for k in keys)
    previous_correct = retained + lost
    current_correct = retained + gained
    union = retained + gained + lost
    return {
        "newly_correct": gained,
        "newly_wrong": lost,
        "retained_correct": retained,
        "stable_wrong": stable_wrong,
        "net_correct_change": gained - lost,
        "answer_choice_flips": flipped,
        "answer_choice_flip_rate": pct(flipped, len(current)),
        "correct_set_churn_rate": pct(gained + lost, len(current)),
        "prior_correct_retention": pct(retained, previous_correct),
        "correct_set_jaccard": pct(retained, union),
        "previous_correct": previous_correct,
        "current_correct": current_correct,
    }


def grouped_accuracy(
    checkpoints: list[tuple[int, dict[str, dict[str, Any]]]], key_fn
) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    all_names = sorted({str(key_fn(row) or "Unknown") for _, rows in checkpoints for row in rows.values()})
    for name in all_names:
        trajectory = []
        for iteration, rows in checkpoints:
            selected = [row for row in rows.values() if str(key_fn(row) or "Unknown") == name]
            correct = sum(bool(row["correct"]) for row in selected)
            trajectory.append(
                {"iteration": iteration, "n": len(selected), "correct": correct, "accuracy": pct(correct, len(selected))}
            )
        best = max(trajectory, key=lambda row: (row["accuracy"], -row["iteration"]))
        first, final = trajectory[0], trajectory[-1]
        groups[name] = {
            "n": final["n"],
            "trajectory": trajectory,
            "initial_accuracy": first["accuracy"],
            "best_accuracy": best["accuracy"],
            "best_iteration": best["iteration"],
            "final_accuracy": final["accuracy"],
            "final_minus_best": final["accuracy"] - best["accuracy"],
            "final_minus_initial": final["accuracy"] - first["accuracy"],
        }
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--initial-predictions", type=Path, required=True)
    parser.add_argument("--initial-receipt", type=Path, required=True)
    parser.add_argument("--clean-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    clean_ids = {line.strip() for line in args.clean_ids.read_text(encoding="utf-8").splitlines() if line.strip()}
    if len(clean_ids) != 16_159:
        raise ValueError(f"clean GreekMMLU count drift: {len(clean_ids)}")

    initial_receipt = read_json(args.initial_receipt)
    initial_artifact = initial_receipt.get("artifacts", {}).get(args.initial_predictions.name)
    if not isinstance(initial_artifact, dict):
        raise ValueError("initial receipt does not bind the selected prediction payload")
    if (
        args.initial_predictions.stat().st_size != int(initial_artifact.get("bytes", -1))
        or sha256_file(args.initial_predictions) != initial_artifact.get("sha256")
    ):
        raise ValueError("initial GreekMMLU prediction payload drift")

    prediction_sources: list[tuple[int, Path, Path]] = [(0, args.initial_predictions, args.initial_receipt)]
    evaluation_root = args.run_root / "checkpoint_evaluations"
    for directory in sorted(evaluation_root.glob("iter_*")):
        authority_path = directory / "authoritative_attempt.json"
        if not authority_path.is_file():
            continue
        authority = read_json(authority_path)
        if authority.get("status") != "completed":
            continue
        receipt_path = Path(authority["greekmmlu_receipt"])
        receipt = read_json(receipt_path)
        iteration = int(receipt["checkpoint"]["iteration"])
        prediction_path = Path(receipt["artifacts"]["predictions"]["path"])
        expected_hash = str(receipt["artifacts"]["predictions"]["sha256"])
        if sha256_file(prediction_path) != expected_hash:
            raise ValueError(f"prediction hash drift: {prediction_path}")
        prediction_sources.append((iteration, prediction_path, receipt_path))
    prediction_sources.sort(key=lambda item: item[0])
    iterations = [item[0] for item in prediction_sources]
    expected = [0, 400, 1192, 2384, 3576, 4768, 5960, 7152, 8344, 9536, 10728, 11920, 13112, 14304, 14627, 15496, 16688, 17880, 18284]
    if iterations != expected:
        raise ValueError(f"checkpoint coverage drift: {iterations}")

    checkpoints: list[tuple[int, dict[str, dict[str, Any]]]] = []
    bindings = []
    for iteration, predictions, receipt in prediction_sources:
        checkpoints.append((iteration, load_predictions(predictions, clean_ids)))
        bindings.append(
            {
                "iteration": iteration,
                "predictions": file_receipt(predictions),
                "receipt": file_receipt(receipt),
            }
        )

    table = []
    for index, (iteration, rows) in enumerate(checkpoints):
        record = checkpoint_record(iteration, rows)
        if index == 0:
            record["vs_previous"] = None
        else:
            record["vs_previous"] = transitions(rows, checkpoints[index - 1][1])
        table.append(record)

    best_iteration, best_rows = max(checkpoints, key=lambda item: checkpoint_record(item[0], item[1])["accuracy"])
    final_iteration, final_rows = checkpoints[-1]
    initial_rows = checkpoints[0][1]
    for record, (_, rows) in zip(table, checkpoints, strict=True):
        record["vs_initial"] = transitions(rows, initial_rows)
        record["vs_best_accuracy_checkpoint"] = transitions(rows, best_rows)

    correctness_frequency = Counter(
        sum(bool(rows[example_id]["correct"]) for _, rows in checkpoints) for example_id in clean_ids
    )
    persistent = {
        "always_correct": correctness_frequency[len(checkpoints)],
        "never_correct": correctness_frequency[0],
        "transiently_correct": len(clean_ids) - correctness_frequency[len(checkpoints)] - correctness_frequency[0],
        "correct_at_exactly_k_checkpoints": {str(k): correctness_frequency[k] for k in range(len(checkpoints) + 1)},
    }

    subject = grouped_accuracy(checkpoints, lambda row: row.get("subject") or row.get("metadata", {}).get("subject"))
    level = grouped_accuracy(checkpoints, lambda row: row.get("metadata", {}).get("level"))

    subject_final_vs_best = []
    for name, payload in subject.items():
        b = next(row for row in payload["trajectory"] if row["iteration"] == best_iteration)
        f = payload["trajectory"][-1]
        subject_final_vs_best.append(
            {
                "subject": name,
                "n": f["n"],
                "best_checkpoint_accuracy": b["accuracy"],
                "final_accuracy": f["accuracy"],
                "final_minus_best_checkpoint": f["accuracy"] - b["accuracy"],
            }
        )
    subject_final_vs_best.sort(key=lambda row: row["final_minus_best_checkpoint"])

    payload = {
        "schema_version": "full8b_clean_greekmmlu_answer_drift_v1",
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(args.run_root.resolve()),
        "clean_subset": {"n": len(clean_ids), "ids": file_receipt(args.clean_ids)},
        "checkpoint_count": len(checkpoints),
        "best_accuracy_iteration": best_iteration,
        "final_iteration": final_iteration,
        "checkpoint_table": table,
        "best_to_final": transitions(final_rows, best_rows),
        "initial_to_final": transitions(final_rows, initial_rows),
        "correctness_persistence": persistent,
        "subjects": subject,
        "levels": level,
        "subject_final_vs_best": subject_final_vs_best,
        "bindings": bindings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output) + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"ok": True, "output": str(args.output), "checkpoints": len(checkpoints), "n": len(clean_ids)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
