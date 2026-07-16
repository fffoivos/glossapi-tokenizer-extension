#!/usr/bin/env python3
"""Audit which connector features actually distinguish FILLER from CONTINUATION."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bibliography_role_experts import ConnectorBundle
from .contract import sha256_file


SCHEMA_VERSION = "bibliography-filler-feature-audit-v1"


class _HistoricalConnectorUnpickler(pickle.Unpickler):
    """Load bundles written by the historical ``python -m`` training entry point."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "__main__" and name == "ConnectorBundle":
            return ConnectorBundle
        return super().find_class(module, name)


def feature_group(name: str) -> str:
    if name.startswith("presence:"):
        return "deterministic_feature_presence"
    if name.startswith("log1p:"):
        return "deterministic_feature_counts"
    if name.startswith("gap:"):
        return "unmatched_character_geometry"
    if name.startswith("nearest_anchor_"):
        return "nearest_entry_anchor"
    if name.startswith("entry_"):
        return "entry_probability_neighbourhoods"
    if name.startswith(("previous_pair:", "next_pair:")):
        return "adjacent_line_shape_pairs"
    if name.startswith("joined_"):
        return "joined_line_entry_gain"
    if "header_probability" in name or name == "heading_probability_max":
        return "heading_probabilities"
    if name in {"inside_anchor_gap", "candidate_window_edge_distance"}:
        return "block_relative_position"
    return "current_line_shape"


def _sklearn() -> tuple[Any, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    return average_precision_score, roc_auc_score


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_text_new(path: Path, value: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


def _conditional_continuation(probability: np.ndarray) -> np.ndarray:
    return np.divide(
        probability[:, 1], probability[:, 0],
        out=np.full(len(probability), 0.5, dtype=np.float32),
        where=probability[:, 0] > 1.0e-7,
    )


def _univariate(
    features: np.ndarray, names: Sequence[str], rows: np.ndarray, filler: np.ndarray,
) -> list[dict[str, Any]]:
    _, roc_auc = _sklearn()
    result = []
    truth = filler[rows]
    for column, name in enumerate(names):
        values = features[rows, column].astype(np.float64)
        if np.all(values == values[0]):
            continue
        auc = float(roc_auc(truth, values))
        filler_values, continuation_values = values[truth], values[~truth]
        pooled = float(np.sqrt((filler_values.var() + continuation_values.var()) / 2.0))
        effect = (
            float((filler_values.mean() - continuation_values.mean()) / pooled)
            if pooled > 1.0e-12 else 0.0
        )
        result.append({
            "feature": name,
            "group": feature_group(name),
            "oriented_roc_auc": max(auc, 1.0 - auc),
            "filler_direction": "higher" if auc >= 0.5 else "lower",
            "standardized_mean_difference": effect,
            "filler_mean": float(filler_values.mean()),
            "continuation_mean": float(continuation_values.mean()),
            "filler_median": float(np.median(filler_values)),
            "continuation_median": float(np.median(continuation_values)),
        })
    return sorted(
        result,
        key=lambda row: (row["oriented_roc_auc"], abs(row["standardized_mean_difference"])),
        reverse=True,
    )


def _permutation_audit(
    *, features: np.ndarray, names: Sequence[str], folds: np.ndarray,
    trusted: np.ndarray, filler: np.ndarray, model_dir: Path, repetitions: int,
    seed: int,
) -> tuple[float, list[dict[str, Any]], list[dict[str, Any]]]:
    average_precision, _ = _sklearn()
    payload = []
    baseline_parts = []
    for fold in sorted(int(value) for value in np.unique(folds)):
        rows = np.flatnonzero(trusted & (folds == fold))
        with (model_dir / "models" / f"fold{fold}.pkl").open("rb") as handle:
            bundle = _HistoricalConnectorUnpickler(handle).load()
        probability = bundle.predict(features[rows])
        filler_probability = 1.0 - _conditional_continuation(probability)
        score = float(average_precision(filler[rows], filler_probability))
        payload.append((rows, features[rows].copy(), bundle))
        baseline_parts.append((len(rows), score))
    baseline = sum(count * score for count, score in baseline_parts) / sum(
        count for count, _ in baseline_parts
    )
    rng = np.random.default_rng(seed)

    def importance(columns: Sequence[int]) -> tuple[float, float]:
        scores = []
        for _ in range(repetitions):
            numerator = denominator = 0.0
            for rows, values, bundle in payload:
                permuted = values.copy()
                order = rng.permutation(len(permuted))
                permuted[:, columns] = permuted[order][:, columns]
                probability = bundle.predict(permuted)
                score = float(average_precision(
                    filler[rows], 1.0 - _conditional_continuation(probability)
                ))
                numerator += len(rows) * score
                denominator += len(rows)
            scores.append(numerator / denominator)
        return baseline - float(np.mean(scores)), float(np.std(scores))

    grouped: dict[str, list[int]] = {}
    for column, name in enumerate(names):
        grouped.setdefault(feature_group(name), []).append(column)
    group_rows = []
    for group, columns in grouped.items():
        drop, deviation = importance(columns)
        group_rows.append({
            "group": group, "feature_count": len(columns),
            "average_precision_drop": drop, "permutation_sd": deviation,
        })
    single_rows = []
    for column, name in enumerate(names):
        drop, deviation = importance((column,))
        single_rows.append({
            "feature": name, "group": feature_group(name),
            "average_precision_drop": drop, "permutation_sd": deviation,
        })
    key = lambda row: row["average_precision_drop"]
    return baseline, sorted(group_rows, key=key, reverse=True), sorted(single_rows, key=key, reverse=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    table_dir, model_dir = Path(args.table_dir).resolve(), Path(args.model_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((table_dir / "manifest.json").read_text(encoding="utf-8"))
    names = tuple(manifest["feature_names"])
    features = np.load(table_dir / "features.npy", mmap_mode="r", allow_pickle=False)
    folds = np.load(table_dir / "folds.npy", mmap_mode="r", allow_pickle=False)
    trusted = np.load(table_dir / "subtype_trusted.npy", mmap_mode="r", allow_pickle=False).astype(bool)
    continuation = np.load(
        table_dir / "subtype_targets.npy", mmap_mode="r", allow_pickle=False
    ).astype(bool)
    if features.shape != (len(folds), len(names)) or trusted.shape != folds.shape:
        raise ValueError("filler audit arrays are not aligned")
    filler = ~continuation
    rows = np.flatnonzero(trusted)
    baseline, group_rows, single_rows = _permutation_audit(
        features=features, names=names, folds=folds, trusted=trusted,
        filler=filler, model_dir=model_dir, repetitions=args.repetitions, seed=args.seed,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_filler_feature_audit",
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "validation_opened": False,
        "feature_count": len(names),
        "trusted_subtype_count": len(rows),
        "filler_count": int(np.count_nonzero(filler[rows])),
        "continuation_count": int(np.count_nonzero(continuation[rows])),
        "oof_filler_average_precision_fold_weighted": baseline,
        "permutation_repetitions": args.repetitions,
        "group_permutation": group_rows,
        "single_feature_permutation": single_rows,
        "univariate": _univariate(features, names, rows, filler),
        "inputs": {
            "table_manifest_sha256": sha256_file(table_dir / "manifest.json"),
            "model_report_sha256": sha256_file(model_dir / "report.json"),
            "model_receipt_sha256": sha256_file(model_dir / "receipt.json"),
        },
    }
    _write_json_new(output / "report.json", report)
    lines = [
        "# FILLER feature audit\n\n",
        f"- trusted subtype rows: {len(rows):,} ({report['filler_count']:,} FILLER; "
        f"{report['continuation_count']:,} CONTINUATION)\n",
        f"- OOF fold-weighted FILLER PR-AUC: {baseline:.6f}\n",
        "- validation opened: no\n\n",
        "## Feature-group permutation importance\n\n",
        "| group | features | PR-AUC drop | permutation SD |\n",
        "|---|---:|---:|---:|\n",
    ]
    lines.extend(
        f"| {row['group']} | {row['feature_count']} | "
        f"{row['average_precision_drop']:.6f} | {row['permutation_sd']:.6f} |\n"
        for row in group_rows
    )
    lines.extend((
        "\n## Top individual permutation importances\n\n",
        "| feature | group | PR-AUC drop |\n",
        "|---|---|---:|\n",
    ))
    lines.extend(
        f"| `{row['feature']}` | {row['group']} | {row['average_precision_drop']:.6f} |\n"
        for row in single_rows[:40]
    )
    _write_text_new(output / "README.md", "".join(lines))
    _write_json_new(output / "receipt.json", {
        **report,
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(output.iterdir()) if path.is_file()
        },
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repetitions", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("--repetitions must be at least 2")
    run(args)


if __name__ == "__main__":
    main()
