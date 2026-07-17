#!/usr/bin/env python3
"""Describe what the frozen bibliography models see on CONTINUATION lines.

This is deliberately an audit, not another fitted classifier.  It combines the
frozen P0D entry probability, the frozen connector feature table, and the
connector subtype OOF models.  Validation is neither read nor scored.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

import numpy as np

from .bibliography_filler_feature_audit import feature_group
from .bibliography_role_experts import ConnectorBundle
from .contract import sha256_file


SCHEMA_VERSION = "bibliography-continuation-feature-audit-v1"
ENTRY_THRESHOLD = 0.25


class _HistoricalConnectorUnpickler(pickle.Unpickler):
    """Load bundles written by the historical ``python -m`` entry point."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "__main__" and name == "ConnectorBundle":
            return ConnectorBundle
        return super().find_class(module, name)


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


def numeric_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    quantiles = np.quantile(values, (0.10, 0.25, 0.50, 0.75, 0.90))
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std()),
        "minimum": float(values.min()),
        "p10": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
        "maximum": float(values.max()),
    }


def _feature_column(names: Sequence[str], name: str) -> int:
    try:
        return names.index(name)
    except ValueError as error:
        raise ValueError(f"missing connector feature: {name}") from error


def _role_rows(
    *, roles: np.ndarray, trusted: np.ndarray, role_encoding: Mapping[str, int],
) -> dict[str, np.ndarray]:
    return {
        name: np.flatnonzero(trusted & (roles == int(identifier)))
        for name, identifier in role_encoding.items() if name != "UNKNOWN"
    }


def deterministic_profiles(
    *, features: np.ndarray, names: Sequence[str], role_rows: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    result = []
    comparison_roles = ("CONTINUATION", "ENTRY", "FILLER", "OTHER")
    for name in names:
        if not name.startswith("presence:"):
            continue
        base_name = name.removeprefix("presence:")
        presence_column = _feature_column(names, name)
        count_column = _feature_column(names, f"log1p:{base_name}")
        row: dict[str, Any] = {"feature": base_name}
        for role in comparison_roles:
            rows = role_rows[role]
            presence = features[rows, presence_column]
            counts = np.expm1(features[rows, count_column].astype(np.float64))
            row[f"{role.lower()}_presence_rate"] = float(presence.mean())
            row[f"{role.lower()}_mean_count"] = float(counts.mean())
        row["continuation_minus_filler_presence"] = (
            row["continuation_presence_rate"] - row["filler_presence_rate"]
        )
        row["continuation_minus_other_presence"] = (
            row["continuation_presence_rate"] - row["other_presence_rate"]
        )
        entry_rate = row["entry_presence_rate"]
        row["continuation_to_entry_presence_ratio"] = (
            row["continuation_presence_rate"] / entry_rate if entry_rate else None
        )
        result.append(row)
    return sorted(result, key=lambda row: row["continuation_presence_rate"], reverse=True)


def _univariate_subtype(
    *, features: np.ndarray, names: Sequence[str], rows: np.ndarray,
    continuation: np.ndarray,
) -> list[dict[str, Any]]:
    _, roc_auc = _sklearn()
    truth = continuation[rows]
    result = []
    for column, name in enumerate(names):
        values = features[rows, column].astype(np.float64)
        if np.all(values == values[0]):
            continue
        auc = float(roc_auc(truth, values))
        positive, negative = values[truth], values[~truth]
        pooled = float(np.sqrt((positive.var() + negative.var()) / 2.0))
        effect = (
            float((positive.mean() - negative.mean()) / pooled)
            if pooled > 1.0e-12 else 0.0
        )
        result.append({
            "feature": name,
            "group": feature_group(name),
            "oriented_roc_auc": max(auc, 1.0 - auc),
            "continuation_direction": "higher" if auc >= 0.5 else "lower",
            "standardized_mean_difference": effect,
            "continuation_mean": float(positive.mean()),
            "filler_mean": float(negative.mean()),
            "continuation_median": float(np.median(positive)),
            "filler_median": float(np.median(negative)),
        })
    return sorted(
        result,
        key=lambda row: (row["oriented_roc_auc"], abs(row["standardized_mean_difference"])),
        reverse=True,
    )


def _subtype_permutation_audit(
    *, features: np.ndarray, names: Sequence[str], folds: np.ndarray,
    subtype_trusted: np.ndarray, continuation: np.ndarray, model_dir: Path,
    repetitions: int, seed: int,
) -> tuple[float, np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    average_precision, _ = _sklearn()
    payload = []
    baseline_parts = []
    oof_probability = np.full(len(features), np.nan, dtype=np.float32)
    for fold in sorted(int(value) for value in np.unique(folds)):
        rows = np.flatnonzero(subtype_trusted & (folds == fold))
        with (model_dir / "models" / f"fold{fold}.pkl").open("rb") as handle:
            bundle = _HistoricalConnectorUnpickler(handle).load()
        probability = _conditional_continuation(bundle.predict(features[rows]))
        oof_probability[rows] = probability
        score = float(average_precision(continuation[rows], probability))
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
                probability = _conditional_continuation(bundle.predict(permuted))
                score = float(average_precision(continuation[rows], probability))
                numerator += len(rows) * score
                denominator += len(rows)
            scores.append(numerator / denominator)
        return baseline - float(np.mean(scores)), float(np.std(scores))

    grouped: dict[str, list[int]] = defaultdict(list)
    for column, name in enumerate(names):
        grouped[feature_group(name)].append(column)
    group_rows = []
    for group, columns in grouped.items():
        drop, deviation = importance(columns)
        group_rows.append({
            "group": group,
            "feature_count": len(columns),
            "average_precision_drop": drop,
            "permutation_sd": deviation,
        })
    single_rows = []
    for column, name in enumerate(names):
        drop, deviation = importance((column,))
        single_rows.append({
            "feature": name,
            "group": feature_group(name),
            "average_precision_drop": drop,
            "permutation_sd": deviation,
        })
    key = lambda row: row["average_precision_drop"]
    return (
        baseline,
        oof_probability,
        sorted(group_rows, key=key, reverse=True),
        sorted(single_rows, key=key, reverse=True),
    )


def _candidate_metadata(
    *, source_jsonl: Path, row_indices: np.ndarray, expected_line_count: int,
    expected_split: str,
) -> list[dict[str, Any]]:
    if len(row_indices) and not np.all(row_indices[1:] > row_indices[:-1]):
        raise ValueError("candidate row indices must be strictly increasing")
    metadata: list[dict[str, Any] | None] = [None] * len(row_indices)
    cursor = candidate_cursor = 0
    with source_jsonl.open("r", encoding="utf-8") as handle:
        for raw in handle:
            document = json.loads(raw)
            if document["split"] != expected_split:
                continue
            lines = document["lines"]
            document_end = cursor + len(lines)
            while candidate_cursor < len(row_indices):
                global_index = int(row_indices[candidate_cursor])
                if global_index >= document_end:
                    break
                if global_index < cursor:
                    raise ValueError("candidate/source ordering mismatch")
                line = lines[global_index - cursor]
                metadata[candidate_cursor] = {
                    "document_id": document["document_id"],
                    "source": document["source"],
                    "line_id": line["line_id"],
                    "abs_idx": int(line["abs_idx"]),
                    "text": line["text"],
                }
                candidate_cursor += 1
            cursor = document_end
    if cursor != expected_line_count:
        raise ValueError(f"source line count {cursor} != expected {expected_line_count}")
    if candidate_cursor != len(row_indices) or any(value is None for value in metadata):
        raise ValueError("not all candidate rows were rejoined to source lines")
    return [value for value in metadata if value is not None]


def _source_and_document_profile(
    *, rows: np.ndarray, metadata: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    source_counts = Counter(str(metadata[row]["source"]) for row in rows)
    document_counts = Counter(str(metadata[row]["document_id"]) for row in rows)
    ordered = document_counts.most_common()
    return {
        "source_counts": dict(sorted(source_counts.items())),
        "document_count": len(document_counts),
        "top_documents": [
            {"document_id": document_id, "line_count": count, "share": count / len(rows)}
            for document_id, count in ordered[:10]
        ],
        "top_five_document_share": sum(count for _, count in ordered[:5]) / len(rows),
    }


def _examples(
    *, continuation_rows: np.ndarray, metadata: Sequence[Mapping[str, Any]],
    features: np.ndarray, names: Sequence[str], entry_probability: np.ndarray,
    subtype_probability: np.ndarray,
) -> dict[str, list[dict[str, Any]]]:
    previous_score = features[:, _feature_column(names, "joined_previous_entry_probability")]
    next_score = features[:, _feature_column(names, "joined_next_entry_probability")]
    previous_gain = features[:, _feature_column(names, "joined_previous_probability_gain")]
    next_gain = features[:, _feature_column(names, "joined_next_probability_gain")]
    char_length = features[:, _feature_column(names, "char_length")]
    presence_columns = [i for i, name in enumerate(names) if name.startswith("presence:")]

    def record(row: int) -> dict[str, Any]:
        detected = [
            names[column].removeprefix("presence:")
            for column in presence_columns if features[row, column] > 0
        ]
        return {
            **metadata[row],
            "current_entry_probability": float(entry_probability[row]),
            "continuation_probability_given_connector": float(subtype_probability[row]),
            "joined_previous_entry_probability": float(previous_score[row]),
            "joined_next_entry_probability": float(next_score[row]),
            "joined_previous_probability_gain": float(previous_gain[row]),
            "joined_next_probability_gain": float(next_gain[row]),
            "detected_features": detected,
        }

    def take(order: np.ndarray, limit: int = 8) -> list[dict[str, Any]]:
        return [record(int(row)) for row in order[:limit]]

    join_gain = np.maximum(previous_gain, next_gain)
    joined_score = np.maximum(previous_score, next_score)
    rescued = continuation_rows[
        (entry_probability[continuation_rows] < ENTRY_THRESHOLD)
        & (joined_score[continuation_rows] >= ENTRY_THRESHOLD)
    ]
    uncertain = continuation_rows[np.argsort(
        np.abs(subtype_probability[continuation_rows] - 0.5)
    )]
    return {
        "lowest_current_entry_probability": take(
            continuation_rows[np.argsort(entry_probability[continuation_rows])]
        ),
        "highest_current_entry_probability": take(
            continuation_rows[np.argsort(-entry_probability[continuation_rows])]
        ),
        "strongest_join_rescues": take(rescued[np.argsort(-join_gain[rescued])]),
        "shortest_nonempty": take(
            continuation_rows[np.argsort(char_length[continuation_rows])]
        ),
        "most_subtype_uncertain": take(uncertain),
    }


def _prototype_counts(
    *, rows: np.ndarray, features: np.ndarray, names: Sequence[str],
    entry_probability: np.ndarray,
) -> dict[str, dict[str, Any]]:
    def column(name: str) -> np.ndarray:
        return features[rows, _feature_column(names, name)]

    joined_previous = column("joined_previous_entry_probability")
    joined_next = column("joined_next_entry_probability")
    gain_previous = column("joined_previous_probability_gain")
    gain_next = column("joined_next_probability_gain")
    current = entry_probability[rows]
    flags = {
        "self_supporting_at_entry_threshold": current >= ENTRY_THRESHOLD,
        "weak_at_entry_threshold": current < ENTRY_THRESHOLD,
        "join_rescued_by_previous": (current < ENTRY_THRESHOLD) & (joined_previous >= ENTRY_THRESHOLD),
        "join_rescued_by_next": (current < ENTRY_THRESHOLD) & (joined_next >= ENTRY_THRESHOLD),
        "join_rescued_by_either": (current < ENTRY_THRESHOLD) & (np.maximum(joined_previous, joined_next) >= ENTRY_THRESHOLD),
        "join_gain_at_least_0_10": np.maximum(gain_previous, gain_next) >= 0.10,
        "short_at_most_40_chars": column("char_length") <= 40,
        "tiny_at_most_3_chars": column("char_length") <= 3,
        "starts_lowercase": column("starts_lowercase") > 0,
        "previous_line_ends_open": column("previous_pair:left_ends_opening_terminal") > 0,
        "table_row_feature": column("presence:table_row_count") > 0,
        "url_or_doi_feature": (column("presence:url_count") > 0) | (column("presence:doi_count") > 0),
        "page_or_volume_feature": (
            (column("presence:page_marker_count") > 0)
            | (column("presence:page_range_count") > 0)
            | (column("presence:article_page_range_count") > 0)
            | (column("presence:volume_marker_count") > 0)
            | (column("presence:volume_shape_count") > 0)
        ),
        "author_or_name_feature": (
            (column("presence:proper_name_word_count") > 0)
            | (column("presence:inverted_author_count") > 0)
            | (column("presence:direct_author_count") > 0)
            | (column("presence:name_initial_pair_count") > 0)
        ),
    }
    return {
        name: {"count": int(np.count_nonzero(flag)), "share": float(np.mean(flag))}
        for name, flag in flags.items()
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    table_dir = Path(args.table_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    entry_oof_path = Path(args.entry_oof).resolve()
    source_jsonl = Path(args.source_jsonl).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)

    manifest = json.loads((table_dir / "manifest.json").read_text(encoding="utf-8"))
    names = tuple(manifest["feature_names"])
    features = np.load(table_dir / "features.npy", mmap_mode="r", allow_pickle=False)
    folds = np.load(table_dir / "folds.npy", mmap_mode="r", allow_pickle=False)
    roles = np.load(table_dir / "roles.npy", mmap_mode="r", allow_pickle=False)
    row_indices = np.load(table_dir / "row_indices.npy", mmap_mode="r", allow_pickle=False)
    trusted = np.load(table_dir / "trusted.npy", mmap_mode="r", allow_pickle=False).astype(bool)
    subtype_trusted = np.load(
        table_dir / "subtype_trusted.npy", mmap_mode="r", allow_pickle=False
    ).astype(bool)
    continuation = np.load(
        table_dir / "subtype_targets.npy", mmap_mode="r", allow_pickle=False
    ).astype(bool)
    full_entry_probability = np.load(entry_oof_path, mmap_mode="r", allow_pickle=False)
    if features.shape != (len(folds), len(names)) or roles.shape != folds.shape:
        raise ValueError("continuation audit arrays are not aligned")
    if row_indices.max(initial=0) >= len(full_entry_probability):
        raise ValueError("candidate row index exceeds the P0D probability table")
    entry_probability = np.asarray(full_entry_probability[row_indices], dtype=np.float32)
    entry_oof_sha256 = sha256_file(entry_oof_path)
    source_jsonl_sha256 = sha256_file(source_jsonl)
    if entry_oof_sha256 != manifest["inputs"]["entry_oof_sha256"]:
        raise ValueError("P0D probability hash does not match the connector table")
    if source_jsonl_sha256 != manifest["inputs"]["source_sha256"]:
        raise ValueError("source JSONL hash does not match the connector table")

    role_rows = _role_rows(
        roles=roles, trusted=trusted, role_encoding=manifest["role_encoding"]
    )
    subtype_rows = np.flatnonzero(subtype_trusted)
    continuation_rows = role_rows["CONTINUATION"]
    baseline, subtype_probability, group_rows, single_rows = _subtype_permutation_audit(
        features=features, names=names, folds=folds,
        subtype_trusted=subtype_trusted, continuation=continuation,
        model_dir=model_dir, repetitions=args.repetitions, seed=args.seed,
    )
    metadata = _candidate_metadata(
        source_jsonl=source_jsonl, row_indices=row_indices,
        expected_line_count=len(full_entry_probability),
        expected_split=manifest["split"],
    )

    role_score_profiles = {}
    selected_shape_names = (
        "char_length", "token_count", "letter_fraction", "digit_fraction",
        "punctuation_fraction", "whitespace_fraction", "starts_lowercase",
        "starts_uppercase", "starts_digit", "ends_sentence_terminal",
        "gap:unmatched_fraction", "gap:unmatched_prefix_fraction",
        "gap:unmatched_suffix_fraction", "nearest_anchor_above_distance",
        "nearest_anchor_below_distance", "inside_anchor_gap",
        "joined_previous_entry_probability", "joined_previous_probability_gain",
        "joined_previous_distinct_feature_gain",
        "joined_previous_unmatched_fraction_gain",
        "joined_next_entry_probability", "joined_next_probability_gain",
        "joined_next_distinct_feature_gain", "joined_next_unmatched_fraction_gain",
    )
    for role, rows in role_rows.items():
        role_score_profiles[role] = {
            "current_entry_probability": numeric_summary(entry_probability[rows]),
            "entry_probability_rates": {
                f"at_least_{str(threshold).replace('.', '_')}": float(
                    np.mean(entry_probability[rows] >= threshold)
                )
                for threshold in (0.05, 0.10, 0.25, 0.50, 0.75)
            },
            "selected_features": {
                name: numeric_summary(features[rows, _feature_column(names, name)])
                for name in selected_shape_names
            },
        }

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_continuation_feature_audit",
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "validation_opened": False,
        "feature_count": len(names),
        "trusted_subtype_count": int(len(subtype_rows)),
        "continuation_count": int(len(continuation_rows)),
        "filler_count": int(len(role_rows["FILLER"])),
        "oof_continuation_average_precision_fold_weighted": baseline,
        "oof_continuation_probability": numeric_summary(
            subtype_probability[continuation_rows]
        ),
        "permutation_repetitions": args.repetitions,
        "group_permutation": group_rows,
        "single_feature_permutation": single_rows,
        "univariate_continuation_vs_filler": _univariate_subtype(
            features=features, names=names, rows=subtype_rows,
            continuation=continuation,
        ),
        "role_score_profiles": role_score_profiles,
        "deterministic_feature_profiles": deterministic_profiles(
            features=features, names=names, role_rows=role_rows
        ),
        "continuation_prototypes": _prototype_counts(
            rows=continuation_rows, features=features, names=names,
            entry_probability=entry_probability,
        ),
        "continuation_corpus_profile": _source_and_document_profile(
            rows=continuation_rows, metadata=metadata
        ),
        "examples": _examples(
            continuation_rows=continuation_rows, metadata=metadata,
            features=features, names=names, entry_probability=entry_probability,
            subtype_probability=subtype_probability,
        ),
        "inputs": {
            "table_manifest_sha256": sha256_file(table_dir / "manifest.json"),
            "model_report_sha256": sha256_file(model_dir / "report.json"),
            "model_receipt_sha256": sha256_file(model_dir / "receipt.json"),
            "entry_oof_sha256": entry_oof_sha256,
            "source_jsonl_sha256": source_jsonl_sha256,
        },
    }
    _write_json_new(output / "report.json", report)

    top_features = report["deterministic_feature_profiles"][:20]
    lines = [
        "# CONTINUATION feature audit\n\n",
        f"- trusted subtype rows: {len(subtype_rows):,} "
        f"({len(continuation_rows):,} CONTINUATION; {len(role_rows['FILLER']):,} FILLER)\n",
        f"- OOF fold-weighted CONTINUATION PR-AUC given connector: {baseline:.6f}\n",
        "- frozen entry model and connector models: unchanged\n",
        "- validation opened: no\n\n",
        "## Continuation forms measured from frozen features\n\n",
        "| form | lines | share |\n",
        "|---|---:|---:|\n",
    ]
    lines.extend(
        f"| {name} | {value['count']} | {value['share']:.1%} |\n"
        for name, value in report["continuation_prototypes"].items()
    )
    lines.extend((
        "\n## Most common deterministic features\n\n",
        "| feature | continuation | entry | filler | other |\n",
        "|---|---:|---:|---:|---:|\n",
    ))
    lines.extend(
        f"| `{row['feature']}` | {row['continuation_presence_rate']:.1%} | "
        f"{row['entry_presence_rate']:.1%} | {row['filler_presence_rate']:.1%} | "
        f"{row['other_presence_rate']:.1%} |\n"
        for row in top_features
    )
    lines.extend((
        "\n## Feature-group permutation importance\n\n",
        "| group | features | CONTINUATION PR-AUC drop | permutation SD |\n",
        "|---|---:|---:|---:|\n",
    ))
    lines.extend(
        f"| {row['group']} | {row['feature_count']} | "
        f"{row['average_precision_drop']:.6f} | {row['permutation_sd']:.6f} |\n"
        for row in group_rows
    )
    lines.extend((
        "\n## Top individual permutation importances\n\n",
        "| feature | group | CONTINUATION PR-AUC drop |\n",
        "|---|---|---:|\n",
    ))
    lines.extend(
        f"| `{row['feature']}` | {row['group']} | "
        f"{row['average_precision_drop']:.6f} |\n"
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
    parser.add_argument("--entry-oof", required=True)
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repetitions", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("--repetitions must be at least 2")
    run(args)


if __name__ == "__main__":
    main()
