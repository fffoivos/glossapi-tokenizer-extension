#!/usr/bin/env python3
"""Compare compact train-only CONTINUATION heads with the frozen cascade."""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contract import sha256_file


SCHEMA_VERSION = "bibliography-continuation-head-ablation-oof-v1"
SELECTED_FAMILIES = {
    "author_name": (
        "initial_count", "proper_name_word_count", "inverted_author_count",
        "name_initial_pair_count", "direct_author_count", "ampersand_count",
    ),
    "date": (
        "year_count", "no_date_count", "numeric_date_count", "month_date_count",
        "access_date_count",
    ),
    "locator": (
        "volume_marker_count", "volume_shape_count", "journal_year_volume_count",
        "page_marker_count", "article_page_range_count", "page_range_count",
    ),
    "identifier": ("url_count", "doi_count", "isbn_count", "issn_count"),
    "container": (
        "editor_term_count", "thesis_term_count", "in_container_count",
        "edition_term_count", "publisher_term_count", "place_name_count",
        "place_publisher_shape_count",
    ),
    "citation_punctuation": (
        "dotted_word_count", "dotted_sequence_count", "quoted_span_count",
        "punctuation_count", "numbered_entry_count",
    ),
    "table": ("table_row_count",),
}
JOIN_NAMES = (
    "joined_previous_entry_probability", "joined_previous_probability_gain",
    "joined_previous_distinct_feature_gain", "joined_previous_unmatched_fraction_gain",
    "joined_next_entry_probability", "joined_next_probability_gain",
    "joined_next_distinct_feature_gain", "joined_next_unmatched_fraction_gain",
)
INTERACTION_NAMES = (
    "char_length", "token_count", "whitespace_fraction", "punctuation_fraction",
    "gap:unmatched_fraction", "gap:unmatched_prefix_fraction",
    "gap:unmatched_suffix_fraction", *JOIN_NAMES,
)


def _sklearn() -> dict[str, Any]:
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, precision_recall_curve
    from sklearn.preprocessing import StandardScaler

    return {
        "version": sklearn.__version__,
        "hist": HistGradientBoostingClassifier,
        "logistic": LogisticRegression,
        "average_precision": average_precision_score,
        "precision_recall_curve": precision_recall_curve,
        "scaler": StandardScaler,
    }


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _save(path: Path, value: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)


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
                }
                candidate_cursor += 1
            cursor = document_end
    if cursor != expected_line_count or candidate_cursor != len(row_indices):
        raise ValueError("candidate metadata does not cover the source table")
    if any(value is None for value in metadata):
        raise ValueError("candidate metadata contains unresolved rows")
    return [value for value in metadata if value is not None]


def _columns(names: Sequence[str], predicate: Any) -> list[int]:
    return [index for index, name in enumerate(names) if predicate(name)]


def feature_arms(
    features: np.ndarray, names: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    """Build nested feature arms without source or document identity."""

    name_to_column = {name: index for index, name in enumerate(names)}
    if len(name_to_column) != len(names):
        raise ValueError("feature names are duplicated")
    presence_columns = {
        name.removeprefix("presence:"): index
        for index, name in enumerate(names) if name.startswith("presence:")
    }
    family_values, family_names = [], []
    for family, members in SELECTED_FAMILIES.items():
        missing = set(members) - set(presence_columns)
        if missing:
            raise ValueError(f"missing deterministic features for {family}: {sorted(missing)}")
        values = features[:, [presence_columns[name] for name in members]]
        family_values.extend((values.max(axis=1), values.sum(axis=1)))
        family_names.extend((f"family_any:{family}", f"family_count:{family}"))

    shape_columns = _columns(names, lambda name: (
        name in {
            "char_length", "log1p_char_length", "token_count", "log1p_token_count",
            "mean_token_length", "maximum_token_length", "leading_whitespace",
            "trailing_whitespace", "letter_fraction", "digit_fraction",
            "uppercase_fraction_of_letters", "lowercase_fraction_of_letters",
            "greek_fraction_of_letters", "latin_fraction_of_letters",
            "punctuation_fraction", "symbol_fraction", "whitespace_fraction",
            "other_fraction", "starts_lowercase", "starts_uppercase", "starts_digit",
            "starts_bullet_or_number", "ends_sentence_terminal",
            "ends_opening_terminal", "parenthesis_balance", "bracket_balance",
            "quote_parity", "is_blank", "is_repeated_rule", "is_table_rule",
            "is_page_number", "is_bullet_only", "has_html_fragment",
            "has_replacement_character",
        }
        or name.startswith("gap:")
        or name.startswith("previous_pair:")
        or name.startswith("next_pair:")
        or name.startswith("entry_above_r1_")
        or name.startswith("entry_above_r3_")
        or name.startswith("entry_below_r1_")
        or name.startswith("entry_below_r3_")
        or name in {"inside_anchor_gap", "candidate_window_edge_distance"}
    ))
    compact_names = [*family_names, *(names[index] for index in shape_columns)]
    compact = np.column_stack((*family_values, features[:, shape_columns])).astype(np.float32)
    join_columns = [name_to_column[name] for name in JOIN_NAMES]
    compact_join = np.column_stack((compact, features[:, join_columns])).astype(np.float32)
    compact_join_names = [*compact_names, *JOIN_NAMES]
    table = family_values[2 * list(SELECTED_FAMILIES).index("table")]
    interaction_columns = [name_to_column[name] for name in INTERACTION_NAMES]
    interactions = features[:, interaction_columns] * table[:, None]
    table_aware = np.column_stack((compact_join, interactions)).astype(np.float32)
    table_names = [*compact_join_names, *(f"table_x:{name}" for name in INTERACTION_NAMES)]
    arms = {
        "all_177": np.asarray(features, dtype=np.float32),
        "compact_core": compact,
        "compact_plus_directional_join": compact_join,
        "compact_join_table_interactions": table_aware,
    }
    arm_names = {
        "all_177": list(names),
        "compact_core": compact_names,
        "compact_plus_directional_join": compact_join_names,
        "compact_join_table_interactions": table_names,
    }
    for name, values in arms.items():
        if not np.isfinite(values).all() or values.shape[1] != len(arm_names[name]):
            raise ValueError(f"malformed feature arm: {name}")
    return arms, arm_names


def _weights(indices: np.ndarray, target: np.ndarray, documents: np.ndarray) -> np.ndarray:
    doc_counts = collections.Counter(str(documents[index]) for index in indices)
    class_counts = np.bincount(target[indices].astype(np.uint8), minlength=2)
    if np.any(class_counts == 0):
        raise ValueError("fit partition needs both continuation classes")
    result = np.asarray([
        1.0 / doc_counts[str(documents[index])]
        * len(indices) / (2.0 * class_counts[int(target[index])])
        for index in indices
    ], dtype=np.float64)
    return result / result.mean()


def _fit(
    x: np.ndarray, target: np.ndarray, indices: np.ndarray, documents: np.ndarray,
    config: Mapping[str, Any], seed: int,
) -> tuple[Any, Any]:
    tools = _sklearn()
    weights = _weights(indices, target, documents)
    if config["kind"] == "logistic":
        scaler = tools["scaler"]().fit(x[indices], sample_weight=weights)
        transformed = scaler.transform(x[indices]).astype(np.float32)
        model = tools["logistic"](
            C=float(config["C"]), solver="lbfgs", max_iter=1000,
            random_state=seed,
        ).fit(transformed, target[indices], sample_weight=weights)
        return scaler, model
    if config["kind"] == "hist":
        model = tools["hist"](
            learning_rate=0.05, max_iter=200, max_depth=int(config["max_depth"]),
            min_samples_leaf=int(config["min_samples_leaf"]), l2_regularization=1.0,
            early_stopping=False, random_state=seed,
        ).fit(x[indices], target[indices], sample_weight=weights)
        return None, model
    raise ValueError(f"unknown model kind: {config['kind']}")


def _predict(bundle: tuple[Any, Any], x: np.ndarray) -> np.ndarray:
    scaler, model = bundle
    values = scaler.transform(x).astype(np.float32) if scaler is not None else x
    return model.predict_proba(values)[:, 1].astype(np.float32)


def _candidate_configs() -> list[dict[str, Any]]:
    return [
        {"kind": "logistic", "C": 0.1},
        {"kind": "logistic", "C": 1.0},
        {"kind": "hist", "max_depth": 2, "min_samples_leaf": 20},
        {"kind": "hist", "max_depth": 3, "min_samples_leaf": 20},
    ]


def _score(target: np.ndarray, probability: np.ndarray) -> float:
    return float(_sklearn()["average_precision"](target, probability))


def fit_oof(
    x: np.ndarray, target: np.ndarray, eligible: np.ndarray, folds: np.ndarray,
    documents: np.ndarray, seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    probability = np.full(len(target), np.nan, dtype=np.float32)
    reports = []
    unique_folds = sorted(int(value) for value in np.unique(folds[eligible]))
    for outer in unique_folds:
        inner = unique_folds[(unique_folds.index(outer) + 1) % len(unique_folds)]
        inner_fit = np.flatnonzero(eligible & (folds != outer) & (folds != inner))
        inner_holdout = np.flatnonzero(eligible & (folds == inner))
        candidates = []
        for offset, config in enumerate(_candidate_configs()):
            model = _fit(x, target, inner_fit, documents, config, seed + outer * 100 + offset)
            candidates.append({**config, "inner_pr_auc": _score(
                target[inner_holdout], _predict(model, x[inner_holdout])
            )})
        selected = max(candidates, key=lambda row: (row["inner_pr_auc"], row["kind"] == "logistic"))
        fit = np.flatnonzero(eligible & (folds != outer))
        holdout = np.flatnonzero(eligible & (folds == outer))
        model = _fit(x, target, fit, documents, selected, seed + outer)
        probability[holdout] = _predict(model, x[holdout])
        reports.append({
            "outer_fold": outer, "inner_fold": inner, "fit_count": len(fit),
            "holdout_count": len(holdout), "selected": selected,
            "candidates": candidates,
        })
    if not np.isfinite(probability[eligible]).all():
        raise RuntimeError("OOF continuation predictions are incomplete")
    return probability, reports


def _metrics(
    target: np.ndarray, probability: np.ndarray, eligible: np.ndarray,
    documents: np.ndarray, sources: np.ndarray,
) -> dict[str, Any]:
    rows = np.flatnonzero(eligible)
    truth, score = target[rows], probability[rows]
    precision_curve, recall_curve, thresholds = _sklearn()["precision_recall_curve"](truth, score)
    f1 = np.divide(
        2 * precision_curve * recall_curve, precision_curve + recall_curve,
        out=np.zeros_like(precision_curve), where=(precision_curve + recall_curve) > 0,
    )
    best = int(np.argmax(f1))
    threshold = float(thresholds[min(best, len(thresholds) - 1)]) if len(thresholds) else 0.5
    prediction = score >= 0.5
    tp = int(np.count_nonzero(prediction & truth))
    fp = int(np.count_nonzero(prediction & ~truth))
    fn = int(np.count_nonzero(~prediction & truth))
    document_scores = []
    for document in sorted(set(str(value) for value in documents[rows])):
        local = rows[documents[rows] == document]
        if len(np.unique(target[local])) == 2:
            document_scores.append(_score(target[local], probability[local]))
    by_source = {}
    for source in sorted(set(str(value) for value in sources[rows])):
        local = rows[sources[rows] == source]
        by_source[source] = {
            "line_count": len(local), "positive_count": int(target[local].sum()),
            "pr_auc": _score(target[local], probability[local])
            if len(np.unique(target[local])) == 2 else None,
        }
    return {
        "line_count": len(rows), "positive_count": int(truth.sum()),
        "pooled_pr_auc": _score(truth, score),
        "document_macro_pr_auc": float(np.mean(document_scores)) if document_scores else None,
        "document_macro_count": len(document_scores),
        "at_threshold_0_5": {
            "precision": tp / (tp + fp) if tp + fp else 1.0,
            "recall": tp / (tp + fn) if tp + fn else 1.0,
            "tp": tp, "fp": fp, "fn": fn,
        },
        "diagnostic_max_f1": {
            "threshold": threshold, "f1": float(f1[best]),
            "precision": float(precision_curve[best]), "recall": float(recall_curve[best]),
            "not_for_downstream_selection": True,
        },
        "by_source": by_source,
    }


def _source_holdouts(
    x: np.ndarray, target: np.ndarray, eligible: np.ndarray, documents: np.ndarray,
    sources: np.ndarray, config: Mapping[str, Any], seed: int,
) -> dict[str, Any]:
    result = {}
    for offset, source in enumerate(sorted(set(str(value) for value in sources[eligible]))):
        fit = np.flatnonzero(eligible & (sources != source))
        holdout = np.flatnonzero(eligible & (sources == source))
        model = _fit(x, target, fit, documents, config, seed + offset)
        probability = _predict(model, x[holdout])
        result[source] = {
            "fit_count": len(fit), "holdout_count": len(holdout),
            "positive_count": int(target[holdout].sum()),
            "pr_auc": _score(target[holdout], probability),
        }
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    table_dir = Path(args.table_dir).resolve()
    frozen_dir = Path(args.frozen_connector_oof_dir).resolve()
    source_path = Path(args.source_jsonl).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    manifest = json.loads((table_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("split") != "train" or manifest.get("validation_opened") is not False:
        raise ValueError("continuation ablation accepts only the frozen train table")
    if sha256_file(source_path) != manifest["inputs"]["source_sha256"]:
        raise ValueError("source JSONL does not match the connector table")
    names = tuple(str(value) for value in manifest["feature_names"])
    features = np.load(table_dir / "features.npy", mmap_mode="r", allow_pickle=False)
    roles = np.load(table_dir / "roles.npy", mmap_mode="r", allow_pickle=False)
    trusted = np.load(table_dir / "trusted.npy", mmap_mode="r", allow_pickle=False).astype(bool)
    folds = np.load(table_dir / "folds.npy", mmap_mode="r", allow_pickle=False)
    row_indices = np.load(table_dir / "row_indices.npy", mmap_mode="r", allow_pickle=False)
    frozen_rows = np.load(frozen_dir / "row_indices.npy", mmap_mode="r", allow_pickle=False)
    frozen_probability = np.load(frozen_dir / "oof_probability.npy", mmap_mode="r", allow_pickle=False)
    if not np.array_equal(row_indices, frozen_rows) or frozen_probability.shape != (len(features), 4):
        raise ValueError("frozen connector OOF output is not aligned with the table")
    role_encoding = manifest["role_encoding"]
    target = roles == int(role_encoding["CONTINUATION"])
    eligible = trusted & (roles != int(role_encoding["ENTRY"])) & (roles != int(role_encoding["UNKNOWN"]))
    metadata = _candidate_metadata(
        source_jsonl=source_path, row_indices=row_indices,
        expected_line_count=int(manifest["line_count"]), expected_split="train",
    )
    documents = np.asarray([str(row["document_id"]) for row in metadata])
    sources = np.asarray([str(row["source"]) for row in metadata])
    arms, arm_names = feature_arms(features, names)

    probabilities: dict[str, np.ndarray] = {
        "frozen_connector_cascade": np.asarray(frozen_probability[:, 1], dtype=np.float32)
    }
    reports: dict[str, Any] = {}
    for offset, (arm, x) in enumerate(arms.items()):
        probability, folds_report = fit_oof(
            x, target, eligible, folds, documents, args.seed + offset * 1000,
        )
        probabilities[arm] = probability
        reports[arm] = {"feature_count": x.shape[1], "folds": folds_report}

    metrics = {
        arm: _metrics(target, probability, eligible, documents, sources)
        for arm, probability in probabilities.items()
    }
    learned_arms = list(arms)
    selected_arm = max(
        learned_arms,
        key=lambda arm: (
            metrics[arm]["pooled_pr_auc"],
            metrics[arm]["document_macro_pr_auc"] or -1.0,
            -arms[arm].shape[1],
        ),
    )
    selected_configs = collections.Counter(
        json.dumps(row["selected"], sort_keys=True) for row in reports[selected_arm]["folds"]
    )
    modal_config = json.loads(selected_configs.most_common(1)[0][0])
    source_holdouts = _source_holdouts(
        arms[selected_arm], target, eligible, documents, sources, modal_config, args.seed + 9000,
    )

    output.mkdir(parents=True)
    for arm, probability in probabilities.items():
        _save(output / f"{arm}.oof_probability.npy", probability)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_train_only_continuation_head_ablation",
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "validation_opened": False,
        "target_contract": (
            "CONTINUATION versus trusted non-ENTRY candidates; ENTRY remains owned by P0D; "
            "UNKNOWN is masked"
        ),
        "eligible_line_count": int(np.count_nonzero(eligible)),
        "continuation_line_count": int(np.count_nonzero(target & eligible)),
        "negative_line_count": int(np.count_nonzero(~target & eligible)),
        "document_count": len(set(documents[eligible])),
        "source_counts": dict(sorted(collections.Counter(sources[eligible]).items())),
        "sklearn_version": _sklearn()["version"],
        "selection_rule": "highest pooled train OOF PR-AUC, then document-macro PR-AUC, then fewer features",
        "selected_learned_arm": selected_arm,
        "selected_modal_model_config_for_source_audit": modal_config,
        "metrics": metrics,
        "source_holdout_audit_for_selected_arm": source_holdouts,
        "arms": reports,
        "feature_names": arm_names,
        "inputs": {
            "table_manifest_sha256": sha256_file(table_dir / "manifest.json"),
            "frozen_connector_report_sha256": sha256_file(frozen_dir / "report.json"),
            "frozen_connector_oof_sha256": sha256_file(frozen_dir / "oof_probability.npy"),
            "source_jsonl_sha256": sha256_file(source_path),
        },
        "restrictions": {
            "may_replace_frozen_connector_without_block_evaluation": False,
            "sealed_consensus_opened": False,
            "validation_opened": False,
        },
    }
    _write_json_new(output / "report.json", report)
    _write_json_new(output / "receipt.json", {
        **report,
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(output.iterdir()) if path.is_file()
        },
    })
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--frozen-connector-oof-dir", required=True)
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", default=os.environ.get("SLURM_JOB_ID", ""))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    print(json.dumps({
        "status": report["status"],
        "selected_learned_arm": report["selected_learned_arm"],
        "metrics": report["metrics"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
