#!/usr/bin/env python3
"""Fit a richer monotonic gate over frozen role-aware OOF components."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .bibliography_auxiliary_scope_veto import decode_with_auxiliary_veto
from .bibliography_deterministic_roles import ROLE_NAMES, SCHEMA_VERSION as ROLE_SCHEMA
from .bibliography_entry_blocks import BlockConfig, evaluate_prediction
from .bibliography_entry_coherence import is_safe_candidate
from .bibliography_entry_component_gate import (
    CandidateSet,
    EXTENT_SATURATION_LINES,
    _load_quality_exclusions,
    _proposal_ceiling,
)
from .bibliography_entry_models import PINNED_SKLEARN_VERSION, load_table


SCHEMA_VERSION = "bibliography-rich-component-gate-oof-v1"
FEATURE_NAMES = (
    "saturated_extent",
    "entry_probability_q10",
    "entry_probability_median",
    "entry_probability_q90",
    "longest_weak_run_fraction",
    "minimum_boundary_probability",
    "exact_header_at_or_before_start",
) + tuple(f"role_{name}_fraction" for name in ROLE_NAMES)
EXPECTED_DIRECTIONS = (1, 1, 1, 1, -1, 1, 1) + (-1,) * len(ROLE_NAMES)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _save_array(path: Path, value: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)


def _longest_true_run(values: np.ndarray) -> int:
    longest = current = 0
    for enabled in values:
        current = current + 1 if bool(enabled) else 0
        longest = max(longest, current)
    return longest


def rich_component_features(
    table: Any,
    candidates: CandidateSet,
    probability: np.ndarray,
    roles: np.ndarray,
    config: BlockConfig,
) -> np.ndarray:
    """Return perpendicular distribution, boundary, extent, and role summaries."""

    if probability.shape != table.targets.shape:
        raise ValueError("line probability does not align with the table")
    if roles.shape != (len(table.targets), len(ROLE_NAMES)):
        raise ValueError("deterministic roles do not align with the table")
    rows = np.empty((len(candidates.labels), len(FEATURE_NAMES)), dtype=np.float32)
    for row, (document_index, start, end) in enumerate(
        zip(
            candidates.document_indices,
            candidates.starts,
            candidates.ends,
            strict=True,
        )
    ):
        document = table.documents[int(document_index)]
        offset = int(document["line_start"])
        local_start, local_end = int(start), int(end)
        values = np.asarray(
            probability[offset + local_start : offset + local_end + 1],
            dtype=np.float64,
        )
        role_values = np.asarray(
            roles[offset + local_start : offset + local_end + 1],
            dtype=np.float64,
        )
        if not len(values):
            raise ValueError("component feature span is empty")
        edge = min(2, len(values))
        boundary_probability = min(
            float(np.mean(values[:edge])),
            float(np.mean(values[-edge:])),
        )
        rows[row] = np.asarray(
            (
                min(len(values), EXTENT_SATURATION_LINES)
                / EXTENT_SATURATION_LINES,
                float(np.quantile(values, 0.10)),
                float(np.median(values)),
                float(np.quantile(values, 0.90)),
                _longest_true_run(values < config.inside_probability)
                / len(values),
                boundary_probability,
                float(candidates.features[row, 3]),
                *(np.mean(role_values, axis=0).tolist()),
            ),
            dtype=np.float32,
        )
    return rows


def _make_model(arm: str, seed: int) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if arm == "logistic_l2":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=seed,
                solver="lbfgs",
            ),
        )
    if arm == "monotonic_hgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=150,
            max_depth=3,
            min_samples_leaf=20,
            l2_regularization=3.0,
            monotonic_cst=list(EXPECTED_DIRECTIONS),
            early_stopping=False,
            random_state=seed,
        )
    raise ValueError(f"unknown rich component arm: {arm}")


def _crossfit_scores(
    table: Any,
    candidates: CandidateSet,
    *,
    arm: str,
    seed: int,
) -> tuple[np.ndarray, list[Any], list[dict[str, Any]], bool]:
    document_folds = np.asarray([int(document["fold"]) for document in table.documents])
    candidate_folds = document_folds[candidates.document_indices]
    labelled = candidates.labels >= 0
    scores = np.full(len(candidates.labels), np.nan, dtype=np.float32)
    models = []
    fold_rows = []
    direction_ok = True
    for fold in range(int(table.manifest["n_folds"])):
        holdout = candidate_folds == fold
        fit = (candidate_folds != fold) & labelled
        model = _make_model(arm, seed + fold)
        model.fit(candidates.features[fit], candidates.labels[fit])
        scores[holdout] = model.predict_proba(candidates.features[holdout])[:, 1]
        coefficient = None
        fold_direction_ok = True
        if arm == "logistic_l2":
            coefficient = model.named_steps["logisticregression"].coef_[0].tolist()
            fold_direction_ok = all(
                float(value) * expected >= 0
                for value, expected in zip(
                    coefficient, EXPECTED_DIRECTIONS, strict=True
                )
            )
            direction_ok &= fold_direction_ok
        models.append(model)
        fold_rows.append(
            {
                "fold": fold,
                "fit_candidate_count": int(np.count_nonzero(fit)),
                "holdout_candidate_count": int(np.count_nonzero(holdout)),
                "masked_fit_candidate_count": int(
                    np.count_nonzero((candidate_folds != fold) & ~labelled)
                ),
                "coefficient_standardized": coefficient,
                "direction_contract_satisfied": fold_direction_ok,
            }
        )
    if np.isnan(scores).any():
        raise ValueError("rich component OOF scores are incomplete")
    return scores, models, fold_rows, direction_ok


def _load_candidates(root: Path, variant: str) -> CandidateSet:
    prefix = root / variant
    return CandidateSet(
        features=np.load(prefix.with_suffix(".features.npy"), mmap_mode="r", allow_pickle=False),
        document_indices=np.load(prefix.with_suffix(".document_indices.npy"), mmap_mode="r", allow_pickle=False),
        starts=np.load(prefix.with_suffix(".starts.npy"), mmap_mode="r", allow_pickle=False),
        ends=np.load(prefix.with_suffix(".ends.npy"), mmap_mode="r", allow_pickle=False),
        labels=np.load(prefix.with_suffix(".labels.npy"), mmap_mode="r", allow_pickle=False),
    )


def _selection_key(row: dict[str, Any]) -> tuple[Any, ...]:
    metrics = row["metrics"]
    return (
        float(metrics["token_recall"]),
        float(metrics["line_recall"]),
        float(metrics["token_precision"]),
        row["model_arm"] == "logistic_l2",
        -float(row["threshold"]),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    import sklearn

    if sklearn.__version__ != PINNED_SKLEARN_VERSION:
        raise RuntimeError(
            f"expected scikit-learn {PINNED_SKLEARN_VERSION}, got {sklearn.__version__}"
        )
    table = load_table(args.table_dir, expected_split="train")
    source_root = Path(args.source_component_dir).resolve()
    source_report_path = source_root / args.source_report_name
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    if source_report.get("validation_opened") is not False:
        raise ValueError("rich gate requires validation-isolated candidates")
    roles_root = Path(args.deterministic_roles_dir).resolve()
    roles_report_path = roles_root / "deterministic_roles_report.json"
    roles_report = json.loads(roles_report_path.read_text(encoding="utf-8"))
    if (
        roles_report.get("schema_version") != ROLE_SCHEMA
        or tuple(roles_report.get("role_names", ())) != ROLE_NAMES
        or roles_report.get("validation_opened") is not False
    ):
        raise ValueError("rich gate requires the frozen deterministic roles")
    roles_path = roles_root / "negative_roles.npy"
    roles = np.load(roles_path, mmap_mode="r", allow_pickle=False)
    probability_path = (
        Path(args.line_oof_dir).resolve() / f"{args.line_arm}.oof_probability.npy"
    )
    probability = np.load(probability_path, mmap_mode="r", allow_pickle=False)
    scope_root = Path(args.scope_veto_dir).resolve()
    scope_report_path = scope_root / "auxiliary_scope_veto_oof_report.json"
    scope_report = json.loads(scope_report_path.read_text(encoding="utf-8"))
    if (
        scope_report.get("validation_opened") is not False
        or int(scope_report.get("auxiliary_scope_inside_silver_bib_count", -1)) != 0
    ):
        raise ValueError("rich gate requires the audited zero-overlap scope veto")
    auxiliary_scope_path = scope_root / "auxiliary_scope_active.npy"
    auxiliary_scope = np.load(auxiliary_scope_path, mmap_mode="r", allow_pickle=False)
    config = BlockConfig(**source_report["block_config"])
    excluded_ids, quality_packet = _load_quality_exclusions(
        Path(args.quality_decisions).resolve()
    )
    qualified_documents = {
        index
        for index, document in enumerate(table.documents)
        if str(document["document_id"]) not in excluded_ids
    }
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    model_dir = output_dir / "models"
    model_dir.mkdir()

    rows = []
    model_reports = {}
    candidate_sets = {}
    score_arrays = {}
    for variant in args.variants:
        base_candidates = _load_candidates(source_root, variant)
        features = rich_component_features(
            table, base_candidates, probability, roles, config
        )
        candidates = CandidateSet(
            features=features,
            document_indices=base_candidates.document_indices,
            starts=base_candidates.starts,
            ends=base_candidates.ends,
            labels=base_candidates.labels,
        )
        candidate_sets[variant] = candidates
        prefix = output_dir / variant
        for suffix, value in (
            (".features.npy", candidates.features),
            (".document_indices.npy", candidates.document_indices),
            (".starts.npy", candidates.starts),
            (".ends.npy", candidates.ends),
            (".labels.npy", candidates.labels),
        ):
            _save_array(prefix.with_suffix(suffix), value)
        for model_arm in args.model_arms:
            scores, models, folds, direction_ok = _crossfit_scores(
                table, candidates, arm=model_arm, seed=int(args.seed)
            )
            score_arrays[(variant, model_arm)] = scores
            _save_array(output_dir / f"{variant}.{model_arm}.oof_scores.npy", scores)
            for fold, model in enumerate(models):
                with (
                    model_dir / f"{variant}.{model_arm}.fold{fold}.pkl"
                ).open("xb") as handle:
                    pickle.dump(model, handle, protocol=5)
            model_reports[f"{variant}:{model_arm}"] = {
                "folds": folds,
                "direction_contract_satisfied": direction_ok,
            }
            for threshold in args.thresholds:
                prediction, vetoed = decode_with_auxiliary_veto(
                    table,
                    candidates,
                    scores,
                    probability,
                    auxiliary_scope,
                    config,
                    threshold=float(threshold),
                    qualified_documents=qualified_documents,
                    apply_veto=True,
                )
                rows.append(
                    {
                        "variant": variant,
                        "model_arm": model_arm,
                        "threshold": float(threshold),
                        "direction_contract_satisfied": direction_ok,
                        "vetoed_candidate_count": len(vetoed),
                        "metrics": evaluate_prediction(
                            table,
                            prediction,
                            document_subset=qualified_documents,
                        ),
                    }
                )

    safe = [
        row
        for row in rows
        if row["direction_contract_satisfied"] and is_safe_candidate(row)
    ]
    selected = max(safe, key=_selection_key) if safe else None
    p95 = [row for row in rows if row["metrics"]["line_precision"] >= 0.95]
    diagnostic_highest = max(rows, key=_selection_key)
    diagnostic_p95 = max(p95, key=_selection_key) if p95 else None
    if selected is not None:
        prediction, _ = decode_with_auxiliary_veto(
            table,
            candidate_sets[selected["variant"]],
            score_arrays[(selected["variant"], selected["model_arm"])],
            probability,
            auxiliary_scope,
            config,
            threshold=float(selected["threshold"]),
            qualified_documents=qualified_documents,
            apply_veto=True,
        )
        _save_array(output_dir / "selected_oof_prediction.npy", prediction)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "passed_train_oof_safety_gate_validation_unopened"
            if selected is not None
            else "research_only_no_candidate_met_safety_gate"
        ),
        "feature_names": list(FEATURE_NAMES),
        "expected_directions": list(EXPECTED_DIRECTIONS),
        "feature_reference": {
            "saturated_extent": "Candidate line count rises to 32 and then stops increasing.",
            "entry_probability_q10": "Lower-tail frozen entry evidence: 90% of candidate lines score at least this high.",
            "entry_probability_median": "Typical frozen entry evidence inside the candidate.",
            "entry_probability_q90": "Upper-tail frozen entry evidence without reducing the component to one maximum line.",
            "longest_weak_run_fraction": "Largest uninterrupted below-0.25 hole divided by candidate extent.",
            "minimum_boundary_probability": "Lower of the mean frozen probabilities at the first and last two candidate lines.",
            "exact_header_at_or_before_start": "Exact bibliography heading on or immediately before the candidate start.",
            **{
                f"role_{name}_fraction": roles_report["role_reference"][name]
                for name in ROLE_NAMES
            },
        },
        "variants": list(args.variants),
        "model_reports": model_reports,
        "candidate_count": len(rows),
        "safe_candidate_count": len(safe),
        "candidates": rows,
        "selected": selected,
        "diagnostic_highest_recall_candidate": diagnostic_highest,
        "diagnostic_highest_recall_at_line_precision_0_95": diagnostic_p95,
        "proposal_ceiling": {
            variant: _proposal_ceiling(table, candidates, qualified_documents)
            for variant, candidates in candidate_sets.items()
        },
        "selection_rule": "require intended directions, exact audited scope veto, line precision>=0.99, and <=0.02 spurious blocks per zero-BIB document; then maximize token and line recall",
        "block_config": source_report["block_config"],
        "quality_filter": {
            "decision_schema": quality_packet["schema_version"],
            "excluded_document_count": len(excluded_ids),
            "qualified_document_count": len(qualified_documents),
        },
        "input_hashes": {
            "source_component_report": _sha256(source_report_path),
            "deterministic_roles_report": _sha256(roles_report_path),
            "negative_roles": _sha256(roles_path),
            "line_oof_probability": _sha256(probability_path),
            "scope_veto_report": _sha256(scope_report_path),
            "auxiliary_scope": _sha256(auxiliary_scope_path),
            "quality_decisions": _sha256(Path(args.quality_decisions).resolve()),
        },
        "code_commit": str(args.code_commit),
        "slurm_job_id": str(args.slurm_job_id),
        "validation_opened": False,
        "production_eligible": False,
    }
    _write_json(output_dir / "rich_component_gate_oof_report.json", result)
    outputs = {
        str(path.relative_to(output_dir)): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }
    _write_json(output_dir / "receipt.json", {**result, "outputs": outputs})
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--line-oof-dir", required=True)
    parser.add_argument("--source-component-dir", required=True)
    parser.add_argument("--source-report-name", required=True)
    parser.add_argument("--deterministic-roles-dir", required=True)
    parser.add_argument("--scope-veto-dir", required=True)
    parser.add_argument("--quality-decisions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--line-arm", default="D1")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=("no_length_roles", "no_length_or_position_roles"),
    )
    parser.add_argument(
        "--model-arms",
        nargs="+",
        choices=("logistic_l2", "monotonic_hgb"),
        default=("logistic_l2", "monotonic_hgb"),
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99, 0.995),
    )
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
