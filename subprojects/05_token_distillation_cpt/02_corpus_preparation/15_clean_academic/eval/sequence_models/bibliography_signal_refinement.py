#!/usr/bin/env python3
"""Train-only refinement experiments for the frozen signal-block decoder.

This module leaves the line model, signal TCN, and recall-first decoder intact.
It tests two independent precision refinements:

* an edge policy that can remove deterministic structural roles only from the
  two-line fringe outside the independently anchored core; and
* a cross-fitted component gate over frozen probability and role summaries.

The user-reviewed unseen packet is deliberately absent from fitting and model
selection.  It is a later diagnostic for any train-OOF candidate retained here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bibliography_deterministic_roles import (
    ROLE_NAMES,
    SCHEMA_VERSION as ROLE_SCHEMA,
)
from .bibliography_entry_blocks import (
    BlockConfig,
    attach_h0_document,
    blocks_from_mask,
    evaluate_prediction,
)
from .bibliography_entry_component_gate import (
    CandidateSet,
    _load_quality_exclusions,
    candidate_supervision,
)
from .bibliography_entry_dataset import HEADER_NONE, LABEL_TO_ID
from .bibliography_entry_models import PINNED_SKLEARN_VERSION, load_table
from .bibliography_entry_role_sequence import _validate_role_matrix
from .bibliography_signal_block_decode import (
    SCHEMA_VERSION as BLOCK_SCHEMA,
    decode_signal_blocks,
)
from .bibliography_signal_tcn import SCHEMA_VERSION as SIGNAL_SCHEMA


SCHEMA_VERSION = "bibliography-signal-refinement-oof-v2"
EDGE_ROLE_ARMS = {
    "none": (),
    "headings": (
        "exact_negative_scope_heading",
        "generic_markdown_heading",
    ),
    "structural": (
        "exact_negative_scope_heading",
        "generic_markdown_heading",
        "figure_caption",
        "table_or_equation",
    ),
    "prose": (
        "exact_negative_scope_heading",
        "generic_markdown_heading",
        "figure_caption",
        "table_or_equation",
        "running_or_enumerated_prose",
        "legal_procedure",
        "other_explicit_negative",
    ),
    "all": ROLE_NAMES,
}
COMPONENT_VARIANTS = ("whole", "heading_split")
MODEL_ARMS = ("logistic_l2", "monotonic_hgb")
FEATURE_NAMES = (
    "signal_q10",
    "signal_median",
    "signal_q90",
    "entry_q10",
    "entry_median",
    "entry_q90",
    "signal_anchor_fraction",
    "entry_positive_fraction",
    "longest_entry_run_fraction",
    "longest_hard_negative_run_fraction",
    "hard_negative_transition_fraction",
    "starts_with_structural_non_bib_heading",
    "exact_header_at_or_before_start",
    *(f"role_{name}_fraction" for name in ROLE_NAMES),
)
EXPECTED_DIRECTIONS = (
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    -1, -1,
    -1, 1,
    *(-1 for _ in ROLE_NAMES),
)


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
    for enabled in np.asarray(values, dtype=bool):
        current = current + 1 if enabled else 0
        longest = max(longest, current)
    return longest


def refine_outer_edges(
    table: Any,
    base_prediction: np.ndarray,
    core_prediction: np.ndarray,
    roles: np.ndarray,
    *,
    role_names: Sequence[str],
    side: str,
    qualified_documents: set[int],
) -> np.ndarray:
    """Trim only role-marked fringe outside an independently anchored core.

    A role inside the core is never touched.  At a fringe, the closest role is
    a boundary: the role and anything farther outside the core are removed.
    """

    if side not in {"left", "right", "both"}:
        raise ValueError(f"unsupported edge side {side!r}")
    if not (
        len(base_prediction) == len(core_prediction) == len(roles) == len(table.targets)
    ):
        raise ValueError("edge-refinement arrays are not aligned")
    indices = [ROLE_NAMES.index(name) for name in role_names]
    boundary = (
        np.any(roles[:, indices] > 0, axis=1)
        if indices
        else np.zeros(len(base_prediction), dtype=bool)
    )
    result = np.asarray(base_prediction, dtype=bool).copy()
    for document_index in sorted(qualified_documents):
        document = table.documents[document_index]
        start, end = int(document["line_start"]), int(document["line_end"])
        local_base = base_prediction[start:end]
        local_core = core_prediction[start:end]
        local_boundary = boundary[start:end]
        local_result = result[start:end]
        for block_start, block_end in blocks_from_mask(
            local_base, table.abs_indices[start:end]
        ):
            core = np.flatnonzero(local_core[block_start : block_end + 1])
            if not len(core):
                continue
            core_start = block_start + int(core[0])
            core_end = block_start + int(core[-1])
            if side in {"left", "both"} and block_start < core_start:
                marked = np.flatnonzero(local_boundary[block_start:core_start])
                if len(marked):
                    boundary_line = block_start + int(marked[-1])
                    local_result[block_start : boundary_line + 1] = False
            if side in {"right", "both"} and core_end < block_end:
                marked = np.flatnonzero(local_boundary[core_end + 1 : block_end + 1])
                if len(marked):
                    boundary_line = core_end + 1 + int(marked[0])
                    local_result[boundary_line : block_end + 1] = False
    return result


def refine_outer_edges_asymmetric(
    table: Any,
    base_prediction: np.ndarray,
    core_prediction: np.ndarray,
    roles: np.ndarray,
    *,
    left_role_names: Sequence[str],
    right_role_names: Sequence[str],
    qualified_documents: set[int],
) -> np.ndarray:
    """Apply independently chosen left and right fringe policies."""

    left = refine_outer_edges(
        table,
        base_prediction,
        core_prediction,
        roles,
        role_names=left_role_names,
        side="left",
        qualified_documents=qualified_documents,
    )
    return refine_outer_edges(
        table,
        left,
        core_prediction,
        roles,
        role_names=right_role_names,
        side="right",
        qualified_documents=qualified_documents,
    )


def _split_span_at_headings(
    start: int, end: int, generic_heading: np.ndarray
) -> list[tuple[int, int]]:
    """Assign each internal heading to the component that follows it."""

    headings = [
        index
        for index in np.flatnonzero(generic_heading[start : end + 1]) + start
        if index > start
    ]
    spans: list[tuple[int, int]] = []
    current = start
    for heading in headings:
        if current <= heading - 1:
            spans.append((current, int(heading) - 1))
        current = int(heading)
    if current <= end:
        spans.append((current, end))
    return spans


def component_feature_vector(
    signal_probability: np.ndarray,
    entry_probability: np.ndarray,
    roles: np.ndarray,
    header_kinds: np.ndarray,
    *,
    start: int,
    end: int,
    config: BlockConfig,
) -> np.ndarray:
    signal = np.asarray(signal_probability[start : end + 1], dtype=np.float64)
    entry = np.asarray(entry_probability[start : end + 1], dtype=np.float64)
    local_roles = np.asarray(roles[start : end + 1], dtype=bool)
    if not len(signal) or len(entry) != len(signal) or len(local_roles) != len(signal):
        raise ValueError("component inputs are empty or misaligned")
    hard = np.any(local_roles, axis=1)
    entry_positive = entry >= 0.5
    transitions = np.count_nonzero(hard[1:] != hard[:-1]) if len(hard) > 1 else 0
    heading_roles = [
        ROLE_NAMES.index("exact_negative_scope_heading"),
        ROLE_NAMES.index("generic_markdown_heading"),
    ]
    header_start = max(0, start - int(config.header_window))
    return np.asarray(
        (
            *np.quantile(signal, (0.1, 0.5, 0.9)),
            *np.quantile(entry, (0.1, 0.5, 0.9)),
            np.mean(signal >= config.anchor_probability),
            np.mean(entry_positive),
            _longest_true_run(entry_positive) / len(entry_positive),
            _longest_true_run(hard) / len(hard),
            transitions / max(len(hard) - 1, 1),
            float(
                np.any(local_roles[0, heading_roles])
                and int(header_kinds[start]) == HEADER_NONE
            ),
            float(np.any(header_kinds[header_start : start + 1] > 0)),
            *np.mean(local_roles, axis=0),
        ),
        dtype=np.float32,
    )


def generate_component_candidates(
    table: Any,
    base_prediction: np.ndarray,
    signal_probability: np.ndarray,
    entry_probability: np.ndarray,
    roles: np.ndarray,
    *,
    variant: str,
    config: BlockConfig,
    qualified_documents: set[int],
) -> CandidateSet:
    if variant not in COMPONENT_VARIANTS:
        raise ValueError(f"unsupported component variant {variant!r}")
    features: list[np.ndarray] = []
    document_indices: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    labels: list[int] = []
    gold = table.original_labels == LABEL_TO_ID["BIB"]
    generic_index = ROLE_NAMES.index("generic_markdown_heading")
    for document_index in sorted(qualified_documents):
        document = table.documents[document_index]
        doc_start, doc_end = int(document["line_start"]), int(document["line_end"])
        local_abs = table.abs_indices[doc_start:doc_end]
        local_prediction = base_prediction[doc_start:doc_end]
        local_generic = roles[doc_start:doc_end, generic_index] > 0
        for block_start, block_end in blocks_from_mask(local_prediction, local_abs):
            spans = (
                [(block_start, block_end)]
                if variant == "whole"
                else _split_span_at_headings(block_start, block_end, local_generic)
            )
            for start, end in spans:
                features.append(
                    component_feature_vector(
                        signal_probability[doc_start:doc_end],
                        entry_probability[doc_start:doc_end],
                        roles[doc_start:doc_end],
                        table.header_kinds[doc_start:doc_end],
                        start=start,
                        end=end,
                        config=config,
                    )
                )
                document_indices.append(document_index)
                starts.append(start)
                ends.append(end)
                labels.append(candidate_supervision(gold[doc_start + start : doc_start + end + 1]))
    if not features:
        raise ValueError("no component candidates were generated")
    return CandidateSet(
        features=np.stack(features),
        document_indices=np.asarray(document_indices, dtype=np.uint32),
        starts=np.asarray(starts, dtype=np.uint32),
        ends=np.asarray(ends, dtype=np.uint32),
        labels=np.asarray(labels, dtype=np.int8),
    )


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
            learning_rate=0.06,
            max_iter=100,
            max_depth=2,
            min_samples_leaf=20,
            l2_regularization=2.0,
            monotonic_cst=list(EXPECTED_DIRECTIONS),
            early_stopping=False,
            random_state=seed,
        )
    raise ValueError(f"unsupported model arm {arm!r}")


def crossfit_component_scores(
    table: Any,
    candidates: CandidateSet,
    *,
    arm: str,
    seed: int,
) -> tuple[np.ndarray, list[Any], list[dict[str, Any]]]:
    document_folds = np.asarray([int(row["fold"]) for row in table.documents])
    candidate_folds = document_folds[candidates.document_indices]
    labelled = candidates.labels >= 0
    scores = np.full(len(candidates.labels), np.nan, dtype=np.float32)
    models: list[Any] = []
    reports: list[dict[str, Any]] = []
    for fold in range(int(table.manifest["n_folds"])):
        holdout = candidate_folds == fold
        fit = (candidate_folds != fold) & labelled
        if len(np.unique(candidates.labels[fit])) != 2:
            raise ValueError(f"fold {fold} fit candidates do not contain both classes")
        model = _make_model(arm, seed + fold)
        model.fit(candidates.features[fit], candidates.labels[fit])
        scores[holdout] = model.predict_proba(candidates.features[holdout])[:, 1]
        coefficient = None
        direction_ok = True
        if arm == "logistic_l2":
            coefficient = model.named_steps["logisticregression"].coef_[0].tolist()
            direction_ok = all(
                float(value) * expected >= 0
                for value, expected in zip(coefficient, EXPECTED_DIRECTIONS, strict=True)
            )
        models.append(model)
        reports.append(
            {
                "fold": fold,
                "fit_candidate_count": int(np.count_nonzero(fit)),
                "holdout_candidate_count": int(np.count_nonzero(holdout)),
                "masked_fit_candidate_count": int(np.count_nonzero((candidate_folds != fold) & ~labelled)),
                "coefficient_standardized": coefficient,
                "direction_contract_satisfied": direction_ok,
            }
        )
    if np.isnan(scores).any():
        raise ValueError("component OOF scores are incomplete")
    return scores, models, reports


def decode_component_candidates(
    table: Any,
    candidates: CandidateSet,
    scores: np.ndarray,
    entry_probability: np.ndarray,
    config: BlockConfig,
    *,
    threshold: float,
    heading_threshold: float | None = None,
    qualified_documents: set[int],
) -> np.ndarray:
    result = np.zeros(len(table.targets), dtype=bool)
    for document_index in sorted(qualified_documents):
        document = table.documents[document_index]
        doc_start, doc_end = int(document["line_start"]), int(document["line_end"])
        local = np.zeros(doc_end - doc_start, dtype=bool)
        document_rows = np.flatnonzero(candidates.document_indices == document_index)
        available = np.zeros_like(local)
        for row in document_rows:
            available[int(candidates.starts[row]) : int(candidates.ends[row]) + 1] = True
        minimum = np.full(len(document_rows), float(threshold))
        if heading_threshold is not None:
            heading = candidates.features[
                document_rows,
                FEATURE_NAMES.index("starts_with_structural_non_bib_heading"),
            ] > 0.5
            minimum[heading] = max(float(threshold), float(heading_threshold))
        rows = document_rows[scores[document_rows] >= minimum]
        for row in rows:
            local[int(candidates.starts[row]) : int(candidates.ends[row]) + 1] = True
        local = attach_h0_document(
            local,
            entry_probability[doc_start:doc_end],
            table.header_kinds[doc_start:doc_end],
            table.abs_indices[doc_start:doc_end],
            config,
        )
        # A component gate is reject-only.  Header attachment may restore a
        # header that was already in the frozen proposal, but can never add a
        # line outside the proposal candidates.
        local &= available
        result[doc_start:doc_end] = local
    return result


def _edge_selection_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    metrics = row["metrics"]
    return (
        float(metrics["token_f0_5"]),
        float(metrics["line_precision"]),
        float(metrics["token_precision"]),
        float(metrics["line_recall"]),
    )


def _component_selection_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    metrics = row["metrics"]
    return (
        float(metrics["token_f0_5"]),
        float(metrics["line_precision"]),
        float(metrics["token_precision"]),
        float(metrics["line_recall"]),
        row["model_arm"] == "logistic_l2",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    import sklearn

    if sklearn.__version__ != PINNED_SKLEARN_VERSION:
        raise RuntimeError(
            f"expected scikit-learn {PINNED_SKLEARN_VERSION}, got {sklearn.__version__}"
        )
    table = load_table(args.table_dir, expected_split="train")
    signal_root = Path(args.signal_tcn_dir).resolve()
    signal_report_path = signal_root / "signal_tcn_oof_report.json"
    signal_report = json.loads(signal_report_path.read_text(encoding="utf-8"))
    if signal_report.get("schema_version") != SIGNAL_SCHEMA or signal_report.get("validation_opened") is not False:
        raise ValueError("refinement requires validation-isolated signal scores")
    block_root = Path(args.signal_block_dir).resolve()
    block_report_path = block_root / "signal_block_decode_oof_report.json"
    block_report = json.loads(block_report_path.read_text(encoding="utf-8"))
    if block_report.get("schema_version") != BLOCK_SCHEMA or block_report.get("validation_opened") is not False:
        raise ValueError("refinement requires validation-isolated block configuration")
    candidate_row = block_report.get(str(args.candidate))
    if not isinstance(candidate_row, Mapping):
        raise ValueError(f"signal-block candidate {args.candidate!r} is unavailable")
    config = BlockConfig(**candidate_row["config"])

    roles_root = Path(args.deterministic_roles_dir).resolve()
    roles_report_path = roles_root / "deterministic_roles_report.json"
    roles_report = json.loads(roles_report_path.read_text(encoding="utf-8"))
    if (
        roles_report.get("schema_version") != ROLE_SCHEMA
        or tuple(roles_report.get("role_names", ())) != ROLE_NAMES
        or roles_report.get("validation_opened") is not False
    ):
        raise ValueError("refinement requires validation-isolated deterministic roles")
    roles_path = roles_root / "negative_roles.npy"
    roles = _validate_role_matrix(roles_path, len(table.targets))

    signal_path = signal_root / "signal_tcn_oof_probability.npy"
    entry_path = Path(args.line_oof_dir).resolve() / f"{args.line_arm}.oof_probability.npy"
    scope_path = block_root / "auxiliary_scope_active.npy"
    signal = np.load(signal_path, mmap_mode="r", allow_pickle=False)
    entry = np.load(entry_path, mmap_mode="r", allow_pickle=False)
    scope = np.load(scope_path, mmap_mode="r", allow_pickle=False)
    if not (len(signal) == len(entry) == len(scope) == len(table.targets)):
        raise ValueError("frozen refinement inputs are not aligned")

    excluded_ids, quality_packet = _load_quality_exclusions(Path(args.quality_decisions).resolve())
    qualified_documents = {
        index
        for index, document in enumerate(table.documents)
        if str(document["document_id"]) not in excluded_ids
    }
    base_prediction, _ = decode_signal_blocks(
        table,
        signal,
        entry,
        scope,
        config,
        qualified_documents=qualified_documents,
        apply_veto=True,
    )
    core_prediction, _ = decode_signal_blocks(
        table,
        signal,
        entry,
        scope,
        replace(config, adjacent_expansion=0),
        qualified_documents=qualified_documents,
        apply_veto=True,
    )
    baseline_metrics = evaluate_prediction(table, base_prediction, document_subset=qualified_documents)
    gold = table.original_labels == LABEL_TO_ID["BIB"]

    edge_rows: list[dict[str, Any]] = []
    edge_predictions: dict[tuple[str, str], np.ndarray] = {}
    for left_arm in args.edge_role_arms:
        for right_arm in args.edge_role_arms:
            prediction = refine_outer_edges_asymmetric(
                table,
                base_prediction,
                core_prediction,
                roles,
                left_role_names=EDGE_ROLE_ARMS[left_arm],
                right_role_names=EDGE_ROLE_ARMS[right_arm],
                qualified_documents=qualified_documents,
            )
            removed = base_prediction & ~prediction
            row = {
                "left_role_arm": left_arm,
                "right_role_arm": right_arm,
                "removed_line_count": int(np.count_nonzero(removed)),
                "removed_silver_bib_line_count": int(np.count_nonzero(removed & gold)),
                "removed_silver_non_bib_line_count": int(np.count_nonzero(removed & ~gold)),
                "metrics": evaluate_prediction(table, prediction, document_subset=qualified_documents),
            }
            edge_rows.append(row)
            edge_predictions[(left_arm, right_arm)] = prediction
    edge_safe = [
        row
        for row in edge_rows
        if (row["left_role_arm"], row["right_role_arm"]) != ("none", "none")
        and row["metrics"]["line_fp"] < baseline_metrics["line_fp"]
        and row["metrics"]["line_recall"] >= baseline_metrics["line_recall"] - float(args.edge_line_recall_budget)
        and row["metrics"]["token_recall"] >= baseline_metrics["token_recall"] - float(args.edge_token_recall_budget)
    ]
    edge_selected = max(edge_safe, key=_edge_selection_key) if edge_safe else None

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    models_dir = output_dir / "models"
    models_dir.mkdir()
    _save_array(output_dir / "baseline_oof_prediction.npy", base_prediction)
    _save_array(output_dir / "core_oof_prediction.npy", core_prediction)
    if edge_selected is not None:
        _save_array(
            output_dir / "selected_edge_oof_prediction.npy",
            edge_predictions[(edge_selected["left_role_arm"], edge_selected["right_role_arm"])],
        )

    component_rows: list[dict[str, Any]] = []
    candidate_sets: dict[str, CandidateSet] = {}
    score_arrays: dict[tuple[str, str], np.ndarray] = {}
    model_reports: dict[str, Any] = {}
    for variant in args.component_variants:
        candidates = generate_component_candidates(
            table,
            base_prediction,
            signal,
            entry,
            roles,
            variant=variant,
            config=config,
            qualified_documents=qualified_documents,
        )
        candidate_sets[variant] = candidates
        prefix = output_dir / f"{variant}.candidates"
        _save_array(prefix.with_suffix(".features.npy"), candidates.features)
        _save_array(prefix.with_suffix(".document_indices.npy"), candidates.document_indices)
        _save_array(prefix.with_suffix(".starts.npy"), candidates.starts)
        _save_array(prefix.with_suffix(".ends.npy"), candidates.ends)
        _save_array(prefix.with_suffix(".labels.npy"), candidates.labels)
        for model_arm in args.model_arms:
            scores, models, folds = crossfit_component_scores(
                table, candidates, arm=model_arm, seed=int(args.seed)
            )
            score_arrays[(variant, model_arm)] = scores
            _save_array(output_dir / f"{variant}.{model_arm}.oof_scores.npy", scores)
            for fold, model in enumerate(models):
                with (models_dir / f"{variant}.{model_arm}.fold{fold}.pkl").open("xb") as handle:
                    pickle.dump(model, handle, protocol=5)
            labelled = candidates.labels >= 0
            full_model = _make_model(model_arm, int(args.seed) + 1000)
            full_model.fit(candidates.features[labelled], candidates.labels[labelled])
            with (models_dir / f"{variant}.{model_arm}.full.pkl").open("xb") as handle:
                pickle.dump(full_model, handle, protocol=5)
            model_reports[f"{variant}:{model_arm}"] = {
                "folds": folds,
                "labelled_candidate_count": int(np.count_nonzero(labelled)),
                "positive_candidate_count": int(np.count_nonzero(candidates.labels[labelled] == 1)),
                "negative_candidate_count": int(np.count_nonzero(candidates.labels[labelled] == 0)),
                "masked_candidate_count": int(np.count_nonzero(~labelled)),
            }
            for threshold in args.thresholds:
                heading_thresholds = (
                    (float(threshold),)
                    if variant == "whole"
                    else tuple(
                        sorted(
                            {
                                float(threshold),
                                *(
                                    float(value)
                                    for value in args.heading_thresholds
                                    if float(value) >= float(threshold)
                                ),
                            }
                        )
                    )
                )
                for heading_threshold in heading_thresholds:
                    prediction = decode_component_candidates(
                        table,
                        candidates,
                        scores,
                        entry,
                        config,
                        threshold=float(threshold),
                        heading_threshold=heading_threshold,
                        qualified_documents=qualified_documents,
                    )
                    component_rows.append(
                        {
                            "variant": variant,
                            "model_arm": model_arm,
                            "threshold": float(threshold),
                            "heading_threshold": heading_threshold,
                            "metrics": evaluate_prediction(table, prediction, document_subset=qualified_documents),
                        }
                    )
    component_safe = [
        row
        for row in component_rows
        if row["metrics"]["line_fp"] < baseline_metrics["line_fp"]
        and row["metrics"]["line_recall"] >= baseline_metrics["line_recall"] - float(args.component_line_recall_budget)
        and row["metrics"]["token_recall"] >= baseline_metrics["token_recall"] - float(args.component_token_recall_budget)
    ]
    component_selected = max(component_safe, key=_component_selection_key) if component_safe else None
    if component_selected is not None:
        candidates = candidate_sets[component_selected["variant"]]
        selected_prediction = decode_component_candidates(
            table,
            candidates,
            score_arrays[(component_selected["variant"], component_selected["model_arm"])],
            entry,
            config,
            threshold=float(component_selected["threshold"]),
            heading_threshold=float(component_selected["heading_threshold"]),
            qualified_documents=qualified_documents,
        )
        _save_array(output_dir / "selected_component_oof_prediction.npy", selected_prediction)

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_train_oof_refinement_experiments_validation_unopened",
        "frozen_decoder_candidate": str(args.candidate),
        "frozen_decoder_config": asdict(config),
        "baseline_metrics": baseline_metrics,
        "edge_experiment": {
            "contract": "only deterministic roles in the fringe outside the independently anchored core may be trimmed; internal lines are immutable",
            "role_arms": {name: list(EDGE_ROLE_ARMS[name]) for name in args.edge_role_arms},
            "line_recall_budget": float(args.edge_line_recall_budget),
            "token_recall_budget": float(args.edge_token_recall_budget),
            "candidates": edge_rows,
            "selected": edge_selected,
        },
        "component_experiment": {
            "contract": "document-grouped cross-fitting over frozen probabilities and deterministic roles; no text, source, document identity, raw length, or position features",
            "feature_names": list(FEATURE_NAMES),
            "expected_directions": list(EXPECTED_DIRECTIONS),
            "variants": list(args.component_variants),
            "model_arms": list(args.model_arms),
            "heading_thresholds": list(args.heading_thresholds),
            "line_recall_budget": float(args.component_line_recall_budget),
            "token_recall_budget": float(args.component_token_recall_budget),
            "models": model_reports,
            "candidates": component_rows,
            "selected": component_selected,
            "highest_token_f0_5": max(component_rows, key=_component_selection_key),
        },
        "quality_filter": {
            "decision_schema": quality_packet["schema_version"],
            "excluded_document_count": len(excluded_ids),
            "qualified_document_count": len(qualified_documents),
        },
        "input_hashes": {
            "signal_report": _sha256(signal_report_path),
            "block_report": _sha256(block_report_path),
            "signal_probability": _sha256(signal_path),
            "entry_probability": _sha256(entry_path),
            "deterministic_roles_report": _sha256(roles_report_path),
            "deterministic_roles": _sha256(roles_path),
            "auxiliary_scope": _sha256(scope_path),
            "quality_decisions": _sha256(Path(args.quality_decisions).resolve()),
        },
        "code_commit": str(args.code_commit),
        "slurm_job_id": str(args.slurm_job_id),
        "unseen_review_used_for_fitting_or_selection": False,
        "development_review_informed_experiment_design": True,
        "independent_fresh_unseen_evaluation_required": True,
        "validation_opened": False,
        "production_eligible": False,
    }
    report_path = output_dir / "signal_refinement_oof_report.json"
    _write_json(report_path, report)
    outputs = {
        str(path.relative_to(output_dir)): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }
    _write_json(output_dir / "receipt.json", {**report, "outputs": outputs})
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--line-oof-dir", required=True)
    parser.add_argument("--signal-tcn-dir", required=True)
    parser.add_argument("--signal-block-dir", required=True)
    parser.add_argument("--deterministic-roles-dir", required=True)
    parser.add_argument("--quality-decisions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--line-arm", default="D1")
    parser.add_argument("--candidate", default="diagnostic_highest_recall_candidate")
    parser.add_argument("--edge-role-arms", nargs="+", choices=tuple(EDGE_ROLE_ARMS), default=tuple(EDGE_ROLE_ARMS))
    parser.add_argument("--component-variants", nargs="+", choices=COMPONENT_VARIANTS, default=COMPONENT_VARIANTS)
    parser.add_argument("--model-arms", nargs="+", choices=MODEL_ARMS, default=MODEL_ARMS)
    parser.add_argument("--thresholds", type=float, nargs="+", default=(0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99))
    parser.add_argument("--heading-thresholds", type=float, nargs="+", default=(0.10, 0.20, 0.30, 0.50))
    parser.add_argument("--edge-line-recall-budget", type=float, default=0.0025)
    parser.add_argument("--edge-token-recall-budget", type=float, default=0.001)
    parser.add_argument("--component-line-recall-budget", type=float, default=0.0025)
    parser.add_argument("--component-token-recall-budget", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
