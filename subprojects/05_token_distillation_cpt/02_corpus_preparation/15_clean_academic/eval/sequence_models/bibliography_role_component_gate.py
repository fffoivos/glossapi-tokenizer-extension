#!/usr/bin/env python3
"""Gate role-aware train-OOF bibliography proposals as coherent components.

The role-aware CRF improves the recall frontier but still proposes isolated
false blocks.  This experiment applies the already-defined five structural
component features to those proposals.  Work folds, extraction exclusions,
and validation isolation remain unchanged.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .bibliography_deterministic_roles import ROLE_NAMES, SCHEMA_VERSION as ROLE_SCHEMA
from .bibliography_entry_blocks import BlockConfig, evaluate_prediction
from .bibliography_entry_coherence import is_safe_candidate
from .bibliography_entry_component_gate import (
    EXPECTED_DIRECTIONS,
    FEATURE_NAMES,
    _candidate_key,
    _crossfit_scores,
    _load_quality_exclusions,
    _proposal_ceiling,
    decode_candidates,
    generate_candidates,
)
from .bibliography_entry_models import PINNED_SKLEARN_VERSION, load_table
from .bibliography_entry_role_sequence import (
    ROLE_FEATURE_NAMES,
    SCHEMA_VERSION as ROLE_SEQUENCE_SCHEMA,
    VARIANTS,
    _validate_role_matrix,
)
from .bibliography_entry_sequence import (
    FEATURE_NAMES_BASE,
    constrained_viterbi,
    make_examples,
)
from .feature_crf import LinearChainCRF
from .features import TAGS


SCHEMA_VERSION = "bibliography-role-component-gate-oof-v1"


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


def _decode_fold(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        table_dir,
        probability_path,
        role_path,
        model_root,
        arm,
        variant,
        fold,
        seed_length_limit,
        deletion_biases,
        excluded_ids,
    ) = task
    table = load_table(table_dir, expected_split="train")
    probability = np.load(probability_path, mmap_mode="r", allow_pickle=False)
    roles = _validate_role_matrix(Path(role_path), len(table.targets))
    model, metadata = LinearChainCRF.load(
        Path(model_root) / f"{arm}.{variant}.fold{fold}.npz"
    )
    if (
        metadata.get("schema_version") != ROLE_SEQUENCE_SCHEMA
        or metadata.get("variant") != variant
        or tuple(metadata.get("dropped_feature_names", ())) != VARIANTS[variant]
        or tuple(metadata.get("feature_names", ()))
        != FEATURE_NAMES_BASE + ROLE_FEATURE_NAMES
    ):
        raise ValueError("role-sequence checkpoint metadata is inconsistent")
    excluded = set(excluded_ids)
    document_indices = [
        index
        for index, document in enumerate(table.documents)
        if int(document["fold"]) == fold
        and str(document["document_id"]) not in excluded
    ]
    examples = make_examples(
        table,
        probability,
        document_indices,
        seed_length_limit=seed_length_limit,
        include_header=False,
        dropped_feature_names=VARIANTS[variant],
        extra_features=roles,
    )
    indices: list[int] = []
    predictions = {float(bias): [] for bias in deletion_biases}
    for example in examples:
        indices.extend(example.line_indices)
        for bias in deletion_biases:
            tags = constrained_viterbi(
                model,
                example.features,
                example.char_lengths,
                seed_length_limit=seed_length_limit,
                deletion_bias=float(bias),
            )
            predictions[float(bias)].extend(
                TAGS[int(tag)] != "O" for tag in tags
            )
    return {
        "variant": variant,
        "fold": fold,
        "indices": np.asarray(indices, dtype=np.uint32),
        "predictions": {
            bias: np.asarray(values, dtype=bool)
            for bias, values in predictions.items()
        },
    }


def _proposal_masks(
    table: Any,
    probability_path: Path,
    role_path: Path,
    model_root: Path,
    *,
    arm: str,
    variants: Sequence[str],
    deletion_biases: Sequence[float],
    seed_length_limit: int,
    excluded_ids: set[str],
    workers: int,
) -> dict[tuple[str, float], np.ndarray]:
    tasks = [
        (
            str(table.root),
            str(probability_path),
            str(role_path),
            str(model_root),
            arm,
            variant,
            fold,
            seed_length_limit,
            tuple(deletion_biases),
            tuple(sorted(excluded_ids)),
        )
        for variant in variants
        for fold in range(int(table.manifest["n_folds"]))
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_decode_fold, tasks, chunksize=1))
    masks = {
        (variant, float(bias)): np.zeros(len(table.targets), dtype=bool)
        for variant in variants
        for bias in deletion_biases
    }
    for result in results:
        for bias, values in result["predictions"].items():
            masks[(result["variant"], float(bias))][result["indices"]] = values
    return masks


def run(args: argparse.Namespace) -> dict[str, Any]:
    import sklearn

    if sklearn.__version__ != PINNED_SKLEARN_VERSION:
        raise RuntimeError(
            f"expected scikit-learn {PINNED_SKLEARN_VERSION}, got {sklearn.__version__}"
        )
    table = load_table(args.table_dir, expected_split="train")
    line_root = Path(args.line_oof_dir).resolve()
    role_sequence_root = Path(args.role_sequence_dir).resolve()
    roles_root = Path(args.deterministic_roles_dir).resolve()
    quality_path = Path(args.quality_decisions).resolve()
    role_sequence_report_path = role_sequence_root / "role_sequence_oof_report.json"
    roles_report_path = roles_root / "deterministic_roles_report.json"
    role_sequence_report = json.loads(
        role_sequence_report_path.read_text(encoding="utf-8")
    )
    roles_report = json.loads(roles_report_path.read_text(encoding="utf-8"))
    if (
        role_sequence_report.get("schema_version") != ROLE_SEQUENCE_SCHEMA
        or role_sequence_report.get("validation_opened") is not False
    ):
        raise ValueError("component gate requires validation-isolated role sequences")
    if (
        roles_report.get("schema_version") != ROLE_SCHEMA
        or tuple(roles_report.get("role_names", ())) != ROLE_NAMES
        or roles_report.get("validation_opened") is not False
    ):
        raise ValueError("component gate requires the frozen deterministic roles")
    arm = str(args.arm)
    config = BlockConfig(**role_sequence_report["block_config"])
    probability_path = line_root / f"{arm}.oof_probability.npy"
    probability = np.load(probability_path, mmap_mode="r", allow_pickle=False)
    role_path = roles_root / "negative_roles.npy"
    roles = _validate_role_matrix(role_path, len(table.targets))
    hard_negative = np.any(roles > 0, axis=1)
    excluded_ids, quality_packet = _load_quality_exclusions(quality_path)
    known_ids = {str(document["document_id"]) for document in table.documents}
    if not excluded_ids <= known_ids:
        raise ValueError("quality decisions exclude an unknown train document")
    qualified_documents = {
        index
        for index, document in enumerate(table.documents)
        if str(document["document_id"]) not in excluded_ids
    }
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)

    masks = _proposal_masks(
        table,
        probability_path,
        role_path,
        role_sequence_root / "models",
        arm=arm,
        variants=args.variants,
        deletion_biases=args.deletion_biases,
        seed_length_limit=config.seed_length_limit,
        excluded_ids=excluded_ids,
        workers=int(args.workers),
    )
    candidate_sets = {
        variant: generate_candidates(
            table,
            probability,
            hard_negative,
            masks,
            variant=variant,
            deletion_biases=args.deletion_biases,
            config=config,
            qualified_documents=qualified_documents,
        )
        for variant in args.variants
    }

    output_dir.mkdir(parents=True)
    model_dir = output_dir / "models"
    model_dir.mkdir()
    rows: list[dict[str, Any]] = []
    model_reports: dict[str, Any] = {}
    score_arrays: dict[tuple[str, str], np.ndarray] = {}
    for variant, candidates in candidate_sets.items():
        prefix = output_dir / variant
        _save_array(prefix.with_suffix(".features.npy"), candidates.features)
        _save_array(
            prefix.with_suffix(".document_indices.npy"),
            candidates.document_indices,
        )
        _save_array(prefix.with_suffix(".starts.npy"), candidates.starts)
        _save_array(prefix.with_suffix(".ends.npy"), candidates.ends)
        _save_array(prefix.with_suffix(".labels.npy"), candidates.labels)
        for model_arm in args.model_arms:
            scores, models, folds, direction_ok = _crossfit_scores(
                table,
                candidates,
                arm=model_arm,
                seed=int(args.seed),
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
                prediction = decode_candidates(
                    table,
                    candidates,
                    scores,
                    probability,
                    config,
                    threshold=float(threshold),
                    qualified_documents=qualified_documents,
                )
                rows.append(
                    {
                        "variant": variant,
                        "model_arm": model_arm,
                        "threshold": float(threshold),
                        "direction_contract_satisfied": direction_ok,
                        "metrics": evaluate_prediction(
                            table,
                            prediction,
                            document_subset=qualified_documents,
                        ),
                    }
                )

    safe_rows = [
        row
        for row in rows
        if row["direction_contract_satisfied"] and is_safe_candidate(row)
    ]
    selected = max(safe_rows, key=_candidate_key) if safe_rows else None
    precision95 = [row for row in rows if row["metrics"]["line_precision"] >= 0.95]
    diagnostic_highest = max(rows, key=_candidate_key)
    diagnostic_p95 = max(precision95, key=_candidate_key) if precision95 else None
    if selected is not None:
        candidates = candidate_sets[selected["variant"]]
        scores = score_arrays[(selected["variant"], selected["model_arm"])]
        prediction = decode_candidates(
            table,
            candidates,
            scores,
            probability,
            config,
            threshold=float(selected["threshold"]),
            qualified_documents=qualified_documents,
        )
        _save_array(output_dir / "selected_oof_prediction.npy", prediction)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "passed_train_oof_safety_gate_validation_unopened"
            if selected is not None
            else "research_only_no_candidate_met_safety_gate"
        ),
        "arm": arm,
        "variants": list(args.variants),
        "deletion_biases": [float(value) for value in args.deletion_biases],
        "feature_names": list(FEATURE_NAMES),
        "expected_directions": list(EXPECTED_DIRECTIONS),
        "model_reports": model_reports,
        "candidate_count": len(rows),
        "safe_candidate_count": len(safe_rows),
        "candidates": rows,
        "selected": selected,
        "diagnostic_highest_recall_candidate": diagnostic_highest,
        "diagnostic_highest_recall_at_line_precision_0_95": diagnostic_p95,
        "proposal_ceiling": {
            variant: _proposal_ceiling(table, candidates, qualified_documents)
            for variant, candidates in candidate_sets.items()
        },
        "selection_rule": "require intended feature directions, line precision>=0.99, and <=0.02 spurious blocks per zero-BIB document; then maximize token and line recall",
        "block_config": asdict(config),
        "quality_filter": {
            "decision_schema": quality_packet["schema_version"],
            "excluded_document_count": len(excluded_ids),
            "qualified_document_count": len(qualified_documents),
        },
        "input_hashes": {
            "role_sequence_report": _sha256(role_sequence_report_path),
            "deterministic_roles_report": _sha256(roles_report_path),
            "negative_roles": _sha256(role_path),
            "line_oof_probability": _sha256(probability_path),
            "quality_decisions": _sha256(quality_path),
        },
        "code_commit": str(args.code_commit),
        "slurm_job_id": str(args.slurm_job_id),
        "validation_opened": False,
        "production_eligible": False,
    }
    _write_json(output_dir / "role_component_gate_oof_report.json", report)
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
    parser.add_argument("--role-sequence-dir", required=True)
    parser.add_argument("--deterministic-roles-dir", required=True)
    parser.add_argument("--quality-decisions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arm", default="D1")
    parser.add_argument(
        "--variants", nargs="+", choices=tuple(VARIANTS), default=tuple(VARIANTS)
    )
    parser.add_argument(
        "--deletion-biases",
        type=float,
        nargs="+",
        default=(-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0),
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
        default=(0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995),
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
