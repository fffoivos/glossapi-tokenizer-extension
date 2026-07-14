#!/usr/bin/env python3
"""Train-only B1 CRF with perpendicular deterministic document-role cues.

The frozen entry probability remains the only positive citation summary.  This
experiment adds one-hot cues for explicit competing line roles such as figure
captions, footnotes, and tables.  The CRF may learn how those cues behave in
sequence; no role is a hard veto, and no validation input is accepted.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .bibliography_deterministic_roles import ROLE_NAMES, SCHEMA_VERSION as ROLE_SCHEMA
from .bibliography_entry_blocks import BlockConfig, evaluate_prediction
from .bibliography_entry_coherence import (
    AnchoredCoherenceConfig,
    filter_anchored_components,
    is_safe_candidate,
)
from .bibliography_entry_component_gate import _load_quality_exclusions
from .bibliography_entry_models import load_table
from .bibliography_entry_sequence import (
    FEATURE_NAMES_BASE,
    constrained_viterbi,
    make_examples,
)
from .feature_crf import LinearChainCRF, train_model
from .features import TAGS


SCHEMA_VERSION = "bibliography-entry-role-sequence-oof-v1"
VARIANTS = {
    "no_length_roles": ("log1p_char_length", "over_seed_length_limit"),
    "no_length_or_position_roles": (
        "log1p_char_length",
        "over_seed_length_limit",
        "physical_position",
    ),
}
ROLE_FEATURE_NAMES = tuple(f"explicit_role_{name}" for name in ROLE_NAMES)


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


def _validate_role_matrix(path: Path, expected_lines: int) -> np.ndarray:
    roles = np.load(path, mmap_mode="r", allow_pickle=False)
    if roles.shape != (expected_lines, len(ROLE_NAMES)):
        raise ValueError("deterministic role matrix does not align with the table")
    if not np.all((roles == 0) | (roles == 1)):
        raise ValueError("deterministic role matrix must be binary")
    if np.any(np.sum(roles, axis=1) > 1):
        raise ValueError("deterministic roles must remain mutually exclusive")
    return roles


def _train_fold(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        table_dir,
        probability_path,
        role_path,
        excluded_ids,
        variant,
        fold,
        seed_length_limit,
        deletion_biases,
        epochs,
        learning_rate,
        l2,
        gradient_clip,
        seed,
    ) = task
    table = load_table(table_dir, expected_split="train")
    probability = np.load(probability_path, mmap_mode="r", allow_pickle=False)
    roles = _validate_role_matrix(Path(role_path), len(table.targets))
    excluded = set(excluded_ids)
    fit_docs = [
        index
        for index, document in enumerate(table.documents)
        if int(document["fold"]) != fold
        and str(document["document_id"]) not in excluded
    ]
    holdout_docs = [
        index
        for index, document in enumerate(table.documents)
        if int(document["fold"]) == fold
        and str(document["document_id"]) not in excluded
    ]
    dropped = VARIANTS[variant]
    fit_examples = make_examples(
        table,
        probability,
        fit_docs,
        seed_length_limit=seed_length_limit,
        include_header=False,
        dropped_feature_names=dropped,
        extra_features=roles,
    )
    holdout_examples = make_examples(
        table,
        probability,
        holdout_docs,
        seed_length_limit=seed_length_limit,
        include_header=False,
        dropped_feature_names=dropped,
        extra_features=roles,
    )
    model = LinearChainCRF(
        len(FEATURE_NAMES_BASE) + len(ROLE_FEATURE_NAMES),
        seed=seed + fold,
        active_classes=("BIB",),
    )
    history = train_model(
        model,
        fit_examples,  # type: ignore[arg-type]
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        gradient_clip=gradient_clip,
        seed=seed + fold,
    )
    indices: list[int] = []
    predictions = {float(bias): [] for bias in deletion_biases}
    for example in holdout_examples:
        indices.extend(example.line_indices)
        for deletion_bias in deletion_biases:
            tags = constrained_viterbi(
                model,
                example.features,
                example.char_lengths,
                seed_length_limit=seed_length_limit,
                deletion_bias=float(deletion_bias),
            )
            predictions[float(deletion_bias)].extend(
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
        "model": model,
        "history": history,
        "fit_document_count": len(fit_docs),
        "holdout_document_count": len(holdout_docs),
    }


def _selection_key(row: dict[str, Any]) -> tuple[Any, ...]:
    metrics = row["metrics"]
    return (
        float(metrics["token_recall"]),
        float(metrics["line_recall"]),
        float(metrics["token_precision"]),
        row["variant"] == "no_length_roles",
        -int(row["coherence_config"]["minimum_anchor_count"]),
        -int(row["coherence_config"]["minimum_component_lines"]),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    table = load_table(args.table_dir, expected_split="train")
    line_root = Path(args.line_oof_dir).resolve()
    block_root = Path(args.block_oof_dir).resolve()
    roles_root = Path(args.deterministic_roles_dir).resolve()
    quality_path = Path(args.quality_decisions).resolve()
    block_report_path = block_root / "block_oof_report.json"
    roles_report_path = roles_root / "deterministic_roles_report.json"
    block_report = json.loads(block_report_path.read_text(encoding="utf-8"))
    roles_report = json.loads(roles_report_path.read_text(encoding="utf-8"))
    if roles_report.get("schema_version") != ROLE_SCHEMA:
        raise ValueError("unsupported deterministic role schema")
    if roles_report.get("validation_opened") is not False:
        raise ValueError("role sequence requires validation-isolated role cues")
    if tuple(roles_report.get("role_names", ())) != ROLE_NAMES:
        raise ValueError("deterministic role ordering has changed")
    role_path = roles_root / "negative_roles.npy"
    _validate_role_matrix(role_path, len(table.targets))
    excluded_ids, quality_packet = _load_quality_exclusions(quality_path)
    known_ids = {str(document["document_id"]) for document in table.documents}
    if not excluded_ids <= known_ids:
        raise ValueError("quality decisions exclude an unknown train document")
    qualified_documents = {
        index
        for index, document in enumerate(table.documents)
        if str(document["document_id"]) not in excluded_ids
    }
    arm = str(args.arm)
    block_config = BlockConfig(**block_report["arms"][arm]["selected_config"])
    probability_path = line_root / f"{arm}.oof_probability.npy"
    probability = np.load(probability_path, mmap_mode="r", allow_pickle=False)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    model_dir = output_dir / "models"
    model_dir.mkdir()

    tasks = [
        (
            str(Path(args.table_dir).resolve()),
            str(probability_path),
            str(role_path),
            tuple(sorted(excluded_ids)),
            variant,
            fold,
            block_config.seed_length_limit,
            tuple(args.deletion_biases),
            int(args.epochs),
            float(args.learning_rate),
            float(args.l2),
            float(args.gradient_clip),
            int(args.seed),
        )
        for variant in args.variants
        for fold in range(int(table.manifest["n_folds"]))
    ]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=int(args.workers)
    ) as executor:
        results = list(executor.map(_train_fold, tasks, chunksize=1))

    rows: list[dict[str, Any]] = []
    raw_predictions: dict[tuple[str, float], np.ndarray] = {}
    fold_rows: dict[str, list[dict[str, Any]]] = {
        variant: [] for variant in args.variants
    }
    for variant in args.variants:
        for bias in args.deletion_biases:
            raw_predictions[(variant, float(bias))] = np.zeros(
                len(table.targets), dtype=bool
            )
        for result in results:
            if result["variant"] != variant:
                continue
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "arm": arm,
                "variant": variant,
                "fold": result["fold"],
                "feature_names": list(FEATURE_NAMES_BASE + ROLE_FEATURE_NAMES),
                "dropped_feature_names": list(VARIANTS[variant]),
                "history": result["history"],
                "active_classes": ["BIB"],
            }
            result["model"].save(
                model_dir / f"{arm}.{variant}.fold{result['fold']}.npz",
                metadata,
            )
            fold_rows[variant].append(
                {
                    "fold": result["fold"],
                    "history": result["history"],
                    "fit_document_count": result["fit_document_count"],
                    "holdout_document_count": result["holdout_document_count"],
                }
            )
            for bias, values in result["predictions"].items():
                raw_predictions[(variant, float(bias))][result["indices"]] = values

        for deletion_bias in args.deletion_biases:
            raw = raw_predictions[(variant, float(deletion_bias))]
            for minimum_anchor_count in args.minimum_anchor_counts:
                for minimum_component_lines in args.minimum_component_lines:
                    coherence = AnchoredCoherenceConfig(
                        deletion_bias=float(deletion_bias),
                        minimum_anchor_count=int(minimum_anchor_count),
                        minimum_component_lines=int(minimum_component_lines),
                    )
                    prediction = filter_anchored_components(
                        table,
                        raw,
                        probability,
                        block_config=block_config,
                        config=coherence,
                    )
                    rows.append(
                        {
                            "variant": variant,
                            "dropped_feature_names": list(VARIANTS[variant]),
                            "coherence_config": asdict(coherence),
                            "metrics": evaluate_prediction(
                                table,
                                prediction,
                                document_subset=qualified_documents,
                            ),
                        }
                    )
    safe_rows = [row for row in rows if is_safe_candidate(row)]
    selected = max(safe_rows, key=_selection_key) if safe_rows else None
    diagnostic_highest_recall = max(rows, key=_selection_key)
    precision95_rows = [
        row for row in rows if row["metrics"]["line_precision"] >= 0.95
    ]
    diagnostic_p95 = (
        max(precision95_rows, key=_selection_key) if precision95_rows else None
    )
    if selected is not None:
        selected_coherence = AnchoredCoherenceConfig(**selected["coherence_config"])
        selected_prediction = filter_anchored_components(
            table,
            raw_predictions[
                (selected["variant"], selected_coherence.deletion_bias)
            ],
            probability,
            block_config=block_config,
            config=selected_coherence,
        )
        _save_array(output_dir / "selected_oof_prediction.npy", selected_prediction)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "passed_train_oof_safety_gate_validation_unopened"
            if selected is not None
            else "research_only_no_candidate_met_safety_gate"
        ),
        "arm": arm,
        "variants": {
            variant: {
                "dropped_feature_names": list(VARIANTS[variant]),
                "folds": sorted(fold_rows[variant], key=lambda row: row["fold"]),
            }
            for variant in args.variants
        },
        "feature_names": list(FEATURE_NAMES_BASE + ROLE_FEATURE_NAMES),
        "candidate_count": len(rows),
        "safe_candidate_count": len(safe_rows),
        "candidates": rows,
        "selected": selected,
        "diagnostic_highest_recall_candidate": diagnostic_highest_recall,
        "diagnostic_highest_recall_at_line_precision_0_95": diagnostic_p95,
        "selection_rule": "require line precision>=0.99 and <=0.02 spurious blocks per zero-BIB document; then maximize token and line recall",
        "feature_reference": {
            feature_name: roles_report["role_reference"][role_name]
            for feature_name, role_name in zip(
                ROLE_FEATURE_NAMES, ROLE_NAMES, strict=True
            )
        },
        "block_config": asdict(block_config),
        "training": {
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "l2": float(args.l2),
            "gradient_clip": float(args.gradient_clip),
            "workers": int(args.workers),
        },
        "quality_filter": {
            "decision_schema": quality_packet["schema_version"],
            "excluded_document_count": len(excluded_ids),
            "qualified_document_count": len(qualified_documents),
        },
        "input_hashes": {
            "block_oof_report": _sha256(block_report_path),
            "line_oof_probability": _sha256(probability_path),
            "deterministic_roles_report": _sha256(roles_report_path),
            "negative_roles": _sha256(role_path),
            "quality_decisions": _sha256(quality_path),
        },
        "code_commit": str(args.code_commit),
        "slurm_job_id": str(args.slurm_job_id),
        "validation_opened": False,
        "production_eligible": False,
    }
    _write_json(output_dir / "role_sequence_oof_report.json", report)
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
    parser.add_argument("--block-oof-dir", required=True)
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
        "--minimum-anchor-counts", type=int, nargs="+", default=(0, 1, 2, 3)
    )
    parser.add_argument(
        "--minimum-component-lines", type=int, nargs="+", default=(1, 3, 5)
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
