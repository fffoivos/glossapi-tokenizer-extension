#!/usr/bin/env python3
"""Train-only B1 observation ablations followed by anchored coherence gates.

The original B1 gave line length two jobs: it prohibited a long line from
starting a block *and* supplied negative length observations to every BIB
state.  The second job contradicts the intended contract that long lines may
be included inside an established bibliography.  This experiment keeps the
hard start constraint and tests removing length observations from CRF
emissions.  Document position is tested independently because bibliographies
can occur after chapters as well as at the end of a document.
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

from .bibliography_entry_blocks import BlockConfig, evaluate_prediction
from .bibliography_entry_coherence import (
    AnchoredCoherenceConfig,
    filter_anchored_components,
    is_safe_candidate,
)
from .bibliography_entry_models import load_table
from .bibliography_entry_sequence import (
    FEATURE_NAMES_BASE,
    constrained_viterbi,
    make_examples,
)
from .feature_crf import LinearChainCRF, train_model
from .features import TAGS


SCHEMA_VERSION = "bibliography-entry-b1-observation-ablation-oof-v1"
VARIANTS = {
    "no_length": ("log1p_char_length", "over_seed_length_limit"),
    "no_position": ("physical_position",),
    "no_length_or_position": (
        "log1p_char_length",
        "over_seed_length_limit",
        "physical_position",
    ),
}


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


def _train_fold(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        table_dir,
        probability_path,
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
    dropped = VARIANTS[variant]
    fit_docs = [
        index
        for index, document in enumerate(table.documents)
        if int(document["fold"]) != fold
    ]
    holdout_docs = [
        index
        for index, document in enumerate(table.documents)
        if int(document["fold"]) == fold
    ]
    fit_examples = make_examples(
        table,
        probability,
        fit_docs,
        seed_length_limit=seed_length_limit,
        include_header=False,
        dropped_feature_names=dropped,
    )
    holdout_examples = make_examples(
        table,
        probability,
        holdout_docs,
        seed_length_limit=seed_length_limit,
        include_header=False,
        dropped_feature_names=dropped,
    )
    model = LinearChainCRF(
        len(FEATURE_NAMES_BASE), seed=seed + fold, active_classes=("BIB",)
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
    predictions = {
        float(bias): [] for bias in deletion_biases
    }
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
        row["variant"] == "no_length",
        -int(row["coherence_config"]["minimum_anchor_count"]),
        -int(row["coherence_config"]["minimum_component_lines"]),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    table = load_table(args.table_dir, expected_split="train")
    line_root = Path(args.line_oof_dir).resolve()
    block_root = Path(args.block_oof_dir).resolve()
    block_report_path = block_root / "block_oof_report.json"
    block_report = json.loads(block_report_path.read_text(encoding="utf-8"))
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
                "feature_names": list(FEATURE_NAMES_BASE),
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
                raw_predictions[(variant, float(bias))][
                    result["indices"]
                ] = values

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
                            "metrics": evaluate_prediction(table, prediction),
                        }
                    )
    safe_rows = [row for row in rows if is_safe_candidate(row)]
    selected = max(safe_rows, key=_selection_key) if safe_rows else None
    diagnostic_highest_recall = max(rows, key=_selection_key)
    if selected is not None:
        selected_coherence = AnchoredCoherenceConfig(
            **selected["coherence_config"]
        )
        selected_prediction = filter_anchored_components(
            table,
            raw_predictions[
                (selected["variant"], selected_coherence.deletion_bias)
            ],
            probability,
            block_config=block_config,
            config=selected_coherence,
        )
        _save_array(
            output_dir / "selected_oof_prediction.npy", selected_prediction
        )
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
                "folds": sorted(
                    fold_rows[variant], key=lambda row: row["fold"]
                ),
            }
            for variant in args.variants
        },
        "candidate_count": len(rows),
        "safe_candidate_count": len(safe_rows),
        "candidates": rows,
        "selected": selected,
        "diagnostic_highest_recall_candidate": diagnostic_highest_recall,
        "selection_rule": "prefer precision>=0.99 and <=0.02 spurious blocks per zero-BIB document, then maximize token and line recall; prefer the narrower no-length ablation on ties",
        "feature_reference": {
            "no_length": "Remove line-length penalties from BIB emissions while retaining the hard rule that a long line cannot start a block.",
            "no_position": "Remove the document-position prior so chapter bibliographies and early publication lists are not disadvantaged.",
            "no_length_or_position": "Test both independent removals together.",
        },
        "block_config": asdict(block_config),
        "training": {
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "l2": float(args.l2),
            "gradient_clip": float(args.gradient_clip),
            "workers": int(args.workers),
        },
        "input_hashes": {
            "block_oof_report": _sha256(block_report_path),
            "line_oof_probability": _sha256(probability_path),
        },
        "code_commit": str(args.code_commit),
        "slurm_job_id": str(args.slurm_job_id),
        "validation_opened": False,
        "production_eligible": False,
    }
    _write_json(output_dir / "sequence_ablation_oof_report.json", report)
    receipt = {
        **report,
        "outputs": {
            str(path.relative_to(output_dir)): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(output_dir.rglob("*"))
            if path.is_file()
        },
    }
    _write_json(output_dir / "receipt.json", receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--line-oof-dir", required=True)
    parser.add_argument("--block-oof-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arm", default="D1")
    parser.add_argument(
        "--variants", nargs="+", choices=tuple(VARIANTS), default=tuple(VARIANTS)
    )
    parser.add_argument(
        "--deletion-biases",
        type=float,
        nargs="+",
        default=(-1.0, -0.5, 0.0, 0.5, 1.0),
    )
    parser.add_argument(
        "--minimum-anchor-counts", type=int, nargs="+", default=(1, 2, 3)
    )
    parser.add_argument(
        "--minimum-component-lines", type=int, nargs="+", default=(3, 5)
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
