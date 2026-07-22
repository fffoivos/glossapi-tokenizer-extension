#!/usr/bin/env python3
"""Freeze train-only selection, then run one retrospective validation pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bibliography_entry_blocks import (
    BlockConfig,
    _breakdowns,
    decode_table,
    evaluate_prediction,
)
from .bibliography_entry_dataset import run as materialize_table
from .bibliography_entry_models import (
    PINNED_SKLEARN_VERSION,
    TARGET_MASK,
    _fit_d1,
    _fit_linear,
    _sigmoid,
    binary_metrics,
    load_table,
)
from .bibliography_entry_semimarkov import decode_candidates, generate_candidates
from .bibliography_entry_sequence import (
    FEATURE_NAMES_BASE,
    _select_bias,
    attach_h0_table,
    constrained_viterbi,
    make_examples,
)
from .feature_crf import LinearChainCRF, train_model
from .features import TAGS


FREEZE_SCHEMA = "bibliography-entry-frozen-selection-v1"
VALIDATION_SCHEMA = "bibliography-entry-retrospective-validation-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _save_array(path: Path, value: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)


def _eligible(metrics: Mapping[str, Any]) -> bool:
    return (
        float(metrics["line_precision"]) >= 0.99
        and float(metrics["spurious_blocks_per_zero_block_document"]) <= 0.02
    )


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    line_root = Path(args.line_oof_dir).resolve()
    block_root = Path(args.block_oof_dir).resolve()
    b1_root = Path(args.b1_oof_dir).resolve()
    b2_root = Path(args.b2_oof_dir).resolve()
    line = _json(line_root / "line_oof_report.json")
    block = _json(block_root / "block_oof_report.json")
    b1 = _json(b1_root / "b1_oof_report.json")
    b2 = _json(b2_root / "b2_oof_report.json")
    if any(report.get("validation_opened") is not False for report in (line, block, b1, b2)):
        raise ValueError("a train-only selection receipt does not prove validation isolation")

    b0_arm = str(block["primary_arm"])
    candidates = [
        {
            "architecture": "B0_H0",
            "complexity": 0,
            "line_arm": b0_arm,
            "metrics": block["arms"][b0_arm]["selected_b0_plus_h0_metrics"],
        },
        {
            "architecture": "B1",
            "complexity": 1,
            "line_arm": str(b1["selected_arm"]),
            "variant": str(b1["selected_key"]).split(":", 1)[1],
            "metrics": b1["variants"][b1["selected_key"]]["metrics"],
        },
    ]
    if b2.get("b2_required") and b2.get("accepted_over_b1"):
        candidates.append(
            {
                "architecture": "B2",
                "complexity": 2,
                "line_arm": str(b2["selected_line_arm"]),
                "metrics": b2["metrics"],
            }
        )
    eligible = [row for row in candidates if _eligible(row["metrics"])]
    if eligible:
        best_recall = max(float(row["metrics"]["token_recall"]) for row in eligible)
        near_best = [
            row
            for row in eligible
            if best_recall - float(row["metrics"]["token_recall"]) <= 0.005
        ]
        selected = min(near_best, key=lambda row: int(row["complexity"]))
        selection_status = "passed_train_only_safety_gate"
    else:
        selected = max(
            candidates,
            key=lambda row: (
                float(row["metrics"]["token_f0_5"]),
                float(row["metrics"]["line_precision"]),
                -int(row["complexity"]),
            ),
        )
        selection_status = "research_only_no_candidate_met_safety_gate"

    line_arm = str(selected["line_arm"])
    line_selection = line["arms"][line_arm]["selected"]
    if line_arm == "L0":
        point_threshold = int(
            statistics.median(
                int(row["point_threshold"])
                for row in line["arms"]["L0"]["folds"]
            )
        )
        line_selection = {**line_selection, "final_point_threshold": point_threshold}
    frozen = {
        "schema_version": FREEZE_SCHEMA,
        "status": selection_status,
        "selection_rule": "among precision>=0.99 and <=0.02 spurious blocks/zero-doc, choose lowest complexity within 0.005 token recall of best",
        "selected": selected,
        "line_model": {"arm": line_arm, "selection": line_selection},
        "b0_h0_config": block["arms"][line_arm]["selected_config"],
        "b1_training": b1["training"],
        "b1_variant": selected.get("variant"),
        "train_candidates": candidates,
        "input_hashes": {
            "line_oof_report": _sha256(line_root / "line_oof_report.json"),
            "block_oof_report": _sha256(block_root / "block_oof_report.json"),
            "b1_oof_report": _sha256(b1_root / "b1_oof_report.json"),
            "b2_oof_report": _sha256(b2_root / "b2_oof_report.json"),
        },
        "code_commit": str(args.code_commit),
        "slurm_job_id": str(args.slurm_job_id),
        "validation_opened": False,
        "production_eligible": False,
    }
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    _write_json(output_dir / "frozen_config.json", frozen)
    receipt = {
        **frozen,
        "outputs": {
            "frozen_config.json": {
                "bytes": (output_dir / "frozen_config.json").stat().st_size,
                "sha256": _sha256(output_dir / "frozen_config.json"),
            }
        },
    }
    _write_json(output_dir / "receipt.json", receipt)
    return receipt


def _fit_final_line_model(
    train: Any,
    validation: Any,
    frozen: Mapping[str, Any],
) -> tuple[np.ndarray, Any]:
    arm = str(frozen["line_model"]["arm"])
    labelled = train.targets != TARGET_MASK
    if arm == "L0":
        threshold = int(frozen["line_model"]["selection"]["final_point_threshold"])
        points = np.count_nonzero(validation.counts, axis=1).astype(np.float32)
        return _sigmoid(points - threshold).astype(np.float32), {
            "kind": "equal_presence",
            "point_threshold": threshold,
        }
    if arm == "D1":
        model, transform = _fit_d1(train.counts, train.targets, labelled, seed=1729)
    else:
        model, transform = _fit_linear(
            arm,
            train.counts,
            train.targets,
            labelled,
            c_value=float(frozen["line_model"]["selection"]["C"]),
            seed=1729,
        )
    probability = model.predict_proba(transform.apply(validation.counts))[:, 1].astype(np.float32)
    return probability, (model, transform)


def _predict_final_b1(
    train: Any,
    validation: Any,
    train_probability: np.ndarray,
    validation_probability: np.ndarray,
    frozen: Mapping[str, Any],
) -> tuple[np.ndarray, LinearChainCRF, float, list[float]]:
    include_header = frozen.get("b1_variant") == "with_header"
    config = BlockConfig(**frozen["b0_h0_config"])
    train_examples = make_examples(
        train,
        train_probability,
        list(range(len(train.documents))),
        seed_length_limit=config.seed_length_limit,
        include_header=include_header,
    )
    validation_examples = make_examples(
        validation,
        validation_probability,
        list(range(len(validation.documents))),
        seed_length_limit=config.seed_length_limit,
        include_header=include_header,
    )
    training = frozen["b1_training"]
    model = LinearChainCRF(
        len(FEATURE_NAMES_BASE) + int(include_header),
        seed=2718,
        active_classes=("BIB",),
    )
    history = train_model(
        model,
        train_examples,  # type: ignore[arg-type]
        epochs=int(training["epochs"]),
        learning_rate=float(training["learning_rate"]),
        l2=float(training["l2"]),
        gradient_clip=float(training["gradient_clip"]),
        seed=2718,
    )
    deletion_bias, _rows = _select_bias(
        train_examples,
        model,
        seed_length_limit=config.seed_length_limit,
    )
    prediction = np.zeros(len(validation.targets), dtype=bool)
    for example in validation_examples:
        tags = constrained_viterbi(
            model,
            example.features,
            example.char_lengths,
            seed_length_limit=config.seed_length_limit,
            deletion_bias=deletion_bias,
        )
        prediction[list(example.line_indices)] = [TAGS[int(tag)] != "O" for tag in tags]
    return prediction, model, deletion_bias, history


def validate(args: argparse.Namespace) -> dict[str, Any]:
    frozen_path = Path(args.frozen_config).resolve()
    frozen = _json(frozen_path)
    if frozen.get("schema_version") != FREEZE_SCHEMA or frozen.get("validation_opened") is not False:
        raise ValueError("invalid or already-open frozen configuration")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    validation_table_dir = output_dir / "validation_table"
    materialize_table(
        argparse.Namespace(
            input=str(Path(args.input).resolve()),
            output_dir=str(validation_table_dir),
            split="validation",
            workers=int(args.workers),
            n_folds=5,
            fold_seed="retrospective-validation-not-for-selection",
            expected_input_sha256=str(args.expected_input_sha256),
            code_commit=str(args.code_commit),
            slurm_job_id=str(args.slurm_job_id),
        )
    )
    train = load_table(args.train_table_dir)
    validation = load_table(validation_table_dir, expected_split="validation")
    if int(validation.manifest["document_count"]) != 274:
        raise ValueError("retrospective validation document count is not 274")
    probability, final_line_model = _fit_final_line_model(train, validation, frozen)
    _save_array(output_dir / "validation_line_probability.npy", probability)
    with (output_dir / "final_line_model.pkl").open("xb") as handle:
        pickle.dump(final_line_model, handle, protocol=5)
    labelled = validation.targets != TARGET_MASK
    line_metrics = binary_metrics(
        validation.targets[labelled],
        probability[labelled],
        float(frozen["line_model"]["selection"]["threshold"]),
    )

    architecture = str(frozen["selected"]["architecture"])
    config = BlockConfig(**frozen["b0_h0_config"])
    if architecture == "B0_H0":
        prediction = decode_table(validation, probability, config, attach_headers=True)
        final_block_model: Any = {"architecture": architecture, "config": frozen["b0_h0_config"]}
    elif architecture == "B1":
        train_probability = np.load(
            Path(args.line_oof_dir) / f"{frozen['line_model']['arm']}.oof_probability.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        prediction, model, deletion_bias, history = _predict_final_b1(
            train,
            validation,
            train_probability,
            probability,
            frozen,
        )
        prediction = attach_h0_table(validation, prediction, probability, config)
        model.save(
            output_dir / "final_b1_model.npz",
            {
                "schema_version": VALIDATION_SCHEMA,
                "active_classes": ["BIB"],
                "deletion_bias": deletion_bias,
                "history": history,
                "frozen_config_sha256": _sha256(frozen_path),
            },
        )
        final_block_model = {"architecture": architecture, "deletion_bias": deletion_bias, "history": history}
    elif architecture == "B2":
        import sklearn
        from sklearn.ensemble import HistGradientBoostingClassifier

        if sklearn.__version__ != PINNED_SKLEARN_VERSION:
            raise RuntimeError("scikit-learn version differs from the pinned ladder")
        train_probability = np.load(
            Path(args.line_oof_dir) / f"{frozen['line_model']['arm']}.oof_probability.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        train_features, _train_docs, _train_starts, _train_ends, train_labels = generate_candidates(train, train_probability, config)
        model = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=75,
            max_depth=3,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=3141,
        )
        model.fit(train_features, train_labels)
        features, documents, starts, ends, _labels = generate_candidates(validation, probability, config)
        scores = model.predict_proba(features)[:, 1]
        prediction = decode_candidates(validation, scores, documents, starts, ends, probability, config)
        with (output_dir / "final_b2_model.pkl").open("xb") as handle:
            pickle.dump(model, handle, protocol=5)
        final_block_model = {"architecture": architecture, "sklearn_version": sklearn.__version__}
    else:
        raise ValueError(f"unknown frozen architecture {architecture}")

    _save_array(output_dir / "validation_block_prediction.npy", prediction)
    block_metrics = evaluate_prediction(validation, prediction)
    report = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "passed_single_retrospective_validation",
        "evidence_scope": "LLM_silver_retrospective_validation_not_human_gold",
        "code_commit": str(args.code_commit),
        "slurm_job_id": str(args.slurm_job_id),
        "frozen_config_sha256": _sha256(frozen_path),
        "selected_architecture": architecture,
        "selected_line_arm": frozen["line_model"]["arm"],
        "line_metrics": line_metrics,
        "block_metrics": block_metrics,
        "block_breakdowns": _breakdowns(validation, prediction),
        "final_block_model": final_block_model,
        "validation_document_count": len(validation.documents),
        "validation_opened_once": True,
        "tuning_after_validation_forbidden": True,
        "production_eligible": False,
    }
    _write_json(output_dir / "validation_report.json", report)
    outputs = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            outputs[str(path.relative_to(output_dir))] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    receipt = {**report, "outputs": outputs}
    _write_json(output_dir / "receipt.json", receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    for name in ("line-oof-dir", "block-oof-dir", "b1-oof-dir", "b2-oof-dir", "output-dir"):
        freeze_parser.add_argument(f"--{name}", required=True)
    freeze_parser.add_argument("--code-commit", required=True)
    freeze_parser.add_argument("--slurm-job-id", required=True)
    validation_parser = subparsers.add_parser("validate")
    for name in ("input", "expected-input-sha256", "train-table-dir", "line-oof-dir", "frozen-config", "output-dir"):
        validation_parser.add_argument(f"--{name}", required=True)
    validation_parser.add_argument("--workers", type=int, default=16)
    validation_parser.add_argument("--code-commit", required=True)
    validation_parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    freeze(arguments) if arguments.command == "freeze" else validate(arguments)
