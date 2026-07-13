#!/usr/bin/env python3
"""Train-only B1 recall sweep with a conservative anchored-component filter.

The original B1 calibration chose a large deletion bias in order to make every
predicted line precise in isolation.  That is stricter than the actual task:
bibliography entries normally occur in blocks.  This experiment lets B1 emit a
higher-recall sequence and then rejects any proposed component that is not
supported by several independently strong entry lines.

Validation input is deliberately not accepted by this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bibliography_entry_blocks import (
    BlockConfig,
    attach_h0_document,
    blocks_from_mask,
    evaluate_prediction,
)
from .bibliography_entry_models import load_table
from .bibliography_entry_sequence import constrained_viterbi, make_examples
from .feature_crf import LinearChainCRF
from .features import TAGS


SCHEMA_VERSION = "bibliography-entry-anchored-coherence-oof-v1"


@dataclass(frozen=True)
class AnchoredCoherenceConfig:
    """Parameters with one distinct job each.

    ``deletion_bias`` controls how readily B1 proposes bibliography lines.
    ``minimum_anchor_count`` is an independent component-level evidence gate.
    ``minimum_component_lines`` removes isolated citation-like fragments.
    """

    deletion_bias: float
    minimum_anchor_count: int
    minimum_component_lines: int
    strong_anchor_probability: float = 0.70


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


def filter_anchored_components(
    table: Any,
    raw_prediction: np.ndarray,
    entry_probability: np.ndarray,
    *,
    block_config: BlockConfig,
    config: AnchoredCoherenceConfig,
    attach_headers: bool = True,
) -> np.ndarray:
    """Keep only coherent predicted components supported by trusted anchors.

    A strong anchor is the already-frozen entry probability threshold combined
    with the existing rule that a long line cannot establish a block.  Long or
    weak lines remain eligible *inside* a retained component.
    """

    if not (
        len(raw_prediction)
        == len(entry_probability)
        == len(table.targets)
    ):
        raise ValueError("coherence arrays must match the feature table")
    result = np.zeros(len(raw_prediction), dtype=bool)
    for document in table.documents:
        start, end = int(document["line_start"]), int(document["line_end"])
        local_raw = np.asarray(raw_prediction[start:end], dtype=bool)
        local_probability = np.asarray(entry_probability[start:end])
        local_lengths = np.asarray(table.char_lengths[start:end])
        local_absolute = np.asarray(table.abs_indices[start:end])
        for component_start, component_end in blocks_from_mask(
            local_raw, local_absolute
        ):
            line_count = component_end - component_start + 1
            component = slice(component_start, component_end + 1)
            anchor_count = int(
                np.count_nonzero(
                    (local_probability[component]
                    >= config.strong_anchor_probability)
                    & (
                        local_lengths[component]
                        <= block_config.seed_length_limit
                    )
                )
            )
            if (
                line_count >= config.minimum_component_lines
                and anchor_count >= config.minimum_anchor_count
            ):
                result[
                    start + component_start : start + component_end + 1
                ] = True
        if attach_headers:
            result[start:end] = attach_h0_document(
                result[start:end],
                local_probability,
                table.header_kinds[start:end],
                local_absolute,
                block_config,
            )
    return result


def _raw_oof_prediction(
    table: Any,
    entry_probability: np.ndarray,
    model_dir: Path,
    *,
    arm: str,
    variant: str,
    block_config: BlockConfig,
    deletion_bias: float,
) -> np.ndarray:
    include_header = variant == "with_header"
    prediction = np.zeros(len(table.targets), dtype=bool)
    for fold in range(int(table.manifest["n_folds"])):
        model, metadata = LinearChainCRF.load(
            model_dir / f"{arm}.{variant}.fold{fold}.npz"
        )
        if metadata.get("arm") != arm or metadata.get("variant") != variant:
            raise ValueError("B1 checkpoint metadata does not match requested arm")
        document_indices = [
            index
            for index, document in enumerate(table.documents)
            if int(document["fold"]) == fold
        ]
        examples = make_examples(
            table,
            entry_probability,
            document_indices,
            seed_length_limit=block_config.seed_length_limit,
            include_header=include_header,
        )
        for example in examples:
            tags = constrained_viterbi(
                model,
                example.features,
                example.char_lengths,
                seed_length_limit=block_config.seed_length_limit,
                deletion_bias=deletion_bias,
            )
            prediction[list(example.line_indices)] = [
                TAGS[int(tag)] != "O" for tag in tags
            ]
    return prediction


def _selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = row["metrics"]
    safe = (
        float(metrics["line_precision"]) >= 0.99
        and float(metrics["spurious_blocks_per_zero_block_document"])
        <= 0.02
    )
    return (
        safe,
        float(metrics["token_recall"]),
        float(metrics["line_recall"]),
        float(metrics["token_precision"]),
        -int(row["config"]["minimum_anchor_count"]),
        -int(row["config"]["minimum_component_lines"]),
        -abs(float(row["config"]["deletion_bias"])),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    table = load_table(args.table_dir, expected_split="train")
    line_root = Path(args.line_oof_dir).resolve()
    b1_root = Path(args.b1_oof_dir).resolve()
    block_root = Path(args.block_oof_dir).resolve()
    b1_report_path = b1_root / "b1_oof_report.json"
    block_report_path = block_root / "block_oof_report.json"
    b1_report = json.loads(b1_report_path.read_text(encoding="utf-8"))
    block_report = json.loads(block_report_path.read_text(encoding="utf-8"))
    if b1_report.get("validation_opened") is not False:
        raise ValueError("B1 input does not prove validation isolation")
    arm = str(args.arm or b1_report["selected_arm"])
    variant = str(args.variant)
    if f"{arm}:{variant}" not in b1_report["variants"]:
        raise ValueError("requested B1 arm/variant is absent")
    block_config = BlockConfig(**block_report["arms"][arm]["selected_config"])
    probability_path = line_root / f"{arm}.oof_probability.npy"
    entry_probability = np.load(
        probability_path, mmap_mode="r", allow_pickle=False
    )
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    rows: list[dict[str, Any]] = []
    raw_by_bias: dict[float, np.ndarray] = {}
    for deletion_bias in args.deletion_biases:
        raw = _raw_oof_prediction(
            table,
            entry_probability,
            b1_root / "models",
            arm=arm,
            variant=variant,
            block_config=block_config,
            deletion_bias=float(deletion_bias),
        )
        raw_by_bias[float(deletion_bias)] = raw
        for minimum_anchor_count in args.minimum_anchor_counts:
            for minimum_component_lines in args.minimum_component_lines:
                config = AnchoredCoherenceConfig(
                    deletion_bias=float(deletion_bias),
                    minimum_anchor_count=int(minimum_anchor_count),
                    minimum_component_lines=int(minimum_component_lines),
                )
                prediction = filter_anchored_components(
                    table,
                    raw,
                    entry_probability,
                    block_config=block_config,
                    config=config,
                )
                rows.append(
                    {
                        "config": asdict(config),
                        "metrics": evaluate_prediction(table, prediction),
                    }
                )
    selected = max(rows, key=_selection_key)
    selected_config = AnchoredCoherenceConfig(**selected["config"])
    selected_prediction = filter_anchored_components(
        table,
        raw_by_bias[selected_config.deletion_bias],
        entry_probability,
        block_config=block_config,
        config=selected_config,
    )
    _save_array(output_dir / "selected_oof_prediction.npy", selected_prediction)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_train_oof_validation_unopened",
        "arm": arm,
        "variant": variant,
        "feature_reference": {
            "deletion_bias": "How readily B1 proposes bibliography lines before block filtering.",
            "minimum_anchor_count": "How many independently strong, normal-length entry lines a proposed component must contain.",
            "minimum_component_lines": "The smallest retained component; isolated citation-like fragments are rejected.",
            "strong_anchor_probability": "The frozen entry-probability threshold used only to certify a proposed component.",
        },
        "block_config": asdict(block_config),
        "candidate_count": len(rows),
        "candidates": rows,
        "selected": selected,
        "selection_rule": "prefer precision>=0.99 and <=0.02 spurious blocks per zero-BIB document, then maximize token and line recall",
        "input_hashes": {
            "b1_oof_report": _sha256(b1_report_path),
            "block_oof_report": _sha256(block_report_path),
            "line_oof_probability": _sha256(probability_path),
        },
        "code_commit": str(args.code_commit),
        "slurm_job_id": str(args.slurm_job_id),
        "validation_opened": False,
        "production_eligible": False,
    }
    _write_json(output_dir / "coherence_oof_report.json", report)
    receipt = {
        **report,
        "outputs": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(output_dir.iterdir())
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
    parser.add_argument("--b1-oof-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arm")
    parser.add_argument("--variant", choices=("no_header", "with_header"), default="no_header")
    parser.add_argument(
        "--deletion-biases",
        type=float,
        nargs="+",
        default=(-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0),
    )
    parser.add_argument(
        "--minimum-anchor-counts", type=int, nargs="+", default=(1, 2, 3, 5)
    )
    parser.add_argument(
        "--minimum-component-lines", type=int, nargs="+", default=(3, 5)
    )
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
