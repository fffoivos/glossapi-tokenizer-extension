#!/usr/bin/env python3
"""One-shot evaluation of frozen candidates on consensus-silver labels."""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .bibliography_entry_blocks import blocks_from_mask
from .bibliography_nextgen_freeze import SCHEMA_VERSION as FREEZE_SCHEMA
from .bibliography_nextgen_unseen_predict import SCHEMA_VERSION as PREDICTION_SCHEMA
from .contract import sha256_file


SCHEMA_VERSION = "bibliography-nextgen-one-shot-consensus-silver-test-v1"
_MARKDOWN_HEADING = re.compile(r"^\s*#{1,6}\s+\S")
_IMAGE_MARKER = re.compile(r"^\s*<!--\s*image\s*-->\s*$", re.IGNORECASE)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected object")
            yield value


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _binary_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    trusted: np.ndarray,
    lengths: np.ndarray,
) -> dict[str, Any]:
    tp_mask = prediction & truth & trusted
    fp_mask = prediction & ~truth & trusted
    fn_mask = ~prediction & truth & trusted
    tp, fp, fn = (int(np.count_nonzero(value)) for value in (tp_mask, fp_mask, fn_mask))
    char_tp, char_fp, char_fn = (
        int(lengths[value].astype(np.int64).sum())
        for value in (tp_mask, fp_mask, fn_mask)
    )
    return {
        "trusted_line_count": int(np.count_nonzero(trusted)),
        "line_precision": tp / (tp + fp) if tp + fp else 1.0,
        "line_recall": tp / (tp + fn) if tp + fn else 0.0,
        "line_tp": tp,
        "line_fp": fp,
        "line_fn": fn,
        "char_precision": char_tp / (char_tp + char_fp) if char_tp + char_fp else 1.0,
        "char_recall": char_tp / (char_tp + char_fn) if char_tp + char_fn else 0.0,
        "char_tp": char_tp,
        "char_fp": char_fp,
        "char_fn": char_fn,
    }


def _fp_kind(index: int, gold: np.ndarray, trusted: np.ndarray) -> str:
    for distance in range(1, 4):
        if index - distance >= 0 and gold[index - distance] and trusted[index - distance]:
            return "downward_boundary_overrun"
        if index + distance < len(gold) and gold[index + distance] and trusted[index + distance]:
            return "upward_boundary_overrun"
    return "spurious_or_distant"


def _evaluate_candidate(
    prediction: np.ndarray,
    truth: np.ndarray,
    trusted: np.ndarray,
    lengths: np.ndarray,
    abs_indices: np.ndarray,
    documents: Sequence[Mapping[str, Any]],
    source_documents: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result = _binary_metrics(prediction, truth, trusted, lengths)
    by_source: dict[str, Any] = {}
    fp_kinds: collections.Counter[str] = collections.Counter()
    fp_structure: collections.Counter[str] = collections.Counter()
    document_errors = []
    predicted_blocks = gold_blocks = spurious = zero_documents = 0
    for metadata, source_document in zip(documents, source_documents, strict=True):
        start, end = int(metadata["line_start"]), int(metadata["line_end"])
        local_prediction = prediction[start:end]
        local_truth = truth[start:end]
        local_trusted = trusted[start:end]
        local_lengths = lengths[start:end]
        local_abs = abs_indices[start:end]
        source = str(metadata["source"])
        source_mask = np.ones(end - start, dtype=bool)
        source_metrics = by_source.setdefault(
            source,
            {"prediction": [], "truth": [], "trusted": [], "lengths": []},
        )
        source_metrics["prediction"].append(local_prediction)
        source_metrics["truth"].append(local_truth)
        source_metrics["trusted"].append(local_trusted)
        source_metrics["lengths"].append(local_lengths)
        local_predicted_blocks = blocks_from_mask(local_prediction, local_abs)
        local_gold_blocks = blocks_from_mask(local_truth & local_trusted, local_abs)
        predicted_blocks += len(local_predicted_blocks)
        gold_blocks += len(local_gold_blocks)
        if not (local_truth & local_trusted).any():
            zero_documents += 1
            spurious += len(local_predicted_blocks)
        fp = local_prediction & ~local_truth & local_trusted
        fn = ~local_prediction & local_truth & local_trusted
        if not fp.any() and not fn.any():
            continue
        lines = source_document["lines"]
        fp_rows = []
        for offset in np.flatnonzero(fp):
            text = str(lines[int(offset)].get("text", ""))
            kind = _fp_kind(int(offset), local_truth, local_trusted)
            fp_kinds[kind] += 1
            if _MARKDOWN_HEADING.match(text):
                fp_structure["markdown_heading"] += 1
            elif _IMAGE_MARKER.match(text):
                fp_structure["image_marker"] += 1
            elif not text.strip():
                fp_structure["blank"] += 1
            else:
                fp_structure["ordinary_line"] += 1
            fp_rows.append(
                {
                    "offset": int(offset),
                    "abs_idx": int(local_abs[offset]),
                    "kind": kind,
                    "text": text,
                }
            )
        document_errors.append(
            {
                "document_id": metadata["document_id"],
                "source": source,
                "false_positive_line_count": int(np.count_nonzero(fp)),
                "false_negative_line_count": int(np.count_nonzero(fn)),
                "false_positive_lines": fp_rows,
            }
        )
    normalized_sources = {}
    for source, values in sorted(by_source.items()):
        normalized_sources[source] = _binary_metrics(
            np.concatenate(values["prediction"]),
            np.concatenate(values["truth"]),
            np.concatenate(values["trusted"]),
            np.concatenate(values["lengths"]),
        )
    result.update(
        {
            "predicted_block_count": predicted_blocks,
            "gold_block_count": gold_blocks,
            "spurious_blocks_per_zero_bib_document": spurious / zero_documents
            if zero_documents
            else 0.0,
            "false_positive_lines_by_topology": dict(sorted(fp_kinds.items())),
            "false_positive_lines_by_structure": dict(sorted(fp_structure.items())),
            "by_source": normalized_sources,
            "top_false_positive_documents": sorted(
                document_errors,
                key=lambda row: (-row["false_positive_line_count"], row["document_id"]),
            )[:25],
            "top_false_negative_documents": sorted(
                document_errors,
                key=lambda row: (-row["false_negative_line_count"], row["document_id"]),
            )[:25],
        }
    )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    freeze_path = Path(args.candidate_freeze).resolve()
    prediction_root = Path(args.prediction_dir).resolve()
    seal_path = Path(args.test_seal).resolve()
    documents_path = Path(args.documents).resolve()
    line_key_path = Path(args.line_key).resolve()
    labels_path = Path(args.labels).resolve()
    feature_root = Path(args.feature_table_dir).resolve()
    output = Path(args.output).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    prediction_report = json.loads(
        (prediction_root / "report.json").read_text(encoding="utf-8")
    )
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if (
        freeze.get("schema_version") != FREEZE_SCHEMA
        or freeze.get("status") != "frozen_before_test_open"
        or prediction_report.get("schema_version") != PREDICTION_SCHEMA
        or prediction_report.get("test_labels_opened") is not False
        or prediction_report.get("inputs", {}).get("candidate_freeze_sha256")
        != sha256_file(freeze_path)
        or freeze.get("test_seal", {}).get("sha256") != sha256_file(seal_path)
    ):
        raise ValueError("prediction/freeze/test seal provenance mismatch")
    wanted_hashes = seal["sealed_hashes"]
    observed_hashes = {
        "documents_sha256": sha256_file(documents_path),
        "line_key_sha256": sha256_file(line_key_path),
        "labels_sha256": sha256_file(labels_path),
    }
    if any(observed_hashes[name] != wanted_hashes[name] for name in observed_hashes):
        raise ValueError("sealed test input hash drift")
    source_documents = list(_iter_jsonl(documents_path))
    line_keys = list(_iter_jsonl(line_key_path))
    labels = list(_iter_jsonl(labels_path))
    feature_documents = list(_iter_jsonl(feature_root / "documents.jsonl"))
    feature_lines = list(_iter_jsonl(feature_root / "line_ids.jsonl"))
    lengths = np.load(feature_root / "char_lengths.npy", mmap_mode="r", allow_pickle=False)
    absolute = np.load(feature_root / "abs_indices.npy", mmap_mode="r", allow_pickle=False)
    if not (len(line_keys) == len(labels) == len(feature_lines) == len(lengths)):
        raise ValueError("sealed evaluation line inventories differ")
    if len(source_documents) != len(feature_documents):
        raise ValueError("sealed evaluation document inventories differ")
    for index, (source_document, feature_document) in enumerate(
        zip(source_documents, feature_documents, strict=True)
    ):
        if str(source_document.get("document_id")) != str(
            feature_document.get("document_id")
        ):
            raise ValueError(f"sealed document alignment failure at {index}")
    truth = np.zeros(len(labels), dtype=bool)
    trusted = np.zeros(len(labels), dtype=bool)
    decisions: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for index, (key, label) in enumerate(zip(line_keys, labels, strict=True)):
        identity = (str(key["document_id"]), str(key["line_id"]), int(key["abs_idx"]))
        if identity != (
            str(label["document_id"]),
            str(label["line_id"]),
            int(label["abs_idx"]),
        ):
            raise ValueError(f"sealed key/label alignment failure at {index}")
        if identity in decisions:
            raise ValueError(f"duplicate sealed line identity at {index}")
        decisions[identity] = label["tasks"]["bibliography_membership"]
    for index, feature_line in enumerate(feature_lines):
        identity = (
            str(feature_line["document_id"]),
            str(feature_line["line_id"]),
            int(feature_line["abs_idx"]),
        )
        decision = decisions.pop(identity, None)
        if decision is None:
            raise ValueError(f"sealed feature line lacks a label at {index}")
        trusted[index] = bool(decision["trusted"])
        truth[index] = decision["label"] == "BIB"
    if decisions:
        raise ValueError("sealed labels contain lines absent from the feature table")
    candidate_metrics = []
    by_name = {row["name"]: row for row in prediction_report["candidates"]}
    for frozen in freeze["candidates"]:
        prediction_row = by_name.get(frozen["name"])
        if prediction_row is None:
            raise ValueError(f"frozen candidate lacks prediction: {frozen['name']}")
        prediction_path = Path(prediction_row["prediction_path"]).resolve()
        if sha256_file(prediction_path) != prediction_row["prediction_sha256"]:
            raise ValueError(f"prediction drift: {frozen['name']}")
        prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False).astype(bool)
        if prediction.shape != truth.shape:
            raise ValueError(f"prediction shape mismatch: {frozen['name']}")
        candidate_metrics.append(
            {
                "name": frozen["name"],
                "model_kind": frozen["model_kind"],
                "development_gate_passed": frozen["development_gate_passed"],
                "development_selection_tier": frozen["development_selection_tier"],
                "development_metrics": frozen["development_metrics"],
                "test_metrics": _evaluate_candidate(
                    prediction,
                    truth,
                    trusted,
                    lengths,
                    absolute,
                    feature_documents,
                    source_documents,
                ),
            }
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_one_shot_frozen_consensus_silver_test",
        "test_opened": True,
        "test_labels_opened": True,
        "test_used_for_training_or_calibration": False,
        "candidate_ranking_or_selection_after_test": False,
        "human_gold": False,
        "label_semantics": seal["label_semantics"],
        "document_count": len(source_documents),
        "line_count": len(labels),
        "trusted_membership_line_count": int(np.count_nonzero(trusted)),
        "trusted_membership_coverage": float(np.mean(trusted)),
        "candidates": candidate_metrics,
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "inputs": {
            "candidate_freeze_sha256": sha256_file(freeze_path),
            "prediction_receipt_sha256": sha256_file(prediction_root / "receipt.json"),
            "test_seal_sha256": sha256_file(seal_path),
            **observed_hashes,
        },
    }
    _write_json_new(output, report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-freeze", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--test-seal", required=True)
    parser.add_argument("--documents", required=True)
    parser.add_argument("--line-key", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--feature-table-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
