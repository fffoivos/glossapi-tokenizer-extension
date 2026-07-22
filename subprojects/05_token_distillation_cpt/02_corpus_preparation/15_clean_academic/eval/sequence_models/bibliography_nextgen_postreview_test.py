#!/usr/bin/env python3
"""Evaluate the post-review decoder/scope corrections on the opened test set."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .bibliography_nextgen_decode import DecoderConfig, decode_table
from .bibliography_nextgen_scope import (
    apply_component_threshold,
    build_component_table,
    predict_component_probability,
)
from .bibliography_nextgen_table import bib_heading_lexicon_match, feature_names
from .bibliography_nextgen_unseen_evaluate import _evaluate_candidate
from .bibliography_nextgen_unseen_features import UnseenTable
from .contract import sha256_file


SCHEMA_VERSION = "bibliography-nextgen-postreview-opened-test-v1"
_MARKDOWN_HEADING = re.compile(r"^\s*#{1,6}\s+\S")
_RULE_LINE = re.compile(r"^[\s_—–\-.]{4,}$")


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


def _load_truth(
    line_keys: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    feature_lines: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    decisions: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for index, (key, label) in enumerate(zip(line_keys, labels, strict=True)):
        identity = (str(key["document_id"]), str(key["line_id"]), int(key["abs_idx"]))
        if identity != (
            str(label["document_id"]),
            str(label["line_id"]),
            int(label["abs_idx"]),
        ):
            raise ValueError(f"test key/label mismatch at {index}")
        decisions[identity] = label["tasks"]["bibliography_membership"]
    truth = np.zeros(len(feature_lines), dtype=bool)
    trusted = np.zeros(len(feature_lines), dtype=bool)
    for index, row in enumerate(feature_lines):
        identity = (str(row["document_id"]), str(row["line_id"]), int(row["abs_idx"]))
        decision = decisions.pop(identity, None)
        if decision is None:
            raise ValueError(f"test feature line lacks decision at {index}")
        truth[index] = decision["label"] == "BIB"
        trusted[index] = bool(decision["trusted"])
    if decisions:
        raise ValueError("test labels contain lines absent from feature table")
    return truth, trusted


def _extend_features(
    old_features: np.ndarray,
    old_names: Sequence[str],
    source_documents: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, tuple[str, ...]]:
    current_names = feature_names()
    if tuple(old_names) != current_names[:-2]:
        raise ValueError("opened-test feature table is not the expected v1 prefix")
    rule: list[float] = []
    heading: list[float] = []
    for document in source_documents:
        for line in document["lines"]:
            text = str(line["text"])
            rule.append(float(bool(_RULE_LINE.fullmatch(text))))
            heading.append(
                float(
                    bool(_MARKDOWN_HEADING.match(text))
                    and bib_heading_lexicon_match(text, int(line["abs_idx"]))
                )
            )
    structural = np.column_stack((rule, heading)).astype(np.float32)
    if len(structural) != len(old_features):
        raise ValueError("opened-test source/feature line counts differ")
    features = np.column_stack((old_features, structural)).astype(np.float32)
    if features.shape != (len(old_features), len(current_names)):
        raise RuntimeError("post-review feature extension failed")
    return features, current_names


def run(args: argparse.Namespace) -> dict[str, Any]:
    feature_root = Path(args.feature_table_dir).resolve()
    documents_path = Path(args.documents).resolve()
    line_key_path = Path(args.line_key).resolve()
    labels_path = Path(args.labels).resolve()
    probability_path = Path(args.line_probability).resolve()
    baseline_path = Path(args.baseline_prediction).resolve()
    decoder_report_path = Path(args.decoder_report).resolve()
    scope_root = Path(args.scope_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)

    manifest = json.loads((feature_root / "manifest.json").read_text(encoding="utf-8"))
    decoder_report = json.loads(decoder_report_path.read_text(encoding="utf-8"))
    scope_report = json.loads((scope_root / "report.json").read_text(encoding="utf-8"))
    selected_scope = scope_report.get("selected")
    selected_decoder = decoder_report.get("deployment_near_miss")
    if not isinstance(selected_scope, Mapping) or not isinstance(selected_decoder, Mapping):
        raise ValueError("post-review development selections are incomplete")

    source_documents = list(_iter_jsonl(documents_path))
    metadata = list(_iter_jsonl(feature_root / "documents.jsonl"))
    feature_lines = list(_iter_jsonl(feature_root / "line_ids.jsonl"))
    line_keys = list(_iter_jsonl(line_key_path))
    labels = list(_iter_jsonl(labels_path))
    if len(source_documents) != len(metadata):
        raise ValueError("opened-test document inventories differ")
    for source, row in zip(source_documents, metadata, strict=True):
        if str(source["document_id"]) != str(row["document_id"]):
            raise ValueError("opened-test document order differs")

    old_features = np.load(feature_root / "features.npy", mmap_mode="r", allow_pickle=False)
    features, names = _extend_features(
        np.asarray(old_features), manifest["feature_names"], source_documents
    )
    absolute = np.load(feature_root / "abs_indices.npy", mmap_mode="r", allow_pickle=False)
    lengths = np.load(feature_root / "char_lengths.npy", mmap_mode="r", allow_pickle=False)
    auxiliary = np.load(
        feature_root / "auxiliary_scope.npy", mmap_mode="r", allow_pickle=False
    ).astype(bool)
    probability = np.load(probability_path, mmap_mode="r", allow_pickle=False)
    baseline = np.load(baseline_path, mmap_mode="r", allow_pickle=False).astype(bool)
    truth, trusted = _load_truth(line_keys, labels, feature_lines)
    if not (
        len(features)
        == len(absolute)
        == len(lengths)
        == len(auxiliary)
        == len(probability)
        == len(baseline)
        == len(truth)
    ):
        raise ValueError("opened-test arrays are not aligned")

    table = UnseenTable(metadata, absolute, lengths, names)
    decoded = decode_table(
        table,
        probability,
        features,
        names,
        DecoderConfig(**selected_decoder["config"]),
        auxiliary,
    )
    component_features, _documents, bounds, _target = build_component_table(
        table, decoded, probability, features, names
    )
    models = []
    for path in sorted((scope_root / "models").glob("fold*.pkl")):
        with path.open("rb") as handle:
            bundle = pickle.load(handle)
        if not isinstance(bundle, dict) or set(bundle) != {"kind", "scaler", "model"}:
            raise ValueError(f"unexpected scope bundle: {path}")
        models.append(bundle)
    if not models:
        raise ValueError("scope fold inventory is empty")
    component_probability = predict_component_probability(models, component_features)
    corrected = apply_component_threshold(
        decoded,
        bounds,
        component_probability,
        float(selected_scope["threshold"]),
        component_features=component_features,
        heading_rescue_floor=float(scope_report["heading_rescue_floor"]),
        image_fraction_veto=float(scope_report["image_fraction_veto"]),
        rule_fraction_veto=float(scope_report["rule_fraction_veto"]),
    )

    output.mkdir(parents=True)
    with (output / "corrected_prediction.npy").open("xb") as handle:
        np.save(handle, corrected, allow_pickle=False)
    with (output / "component_probability.npy").open("xb") as handle:
        np.save(handle, component_probability, allow_pickle=False)
    candidates = []
    for name, prediction in (("frozen_test_baseline", baseline), ("postreview_corrected", corrected)):
        candidates.append(
            {
                "name": name,
                "metrics": _evaluate_candidate(
                    prediction,
                    truth,
                    trusted,
                    lengths,
                    absolute,
                    metadata,
                    source_documents,
                ),
            }
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_postreview_opened_test_evaluation",
        "test_opened": True,
        "test_labels_opened": True,
        "test_used_to_derive_corrections": True,
        "unbiased_test_claim": False,
        "human_gold": False,
        "document_count": len(metadata),
        "line_count": len(features),
        "trusted_line_count": int(np.count_nonzero(trusted)),
        "decoder_config": selected_decoder["config"],
        "scope_threshold": selected_scope["threshold"],
        "scope_guards": {
            "heading_rescue_floor": scope_report["heading_rescue_floor"],
            "image_fraction_veto": scope_report["image_fraction_veto"],
            "rule_fraction_veto": scope_report["rule_fraction_veto"],
        },
        "component_count_before_scope": len(bounds),
        "component_count_after_scope": int(
            sum(
                bool(corrected[int(start) : int(end) + 1].any())
                for start, end in bounds
            )
        ),
        "candidates": candidates,
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "inputs": {
            "feature_receipt_sha256": sha256_file(feature_root / "receipt.json"),
            "documents_sha256": sha256_file(documents_path),
            "line_key_sha256": sha256_file(line_key_path),
            "labels_sha256": sha256_file(labels_path),
            "line_probability_sha256": sha256_file(probability_path),
            "baseline_prediction_sha256": sha256_file(baseline_path),
            "decoder_receipt_sha256": sha256_file(decoder_report_path.parent / "receipt.json"),
            "scope_receipt_sha256": sha256_file(scope_root / "receipt.json"),
        },
    }
    _write_json_new(output / "report.json", report)
    _write_json_new(
        output / "receipt.json",
        {
            **report,
            "outputs": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in sorted(output.iterdir())
                if path.is_file()
            },
        },
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-table-dir", required=True)
    parser.add_argument("--documents", required=True)
    parser.add_argument("--line-key", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--line-probability", required=True)
    parser.add_argument("--baseline-prediction", required=True)
    parser.add_argument("--decoder-report", required=True)
    parser.add_argument("--scope-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
