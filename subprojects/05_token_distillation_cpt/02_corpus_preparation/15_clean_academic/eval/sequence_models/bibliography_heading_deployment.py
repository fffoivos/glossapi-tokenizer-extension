#!/usr/bin/env python3
"""Receipt-owned deployment of the grouped-OOF three-role heading expert.

The historical heading fit is already a deployable fold ensemble: every fold
pickle contains its fitted character/word vectorizers, numeric scaler, binary
heading model, and conditional three-way type model.  This module turns those
training artifacts into a candidate-owned package only after reproducing the
stored grouped-OOF probabilities byte for byte.

No labels are accepted by the prediction API.  New documents are scored from
their text, physical line coordinates, and the frozen D1 entry probability.
The high-recall candidate predicate and numeric feature contract are exactly
the ones used to materialize the original heading table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bibliography_entry_models import PINNED_SKLEARN_VERSION, Table, load_table
from .bibliography_evolution_contract import (
    ContractError,
    canonical_json_bytes,
    load_json,
    sha256_file,
    write_json_exclusive,
)
from .bibliography_evolution_headers import ROLE_TO_ID as CONTROLLER_ROLE_TO_ID
from .bibliography_role_experts import (
    HEADING_PROBABILITY_COLUMNS,
    HEADING_SCHEMA,
    HeadingBundle,
    HeadingTransform,
)
from .bibliography_role_features import (
    HEADING_NUMERIC_NAMES,
    broad_heading_candidate,
    candidate_window_mask,
    heading_numeric_features,
)


DEPLOYMENT_SCHEMA = "bibliography-heading-fold-ensemble-deployment-v1"
FEATURE_SCHEMA = "bibliography-heading-fold-ensemble-feature-contract-v1"
ASSIGNMENT_THRESHOLD = 0.5
ENTRY_NEIGHBOURHOOD_THRESHOLD = 0.25
ENTRY_NEIGHBOURHOOD_RADIUS = 30
EXPECTED_FOLDS = 5
_LEGACY_GLOBALS = {
    ("__main__", "HeadingBundle"): HeadingBundle,
    ("__main__", "HeadingTransform"): HeadingTransform,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _regular_file(path: Path, label: str) -> Path:
    _require(path.is_file() and not path.is_symlink(), f"{label} is missing or a symlink")
    return path.resolve()


def _real_directory(path: Path, label: str) -> Path:
    _require(path.is_dir() and not path.is_symlink(), f"{label} is missing or a symlink")
    return path.resolve()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ContractError(f"{path}:{number}: expected an object")
            rows.append(value)
    return rows


def load_document_rows(path: Path) -> list[dict[str, Any]]:
    """Load a receipt-bound document JSONL for prediction-only inference."""

    return _jsonl(_regular_file(path, "heading inference documents"))


def materialize_prediction_documents(
    source_path: Path,
    table_root: Path,
    output: Path,
    *,
    split: str = "validation",
) -> Path:
    """Strip labels and non-selected splits from a canonical document source."""

    source_path = _regular_file(source_path, "canonical heading document source")
    table = load_table(table_root, expected_split=split)
    source_rows = _jsonl(source_path)
    selected = [row for row in source_rows if row.get("split") == split]
    aligned = _aligned_documents(selected, table)
    _require(not output.exists() and not output.is_symlink(), "document output must be new")
    output.mkdir(parents=True)
    prediction_rows = []
    for source, lines, metadata in zip(selected, aligned, table.documents, strict=True):
        prediction_rows.append(
            {
                "document_id": str(metadata["document_id"]),
                "work_id": str(metadata["work_id"]),
                "source": str(metadata["source"]),
                "split": split,
                "lines": [
                    {
                        "line_id": str(
                            line.get("line_id")
                            or f"{metadata['document_id']}:{int(line['abs_idx'])}"
                        ),
                        "abs_idx": int(line["abs_idx"]),
                        "text": str(line["text"]),
                    }
                    for line in lines
                ],
            }
        )
    documents_path = output / "documents.jsonl"
    with documents_path.open("xb") as handle:
        for row in prediction_rows:
            handle.write(canonical_json_bytes(row))
    receipt = {
        "schema_version": "bibliography-heading-prediction-documents-v1",
        "status": "passed_label_stripped_single_split_materialization",
        "split": split,
        "labels_present": False,
        "source": {"path": str(source_path), "sha256": sha256_file(source_path)},
        "table_manifest_sha256": sha256_file(Path(table_root) / "manifest.json"),
        "document_count": len(prediction_rows),
        "work_count": len({row["work_id"] for row in prediction_rows}),
        "line_count": sum(len(row["lines"]) for row in prediction_rows),
        "output": {
            "path": "documents.jsonl",
            "bytes": documents_path.stat().st_size,
            "sha256": sha256_file(documents_path),
        },
    }
    receipt_path = output / "receipt.json"
    write_json_exclusive(receipt_path, receipt)
    return receipt_path


class _LegacyHeadingUnpickler(pickle.Unpickler):
    """Map only the two classes historically serialized as ``__main__``."""

    def find_class(self, module: str, name: str) -> Any:
        replacement = _LEGACY_GLOBALS.get((module, name))
        if replacement is not None:
            return replacement
        return super().find_class(module, name)


def _load_bundle(path: Path) -> HeadingBundle:
    import sklearn

    _require(
        sklearn.__version__ == PINNED_SKLEARN_VERSION,
        "heading deployment sklearn version differs from training",
    )
    with _regular_file(path, "heading fold model").open("rb") as handle:
        value = _LegacyHeadingUnpickler(handle).load()
    _require(isinstance(value, HeadingBundle), "heading fold pickle has an unexpected type")
    return value


def _verify_output_inventory(root: Path, receipt: Mapping[str, Any]) -> None:
    outputs = receipt.get("outputs")
    _require(isinstance(outputs, Mapping) and bool(outputs), "artifact receipt has no outputs")
    actual = {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "receipt.json"
    }
    _require(set(actual) == set(outputs), "artifact receipt output inventory differs")
    for name, path in actual.items():
        row = outputs[name]
        _require(
            isinstance(row, Mapping)
            and int(row.get("bytes", -1)) == path.stat().st_size
            and row.get("sha256") == sha256_file(path),
            f"artifact receipt output changed: {name}",
        )


def _source_arrays(
    model_root: Path, table_root: Path
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model_root = _real_directory(model_root, "heading OOF root")
    table_root = _real_directory(table_root, "heading table root")
    model_receipt = load_json(_regular_file(model_root / "receipt.json", "heading OOF receipt"))
    model_report = load_json(_regular_file(model_root / "report.json", "heading OOF report"))
    table_receipt = load_json(_regular_file(table_root / "receipt.json", "heading table receipt"))
    table_manifest = load_json(_regular_file(table_root / "manifest.json", "heading table manifest"))
    _verify_output_inventory(model_root, model_receipt)
    _verify_output_inventory(table_root, table_receipt)
    _require(
        model_report.get("schema_version") == HEADING_SCHEMA
        and model_report.get("status") == "passed_grouped_oof_expert_training"
        and model_report.get("kind") == "heading"
        and model_report.get("probability_columns") == list(HEADING_PROBABILITY_COLUMNS),
        "heading OOF report contract differs",
    )
    _require(
        table_manifest.get("schema_version") == "bibliography-heading-expert-table-v1"
        and table_manifest.get("status") == "passed_heading_table_materialization"
        and table_manifest.get("split") == "train"
        and model_report.get("table_manifest_sha256") == sha256_file(table_root / "manifest.json"),
        "heading table is not the table bound by the OOF fit",
    )
    model_rows = np.load(model_root / "row_indices.npy", allow_pickle=False)
    table_rows = np.load(table_root / "row_indices.npy", allow_pickle=False)
    _require(np.array_equal(model_rows, table_rows), "heading OOF/table row inventories differ")
    texts = [str(row["text"]) for row in _jsonl(table_root / "texts.jsonl")]
    numeric = np.load(table_root / "features.npy", allow_pickle=False)
    folds = np.load(table_root / "folds.npy", allow_pickle=False)
    expected = np.load(model_root / "oof_probability.npy", allow_pickle=False)
    _require(
        len(texts) == len(table_rows)
        and numeric.shape == (len(table_rows), len(HEADING_NUMERIC_NAMES))
        and folds.shape == table_rows.shape
        and expected.shape == (len(table_rows), len(HEADING_PROBABILITY_COLUMNS))
        and np.isfinite(numeric).all()
        and np.isfinite(expected).all()
        and set(np.unique(folds).tolist()) == set(range(EXPECTED_FOLDS)),
        "heading source arrays have an invalid shape, fold inventory, or value",
    )
    return texts, numeric, folds, expected, table_rows


def replay_grouped_oof(model_root: Path, table_root: Path) -> dict[str, Any]:
    """Reproduce every stored held-out probability with its fold model."""

    texts, numeric, folds, expected, _rows = _source_arrays(model_root, table_root)
    observed = np.full_like(expected, np.nan)
    for fold in range(EXPECTED_FOLDS):
        indices = np.flatnonzero(folds == fold)
        bundle = _load_bundle(model_root / "models" / f"fold{fold}.pkl")
        observed[indices] = bundle.predict(
            [texts[int(index)] for index in indices], numeric[indices]
        )
    _require(np.array_equal(observed, expected), "heading fold models do not replay OOF bytes")
    array_sha256 = hashlib.sha256(np.ascontiguousarray(observed).tobytes()).hexdigest()
    return {
        "status": "passed_byte_identical_grouped_oof_replay",
        "row_count": len(observed),
        "cell_count": int(observed.size),
        "maximum_absolute_difference": 0.0,
        "probability_array_sha256": array_sha256,
    }


def _feature_contract() -> dict[str, Any]:
    source_root = Path(__file__).resolve().parent
    code_files = (
        "bibliography_heading_deployment.py",
        "bibliography_role_features.py",
        "bibliography_role_experts.py",
    )
    return {
        "schema_version": FEATURE_SCHEMA,
        "numeric_feature_names": list(HEADING_NUMERIC_NAMES),
        "probability_columns": list(HEADING_PROBABILITY_COLUMNS),
        "candidate_policy": {
            "predicate": "broad_heading_candidate",
            "entry_probability_threshold": ENTRY_NEIGHBOURHOOD_THRESHOLD,
            "maximum_physical_line_radius": ENTRY_NEIGHBOURHOOD_RADIUS,
            "trusted_training_only_forcing_disabled_at_inference": True,
        },
        "ensemble_policy": "arithmetic_mean_of_five_fold_probability_vectors",
        "assignment_policy": {
            "any_header_threshold": "candidate_spec_bound",
            "control_threshold": ASSIGNMENT_THRESHOLD,
            "predeclared_evolution_grid": [0.3, 0.4, 0.5, 0.6, 0.7],
            "type": "argmax_of_marginal_bib_header_bib_subheader_non_bib_header",
            "below_threshold": "NONE",
        },
        "code": [
            {"path": name, "sha256": sha256_file(source_root / name)} for name in code_files
        ],
    }


def materialize_deployment(
    model_root: Path, table_root: Path, training_base_table_root: Path, output: Path
) -> Path:
    """Copy a verified fold ensemble into a new candidate-owned directory."""

    model_root = _real_directory(model_root, "heading OOF root")
    table_root = _real_directory(table_root, "heading table root")
    training_base_table_root = _real_directory(
        training_base_table_root, "heading training base table root"
    )
    _require(not output.exists() and not output.is_symlink(), "deployment output must be new")
    replay = replay_grouped_oof(model_root, table_root)
    heading_manifest = load_json(table_root / "manifest.json")
    training_manifest = load_json(
        _regular_file(training_base_table_root / "manifest.json", "heading training manifest")
    )
    _require(
        heading_manifest.get("inputs", {}).get("base_manifest_sha256")
        == sha256_file(training_base_table_root / "manifest.json")
        and training_manifest.get("split") == "train",
        "heading training base table is not the one bound by the heading table",
    )
    training_documents = _jsonl(
        _regular_file(
            training_base_table_root / "documents.jsonl", "heading training documents"
        )
    )
    training_work_ids = sorted({str(row.get("work_id", "")) for row in training_documents})
    _require(
        bool(training_work_ids)
        and "" not in training_work_ids
        and len(training_documents) == int(training_manifest.get("document_count", -1))
        and len(training_work_ids) == int(training_manifest.get("work_count", -1)),
        "heading training work inventory is malformed",
    )
    output.mkdir(parents=True)
    model_output = output / "models"
    model_output.mkdir()
    for fold in range(EXPECTED_FOLDS):
        source = _regular_file(model_root / "models" / f"fold{fold}.pkl", "heading fold model")
        destination = model_output / source.name
        with source.open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
    write_json_exclusive(output / "feature_contract.json", _feature_contract())
    write_json_exclusive(
        output / "training_work_ids.json",
        {
            "schema_version": "bibliography-heading-training-work-inventory-v1",
            "split": "train",
            "document_count": len(training_documents),
            "work_count": len(training_work_ids),
            "work_ids": training_work_ids,
        },
    )
    outputs = {
        path.relative_to(output).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    receipt = {
        "schema_version": DEPLOYMENT_SCHEMA,
        "status": "passed_candidate_owned_fold_ensemble_materialization",
        "sklearn_version": PINNED_SKLEARN_VERSION,
        "fold_count": EXPECTED_FOLDS,
        "source": {
            "heading_oof_receipt_sha256": sha256_file(model_root / "receipt.json"),
            "heading_table_receipt_sha256": sha256_file(table_root / "receipt.json"),
            "training_base_manifest_sha256": sha256_file(
                training_base_table_root / "manifest.json"
            ),
        },
        "grouped_oof_replay": replay,
        "prediction_modes": {
            "receipt_bound_grouped_oof_replay": {
                "allowed_rows": replay["row_count"],
                "fold_selection": "stored row fold only",
                "arbitrary_rows_allowed": False,
            },
            "unseen_ensemble_mean": {
                "required_work_overlap_with_training": 0,
                "fold_selection": "arithmetic mean of all five fold probability vectors",
            },
        },
        "outputs": outputs,
    }
    receipt_path = output / "receipt.json"
    write_json_exclusive(receipt_path, receipt)
    return receipt_path


def verify_deployment(root: Path) -> dict[str, Any]:
    root = _real_directory(root, "heading deployment root")
    receipt = load_json(_regular_file(root / "receipt.json", "heading deployment receipt"))
    _require(
        receipt.get("schema_version") == DEPLOYMENT_SCHEMA
        and receipt.get("status") == "passed_candidate_owned_fold_ensemble_materialization"
        and receipt.get("sklearn_version") == PINNED_SKLEARN_VERSION
        and int(receipt.get("fold_count", -1)) == EXPECTED_FOLDS
        and receipt.get("grouped_oof_replay", {}).get("status")
        == "passed_byte_identical_grouped_oof_replay",
        "heading deployment receipt contract differs",
    )
    _verify_output_inventory(root, receipt)
    feature = load_json(root / "feature_contract.json")
    _require(feature == _feature_contract(), "heading deployment feature contract changed")
    work_inventory = load_json(root / "training_work_ids.json")
    work_ids = work_inventory.get("work_ids")
    _require(
        work_inventory.get("schema_version")
        == "bibliography-heading-training-work-inventory-v1"
        and work_inventory.get("split") == "train"
        and isinstance(work_ids, list)
        and work_ids == sorted(set(work_ids))
        and int(work_inventory.get("work_count", -1)) == len(work_ids),
        "heading deployment training-work inventory changed",
    )
    for fold in range(EXPECTED_FOLDS):
        _load_bundle(root / "models" / f"fold{fold}.pkl")
    return receipt


def probabilities_to_role_ids(
    probability: np.ndarray, candidate_mask: np.ndarray, *, threshold: float = ASSIGNMENT_THRESHOLD
) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float32)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    _require(
        probability.ndim == 2
        and probability.shape[1] == len(HEADING_PROBABILITY_COLUMNS)
        and candidate_mask.shape == (len(probability),)
        and np.isfinite(probability).all()
        and 0.0 <= threshold <= 1.0,
        "heading assignment inputs are malformed",
    )
    role_ids = np.zeros(len(probability), dtype=np.uint8)
    assigned = candidate_mask & (probability[:, 0] >= float(threshold))
    winners = np.argmax(probability[:, 1:], axis=1)
    controller_ids = np.asarray(
        [
            CONTROLLER_ROLE_TO_ID["BIB_HEADER"],
            CONTROLLER_ROLE_TO_ID["BIB_SUBHEADER"],
            CONTROLLER_ROLE_TO_ID["NON_BIB_HEADER"],
        ],
        dtype=np.uint8,
    )
    role_ids[assigned] = controller_ids[winners[assigned]]
    return role_ids


def _aligned_documents(
    documents: Sequence[Mapping[str, Any]], table: Table
) -> list[list[Mapping[str, Any]]]:
    rows = list(documents)
    if rows and any("split" in row for row in rows):
        rows = [row for row in rows if row.get("split") == table.manifest.get("split")]
    _require(len(rows) == len(table.documents), "heading documents/table count differs")
    aligned: list[list[Mapping[str, Any]]] = []
    for source, metadata in zip(rows, table.documents, strict=True):
        lines = source.get("lines")
        _require(
            source.get("document_id") == metadata["document_id"]
            and isinstance(lines, list)
            and len(lines) == int(metadata["line_count"]),
            "heading documents/table identity or line count differs",
        )
        start = int(metadata["line_start"])
        for offset, line in enumerate(lines):
            _require(
                isinstance(line, Mapping)
                and isinstance(line.get("text"), str)
                and int(line.get("abs_idx", -1)) == int(table.abs_indices[start + offset]),
                "heading document line text/coordinate differs from table",
            )
        aligned.append(lines)
    return aligned


def predict_documents(
    deployment_root: Path,
    table: Table,
    documents: Sequence[Mapping[str, Any]],
    entry_probability: np.ndarray,
    *,
    assignment_threshold: float = ASSIGNMENT_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return full-line role IDs, probabilities, and candidate mask."""

    verify_deployment(deployment_root)
    entry = np.asarray(entry_probability, dtype=np.float32)
    _require(
        entry.shape == (len(table.targets),)
        and np.isfinite(entry).all()
        and np.all((entry >= 0.0) & (entry <= 1.0)),
        "heading D1 probability is malformed",
    )
    lines_by_doc = _aligned_documents(documents, table)
    training_work_ids = set(
        load_json(deployment_root / "training_work_ids.json")["work_ids"]
    )
    inference_work_ids = {str(row.get("work_id", "")) for row in table.documents}
    _require(
        "" not in inference_work_ids
        and not (training_work_ids & inference_work_ids),
        "unseen heading ensemble input overlaps a training work group",
    )
    candidate_mask = np.zeros(len(entry), dtype=bool)
    feature_rows: list[np.ndarray] = []
    texts: list[str] = []
    row_indices: list[int] = []
    for lines, metadata in zip(lines_by_doc, table.documents, strict=True):
        start, end = int(metadata["line_start"]), int(metadata["line_end"])
        local_entry = entry[start:end]
        local_abs = table.abs_indices[start:end]
        broad = np.asarray(
            [
                broad_heading_candidate(
                    str(line["text"]),
                    previous_blank=offset > 0 and not str(lines[offset - 1]["text"]).strip(),
                    next_blank=offset + 1 < len(lines)
                    and not str(lines[offset + 1]["text"]).strip(),
                )
                for offset, line in enumerate(lines)
            ],
            dtype=bool,
        )
        local_candidates = broad & candidate_window_mask(
            local_entry,
            np.zeros(len(local_entry), dtype=bool),
            local_abs,
            entry_threshold=ENTRY_NEIGHBOURHOOD_THRESHOLD,
            radius=ENTRY_NEIGHBOURHOOD_RADIUS,
        )
        for offset in np.flatnonzero(local_candidates):
            absolute = start + int(offset)
            text = str(lines[int(offset)]["text"])
            above = entry[max(start, absolute - 30) : absolute]
            below = entry[absolute + 1 : min(end, absolute + 31)]
            feature_rows.append(
                heading_numeric_features(
                    text,
                    previous_blank=offset > 0 and not str(lines[int(offset) - 1]["text"]).strip(),
                    next_blank=offset + 1 < len(lines)
                    and not str(lines[int(offset) + 1]["text"]).strip(),
                    position_fraction=float(offset) / max(1, len(lines) - 1),
                    entry_probabilities_above=above,
                    entry_probabilities_below=below,
                )
            )
            texts.append(text)
            row_indices.append(absolute)
            candidate_mask[absolute] = True
    probability = np.zeros(
        (len(entry), len(HEADING_PROBABILITY_COLUMNS)), dtype=np.float32
    )
    if row_indices:
        numeric = np.stack(feature_rows).astype(np.float32)
        ensemble = np.zeros((len(row_indices), len(HEADING_PROBABILITY_COLUMNS)), dtype=np.float64)
        for fold in range(EXPECTED_FOLDS):
            bundle = _load_bundle(deployment_root / "models" / f"fold{fold}.pkl")
            ensemble += bundle.predict(texts, numeric).astype(np.float64)
        probability[np.asarray(row_indices)] = (ensemble / EXPECTED_FOLDS).astype(np.float32)
    roles = probabilities_to_role_ids(
        probability, candidate_mask, threshold=assignment_threshold
    )
    return roles, probability, candidate_mask


def deployment_proofs(root: Path) -> list[dict[str, Any]]:
    """Return every package file proof for candidate-ownership checks."""

    verify_deployment(root)
    return [
        {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    replay = sub.add_parser("replay-oof")
    replay.add_argument("--heading-model-dir", required=True)
    replay.add_argument("--heading-table-dir", required=True)
    documents = sub.add_parser("materialize-prediction-documents")
    documents.add_argument("--source", required=True)
    documents.add_argument("--table-dir", required=True)
    documents.add_argument("--split", default="validation")
    documents.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "replay-oof":
        print(
            json.dumps(
                replay_grouped_oof(
                    Path(args.heading_model_dir), Path(args.heading_table_dir)
                ),
                sort_keys=True,
            )
        )
        return
    receipt = materialize_prediction_documents(
        Path(args.source),
        Path(args.table_dir),
        Path(args.output_dir),
        split=str(args.split),
    )
    print(receipt)


if __name__ == "__main__":
    main()
