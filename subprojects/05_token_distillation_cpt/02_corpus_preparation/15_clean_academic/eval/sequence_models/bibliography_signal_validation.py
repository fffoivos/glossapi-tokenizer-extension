#!/usr/bin/env python3
"""Evaluate frozen signal-TCN candidates on retrospective validation.

All thresholds and the anchored candidate are selected from train OOF evidence
recorded before this command.  Validation is evaluated for diagnosis; this
command never fits a model or searches a validation threshold.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

from .bibliography_auxiliary_scope_veto import materialize_auxiliary_headings
from .bibliography_deterministic_roles import (
    ROLE_NAMES,
    _analyze_document,
)
from .bibliography_entry_blocks import BlockConfig, evaluate_prediction
from .bibliography_entry_dataset import LABEL_TO_ID
from .bibliography_entry_models import load_table
from .bibliography_signal_block_decode import (
    SCHEMA_VERSION as BLOCK_SCHEMA,
    decode_signal_blocks,
)
from .bibliography_signal_tcn import (
    FEATURE_NAMES,
    SCHEMA_VERSION as SIGNAL_SCHEMA,
    SignalTCN,
    _collate,
    build_signal_features,
    decode_signal_probability,
    make_signal_chunks,
    require_torch,
)


SCHEMA_VERSION = "bibliography-signal-validation-v1"
POLICY_SCHEMA = "bibliography-signal-validation-policy-v1"


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


def _validation_quality_exclusions(path: Path) -> tuple[set[str], dict[str, Any]]:
    packet = json.loads(path.read_text(encoding="utf-8"))
    if packet.get("schema_version") != "bibliography-validation-quality-decisions-v1":
        raise ValueError("unsupported validation-quality decisions")
    rows = packet.get("documents")
    if not isinstance(rows, list):
        raise ValueError("validation-quality decisions are incomplete")
    ids = [str(row["document_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("validation-quality decisions contain duplicate documents")
    return {
        str(row["document_id"])
        for row in rows
        if row.get("decision") == "exclude"
    }, packet


def _materialize_roles(
    table: Any, input_path: Path, *, workers: int
) -> tuple[np.ndarray, dict[str, int]]:
    expected = {
        str(document["document_id"]): (
            int(document["line_start"]),
            int(document["line_end"]),
        )
        for document in table.documents
    }
    tasks: list[tuple[str, list[Any]]] = []
    seen: set[str] = set()
    with input_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            # Fail closed: train and test text are never deserialized here.
            if '"split": "validation"' not in raw:
                continue
            row = json.loads(raw)
            document_id = str(row.get("document_id"))
            if document_id not in expected:
                continue
            lines = row.get("lines")
            if not isinstance(lines, list):
                raise ValueError(f"{document_id}: missing line inventory")
            start, end = expected[document_id]
            if len(lines) != end - start:
                raise ValueError(f"{document_id}: source/table line alignment failure")
            if document_id in seen:
                raise ValueError("validation source contains a duplicate document")
            tasks.append((document_id, lines))
            seen.add(document_id)
    if seen != set(expected):
        raise ValueError("validation source/table document inventory mismatch")
    roles = np.zeros((len(table.targets), len(ROLE_NAMES)), dtype=np.uint8)
    counts = {name: 0 for name in ROLE_NAMES}
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(workers)) as executor:
        for document_id, document_roles, document_counts in executor.map(
            _analyze_document, tasks, chunksize=1
        ):
            start, end = expected[document_id]
            roles[start:end] = document_roles
            for name, count in document_counts.items():
                counts[name] += int(count)
    return roles, counts


def _load_signal_model(checkpoint_path: Path, architecture: dict[str, Any]) -> Any:
    require_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("schema_version") != SIGNAL_SCHEMA
        or tuple(checkpoint.get("feature_names", ())) != FEATURE_NAMES
        or checkpoint.get("architecture") != architecture
    ):
        raise ValueError("signal-TCN checkpoint contract changed")
    model = SignalTCN(
        len(FEATURE_NAMES),
        hidden_dim=int(architecture["hidden_dim"]),
        dilations=tuple(int(value) for value in architecture["dilations"]),
        dropout=float(architecture["dropout"]),
    ).cpu()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval()


def _ensemble_probability(
    table: Any,
    features: np.ndarray,
    model_paths: Sequence[Path],
    architecture: dict[str, Any],
    *,
    central_width: int,
    context: int,
    batch_size: int,
) -> np.ndarray:
    require_torch()
    chunks = make_signal_chunks(
        table,
        list(range(len(table.documents))),
        central_width=int(central_width),
        context=int(context),
    )
    total = np.zeros(len(table.targets), dtype=np.float64)
    count = np.zeros(len(table.targets), dtype=np.uint8)
    for model_path in model_paths:
        model = _load_signal_model(model_path, architecture)
        with torch.inference_mode():
            for offset in range(0, len(chunks), int(batch_size)):
                batch = chunks[offset : offset + int(batch_size)]
                x, line_mask, _loss_mask, _target = _collate(
                    batch, features, table.original_labels
                )
                values = torch.sigmoid(model(x, line_mask)).cpu().numpy()
                for batch_index, chunk in enumerate(batch):
                    local_start = chunk.target_start - chunk.input_start
                    local_end = chunk.target_end - chunk.input_start
                    indices = slice(chunk.target_start, chunk.target_end)
                    total[indices] += values[
                        batch_index, local_start:local_end
                    ]
                    count[indices] += 1
    known = table.original_labels != LABEL_TO_ID["UNKNOWN"]
    if not np.all(count[known] == len(model_paths)):
        raise ValueError("validation ensemble coverage is incomplete")
    probability = np.zeros(len(table.targets), dtype=np.float32)
    probability[known] = (total[known] / count[known]).astype(np.float32)
    return probability


def select_train_recall_candidate(report: dict[str, Any]) -> dict[str, Any]:
    if (
        report.get("schema_version") != BLOCK_SCHEMA
        or report.get("validation_opened") is not False
    ):
        raise ValueError("recall block report is not train-only")
    eligible = [
        row
        for row in report["candidates"]
        if float(row["metrics"]["line_precision"]) >= 0.90
    ]
    if not eligible:
        raise ValueError("recall block grid has no candidate above 0.90 precision")
    return max(
        eligible,
        key=lambda row: (
            float(row["metrics"]["line_recall"]),
            float(row["metrics"]["token_recall"]),
            float(row["metrics"]["line_precision"]),
        ),
    )


def _breakdowns(
    table: Any, prediction: np.ndarray, qualified: set[int]
) -> dict[str, Any]:
    by_source: dict[str, Any] = {}
    for source in sorted({str(document.get("source")) for document in table.documents}):
        subset = {
            index
            for index in qualified
            if str(table.documents[index].get("source")) == source
        }
        by_source[source] = evaluate_prediction(
            table, prediction, document_subset=subset
        )
    return {
        "full": evaluate_prediction(table, prediction),
        "extraction_qualified": evaluate_prediction(
            table, prediction, document_subset=qualified
        ),
        "qualified_by_source": by_source,
    }


def _worst_misses(
    table: Any, prediction: np.ndarray, qualified: set[int], *, limit: int = 50
) -> list[dict[str, Any]]:
    gold = table.original_labels == LABEL_TO_ID["BIB"]
    rows = []
    for document_index in sorted(qualified):
        document = table.documents[document_index]
        start, end = int(document["line_start"]), int(document["line_end"])
        local_gold = gold[start:end]
        local_prediction = prediction[start:end]
        tokens = table.token_counts[start:end].astype(np.int64)
        gold_lines = int(np.count_nonzero(local_gold))
        missed_lines = int(np.count_nonzero(local_gold & ~local_prediction))
        gold_tokens = int(tokens[local_gold].sum())
        missed_tokens = int(tokens[local_gold & ~local_prediction].sum())
        if not gold_lines:
            continue
        rows.append(
            {
                "document_id": str(document["document_id"]),
                "source": str(document.get("source")),
                "coverage": document.get("coverage"),
                "gold_line_count": gold_lines,
                "missed_line_count": missed_lines,
                "line_recall": (gold_lines - missed_lines) / gold_lines,
                "gold_token_count": gold_tokens,
                "missed_token_count": missed_tokens,
                "token_recall": (
                    (gold_tokens - missed_tokens) / gold_tokens if gold_tokens else 0.0
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            -int(row["missed_token_count"]),
            -int(row["missed_line_count"]),
            row["document_id"],
        )
    )
    return rows[:limit]


def run(args: argparse.Namespace) -> dict[str, Any]:
    require_torch()
    torch.set_num_threads(int(args.torch_threads))
    table = load_table(args.validation_table_dir, expected_split="validation")
    input_path = Path(args.input).resolve()
    line_probability_path = Path(args.validation_line_probability).resolve()
    line_probability = np.load(
        line_probability_path, mmap_mode="r", allow_pickle=False
    )
    if len(line_probability) != len(table.targets):
        raise ValueError("validation line probability is not aligned")
    policy_path = Path(args.policy).resolve()
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if (
        policy.get("schema_version") != POLICY_SCHEMA
        or policy.get("status") != "frozen_before_signal_tcn_validation"
    ):
        raise ValueError("signal validation policy is not frozen")
    signal_root = Path(args.signal_tcn_dir).resolve()
    signal_report_path = signal_root / "signal_tcn_oof_report.json"
    signal_report = json.loads(signal_report_path.read_text(encoding="utf-8"))
    if (
        signal_report.get("schema_version") != SIGNAL_SCHEMA
        or signal_report.get("validation_opened") is not False
    ):
        raise ValueError("signal model report is not train-only")
    architecture = dict(signal_report["architecture"])
    checkpoint_architecture = {
        "hidden_dim": int(architecture["hidden_dim"]),
        "dilations": [int(value) for value in architecture["dilations"]],
        "dropout": float(architecture["dropout"]),
    }
    model_paths = sorted((signal_root / "models").glob("fold*.pt"))
    if len(model_paths) != 5:
        raise ValueError("expected exactly five signal-TCN fold models")
    recall_report_path = (
        Path(args.train_recall_block_dir).resolve()
        / "signal_block_decode_oof_report.json"
    )
    recall_report = json.loads(recall_report_path.read_text(encoding="utf-8"))
    anchored_row = select_train_recall_candidate(recall_report)
    quality_excluded, quality_packet = _validation_quality_exclusions(
        Path(args.quality_decisions).resolve()
    )
    known_ids = {str(document["document_id"]) for document in table.documents}
    if not quality_excluded <= known_ids:
        raise ValueError("validation-quality decision references unknown document")
    qualified = {
        index
        for index, document in enumerate(table.documents)
        if str(document["document_id"]) not in quality_excluded
    }
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    roles, role_counts = _materialize_roles(
        table, input_path, workers=int(args.workers)
    )
    features = build_signal_features(line_probability, roles, table.header_kinds)
    ensemble = _ensemble_probability(
        table,
        features,
        model_paths,
        checkpoint_architecture,
        central_width=int(architecture["central_width"]),
        context=int(architecture["context_lines"]),
        batch_size=int(args.batch_size),
    )
    headings, scope = materialize_auxiliary_headings(
        table, input_path, expected_split="validation"
    )
    if np.any(scope & (table.original_labels == LABEL_TO_ID["BIB"])):
        raise ValueError("validation auxiliary scope overlaps silver bibliography")
    _save_array(output_dir / "validation_roles.npy", roles)
    _save_array(output_dir / "validation_signal_ensemble_probability.npy", ensemble)
    _save_array(output_dir / "validation_auxiliary_scope.npy", scope)

    base_config = BlockConfig(**signal_report["block_config"])
    candidate_predictions: dict[str, np.ndarray] = {}
    train_rows: dict[str, Any] = {}
    for name in ("precision_first_plain", "recall_first_plain"):
        threshold = float(policy["validation_candidates"][name]["threshold"])
        matching = [
            row
            for row in signal_report["candidates"]
            if float(row["threshold"]) == threshold
        ]
        if len(matching) != 1:
            raise ValueError(f"{name} threshold is absent from train report")
        prediction, _vetoed = decode_signal_probability(
            table,
            ensemble,
            line_probability,
            scope,
            base_config,
            threshold=threshold,
            qualified_documents=set(range(len(table.documents))),
            apply_veto=True,
        )
        candidate_predictions[name] = prediction
        train_rows[name] = matching[0]
    anchored_prediction, _vetoed = decode_signal_blocks(
        table,
        ensemble,
        line_probability,
        scope,
        BlockConfig(**anchored_row["config"]),
        qualified_documents=set(range(len(table.documents))),
        apply_veto=True,
    )
    candidate_predictions["recall_first_anchored"] = anchored_prediction
    train_rows["recall_first_anchored"] = anchored_row

    candidates: dict[str, Any] = {}
    for name, prediction in candidate_predictions.items():
        _save_array(output_dir / f"{name}.prediction.npy", prediction)
        candidates[name] = {
            "train_oof_frozen_row": train_rows[name],
            "validation": _breakdowns(table, prediction, qualified),
            "worst_qualified_misses": _worst_misses(
                table, prediction, qualified
            ),
        }
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_frozen_signal_candidates_on_retrospective_validation",
        "evidence_scope": "LLM_silver_validation_not_human_gold",
        "candidate_selection": "all thresholds and decoder settings frozen from train OOF before this run; no validation threshold search",
        "candidates": candidates,
        "role_counts": role_counts,
        "auxiliary_heading_line_count": int(np.count_nonzero(headings)),
        "auxiliary_scope_line_count": int(np.count_nonzero(scope)),
        "quality_filter": {
            "decision_schema": quality_packet["schema_version"],
            "excluded_document_count": len(quality_excluded),
            "qualified_document_count": len(qualified),
            "interpretation": quality_packet.get("metric_interpretation"),
        },
        "input_hashes": {
            "source": _sha256(input_path),
            "validation_table_receipt": _sha256(
                Path(args.validation_table_dir).resolve() / "receipt.json"
            ),
            "validation_line_probability": _sha256(line_probability_path),
            "signal_train_report": _sha256(signal_report_path),
            "train_recall_report": _sha256(recall_report_path),
            "policy": _sha256(policy_path),
            "quality_decisions": _sha256(Path(args.quality_decisions).resolve()),
        },
        "policy": policy,
        "code_commit": str(args.code_commit),
        "slurm_job_id": str(args.slurm_job_id),
        "validation_opened": True,
        "production_eligible": False,
    }
    _write_json(output_dir / "signal_validation_report.json", result)
    outputs = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    _write_json(output_dir / "receipt.json", {**result, "outputs": outputs})
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--validation-table-dir", required=True)
    parser.add_argument("--validation-line-probability", required=True)
    parser.add_argument("--signal-tcn-dir", required=True)
    parser.add_argument("--train-recall-block-dir", required=True)
    parser.add_argument("--quality-decisions", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--torch-threads", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
