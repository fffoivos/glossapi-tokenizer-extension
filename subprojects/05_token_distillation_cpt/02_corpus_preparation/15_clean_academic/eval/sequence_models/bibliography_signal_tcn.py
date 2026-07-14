#!/usr/bin/env python3
"""Train-only document-grouped TCN over frozen bibliography line signals.

This is deliberately a small block model, not another text classifier.  It sees
only the frozen out-of-fold bibliography-entry probability, mutually exclusive
deterministic competing roles, and the exact-header flag.  It does not see raw
text, character length, document position, source name, or validation rows.

The symmetric dilated convolutions learn whether a line sits inside a coherent
bibliography neighbourhood.  Every reported training prediction is made by a
model that was fitted without that document.  The already-audited exact
auxiliary-scope veto is applied after decoding and can only remove components.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:  # Available in the pinned Clariden PyTorch uenv.
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - dependency-free unit tests
    torch = None
    nn = None

from .bibliography_auxiliary_scope_veto import (
    has_auxiliary_scope,
    materialize_auxiliary_headings,
)
from .bibliography_deterministic_roles import ROLE_NAMES, SCHEMA_VERSION as ROLE_SCHEMA
from .bibliography_entry_blocks import (
    BlockConfig,
    attach_h0_document,
    blocks_from_mask,
    evaluate_prediction,
)
from .bibliography_entry_coherence import is_safe_candidate
from .bibliography_entry_component_gate import _load_quality_exclusions
from .bibliography_entry_dataset import LABEL_TO_ID, MAX_PHYSICAL_GAP
from .bibliography_entry_models import load_table
from .bibliography_entry_role_sequence import _validate_role_matrix


SCHEMA_VERSION = "bibliography-signal-tcn-oof-v1"
FEATURE_NAMES = (
    "frozen_entry_probability",
    *(f"explicit_role_{name}" for name in ROLE_NAMES),
    "exact_bibliography_header",
)


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("bibliography_signal_tcn requires PyTorch on a CPU worker")


if torch is not None:

    class ResidualSignalBlock(nn.Module):
        def __init__(self, hidden_dim: int, dilation: int, dropout: float):
            super().__init__()
            self.norm = nn.LayerNorm(hidden_dim)
            self.convolution = nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, values: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
            residual = values
            values = self.norm(values).transpose(1, 2)
            values = torch.nn.functional.gelu(self.convolution(values)).transpose(1, 2)
            return (residual + self.dropout(values)) * mask.unsqueeze(-1)


    class SignalTCN(nn.Module):
        def __init__(
            self,
            input_dim: int,
            *,
            hidden_dim: int = 32,
            dilations: Sequence[int] = (1, 2, 4, 8),
            dropout: float = 0.10,
        ):
            super().__init__()
            self.input_projection = nn.Linear(input_dim, hidden_dim)
            self.blocks = nn.ModuleList(
                ResidualSignalBlock(hidden_dim, int(dilation), dropout)
                for dilation in dilations
            )
            self.output_norm = nn.LayerNorm(hidden_dim)
            self.output = nn.Linear(hidden_dim, 1)

        def forward(self, features: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
            values = self.input_projection(features) * mask.unsqueeze(-1)
            for block in self.blocks:
                values = block(values, mask)
            return self.output(self.output_norm(values)).squeeze(-1)

else:

    class SignalTCN:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            require_torch()


@dataclass(frozen=True)
class SignalChunk:
    document_index: int
    input_start: int
    input_end: int
    target_start: int
    target_end: int


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


def build_signal_features(
    probability: np.ndarray,
    roles: np.ndarray,
    header_kinds: np.ndarray,
) -> np.ndarray:
    """Build the intentionally narrow, line-aligned signal matrix."""

    probability = np.asarray(probability)
    roles = np.asarray(roles)
    header_kinds = np.asarray(header_kinds)
    if probability.ndim != 1 or not (
        len(probability) == len(roles) == len(header_kinds)
    ):
        raise ValueError("signal inputs must be line aligned")
    if roles.shape != (len(probability), len(ROLE_NAMES)):
        raise ValueError("deterministic role width changed")
    if not np.isfinite(probability).all() or np.any(
        (probability < 0.0) | (probability > 1.0)
    ):
        raise ValueError("entry probabilities must be finite values in [0, 1]")
    if np.any((roles != 0) & (roles != 1)) or np.any(roles.sum(axis=1) > 1):
        raise ValueError("deterministic roles must be mutually exclusive binary cues")
    return np.column_stack(
        (
            probability.astype(np.float32),
            roles.astype(np.float32),
            (header_kinds > 0).astype(np.float32),
        )
    )


def make_signal_chunks(
    table: Any,
    document_indices: Sequence[int],
    *,
    central_width: int,
    context: int,
) -> list[SignalChunk]:
    """Chunk known physical segments while assigning every known line once."""

    if central_width < 1 or context < 0:
        raise ValueError("central width and context are invalid")
    unknown = LABEL_TO_ID["UNKNOWN"]
    chunks: list[SignalChunk] = []
    for document_index in document_indices:
        document = table.documents[int(document_index)]
        doc_start, doc_end = int(document["line_start"]), int(document["line_end"])
        cursor = doc_start
        while cursor < doc_end:
            while cursor < doc_end and int(table.original_labels[cursor]) == unknown:
                cursor += 1
            segment_start = cursor
            while cursor < doc_end and int(table.original_labels[cursor]) != unknown:
                if (
                    cursor > segment_start
                    and int(table.abs_indices[cursor])
                    - int(table.abs_indices[cursor - 1])
                    > MAX_PHYSICAL_GAP
                ):
                    break
                cursor += 1
            segment_end = cursor
            for target_start in range(segment_start, segment_end, central_width):
                target_end = min(segment_end, target_start + central_width)
                chunks.append(
                    SignalChunk(
                        document_index=int(document_index),
                        input_start=max(segment_start, target_start - context),
                        input_end=min(segment_end, target_end + context),
                        target_start=target_start,
                        target_end=target_end,
                    )
                )
            if cursor == segment_start:
                cursor += 1
    return chunks


def _collate(
    chunks: Sequence[SignalChunk],
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[Any, Any, Any, Any]:
    require_torch()
    if not chunks:
        raise ValueError("cannot collate an empty signal-TCN batch")
    width = max(chunk.input_end - chunk.input_start for chunk in chunks)
    x = torch.zeros((len(chunks), width, features.shape[1]), dtype=torch.float32)
    line_mask = torch.zeros((len(chunks), width), dtype=torch.bool)
    loss_mask = torch.zeros((len(chunks), width), dtype=torch.bool)
    y = torch.zeros((len(chunks), width), dtype=torch.float32)
    bib = LABEL_TO_ID["BIB"]
    for batch_index, chunk in enumerate(chunks):
        length = chunk.input_end - chunk.input_start
        x[batch_index, :length] = torch.from_numpy(
            np.asarray(features[chunk.input_start : chunk.input_end])
        )
        line_mask[batch_index, :length] = True
        local_start = chunk.target_start - chunk.input_start
        local_end = chunk.target_end - chunk.input_start
        loss_mask[batch_index, local_start:local_end] = True
        y[batch_index, :length] = torch.from_numpy(
            (np.asarray(labels[chunk.input_start : chunk.input_end]) == bib).astype(
                np.float32
            )
        )
    return x, line_mask, loss_mask, y


def _model_parameter_count(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _fit_fold(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        table_dir,
        probability_path,
        role_path,
        excluded_ids,
        output_model_path,
        fold,
        central_width,
        context,
        hidden_dim,
        dilations,
        dropout,
        epochs,
        batch_size,
        learning_rate,
        weight_decay,
        positive_weight,
        gradient_clip,
        seed,
        torch_threads,
    ) = task
    require_torch()
    torch.set_num_threads(int(torch_threads))
    torch.set_num_interop_threads(1)
    fold_seed = int(seed) + int(fold)
    random.seed(fold_seed)
    np.random.seed(fold_seed)
    torch.manual_seed(fold_seed)
    torch.use_deterministic_algorithms(True)

    table = load_table(table_dir, expected_split="train")
    probability = np.load(probability_path, mmap_mode="r", allow_pickle=False)
    roles = _validate_role_matrix(Path(role_path), len(table.targets))
    features = build_signal_features(probability, roles, table.header_kinds)
    excluded = set(excluded_ids)
    fit_docs = [
        index
        for index, document in enumerate(table.documents)
        if int(document["fold"]) != int(fold)
        and str(document["document_id"]) not in excluded
    ]
    holdout_docs = [
        index
        for index, document in enumerate(table.documents)
        if int(document["fold"]) == int(fold)
        and str(document["document_id"]) not in excluded
    ]
    fit_chunks = make_signal_chunks(
        table, fit_docs, central_width=int(central_width), context=int(context)
    )
    holdout_chunks = make_signal_chunks(
        table, holdout_docs, central_width=int(central_width), context=int(context)
    )
    model = SignalTCN(
        len(FEATURE_NAMES),
        hidden_dim=int(hidden_dim),
        dilations=tuple(int(value) for value in dilations),
        dropout=float(dropout),
    ).cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    rng = random.Random(fold_seed)
    order = list(range(len(fit_chunks)))
    history: list[float] = []
    positive = torch.tensor(float(positive_weight), dtype=torch.float32)
    for _epoch in range(int(epochs)):
        rng.shuffle(order)
        model.train()
        total_loss = 0.0
        total_lines = 0
        for offset in range(0, len(order), int(batch_size)):
            batch = [fit_chunks[index] for index in order[offset : offset + int(batch_size)]]
            x, line_mask, loss_mask, y = _collate(batch, features, table.original_labels)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, line_mask)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[loss_mask], y[loss_mask], pos_weight=positive
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("signal TCN produced a non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
            optimizer.step()
            lines = int(loss_mask.sum().item())
            total_loss += float(loss.detach().item()) * lines
            total_lines += lines
        history.append(total_loss / max(total_lines, 1))

    indices: list[np.ndarray] = []
    values: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(holdout_chunks), int(batch_size)):
            batch = holdout_chunks[offset : offset + int(batch_size)]
            x, line_mask, _loss_mask, _y = _collate(
                batch, features, table.original_labels
            )
            probability_batch = torch.sigmoid(model(x, line_mask)).cpu().numpy()
            for batch_index, chunk in enumerate(batch):
                local_start = chunk.target_start - chunk.input_start
                local_end = chunk.target_end - chunk.input_start
                indices.append(np.arange(chunk.target_start, chunk.target_end, dtype=np.uint32))
                values.append(
                    probability_batch[batch_index, local_start:local_end].astype(
                        np.float32
                    )
                )
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "fold": int(fold),
        "feature_names": FEATURE_NAMES,
        "architecture": {
            "hidden_dim": int(hidden_dim),
            "dilations": [int(value) for value in dilations],
            "dropout": float(dropout),
        },
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, output_model_path)
    return {
        "fold": int(fold),
        "indices": np.concatenate(indices) if indices else np.empty(0, dtype=np.uint32),
        "probability": np.concatenate(values) if values else np.empty(0, dtype=np.float32),
        "fit_document_count": len(fit_docs),
        "holdout_document_count": len(holdout_docs),
        "fit_chunk_count": len(fit_chunks),
        "holdout_chunk_count": len(holdout_chunks),
        "history": history,
        "parameter_count": _model_parameter_count(model),
    }


def decode_signal_probability(
    table: Any,
    signal_probability: np.ndarray,
    frozen_entry_probability: np.ndarray,
    auxiliary_scope: np.ndarray,
    config: BlockConfig,
    *,
    threshold: float,
    qualified_documents: set[int],
    apply_veto: bool,
) -> tuple[np.ndarray, int]:
    """Threshold contextual scores, veto scoped components, then attach H0."""

    prediction = np.zeros(len(table.targets), dtype=bool)
    vetoed = 0
    for document_index in sorted(qualified_documents):
        document = table.documents[document_index]
        start, end = int(document["line_start"]), int(document["line_end"])
        local = signal_probability[start:end] >= float(threshold)
        local_scope = auxiliary_scope[start:end]
        local_absolute = table.abs_indices[start:end]
        if apply_veto:
            for block_start, block_end in blocks_from_mask(local, local_absolute):
                if has_auxiliary_scope(
                    local_scope,
                    local_absolute,
                    block_start,
                    end=block_end,
                    window=config.header_window,
                ):
                    local[block_start : block_end + 1] = False
                    vetoed += 1
        local = attach_h0_document(
            local,
            frozen_entry_probability[start:end],
            table.header_kinds[start:end],
            local_absolute,
            config,
        )
        prediction[start:end] = local
    return prediction, vetoed


def _selection_key(row: dict[str, Any]) -> tuple[float, ...]:
    metrics = row["metrics"]
    return (
        float(metrics["token_recall"]),
        float(metrics["line_recall"]),
        float(metrics["token_precision"]),
        -float(row["threshold"]),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    require_torch()
    table = load_table(args.table_dir, expected_split="train")
    line_root = Path(args.line_oof_dir).resolve()
    roles_root = Path(args.deterministic_roles_dir).resolve()
    role_report_path = roles_root / "deterministic_roles_report.json"
    role_report = json.loads(role_report_path.read_text(encoding="utf-8"))
    if (
        role_report.get("schema_version") != ROLE_SCHEMA
        or tuple(role_report.get("role_names", ())) != ROLE_NAMES
        or role_report.get("validation_opened") is not False
    ):
        raise ValueError("signal TCN requires frozen train-only deterministic roles")
    role_path = roles_root / "negative_roles.npy"
    _validate_role_matrix(role_path, len(table.targets))
    probability_path = line_root / f"{args.line_arm}.oof_probability.npy"
    frozen_probability = np.load(probability_path, mmap_mode="r", allow_pickle=False)
    if len(frozen_probability) != len(table.targets):
        raise ValueError("frozen entry probability is not line aligned")
    block_report_path = Path(args.block_oof_dir).resolve() / "block_oof_report.json"
    block_report = json.loads(block_report_path.read_text(encoding="utf-8"))
    config = BlockConfig(**block_report["arms"][args.line_arm]["selected_config"])
    excluded_ids, quality_packet = _load_quality_exclusions(
        Path(args.quality_decisions).resolve()
    )
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
    output_dir.mkdir(parents=True)
    model_dir = output_dir / "models"
    model_dir.mkdir()

    folds = int(table.manifest["n_folds"])
    worker_count = min(int(args.workers), folds)
    torch_threads = max(1, int(args.cpus) // worker_count)
    tasks = [
        (
            str(Path(args.table_dir).resolve()),
            str(probability_path),
            str(role_path),
            tuple(sorted(excluded_ids)),
            str(model_dir / f"fold{fold}.pt"),
            fold,
            int(args.central_width),
            int(args.context),
            int(args.hidden_dim),
            tuple(args.dilations),
            float(args.dropout),
            int(args.epochs),
            int(args.batch_size),
            float(args.learning_rate),
            float(args.weight_decay),
            float(args.positive_weight),
            float(args.gradient_clip),
            int(args.seed),
            torch_threads,
        )
        for fold in range(folds)
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        fold_results = list(executor.map(_fit_fold, tasks, chunksize=1))

    oof_probability = np.full(len(table.targets), np.nan, dtype=np.float32)
    fold_rows: list[dict[str, Any]] = []
    for result in sorted(fold_results, key=lambda row: int(row["fold"])):
        if np.any(np.isfinite(oof_probability[result["indices"]])):
            raise ValueError("a line received multiple OOF predictions")
        oof_probability[result["indices"]] = result["probability"]
        fold_rows.append({key: value for key, value in result.items() if key not in {"indices", "probability"}})
    qualified_line_mask = np.isin(
        table.document_indices, np.asarray(sorted(qualified_documents), dtype=table.document_indices.dtype)
    )
    known_line_mask = table.original_labels != LABEL_TO_ID["UNKNOWN"]
    required = qualified_line_mask & known_line_mask
    if not np.isfinite(oof_probability[required]).all():
        raise ValueError("OOF signal probabilities are incomplete")
    oof_probability[~np.isfinite(oof_probability)] = 0.0
    _save_array(output_dir / "signal_tcn_oof_probability.npy", oof_probability)

    auxiliary_headings, auxiliary_scope = materialize_auxiliary_headings(
        table, Path(args.input).resolve()
    )
    if np.any(
        auxiliary_scope & (table.original_labels == LABEL_TO_ID["BIB"])
    ):
        raise ValueError("audited auxiliary scope now overlaps silver bibliography")
    rows: list[dict[str, Any]] = []
    predictions: dict[float, np.ndarray] = {}
    for threshold in sorted(set(float(value) for value in args.thresholds)):
        baseline, _ = decode_signal_probability(
            table,
            oof_probability,
            frozen_probability,
            auxiliary_scope,
            config,
            threshold=threshold,
            qualified_documents=qualified_documents,
            apply_veto=False,
        )
        prediction, vetoed = decode_signal_probability(
            table,
            oof_probability,
            frozen_probability,
            auxiliary_scope,
            config,
            threshold=threshold,
            qualified_documents=qualified_documents,
            apply_veto=True,
        )
        predictions[threshold] = prediction
        rows.append(
            {
                "threshold": threshold,
                "vetoed_component_count": vetoed,
                "baseline_metrics": evaluate_prediction(
                    table, baseline, document_subset=qualified_documents
                ),
                "metrics": evaluate_prediction(
                    table, prediction, document_subset=qualified_documents
                ),
            }
        )
    safe = [row for row in rows if is_safe_candidate(row)]
    selected = max(safe, key=_selection_key) if safe else None
    if selected is not None:
        _save_array(
            output_dir / "selected_oof_prediction.npy",
            predictions[float(selected["threshold"])],
        )
    p95 = [row for row in rows if row["metrics"]["line_precision"] >= 0.95]
    diagnostic_p95 = max(p95, key=_selection_key) if p95 else None
    diagnostic_highest = max(rows, key=_selection_key)

    if selected is not None:
        selected_prediction = predictions[float(selected["threshold"])]
        for fold_row in fold_rows:
            fold = int(fold_row["fold"])
            subset = {
                index
                for index in qualified_documents
                if int(table.documents[index]["fold"]) == fold
            }
            fold_row["selected_metrics"] = evaluate_prediction(
                table, selected_prediction, document_subset=subset
            )
    receptive_field = 1 + 2 * sum(int(value) for value in args.dilations)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "passed_train_oof_safety_gate_validation_unopened"
            if selected is not None
            else "research_only_no_candidate_met_safety_gate"
        ),
        "feature_names": list(FEATURE_NAMES),
        "feature_contract": "no text, character length, document position, source identity, or validation input",
        "architecture": {
            "kind": "symmetric_residual_line_tcn_binary_segmentation",
            "hidden_dim": int(args.hidden_dim),
            "dilations": [int(value) for value in args.dilations],
            "dropout": float(args.dropout),
            "receptive_field_lines": receptive_field,
            "central_width": int(args.central_width),
            "context_lines": int(args.context),
        },
        "optimization": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "positive_weight": float(args.positive_weight),
            "gradient_clip": float(args.gradient_clip),
            "seed": int(args.seed),
        },
        "folds": fold_rows,
        "candidates": rows,
        "safe_candidate_count": len(safe),
        "selected": selected,
        "diagnostic_highest_recall_at_line_precision_0_95": diagnostic_p95,
        "diagnostic_highest_recall_candidate": diagnostic_highest,
        "selection_rule": "require line precision>=0.99 and <=0.02 spurious blocks per zero-BIB document; then maximize token and line recall",
        "scope_rule": "the audited exact auxiliary scope veto can remove a predicted component but cannot create or expand one",
        "auxiliary_heading_line_count": int(np.count_nonzero(auxiliary_headings)),
        "auxiliary_scope_line_count": int(np.count_nonzero(auxiliary_scope)),
        "quality_filter": {
            "decision_schema": quality_packet["schema_version"],
            "excluded_document_count": len(excluded_ids),
            "qualified_document_count": len(qualified_documents),
        },
        "block_config": asdict(config),
        "input_hashes": {
            "source": _sha256(Path(args.input).resolve()),
            "table_receipt": _sha256(Path(args.table_dir).resolve() / "receipt.json"),
            "line_probability": _sha256(probability_path),
            "role_report": _sha256(role_report_path),
            "role_matrix": _sha256(role_path),
            "block_report": _sha256(block_report_path),
            "quality_decisions": _sha256(Path(args.quality_decisions).resolve()),
        },
        "code_commit": str(args.code_commit),
        "slurm_job_id": str(args.slurm_job_id),
        "validation_opened": False,
        "production_eligible": False,
    }
    _write_json(output_dir / "signal_tcn_oof_report.json", result)
    outputs = {
        path.relative_to(output_dir).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }
    _write_json(output_dir / "receipt.json", {**result, "outputs": outputs})
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--line-oof-dir", required=True)
    parser.add_argument("--block-oof-dir", required=True)
    parser.add_argument("--deterministic-roles-dir", required=True)
    parser.add_argument("--quality-decisions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--line-arm", default="D1")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dilations", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--central-width", type=int, default=256)
    parser.add_argument("--context", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--cpus", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.85,
            0.875,
            0.90,
            0.925,
            0.95,
            0.96,
            0.97,
            0.975,
            0.98,
            0.985,
            0.99,
            0.9925,
            0.995,
            0.9975,
            0.999,
        ),
    )
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
