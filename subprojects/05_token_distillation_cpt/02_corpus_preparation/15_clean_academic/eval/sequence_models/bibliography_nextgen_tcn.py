#!/usr/bin/env python3
"""Train a grouped out-of-fold TCN over full-document line features."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - permits dependency-light imports
    torch = None
    nn = None

from .bibliography_entry_dataset import LABEL_TO_ID, MAX_PHYSICAL_GAP
from .bibliography_entry_models import load_table
from .bibliography_nextgen_models import SCHEMA_VERSION as MODEL_SCHEMA
from .bibliography_nextgen_table import SCHEMA_VERSION as TABLE_SCHEMA
from .contract import sha256_file


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("bibliography_nextgen_tcn requires PyTorch")


if torch is not None:

    class ResidualBlock(nn.Module):
        def __init__(self, width: int, dilation: int, dropout: float):
            super().__init__()
            self.norm = nn.LayerNorm(width)
            self.convolution = nn.Conv1d(
                width, width, kernel_size=3, padding=dilation, dilation=dilation
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, values: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
            residual = values
            values = self.norm(values).transpose(1, 2)
            values = torch.nn.functional.gelu(self.convolution(values)).transpose(1, 2)
            return (residual + self.dropout(values)) * mask.unsqueeze(-1)


    class FeatureTCN(nn.Module):
        def __init__(
            self,
            input_dim: int,
            *,
            width: int,
            dilations: Sequence[int],
            dropout: float,
        ):
            super().__init__()
            self.input_projection = nn.Linear(input_dim, width)
            self.blocks = nn.ModuleList(
                ResidualBlock(width, int(dilation), dropout) for dilation in dilations
            )
            self.output_norm = nn.LayerNorm(width)
            self.output = nn.Linear(width, 1)

        def forward(self, features: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
            values = self.input_projection(features) * mask.unsqueeze(-1)
            for block in self.blocks:
                values = block(values, mask)
            return self.output(self.output_norm(values)).squeeze(-1)

else:

    class FeatureTCN:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            require_torch()


@dataclass(frozen=True)
class Chunk:
    input_start: int
    input_end: int
    target_start: int
    target_end: int


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _save(path: Path, value: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)


def make_chunks(
    table: Any,
    document_indices: Sequence[int],
    *,
    central_width: int,
    context: int,
) -> list[Chunk]:
    """Cover each trusted physical segment once, with bounded context."""

    chunks: list[Chunk] = []
    unknown = LABEL_TO_ID["UNKNOWN"]
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
                    Chunk(
                        input_start=max(segment_start, target_start - context),
                        input_end=min(segment_end, target_end + context),
                        target_start=target_start,
                        target_end=target_end,
                    )
                )
            if cursor == segment_start:
                cursor += 1
    return chunks


def _moments(
    features: np.ndarray, table: Any, document_indices: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(features.shape[1], dtype=np.float64)
    squares = np.zeros(features.shape[1], dtype=np.float64)
    count = 0
    unknown = LABEL_TO_ID["UNKNOWN"]
    for document_index in document_indices:
        document = table.documents[int(document_index)]
        start, end = int(document["line_start"]), int(document["line_end"])
        trusted = table.original_labels[start:end] != unknown
        values = np.asarray(features[start:end][trusted], dtype=np.float64)
        total += values.sum(axis=0)
        squares += np.square(values).sum(axis=0)
        count += len(values)
    if not count:
        raise ValueError("TCN fold has no trusted fit lines")
    mean = total / count
    variance = np.maximum(0.0, squares / count - np.square(mean))
    scale = np.sqrt(variance)
    scale[scale < 1.0e-6] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def _collate(
    chunks: Sequence[Chunk],
    features: np.ndarray,
    labels: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> tuple[Any, Any, Any, Any]:
    require_torch()
    width = max(chunk.input_end - chunk.input_start for chunk in chunks)
    x = torch.zeros((len(chunks), width, features.shape[1]), dtype=torch.float32)
    mask = torch.zeros((len(chunks), width), dtype=torch.bool)
    loss_mask = torch.zeros((len(chunks), width), dtype=torch.bool)
    y = torch.zeros((len(chunks), width), dtype=torch.float32)
    for batch_index, chunk in enumerate(chunks):
        length = chunk.input_end - chunk.input_start
        values = np.asarray(features[chunk.input_start : chunk.input_end])
        x[batch_index, :length] = torch.from_numpy((values - mean) / scale)
        mask[batch_index, :length] = True
        local_start = chunk.target_start - chunk.input_start
        local_end = chunk.target_end - chunk.input_start
        loss_mask[batch_index, local_start:local_end] = True
        y[batch_index, :length] = torch.from_numpy(
            (
                np.asarray(labels[chunk.input_start : chunk.input_end])
                == LABEL_TO_ID["BIB"]
            ).astype(np.float32)
        )
    return x, mask, loss_mask, y


def _fit_fold(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        table_dir,
        feature_path,
        model_path,
        fold,
        central_width,
        context,
        width,
        dilations,
        dropout,
        epochs,
        batch_size,
        learning_rate,
        weight_decay,
        maximum_positive_weight,
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
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    fit_docs = [
        index
        for index, document in enumerate(table.documents)
        if int(document["fold"]) != int(fold)
    ]
    holdout_docs = [
        index
        for index, document in enumerate(table.documents)
        if int(document["fold"]) == int(fold)
    ]
    fit_chunks = make_chunks(
        table, fit_docs, central_width=int(central_width), context=int(context)
    )
    holdout_chunks = make_chunks(
        table, holdout_docs, central_width=int(central_width), context=int(context)
    )
    mean, scale = _moments(features, table, fit_docs)
    fit_line_mask = np.isin(table.document_indices, fit_docs) & (
        table.original_labels != LABEL_TO_ID["UNKNOWN"]
    )
    positive_count = int(
        np.count_nonzero(fit_line_mask & (table.original_labels == LABEL_TO_ID["BIB"]))
    )
    negative_count = int(np.count_nonzero(fit_line_mask)) - positive_count
    positive_weight = min(
        float(maximum_positive_weight), negative_count / max(positive_count, 1)
    )
    model = FeatureTCN(
        features.shape[1],
        width=int(width),
        dilations=tuple(int(value) for value in dilations),
        dropout=float(dropout),
    ).cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    positive = torch.tensor(positive_weight, dtype=torch.float32)
    order = list(range(len(fit_chunks)))
    rng = random.Random(fold_seed)
    history: list[float] = []
    for _epoch in range(int(epochs)):
        rng.shuffle(order)
        model.train()
        total_loss = 0.0
        total_lines = 0
        for offset in range(0, len(order), int(batch_size)):
            batch = [fit_chunks[index] for index in order[offset : offset + int(batch_size)]]
            x, mask, loss_mask, y = _collate(
                batch, features, table.original_labels, mean, scale
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, mask)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[loss_mask], y[loss_mask], pos_weight=positive
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nextgen TCN produced non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lines = int(loss_mask.sum().item())
            total_loss += float(loss.detach()) * lines
            total_lines += lines
        history.append(total_loss / max(total_lines, 1))
    indices: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(holdout_chunks), int(batch_size)):
            batch = holdout_chunks[offset : offset + int(batch_size)]
            x, mask, _loss_mask, _y = _collate(
                batch, features, table.original_labels, mean, scale
            )
            values = torch.sigmoid(model(x, mask)).numpy()
            for batch_index, chunk in enumerate(batch):
                local_start = chunk.target_start - chunk.input_start
                local_end = chunk.target_end - chunk.input_start
                indices.append(
                    np.arange(chunk.target_start, chunk.target_end, dtype=np.uint32)
                )
                probabilities.append(
                    values[batch_index, local_start:local_end].astype(np.float32)
                )
    torch.save(
        {
            "kind": "nextgen_tcn",
            "fold": int(fold),
            "feature_count": features.shape[1],
            "architecture": {
                "width": int(width),
                "dilations": [int(value) for value in dilations],
                "dropout": float(dropout),
            },
            "mean": torch.from_numpy(mean),
            "scale": torch.from_numpy(scale),
            "state_dict": model.state_dict(),
        },
        model_path,
    )
    return {
        "fold": int(fold),
        "indices": np.concatenate(indices),
        "probability": np.concatenate(probabilities),
        "fit_document_count": len(fit_docs),
        "holdout_document_count": len(holdout_docs),
        "fit_chunk_count": len(fit_chunks),
        "holdout_chunk_count": len(holdout_chunks),
        "positive_weight": positive_weight,
        "history": history,
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
    }


def _metrics(table: Any, probability: np.ndarray) -> dict[str, Any]:
    trusted = table.original_labels != LABEL_TO_ID["UNKNOWN"]
    truth = table.original_labels == LABEL_TO_ID["BIB"]
    prediction = probability >= 0.5
    tp = int(np.count_nonzero(prediction & truth & trusted))
    fp = int(np.count_nonzero(prediction & ~truth & trusted))
    fn = int(np.count_nonzero(~prediction & truth & trusted))
    return {
        "precision_at_0_5": tp / (tp + fp) if tp + fp else 1.0,
        "recall_at_0_5": tp / (tp + fn) if tp + fn else 0.0,
        "tp_at_0_5": tp,
        "fp_at_0_5": fp,
        "fn_at_0_5": fn,
        "brier": float(np.mean(np.square(probability[trusted] - truth[trusted]))),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    require_torch()
    table_root = Path(args.table_dir).resolve()
    base_root = Path(args.base_table_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    manifest = json.loads((table_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != TABLE_SCHEMA or manifest.get("test_opened") is not False:
        raise ValueError("TCN requires the sealed full-development feature table")
    table = load_table(base_root, expected_split="train")
    feature_path = table_root / "features.npy"
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    if features.shape != (len(table.targets), int(manifest["feature_count"])):
        raise ValueError("nextgen feature table shape mismatch")
    output.mkdir(parents=True)
    model_root = output / "models"
    model_root.mkdir()
    folds = int(table.manifest["n_folds"])
    workers = min(int(args.workers), folds)
    tasks = [
        (
            str(base_root),
            str(feature_path),
            str(model_root / f"fold{fold}.pt"),
            fold,
            int(args.central_width),
            int(args.context),
            int(args.width),
            tuple(args.dilations),
            float(args.dropout),
            int(args.epochs),
            int(args.batch_size),
            float(args.learning_rate),
            float(args.weight_decay),
            float(args.maximum_positive_weight),
            int(args.seed),
            max(1, int(args.cpus) // workers),
        )
        for fold in range(folds)
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_fit_fold, tasks, chunksize=1))
    probability = np.zeros(len(table.targets), dtype=np.float32)
    assigned = np.zeros(len(table.targets), dtype=np.uint8)
    fold_rows = []
    for result in results:
        indices = result.pop("indices")
        values = result.pop("probability")
        if np.any(assigned[indices]):
            raise RuntimeError("TCN OOF rows were assigned more than once")
        probability[indices] = values
        assigned[indices] = 1
        fold_rows.append(result)
    trusted = table.original_labels != LABEL_TO_ID["UNKNOWN"]
    if not np.all(assigned[trusted]) or not np.isfinite(probability).all():
        raise RuntimeError("TCN OOF prediction coverage is incomplete")
    _save(output / "oof_probability.npy", probability)
    report = {
        "schema_version": MODEL_SCHEMA,
        "status": "passed_grouped_oof_full_document_tcn_training",
        "kind": "tcn",
        "validation_opened": False,
        "test_opened": False,
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "feature_count": features.shape[1],
        "feature_names": manifest["feature_names"],
        "unknown_default_probability": 0.0,
        "architecture": {
            "central_width": int(args.central_width),
            "context": int(args.context),
            "width": int(args.width),
            "dilations": [int(value) for value in args.dilations],
            "dropout": float(args.dropout),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
        },
        "folds": fold_rows,
        "oof_metrics_before_block_decoding": _metrics(table, probability),
        "inputs": {
            "table_manifest_sha256": sha256_file(table_root / "manifest.json"),
            "table_features_sha256": sha256_file(feature_path),
            "base_manifest_sha256": sha256_file(base_root / "manifest.json"),
        },
    }
    _write_json_new(output / "report.json", report)
    _write_json_new(
        output / "receipt.json",
        {
            **report,
            "outputs": {
                str(path.relative_to(output)): {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in sorted(output.rglob("*"))
                if path.is_file()
            },
        },
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--base-table-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--central-width", type=int, default=256)
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--dilations", type=int, nargs="+", default=(1, 2, 4, 8, 16))
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--maximum-positive-weight", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--cpus", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
