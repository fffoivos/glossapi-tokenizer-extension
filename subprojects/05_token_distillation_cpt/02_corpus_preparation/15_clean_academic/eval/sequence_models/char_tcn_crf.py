#!/usr/bin/env python3
"""N1 byte-CNN + dilated line-TCN + masked BIOES CRF scaffold.

The byte encoder avoids a Greek/Latin/OCR vocabulary and performs no Unicode
normalization.  The TCN contextualizes complete line sequences; the CRF uses the
same legal BIOES transition mask as the feature baselines.  This module is an
offline CPU shadow candidate, never a replacement for the Rust hot path without
the separate promotion and runtime receipts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import re
import resource
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # Optional locally; available in the pinned Clariden PyTorch uenv.
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised by dependency-free smoke tests
    torch = None
    nn = None

from .bib_ladder import (
    configure_runtime,
    select_shared_calibration,
    verify_selection_bundle,
)
from .contract import GoldDocument, sha256_file
from .evaluate import evaluate, read_predictions
from .features import (
    TAGS,
    FeatureEncoder,
    allowed_transition_mask,
    bioes_to_classes,
    document_tag_ids,
)


def encode_utf8_lines(lines: Sequence[str], max_bytes: int = 256) -> list[list[int]]:
    """Encode each line as 1..256 byte IDs (0 is padding), without normalization."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    return [[byte + 1 for byte in text.encode("utf-8")[:max_bytes]] for text in lines]


def require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "char_tcn_crf requires PyTorch; run it on a Clariden CPU node with "
            "uenv run pytorch/v2.6.0:v1 --view=default"
        )


if torch is not None:

    class ByteCNN(nn.Module):
        def __init__(self, embedding_dim: int, channels: int, kernels: Sequence[int]):
            super().__init__()
            self.embedding = nn.Embedding(257, embedding_dim, padding_idx=0)
            self.convolutions = nn.ModuleList(
                nn.Conv1d(embedding_dim, channels, kernel_size=kernel, padding=kernel // 2)
                for kernel in kernels
            )
            self.output_dim = channels * len(kernels)

        def forward(self, byte_ids: "torch.Tensor") -> "torch.Tensor":
            # [batch, lines, bytes] -> [batch, lines, char_features]
            batch, lines, width = byte_ids.shape
            flat = byte_ids.reshape(batch * lines, width)
            embedded = self.embedding(flat).transpose(1, 2)
            pooled = []
            valid = flat.ne(0).any(dim=1, keepdim=True)
            byte_mask = flat.ne(0).unsqueeze(1)
            for convolution in self.convolutions:
                values = torch.relu(convolution(embedded))
                # A shorter line must not pool the learned convolution bias from
                # the batch padding region as if it were character evidence.
                values = values.masked_fill(~byte_mask, torch.finfo(values.dtype).min)
                values = values.amax(dim=2)
                pooled.append(values * valid)
            return torch.cat(pooled, dim=1).reshape(batch, lines, self.output_dim)


    class ResidualTCNBlock(nn.Module):
        def __init__(self, hidden_dim: int, dilation: int, dropout: float):
            super().__init__()
            self.norm = nn.LayerNorm(hidden_dim)
            self.conv = nn.Conv1d(
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
            values = torch.nn.functional.gelu(self.conv(values)).transpose(1, 2)
            values = residual + self.dropout(values)
            return values * mask.unsqueeze(-1)


    class MaskedBIOESCRF(nn.Module):
        """Batched log-likelihood and Viterbi with forbidden BIOES transitions masked."""

        def __init__(
            self, n_tags: int, active_classes: Sequence[str] = ("BIB", "TOC")
        ):
            super().__init__()
            transition_mask, start_mask, end_mask = allowed_transition_mask()
            unknown = set(active_classes) - {"BIB", "TOC"}
            if unknown:
                raise ValueError(f"unknown active classes: {sorted(unknown)!r}")
            active = torch.as_tensor([
                tag == "O" or any(tag.endswith(f"-{target}") for target in active_classes)
                for tag in TAGS
            ])
            self.transitions = nn.Parameter(torch.zeros(n_tags, n_tags))
            self.start = nn.Parameter(torch.zeros(n_tags))
            self.end = nn.Parameter(torch.zeros(n_tags))
            self.register_buffer(
                "transition_mask", torch.as_tensor(transition_mask) & active[:, None] & active[None, :]
            )
            self.register_buffer("start_mask", torch.as_tensor(start_mask) & active)
            self.register_buffer("end_mask", torch.as_tensor(end_mask) & active)
            self.register_buffer("active_tag_mask", active)

        @staticmethod
        def _masked(values: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
            return values.masked_fill(~mask, -torch.inf)

        def _local_graph(
            self,
        ) -> tuple[
            "torch.Tensor",
            "torch.Tensor",
            "torch.Tensor",
            "torch.Tensor",
            "torch.Tensor",
            "torch.Tensor",
        ]:
            active = self.active_tag_mask.nonzero(as_tuple=False).flatten()
            transition_mask = self.transition_mask.index_select(0, active).index_select(
                1, active
            )
            start_mask = self.start_mask.index_select(0, active)
            end_mask = self.end_mask.index_select(0, active)
            if (
                not bool(start_mask.any())
                or not bool(end_mask.any())
                or not bool(transition_mask.any(dim=0).all())
                or not bool(transition_mask.any(dim=1).all())
            ):
                raise ValueError("active CRF graph is not a complete BIOES state machine")
            transitions = self.transitions.index_select(0, active).index_select(1, active)
            return active, transition_mask, start_mask, end_mask, transitions, self.start.index_select(0, active)

        @staticmethod
        def _validate_sequence_mask(mask: "torch.Tensor") -> None:
            if mask.ndim != 2 or mask.shape[1] < 1:
                raise ValueError("CRF mask must be a non-empty batch-by-time tensor")
            if not bool(mask[:, 0].all()) or bool((mask[:, 1:] & ~mask[:, :-1]).any()):
                raise ValueError("CRF masks must be non-empty contiguous prefixes")

        def log_partition(self, emissions: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
            self._validate_sequence_mask(mask)
            active, transition_mask, start_mask, end_mask, transitions, start = (
                self._local_graph()
            )
            emissions = emissions.index_select(2, active)
            transitions = self._masked(transitions, transition_mask)
            score = self._masked(start, start_mask) + emissions[:, 0]
            for t in range(1, emissions.shape[1]):
                candidate = score.unsqueeze(2) + transitions.unsqueeze(0) + emissions[:, t].unsqueeze(1)
                next_score = torch.logsumexp(candidate, dim=1)
                score = torch.where(mask[:, t].unsqueeze(1), next_score, score)
            end = self.end.index_select(0, active)
            return torch.logsumexp(score[:, end_mask] + end[end_mask], dim=1)

        def gold_score(
            self, emissions: "torch.Tensor", tags: "torch.Tensor", mask: "torch.Tensor"
        ) -> "torch.Tensor":
            self._validate_sequence_mask(mask)
            if not bool(self.active_tag_mask[tags[mask]].all()):
                raise ValueError("gold contains a class disabled for this evidence tier")
            lengths = mask.long().sum(dim=1) - 1
            last_tags = tags.gather(1, lengths.unsqueeze(1)).squeeze(1)
            if not bool(self.start_mask[tags[:, 0]].all()) or not bool(
                self.end_mask[last_tags].all()
            ):
                raise ValueError("gold contains an illegal BIOES boundary tag")
            if tags.shape[1] > 1:
                legal_steps = self.transition_mask[tags[:, :-1], tags[:, 1:]]
                if not bool(legal_steps[mask[:, 1:]].all()):
                    raise ValueError("gold contains an illegal BIOES transition")
            if bool((tags[mask] < 0).any()) or bool((tags[mask] >= len(TAGS)).any()):
                raise ValueError("gold contains a tag index outside the BIOES inventory")
            batch = torch.arange(emissions.shape[0], device=emissions.device)
            score = self.start[tags[:, 0]]
            score = score + emissions[batch, 0, tags[:, 0]]
            for t in range(1, emissions.shape[1]):
                step = self.transitions[tags[:, t - 1], tags[:, t]] + emissions[batch, t, tags[:, t]]
                score = score + torch.where(mask[:, t], step, torch.zeros_like(step))
            return score + self.end[last_tags]

        def neg_log_likelihood(
            self, emissions: "torch.Tensor", tags: "torch.Tensor", mask: "torch.Tensor"
        ) -> "torch.Tensor":
            if not bool(mask[:, 0].all()):
                raise ValueError("every sequence must contain at least one line")
            return (self.log_partition(emissions, mask) - self.gold_score(emissions, tags, mask)).mean()

        def decode(
            self,
            emissions: "torch.Tensor",
            mask: "torch.Tensor",
            *,
            deletion_bias: float = 0.0,
        ) -> list[list[int]]:
            self._validate_sequence_mask(mask)
            if deletion_bias:
                emissions = emissions.clone()
                emissions[:, :, 1:] -= float(deletion_bias)
            active, transition_mask, start_mask, end_mask, transitions, start = (
                self._local_graph()
            )
            emissions = emissions.index_select(2, active)
            transitions = self._masked(transitions, transition_mask)
            score = self._masked(start, start_mask) + emissions[:, 0]
            histories: list[torch.Tensor] = []
            for t in range(1, emissions.shape[1]):
                candidate = score.unsqueeze(2) + transitions.unsqueeze(0)
                best_score, best_tag = candidate.max(dim=1)
                next_score = best_score + emissions[:, t]
                score = torch.where(mask[:, t].unsqueeze(1), next_score, score)
                histories.append(best_tag)
            end = self.end.index_select(0, active)
            allowed_end = end_mask.nonzero(as_tuple=False).flatten()
            final_score = score.index_select(1, allowed_end) + end.index_select(
                0, allowed_end
            )
            last = allowed_end[final_score.argmax(dim=1)]
            decoded: list[list[int]] = []
            for b in range(emissions.shape[0]):
                length = int(mask[b].sum().item())
                path = [int(last[b].item())]
                for t in range(length - 2, -1, -1):
                    path.append(int(histories[t][b, path[-1]].item()))
                decoded.append(active[list(reversed(path))].tolist())
            return decoded


    class CharTCNCRF(nn.Module):
        def __init__(
            self,
            *,
            engineered_dim: int,
            byte_embedding_dim: int = 32,
            char_channels_per_kernel: int = 48,
            char_kernels: Sequence[int] = (3, 5, 7),
            hidden_dim: int = 128,
            tcn_dilations: Sequence[int] = (1, 2, 4, 8),
            dropout: float = 0.15,
            target_classes: Sequence[str] = ("BIB", "TOC"),
        ):
            super().__init__()
            self.byte_cnn = ByteCNN(byte_embedding_dim, char_channels_per_kernel, char_kernels)
            self.input_projection = nn.Linear(self.byte_cnn.output_dim + engineered_dim, hidden_dim)
            self.blocks = nn.ModuleList(
                ResidualTCNBlock(hidden_dim, dilation, dropout) for dilation in tcn_dilations
            )
            self.output_norm = nn.LayerNorm(hidden_dim)
            self.emissions = nn.Linear(hidden_dim, len(TAGS))
            self.crf = MaskedBIOESCRF(len(TAGS), target_classes)

        def forward(
            self,
            byte_ids: "torch.Tensor",
            engineered: "torch.Tensor",
            line_mask: "torch.Tensor",
        ) -> "torch.Tensor":
            chars = self.byte_cnn(byte_ids)
            values = self.input_projection(torch.cat((chars, engineered), dim=-1))
            values = values * line_mask.unsqueeze(-1)
            for block in self.blocks:
                values = block(values, line_mask)
            return self.emissions(self.output_norm(values))

        def loss(
            self,
            byte_ids: "torch.Tensor",
            engineered: "torch.Tensor",
            line_mask: "torch.Tensor",
            tags: "torch.Tensor",
        ) -> "torch.Tensor":
            return self.crf.neg_log_likelihood(self(byte_ids, engineered, line_mask), tags, line_mask)

        def decode(
            self,
            byte_ids: "torch.Tensor",
            engineered: "torch.Tensor",
            line_mask: "torch.Tensor",
            *,
            deletion_bias: float = 0.0,
        ) -> list[list[int]]:
            return self.crf.decode(
                self(byte_ids, engineered, line_mask),
                line_mask,
                deletion_bias=deletion_bias,
            )

else:

    class CharTCNCRF:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            require_torch()


def export_torchscript_emissions(
    model: Any,
    example_inputs: tuple[Any, Any, Any],
    output_path: str | Path,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Export deterministic CPU emissions and verify eager↔TorchScript parity.

    CRF decoding remains in the separately testable masked decoder; the exported
    graph contains byte-CNN + TCN + emission projection only.
    """
    require_torch()
    model = model.cpu().eval()
    inputs = tuple(value.cpu() for value in example_inputs)
    with torch.inference_mode():
        eager = model(*inputs)
        traced = torch.jit.trace(model, inputs, strict=True)
        scripted = traced(*inputs)
    maximum_absolute_delta = float((eager - scripted).abs().max().item())
    if maximum_absolute_delta > 1.0e-6:
        raise RuntimeError(f"TorchScript parity failed: max |delta|={maximum_absolute_delta}")
    output_path = Path(output_path)
    traced.save(str(output_path))
    receipt = {
        "schema_version": "academic-structure-char-tcn-export-v1",
        "device": "cpu",
        "format": "torchscript-emissions",
        "maximum_absolute_delta": maximum_absolute_delta,
        "metadata": dict(metadata),
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt


@dataclass(frozen=True)
class NeuralExample:
    document: GoldDocument
    line_indices: tuple[int, ...]
    byte_ids: tuple[tuple[int, ...], ...]
    engineered: tuple[tuple[float, ...], ...]
    tags: tuple[int, ...]


def make_neural_examples(
    documents: Sequence[GoldDocument],
    encoder: FeatureEncoder,
    *,
    max_bytes: int,
) -> list[NeuralExample]:
    """Build known-label segments without ever coercing UNKNOWN to ``O``."""
    examples: list[NeuralExample] = []
    for document in documents:
        sparse = encoder.encode_document(document)
        start = 0
        while start < len(document.lines):
            while start < len(document.lines) and document.lines[start].label == "UNKNOWN":
                start += 1
            end = start
            while end < len(document.lines) and document.lines[end].label != "UNKNOWN":
                end += 1
            if end > start:
                segment = type("Segment", (), {"lines": document.lines[start:end]})()
                tag_ids = document_tag_ids(segment)
                dense_rows = []
                for row in sparse[start:end]:
                    dense = [0.0] * encoder.n_features
                    for index, value in row.items():
                        dense[int(index)] = float(value)
                    dense_rows.append(tuple(dense))
                examples.append(
                    NeuralExample(
                        document=document,
                        line_indices=tuple(range(start, end)),
                        byte_ids=tuple(
                            tuple(row)
                            for row in encode_utf8_lines(
                                [line.text for line in document.lines[start:end]],
                                max_bytes=max_bytes,
                            )
                        ),
                        engineered=tuple(dense_rows),
                        tags=tuple(int(value) for value in tag_ids),
                    )
                )
            start = end + 1
    return examples


def count_neural_sequences(documents: Sequence[GoldDocument]) -> int:
    """Count contiguous known-label segments without materializing model inputs."""
    count = 0
    for document in documents:
        previous_known = False
        for line in document.lines:
            known = line.label != "UNKNOWN"
            if known and not previous_known:
                count += 1
            previous_known = known
    return count


def _collate(
    examples: Sequence[NeuralExample],
) -> tuple[Any, Any, Any, Any]:
    require_torch()
    if not examples:
        raise ValueError("cannot collate an empty N1 batch")
    line_count = max(len(example.line_indices) for example in examples)
    byte_count = max(
        1,
        max(len(row) for example in examples for row in example.byte_ids),
    )
    engineered_dim = len(examples[0].engineered[0])
    byte_ids = torch.zeros((len(examples), line_count, byte_count), dtype=torch.long)
    engineered = torch.zeros(
        (len(examples), line_count, engineered_dim), dtype=torch.float32
    )
    line_mask = torch.zeros((len(examples), line_count), dtype=torch.bool)
    tags = torch.zeros((len(examples), line_count), dtype=torch.long)
    for batch_index, example in enumerate(examples):
        length = len(example.line_indices)
        line_mask[batch_index, :length] = True
        tags[batch_index, :length] = torch.as_tensor(example.tags, dtype=torch.long)
        engineered[batch_index, :length] = torch.as_tensor(
            example.engineered, dtype=torch.float32
        )
        for line_index, row in enumerate(example.byte_ids):
            if row:
                byte_ids[batch_index, line_index, : len(row)] = torch.as_tensor(
                    row, dtype=torch.long
                )
    return byte_ids, engineered, line_mask, tags


def _architecture(config: Mapping[str, Any]) -> Mapping[str, Any]:
    for architecture in config["architecture_ladder"]:
        if architecture["id"] == "n1-bytecnn-tcn-masked-crf":
            if architecture.get("kind") != "char_tcn_crf":
                raise ValueError("N1 config is not a char_tcn_crf architecture")
            return architecture
    raise ValueError("N1 architecture is absent from config")


def _make_model(
    architecture: Mapping[str, Any], engineered_dim: int
) -> Any:
    require_torch()
    return CharTCNCRF(
        engineered_dim=engineered_dim,
        byte_embedding_dim=int(architecture["byte_embedding_dim"]),
        char_channels_per_kernel=int(architecture["char_channels_per_kernel"]),
        char_kernels=tuple(int(value) for value in architecture["char_kernels"]),
        hidden_dim=int(architecture["hidden_dim"]),
        tcn_dilations=tuple(int(value) for value in architecture["tcn_dilations"]),
        dropout=float(architecture["dropout"]),
        target_classes=("BIB",),
    ).cpu()


def train_n1(
    model: Any,
    examples: Sequence[NeuralExample],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    seed: int,
) -> tuple[list[float], float]:
    """Run deterministic AdamW fitting on CPU and return epoch NLLs."""
    require_torch()
    if not examples or epochs < 1 or batch_size < 1:
        raise ValueError("N1 requires examples, positive epochs, and a positive batch size")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    order = list(range(len(examples)))
    rng = random.Random(seed)
    history: list[float] = []
    started = time.perf_counter()
    for _epoch in range(int(epochs)):
        rng.shuffle(order)
        total_loss = 0.0
        total_sequences = 0
        model.train()
        for offset in range(0, len(order), int(batch_size)):
            batch_examples = [examples[index] for index in order[offset : offset + batch_size]]
            byte_ids, engineered, line_mask, tags = _collate(batch_examples)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(byte_ids, engineered, line_mask, tags)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("N1 training produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
            optimizer.step()
            total_loss += float(loss.detach().item()) * len(batch_examples)
            total_sequences += len(batch_examples)
        history.append(total_loss / total_sequences)
    return history, time.perf_counter() - started


def _cache_emissions(
    model: Any,
    examples: Sequence[NeuralExample],
    *,
    batch_size: int,
) -> list[Any]:
    require_torch()
    cached: list[Any] = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(examples), batch_size):
            batch_examples = examples[offset : offset + batch_size]
            byte_ids, engineered, line_mask, _tags = _collate(batch_examples)
            emissions = model(byte_ids, engineered, line_mask).cpu()
            for batch_index, example in enumerate(batch_examples):
                length = len(example.line_indices)
                cached.append(emissions[batch_index, :length].contiguous())
    return cached


def _predictions_from_emissions(
    documents: Sequence[GoldDocument],
    examples: Sequence[NeuralExample],
    emissions: Sequence[Any],
    model: Any,
    *,
    deletion_bias: float,
) -> dict[str, list[str]]:
    require_torch()
    result = {document.document_id: ["O"] * len(document.lines) for document in documents}
    for example, values in zip(examples, emissions):
        mask = torch.ones((1, len(example.line_indices)), dtype=torch.bool)
        path = model.crf.decode(
            values.unsqueeze(0), mask, deletion_bias=float(deletion_bias)
        )[0]
        classes = bioes_to_classes([TAGS[index] for index in path])
        for line_index, label in zip(example.line_indices, classes):
            result[example.document.document_id][line_index] = label
    return result


def calibrate_n1(
    documents: Sequence[GoldDocument],
    examples: Sequence[NeuralExample],
    emissions: Sequence[Any],
    model: Any,
    candidates: Sequence[float],
    *,
    reference_action_precision: float,
) -> dict[str, Any]:
    """Select comparison-only deletion bias on validation silver."""
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        predictions = _predictions_from_emissions(
            documents, examples, emissions, model, deletion_bias=float(candidate)
        )
        metrics, _per_document = evaluate(documents, predictions, split="validation")
        predicted_tokens = sum(
            line.token_count
            for document in documents
            for line, guess in zip(
                document.lines, predictions[document.document_id]
            )
            if line.label != "UNKNOWN" and guess != "O"
        )
        rows.append(
            {
                "deletion_bias": float(candidate),
                "action_precision": metrics["token"]["action_precision"],
                "action_recall": metrics["token"]["action_recall"],
                "bib_recall": metrics["token"]["bib_recall"],
                "predicted_action_tokens": predicted_tokens,
            }
        )
    return select_shared_calibration(
        rows, reference_action_precision=reference_action_precision
    )


def _atomic_torch_save(path: str | Path, value: Any) -> None:
    require_torch()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable output {output}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial", dir=output.parent
    )
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        with Path(temporary).open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, output)
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable output {output}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_n1_predictions(
    path: str | Path,
    documents: Sequence[GoldDocument],
    predictions: Mapping[str, Sequence[str]],
) -> None:
    output = Path(path)
    rows = []
    for document in documents:
        guesses = predictions[document.document_id]
        rows.append(
            {
                "schema_version": "academic-structure-predictions-v1",
                "model_id": "n1-bytecnn-tcn-masked-crf",
                "document_id": document.document_id,
                "work_id": document.work_id,
                "source": document.source,
                "split": document.split,
                "lines": [
                    {
                        "line_id": line.line_id,
                        "abs_idx": line.abs_idx,
                        "prediction": guess,
                    }
                    for line, guess in zip(document.lines, guesses)
                ],
            }
        )
    payload = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in rows
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable output {output}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_n1_checkpoint(
    path: str | Path, config: Mapping[str, Any]
) -> tuple[Any, dict[str, Any]]:
    """Strictly load, reconstruct, and validate a real N1 checkpoint."""
    require_torch()
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before weights_only was added.
        value = torch.load(path, map_location="cpu")
    required = {
        "schema_version",
        "architecture_id",
        "architecture",
        "engineered_dim",
        "feature_encoder",
        "active_classes",
        "deletion_bias",
        "state_dict",
        "inputs",
        "seed",
        "production_eligible",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("N1 checkpoint has an unexpected field inventory")
    architecture = _architecture(config)
    encoder = FeatureEncoder(char_hash_dim=0)
    if (
        value.get("schema_version") != "n1-checkpoint-v2"
        or value.get("architecture_id") != "n1-bytecnn-tcn-masked-crf"
        or value.get("architecture") != architecture
        or value.get("engineered_dim") != encoder.n_features
        or value.get("feature_encoder") != encoder.metadata()
        or value.get("active_classes") != ["BIB"]
        or value.get("seed") != int(config["execution"]["seed"])
        or value.get("production_eligible") is not False
        or value.get("deletion_bias") not in config["calibration"]["deletion_bias_grid"]
    ):
        raise ValueError("N1 checkpoint metadata differs from the current config")
    inputs = value.get("inputs")
    if not isinstance(inputs, dict) or inputs.get("config_sha256") is None:
        raise ValueError("N1 checkpoint input bindings are missing")
    state = value.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("N1 checkpoint state_dict is missing")
    model = _make_model(architecture, encoder.n_features)
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("N1 checkpoint state_dict key inventory differs")
    for name, expected_tensor in expected.items():
        observed = state[name]
        if (
            not isinstance(observed, torch.Tensor)
            or observed.shape != expected_tensor.shape
            or observed.dtype != expected_tensor.dtype
            or not bool(torch.isfinite(observed).all())
        ):
            raise ValueError(f"N1 checkpoint tensor {name!r} is invalid")
        if name in {
            "crf.active_tag_mask",
            "crf.transition_mask",
            "crf.start_mask",
            "crf.end_mask",
        } and not torch.equal(observed, expected_tensor):
            raise ValueError(f"N1 checkpoint derived mask {name!r} differs from metadata")
    model.load_state_dict(state, strict=True)
    return model.cpu().eval(), value


def _require_clariden_cpu(confirmed: bool) -> None:
    if not confirmed:
        raise RuntimeError("N1 fitting requires --confirm-clariden-cpu-only")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("N1 fitting is forbidden outside a Clariden Slurm allocation")
    if os.environ.get("SLURM_JOB_PARTITION") not in {"normal", "debug"}:
        raise RuntimeError("N1 fitting requires a Clariden compute partition")
    if platform.machine() != "aarch64":
        raise RuntimeError("N1 fitting requires the Clariden aarch64 CPU runtime")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("N1 fitting requires CUDA_VISIBLE_DEVICES to disable accelerators")


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if platform.system() == "Darwin" else peak * 1024)


def _scrub_silver_safety(metrics: dict[str, Any]) -> None:
    for row in [metrics, *metrics.get("by_source", {}).values()]:
        row["token"]["prose_contamination"] = None
        row["token"]["true_main_text_retention"] = None
        row["token"]["toc_recall"] = None
        row["line"]["toc_recall"] = None
        row["span"]["toc"] = {key: None for key in row["span"]["toc"]}
        row["document"]["catastrophic_prose_deletions"] = None
        row["document"]["maximum_contiguous_false_deletion_tokens"] = None
    metrics["metric_availability"] = {
        "LLM_silver_agreement": True,
        "independent_running_prose_safety": False,
        "toc_supervision": False,
    }


def _state_digest(model: Any) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _tiny_determinism_smoke(seed: int) -> dict[str, Any]:
    """Fit two tiny synthetic models; callable only behind the Clariden gate."""
    require_torch()

    def once() -> tuple[str, list[float]]:
        torch.manual_seed(seed)
        model = CharTCNCRF(
            engineered_dim=2,
            byte_embedding_dim=4,
            char_channels_per_kernel=3,
            char_kernels=(3,),
            hidden_dim=8,
            tcn_dilations=(1,),
            dropout=0.0,
            target_classes=("BIB",),
        ).cpu()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        byte_ids = torch.tensor([[[66, 67, 0], [68, 69, 70], [71, 0, 0]]])
        engineered = torch.tensor([[[0.1, 0.2], [0.4, -0.1], [0.0, 0.3]]])
        mask = torch.ones((1, 3), dtype=torch.bool)
        tags = torch.tensor([[0, TAGS.index("S-BIB"), 0]])
        losses = []
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(byte_ids, engineered, mask, tags)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        return _state_digest(model), losses

    first_digest, first_loss = once()
    second_digest, second_loss = once()
    if first_digest != second_digest or first_loss != second_loss:
        raise RuntimeError("tiny N1 deterministic fitting smoke diverged")
    return {
        "status": "pass",
        "state_sha256": first_digest,
        "losses": first_loss,
        "replicas": 2,
        "steps_per_replica": 2,
    }


def validate_n1_profile_receipt(
    profile: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    expected_inputs: Mapping[str, str],
    expected_uenv: str,
    expected_code_commit: str,
    current_runtime: Mapping[str, Any],
    expected_counts: Mapping[str, int],
) -> None:
    """Validate that a separate profile authorizes this exact code/runtime fit."""
    require_torch()
    if not re.fullmatch(r"[0-9a-f]{40}", expected_code_commit):
        raise RuntimeError("N1 profile code commit must be an exact Git SHA")
    architecture = _architecture(config)
    smoke = profile.get("determinism_smoke")
    counts = profile.get("counts")
    one_epoch = profile.get("one_epoch_seconds")
    projected = profile.get("projected_full_fit_seconds_with_15pct_margin")
    maximum = profile.get("maximum_full_fit_seconds")
    numeric_timing = all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in (one_epoch, projected, maximum)
    )
    expected_projected = (
        float(one_epoch) * int(architecture["epochs"]) * 1.15
        if numeric_timing
        else math.nan
    )
    smoke_valid = (
        isinstance(smoke, Mapping)
        and set(smoke)
        == {"status", "state_sha256", "losses", "replicas", "steps_per_replica"}
        and smoke.get("status") == "pass"
        and isinstance(smoke.get("state_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", smoke["state_sha256"]) is not None
        and isinstance(smoke.get("losses"), list)
        and len(smoke["losses"]) == 2
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in smoke["losses"]
        )
        and smoke.get("replicas") == 2
        and smoke.get("steps_per_replica") == 2
    )
    counts_valid = counts == dict(expected_counts) and all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in expected_counts.values()
    )
    execution = profile.get("execution")
    runtime_keys = (
        "device",
        "accelerator_used",
        "uenv",
        "python_version",
        "python_implementation",
        "numpy_version",
        "torch_version",
        "python_hash_seed",
        "effective_seed",
        "deterministic_algorithms",
        "torch_intraop_threads",
        "torch_interop_threads",
        "omp_num_threads",
        "mkl_num_threads",
        "slurm_cpus_per_task",
    )
    if (
        profile.get("schema_version") != "academic-structure-n1-profile-v1"
        or profile.get("status")
        != "passed_one_epoch_profile_and_determinism_smoke"
        or profile.get("architecture_id") != "n1-bytecnn-tcn-masked-crf"
        or profile.get("production_eligible") is not False
        or profile.get("effective_seed") != int(config["execution"]["seed"])
        or profile.get("inputs") != dict(expected_inputs)
        or not isinstance(execution, Mapping)
        or execution.get("uenv") != expected_uenv
        or execution.get("code_commit") != expected_code_commit
        or any(execution.get(key) != current_runtime.get(key) for key in runtime_keys)
        or not smoke_valid
        or not counts_valid
        or not numeric_timing
        or float(one_epoch) <= 0.0
        or float(maximum) != 32400.0
        or not math.isclose(
            float(projected), expected_projected, rel_tol=1.0e-12, abs_tol=1.0e-9
        )
        or float(projected) > float(maximum)
        or profile.get("within_full_fit_budget") is not True
        or not isinstance(profile.get("peak_rss_bytes"), int)
        or isinstance(profile.get("peak_rss_bytes"), bool)
        or profile["peak_rss_bytes"] <= 0
    ):
        raise RuntimeError("N1 profile receipt does not authorize this exact full fit")


def profile_cli(args: argparse.Namespace) -> dict[str, Any]:
    require_torch()
    _require_clariden_cpu(args.confirm_clariden_cpu_only)
    if Path(args.receipt_out).exists() or Path(args.receipt_out).is_symlink():
        raise FileExistsError(f"refusing immutable output overwrite: {args.receipt_out}")
    documents, validation, selection_receipt = verify_selection_bundle(
        selection_silver_path=args.selection_silver,
        selection_manifest_path=args.selection_manifest,
        validation_silver_path=args.validation_silver,
        selection_receipt_path=args.selection_receipt,
        config_path=args.config,
    )
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    seed = int(config["execution"]["seed"])
    runtime = configure_runtime(config, uenv=args.uenv, effective_seed=seed)
    if not re.fullmatch(r"[0-9a-f]{40}", args.code_commit):
        raise RuntimeError("N1 profile code commit must be an exact Git SHA")
    runtime["code_commit"] = args.code_commit
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(bool(config["execution"]["deterministic_algorithms"]))
    smoke = _tiny_determinism_smoke(seed)
    architecture = _architecture(config)
    encoder = FeatureEncoder(char_hash_dim=0)
    train_documents = [document for document in documents if document.split == "train"]
    examples = make_neural_examples(
        train_documents,
        encoder,
        max_bytes=int(architecture["max_utf8_bytes_per_line"]),
    )
    model = _make_model(architecture, encoder.n_features)
    _history, epoch_seconds = train_n1(
        model,
        examples,
        epochs=1,
        batch_size=int(architecture["batch_size"]),
        learning_rate=float(architecture["learning_rate"]),
        weight_decay=float(architecture["weight_decay"]),
        gradient_clip=float(architecture["gradient_clip"]),
        seed=seed,
    )
    projected = epoch_seconds * int(architecture["epochs"]) * 1.15
    receipt = {
        "schema_version": "academic-structure-n1-profile-v1",
        "status": "passed_one_epoch_profile_and_determinism_smoke",
        "architecture_id": "n1-bytecnn-tcn-masked-crf",
        "production_eligible": False,
        "inputs": {
            "selection_silver_sha256": sha256_file(args.selection_silver),
            "selection_manifest_sha256": sha256_file(args.selection_manifest),
            "validation_silver_sha256": sha256_file(args.validation_silver),
            "selection_receipt_sha256": sha256_file(args.selection_receipt),
            "config_sha256": sha256_file(args.config),
            "source_rehydration_receipt_sha256": selection_receipt["source"][
                "rehydration_receipt_sha256"
            ],
        },
        "execution": runtime,
        "effective_seed": seed,
        "counts": {
            "train_documents": len(train_documents),
            "validation_documents_contract_checked_not_scored_by_profile": len(validation),
            "train_sequences": len(examples),
        },
        "determinism_smoke": smoke,
        "one_epoch_seconds": epoch_seconds,
        "projected_full_fit_seconds_with_15pct_margin": projected,
        "maximum_full_fit_seconds": 32400.0,
        "within_full_fit_budget": projected <= 32400.0,
        "peak_rss_bytes": _peak_rss_bytes(),
        "note": "config deployment resource gates are promotion-only and are not applied here",
    }
    if not receipt["within_full_fit_budget"]:
        raise RuntimeError("N1 profile projects beyond the non-resumable full-fit budget")
    _atomic_json(args.receipt_out, receipt)
    return receipt


def _validate_profile_receipt(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    documents: Sequence[GoldDocument],
    validation: Sequence[GoldDocument],
) -> dict[str, Any]:
    profile = json.loads(Path(args.profile_receipt).read_text(encoding="utf-8"))
    selection_receipt = json.loads(
        Path(args.selection_receipt).read_text(encoding="utf-8")
    )
    expected_inputs = {
        "selection_silver_sha256": sha256_file(args.selection_silver),
        "selection_manifest_sha256": sha256_file(args.selection_manifest),
        "validation_silver_sha256": sha256_file(args.validation_silver),
        "selection_receipt_sha256": sha256_file(args.selection_receipt),
        "config_sha256": sha256_file(args.config),
        "source_rehydration_receipt_sha256": selection_receipt["source"][
            "rehydration_receipt_sha256"
        ],
    }
    current_runtime = configure_runtime(
        config, uenv=args.uenv, effective_seed=int(config["execution"]["seed"])
    )
    validate_n1_profile_receipt(
        profile,
        config,
        expected_inputs=expected_inputs,
        expected_uenv=args.uenv,
        expected_code_commit=args.code_commit,
        current_runtime=current_runtime,
        expected_counts={
            "train_documents": sum(
                document.split == "train" for document in documents
            ),
            "validation_documents_contract_checked_not_scored_by_profile": len(
                validation
            ),
            "train_sequences": count_neural_sequences(
                [document for document in documents if document.split == "train"]
            ),
        },
    )
    return profile


def train_cli(args: argparse.Namespace) -> dict[str, Any]:
    require_torch()
    _require_clariden_cpu(args.confirm_clariden_cpu_only)
    for path in (args.model_out, args.validation_predictions, args.receipt_out):
        if Path(path).exists() or Path(path).is_symlink():
            raise FileExistsError(f"refusing immutable output overwrite: {path}")
    documents, validation, selection_receipt = verify_selection_bundle(
        selection_silver_path=args.selection_silver,
        selection_manifest_path=args.selection_manifest,
        validation_silver_path=args.validation_silver,
        selection_receipt_path=args.selection_receipt,
        config_path=args.config,
    )
    train_documents = [document for document in documents if document.split == "train"]
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    architecture = _architecture(config)
    seed = int(config["execution"]["seed"])
    runtime = configure_runtime(config, uenv=args.uenv, effective_seed=seed)
    if not re.fullmatch(r"[0-9a-f]{40}", args.code_commit):
        raise RuntimeError("N1 fit code commit must be an exact Git SHA")
    runtime["code_commit"] = args.code_commit
    profile = _validate_profile_receipt(args, config, documents, validation)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(bool(config["execution"]["deterministic_algorithms"]))
    reference_predictions = read_predictions(args.reference_predictions, validation)
    reference_metrics, _ = evaluate(
        validation, reference_predictions, split="validation"
    )
    reference_precision = float(reference_metrics["token"]["action_precision"])
    wall_started = time.perf_counter()
    encoder = FeatureEncoder(char_hash_dim=0)
    max_bytes = int(architecture["max_utf8_bytes_per_line"])
    train_examples = make_neural_examples(train_documents, encoder, max_bytes=max_bytes)
    validation_examples = make_neural_examples(validation, encoder, max_bytes=max_bytes)
    model = _make_model(architecture, encoder.n_features)
    history, training_seconds = train_n1(
        model,
        train_examples,
        epochs=int(architecture["epochs"]),
        batch_size=int(architecture["batch_size"]),
        learning_rate=float(architecture["learning_rate"]),
        weight_decay=float(architecture["weight_decay"]),
        gradient_clip=float(architecture["gradient_clip"]),
        seed=seed,
    )
    emissions = _cache_emissions(
        model, validation_examples, batch_size=int(architecture["batch_size"])
    )
    calibration_receipt = calibrate_n1(
        validation,
        validation_examples,
        emissions,
        model,
        config["calibration"]["deletion_bias_grid"],
        reference_action_precision=reference_precision,
    )
    deletion_bias = float(calibration_receipt["selected"]["deletion_bias"])
    predictions = _predictions_from_emissions(
        validation,
        validation_examples,
        emissions,
        model,
        deletion_bias=deletion_bias,
    )
    checkpoint = {
        "schema_version": "n1-checkpoint-v2",
        "architecture_id": "n1-bytecnn-tcn-masked-crf",
        "architecture": dict(architecture),
        "engineered_dim": encoder.n_features,
        "feature_encoder": encoder.metadata(),
        "active_classes": ["BIB"],
        "deletion_bias": deletion_bias,
        "state_dict": model.state_dict(),
        "inputs": {
            "selection_silver_sha256": sha256_file(args.selection_silver),
            "selection_manifest_sha256": sha256_file(args.selection_manifest),
            "validation_silver_sha256": sha256_file(args.validation_silver),
            "selection_receipt_sha256": sha256_file(args.selection_receipt),
            "config_sha256": sha256_file(args.config),
            "reference_predictions_sha256": sha256_file(args.reference_predictions),
            "profile_receipt_sha256": sha256_file(args.profile_receipt),
        },
        "seed": seed,
        "production_eligible": False,
    }
    _atomic_torch_save(args.model_out, checkpoint)
    loaded_model, loaded_checkpoint = load_n1_checkpoint(args.model_out, config)
    if loaded_checkpoint["inputs"] != checkpoint["inputs"]:
        raise RuntimeError("reloaded N1 checkpoint input bindings differ")
    loaded_emissions = _cache_emissions(
        loaded_model,
        validation_examples,
        batch_size=int(architecture["batch_size"]),
    )
    loaded_predictions = _predictions_from_emissions(
        validation,
        validation_examples,
        loaded_emissions,
        loaded_model,
        deletion_bias=deletion_bias,
    )
    if loaded_predictions != predictions:
        raise RuntimeError("reloaded N1 checkpoint predictions differ before publication")
    write_n1_predictions(args.validation_predictions, validation, loaded_predictions)
    validation_metrics, _ = evaluate(
        validation, loaded_predictions, split="validation"
    )
    _scrub_silver_safety(validation_metrics)
    runtime["wall_seconds"] = time.perf_counter() - wall_started
    runtime["training_seconds"] = training_seconds
    runtime["peak_rss_bytes"] = _peak_rss_bytes()
    receipt = {
        "schema_version": "academic-structure-n1-training-v2",
        "status": "passed_cpu_fit_checkpoint_reload_and_validation_prediction",
        "architecture_id": "n1-bytecnn-tcn-masked-crf",
        "target": "BIB",
        "evidence_tier": "LLM_silver",
        "production_eligible": False,
        "execution": runtime,
        "effective_seed": seed,
        "inputs": checkpoint["inputs"],
        "source_rehydration_receipt_sha256": selection_receipt["source"][
            "rehydration_receipt_sha256"
        ],
        "counts": {
            "train_documents": len(train_documents),
            "validation_documents": len(validation),
            "train_sequences": len(train_examples),
            "validation_sequences": len(validation_examples),
        },
        "training_loss": history,
        "deletion_bias": deletion_bias,
        "calibration": calibration_receipt,
        "validation_metrics": validation_metrics,
        "profile": {
            "receipt_sha256": sha256_file(args.profile_receipt),
            "one_epoch_seconds": profile["one_epoch_seconds"],
            "projected_full_fit_seconds_with_15pct_margin": profile[
                "projected_full_fit_seconds_with_15pct_margin"
            ],
        },
        "outputs": {
            "model_sha256": sha256_file(args.model_out),
            "validation_predictions_sha256": sha256_file(args.validation_predictions),
        },
        "historically_named_test_partition": {
            "documents_loaded": 0,
            "predictions_written": 0,
            "semantics": "sealed_retrospective_comparison_not_unbiased_test",
        },
        "resource_gate_note": (
            "config deployment resource limits are promotion-only; this run records wall/RSS "
            "but makes no promotion claim"
        ),
    }
    _atomic_json(args.receipt_out, receipt)
    return receipt


def predict_cli(args: argparse.Namespace) -> dict[str, Any]:
    require_torch()
    documents, validation, selection_receipt = verify_selection_bundle(
        selection_silver_path=args.selection_silver,
        selection_manifest_path=args.selection_manifest,
        validation_silver_path=args.validation_silver,
        selection_receipt_path=args.selection_receipt,
        config_path=args.config,
    )
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    runtime = configure_runtime(
        config, uenv=args.uenv, effective_seed=int(config["execution"]["seed"])
    )
    selected = validation if args.split == "validation" else [
        document for document in documents if document.split == "train"
    ]
    if not selected or any(document.split == "test" for document in selected):
        raise RuntimeError("N1 prediction entry point rejects the historical test partition")
    training_receipt = json.loads(Path(args.training_receipt).read_text(encoding="utf-8"))
    expected_inputs = {
        "selection_silver_sha256": sha256_file(args.selection_silver),
        "selection_manifest_sha256": sha256_file(args.selection_manifest),
        "validation_silver_sha256": sha256_file(args.validation_silver),
        "selection_receipt_sha256": sha256_file(args.selection_receipt),
        "config_sha256": sha256_file(args.config),
    }
    if (
        training_receipt.get("schema_version") != "academic-structure-n1-training-v2"
        or training_receipt.get("status")
        != "passed_cpu_fit_checkpoint_reload_and_validation_prediction"
        or training_receipt.get("production_eligible") is not False
        or training_receipt.get("outputs", {}).get("model_sha256")
        != sha256_file(args.model)
        or any(
            training_receipt.get("inputs", {}).get(key) != value
            for key, value in expected_inputs.items()
        )
    ):
        raise RuntimeError("N1 model/training/selection receipt binding failed")
    model, checkpoint = load_n1_checkpoint(args.model, config)
    if any(checkpoint["inputs"].get(key) != value for key, value in expected_inputs.items()):
        raise RuntimeError("N1 checkpoint differs from current config/selection inputs")
    architecture = checkpoint["architecture"]
    encoder = FeatureEncoder(char_hash_dim=0)
    examples = make_neural_examples(
        selected, encoder, max_bytes=int(architecture["max_utf8_bytes_per_line"])
    )
    emissions = _cache_emissions(model, examples, batch_size=int(architecture["batch_size"]))
    predictions = _predictions_from_emissions(
        selected,
        examples,
        emissions,
        model,
        deletion_bias=float(checkpoint["deletion_bias"]),
    )
    write_n1_predictions(args.predictions, selected, predictions)
    runtime["peak_rss_bytes"] = _peak_rss_bytes()
    receipt = {
        "schema_version": "academic-structure-n1-prediction-v1",
        "status": "pass",
        "split": args.split,
        "document_count": len(selected),
        "predictions_sha256": sha256_file(args.predictions),
        "model_sha256": sha256_file(args.model),
        "selection_receipt_sha256": sha256_file(args.selection_receipt),
        "source_rehydration_receipt_sha256": selection_receipt["source"][
            "rehydration_receipt_sha256"
        ],
        "execution": runtime,
        "historically_named_test_documents_loaded": 0,
        "production_eligible": False,
    }
    _atomic_json(args.receipt_out, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument("--selection-silver", required=True)
    train.add_argument("--selection-manifest", required=True)
    train.add_argument("--validation-silver", required=True)
    train.add_argument("--selection-receipt", required=True)
    train.add_argument("--config", required=True)
    train.add_argument("--reference-predictions", required=True)
    train.add_argument("--profile-receipt", required=True)
    train.add_argument("--model-out", required=True)
    train.add_argument("--validation-predictions", required=True)
    train.add_argument("--receipt-out", required=True)
    train.add_argument("--uenv", required=True)
    train.add_argument("--code-commit", required=True)
    train.add_argument("--confirm-clariden-cpu-only", action="store_true")

    profile = sub.add_parser("profile")
    profile.add_argument("--selection-silver", required=True)
    profile.add_argument("--selection-manifest", required=True)
    profile.add_argument("--validation-silver", required=True)
    profile.add_argument("--selection-receipt", required=True)
    profile.add_argument("--config", required=True)
    profile.add_argument("--receipt-out", required=True)
    profile.add_argument("--uenv", required=True)
    profile.add_argument("--code-commit", required=True)
    profile.add_argument("--confirm-clariden-cpu-only", action="store_true")

    predict = sub.add_parser("predict")
    predict.add_argument("--model", required=True)
    predict.add_argument("--training-receipt", required=True)
    predict.add_argument("--selection-silver", required=True)
    predict.add_argument("--selection-manifest", required=True)
    predict.add_argument("--validation-silver", required=True)
    predict.add_argument("--selection-receipt", required=True)
    predict.add_argument("--config", required=True)
    predict.add_argument("--uenv", required=True)
    predict.add_argument("--split", choices=("train", "validation"), default="validation")
    predict.add_argument("--predictions", required=True)
    predict.add_argument("--receipt-out", required=True)

    args = parser.parse_args(argv)
    if args.command == "train":
        receipt = train_cli(args)
    elif args.command == "profile":
        receipt = profile_cli(args)
    else:
        receipt = predict_cli(args)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


__all__ = [
    "CharTCNCRF",
    "NeuralExample",
    "calibrate_n1",
    "count_neural_sequences",
    "encode_utf8_lines",
    "export_torchscript_emissions",
    "make_neural_examples",
    "load_n1_checkpoint",
    "validate_n1_profile_receipt",
    "require_torch",
    "train_n1",
    "write_n1_predictions",
]


if __name__ == "__main__":
    raise SystemExit(main())
