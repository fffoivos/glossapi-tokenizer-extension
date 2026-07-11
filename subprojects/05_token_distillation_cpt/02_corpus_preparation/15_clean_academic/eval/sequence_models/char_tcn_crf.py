#!/usr/bin/env python3
"""N1 byte-CNN + dilated line-TCN + masked BIOES CRF scaffold.

The byte encoder avoids a Greek/Latin/OCR vocabulary and performs no Unicode
normalization.  The TCN contextualizes complete line sequences; the CRF uses the
same legal BIOES transition mask as the feature baselines.  This module is an
offline CPU shadow candidate, never a replacement for the Rust hot path without
the separate promotion and runtime receipts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # Optional locally; available in the pinned Clariden PyTorch uenv.
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised by dependency-free smoke tests
    torch = None
    nn = None

from .features import TAGS, allowed_transition_mask


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
            for convolution in self.convolutions:
                values = torch.relu(convolution(embedded))
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

        def __init__(self, n_tags: int):
            super().__init__()
            transition_mask, start_mask, end_mask = allowed_transition_mask()
            self.transitions = nn.Parameter(torch.zeros(n_tags, n_tags))
            self.start = nn.Parameter(torch.zeros(n_tags))
            self.end = nn.Parameter(torch.zeros(n_tags))
            self.register_buffer("transition_mask", torch.as_tensor(transition_mask))
            self.register_buffer("start_mask", torch.as_tensor(start_mask))
            self.register_buffer("end_mask", torch.as_tensor(end_mask))

        @staticmethod
        def _masked(values: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
            return values.masked_fill(~mask, -1.0e4)

        def log_partition(self, emissions: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
            transitions = self._masked(self.transitions, self.transition_mask)
            score = self._masked(self.start, self.start_mask) + emissions[:, 0]
            for t in range(1, emissions.shape[1]):
                candidate = score.unsqueeze(2) + transitions.unsqueeze(0) + emissions[:, t].unsqueeze(1)
                next_score = torch.logsumexp(candidate, dim=1)
                score = torch.where(mask[:, t].unsqueeze(1), next_score, score)
            return torch.logsumexp(score + self._masked(self.end, self.end_mask), dim=1)

        def gold_score(
            self, emissions: "torch.Tensor", tags: "torch.Tensor", mask: "torch.Tensor"
        ) -> "torch.Tensor":
            transitions = self._masked(self.transitions, self.transition_mask)
            batch = torch.arange(emissions.shape[0], device=emissions.device)
            score = self._masked(self.start, self.start_mask)[tags[:, 0]]
            score = score + emissions[batch, 0, tags[:, 0]]
            for t in range(1, emissions.shape[1]):
                step = transitions[tags[:, t - 1], tags[:, t]] + emissions[batch, t, tags[:, t]]
                score = score + step * mask[:, t]
            lengths = mask.long().sum(dim=1) - 1
            last_tags = tags.gather(1, lengths.unsqueeze(1)).squeeze(1)
            return score + self._masked(self.end, self.end_mask)[last_tags]

        def neg_log_likelihood(
            self, emissions: "torch.Tensor", tags: "torch.Tensor", mask: "torch.Tensor"
        ) -> "torch.Tensor":
            if not bool(mask[:, 0].all()):
                raise ValueError("every sequence must contain at least one line")
            return (self.log_partition(emissions, mask) - self.gold_score(emissions, tags, mask)).mean()

        def decode(self, emissions: "torch.Tensor", mask: "torch.Tensor") -> list[list[int]]:
            transitions = self._masked(self.transitions, self.transition_mask)
            score = self._masked(self.start, self.start_mask) + emissions[:, 0]
            histories: list[torch.Tensor] = []
            for t in range(1, emissions.shape[1]):
                candidate = score.unsqueeze(2) + transitions.unsqueeze(0)
                best_score, best_tag = candidate.max(dim=1)
                next_score = best_score + emissions[:, t]
                score = torch.where(mask[:, t].unsqueeze(1), next_score, score)
                histories.append(best_tag)
            score = score + self._masked(self.end, self.end_mask)
            last = score.argmax(dim=1)
            decoded: list[list[int]] = []
            for b in range(emissions.shape[0]):
                length = int(mask[b].sum().item())
                path = [int(last[b].item())]
                for t in range(length - 2, -1, -1):
                    path.append(int(histories[t][b, path[-1]].item()))
                decoded.append(list(reversed(path)))
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
        ):
            super().__init__()
            self.byte_cnn = ByteCNN(byte_embedding_dim, char_channels_per_kernel, char_kernels)
            self.input_projection = nn.Linear(self.byte_cnn.output_dim + engineered_dim, hidden_dim)
            self.blocks = nn.ModuleList(
                ResidualTCNBlock(hidden_dim, dilation, dropout) for dilation in tcn_dilations
            )
            self.output_norm = nn.LayerNorm(hidden_dim)
            self.emissions = nn.Linear(hidden_dim, len(TAGS))
            self.crf = MaskedBIOESCRF(len(TAGS))

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
        ) -> list[list[int]]:
            return self.crf.decode(self(byte_ids, engineered, line_mask), line_mask)

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


__all__ = [
    "CharTCNCRF", "encode_utf8_lines", "export_torchscript_emissions", "require_torch"
]
