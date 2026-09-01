#!/usr/bin/env python3
"""Megatron Dataset adapter for receipt-bound D0-D4 packed schedules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from megatron.core.datasets.gpt_dataset import (
    _create_hash_table,
    _get_ltor_masks_and_position_ids,
    apply_goldfish,
)

from scheduled_sequence_reader import ScheduledSequenceReader


class ScheduledPackedGPTDataset(Dataset):
    def __init__(
        self,
        schedule_manifest: Path,
        arm_id: str,
        *,
        eod_token_id: int,
        reset_position_ids: bool,
        reset_attention_mask: bool,
        eod_mask_loss: bool,
        create_attention_mask: bool,
        goldfish_loss: bool,
        goldfish_k: int,
        goldfish_h: int,
        max_samples: int | None = None,
    ) -> None:
        self.reader = ScheduledSequenceReader(schedule_manifest, arm_id)
        self.eod_token_id = int(eod_token_id)
        self.reset_position_ids = bool(reset_position_ids)
        self.reset_attention_mask = bool(reset_attention_mask)
        self.eod_mask_loss = bool(eod_mask_loss)
        self.create_attention_mask = bool(create_attention_mask)
        self.goldfish_loss = bool(goldfish_loss)
        self.goldfish_k = int(goldfish_k)
        self.goldfish_h = int(goldfish_h)
        if max_samples is not None and not 0 < int(max_samples) <= len(self.reader):
            raise ValueError(f"invalid scheduled-dataset prefix: {max_samples}/{len(self.reader)}")
        self.max_samples = int(max_samples) if max_samples is not None else len(self.reader)
        self._goldfish_hash_table: torch.Tensor | None = None
        self._goldfish_token_id = -1

    def __len__(self) -> int:
        return self.max_samples

    def __getitem__(self, idx: int | None) -> dict[str, Any]:
        padding_sample = idx is None
        payload = self.reader[0 if padding_sample else int(idx)]
        text = torch.from_numpy(payload.tokens).long()
        tokens = text[:-1].contiguous()
        labels = text[1:].contiguous()
        attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
            tokens,
            self.eod_token_id,
            self.reset_position_ids,
            self.reset_attention_mask,
            self.eod_mask_loss,
            self.create_attention_mask,
        )
        if payload.active_tokens < labels.numel():
            loss_mask[payload.active_tokens :] = 0.0
        pad_token_id = self.reader.pad_token_id
        loss_mask[labels == pad_token_id] = 0.0
        tokens[tokens == pad_token_id] = 0
        labels[labels == pad_token_id] = 0

        if self.goldfish_loss:
            if self._goldfish_hash_table is None:
                self._goldfish_hash_table = _create_hash_table(device=labels.device)
            goldfish_labels = apply_goldfish(
                labels,
                goldfish_token_id=self._goldfish_token_id,
                k=self.goldfish_k,
                goldfish_hash_table=self._goldfish_hash_table,
                goldfish_context_width=self.goldfish_h,
            )
            loss_mask[goldfish_labels == self._goldfish_token_id] = 0.0
        if padding_sample or payload.filler:
            loss_mask.zero_()

        result: dict[str, Any] = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }
        if self.create_attention_mask:
            result["attention_mask"] = attention_mask
        return result
