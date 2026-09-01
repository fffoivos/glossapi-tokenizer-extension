#!/usr/bin/env python3
"""Deterministic 2253/3218 replay split for the 13.5B LR-floor study."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


UINT64_RANGE = 1 << 64
SPLIT_NUMERATOR = 2253
SPLIT_DENOMINATOR = 3218


@dataclass(frozen=True)
class Assignment:
    phase: int
    logical_pool: str
    score_u64: int


def partition_score(seed: int, logical_pool: str, document_id: str) -> int:
    if seed < 0 or logical_pool not in {"foreign_replay", "old_greek_replay"} or not document_id:
        raise ValueError("invalid replay phase-partition input")
    payload = f"greek-cpt-lr13-replay-partition-v1\0{seed}\0{logical_pool}\0{document_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def split_phase(score_u64: int) -> int:
    if score_u64 < 0 or score_u64 >= UINT64_RANGE:
        raise ValueError("score must be an unsigned 64-bit integer")
    return 1 if score_u64 * SPLIT_DENOMINATOR < SPLIT_NUMERATOR * UINT64_RANGE else 2


def assign_replay(*, seed: int, logical_pool: str, document_id: str) -> Assignment:
    score = partition_score(seed, logical_pool, document_id)
    return Assignment(phase=split_phase(score), logical_pool=logical_pool, score_u64=score)
