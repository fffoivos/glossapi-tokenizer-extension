#!/usr/bin/env python3
"""Deterministic, document-disjoint phase assignment for the 25B CPT run."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


UINT64_RANGE = 1 << 64
HPLT_SOURCE = re.compile(r"^HPLT/")
SPLIT_NUMERATOR = 3
SPLIT_DENOMINATOR = 5


@dataclass(frozen=True)
class Assignment:
    phase: int
    logical_pool: str
    score_u64: int | None


def composite_document_id(source_dataset: object, source_doc_id: object) -> str:
    """Create the only accepted identity for rows in the cleaned Greek v2 set."""

    source = str(source_dataset or "")
    upstream_id = str(source_doc_id or "")
    if not source or not upstream_id:
        raise ValueError("source_dataset and source_doc_id must both be non-empty")
    payload = f"greek-cpt-document-v1\0{source}\0{upstream_id}".encode("utf-8")
    return "gdocv1:" + hashlib.sha256(payload).hexdigest()


def partition_score(seed: int, logical_pool: str, document_id: str) -> int:
    if seed < 0 or not logical_pool or not document_id:
        raise ValueError("invalid phase-partition input")
    payload = (
        f"greek-cpt-phase-partition-v1\0{seed}\0{logical_pool}\0{document_id}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def split_phase(score_u64: int) -> int:
    """Return phase 1 for the first 3/5 of the hash space, else phase 2."""

    if score_u64 < 0 or score_u64 >= UINT64_RANGE:
        raise ValueError("score must be an unsigned 64-bit integer")
    return (
        1
        if score_u64 * SPLIT_DENOMINATOR
        < SPLIT_NUMERATOR * UINT64_RANGE
        else 2
    )


def classify_new_greek(source_dataset: object) -> str:
    source = str(source_dataset or "")
    if not source:
        raise ValueError("source_dataset must be non-empty")
    return "hplt_new_greek" if HPLT_SOURCE.search(source) else "non_hplt_new_greek"


def assign_new_greek(
    *, seed: int, source_dataset: object, source_doc_id: object
) -> Assignment:
    logical_pool = classify_new_greek(source_dataset)
    document_id = composite_document_id(source_dataset, source_doc_id)
    if logical_pool == "non_hplt_new_greek":
        return Assignment(phase=2, logical_pool=logical_pool, score_u64=None)
    score = partition_score(seed, logical_pool, document_id)
    return Assignment(
        phase=split_phase(score), logical_pool=logical_pool, score_u64=score
    )


def assign_replay(*, seed: int, logical_pool: str, document_id: str) -> Assignment:
    if logical_pool not in {"foreign_replay", "old_greek_replay"}:
        raise ValueError(f"unsupported replay pool: {logical_pool}")
    score = partition_score(seed, logical_pool, document_id)
    return Assignment(
        phase=split_phase(score), logical_pool=logical_pool, score_u64=score
    )
