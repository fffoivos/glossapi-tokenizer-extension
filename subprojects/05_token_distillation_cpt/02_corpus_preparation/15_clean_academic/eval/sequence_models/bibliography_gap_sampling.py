"""Deterministic work-grouped subsets for gap-connection learning curves."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from .bibliography_gap_candidates import REGIME_RANK
from .bibliography_gap_connect_table import gap_length_bucket


SIZE_LADDER = (250, 500, 1000, 2000)


def _rank(seed: int, *values: object) -> str:
    return hashlib.sha256(
        f"{seed}:".encode("utf-8") + ":".join(map(str, values)).encode("utf-8")
    ).hexdigest()


def _eligible(row: Mapping[str, Any], regime: str) -> bool:
    return REGIME_RANK[str(row["regime"])] <= REGIME_RANK[regime]


def _representative_rows(
    metadata: Sequence[Mapping[str, Any]], targets: np.ndarray, regime: str,
) -> dict[str, int]:
    representatives = {}
    for index, row in enumerate(metadata):
        if not _eligible(row, regime):
            continue
        group = str(row["boundary_group_id"])
        if group in representatives:
            previous = representatives[group]
            if int(targets[previous]) != int(targets[index]):
                raise ValueError("one boundary group has contradictory targets")
            continue
        representatives[group] = index
    return representatives


def available_negative_group_count(
    metadata: Sequence[Mapping[str, Any]], targets: np.ndarray, regime: str,
) -> int:
    representatives = _representative_rows(metadata, targets, regime)
    return sum(int(targets[index]) == 0 for index in representatives.values())


def size_rungs(available_negatives: int) -> tuple[int | None, ...]:
    rungs: list[int | None] = [value for value in SIZE_LADDER if value < available_negatives]
    rungs.append(None)
    return tuple(rungs)


def _stratum(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return (
        str(row["source"]),
        int(row["fold"]),
        gap_length_bucket(int(row["model_line_count"])),
    )


def select_training_rows(
    metadata: Sequence[Mapping[str, Any]], targets: np.ndarray, *,
    regime: str, negative_group_limit: int | None, seed: int,
    positive_per_negative: int = 2, maximum_positive_per_gold_block: int = 4,
) -> np.ndarray:
    """Select nested negative groups and length/source-matched positives."""

    if len(metadata) != len(targets):
        raise ValueError("metadata and targets are not aligned")
    representatives = _representative_rows(metadata, targets, regime)
    negative_groups = sorted(
        (group for group, index in representatives.items() if int(targets[index]) == 0),
        key=lambda group: _rank(seed, "negative", group),
    )
    if negative_group_limit is not None:
        negative_groups = negative_groups[:negative_group_limit]
    if not negative_groups:
        raise ValueError(f"{regime} has no selected negative boundary groups")

    quota = Counter()
    for group in negative_groups:
        quota[_stratum(metadata[representatives[group]])] += positive_per_negative
    positive_groups: defaultdict[tuple[str, int, str], list[str]] = defaultdict(list)
    for group, index in representatives.items():
        if int(targets[index]) == 1:
            positive_groups[_stratum(metadata[index])].append(group)
    selected_positive: list[str] = []
    gold_counts = Counter()
    selected_by_stratum = Counter()
    for stratum, count in sorted(quota.items()):
        candidates = sorted(
            positive_groups.get(stratum, ()),
            key=lambda group: _rank(seed, "positive", group),
        )
        for group in candidates:
            index = representatives[group]
            gold = str(metadata[index].get("gold_block_group_id") or group)
            if gold_counts[gold] >= maximum_positive_per_gold_block:
                continue
            selected_positive.append(group)
            gold_counts[gold] += 1
            selected_by_stratum[stratum] += 1
            if selected_by_stratum[stratum] >= count:
                break

    desired = positive_per_negative * len(negative_groups)
    if len(selected_positive) < desired:
        selected_set = set(selected_positive)
        fallback = sorted(
            (
                group for group, index in representatives.items()
                if int(targets[index]) == 1 and group not in selected_set
            ),
            key=lambda group: _rank(seed, "positive-fallback", group),
        )
        for group in fallback:
            index = representatives[group]
            gold = str(metadata[index].get("gold_block_group_id") or group)
            if gold_counts[gold] >= maximum_positive_per_gold_block:
                continue
            selected_positive.append(group)
            gold_counts[gold] += 1
            if len(selected_positive) >= desired:
                break

    selected_groups = set((*negative_groups, *selected_positive))
    rows = np.asarray([
        index for index, row in enumerate(metadata)
        if _eligible(row, regime) and str(row["boundary_group_id"]) in selected_groups
    ], dtype=np.int64)
    if {int(value) for value in np.unique(targets[rows])} != {0, 1}:
        raise ValueError("selected learning-curve rows lack a class")
    return rows


def fit_weights(
    metadata: Sequence[Mapping[str, Any]], targets: np.ndarray, rows: np.ndarray,
    *, maximum_synthetic_fraction: float = 0.5,
) -> np.ndarray:
    """Work-balance a subset, then class-balance without synthetic domination."""

    base = np.asarray([
        float(metadata[int(index)].get("base_weight", 1.0)) for index in rows
    ], dtype=np.float64)
    works = np.asarray([str(metadata[int(index)]["work_id"]) for index in rows])
    for work in np.unique(works):
        local = works == work
        base[local] /= base[local].sum()
    local_targets = targets[rows]
    for target in (0, 1):
        local = local_targets == target
        if not np.any(local):
            raise ValueError("fit weights require both classes")
        base[local] *= 0.5 / base[local].sum()
    synthetic = np.asarray([
        str(metadata[int(index)]["regime"]) != "deployment_real" for index in rows
    ])
    natural_weight = float(base[~synthetic].sum())
    synthetic_weight = float(base[synthetic].sum())
    if natural_weight <= 0:
        raise ValueError("a training subset cannot be entirely synthetic")
    maximum_synthetic = maximum_synthetic_fraction * (natural_weight + synthetic_weight)
    if synthetic_weight > maximum_synthetic and synthetic_weight > 0:
        desired = maximum_synthetic_fraction * natural_weight / max(
            1.0e-12, 1.0 - maximum_synthetic_fraction
        )
        base[synthetic] *= desired / synthetic_weight
    base /= base.sum()
    if not np.isfinite(base).all() or np.any(base <= 0):
        raise RuntimeError("fit weights are invalid")
    return base
