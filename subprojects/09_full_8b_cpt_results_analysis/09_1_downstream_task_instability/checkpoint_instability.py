#!/usr/bin/env python3
"""Metrics from Nishida et al. (2025), arXiv:2510.04848, equations 1-2."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from numbers import Real
from typing import Any, TypeVar


OutputT = TypeVar("OutputT")


def _validated_real(value: Real, *, name: str) -> float:
    if isinstance(value, bool):
        result = float(value)
    elif isinstance(value, Real):
        result = float(value)
    else:
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    return result


def _require_trajectory(values: Sequence[Any], *, name: str) -> None:
    if len(values) < 2:
        raise ValueError(f"{name} requires at least two checkpoints, got {len(values)}")


def mean_total_variation(scores: Sequence[Real]) -> float:
    """Return equation (1), the mean absolute change between adjacent scores."""

    _require_trajectory(scores, name="mean total variation")
    numeric = [
        _validated_real(score, name=f"scores[{index}]")
        for index, score in enumerate(scores)
    ]
    return math.fsum(
        abs(current - previous)
        for previous, current in zip(numeric, numeric[1:], strict=False)
    ) / (len(numeric) - 1)


def exact_match_similarity(left: OutputT, right: OutputT) -> float:
    """Return one for equal outputs and zero otherwise."""

    return float(left == right)


def instability_score(
    outputs: Sequence[OutputT],
    similarity: Callable[[OutputT, OutputT], Real] = exact_match_similarity,
) -> float:
    """Return equation (2), mean dissimilarity between adjacent outputs."""

    _require_trajectory(outputs, name="instability score")
    dissimilarities: list[float] = []
    for index, (previous, current) in enumerate(
        zip(outputs, outputs[1:], strict=False)
    ):
        similarity_value = _validated_real(
            similarity(previous, current), name=f"similarity[{index}]"
        )
        if not 0.0 <= similarity_value <= 1.0:
            raise ValueError(
                f"similarity[{index}] must be in [0, 1], got {similarity_value}"
            )
        dissimilarities.append(1.0 - similarity_value)
    return math.fsum(dissimilarities) / len(dissimilarities)


def analyze_example_trajectories(
    scores_by_example: Mapping[str, Sequence[Real]],
    outputs_by_example: Mapping[str, Sequence[OutputT]],
    similarity: Callable[[OutputT, OutputT], Real] = exact_match_similarity,
) -> dict[str, Any]:
    """Compute equations (1)-(2) per aligned example and macro-average them."""

    score_ids = set(scores_by_example)
    output_ids = set(outputs_by_example)
    if score_ids != output_ids:
        raise ValueError(
            "score/output example IDs differ: "
            f"scores_only={sorted(score_ids-output_ids)[:5]} "
            f"outputs_only={sorted(output_ids-score_ids)[:5]}"
        )
    if not score_ids:
        raise ValueError("at least one example is required")

    ordered_ids = sorted(score_ids)
    checkpoint_count: int | None = None
    per_example: dict[str, dict[str, float]] = {}
    for example_id in ordered_ids:
        scores = scores_by_example[example_id]
        outputs = outputs_by_example[example_id]
        if len(scores) != len(outputs):
            raise ValueError(
                f"{example_id}: score/output checkpoint count differs: "
                f"{len(scores)} != {len(outputs)}"
            )
        if checkpoint_count is None:
            checkpoint_count = len(scores)
        elif len(scores) != checkpoint_count:
            raise ValueError(
                f"{example_id}: checkpoint count drift: "
                f"{len(scores)} != {checkpoint_count}"
            )
        per_example[example_id] = {
            "mean_total_variation": mean_total_variation(scores),
            "instability_score": instability_score(outputs, similarity),
        }

    assert checkpoint_count is not None
    return {
        "example_count": len(ordered_ids),
        "checkpoint_count": checkpoint_count,
        "mean_example_mtv": math.fsum(
            row["mean_total_variation"] for row in per_example.values()
        )
        / len(per_example),
        "mean_example_is": math.fsum(
            row["instability_score"] for row in per_example.values()
        )
        / len(per_example),
        "per_example": per_example,
    }
