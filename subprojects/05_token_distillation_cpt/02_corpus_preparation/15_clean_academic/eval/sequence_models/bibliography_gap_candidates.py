"""Pure candidate construction for bibliography gap-connection research.

The deployed line threshold creates seed components.  A gap candidate is the
strictly interior sequence between two adjacent components; component lines
are metadata only and never enter a gap tensor.  Threshold and synthetic
variants share a boundary-group identifier so correlated views cannot be
mistaken for independent examples.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import hashlib
from typing import Iterable, Sequence

import numpy as np

from .bibliography_entry_dataset import MAX_PHYSICAL_GAP


DEPLOYMENT_THRESHOLD = 0.25
THRESHOLD_LADDER = (0.10, 0.15, 0.25, 0.40, 0.60, 0.80)
REGIME_ORDER = (
    "deployment_real",
    "threshold_ladder",
    "header_ablation",
    "hard_nonbib",
    "easy_nonbib",
)
REGIME_RANK = {name: index for index, name in enumerate(REGIME_ORDER)}


@dataclass(frozen=True)
class CandidateContext:
    document_index: int
    document_id: str
    work_id: str
    source: str
    fold: int


@dataclass(frozen=True)
class GapCandidate:
    context: CandidateContext
    left: int
    right: int
    target: int
    regime: str
    generation_thresholds: tuple[float, ...]
    synthetic_kind: str = "none"
    removed_offsets: tuple[int, ...] = ()
    mask_heading_probability: bool = False
    base_weight: float = 1.0
    virtual_boundaries: bool = False
    entry_mean: float = 0.0
    entry_max: float = 0.0

    @property
    def gap_length(self) -> int:
        return self.right - self.left - 1

    @property
    def sequence_length(self) -> int:
        return self.gap_length - len(self.removed_offsets)

    @property
    def boundary_group_id(self) -> str:
        return f"{self.context.document_id}:{self.left}:{self.right}"

    @property
    def variant_id(self) -> str:
        return f"{self.boundary_group_id}:{self.synthetic_kind}"


def stable_integer(*values: object) -> int:
    digest = hashlib.sha256(":".join(map(str, values)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def seed_components(
    probability: np.ndarray,
    char_lengths: np.ndarray,
    abs_indices: np.ndarray,
    forbidden: np.ndarray,
    *,
    threshold: float,
    seed_length_limit: int,
) -> list[tuple[int, int]]:
    """Return maximal exactly-adjacent runs of accepted seed lines."""

    if not (
        probability.shape == char_lengths.shape == abs_indices.shape == forbidden.shape
        and probability.ndim == 1
    ):
        raise ValueError("seed component arrays are not aligned")
    seeds = np.flatnonzero(
        (probability >= threshold)
        & (char_lengths <= seed_length_limit)
        & ~forbidden.astype(bool)
    )
    if not len(seeds):
        return []
    result: list[tuple[int, int]] = []
    start = previous = int(seeds[0])
    for value in seeds[1:]:
        index = int(value)
        physical_break = int(abs_indices[index]) - int(abs_indices[previous]) > MAX_PHYSICAL_GAP
        if index != previous + 1 or physical_break:
            result.append((start, previous))
            start = index
        previous = index
    result.append((start, previous))
    return result


def enumerate_component_gaps(
    *,
    context: CandidateContext,
    entry_probability: np.ndarray,
    char_lengths: np.ndarray,
    abs_indices: np.ndarray,
    gold_bib: np.ndarray,
    typed_heading_barrier: np.ndarray,
    exact_scope: np.ndarray,
    thresholds: Sequence[float] = THRESHOLD_LADDER,
    deployment_threshold: float = DEPLOYMENT_THRESHOLD,
    seed_length_limit: int = 330,
    max_gap_lines: int = 384,
) -> tuple[list[GapCandidate], dict[str, int]]:
    """Enumerate real and header-miss candidates from frozen line outputs."""

    arrays = (
        entry_probability,
        char_lengths,
        abs_indices,
        gold_bib,
        typed_heading_barrier,
        exact_scope,
    )
    if any(value.ndim != 1 or value.shape != entry_probability.shape for value in arrays):
        raise ValueError("component-gap arrays are not aligned")
    original: dict[tuple[int, int], GapCandidate] = {}
    header_variants: dict[tuple[int, int, str], GapCandidate] = {}
    counts: defaultdict[str, int] = defaultdict(int)
    forbidden_seed = typed_heading_barrier.astype(bool) | exact_scope.astype(bool)

    for threshold in thresholds:
        components = seed_components(
            entry_probability,
            char_lengths,
            abs_indices,
            forbidden_seed,
            threshold=float(threshold),
            seed_length_limit=seed_length_limit,
        )
        counts[f"threshold:{threshold:.2f}:components"] += len(components)
        for left_component, right_component in zip(components[:-1], components[1:]):
            left, right = int(left_component[1]), int(right_component[0])
            length = right - left - 1
            counts[f"threshold:{threshold:.2f}:adjacent_component_gaps"] += 1
            if length <= 0:
                counts["zero_length"] += 1
                continue
            if length > max_gap_lines:
                counts["over_length"] += 1
                continue
            physical = bool(np.any(
                np.diff(abs_indices[left : right + 1].astype(np.int64)) > MAX_PHYSICAL_GAP
            ))
            interior = slice(left + 1, right)
            typed = bool(np.any(typed_heading_barrier[interior]))
            scoped = bool(np.any(exact_scope[interior]))
            target = int(np.all(gold_bib[left : right + 1]))
            key = (left, right)
            if physical or scoped:
                counts["hard_barrier"] += 1
                continue
            if typed:
                counts["typed_heading_barrier"] += 1
                if target:
                    counts["typed_heading_positive_excluded"] += 1
                    continue
                removed = tuple(
                    int(index) for index in np.flatnonzero(typed_heading_barrier[interior]) + left + 1
                )
                for kind, mask, removed_offsets in (
                    ("heading_probability_masked", True, ()),
                    ("heading_line_removed", False, removed),
                ):
                    if length - len(removed_offsets) <= 0:
                        continue
                    variant_key = (left, right, kind)
                    existing = header_variants.get(variant_key)
                    thresholds_seen = (float(threshold),) if existing is None else tuple(sorted(set(
                        (*existing.generation_thresholds, float(threshold))
                    )))
                    header_variants[variant_key] = GapCandidate(
                        context=context,
                        left=left,
                        right=right,
                        target=0,
                        regime="header_ablation",
                        generation_thresholds=thresholds_seen,
                        synthetic_kind=kind,
                        removed_offsets=removed_offsets,
                        mask_heading_probability=mask,
                        base_weight=0.25,
                        entry_mean=float(entry_probability[interior].mean()),
                        entry_max=float(entry_probability[interior].max(initial=0.0)),
                    )
                continue

            existing = original.get(key)
            thresholds_seen = (float(threshold),) if existing is None else tuple(sorted(set(
                (*existing.generation_thresholds, float(threshold))
            )))
            deployment = any(np.isclose(value, deployment_threshold) for value in thresholds_seen)
            original[key] = GapCandidate(
                context=context,
                left=left,
                right=right,
                target=target,
                regime="deployment_real" if deployment else "threshold_ladder",
                generation_thresholds=thresholds_seen,
                base_weight=1.0 if deployment else 0.5,
                entry_mean=float(entry_probability[interior].mean()),
                entry_max=float(entry_probability[interior].max(initial=0.0)),
            )

    candidates = [*original.values(), *header_variants.values()]
    for candidate in candidates:
        counts[f"candidate:{candidate.regime}:{candidate.target}"] += 1
    return sorted(candidates, key=lambda row: (
        row.context.document_index, row.left, row.right, REGIME_RANK[row.regime], row.synthetic_kind,
    )), dict(sorted(counts.items()))


def _physical_runs(mask: np.ndarray, abs_indices: np.ndarray) -> Iterable[tuple[int, int]]:
    start: int | None = None
    for index, enabled in enumerate(mask.astype(bool)):
        physical_break = index > 0 and int(abs_indices[index]) - int(abs_indices[index - 1]) > MAX_PHYSICAL_GAP
        if enabled and (start is None or physical_break):
            if start is not None:
                yield start, index
            start = index
        elif not enabled and start is not None:
            yield start, index
            start = None
    if start is not None:
        yield start, len(mask)


def sample_nonbib_spans(
    *,
    context: CandidateContext,
    entry_probability: np.ndarray,
    abs_indices: np.ndarray,
    gold_bib: np.ndarray,
    typed_heading_barrier: np.ndarray,
    exact_scope: np.ndarray,
    target_lengths: Sequence[int],
) -> list[GapCandidate]:
    """Return at most one hard and one easy non-BIB span for one document."""

    if not target_lengths or len(entry_probability) < 3:
        return []
    allowed = ~(gold_bib.astype(bool) | typed_heading_barrier.astype(bool) | exact_scope.astype(bool))
    allowed[0] = False
    allowed[-1] = False
    runs = list(_physical_runs(allowed, abs_indices))
    result: list[GapCandidate] = []
    chosen_boundaries: set[tuple[int, int]] = set()
    for kind, largest in (("hard_nonbib", True), ("easy_nonbib", False)):
        length = int(target_lengths[
            stable_integer(context.work_id, context.document_id, kind) % len(target_lengths)
        ])
        length = max(1, min(length, 384))
        proposals: list[tuple[float, int, int, float]] = []
        for run_start, run_end in runs:
            run_start = max(1, run_start)
            run_end = min(len(entry_probability) - 1, run_end)
            if run_end - run_start < length:
                continue
            prefix = np.concatenate(([0.0], np.cumsum(entry_probability[run_start:run_end], dtype=np.float64)))
            for offset in range(0, run_end - run_start - length + 1):
                start, end = run_start + offset, run_start + offset + length
                mean = float((prefix[offset + length] - prefix[offset]) / length)
                maximum = float(entry_probability[start:end].max(initial=0.0))
                score = mean + maximum
                proposals.append((score, start, end, maximum))
        if not proposals:
            continue
        proposals.sort(key=lambda row: (
            row[0],
            stable_integer(context.document_id, kind, row[1], row[2]),
        ), reverse=largest)
        score, start, end, maximum = proposals[0]
        if (start, end) in chosen_boundaries:
            continue
        chosen_boundaries.add((start, end))
        mean = float(entry_probability[start:end].mean())
        result.append(GapCandidate(
            context=context,
            left=start - 1,
            right=end,
            target=0,
            regime=kind,
            generation_thresholds=(),
            synthetic_kind=f"{kind}_span",
            base_weight=0.25,
            virtual_boundaries=True,
            entry_mean=mean,
            entry_max=maximum,
        ))
    return result


def cap_nonbib_by_work(
    candidates: Sequence[GapCandidate], *, maximum_per_kind: int = 1,
) -> list[GapCandidate]:
    """Cap correlated random spans without depending on input ordering."""

    grouped: defaultdict[tuple[str, str], list[GapCandidate]] = defaultdict(list)
    passthrough = []
    for candidate in candidates:
        if candidate.regime not in {"hard_nonbib", "easy_nonbib"}:
            passthrough.append(candidate)
            continue
        grouped[(candidate.context.work_id, candidate.regime)].append(candidate)
    selected = list(passthrough)
    for rows in grouped.values():
        selected.extend(sorted(
            rows,
            key=lambda row: stable_integer(row.variant_id, row.context.document_index),
        )[:maximum_per_kind])
    return sorted(selected, key=lambda row: (
        row.context.document_index, row.left, row.right, REGIME_RANK[row.regime], row.synthetic_kind,
    ))


def normalize_boundary_weights(candidates: Sequence[GapCandidate]) -> list[GapCandidate]:
    """Keep each correlated boundary family within its declared total weight."""

    grouped: defaultdict[str, list[GapCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.boundary_group_id].append(candidate)
    result = []
    for rows in grouped.values():
        total = sum(row.base_weight for row in rows)
        scale = min(1.0, 1.0 / total) if total else 0.0
        result.extend(replace(row, base_weight=row.base_weight * scale) for row in rows)
    return sorted(result, key=lambda row: (
        row.context.document_index, row.left, row.right, REGIME_RANK[row.regime], row.synthetic_kind,
    ))
