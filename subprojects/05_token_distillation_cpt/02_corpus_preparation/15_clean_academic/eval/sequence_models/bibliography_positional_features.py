#!/usr/bin/env python3
"""Sparse intra-line bibliography match and non-match position encodings.

All offsets refer to the NFKC-normalized string returned by the existing
ownership-resolved feature extractor.  The sparse event representation is the
lossless table artifact; coarse linear maps and 64-bin neural tensors are
derived from it per arm or per batch.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .bibliography_feature_explorer import FEATURE_SPECS
from .bibliography_v2 import extract_bibliography_feature_review


SCHEMA_VERSION = "bibliography-positional-line-encoding-v1"
FEATURE_NAMES = tuple(spec.key for spec in FEATURE_SPECS)
FEATURE_TO_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
SEMANTIC_UNION_EXCLUDED = frozenset(
    {"punctuation_count", "prose_lead_count", "table_row_count"}
)
NONMATCH_CATEGORIES = (
    "letters",
    "digits",
    "whitespace",
    "punctuation_symbols",
    "other_ocr",
)
NONMATCH_TO_INDEX = {name: index for index, name in enumerate(NONMATCH_CATEGORIES)}
POSITION_SUMMARY_NAMES = ("first_start", "last_end", "mean_center", "union_coverage")
GAP_SUMMARY_NAMES = (
    "unmatched_fraction",
    "unmatched_prefix_fraction",
    "unmatched_suffix_fraction",
    "longest_unmatched_fraction",
    "longest_unmatched_center",
    "unmatched_run_count",
    "mean_unmatched_run_fraction",
)


@dataclass(frozen=True)
class PositionalLineEncoding:
    normalized_text: str
    counts: np.ndarray
    match_feature: np.ndarray
    match_start: np.ndarray
    match_end: np.ndarray
    nonmatch_category: np.ndarray
    nonmatch_start: np.ndarray
    nonmatch_end: np.ndarray
    position_summaries: np.ndarray
    gap_summaries: np.ndarray

    @property
    def nfkc_length(self) -> int:
        return len(self.normalized_text)


def _merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted((int(start), int(end)) for start, end in intervals if start < end)
    merged: list[list[int]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _complement(intervals: Sequence[tuple[int, int]], length: int) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    cursor = 0
    for start, end in _merge_intervals(intervals):
        if start < 0 or end > length:
            raise ValueError("feature span lies outside normalized text")
        if cursor < start:
            result.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < length:
        result.append((cursor, length))
    return result


def _character_category(character: str) -> int:
    if character.isspace():
        return NONMATCH_TO_INDEX["whitespace"]
    category = unicodedata.category(character)
    if category.startswith("L"):
        return NONMATCH_TO_INDEX["letters"]
    if category.startswith("N"):
        return NONMATCH_TO_INDEX["digits"]
    if category.startswith(("P", "S")):
        return NONMATCH_TO_INDEX["punctuation_symbols"]
    return NONMATCH_TO_INDEX["other_ocr"]


def _categorize_intervals(
    text: str, intervals: Sequence[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    categories: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    for interval_start, interval_end in intervals:
        run_start = interval_start
        run_category: int | None = None
        for index in range(interval_start, interval_end):
            category = _character_category(text[index])
            if run_category is None:
                run_category = category
                run_start = index
            elif category != run_category:
                categories.append(run_category)
                starts.append(run_start)
                ends.append(index)
                run_category = category
                run_start = index
        if run_category is not None:
            categories.append(run_category)
            starts.append(run_start)
            ends.append(interval_end)
    return (
        np.asarray(categories, dtype=np.uint8),
        np.asarray(starts, dtype=np.uint32),
        np.asarray(ends, dtype=np.uint32),
    )


def _gap_summaries(
    intervals: Sequence[tuple[int, int]], length: int
) -> np.ndarray:
    if length == 0 or not intervals:
        return np.zeros(len(GAP_SUMMARY_NAMES), dtype=np.float32)
    lengths = np.asarray([end - start for start, end in intervals], dtype=np.float64)
    longest_index = int(np.argmax(lengths))
    longest_start, longest_end = intervals[longest_index]
    return np.asarray(
        [
            float(lengths.sum() / length),
            float((intervals[0][1] - intervals[0][0]) / length)
            if intervals[0][0] == 0
            else 0.0,
            float((intervals[-1][1] - intervals[-1][0]) / length)
            if intervals[-1][1] == length
            else 0.0,
            float(lengths[longest_index] / length),
            float((longest_start + longest_end) / (2 * length)),
            float(len(intervals)),
            float(lengths.mean() / length),
        ],
        dtype=np.float32,
    )


def extract_positional_line(text: str) -> PositionalLineEncoding:
    """Extract lossless sparse spans plus normalized positional summaries."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    review = extract_bibliography_feature_review(text)
    normalized = review.normalized_text
    length = len(normalized)
    values = review.features.as_dict()
    counts = np.asarray([int(values[name]) for name in FEATURE_NAMES], dtype=np.uint32)

    features: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    by_feature: list[list[tuple[int, int]]] = [[] for _ in FEATURE_NAMES]
    for match in review.matches:
        feature_index = FEATURE_TO_INDEX.get(match.feature)
        if feature_index is None:
            raise RuntimeError(f"unregistered feature span: {match.feature}")
        if not 0 <= match.start < match.end <= length:
            raise RuntimeError(f"invalid feature span for {match.feature}")
        features.append(feature_index)
        starts.append(match.start)
        ends.append(match.end)
        by_feature[feature_index].append((match.start, match.end))
    observed = np.bincount(features, minlength=len(FEATURE_NAMES)).astype(np.uint32)
    if not np.array_equal(observed, counts):
        mismatches = {
            name: {"count": int(counts[index]), "spans": int(observed[index])}
            for index, name in enumerate(FEATURE_NAMES)
            if counts[index] != observed[index]
        }
        raise RuntimeError(f"feature count/span parity failure: {mismatches}")

    summaries = np.zeros(
        (len(FEATURE_NAMES), len(POSITION_SUMMARY_NAMES)), dtype=np.float32
    )
    if length:
        for feature_index, intervals in enumerate(by_feature):
            if not intervals:
                continue
            merged = _merge_intervals(intervals)
            centres = [(start + end) / 2 for start, end in intervals]
            summaries[feature_index] = np.asarray(
                [
                    min(start for start, _ in intervals) / length,
                    max(end for _, end in intervals) / length,
                    float(np.mean(centres)) / length,
                    sum(end - start for start, end in merged) / length,
                ],
                dtype=np.float32,
            )

    semantic_spans = [
        (start, end)
        for feature, start, end in zip(features, starts, ends, strict=True)
        if FEATURE_NAMES[feature] not in SEMANTIC_UNION_EXCLUDED
    ]
    unmatched = _complement(semantic_spans, length)
    nonmatch_category, nonmatch_start, nonmatch_end = _categorize_intervals(
        normalized, unmatched
    )
    return PositionalLineEncoding(
        normalized_text=normalized,
        counts=counts,
        match_feature=np.asarray(features, dtype=np.uint8),
        match_start=np.asarray(starts, dtype=np.uint32),
        match_end=np.asarray(ends, dtype=np.uint32),
        nonmatch_category=nonmatch_category,
        nonmatch_start=nonmatch_start,
        nonmatch_end=nonmatch_end,
        position_summaries=summaries,
        gap_summaries=_gap_summaries(unmatched, length),
    )


def _rasterize_spans(
    rows: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    *,
    row_offset: int,
    row_count: int,
    length: int,
    bins: int,
    output: np.ndarray,
) -> None:
    if length == 0:
        return
    bin_width = length / bins
    for raw_row, raw_start, raw_end in zip(rows, starts, ends, strict=True):
        row = row_offset + int(raw_row)
        if not row_offset <= row < row_offset + row_count:
            raise ValueError("span row is outside raster inventory")
        start, end = int(raw_start), int(raw_end)
        first = min(bins - 1, int(math.floor(start / bin_width)))
        last = min(bins - 1, int(math.ceil(end / bin_width)) - 1)
        for column in range(first, last + 1):
            left, right = column * bin_width, (column + 1) * bin_width
            overlap = max(0.0, min(end, right) - max(start, left))
            output[row, column] = min(1.0, output[row, column] + overlap / bin_width)


def rasterize_position_map(
    encoding: PositionalLineEncoding,
    *,
    bins: int,
    include_coordinate_channels: bool = False,
) -> np.ndarray:
    """Rasterize detector and non-match runs into normalized coverage bins."""

    if bins < 2:
        raise ValueError("bins must be at least two")
    base_channels = len(FEATURE_NAMES) + len(NONMATCH_CATEGORIES)
    channels = base_channels + (2 if include_coordinate_channels else 0)
    result = np.zeros((channels, bins), dtype=np.float32)
    _rasterize_spans(
        encoding.match_feature,
        encoding.match_start,
        encoding.match_end,
        row_offset=0,
        row_count=len(FEATURE_NAMES),
        length=encoding.nfkc_length,
        bins=bins,
        output=result,
    )
    _rasterize_spans(
        encoding.nonmatch_category,
        encoding.nonmatch_start,
        encoding.nonmatch_end,
        row_offset=len(FEATURE_NAMES),
        row_count=len(NONMATCH_CATEGORIES),
        length=encoding.nfkc_length,
        bins=bins,
        output=result,
    )
    if include_coordinate_channels:
        centres = (np.arange(bins, dtype=np.float32) + 0.5) / bins
        result[base_channels] = centres
        result[base_channels + 1] = 1.0 - centres
    return result


def count_gap_scalars(encoding: PositionalLineEncoding) -> np.ndarray:
    """Return binary counts, log counts, and unmatched-gap summaries (77)."""

    counts = encoding.counts.astype(np.float32)
    return np.concatenate(
        [(counts > 0).astype(np.float32), np.log1p(counts), encoding.gap_summaries]
    ).astype(np.float32, copy=False)


def positional_summary_scalars(encoding: PositionalLineEncoding) -> np.ndarray:
    """Return the P1 count features and four normalized values per feature."""

    counts = encoding.counts.astype(np.float32)
    return np.concatenate(
        [
            (counts > 0).astype(np.float32),
            np.log1p(counts),
            encoding.position_summaries.reshape(-1),
        ]
    ).astype(np.float32, copy=False)
