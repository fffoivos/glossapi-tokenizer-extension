#!/usr/bin/env python3
"""Leakage-safe shape, heading, and connector features for bibliography roles."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .bibliography_entry_dataset import MAX_PHYSICAL_GAP
from .bibliography_positional_features import FEATURE_NAMES, GAP_SUMMARY_NAMES, extract_positional_line
from .deterministic_structure import BibRole, analyze_bib_line


SCHEMA_VERSION = "bibliography-role-feature-contract-v1"
WINDOW_RADII = (1, 3, 5, 10, 30)
HEADING_PROBABILITY_NAMES = (
    "bib_header_probability",
    "bib_subheader_probability",
    "non_bib_header_probability",
)
_WORD = re.compile(r"[^\W_]+(?:[’'\-][^\W_]+)*", re.UNICODE)
_NUMBERED_HEADING = re.compile(
    r"^\s*(?:#{1,6}\s+|(?:\d+(?:\.\d+)*|[IVXLCDM]+|[Α-Ωʹ]+)[.)\]:-]\s+)\S",
    re.IGNORECASE,
)
_REPEATED_RULE = re.compile(r"^\s*(?:[-_=.*·•|:]\s*){3,}$")
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{3,}:?(?:\s*\|\s*:?-{3,}:?)+\s*\|?\s*$")
_HTML_FRAGMENT = re.compile(r"</?[A-Za-z][^>]{0,120}>")
_PAGE_NUMBER = re.compile(r"^\s*(?:p(?:age)?\.?\s*)?[ivxlcdm\d]{1,8}\s*$", re.IGNORECASE)
_BULLET_ONLY = re.compile(r"^\s*(?:[-*•·–—]|\(?\d{1,4}[.)])\s*$")
_SENTENCE_TERMINAL = frozenset(".!?;·")
_OPENING_TERMINAL = frozenset(",:;-/–—([{'\"«“")


LINE_SHAPE_NAMES = (
    "char_length",
    "log1p_char_length",
    "token_count",
    "log1p_token_count",
    "mean_token_length",
    "maximum_token_length",
    "leading_whitespace",
    "trailing_whitespace",
    "letter_fraction",
    "digit_fraction",
    "uppercase_fraction_of_letters",
    "lowercase_fraction_of_letters",
    "greek_fraction_of_letters",
    "latin_fraction_of_letters",
    "punctuation_fraction",
    "symbol_fraction",
    "whitespace_fraction",
    "other_fraction",
    "starts_lowercase",
    "starts_uppercase",
    "starts_digit",
    "starts_bullet_or_number",
    "ends_sentence_terminal",
    "ends_opening_terminal",
    "parenthesis_balance",
    "bracket_balance",
    "quote_parity",
    "is_blank",
    "is_repeated_rule",
    "is_table_rule",
    "is_page_number",
    "is_bullet_only",
    "has_html_fragment",
    "has_replacement_character",
)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _letters_by_script(text: str) -> tuple[int, int, int]:
    letters = greek = latin = 0
    for character in text:
        if not unicodedata.category(character).startswith("L"):
            continue
        letters += 1
        name = unicodedata.name(character, "")
        greek += int("GREEK" in name)
        latin += int("LATIN" in name)
    return letters, greek, latin


def line_shape(text: str) -> np.ndarray:
    """Return language-agnostic shape features over NFKC text."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFKC", text)
    stripped = normalized.strip()
    tokens = _WORD.findall(normalized)
    lengths = [len(token) for token in tokens]
    n = len(normalized)
    categories = [unicodedata.category(character) for character in normalized]
    letters, greek, latin = _letters_by_script(normalized)
    digits = sum(category.startswith("N") for category in categories)
    upper = sum(character.isupper() for character in normalized)
    lower = sum(character.islower() for character in normalized)
    punctuation = sum(category.startswith("P") for category in categories)
    symbols = sum(category.startswith("S") for category in categories)
    spaces = sum(character.isspace() for character in normalized)
    accounted = letters + digits + punctuation + symbols + spaces
    first = stripped[:1]
    last = stripped[-1:] if stripped else ""
    starts_list = bool(_NUMBERED_HEADING.match(stripped)) or bool(
        re.match(r"^\s*[-*•·–—]\s+", normalized)
    )
    quote_count = sum(character in "'\"«»“”‘’" for character in normalized)
    values = (
        n,
        math.log1p(n),
        len(tokens),
        math.log1p(len(tokens)),
        float(np.mean(lengths)) if lengths else 0.0,
        max(lengths, default=0),
        n - len(normalized.lstrip()),
        n - len(normalized.rstrip()),
        _ratio(letters, n),
        _ratio(digits, n),
        _ratio(upper, letters),
        _ratio(lower, letters),
        _ratio(greek, letters),
        _ratio(latin, letters),
        _ratio(punctuation, n),
        _ratio(symbols, n),
        _ratio(spaces, n),
        _ratio(max(0, n - accounted), n),
        int(bool(first) and first.islower()),
        int(bool(first) and first.isupper()),
        int(bool(first) and first.isdigit()),
        int(starts_list),
        int(last in _SENTENCE_TERMINAL),
        int(last in _OPENING_TERMINAL),
        normalized.count("(") - normalized.count(")"),
        normalized.count("[") - normalized.count("]"),
        quote_count % 2,
        int(not stripped),
        int(bool(_REPEATED_RULE.match(normalized))),
        int(bool(_TABLE_RULE.match(normalized))),
        int(bool(_PAGE_NUMBER.match(normalized))),
        int(bool(_BULLET_ONLY.match(normalized))),
        int(bool(_HTML_FRAGMENT.search(normalized))),
        int("\ufffd" in normalized),
    )
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (len(LINE_SHAPE_NAMES),) or not np.isfinite(result).all():
        raise RuntimeError("line shape feature contract failure")
    return result


def broad_heading_candidate(
    text: str, *, previous_blank: bool = False, next_blank: bool = False,
) -> bool:
    """High-recall candidate predicate; classification remains a separate step."""

    normalized = unicodedata.normalize("NFKC", text)
    stripped = normalized.strip()
    if not stripped:
        return False
    deterministic = analyze_bib_line(text, 0).role
    if deterministic in {BibRole.HEADING, BibRole.SUBHEADING}:
        return True
    tokens = _WORD.findall(stripped)
    if len(stripped) > 200 or len(tokens) > 24:
        return False
    letters = [character for character in stripped if character.isalpha()]
    upper_fraction = _ratio(sum(character.isupper() for character in letters), len(letters))
    lexical_tokens = [token for token in tokens if any(character.isalpha() for character in token)]
    title_like = bool(lexical_tokens) and _ratio(
        sum(token[:1].isupper() for token in lexical_tokens), len(lexical_tokens)
    ) >= 0.6
    structural = bool(_NUMBERED_HEADING.match(stripped)) or stripped.startswith("#")
    isolated = previous_blank or next_blank
    non_sentence = stripped[-1] not in _SENTENCE_TERMINAL
    return structural or upper_fraction >= 0.75 or (isolated and non_sentence and title_like)


def heading_numeric_features(
    text: str, *, previous_blank: bool, next_blank: bool,
    position_fraction: float, entry_probabilities_above: Sequence[float],
    entry_probabilities_below: Sequence[float],
) -> np.ndarray:
    above = np.asarray(entry_probabilities_above, dtype=np.float32)
    below = np.asarray(entry_probabilities_below, dtype=np.float32)
    result = np.concatenate(
        (
            line_shape(text),
            np.asarray(
                (
                    int(previous_blank), int(next_blank), float(position_fraction),
                    float(above.max(initial=0.0)), float(below.max(initial=0.0)),
                    float(above.mean()) if len(above) else 0.0,
                    float(below.mean()) if len(below) else 0.0,
                    int(np.count_nonzero(above >= 0.25)),
                    int(np.count_nonzero(below >= 0.25)),
                ),
                dtype=np.float32,
            ),
        )
    )
    if not np.isfinite(result).all():
        raise RuntimeError("non-finite heading features")
    return result


HEADING_NUMERIC_NAMES = (*LINE_SHAPE_NAMES,
    "previous_line_blank", "next_line_blank", "document_position_fraction",
    "entry_max_above_30", "entry_max_below_30", "entry_mean_above_30",
    "entry_mean_below_30", "entry_count_above_025", "entry_count_below_025",
)


def candidate_window_mask(
    entry_probability: np.ndarray, heading_candidates: np.ndarray,
    abs_indices: np.ndarray, *, entry_threshold: float = 0.25, radius: int = 30,
) -> np.ndarray:
    return candidate_seed_distances(
        entry_probability, heading_candidates, abs_indices,
        entry_threshold=entry_threshold,
    ) <= radius


def candidate_seed_distances(
    entry_probability: np.ndarray, heading_candidates: np.ndarray,
    abs_indices: np.ndarray, *, entry_threshold: float = 0.25,
) -> np.ndarray:
    """Return line distance to the nearest proposal seed without crossing physical gaps."""
    if not (
        entry_probability.shape == heading_candidates.shape == abs_indices.shape
        and entry_probability.ndim == 1
    ):
        raise ValueError("candidate-window arrays must be aligned vectors")
    size = len(entry_probability)
    sentinel = np.iinfo(np.int32).max
    distances = np.full(size, sentinel, dtype=np.int32)
    seeds = (entry_probability >= entry_threshold) | heading_candidates.astype(bool)
    previous: int | None = None
    for index in range(size):
        if index and int(abs_indices[index]) - int(abs_indices[index - 1]) > MAX_PHYSICAL_GAP:
            previous = None
        if seeds[index]:
            previous = index
        if previous is not None:
            distances[index] = index - previous
    following: int | None = None
    for index in range(size - 1, -1, -1):
        if index + 1 < size and int(abs_indices[index + 1]) - int(abs_indices[index]) > MAX_PHYSICAL_GAP:
            following = None
        if seeds[index]:
            following = index
        if following is not None:
            distances[index] = min(int(distances[index]), following - index)
    return distances


def _nearest_anchor(
    probability: np.ndarray, abs_indices: np.ndarray, index: int, direction: int,
    threshold: float,
) -> tuple[int, float]:
    for distance in range(1, 31):
        candidate = index + direction * distance
        if candidate < 0 or candidate >= len(probability):
            break
        low, high = sorted((index, candidate))
        if np.any(np.diff(abs_indices[low : high + 1].astype(np.int64)) > MAX_PHYSICAL_GAP):
            break
        if float(probability[candidate]) >= threshold:
            return distance, float(probability[candidate])
    return 31, 0.0


def _window_values(
    values: np.ndarray, abs_indices: np.ndarray, index: int, radius: int, direction: int,
) -> np.ndarray:
    selected: list[float] = []
    cursor = index
    for _ in range(radius):
        candidate = cursor + direction
        if candidate < 0 or candidate >= len(values):
            break
        low, high = sorted((cursor, candidate))
        if int(abs_indices[high]) - int(abs_indices[low]) > MAX_PHYSICAL_GAP:
            break
        selected.append(float(values[candidate]))
        cursor = candidate
    if direction < 0:
        selected.reverse()
    return np.asarray(selected, dtype=np.float32)


def _pair_features(left: str, right: str) -> np.ndarray:
    left_shape, right_shape = line_shape(left), line_shape(right)
    left_length, right_length = max(1.0, left_shape[0]), max(1.0, right_shape[0])
    joined = unicodedata.normalize("NFKC", left).rstrip() + " " + unicodedata.normalize("NFKC", right).lstrip()
    return np.asarray(
        (
            right_length / left_length,
            abs(float(right_shape[6] - left_shape[6])),
            abs(float(right_shape[12] - left_shape[12])),
            abs(float(right_shape[13] - left_shape[13])),
            float(left_shape[23]),
            float(right_shape[18]),
            abs(joined.count("(") - joined.count(")")),
            abs(joined.count("[") - joined.count("]")),
            sum(character in "'\"«»“”‘’" for character in joined) % 2,
        ),
        dtype=np.float32,
    )


PAIR_NAMES = (
    "right_to_left_length_ratio", "indentation_difference", "greek_fraction_difference",
    "latin_fraction_difference", "left_ends_opening_terminal", "right_starts_lowercase",
    "joined_parenthesis_imbalance", "joined_bracket_imbalance", "joined_quote_parity",
)


@dataclass(frozen=True)
class ConnectorFeatureRow:
    values: np.ndarray
    joined_previous_score: float
    joined_next_score: float


def connector_feature_names() -> tuple[str, ...]:
    count_names = tuple(f"presence:{name}" for name in FEATURE_NAMES) + tuple(
        f"log1p:{name}" for name in FEATURE_NAMES
    )
    gap_names = tuple(f"gap:{name}" for name in GAP_SUMMARY_NAMES)
    context = []
    for direction in ("above", "below"):
        for radius in WINDOW_RADII:
            context.extend(
                (f"entry_{direction}_r{radius}_max", f"entry_{direction}_r{radius}_mean",
                 f"entry_{direction}_r{radius}_count025")
            )
    return (
        *count_names, *LINE_SHAPE_NAMES, *gap_names,
        "nearest_anchor_above_distance", "nearest_anchor_above_probability",
        "nearest_anchor_below_distance", "nearest_anchor_below_probability",
        *context,
        *(f"previous_pair:{name}" for name in PAIR_NAMES),
        *(f"next_pair:{name}" for name in PAIR_NAMES),
        "joined_previous_entry_probability", "joined_previous_probability_gain",
        "joined_previous_distinct_feature_gain", "joined_previous_unmatched_fraction_gain",
        "joined_next_entry_probability", "joined_next_probability_gain",
        "joined_next_distinct_feature_gain", "joined_next_unmatched_fraction_gain",
        *HEADING_PROBABILITY_NAMES,
        "heading_probability_max", "inside_anchor_gap", "candidate_window_edge_distance",
    )


def connector_feature_row(
    *, index: int, texts: Sequence[str], counts: np.ndarray, gap_summaries: np.ndarray,
    abs_indices: np.ndarray, entry_probability: np.ndarray,
    heading_probability: np.ndarray, candidate_mask: np.ndarray,
    score_counts: Callable[[np.ndarray], float], entry_threshold: float = 0.25,
) -> ConnectorFeatureRow:
    n = len(texts)
    if not 0 <= index < n:
        raise IndexError(index)
    if counts.shape != (n, len(FEATURE_NAMES)) or gap_summaries.shape != (n, len(GAP_SUMMARY_NAMES)):
        raise ValueError("count/gap feature table shape mismatch")
    if heading_probability.shape != (n, len(HEADING_PROBABILITY_NAMES)):
        raise ValueError("heading probability shape mismatch")
    current_counts = counts[index].astype(np.float32)
    parts: list[np.ndarray] = [
        (current_counts > 0).astype(np.float32),
        np.log1p(current_counts),
        line_shape(texts[index]),
        gap_summaries[index].astype(np.float32),
    ]
    above_distance, above_probability = _nearest_anchor(
        entry_probability, abs_indices, index, -1, entry_threshold
    )
    below_distance, below_probability = _nearest_anchor(
        entry_probability, abs_indices, index, 1, entry_threshold
    )
    parts.append(np.asarray(
        (above_distance, above_probability, below_distance, below_probability), dtype=np.float32
    ))
    context: list[float] = []
    for direction in (-1, 1):
        for radius in WINDOW_RADII:
            values = _window_values(
                entry_probability, abs_indices, index, radius, direction,
            )
            context.extend((
                float(values.max(initial=0.0)), float(values.mean()) if len(values) else 0.0,
                float(np.count_nonzero(values >= entry_threshold)),
            ))
    parts.append(np.asarray(context, dtype=np.float32))
    previous = index - 1 if index > 0 and int(abs_indices[index]) - int(abs_indices[index - 1]) <= MAX_PHYSICAL_GAP else None
    following = index + 1 if index + 1 < n and int(abs_indices[index + 1]) - int(abs_indices[index]) <= MAX_PHYSICAL_GAP else None
    parts.append(_pair_features(texts[previous], texts[index]) if previous is not None else np.zeros(len(PAIR_NAMES), dtype=np.float32))
    parts.append(_pair_features(texts[index], texts[following]) if following is not None else np.zeros(len(PAIR_NAMES), dtype=np.float32))

    base_score = float(entry_probability[index])
    joined_scores: list[float] = []
    for neighbour, left_first in ((previous, True), (following, False)):
        if neighbour is None:
            parts.append(np.zeros(4, dtype=np.float32))
            joined_scores.append(0.0)
            continue
        joined_text = (
            texts[neighbour].rstrip() + " " + texts[index].lstrip()
            if left_first else texts[index].rstrip() + " " + texts[neighbour].lstrip()
        )
        encoding = extract_positional_line(joined_text)
        joined_score = float(score_counts(encoding.counts.reshape(1, -1)))
        neighbour_score = float(entry_probability[neighbour])
        distinct_gain = max(
            0,
            int(np.count_nonzero(encoding.counts))
            - max(int(np.count_nonzero(current_counts)), int(np.count_nonzero(counts[neighbour]))),
        )
        unmatched_gain = max(
            0.0,
            min(float(gap_summaries[index, 0]), float(gap_summaries[neighbour, 0]))
            - float(encoding.gap_summaries[0]),
        )
        parts.append(np.asarray(
            (joined_score, joined_score - max(base_score, neighbour_score), distinct_gain, unmatched_gain),
            dtype=np.float32,
        ))
        joined_scores.append(joined_score)
    parts.append(heading_probability[index].astype(np.float32))
    parts.append(np.asarray((float(heading_probability[index].max(initial=0.0)),), dtype=np.float32))
    inside_gap = int(above_distance <= 30 and below_distance <= 30)
    if candidate_mask[index]:
        left_edge = index
        while left_edge > 0 and candidate_mask[left_edge - 1]:
            left_edge -= 1
        right_edge = index
        while right_edge + 1 < n and candidate_mask[right_edge + 1]:
            right_edge += 1
        edge_distance = min(index - left_edge, right_edge - index)
    else:
        edge_distance = 0
    parts.append(np.asarray((inside_gap, edge_distance), dtype=np.float32))
    result = np.concatenate(parts).astype(np.float32)
    names = connector_feature_names()
    if result.shape != (len(names),) or not np.isfinite(result).all():
        raise RuntimeError(f"connector feature contract failure: {result.shape} != {(len(names),)}")
    return ConnectorFeatureRow(result, joined_scores[0], joined_scores[1])


def p0d_matrix(counts: np.ndarray) -> np.ndarray:
    if counts.ndim != 2 or counts.shape[1] != len(FEATURE_NAMES):
        raise ValueError("P0D counts have the wrong shape")
    values = counts.astype(np.float32)
    return np.concatenate(((values > 0).astype(np.float32), np.log1p(values)), axis=1)
