#!/usr/bin/env python3
"""Describe firing frequencies in the repo-defined Greek tokenizer ID ranges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

APERTUS_BASE_END = 131_072
MODERN_GREEK_END = 148_480
POLYTONIC_END = 148_992

FREQUENCY_BINS = (
    ("0", 0, 0),
    ("1", 1, 1),
    ("2-9", 2, 9),
    ("10-99", 10, 99),
    ("100-999", 100, 999),
    ("1k-9,999", 1_000, 9_999),
    ("10k-99,999", 10_000, 99_999),
    ("100k-999,999", 100_000, 999_999),
    ("1m-9,999,999", 1_000_000, 9_999_999),
    (">=10m", 10_000_000, np.iinfo(np.uint64).max),
)


def quantiles(values: np.ndarray) -> dict[str, int]:
    probabilities = (0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1)
    labels = ("min", "p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max")
    if not values.size:
        return {label: 0 for label in labels}
    return {
        label: int(value)
        for label, value in zip(
            labels, np.quantile(values, probabilities, method="nearest")
        )
    }


def token_row(
    token_id: int, counts: np.ndarray, metadata: dict[str, list[Any]]
) -> dict[str, Any]:
    return {
        "token_id": token_id,
        "decoded_text": metadata["decoded_text"][token_id],
        "model_token": metadata["model_token"][token_id],
        "count": int(counts[token_id]),
    }


def range_stats(
    name: str,
    start: int,
    end: int,
    counts: np.ndarray,
    metadata: dict[str, list[Any]],
    all_text_tokens: int,
) -> dict[str, Any]:
    values = counts[start:end]
    active = values > 0
    occurrences = int(values.sum(dtype=np.uint64))
    sorted_descending = np.sort(values)[::-1]
    cumulative = np.cumsum(sorted_descending, dtype=np.uint64)
    token_ids = np.arange(start, end)
    active_ids = token_ids[active]
    zero_ids = token_ids[~active]
    lowest_ids = active_ids[
        np.argsort(counts[active_ids], kind="stable")[:20]
    ]
    highest_ids = token_ids[
        np.argsort(counts[token_ids], kind="stable")[::-1][:20]
    ]

    histogram = []
    for label, lower, upper in FREQUENCY_BINS:
        mask = (values >= lower) & (values <= upper)
        bin_occurrences = int(values[mask].sum(dtype=np.uint64))
        histogram.append(
            {
                "range": label,
                "token_types": int(mask.sum()),
                "type_share": float(mask.mean()),
                "occurrences": bin_occurrences,
                "mass_share": bin_occurrences / occurrences if occurrences else 0,
            }
        )

    mass_coverage = {}
    for fraction in (0.5, 0.8, 0.9, 0.95, 0.99):
        required = int(
            np.searchsorted(cumulative, occurrences * fraction, side="left") + 1
        )
        mass_coverage[str(fraction)] = {
            "token_types": required,
            "type_share": required / len(values),
        }

    top_mass_shares = {}
    for amount in (1, 10, 100, 1_000):
        amount = min(amount, len(values))
        top_mass_shares[str(amount)] = (
            int(sorted_descending[:amount].sum(dtype=np.uint64)) / occurrences
            if occurrences
            else 0
        )

    sorted_float = np.sort(values.astype(np.float64))
    indexes = np.arange(1, len(values) + 1)
    gini = (
        float(
            np.sum((2 * indexes - len(values) - 1) * sorted_float)
            / (len(values) * sorted_float.sum())
        )
        if sorted_float.sum()
        else 0
    )

    char_mask = np.asarray(
        metadata["is_char_sized_or_smaller"][start:end], dtype=bool
    )
    single_mask = np.asarray(
        metadata["is_single_unicode_codepoint"][start:end], dtype=bool
    )
    fragment_mask = np.asarray(
        metadata["is_subcharacter_utf8_fragment"][start:end], dtype=bool
    )
    greek_bearing_mask = np.asarray(metadata["contains_greek"][start:end], dtype=bool)
    non_greek_ids = token_ids[~greek_bearing_mask]
    non_greek_by_frequency = non_greek_ids[
        np.argsort(counts[non_greek_ids], kind="stable")
    ]

    return {
        "name": name,
        "id_range": [start, end],
        "token_types": len(values),
        "active_token_types": int(active.sum()),
        "never_fired_token_types": int((~active).sum()),
        "occurrences": occurrences,
        "share_of_all_text_tokens": occurrences / all_text_tokens,
        "mean_all_types": occurrences / len(values),
        "mean_positive_types": occurrences / int(active.sum()) if active.any() else 0,
        "quantiles_all_types": quantiles(values),
        "quantiles_positive_types": quantiles(values[active]),
        "gini_all_types": gini,
        "histogram": histogram,
        "mass_coverage": mass_coverage,
        "top_mass_shares": top_mass_shares,
        "char_sized_or_smaller": {
            "token_types": int(char_mask.sum()),
            "active_token_types": int((char_mask & active).sum()),
            "occurrences": int(values[char_mask].sum(dtype=np.uint64)),
        },
        "single_unicode_codepoint": {
            "token_types": int(single_mask.sum()),
            "occurrences": int(values[single_mask].sum(dtype=np.uint64)),
        },
        "subcharacter_utf8_fragment": {
            "token_types": int(fragment_mask.sum()),
            "occurrences": int(values[fragment_mask].sum(dtype=np.uint64)),
        },
        "unicode_greek_bearing": {
            "token_types": int(greek_bearing_mask.sum()),
            "active_token_types": int((greek_bearing_mask & active).sum()),
            "occurrences": int(values[greek_bearing_mask].sum(dtype=np.uint64)),
        },
        "not_unicode_greek_bearing": {
            "token_types": int((~greek_bearing_mask).sum()),
            "active_token_types": int(((~greek_bearing_mask) & active).sum()),
            "occurrences": int(values[~greek_bearing_mask].sum(dtype=np.uint64)),
            "lowest": [
                token_row(int(token_id), counts, metadata)
                for token_id in non_greek_by_frequency[:20]
            ],
            "highest": [
                token_row(int(token_id), counts, metadata)
                for token_id in non_greek_by_frequency[-20:][::-1]
            ],
        },
        "never_fired": [
            token_row(int(token_id), counts, metadata) for token_id in zero_ids
        ],
        "lowest_positive": [
            token_row(int(token_id), counts, metadata) for token_id in lowest_ids
        ],
        "highest": [
            token_row(int(token_id), counts, metadata) for token_id in highest_ids
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", required=True)
    parser.add_argument("--frequency-table", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    counts = np.load(args.counts, allow_pickle=False)
    if counts.dtype != np.uint64 or counts.shape != (POLYTONIC_END,):
        raise ValueError(f"expected uint64[{POLYTONIC_END}], got {counts.dtype}{counts.shape}")
    table = pq.read_table(
        args.frequency_table,
        columns=[
            "token_id",
            "model_token",
            "decoded_text",
            "is_char_sized_or_smaller",
            "is_single_unicode_codepoint",
            "is_subcharacter_utf8_fragment",
            "contains_greek",
        ],
    )
    if table.num_rows != POLYTONIC_END:
        raise ValueError("frequency table row count differs from the vocabulary")
    metadata = table.to_pydict()
    if metadata["token_id"] != list(range(POLYTONIC_END)):
        raise ValueError("frequency table token IDs are not contiguous")

    all_text_tokens = int(counts.sum(dtype=np.uint64))
    result = {
        "schema_version": "added-greek-token-frequency-analysis-v1",
        "status": "passed",
        "all_text_tokens": all_text_tokens,
        "ranges": [
            range_stats(
                "all_added_greek",
                APERTUS_BASE_END,
                POLYTONIC_END,
                counts,
                metadata,
                all_text_tokens,
            ),
            range_stats(
                "modern_greek_c3",
                APERTUS_BASE_END,
                MODERN_GREEK_END,
                counts,
                metadata,
                all_text_tokens,
            ),
            range_stats(
                "polytonic_512",
                MODERN_GREEK_END,
                POLYTONIC_END,
                counts,
                metadata,
                all_text_tokens,
            ),
        ],
    }
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
