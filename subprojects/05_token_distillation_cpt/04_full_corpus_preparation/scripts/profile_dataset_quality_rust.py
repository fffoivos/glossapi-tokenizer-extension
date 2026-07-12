#!/usr/bin/env python3
"""Receipt-bound GlossAPI Rust diagnostics for canonical Phase-04 Parquet.

The command deliberately treats ``glossapi_rs_cleaner`` as an audit.  It
materializes one bounded Markdown batch at a time, asks the cleaner for metrics
without persisting cleaned files, and deletes the temporary Markdown before the
next batch.  Canonical text is never rewritten by this program.

Two subcommands are exposed:

``build-receipt``
    Attest already-built PyO3 modules to the pinned, clean GlossAPI checkout.

``run``
    Validate that attestation, stream normalized Parquet shards, checkpoint each
    batch, and emit a consolidated per-document Parquet plus summary JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import importlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from profile_source_quality import (
    ADA_LINE,
    BIB_HEADER,
    DIGITAL_GOVERNANCE,
    PERSONNEL_CUE,
    TOC_HEADER,
    line_quality,
    normalized_template,
)


PINNED_GLOSSAPI_COMMIT = "6f29a2825559c540ab342fc77ae4457cf3556f2a"
BUILD_RECEIPT_SCHEMA = "glossapi_rust_quality_build_receipt_v1"
DOCUMENT_SCHEMA = "dataset_quality_document_v1"
BATCH_RECEIPT_SCHEMA = "dataset_quality_rust_batch_receipt_v1"
SUMMARY_SCHEMA = "dataset_quality_summary_v1"
CONTRACT_SCHEMA = "dataset_quality_rust_contract_v1"
QUALITY_HANDOFF_SCHEMA = "dataset_quality_site_handoff_v1"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GREEK_RE = re.compile(r"[\u0370-\u03ff\u1f00-\u1fff]")
LATIN_RE = re.compile(r"[A-Za-z\u00c0-\u024f]")
HTML_RE = re.compile(r"<\s*/?\s*[A-Za-z][^>]{0,200}>")
MOJIBAKE_RE = re.compile(r"(?:Ã.|Â.|â€|Î[\x80-\xbf]|Ï[\x80-\xbf])")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
IBAN_RE = re.compile(r"(?i)\bGR\s*\d{2}(?:[\s-]*[0-9A-Z]){23}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?30[\s.-]*)?(?:2\d{9}|69\d{8})(?!\d)")
AFM_RE = re.compile(r"(?i)(?:Α\.?\s*Φ\.?\s*Μ\.?|ΑΦΜ)\s*[:#-]?\s*\d{9}\b")
AMKA_RE = re.compile(r"(?i)(?:Α\.?\s*Μ\.?\s*Κ\.?\s*Α\.?|ΑΜΚΑ)\s*[:#-]?\s*\d{11}\b")
IDENTITY_RE = re.compile(
    r"(?i)(?:Α\.?\s*Δ\.?\s*Τ\.?|ΑΔΤ|ταυτότητας|passport)\s*[:#-]?\s*[A-ZΑ-Ω]{1,3}[\s-]?\d{5,10}\b"
)

PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", EMAIL_RE),
    ("iban", IBAN_RE),
    ("phone", PHONE_RE),
    ("afm_labelled", AFM_RE),
    ("amka_labelled", AMKA_RE),
    ("identity_labelled", IDENTITY_RE),
)

NOISE_FIELDS: tuple[str, ...] = (
    "rust_noise_badness_score",
    "rust_noise_latin_percentage",
    "rust_noise_table_ratio",
    "rust_noise_polytonic_ratio",
    "rust_noise_greek_characters",
    "rust_noise_total_words",
    "rust_noise_vowel_penalty",
    "rust_noise_consonant_penalty",
    "rust_noise_bad_double_count",
    "rust_noise_misplaced_final_sigma_count",
    "rust_noise_invalid_bigram_count",
    "rust_noise_long_word_count",
    "rust_noise_longest_word",
    "rust_noise_short_word_count",
    "rust_noise_max_character_run",
    "rust_noise_vowel_penalty_rate",
    "rust_noise_consonant_penalty_rate",
    "rust_noise_bad_double_rate",
    "rust_noise_final_sigma_rate",
    "rust_noise_invalid_bigram_rate",
    "rust_noise_long_word_rate",
    "rust_noise_short_word_ratio",
    "rust_noise_short_word_penalty",
    "rust_noise_flags",
)

FLOAT_NOISE_FIELDS = {
    "rust_noise_badness_score",
    "rust_noise_latin_percentage",
    "rust_noise_table_ratio",
    "rust_noise_polytonic_ratio",
    "rust_noise_vowel_penalty_rate",
    "rust_noise_consonant_penalty_rate",
    "rust_noise_bad_double_rate",
    "rust_noise_final_sigma_rate",
    "rust_noise_invalid_bigram_rate",
    "rust_noise_long_word_rate",
    "rust_noise_short_word_ratio",
    "rust_noise_short_word_penalty",
}

INTEGER_NOISE_FIELDS = set(NOISE_FIELDS) - FLOAT_NOISE_FIELDS - {"rust_noise_flags"}

DISTRIBUTION_METRICS: tuple[str, ...] = (
    "original_characters",
    "raw_greek_letter_fraction",
    "raw_html_tags_per_1000_chars",
    "raw_mojibake_per_1000_chars",
    "raw_replacement_per_1000_chars",
    "raw_control_per_1000_chars",
    "raw_repeated_line_fraction",
    "raw_one_token_line_fraction",
    "raw_markdown_table_lines",
    "rust_noise_badness_score",
    "rust_noise_latin_percentage",
    "rust_noise_table_ratio",
    "cleaner_badness_score",
    "cleaner_removed_character_fraction",
)

DOCUMENT_COUNTERS: tuple[str, ...] = (
    "empty_input_documents",
    "html_documents",
    "mojibake_documents",
    "replacement_character_documents",
    "control_character_documents",
    "low_unique_line_fraction_documents",
    "one_token_per_line_documents",
    "markdown_table_documents",
    "large_markdown_table_documents",
    "bibliography_header_documents",
    "toc_header_documents",
    "digital_governance_footer_documents",
    "personnel_cue_documents",
    "isolated_ada_stamp_documents",
    "private_data_true_documents",
    "corrected_version_documents",
    "direct_identifier_documents",
    "cleaner_empty_documents",
    "zero_badness_zero_greek_guard_documents",
)

UI_RATE_METRICS: tuple[str, ...] = (
    "html_rate",
    "mojibake_rate",
    "replacement_character_rate",
    "control_character_rate",
    "one_token_per_line_rate",
    "markdown_table_rate",
    "large_markdown_table_rate",
    "bibliography_header_rate",
    "toc_header_rate",
    "digital_governance_footer_rate",
    "personnel_cue_rate",
    "isolated_ada_stamp_rate",
    "private_data_true_rate",
    "corrected_version_rate",
    "direct_identifier_rate",
    "zero_badness_zero_greek_guard_rate",
)

UI_DISTRIBUTION_METRICS: tuple[str, ...] = (
    "original_characters",
    "raw_greek_letter_fraction",
    "raw_mojibake_per_1000_chars",
    "raw_replacement_per_1000_chars",
    "raw_repeated_line_fraction",
    "raw_one_token_line_fraction",
    "rust_noise_badness_score",
    "cleaner_removed_character_fraction",
)

SUMMARY_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "created_at",
        "mode",
        "scan_mode",
        "contract_sha256",
        "contract",
        "normalization_manifest",
        "normalization_schema_version",
        "glossapi_build_receipt",
        "glossapi_commit",
        "batch_size",
        "threads",
        "quantile_sample_size",
        "selected_source_ids",
        "excluded_source_ids",
        "input_shards",
        "batch_checkpoints",
        "document_output",
        "global",
        "repositories",
        "metric_notes",
    }
)


def require_exact_keys(
    value: Mapping[str, Any],
    *,
    required: Iterable[str],
    optional: Iterable[str] = (),
    context: str,
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    actual = set(value)
    missing = required_set - actual
    unexpected = actual - allowed
    if missing or unexpected:
        raise ValueError(
            f"{context}: key contract drift; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )


def require_sha256(value: Any, *, context: str) -> str:
    result = str(value)
    if not SHA256_RE.fullmatch(result):
        raise ValueError(f"{context}: expected SHA-256")
    return result


def require_nonnegative_int(value: Any, *, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{context}: expected a nonnegative integer")
    return value


def require_finite_number(
    value: Any, *, context: str, nullable: bool = False
) -> float | None:
    if value is None and nullable:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{context}: expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context}: expected a finite number")
    return result


def validate_receipt_object(
    value: Any,
    *,
    context: str,
    require_rows: bool = False,
    allow_rows: bool = True,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}: receipt must be an object")
    required = {"path", "bytes", "sha256"}
    if require_rows:
        required.add("rows")
    optional = {"rows"} if allow_rows else set()
    require_exact_keys(value, required=required, optional=optional, context=context)
    path = str(value["path"])
    if not path:
        raise ValueError(f"{context}: empty receipt path")
    result = {
        "path": path,
        "bytes": require_nonnegative_int(value["bytes"], context=f"{context}.bytes"),
        "sha256": require_sha256(value["sha256"], context=f"{context}.sha256"),
    }
    if "rows" in value:
        result["rows"] = require_nonnegative_int(
            value["rows"], context=f"{context}.rows"
        )
    return result


def _validate_distribution(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}: distribution must be an object")
    keys = {
        "count",
        "min",
        "mean",
        "p10_approx",
        "p50_approx",
        "p90_approx",
        "p99_approx",
        "max",
        "quantile_sample_documents",
    }
    require_exact_keys(value, required=keys, context=context)
    count = require_nonnegative_int(value["count"], context=f"{context}.count")
    sampled = require_nonnegative_int(
        value["quantile_sample_documents"],
        context=f"{context}.quantile_sample_documents",
    )
    if sampled > count:
        raise ValueError(f"{context}: quantile sample exceeds metric count")
    result: dict[str, Any] = {"count": count, "quantile_sample_documents": sampled}
    for name in (
        "min",
        "mean",
        "p10_approx",
        "p50_approx",
        "p90_approx",
        "p99_approx",
        "max",
    ):
        result[name] = require_finite_number(
            value[name], context=f"{context}.{name}", nullable=True
        )
    nonnull_quantiles = [
        result[name]
        for name in (
            "min",
            "p10_approx",
            "p50_approx",
            "p90_approx",
            "p99_approx",
            "max",
        )
        if result[name] is not None
    ]
    if nonnull_quantiles != sorted(nonnull_quantiles):
        raise ValueError(f"{context}: quantiles are not monotonic")
    statistic_names = (
        "min",
        "mean",
        "p10_approx",
        "p50_approx",
        "p90_approx",
        "p99_approx",
        "max",
    )
    if (
        count == 0
        and (sampled != 0 or any(result[name] is not None for name in statistic_names))
    ) or (
        count > 0
        and (sampled < 1 or any(result[name] is None for name in statistic_names))
    ):
        raise ValueError(f"{context}: distribution count/statistic nullability drift")
    return result


def _validate_statistics(
    value: Any, *, context: str, require_repo_id: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}: statistics must be an object")
    keys = {
        "documents",
        "characters",
        "bytes_utf8",
        "source_datasets",
        "document_counts",
        "document_rates",
        "distributions",
        "template_concentration",
    }
    if require_repo_id:
        keys.add("repo_id")
    require_exact_keys(value, required=keys, context=context)
    documents = require_nonnegative_int(
        value["documents"], context=f"{context}.documents"
    )
    characters = require_nonnegative_int(
        value["characters"], context=f"{context}.characters"
    )
    bytes_utf8 = require_nonnegative_int(
        value["bytes_utf8"], context=f"{context}.bytes_utf8"
    )
    source_datasets = value["source_datasets"]
    if (
        not isinstance(source_datasets, list)
        or documents < 1
        or not source_datasets
        or any(not isinstance(item, str) or not item for item in source_datasets)
        or source_datasets != sorted(set(source_datasets))
    ):
        raise ValueError(f"{context}.source_datasets: invalid sorted unique strings")

    counts = value["document_counts"]
    rates = value["document_rates"]
    if not isinstance(counts, Mapping) or not isinstance(rates, Mapping):
        raise ValueError(f"{context}: count/rate maps must be objects")
    expected_rate_names = {
        name.removesuffix("_documents") + "_rate" for name in DOCUMENT_COUNTERS
    }
    require_exact_keys(
        counts, required=DOCUMENT_COUNTERS, context=f"{context}.document_counts"
    )
    require_exact_keys(
        rates, required=expected_rate_names, context=f"{context}.document_rates"
    )
    validated_counts: dict[str, int] = {}
    validated_rates: dict[str, float] = {}
    for counter_name in DOCUMENT_COUNTERS:
        count = require_nonnegative_int(
            counts[counter_name], context=f"{context}.document_counts.{counter_name}"
        )
        if count > documents:
            raise ValueError(f"{context}: document counter exceeds denominator")
        validated_counts[counter_name] = count
        rate_name = counter_name.removesuffix("_documents") + "_rate"
        rate = require_finite_number(rates[rate_name], context=f"{context}.{rate_name}")
        assert rate is not None
        expected = count / documents if documents else 0.0
        if not 0.0 <= rate <= 1.0 or not math.isclose(rate, expected, abs_tol=1e-12):
            raise ValueError(f"{context}: inconsistent document rate {rate_name}")
        validated_rates[rate_name] = rate

    distributions = value["distributions"]
    if not isinstance(distributions, Mapping):
        raise ValueError(f"{context}.distributions: expected object")
    require_exact_keys(
        distributions,
        required=DISTRIBUTION_METRICS,
        context=f"{context}.distributions",
    )
    validated_distributions = {
        name: _validate_distribution(
            distributions[name], context=f"{context}.distributions.{name}"
        )
        for name in DISTRIBUTION_METRICS
    }
    template = value["template_concentration"]
    if not isinstance(template, Mapping):
        raise ValueError(f"{context}.template_concentration: expected object")
    require_exact_keys(
        template,
        required=(
            "documents_with_template",
            "unique_templates",
            "top_1_fraction",
            "top_10_fraction",
        ),
        context=f"{context}.template_concentration",
    )
    template_documents = require_nonnegative_int(
        template["documents_with_template"],
        context=f"{context}.template_concentration.documents_with_template",
    )
    unique_templates = require_nonnegative_int(
        template["unique_templates"],
        context=f"{context}.template_concentration.unique_templates",
    )
    if template_documents > documents or unique_templates > template_documents:
        raise ValueError(f"{context}: invalid template denominators")
    top_1 = require_finite_number(
        template["top_1_fraction"],
        context=f"{context}.template_concentration.top_1_fraction",
    )
    top_10 = require_finite_number(
        template["top_10_fraction"],
        context=f"{context}.template_concentration.top_10_fraction",
    )
    assert top_1 is not None and top_10 is not None
    if not 0 <= top_1 <= top_10 <= 1:
        raise ValueError(f"{context}: invalid template concentration")

    repo_id = str(value.get("repo_id", ""))
    if require_repo_id and not repo_id:
        raise ValueError(f"{context}: empty repo_id")
    validated = {
        "repo_id": repo_id,
        "documents": documents,
        "characters": characters,
        "bytes_utf8": bytes_utf8,
        "source_datasets": list(source_datasets),
        "document_counts": validated_counts,
        "document_rates": validated_rates,
        "distributions": validated_distributions,
        "template_concentration": {
            "documents_with_template": template_documents,
            "unique_templates": unique_templates,
            "top_1_fraction": top_1,
            "top_10_fraction": top_10,
        },
    }
    projected = {
        "documents": documents,
        "document_rates": {name: validated_rates[name] for name in UI_RATE_METRICS},
        "distributions": {
            name: {
                key: validated_distributions[name][key]
                for key in ("p50_approx", "p90_approx", "p99_approx")
            }
            for name in UI_DISTRIBUTION_METRICS
        },
        "template_concentration": dict(validated["template_concentration"]),
    }
    if require_repo_id:
        projected["repo_id"] = repo_id
    return validated, projected


def validate_and_project_quality_summary(value: Any) -> dict[str, Any]:
    """Strictly validate a quality summary and return only site UI aggregates."""

    if not isinstance(value, Mapping):
        raise ValueError("quality summary root must be an object")
    require_exact_keys(
        value, required=SUMMARY_TOP_LEVEL_KEYS, context="quality_summary"
    )
    if (
        value["schema_version"] != SUMMARY_SCHEMA
        or value["status"] != "passed"
        or value["mode"] != "diagnostic_only_no_cleaned_text_persisted"
    ):
        raise ValueError("quality_summary: unsupported schema/status/mode")
    scan_mode = str(value["scan_mode"])
    if scan_mode not in {"review_sample", "full_scan"}:
        raise ValueError("quality_summary.scan_mode: invalid")
    require_sha256(value["contract_sha256"], context="quality_summary.contract_sha256")
    validate_receipt_object(value["contract"], context="quality_summary.contract")
    validate_receipt_object(
        value["normalization_manifest"],
        context="quality_summary.normalization_manifest",
    )
    validate_receipt_object(
        value["glossapi_build_receipt"],
        context="quality_summary.glossapi_build_receipt",
    )
    if value["normalization_schema_version"] != "full_cpt_normalization_manifest_v1":
        raise ValueError("quality_summary: normalization schema drift")
    if not re.fullmatch(r"[0-9a-f]{40}", str(value["glossapi_commit"])):
        raise ValueError("quality_summary.glossapi_commit: invalid")
    for name in ("batch_size", "threads", "quantile_sample_size"):
        if require_nonnegative_int(value[name], context=f"quality_summary.{name}") < 1:
            raise ValueError(f"quality_summary.{name}: must be positive")

    selected = value["selected_source_ids"]
    excluded = value["excluded_source_ids"]
    for name, items, allow_empty in (
        ("selected_source_ids", selected, False),
        ("excluded_source_ids", excluded, True),
    ):
        if (
            not isinstance(items, list)
            or (not allow_empty and not items)
            or any(not isinstance(item, str) or not item for item in items)
            or items != sorted(set(items))
        ):
            raise ValueError(f"quality_summary.{name}: invalid sorted unique strings")
    if set(selected) & set(excluded):
        raise ValueError("quality_summary: selected/excluded source overlap")

    input_shards = value["input_shards"]
    if not isinstance(input_shards, list) or not input_shards:
        raise ValueError("quality_summary.input_shards: expected nonempty list")
    normalized_shards: list[dict[str, Any]] = []
    for index, row in enumerate(input_shards):
        if not isinstance(row, Mapping):
            raise ValueError(f"quality_summary.input_shards[{index}]: expected object")
        require_exact_keys(
            row,
            required=("source_id", "path", "bytes", "sha256", "rows"),
            optional=("batches",),
            context=f"quality_summary.input_shards[{index}]",
        )
        source_id = str(row["source_id"])
        path = str(row["path"])
        if not source_id or not path:
            raise ValueError("quality_summary.input_shards: empty identity")
        item = {
            "source_id": source_id,
            "path": path,
            "bytes": require_nonnegative_int(
                row["bytes"], context=f"quality_summary.input_shards[{index}].bytes"
            ),
            "sha256": require_sha256(
                row["sha256"], context=f"quality_summary.input_shards[{index}].sha256"
            ),
            "rows": require_nonnegative_int(
                row["rows"], context=f"quality_summary.input_shards[{index}].rows"
            ),
        }
        if "batches" in row:
            item["batches"] = require_nonnegative_int(
                row["batches"], context=f"quality_summary.input_shards[{index}].batches"
            )
        normalized_shards.append(item)
    if normalized_shards != sorted(
        normalized_shards, key=lambda row: (str(row["source_id"]), str(row["path"]))
    ):
        raise ValueError("quality_summary.input_shards: inventory is not canonical")

    checkpoints = value["batch_checkpoints"]
    if not isinstance(checkpoints, Mapping):
        raise ValueError("quality_summary.batch_checkpoints: expected object")
    require_exact_keys(
        checkpoints,
        required=("count", "rows", "inventory_sha256", "inventory"),
        context="quality_summary.batch_checkpoints",
    )
    checkpoint_count = require_nonnegative_int(
        checkpoints["count"], context="quality_summary.batch_checkpoints.count"
    )
    checkpoint_rows = require_nonnegative_int(
        checkpoints["rows"], context="quality_summary.batch_checkpoints.rows"
    )
    require_sha256(
        checkpoints["inventory_sha256"],
        context="quality_summary.batch_checkpoints.inventory_sha256",
    )
    inventory = checkpoints["inventory"]
    if not isinstance(inventory, list) or len(inventory) != checkpoint_count:
        raise ValueError("quality_summary.batch_checkpoints: count/inventory drift")
    for index, row in enumerate(inventory):
        if not isinstance(row, Mapping):
            raise ValueError(
                "quality_summary.batch_checkpoints.inventory: expected objects"
            )
        require_exact_keys(
            row,
            required=(
                "receipt_path",
                "receipt_sha256",
                "output_sha256",
                "rows",
                "input_shard_sha256",
                "batch_index",
            ),
            context=f"quality_summary.batch_checkpoints.inventory[{index}]",
        )
        if not str(row["receipt_path"]):
            raise ValueError("quality_summary: empty checkpoint receipt path")
        for name in ("receipt_sha256", "output_sha256", "input_shard_sha256"):
            require_sha256(row[name], context=f"quality_summary.checkpoint.{name}")
        require_nonnegative_int(row["rows"], context="quality_summary.checkpoint.rows")
        require_nonnegative_int(
            row["batch_index"], context="quality_summary.checkpoint.batch_index"
        )
    if sha256_json(inventory) != checkpoints["inventory_sha256"]:
        raise ValueError("quality_summary: checkpoint inventory hash drift")

    document_output = validate_receipt_object(
        value["document_output"],
        context="quality_summary.document_output",
        require_rows=True,
    )
    global_validated, _ = _validate_statistics(
        value["global"], context="quality_summary.global", require_repo_id=False
    )
    repositories = value["repositories"]
    if not isinstance(repositories, list) or not repositories:
        raise ValueError("quality_summary.repositories: expected nonempty list")
    validated_repositories: list[dict[str, Any]] = []
    projected_repositories: list[dict[str, Any]] = []
    seen_repos: set[str] = set()
    for index, row in enumerate(repositories):
        validated, projected = _validate_statistics(
            row, context=f"quality_summary.repositories[{index}]", require_repo_id=True
        )
        if validated["repo_id"] in seen_repos:
            raise ValueError("quality_summary.repositories: duplicate repo_id")
        seen_repos.add(validated["repo_id"])
        validated_repositories.append(validated)
        projected_repositories.append(projected)
    if repositories != sorted(
        repositories, key=lambda row: str(row.get("repo_id", ""))
    ):
        raise ValueError("quality_summary.repositories: not sorted")
    documents = global_validated["documents"]
    if (
        documents != document_output.get("rows")
        or documents != checkpoint_rows
        or documents != sum(row["documents"] for row in validated_repositories)
        or checkpoint_rows
        != sum(
            require_nonnegative_int(
                row["rows"], context="quality_summary.checkpoint.rows"
            )
            for row in inventory
        )
    ):
        raise ValueError(
            "quality_summary: document/checkpoint/repository denominator drift"
        )
    if (
        global_validated["characters"]
        != sum(row["characters"] for row in validated_repositories)
        or global_validated["bytes_utf8"]
        != sum(row["bytes_utf8"] for row in validated_repositories)
        or global_validated["source_datasets"]
        != sorted(
            {
                dataset
                for row in validated_repositories
                for dataset in row["source_datasets"]
            }
        )
        or global_validated["template_concentration"]["documents_with_template"]
        != sum(
            row["template_concentration"]["documents_with_template"]
            for row in validated_repositories
        )
    ):
        raise ValueError("quality_summary: global/repository aggregate drift")
    for counter_name in DOCUMENT_COUNTERS:
        if global_validated["document_counts"][counter_name] != sum(
            row["document_counts"][counter_name] for row in validated_repositories
        ):
            raise ValueError(
                f"quality_summary: global counter drift for {counter_name}"
            )
    for metric_name in DISTRIBUTION_METRICS:
        global_metric = global_validated["distributions"][metric_name]
        repository_metrics = [
            row["distributions"][metric_name] for row in validated_repositories
        ]
        count = sum(row["count"] for row in repository_metrics)
        if global_metric["count"] != count:
            raise ValueError(
                f"quality_summary: global distribution count drift for {metric_name}"
            )
        if count:
            mean = (
                sum(
                    float(row["mean"]) * int(row["count"])
                    for row in repository_metrics
                    if row["mean"] is not None
                )
                / count
            )
            minimum = min(
                float(row["min"])
                for row in repository_metrics
                if row["min"] is not None
            )
            maximum = max(
                float(row["max"])
                for row in repository_metrics
                if row["max"] is not None
            )
            if (
                global_metric["mean"] is None
                or global_metric["min"] is None
                or global_metric["max"] is None
                or not math.isclose(float(global_metric["mean"]), mean, rel_tol=1e-12)
                or not math.isclose(float(global_metric["min"]), minimum, rel_tol=1e-12)
                or not math.isclose(float(global_metric["max"]), maximum, rel_tol=1e-12)
            ):
                raise ValueError(
                    f"quality_summary: global distribution aggregate drift for {metric_name}"
                )

    metric_notes = value["metric_notes"]
    expected_notes = {
        "rust_noise_badness_score",
        "cleaner_removed_character_fraction",
        "approximate_quantiles",
        "zero_badness_zero_greek_guard",
        "profile_scope",
    }
    if not isinstance(metric_notes, Mapping):
        raise ValueError("quality_summary.metric_notes: expected object")
    require_exact_keys(
        metric_notes, required=expected_notes, context="quality_summary.metric_notes"
    )
    if any(
        not isinstance(metric_notes[name], str) or not metric_notes[name]
        for name in expected_notes
    ):
        raise ValueError("quality_summary.metric_notes: invalid text")

    return {
        "scan_mode": scan_mode,
        "documents": documents,
        "selected_source_ids": list(selected),
        "excluded_source_ids": list(excluded),
        "repositories": projected_repositories,
        "input_shard_inventory_sha256": sha256_json(normalized_shards),
        "checkpoint_inventory_sha256": str(checkpoints["inventory_sha256"]),
        "document_output": document_output,
    }


def _portable_file_receipt(path: Path, *, label: str) -> dict[str, Any]:
    return {
        "path": label,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def normalization_identity_closure(manifest_path: Path) -> dict[str, Any]:
    """Validate text-free normalization/acquisition identities and project them."""

    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != "full_cpt_normalization_manifest_v1":
        raise ValueError(f"{manifest_path}: unsupported normalization manifest")
    sources_config_path = Path(str(manifest.get("sources_config", ""))).resolve()
    acquisition_path = Path(str(manifest.get("acquisition_receipt", ""))).resolve()
    if (
        not sources_config_path.is_file()
        or sha256_file(sources_config_path) != manifest.get("sources_config_sha256")
        or not acquisition_path.is_file()
        or sha256_file(acquisition_path) != manifest.get("acquisition_receipt_sha256")
    ):
        raise ValueError("normalization dependency receipt drift")
    sources_config = read_json(sources_config_path)
    acquisition = read_json(acquisition_path)
    if (
        acquisition.get("schema_version") != "full_cpt_acquisition_receipt_v1"
        or acquisition.get("status") != "passed"
        or acquisition.get("sources_config_sha256") != sha256_file(sources_config_path)
    ):
        raise ValueError("normalization acquisition receipt is unsupported/unbound")
    if (
        sources_config.get("schema_version") != "full_cpt_sources_v1"
        or not isinstance(sources_config.get("base"), Mapping)
        or not isinstance(sources_config.get("sources"), list)
    ):
        raise ValueError("unsupported sources config identity closure")
    configured_sources: dict[str, Mapping[str, Any]] = {}
    configured_rows = [
        ("nanochat_base", sources_config["base"]),
        *(
            (str(row.get("source_id", "")), row)
            for row in sources_config["sources"]
            if isinstance(row, Mapping)
        ),
    ]
    if len(configured_rows) != 1 + len(sources_config["sources"]):
        raise ValueError("sources config contains a non-object source")
    for source_id, row in configured_rows:
        if not source_id or source_id in configured_sources:
            raise ValueError(
                "sources config contains a duplicate/empty source identity"
            )
        configured_sources[source_id] = row
    acquisition_sources: dict[str, Mapping[str, Any]] = {}
    for row in acquisition.get("sources", []):
        if not isinstance(row, Mapping):
            raise ValueError("acquisition source identity must be an object")
        source_id = str(row.get("source_id", ""))
        if not source_id or source_id in acquisition_sources:
            raise ValueError("duplicate/empty acquisition source identity")
        acquisition_sources[source_id] = row

    identities: list[dict[str, Any]] = []
    normalized_shards: list[dict[str, Any]] = []
    seen_manifest_source_ids: set[str] = set()
    canonical_root = Path(str(manifest.get("output", ""))).resolve()
    for source in manifest.get("sources", []):
        if not isinstance(source, Mapping):
            raise ValueError("normalization source identity must be an object")
        source_id = str(source.get("source_id", ""))
        repo_id = str(source.get("repo_id", ""))
        revision = str(source.get("revision", ""))
        if (
            not source_id
            or not repo_id
            or not revision
            or source_id in seen_manifest_source_ids
        ):
            raise ValueError("normalization source identity is incomplete")
        seen_manifest_source_ids.add(source_id)
        configured = configured_sources.get(source_id)
        if (
            configured is None
            or configured.get("repo_id") != repo_id
            or configured.get("revision") != revision
            or configured.get("role") != source.get("role")
        ):
            raise ValueError(f"{source_id}: config/normalization identity drift")
        acquisition_source = acquisition_sources.get(source_id)
        if (
            acquisition_source is None
            or acquisition_source.get("repo_id") != repo_id
            or acquisition_source.get("revision") != revision
            or acquisition_source.get("role") != source.get("role")
        ):
            raise ValueError(f"{source_id}: normalization/acquisition identity drift")
        source_receipt = source.get("receipt")
        if not isinstance(source_receipt, Mapping):
            raise ValueError(f"{source_id}: missing normalization source receipt")
        source_receipt_path = Path(str(source_receipt.get("path", ""))).resolve()
        validate_file_receipt(source_receipt_path, source_receipt)
        source_receipt_value = read_json(source_receipt_path)
        if any(
            source_receipt_value.get(name) != expected
            for name, expected in (
                ("schema_version", "full_cpt_normalization_source_receipt_v1"),
                ("source_id", source_id),
                ("repo_id", repo_id),
                ("revision", revision),
            )
        ):
            raise ValueError(
                f"{source_id}: normalization source receipt identity drift"
            )

        shards: list[dict[str, Any]] = []
        for shard in source.get("shards", []):
            if not isinstance(shard, Mapping):
                raise ValueError(f"{source_id}: invalid normalized shard")
            core = {
                "source_id": source_id,
                "path": _safe_under(
                    Path(str(shard.get("path", ""))).resolve(), canonical_root
                ),
                "bytes": require_nonnegative_int(
                    shard.get("bytes"), context=f"{source_id}.shard.bytes"
                ),
                "sha256": require_sha256(
                    shard.get("sha256"), context=f"{source_id}.shard.sha256"
                ),
                "rows": require_nonnegative_int(
                    shard.get("rows"), context=f"{source_id}.shard.rows"
                ),
            }
            if not core["path"] or core["rows"] < 1:
                raise ValueError(f"{source_id}: invalid normalized shard identity")
            shard_receipt = shard.get("receipt")
            if not isinstance(shard_receipt, Mapping):
                raise ValueError(f"{source_id}: missing normalized shard receipt")
            shard_receipt_path = Path(str(shard_receipt.get("path", ""))).resolve()
            validate_file_receipt(shard_receipt_path, shard_receipt)
            shard_receipt_value = read_json(shard_receipt_path)
            shard_output = shard_receipt_value.get("output")
            if (
                shard_receipt_value.get("schema_version")
                != "full_cpt_normalization_shard_receipt_v1"
                or shard_receipt_value.get("source_id") != source_id
                or not isinstance(shard_output, Mapping)
                or _safe_under(
                    Path(str(shard_output.get("path", ""))).resolve(), canonical_root
                )
                != core["path"]
                or int(shard_output.get("bytes", -1)) != core["bytes"]
                or shard_output.get("sha256") != core["sha256"]
                or int(shard_output.get("rows", -1)) != core["rows"]
            ):
                raise ValueError(f"{source_id}: normalized shard receipt drift")
            shards.append(core)
            normalized_shards.append(core)
        if not shards:
            raise ValueError(f"{source_id}: normalized source has no shards")
        source_receipt_shards = source_receipt_value.get("shards")
        if not isinstance(source_receipt_shards, list):
            raise ValueError(f"{source_id}: source receipt lacks shard closure")
        source_receipt_shard_core: list[dict[str, Any]] = []
        for row in source_receipt_shards:
            if not isinstance(row, Mapping):
                raise ValueError(f"{source_id}: invalid source-receipt shard")
            source_receipt_shard_core.append(
                {
                    "source_id": source_id,
                    "path": _safe_under(
                        Path(str(row.get("path", ""))).resolve(), canonical_root
                    ),
                    "bytes": int(row.get("bytes", -1)),
                    "sha256": str(row.get("sha256", "")),
                    "rows": int(row.get("rows", -1)),
                }
            )
        if sorted(
            source_receipt_shard_core, key=lambda row: str(row["path"])
        ) != sorted(
            shards, key=lambda row: str(row["path"])
        ) or source_receipt_value.get("role") != source.get("role"):
            raise ValueError(f"{source_id}: source receipt shard/role drift")

        acquisition_files = acquisition_source.get("files", [])
        if not isinstance(acquisition_files, list) or not acquisition_files:
            raise ValueError(f"{source_id}: acquisition source has no files")
        file_projection: list[dict[str, Any]] = []
        for file_row in acquisition_files:
            if not isinstance(file_row, Mapping):
                raise ValueError(f"{source_id}: invalid acquisition file receipt")
            file_projection.append(
                {
                    "path": str(file_row.get("path", "")),
                    "size": require_nonnegative_int(
                        file_row.get("size"), context=f"{source_id}.acquisition.size"
                    ),
                    "hash_kind": str(file_row.get("hash_kind", "")),
                    "expected_hash": str(file_row.get("expected_hash", "")),
                }
            )
        if any(
            not row["path"]
            or row["hash_kind"] not in {"sha256", "lfs_sha256", "git_blob_id"}
            or not re.fullmatch(
                (
                    r"[0-9a-f]{40}|[0-9a-f]{64}"
                    if row["hash_kind"] == "git_blob_id"
                    else r"[0-9a-f]{64}"
                ),
                row["expected_hash"],
            )
            for row in file_projection
        ):
            raise ValueError(f"{source_id}: incomplete acquisition file identity")
        file_projection.sort(key=lambda row: str(row["path"]))
        if len({row["path"] for row in file_projection}) != len(file_projection):
            raise ValueError(f"{source_id}: duplicate acquisition file identity")
        declared_file_count = require_nonnegative_int(
            acquisition_source.get("selected_file_count"),
            context=f"{source_id}.acquisition.selected_file_count",
        )
        declared_file_bytes = require_nonnegative_int(
            acquisition_source.get("selected_bytes"),
            context=f"{source_id}.acquisition.selected_bytes",
        )
        if declared_file_count != len(file_projection) or declared_file_bytes != sum(
            int(row["size"]) for row in file_projection
        ):
            raise ValueError(f"{source_id}: acquisition file totals drift")
        counts = source.get("counts", {})
        documents = (
            int(counts.get("documents_emitted", 0))
            if isinstance(counts, Mapping)
            else 0
        )
        receipt_counts = source_receipt_value.get("counts")
        if (
            documents < 1
            or documents != sum(int(row["rows"]) for row in shards)
            or not isinstance(receipt_counts, Mapping)
            or int(receipt_counts.get("documents_emitted", -1)) != documents
        ):
            raise ValueError(f"{source_id}: normalized document count drift")
        identities.append(
            {
                "source_id": source_id,
                "repo_id": repo_id,
                "revision": revision,
                "role": str(source.get("role", "")),
                "documents": documents,
                "shards": len(shards),
                "shard_inventory_sha256": sha256_json(shards),
                "acquisition_selected_file_count": declared_file_count,
                "acquisition_selected_bytes": declared_file_bytes,
                "acquisition_file_inventory_sha256": sha256_json(file_projection),
            }
        )
    identities.sort(key=lambda row: str(row["source_id"]))
    normalized_shards.sort(key=lambda row: (str(row["source_id"]), str(row["path"])))
    return {
        "schema_version": "full_cpt_normalization_manifest_v1",
        "manifest": _portable_file_receipt(
            manifest_path, label="normalization_manifest.json"
        ),
        "sources_config_sha256": sha256_file(sources_config_path),
        "acquisition_receipt_sha256": sha256_file(acquisition_path),
        "source_identities": identities,
        "source_identity_inventory_sha256": sha256_json(identities),
        "normalized_shard_inventory_sha256": sha256_json(normalized_shards),
        "_normalized_shards": normalized_shards,
    }


def build_quality_site_handoff(
    *,
    summary_path: Path,
    output_root: Path,
    normalization_manifest: Path,
    build_receipt: Path,
    contract_path: Path,
) -> dict[str, Any]:
    """Revalidate the full on-cluster closure and emit a compact site handoff."""

    summary = read_json(summary_path)
    projection = validate_and_project_quality_summary(summary)
    contract = read_json(contract_path)
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError("quality contract schema drift")
    contract_sha256 = sha256_json(contract)
    if (
        contract_sha256 != summary["contract_sha256"]
        or sha256_file(contract_path) != summary["contract"]["sha256"]
        or sha256_file(normalization_manifest)
        != summary["normalization_manifest"]["sha256"]
        or sha256_file(build_receipt) != summary["glossapi_build_receipt"]["sha256"]
    ):
        raise ValueError("quality summary dependency closure drift")
    if (
        contract.get("scan_mode") != summary["scan_mode"]
        or contract.get("batch_size") != summary["batch_size"]
        or contract.get("threads") != summary["threads"]
        or contract.get("quantile_sample_size") != summary["quantile_sample_size"]
        or contract.get("excluded_source_ids") != summary["excluded_source_ids"]
        or contract.get("normalization_manifest", {}).get("sha256")
        != summary["normalization_manifest"]["sha256"]
        or contract.get("build_receipt", {}).get("sha256")
        != summary["glossapi_build_receipt"]["sha256"]
        or contract.get("expected_glossapi_commit") != summary["glossapi_commit"]
        or contract.get("document_schema") != DOCUMENT_SCHEMA
        or contract.get("zero_badness_zero_greek_guard") is not True
    ):
        raise ValueError("quality contract/summary semantic drift")

    document_path = output_root / str(summary["document_output"]["path"])
    validate_file_receipt(document_path, summary["document_output"])
    checkpoints = summary["batch_checkpoints"]
    receipt_closure: list[dict[str, Any]] = []
    for item in checkpoints["inventory"]:
        receipt_path = output_root / str(item["receipt_path"])
        if sha256_file(receipt_path) != item["receipt_sha256"]:
            raise ValueError(f"checkpoint receipt drift: {receipt_path}")
        receipt_value = read_json(receipt_path)
        if (
            receipt_value.get("schema_version") != BATCH_RECEIPT_SCHEMA
            or receipt_value.get("contract_sha256") != contract_sha256
            or receipt_value.get("output", {}).get("sha256") != item["output_sha256"]
            or int(receipt_value.get("output", {}).get("rows", -1)) != item["rows"]
            or receipt_value.get("input_shard", {}).get("sha256")
            != item["input_shard_sha256"]
            or int(receipt_value.get("batch_index", -1)) != item["batch_index"]
        ):
            raise ValueError(f"checkpoint receipt semantic drift: {receipt_path}")
        output_path = receipt_path.parent / str(receipt_value["output"]["path"])
        if (
            not output_path.is_file()
            or output_path.stat().st_size != int(receipt_value["output"]["bytes"])
            or sha256_file(output_path) != item["output_sha256"]
        ):
            raise ValueError(f"checkpoint output stat drift: {output_path}")
        receipt_closure.append(
            {
                "receipt_sha256": str(item["receipt_sha256"]),
                "output_sha256": str(item["output_sha256"]),
                "rows": int(item["rows"]),
                "input_shard_sha256": str(item["input_shard_sha256"]),
                "batch_index": int(item["batch_index"]),
            }
        )
    build = read_json(build_receipt)
    if (
        build.get("schema_version") != BUILD_RECEIPT_SCHEMA
        or build.get("status") != "passed"
        or build.get("source", {}).get("commit") != summary["glossapi_commit"]
    ):
        raise ValueError("quality build receipt drift")
    cargo_locks = [
        {name: row[name] for name in ("path", "bytes", "sha256")}
        for row in build.get("source", {}).get("cargo_locks", [])
    ]
    modules = [
        {name: row[name] for name in ("name", "bytes", "sha256")}
        for row in build.get("modules", [])
    ]
    if len(cargo_locks) != 2 or len(modules) != 2:
        raise ValueError("quality build dependency inventory drift")
    normalization = normalization_identity_closure(normalization_manifest)
    normalized_shards = normalization.pop("_normalized_shards")
    summary_shards = [
        {name: row[name] for name in ("source_id", "path", "bytes", "sha256", "rows")}
        for row in summary["input_shards"]
    ]
    contract_shards = [
        {name: row[name] for name in ("source_id", "path", "bytes", "sha256", "rows")}
        for row in contract.get("selected_shards", [])
    ]
    if summary_shards != contract_shards:
        raise ValueError("quality summary/contract selected-shard inventory drift")
    normalized_by_identity = {
        (str(row["source_id"]), str(row["path"]), str(row["sha256"])): row
        for row in normalized_shards
    }
    if len(normalized_by_identity) != len(normalized_shards):
        raise ValueError("normalization shard identity collision")
    for row in contract_shards:
        normalized = normalized_by_identity.get(
            (str(row["source_id"]), str(row["path"]), str(row["sha256"]))
        )
        if normalized != row:
            raise ValueError(
                "quality contract shard is absent from normalization manifest"
            )
    selected_normalized_source_ids = sorted(
        {str(row["source_id"]) for row in contract_shards}
    )
    if (
        summary["scan_mode"] == "full_scan"
        and selected_normalized_source_ids != summary["selected_source_ids"]
    ):
        raise ValueError("full-scan selected source IDs differ from selected shards")
    normalization["selected_normalized_source_ids"] = selected_normalized_source_ids
    normalization["selected_normalized_shard_inventory_sha256"] = sha256_json(
        contract_shards
    )
    return {
        "schema_version": QUALITY_HANDOFF_SCHEMA,
        "status": "passed",
        "created_at": utc_now(),
        "summary": _portable_file_receipt(summary_path, label=summary_path.name),
        "scan_mode": str(summary["scan_mode"]),
        "aggregate_projection_sha256": sha256_json(projection),
        "normalization": normalization,
        "build": {
            "receipt_sha256": sha256_file(build_receipt),
            "commit": str(summary["glossapi_commit"]),
            "cargo_lock_inventory_sha256": sha256_json(cargo_locks),
            "module_inventory_sha256": sha256_json(modules),
            "runtime": {
                name: str(build.get("runtime", {}).get(name, ""))
                for name in (
                    "python",
                    "platform",
                    "machine",
                    "rustc",
                    "cargo",
                    "maturin",
                )
            },
        },
        "contract": {
            "receipt": _portable_file_receipt(contract_path, label="contract.json"),
            "canonical_sha256": contract_sha256,
            "schema_version": CONTRACT_SCHEMA,
            "selected_shard_inventory_sha256": sha256_json(contract_shards),
            "excluded_source_ids": list(summary["excluded_source_ids"]),
            "profiler_script_sha256": str(contract.get("profiler_script_sha256", "")),
            "review_sample": contract.get("review_sample"),
        },
        "document_output": dict(projection["document_output"]),
        "checkpoint_closure": {
            "count": int(checkpoints["count"]),
            "rows": int(checkpoints["rows"]),
            "inventory_sha256": str(checkpoints["inventory_sha256"]),
            "receipt_closure_sha256": sha256_json(receipt_closure),
            "checkpoint_outputs_rehashed_for_handoff": True,
            "consolidated_document_output_rehashed_for_handoff": True,
        },
    }


def snapshot_quality_handoff_outputs(
    summary_path: Path, output_root: Path
) -> dict[Path, InputSnapshot]:
    """Snapshot every generated artifact parsed or hashed into the handoff.

    The caller must snapshot ``summary_path`` first.  Receipt files are then
    snapshotted before they are parsed to discover their checkpoint outputs.
    """

    summary = read_json(summary_path)
    document_path = output_root / str(
        summary.get("document_output", {}).get("path", "")
    )
    inventory = summary.get("batch_checkpoints", {}).get("inventory", [])
    if not isinstance(inventory, list):
        raise ValueError("quality summary checkpoint inventory must be a list")
    receipt_paths = [
        output_root / str(row.get("receipt_path", ""))
        for row in inventory
        if isinstance(row, Mapping)
    ]
    if len(receipt_paths) != len(inventory):
        raise ValueError("quality summary checkpoint inventory contains non-objects")
    snapshots = snapshot_inputs((document_path, *receipt_paths))
    output_paths: list[Path] = []
    for receipt_path in receipt_paths:
        checkpoint = read_json(receipt_path)
        output = checkpoint.get("output")
        if not isinstance(output, Mapping) or not str(output.get("path", "")):
            raise ValueError(f"{receipt_path}: checkpoint output path missing")
        output_paths.append(receipt_path.parent / str(output["path"]))
    snapshots.update(snapshot_inputs(output_paths))
    return snapshots


def validate_quality_site_handoff(
    *, summary_path: Path, handoff_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate the compact on-cluster attestation and return safe projections."""

    summary = read_json(summary_path)
    projection = validate_and_project_quality_summary(summary)
    handoff = read_json(handoff_path)
    require_exact_keys(
        handoff,
        required=(
            "schema_version",
            "status",
            "created_at",
            "summary",
            "scan_mode",
            "aggregate_projection_sha256",
            "normalization",
            "build",
            "contract",
            "document_output",
            "checkpoint_closure",
        ),
        context="quality_handoff",
    )
    if (
        handoff["schema_version"] != QUALITY_HANDOFF_SCHEMA
        or handoff["status"] != "passed"
        or handoff["scan_mode"] != projection["scan_mode"]
        or handoff["aggregate_projection_sha256"] != sha256_json(projection)
    ):
        raise ValueError("quality_handoff: schema/status/projection drift")
    summary_receipt = validate_receipt_object(
        handoff["summary"], context="quality_handoff.summary", allow_rows=False
    )
    if (
        summary_receipt["path"] != summary_path.name
        or summary_receipt["bytes"] != summary_path.stat().st_size
        or summary_receipt["sha256"] != sha256_file(summary_path)
    ):
        raise ValueError("quality_handoff: summary receipt drift")

    normalization = handoff["normalization"]
    if not isinstance(normalization, Mapping):
        raise ValueError("quality_handoff.normalization: expected object")
    require_exact_keys(
        normalization,
        required=(
            "schema_version",
            "manifest",
            "sources_config_sha256",
            "acquisition_receipt_sha256",
            "source_identities",
            "source_identity_inventory_sha256",
            "normalized_shard_inventory_sha256",
            "selected_normalized_source_ids",
            "selected_normalized_shard_inventory_sha256",
        ),
        context="quality_handoff.normalization",
    )
    if normalization["schema_version"] != "full_cpt_normalization_manifest_v1":
        raise ValueError("quality_handoff: normalization schema drift")
    normalization_manifest_receipt = validate_receipt_object(
        normalization["manifest"],
        context="quality_handoff.normalization.manifest",
        allow_rows=False,
    )
    if (
        normalization_manifest_receipt["bytes"]
        != summary["normalization_manifest"]["bytes"]
        or normalization_manifest_receipt["sha256"]
        != summary["normalization_manifest"]["sha256"]
    ):
        raise ValueError("quality_handoff: normalization manifest receipt drift")
    for name in (
        "sources_config_sha256",
        "acquisition_receipt_sha256",
        "source_identity_inventory_sha256",
        "normalized_shard_inventory_sha256",
        "selected_normalized_shard_inventory_sha256",
    ):
        require_sha256(
            normalization[name], context=f"quality_handoff.normalization.{name}"
        )
    identities = normalization["source_identities"]
    if not isinstance(identities, list) or not identities:
        raise ValueError("quality_handoff: source identities must be nonempty")
    validated_identities: list[dict[str, Any]] = []
    for index, row in enumerate(identities):
        if not isinstance(row, Mapping):
            raise ValueError("quality_handoff: source identity must be object")
        require_exact_keys(
            row,
            required=(
                "source_id",
                "repo_id",
                "revision",
                "role",
                "documents",
                "shards",
                "shard_inventory_sha256",
                "acquisition_selected_file_count",
                "acquisition_selected_bytes",
                "acquisition_file_inventory_sha256",
            ),
            context=f"quality_handoff.source_identities[{index}]",
        )
        identity = {
            "source_id": str(row["source_id"]),
            "repo_id": str(row["repo_id"]),
            "revision": str(row["revision"]),
            "role": str(row["role"]),
            "documents": require_nonnegative_int(
                row["documents"], context="quality_handoff.identity.documents"
            ),
            "shards": require_nonnegative_int(
                row["shards"], context="quality_handoff.identity.shards"
            ),
            "shard_inventory_sha256": require_sha256(
                row["shard_inventory_sha256"],
                context="quality_handoff.identity.shard_inventory_sha256",
            ),
            "acquisition_selected_file_count": require_nonnegative_int(
                row["acquisition_selected_file_count"],
                context="quality_handoff.identity.acquisition_selected_file_count",
            ),
            "acquisition_selected_bytes": require_nonnegative_int(
                row["acquisition_selected_bytes"],
                context="quality_handoff.identity.acquisition_selected_bytes",
            ),
            "acquisition_file_inventory_sha256": require_sha256(
                row["acquisition_file_inventory_sha256"],
                context="quality_handoff.identity.acquisition_file_inventory_sha256",
            ),
        }
        if (
            not identity["source_id"]
            or not identity["repo_id"]
            or not identity["revision"]
            or identity["documents"] < 1
            or identity["shards"] < 1
            or identity["acquisition_selected_file_count"] < 1
        ):
            raise ValueError("quality_handoff: incomplete source identity")
        validated_identities.append(identity)
    if validated_identities != sorted(
        validated_identities, key=lambda row: str(row["source_id"])
    ) or len({row["source_id"] for row in validated_identities}) != len(
        validated_identities
    ):
        raise ValueError("quality_handoff: source identity ordering/uniqueness drift")
    if (
        sha256_json(validated_identities)
        != normalization["source_identity_inventory_sha256"]
    ):
        raise ValueError("quality_handoff: source identity inventory hash drift")
    selected_normalized_source_ids = normalization["selected_normalized_source_ids"]
    if (
        not isinstance(selected_normalized_source_ids, list)
        or not selected_normalized_source_ids
        or selected_normalized_source_ids != sorted(set(selected_normalized_source_ids))
        or any(
            not isinstance(source_id, str) or not source_id
            for source_id in selected_normalized_source_ids
        )
        or not set(selected_normalized_source_ids).issubset(
            {row["source_id"] for row in validated_identities}
        )
    ):
        raise ValueError("quality_handoff: selected normalized source identity drift")
    if (
        projection["scan_mode"] == "full_scan"
        and selected_normalized_source_ids != projection["selected_source_ids"]
    ):
        raise ValueError("quality_handoff: full-scan selected sources drift")

    build = handoff["build"]
    if not isinstance(build, Mapping):
        raise ValueError("quality_handoff.build: expected object")
    require_exact_keys(
        build,
        required=(
            "receipt_sha256",
            "commit",
            "cargo_lock_inventory_sha256",
            "module_inventory_sha256",
            "runtime",
        ),
        context="quality_handoff.build",
    )
    for name in (
        "receipt_sha256",
        "cargo_lock_inventory_sha256",
        "module_inventory_sha256",
    ):
        require_sha256(build[name], context=f"quality_handoff.build.{name}")
    if (
        build["receipt_sha256"] != summary["glossapi_build_receipt"]["sha256"]
        or build["commit"] != summary["glossapi_commit"]
    ):
        raise ValueError("quality_handoff: build receipt/commit drift")
    runtime = build["runtime"]
    if not isinstance(runtime, Mapping):
        raise ValueError("quality_handoff.build.runtime: expected object")
    require_exact_keys(
        runtime,
        required=("python", "platform", "machine", "rustc", "cargo", "maturin"),
        context="quality_handoff.build.runtime",
    )
    if any(not isinstance(runtime[name], str) or not runtime[name] for name in runtime):
        raise ValueError("quality_handoff: incomplete build runtime")

    contract = handoff["contract"]
    if not isinstance(contract, Mapping):
        raise ValueError("quality_handoff.contract: expected object")
    require_exact_keys(
        contract,
        required=(
            "receipt",
            "canonical_sha256",
            "schema_version",
            "selected_shard_inventory_sha256",
            "excluded_source_ids",
            "profiler_script_sha256",
            "review_sample",
        ),
        context="quality_handoff.contract",
    )
    contract_receipt = validate_receipt_object(
        contract["receipt"],
        context="quality_handoff.contract.receipt",
        allow_rows=False,
    )
    if (
        contract_receipt["path"] != "contract.json"
        or contract_receipt["bytes"] != summary["contract"]["bytes"]
        or contract_receipt["sha256"] != summary["contract"]["sha256"]
        or contract["canonical_sha256"] != summary["contract_sha256"]
        or contract["schema_version"] != CONTRACT_SCHEMA
        or contract["excluded_source_ids"] != projection["excluded_source_ids"]
        or contract["selected_shard_inventory_sha256"]
        != normalization["selected_normalized_shard_inventory_sha256"]
    ):
        raise ValueError("quality_handoff: contract closure drift")
    for name in (
        "canonical_sha256",
        "selected_shard_inventory_sha256",
        "profiler_script_sha256",
    ):
        require_sha256(contract[name], context=f"quality_handoff.contract.{name}")
    review_sample = contract["review_sample"]
    if projection["scan_mode"] == "review_sample":
        if not isinstance(review_sample, Mapping):
            raise ValueError(
                "quality_handoff: sample mode lacks review-sample contract"
            )
        require_exact_keys(
            review_sample,
            required=(
                "review_sample_packet",
                "review_sample_receipt",
                "review_sample_attestation",
                "review_requests",
                "documents",
                "text_variant",
            ),
            context="quality_handoff.contract.review_sample",
        )
        for name in (
            "review_sample_packet",
            "review_sample_receipt",
            "review_sample_attestation",
            "review_requests",
        ):
            validate_receipt_object(
                review_sample[name],
                context=f"quality_handoff.contract.review_sample.{name}",
                allow_rows=False,
            )
        if (
            require_nonnegative_int(
                review_sample["documents"],
                context="quality_handoff.contract.review_sample.documents",
            )
            != projection["documents"]
            or review_sample["text_variant"]
            != "high_precision_identifier_masked_review_sample"
        ):
            raise ValueError("quality_handoff: review-sample contract drift")
    elif review_sample is not None:
        raise ValueError(
            "quality_handoff: full scan unexpectedly has review-sample contract"
        )

    document_output = validate_receipt_object(
        handoff["document_output"],
        context="quality_handoff.document_output",
        require_rows=True,
    )
    if document_output != projection["document_output"]:
        raise ValueError("quality_handoff: document output receipt drift")
    checkpoints = handoff["checkpoint_closure"]
    if not isinstance(checkpoints, Mapping):
        raise ValueError("quality_handoff.checkpoint_closure: expected object")
    require_exact_keys(
        checkpoints,
        required=(
            "count",
            "rows",
            "inventory_sha256",
            "receipt_closure_sha256",
            "checkpoint_outputs_rehashed_for_handoff",
            "consolidated_document_output_rehashed_for_handoff",
        ),
        context="quality_handoff.checkpoint_closure",
    )
    if (
        require_nonnegative_int(
            checkpoints["count"], context="quality_handoff.checkpoints.count"
        )
        != summary["batch_checkpoints"]["count"]
        or require_nonnegative_int(
            checkpoints["rows"], context="quality_handoff.checkpoints.rows"
        )
        != projection["documents"]
        or checkpoints["inventory_sha256"] != projection["checkpoint_inventory_sha256"]
        or checkpoints["checkpoint_outputs_rehashed_for_handoff"] is not True
        or checkpoints["consolidated_document_output_rehashed_for_handoff"] is not True
    ):
        raise ValueError("quality_handoff: checkpoint closure drift")
    require_sha256(
        checkpoints["receipt_closure_sha256"],
        context="quality_handoff.checkpoint_closure.receipt_closure_sha256",
    )
    selected_identity_set = set(selected_normalized_source_ids)
    return projection, [
        row for row in validated_identities if row["source_id"] in selected_identity_set
    ]


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class InputSnapshot:
    path: Path
    bytes: int
    sha256: str
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int

    def receipt(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


def snapshot_input(path: Path) -> InputSnapshot:
    resolved = path.expanduser().resolve()
    if path.is_symlink() or not resolved.is_file():
        raise ValueError(f"input must be a regular non-symlinked file: {path}")
    before = resolved.stat()
    digest = sha256_file(resolved)
    after = resolved.stat()
    identity = (
        before.st_size,
        before.st_dev,
        before.st_ino,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if identity != (
        after.st_size,
        after.st_dev,
        after.st_ino,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError(f"input changed while hashing: {resolved}")
    return InputSnapshot(
        path=resolved,
        bytes=after.st_size,
        sha256=digest,
        device=after.st_dev,
        inode=after.st_ino,
        mtime_ns=after.st_mtime_ns,
        ctime_ns=after.st_ctime_ns,
    )


def snapshot_inputs(paths: Iterable[Path | None]) -> dict[Path, InputSnapshot]:
    result: dict[Path, InputSnapshot] = {}
    for path in paths:
        if path is None:
            continue
        snapshot = snapshot_input(path)
        result[snapshot.path] = snapshot
    return result


def verify_input_snapshots(snapshots: Mapping[Path, InputSnapshot]) -> None:
    for path, expected in snapshots.items():
        actual = snapshot_input(path)
        if actual != expected:
            raise ValueError(f"input drift before atomic publication: {path}")


def normalization_dependency_receipt_paths(manifest_path: Path) -> list[Path]:
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != "full_cpt_normalization_manifest_v1":
        raise ValueError(f"{manifest_path}: unsupported normalization manifest")
    paths = [
        Path(str(manifest.get("sources_config", ""))),
        Path(str(manifest.get("acquisition_receipt", ""))),
    ]
    contract = manifest.get("contract")
    if isinstance(contract, Mapping):
        paths.append(Path(str(contract.get("path", ""))))
    for source in manifest.get("sources", []):
        if not isinstance(source, Mapping):
            raise ValueError("normalization manifest source must be an object")
        source_receipt = source.get("receipt")
        if not isinstance(source_receipt, Mapping):
            raise ValueError("normalization source lacks receipt")
        paths.append(Path(str(source_receipt.get("path", ""))))
        for shard in source.get("shards", []):
            if not isinstance(shard, Mapping) or not isinstance(
                shard.get("receipt"), Mapping
            ):
                raise ValueError("normalization shard lacks receipt")
            paths.append(Path(str(shard["receipt"].get("path", ""))))
    if any(not str(path) for path in paths):
        raise ValueError("normalization dependency has an empty receipt path")
    return paths


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def display_document_id(value: str) -> str:
    return hashlib.sha256(
        f"dataset-review-display-id-v1\0{value}".encode("utf-8")
    ).hexdigest()[:16]


def metadata_flags(value: Any) -> tuple[bool, bool]:
    """Extract only the two source-policy flags needed by diagnostics.

    Canonical metadata is JSON, but a few adapters preserve the upstream
    object below a ``metadata_json``/``source_metadata_json`` key.  Walk only
    those known envelopes and never copy the metadata itself into diagnostics.
    """

    pending: list[Any] = [value]
    mappings: list[Mapping[str, Any]] = []
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if isinstance(current, str):
            if not current.strip():
                continue
            try:
                current = json.loads(current)
            except json.JSONDecodeError as exc:
                raise ValueError("invalid canonical source_metadata_json") from exc
        if not isinstance(current, Mapping) or id(current) in seen:
            continue
        seen.add(id(current))
        mappings.append(current)
        for key in (
            "metadata",
            "source_metadata",
            "metadata_json",
            "source_metadata_json",
        ):
            if key in current:
                pending.append(current[key])

    private = False
    corrected = False
    for current in mappings:
        flag = current.get("privateData", current.get("private_data"))
        private = (
            private
            or flag is True
            or (isinstance(flag, str) and flag.strip().casefold() == "true")
        )
        corrected_value = current.get(
            "correctedVersionId", current.get("corrected_version_id")
        )
        corrected = corrected or corrected_value not in (None, "", 0, False)
    return private, corrected


def strict_json_loads(text: str, *, context: str) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{context}: duplicate JSON key {key!r}")
            result[key] = item
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"{context}: non-finite JSON constant {value}")

    return json.loads(
        text,
        object_pairs_hook=object_pairs,
        parse_constant=reject_constant,
    )


def read_json(path: Path) -> dict[str, Any]:
    value = strict_json_loads(path.read_text(encoding="utf-8"), context=str(path))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return value


def write_json_atomic(
    path: Path, value: Mapping[str, Any], *, immutable: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        raise FileExistsError(f"refusing to overwrite immutable JSON: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def file_receipt(
    path: Path, *, relative_to: Path | None = None, rows: int | None = None
) -> dict[str, Any]:
    resolved = path.resolve()
    result: dict[str, Any] = {
        "path": resolved.relative_to(relative_to.resolve()).as_posix()
        if relative_to
        else str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def validate_file_receipt(
    path: Path, receipt: Mapping[str, Any], *, rows: int | None = None
) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(receipt.get("bytes", -1)):
        raise ValueError(f"byte-size drift for {path}")
    if sha256_file(path) != str(receipt.get("sha256", "")):
        raise ValueError(f"SHA-256 drift for {path}")
    expected_rows = rows if rows is not None else receipt.get("rows")
    if expected_rows is not None:
        import pyarrow.parquet as pq

        if pq.ParquetFile(path).metadata.num_rows != int(expected_rows):
            raise ValueError(f"row-count drift for {path}")


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, encoding="utf-8"
    ).strip()


def module_path(name: str) -> Path:
    module = importlib.import_module(name)
    value = getattr(module, "__file__", None)
    if not value:
        raise ValueError(f"{name}: imported module has no filesystem path")
    path = Path(str(value)).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def tool_version(command: str, *arguments: str) -> str:
    value = subprocess.check_output(
        [command, *arguments], text=True, encoding="utf-8", stderr=subprocess.STDOUT
    ).strip()
    if not value:
        raise ValueError(f"{command}: empty version output")
    return value


def build_runtime_receipt(args: argparse.Namespace) -> int:
    root = args.glossapi_root.resolve()
    if git_output(root, "rev-parse", "--is-inside-work-tree") != "true":
        raise ValueError(f"not a Git checkout: {root}")
    commit = git_output(root, "rev-parse", "HEAD")
    if commit != args.expected_commit:
        raise ValueError(
            f"GlossAPI checkout is {commit}, expected {args.expected_commit}"
        )
    if git_output(root, "status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("GlossAPI build receipt requires a clean checkout")

    locks: list[dict[str, Any]] = []
    for relative in (
        "rust/glossapi_rs_noise/Cargo.lock",
        "rust/glossapi_rs_cleaner/Cargo.lock",
    ):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        locks.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    modules = []
    for name in ("glossapi_rs_noise", "glossapi_rs_cleaner"):
        path = module_path(name)
        published_path = path
        if args.module_root is not None or args.published_module_root is not None:
            if args.module_root is None or args.published_module_root is None:
                raise ValueError(
                    "--module-root and --published-module-root must be supplied together"
                )
            try:
                relative = path.relative_to(args.module_root.resolve())
            except ValueError as exc:
                raise ValueError(
                    f"{name}: imported module is outside --module-root"
                ) from exc
            published_path = args.published_module_root.resolve() / relative
        modules.append(
            {
                "name": name,
                "path": str(published_path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    payload = {
        "schema_version": BUILD_RECEIPT_SCHEMA,
        "status": "passed",
        "created_at": utc_now(),
        "source": {
            "root": str(root),
            "commit": commit,
            "cargo_locks": locks,
        },
        "runtime": {
            "python": sys.version,
            "python_executable": str(Path(sys.executable).resolve()),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "rustc": tool_version("rustc", "--version", "--verbose"),
            "cargo": tool_version("cargo", "--version", "--verbose"),
            "maturin": str(args.maturin_version),
        },
        "modules": modules,
    }
    write_json_atomic(args.output, payload, immutable=True)
    print(canonical_json({"ok": True, "receipt": str(args.output.resolve())}))
    return 0


@dataclass(frozen=True)
class RustRuntime:
    noise: Any
    cleaner: Any
    receipt: dict[str, Any]
    receipt_path: Path


def validate_runtime_receipt(path: Path, expected_commit: str) -> RustRuntime:
    receipt = read_json(path)
    if (
        receipt.get("schema_version") != BUILD_RECEIPT_SCHEMA
        or receipt.get("status") != "passed"
    ):
        raise ValueError(f"{path}: unsupported or unsuccessful Rust build receipt")
    source = receipt.get("source")
    if not isinstance(source, dict) or source.get("commit") != expected_commit:
        raise ValueError(f"{path}: GlossAPI commit is not the pinned commit")
    root = Path(str(source.get("root", ""))).resolve()
    if git_output(root, "rev-parse", "HEAD") != expected_commit:
        raise ValueError(f"{path}: GlossAPI checkout commit drift")
    if git_output(root, "status", "--porcelain", "--untracked-files=normal"):
        raise ValueError(f"{path}: pinned GlossAPI checkout is no longer clean")
    for lock in source.get("cargo_locks", []):
        lock_path = root / str(lock.get("path", ""))
        validate_file_receipt(lock_path, lock)

    declared = {str(row.get("name")): row for row in receipt.get("modules", [])}
    loaded: dict[str, Any] = {}
    for name in ("glossapi_rs_noise", "glossapi_rs_cleaner"):
        if name not in declared:
            raise ValueError(f"{path}: missing module receipt for {name}")
        module = importlib.import_module(name)
        actual_path = module_path(name)
        expected_path = Path(str(declared[name].get("path", ""))).resolve()
        if actual_path != expected_path:
            raise ValueError(f"{name}: imported module path differs from build receipt")
        validate_file_receipt(actual_path, declared[name])
        loaded[name] = module
    return RustRuntime(
        noise=loaded["glossapi_rs_noise"],
        cleaner=loaded["glossapi_rs_cleaner"],
        receipt=receipt,
        receipt_path=path.resolve(),
    )


def validate_runtime_receipt_command(args: argparse.Namespace) -> int:
    runtime = validate_runtime_receipt(args.receipt, args.expected_commit)
    print(
        canonical_json(
            {
                "ok": True,
                "receipt": str(runtime.receipt_path),
                "commit": args.expected_commit,
                "modules": [
                    str(module.get("path"))
                    for module in runtime.receipt.get("modules", [])
                ],
            }
        )
    )
    return 0


@dataclass(frozen=True)
class ShardBinding:
    source_id: str
    path: Path
    relative_path: str
    bytes: int
    sha256: str
    rows: int

    def receipt(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "path": self.relative_path,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "rows": self.rows,
        }


def _safe_under(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path escapes canonical root: {path}") from exc


def load_normalized_shards(
    manifest_path: Path,
    canonical_root: Path,
    *,
    include_source_ids: set[str],
    include_base: bool,
) -> tuple[dict[str, Any], list[ShardBinding], list[str]]:
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != "full_cpt_normalization_manifest_v1":
        raise ValueError(f"{manifest_path}: unsupported normalization manifest")
    if Path(str(manifest.get("output", ""))).resolve() != canonical_root.resolve():
        raise ValueError("normalization manifest output root drift")

    all_declared: set[Path] = set()
    selected: list[ShardBinding] = []
    excluded: list[str] = []
    seen_sources: set[str] = set()
    for source in manifest.get("sources", []):
        if not isinstance(source, dict):
            raise ValueError("normalization source entry must be an object")
        source_id = str(source.get("source_id", ""))
        if not source_id or source_id in seen_sources:
            raise ValueError(f"duplicate or empty normalized source_id: {source_id!r}")
        seen_sources.add(source_id)
        wanted = (include_base or source_id != "nanochat_base") and (
            not include_source_ids or source_id in include_source_ids
        )
        if not wanted:
            excluded.append(source_id)
        for row in source.get("shards", []):
            if not isinstance(row, dict):
                raise ValueError(f"{source_id}: shard receipt must be an object")
            path = Path(str(row.get("path", ""))).resolve()
            relative = _safe_under(path, canonical_root)
            if path in all_declared:
                raise ValueError(f"duplicate normalized shard: {path}")
            all_declared.add(path)
            binding = ShardBinding(
                source_id=source_id,
                path=path,
                relative_path=relative,
                bytes=int(row.get("bytes", -1)),
                sha256=str(row.get("sha256", "")),
                rows=int(row.get("rows", -1)),
            )
            if (
                binding.bytes < 1
                or binding.rows < 1
                or not SHA256_RE.fullmatch(binding.sha256)
            ):
                raise ValueError(
                    f"{source_id}: invalid normalized shard receipt for {path}"
                )
            if wanted:
                selected.append(binding)

    actual = {
        path.resolve() for path in canonical_root.rglob("*.parquet") if path.is_file()
    }
    if actual != all_declared:
        missing = sorted(str(path) for path in all_declared - actual)
        unexpected = sorted(str(path) for path in actual - all_declared)
        raise ValueError(
            f"canonical Parquet inventory differs from manifest; missing={missing[:10]}, "
            f"unexpected={unexpected[:10]}"
        )
    if include_source_ids - seen_sources:
        raise ValueError(
            f"unknown requested source IDs: {sorted(include_source_ids - seen_sources)}"
        )
    if not selected:
        raise ValueError("no normalized shards selected for Rust diagnostics")
    return (
        manifest,
        sorted(selected, key=lambda row: (row.source_id, row.relative_path)),
        sorted(excluded),
    )


def load_review_sample_packet(
    *,
    packet_path: Path,
    receipt_path: Path,
    attestation_path: Path,
    requests_path: Path,
    normalization_manifest: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate the exact redacted review sample selected by Stage 30."""

    receipt_value = read_json(receipt_path)
    if (
        receipt_value.get("schema_version")
        != "dataset_review_complete_sample_packet_receipt_v1"
        or receipt_value.get("status") != "passed"
        or receipt_value.get("high_precision_identifier_patterns_masked") is not True
    ):
        raise ValueError(
            f"{receipt_path}: unsupported or incomplete review sample receipt"
        )
    require_exact_keys(
        receipt_value,
        required=(
            "schema_version",
            "status",
            "normalization_manifest",
            "canonical_root",
            "review_requests",
            "export_contract",
            "site_attestation",
            "input_shards",
            "checkpoint_inventory",
            "checkpoint_inventory_sha256",
            "output",
            "redaction_totals",
            "high_precision_identifier_patterns_masked",
        ),
        context="review_sample_receipt",
    )
    output = receipt_value.get("output")
    if not isinstance(output, dict):
        raise ValueError(f"{receipt_path}: missing sample output receipt")
    declared_packet = Path(str(output.get("path", "")))
    if not declared_packet.is_absolute():
        declared_packet = receipt_path.resolve().parent / declared_packet
    if declared_packet.resolve() != packet_path.resolve():
        raise ValueError(f"{receipt_path}: review sample packet path drift")
    validate_file_receipt(
        packet_path,
        {name: output.get(name) for name in ("path", "bytes", "sha256")},
    )
    if receipt_value.get("normalization_manifest", {}).get("sha256") != sha256_file(
        normalization_manifest
    ) or receipt_value.get("review_requests", {}).get("sha256") != sha256_file(
        requests_path
    ):
        raise ValueError(f"{receipt_path}: review sample upstream receipt drift")

    attestation_receipt = validate_receipt_object(
        receipt_value["site_attestation"],
        context="review_sample_receipt.site_attestation",
        allow_rows=False,
    )
    declared_attestation = Path(attestation_receipt["path"])
    if not declared_attestation.is_absolute():
        declared_attestation = receipt_path.resolve().parent / declared_attestation
    if declared_attestation.resolve() != attestation_path.resolve():
        raise ValueError(f"{receipt_path}: review sample attestation path drift")
    validate_file_receipt(attestation_path, attestation_receipt)
    attestation = read_json(attestation_path)
    require_exact_keys(
        attestation,
        required=(
            "schema_version",
            "status",
            "created_at",
            "packet",
            "review_requests",
            "primary_sample_count",
            "primary_sample_id_inventory_sha256",
            "normalization",
            "export_contract",
            "checkpoint_closure",
            "masking",
        ),
        context="review_sample_attestation",
    )
    if (
        attestation["schema_version"]
        != "dataset_review_complete_sample_site_attestation_v1"
        or attestation["status"] != "passed"
    ):
        raise ValueError(f"{attestation_path}: unsupported sample attestation")
    packet_receipt = validate_receipt_object(
        attestation["packet"],
        context="review_sample_attestation.packet",
        require_rows=True,
    )
    requests_receipt = validate_receipt_object(
        attestation["review_requests"],
        context="review_sample_attestation.review_requests",
        allow_rows=False,
    )
    if (
        packet_receipt
        != validate_receipt_object(
            output,
            context="review_sample_receipt.output",
            require_rows=True,
        )
        or requests_receipt["bytes"] != requests_path.stat().st_size
        or requests_receipt["sha256"] != sha256_file(requests_path)
        or int(attestation["primary_sample_count"]) != int(output.get("rows", -1))
    ):
        raise ValueError(f"{attestation_path}: packet/request coverage drift")
    normalization = attestation["normalization"]
    checkpoint_closure = attestation["checkpoint_closure"]
    masking = attestation["masking"]
    if (
        not isinstance(normalization, Mapping)
        or normalization.get("manifest", {}).get("sha256")
        != sha256_file(normalization_manifest)
        or not isinstance(checkpoint_closure, Mapping)
        or checkpoint_closure.get("checkpoint_text_outputs_rehashed_for_attestation")
        is not True
        or int(checkpoint_closure.get("selected_rows", -1))
        != int(output.get("rows", -1))
        or not isinstance(masking, Mapping)
        or masking.get("pipeline") != "high_precision_identifier_patterns_v1"
        or masking.get("high_precision_identifier_patterns_masked") is not True
        or int(masking.get("private_data_true_rows", -1)) != 0
    ):
        raise ValueError(f"{attestation_path}: incomplete sample dependency closure")

    requested: dict[str, dict[str, str]] = {}
    with requests_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = strict_json_loads(line, context=f"{requests_path}:{line_number}")
            if (
                not isinstance(row, dict)
                or row.get("schema_version") != "source_quality_review_request_v1"
            ):
                raise ValueError(
                    f"{requests_path}:{line_number}: unsupported review request"
                )
            if row.get("reviewer_slot") != "primary":
                continue
            source = row.get("source")
            sample_id = str(row.get("sample_id", ""))
            if not isinstance(source, dict) or not SHA256_RE.fullmatch(sample_id):
                raise ValueError(
                    f"{requests_path}:{line_number}: invalid review sample"
                )
            if sample_id in requested:
                raise ValueError(
                    f"{requests_path}:{line_number}: duplicate primary sample"
                )
            requested[sample_id] = {
                "source_id": str(source.get("source_id", "")),
                "source_dataset": str(row.get("source_dataset", "")),
                "source_repo_id": str(source.get("source_repo_id", "")),
                "source_revision": str(source.get("source_revision", "")),
                "display_document_id": display_document_id(
                    str(source.get("source_doc_id", ""))
                ),
            }
    if not requested:
        raise ValueError(f"{requests_path}: no primary review samples")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with packet_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = strict_json_loads(line, context=f"{packet_path}:{line_number}")
            if (
                not isinstance(row, dict)
                or row.get("schema_version") != "dataset_review_complete_sample_v1"
            ):
                raise ValueError(f"{packet_path}:{line_number}: unsupported sample row")
            sample_id = str(row.get("sample_id", ""))
            if sample_id not in requested or sample_id in seen:
                raise ValueError(
                    f"{packet_path}:{line_number}: unknown or duplicate sample"
                )
            if row.get("high_precision_identifier_patterns_masked") is not True:
                raise ValueError(
                    f"{packet_path}:{line_number}: sample lacks the required identifier-pattern masking"
                )
            if row.get("private_data_true") is not False or not isinstance(
                row.get("corrected_version_present"), bool
            ):
                raise ValueError(
                    f"{packet_path}:{line_number}: sample metadata flags are invalid/private"
                )
            actual = {
                name: str(row.get(name, ""))
                for name in (
                    "source_id",
                    "source_dataset",
                    "source_repo_id",
                    "source_revision",
                    "display_document_id",
                )
            }
            if actual != requested[sample_id]:
                raise ValueError(
                    f"{packet_path}:{line_number}: request/sample source identity drift"
                )
            text = row.get("text")
            normalized_sha = str(row.get("normalized_text_sha256", ""))
            profile_sha = str(row.get("profile_text_sha256", ""))
            input_sha = str(row.get("input_shard_sha256", ""))
            if (
                not isinstance(text, str)
                or not SHA256_RE.fullmatch(normalized_sha)
                or not SHA256_RE.fullmatch(profile_sha)
                or hashlib.sha256(text.encode("utf-8")).hexdigest() != profile_sha
                or not SHA256_RE.fullmatch(input_sha)
                or int(row.get("input_row_index", -1)) < 0
            ):
                raise ValueError(
                    f"{packet_path}:{line_number}: invalid sample text/input binding"
                )
            rows.append(
                {
                    "source_id": actual["source_id"],
                    "source_dataset": actual["source_dataset"],
                    "source_repo_id": actual["source_repo_id"],
                    "source_revision": actual["source_revision"],
                    "stable_uid": sample_id,
                    "normalized_text_sha256": normalized_sha,
                    "profile_text_sha256": profile_sha,
                    "profile_text_variant": (
                        "high_precision_identifier_masked_review_sample"
                    ),
                    "input_shard_path": str(row.get("input_shard_path", "")),
                    "input_shard_sha256": input_sha,
                    "input_row_index": int(row["input_row_index"]),
                    "private_data_true": False,
                    "corrected_version_present": row["corrected_version_present"],
                    "text": text,
                }
            )
            seen.add(sample_id)
    if seen != set(requested) or len(rows) != int(output.get("rows", -1)):
        raise ValueError(
            f"review sample coverage mismatch; missing={sorted(set(requested) - seen)[:20]}, "
            f"unexpected={sorted(seen - set(requested))[:20]}"
        )
    input_shards = receipt_value.get("input_shards")
    if not isinstance(input_shards, list) or not input_shards:
        raise ValueError(
            f"{receipt_path}: review sample receipt lacks canonical input shards"
        )
    if (
        attestation["primary_sample_id_inventory_sha256"]
        != sha256_json(sorted(requested))
        or normalization.get("input_shards") != input_shards
        or normalization.get("input_shard_inventory_sha256")
        != sha256_json(input_shards)
        or checkpoint_closure.get("count")
        != len(receipt_value.get("checkpoint_inventory", []))
        or checkpoint_closure.get("inventory_sha256")
        != receipt_value.get("checkpoint_inventory_sha256")
        or masking.get("redaction_totals") != receipt_value.get("redaction_totals")
        or masking.get("redaction_totals_sha256")
        != sha256_json(receipt_value.get("redaction_totals"))
    ):
        raise ValueError(f"{attestation_path}: sample attestation semantic drift")
    return sorted(rows, key=lambda row: str(row["stable_uid"])), [
        dict(row) for row in input_shards if isinstance(row, dict)
    ]


def document_schema():
    import pyarrow as pa

    fields: list[tuple[str, Any]] = [
        ("schema_version", pa.string()),
        ("source_id", pa.string()),
        ("source_dataset", pa.string()),
        ("source_repo_id", pa.string()),
        ("source_revision", pa.string()),
        ("document_id", pa.string()),
        ("normalized_text_sha256", pa.string()),
        ("profile_text_sha256", pa.string()),
        ("profile_text_variant", pa.string()),
        ("input_shard_path", pa.string()),
        ("input_shard_sha256", pa.string()),
        ("input_row_index", pa.int64()),
        ("original_characters", pa.int64()),
        ("original_bytes_utf8", pa.int64()),
        ("original_non_whitespace_characters", pa.int64()),
        ("raw_greek_letters", pa.int64()),
        ("raw_latin_letters", pa.int64()),
        ("raw_greek_letter_fraction", pa.float64()),
        ("raw_html_tags", pa.int64()),
        ("raw_html_tags_per_1000_chars", pa.float64()),
        ("raw_mojibake_markers", pa.int64()),
        ("raw_replacement_characters", pa.int64()),
        ("raw_mojibake_per_1000_chars", pa.float64()),
        ("raw_replacement_per_1000_chars", pa.float64()),
        ("raw_control_characters", pa.int64()),
        ("raw_control_per_1000_chars", pa.float64()),
        ("raw_nonempty_lines", pa.int64()),
        ("raw_unique_line_fraction", pa.float64()),
        ("raw_repeated_line_fraction", pa.float64()),
        ("raw_one_token_line_fraction", pa.float64()),
        ("raw_markdown_table_lines", pa.int64()),
        ("bibliography_header_detected", pa.bool_()),
        ("toc_header_detected", pa.bool_()),
        ("digital_governance_footer_detected", pa.bool_()),
        ("personnel_cue_detected", pa.bool_()),
        ("isolated_ada_stamp_lines", pa.int64()),
        ("private_data_true", pa.bool_()),
        ("corrected_version_present", pa.bool_()),
        ("structural_template_id", pa.string()),
        ("direct_identifier_match_count", pa.int64()),
        ("direct_identifier_types", pa.string()),
    ]
    for name in NOISE_FIELDS:
        if name in FLOAT_NOISE_FIELDS:
            dtype = pa.float64()
        elif name in INTEGER_NOISE_FIELDS:
            dtype = pa.int64()
        else:
            dtype = pa.string()
        fields.append((name, dtype))
    fields.extend(
        [
            ("cleaner_badness_score", pa.float64()),
            ("cleaner_greek_percentage", pa.float64()),
            ("cleaner_latin_percentage", pa.float64()),
            ("cleaner_characters_no_comments", pa.int64()),
            ("cleaner_is_empty", pa.bool_()),
            ("cleaner_retained_character_ratio", pa.float64()),
            ("cleaner_removed_character_fraction", pa.float64()),
            ("zero_badness_zero_greek_guard", pa.bool_()),
            ("noise_score_interpretation", pa.string()),
        ]
    )
    return pa.schema(fields)


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def raw_metrics(
    text: str,
    *,
    private_data_true: bool = False,
    corrected_version_present: bool = False,
) -> dict[str, Any]:
    characters = len(text)
    greek = len(GREEK_RE.findall(text))
    latin = len(LATIN_RE.findall(text))
    letters = greek + latin
    html_tags = len(HTML_RE.findall(text))
    mojibake = len(MOJIBAKE_RE.findall(text))
    replacement = text.count("\ufffd")
    control = len(CONTROL_RE.findall(text))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    repeated = len(lines) - len(set(lines))
    unique_fraction, one_token_fraction, markdown_table_lines = line_quality(text)
    template = normalized_template(text)
    pii_counts = {name: len(pattern.findall(text)) for name, pattern in PII_PATTERNS}
    pii_counts = {name: count for name, count in pii_counts.items() if count}
    denominator = max(characters, 1)
    return {
        "original_characters": characters,
        "original_bytes_utf8": len(text.encode("utf-8")),
        "original_non_whitespace_characters": sum(not char.isspace() for char in text),
        "raw_greek_letters": greek,
        "raw_latin_letters": latin,
        "raw_greek_letter_fraction": greek / letters if letters else 0.0,
        "raw_html_tags": html_tags,
        "raw_html_tags_per_1000_chars": html_tags * 1000.0 / denominator,
        "raw_mojibake_markers": mojibake,
        "raw_replacement_characters": replacement,
        "raw_mojibake_per_1000_chars": mojibake * 1000.0 / denominator,
        "raw_replacement_per_1000_chars": replacement * 1000.0 / denominator,
        "raw_control_characters": control,
        "raw_control_per_1000_chars": control * 1000.0 / denominator,
        "raw_nonempty_lines": len(lines),
        "raw_unique_line_fraction": unique_fraction,
        "raw_repeated_line_fraction": repeated / len(lines) if lines else 0.0,
        "raw_one_token_line_fraction": one_token_fraction,
        "raw_markdown_table_lines": markdown_table_lines,
        "bibliography_header_detected": bool(BIB_HEADER.search(text)),
        "toc_header_detected": bool(TOC_HEADER.search(text)),
        "digital_governance_footer_detected": bool(DIGITAL_GOVERNANCE.search(text)),
        "personnel_cue_detected": bool(PERSONNEL_CUE.search(text)),
        "isolated_ada_stamp_lines": sum(
            bool(ADA_LINE.fullmatch(line)) for line in lines
        ),
        "private_data_true": private_data_true,
        "corrected_version_present": corrected_version_present,
        "structural_template_id": (
            hashlib.sha256(template.encode("utf-8")).hexdigest() if template else ""
        ),
        "direct_identifier_match_count": sum(pii_counts.values()),
        "direct_identifier_types": ",".join(sorted(pii_counts)),
    }


def parse_noise_rows(
    rows: Iterable[Sequence[Any]], expected: set[str]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for values in rows:
        if len(values) != len(NOISE_FIELDS) + 1:
            raise ValueError(
                f"unexpected glossapi_rs_noise detailed tuple length: {len(values)}"
            )
        key = Path(str(values[0])).stem
        if key in result:
            raise ValueError(f"duplicate Rust noise result: {key}")
        metrics: dict[str, Any] = {}
        for name, value in zip(NOISE_FIELDS, values[1:], strict=True):
            if name in FLOAT_NOISE_FIELDS:
                metrics[name] = _finite_float(value)
            elif name in INTEGER_NOISE_FIELDS:
                metrics[name] = int(value)
            else:
                metrics[name] = str(value)
        result[key] = metrics
    if set(result) != expected:
        raise ValueError(
            f"Rust noise coverage mismatch; missing={sorted(expected - set(result))[:10]}, "
            f"unexpected={sorted(set(result) - expected)[:10]}"
        )
    return result


def parse_cleaner_report(path: Path, expected: set[str]) -> dict[str, dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    required = {
        "file_name",
        "badness_score_all_chars",
        "percentage_greek_cleaned",
        "percentage_latin_cleaned",
        "char_count_no_comments",
        "is_empty",
    }
    if not required.issubset(table.column_names):
        raise ValueError(
            f"Rust cleaner report lacks columns: {sorted(required - set(table.column_names))}"
        )
    result: dict[str, dict[str, Any]] = {}
    for row in table.to_pylist():
        key = Path(str(row["file_name"])).stem
        if key in result:
            raise ValueError(f"duplicate Rust cleaner result: {key}")
        result[key] = {
            "cleaner_badness_score": _finite_float(row["badness_score_all_chars"]),
            "cleaner_greek_percentage": _finite_float(row["percentage_greek_cleaned"]),
            "cleaner_latin_percentage": _finite_float(row["percentage_latin_cleaned"]),
            "cleaner_characters_no_comments": int(row["char_count_no_comments"]),
            "cleaner_is_empty": bool(row["is_empty"]),
        }
    if set(result) != expected:
        raise ValueError(
            f"Rust cleaner coverage mismatch; missing={sorted(expected - set(result))[:10]}, "
            f"unexpected={sorted(set(result) - expected)[:10]}"
        )
    return result


def batch_directory(output: Path, shard: ShardBinding, batch_index: int) -> Path:
    source_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", shard.source_id)
    shard_key = hashlib.sha256(shard.relative_path.encode("utf-8")).hexdigest()[:12]
    return (
        output
        / "batches"
        / source_slug
        / f"{shard_key}-{shard.sha256[:12]}"
        / f"batch-{batch_index:06d}"
    )


def validate_batch_checkpoint(
    directory: Path,
    *,
    contract_sha256: str,
    shard: ShardBinding,
    batch_index: int,
    row_start: int,
    row_end: int,
) -> dict[str, Any]:
    receipt_path = directory / "receipt.json"
    output_path = directory / "documents.parquet"
    if not receipt_path.is_file() or not output_path.is_file():
        raise ValueError(f"incomplete Rust quality checkpoint: {directory}")
    receipt = read_json(receipt_path)
    expected = {
        "schema_version": BATCH_RECEIPT_SCHEMA,
        "contract_sha256": contract_sha256,
        "input_shard": shard.receipt(),
        "batch_index": batch_index,
        "row_start": row_start,
        "row_end_exclusive": row_end,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"{receipt_path}: checkpoint drift for {key}")
    output = receipt.get("output")
    if not isinstance(output, dict) or output.get("path") != "documents.parquet":
        raise ValueError(f"{receipt_path}: invalid output receipt")
    validate_file_receipt(output_path, output, rows=row_end - row_start)
    return {**receipt, "receipt": file_receipt(receipt_path)}


def process_batch(
    *,
    rows: list[dict[str, Any]],
    shard: ShardBinding,
    batch_index: int,
    row_start: int,
    output_root: Path,
    scratch_root: Path,
    contract_sha256: str,
    runtime: RustRuntime,
    threads: int,
) -> dict[str, Any]:
    final = batch_directory(output_root, shard, batch_index)
    row_end = row_start + len(rows)
    if final.exists():
        return validate_batch_checkpoint(
            final,
            contract_sha256=contract_sha256,
            shard=shard,
            batch_index=batch_index,
            row_start=row_start,
            row_end=row_end,
        )

    final.parent.mkdir(parents=True, exist_ok=True)
    partial = final.parent / f".{final.name}.partial-{os.getpid()}"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir()
    try:
        with tempfile.TemporaryDirectory(
            prefix="glossapi-rust-quality-", dir=scratch_root
        ) as raw_temp:
            temporary = Path(raw_temp)
            markdown = temporary / "markdown"
            cleaned = temporary / "cleaned-not-persisted"
            cleaner_report = temporary / "cleaner_metrics.parquet"
            markdown.mkdir()

            mapping: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
            for offset, row in enumerate(rows):
                key = f"d{offset:07d}"
                uid = str(row.get("stable_uid", ""))
                text = "" if row.get("text") is None else str(row["text"])
                if not SHA256_RE.fullmatch(uid):
                    raise ValueError(
                        f"{shard.path}:{row_start + offset}: invalid stable_uid"
                    )
                text_sha = str(row.get("normalized_text_sha256", ""))
                profile_text_sha = str(row.get("profile_text_sha256") or text_sha)
                actual_profile_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if (
                    not SHA256_RE.fullmatch(text_sha)
                    or profile_text_sha != actual_profile_sha
                ):
                    raise ValueError(f"{uid}: profile text hash drift")
                if not isinstance(
                    row.get("private_data_true", False), bool
                ) or not isinstance(row.get("corrected_version_present", False), bool):
                    raise ValueError(f"{uid}: invalid source metadata flags")
                if (
                    row.get("private_data_true") is True
                    and row.get("profile_text_variant") != "canonical"
                ):
                    raise ValueError(
                        f"{uid}: privateData=true is forbidden in review samples"
                    )
                (markdown / f"{key}.md").write_text(text, encoding="utf-8")
                mapping[key] = (
                    {**row, "_profile_text_sha256": profile_text_sha},
                    raw_metrics(
                        text,
                        private_data_true=bool(row.get("private_data_true", False)),
                        corrected_version_present=bool(
                            row.get("corrected_version_present", False)
                        ),
                    ),
                )

            expected = set(mapping)
            noise_rows = runtime.noise.score_markdown_directory_detailed(
                str(markdown), threads
            )
            noise = parse_noise_rows(noise_rows, expected)
            runtime.cleaner.run_complete_pipeline(
                str(markdown),
                str(cleaned),
                str(cleaner_report),
                ["greek", "latin"],
                threads,
                False,
            )
            cleaner = parse_cleaner_report(cleaner_report, expected)

            documents: list[dict[str, Any]] = []
            for offset, key in enumerate(sorted(mapping)):
                source, raw = mapping[key]
                noise_values = noise[key]
                cleaner_values = cleaner[key]
                original_non_ws = int(raw["original_non_whitespace_characters"])
                retained = int(cleaner_values["cleaner_characters_no_comments"])
                retained_ratio = (
                    retained / original_non_ws
                    if original_non_ws
                    else (0.0 if retained == 0 else 1.0)
                )
                removed_fraction = max(0.0, min(1.0, 1.0 - retained_ratio))
                noise_score = noise_values.get("rust_noise_badness_score")
                if noise_score is None:
                    raise ValueError(
                        f"{source['stable_uid']}: Rust noise score is non-finite"
                    )
                zero_guard = (
                    float(noise_score) == 0.0
                    and int(noise_values["rust_noise_greek_characters"]) == 0
                )
                if zero_guard:
                    interpretation = "guarded_zero_score_without_greek"
                elif float(noise_score) == 0.0:
                    interpretation = "zero_score_with_greek"
                else:
                    interpretation = "scored"
                documents.append(
                    {
                        "schema_version": DOCUMENT_SCHEMA,
                        "source_id": str(source["source_id"]),
                        "source_dataset": str(source["source_dataset"]),
                        "source_repo_id": str(source["source_repo_id"]),
                        "source_revision": str(source["source_revision"]),
                        "document_id": hashlib.sha256(
                            (
                                "dataset-quality-document-v1\0"
                                + str(source["stable_uid"])
                            ).encode("utf-8")
                        ).hexdigest(),
                        "normalized_text_sha256": str(source["normalized_text_sha256"]),
                        "profile_text_sha256": str(source["_profile_text_sha256"]),
                        "profile_text_variant": str(
                            source.get("profile_text_variant") or "canonical"
                        ),
                        "input_shard_path": str(
                            source.get("input_shard_path") or shard.relative_path
                        ),
                        "input_shard_sha256": str(
                            source.get("input_shard_sha256") or shard.sha256
                        ),
                        "input_row_index": int(
                            source.get("input_row_index", row_start + offset)
                        ),
                        **raw,
                        **noise_values,
                        **cleaner_values,
                        "cleaner_retained_character_ratio": retained_ratio,
                        "cleaner_removed_character_fraction": removed_fraction,
                        "zero_badness_zero_greek_guard": zero_guard,
                        "noise_score_interpretation": interpretation,
                    }
                )

        import pyarrow as pa
        import pyarrow.parquet as pq

        output_path = partial / "documents.parquet"
        table = pa.Table.from_pylist(documents, schema=document_schema())
        pq.write_table(
            table, output_path, compression="zstd", row_group_size=len(documents)
        )
        output_receipt = file_receipt(
            output_path, relative_to=partial, rows=len(documents)
        )
        receipt = {
            "schema_version": BATCH_RECEIPT_SCHEMA,
            "contract_sha256": contract_sha256,
            "input_shard": shard.receipt(),
            "batch_index": batch_index,
            "row_start": row_start,
            "row_end_exclusive": row_end,
            "rust_build_receipt_sha256": sha256_file(runtime.receipt_path),
            "scripts_to_keep": ["greek", "latin"],
            "write_cleaned_files": False,
            "threads": threads,
            "output": output_receipt,
        }
        write_json_atomic(partial / "receipt.json", receipt, immutable=True)
        os.replace(partial, final)
        return {**receipt, "receipt": file_receipt(final / "receipt.json")}
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


@dataclass
class ExactMetric:
    count: int = 0
    total: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def add(self, value: Any) -> None:
        if value is None:
            return
        number = float(value)
        if not math.isfinite(number):
            return
        self.count += 1
        self.total += number
        self.minimum = number if self.minimum is None else min(self.minimum, number)
        self.maximum = number if self.maximum is None else max(self.maximum, number)


@dataclass
class GroupStats:
    reservoir_size: int
    rows: int = 0
    characters: int = 0
    bytes_utf8: int = 0
    datasets: set[str] = field(default_factory=set)
    counters: Counter[str] = field(default_factory=Counter)
    templates: Counter[str] = field(default_factory=Counter)
    exact: dict[str, ExactMetric] = field(
        default_factory=lambda: {name: ExactMetric() for name in DISTRIBUTION_METRICS}
    )
    reservoir: list[tuple[int, str, dict[str, float | None]]] = field(
        default_factory=list
    )

    def add(self, row: Mapping[str, Any]) -> None:
        self.rows += 1
        self.characters += int(row["original_characters"])
        self.bytes_utf8 += int(row["original_bytes_utf8"])
        self.datasets.add(str(row["source_dataset"]))
        for name in DISTRIBUTION_METRICS:
            self.exact[name].add(row.get(name))
        flags = {
            "empty_input_documents": int(row["original_characters"]) == 0,
            "html_documents": int(row["raw_html_tags"]) > 0,
            "mojibake_documents": int(row["raw_mojibake_markers"]) > 0,
            "replacement_character_documents": int(row["raw_replacement_characters"])
            > 0,
            "control_character_documents": int(row["raw_control_characters"]) > 0,
            "low_unique_line_fraction_documents": float(row["raw_unique_line_fraction"])
            < 0.50,
            "one_token_per_line_documents": float(row["raw_one_token_line_fraction"])
            > 0.50,
            "markdown_table_documents": int(row["raw_markdown_table_lines"]) > 0,
            "large_markdown_table_documents": int(row["raw_markdown_table_lines"])
            >= 20,
            "bibliography_header_documents": bool(row["bibliography_header_detected"]),
            "toc_header_documents": bool(row["toc_header_detected"]),
            "digital_governance_footer_documents": bool(
                row["digital_governance_footer_detected"]
            ),
            "personnel_cue_documents": bool(row["personnel_cue_detected"]),
            "isolated_ada_stamp_documents": int(row["isolated_ada_stamp_lines"]) > 0,
            "private_data_true_documents": bool(row["private_data_true"]),
            "corrected_version_documents": bool(row["corrected_version_present"]),
            "direct_identifier_documents": int(row["direct_identifier_match_count"])
            > 0,
            "cleaner_empty_documents": bool(row["cleaner_is_empty"]),
            "zero_badness_zero_greek_guard_documents": bool(
                row["zero_badness_zero_greek_guard"]
            ),
        }
        for name, enabled in flags.items():
            if enabled:
                self.counters[name] += 1
        if set(flags) != set(DOCUMENT_COUNTERS):
            raise AssertionError("document counter registry drift")
        template_id = str(row.get("structural_template_id", ""))
        if template_id:
            self.templates[template_id] += 1
        uid = str(row["document_id"])
        rank = int.from_bytes(
            hashlib.sha256(f"quality-reservoir-v1\0{uid}".encode()).digest(), "big"
        )
        sampled = {name: _finite_float(row.get(name)) for name in DISTRIBUTION_METRICS}
        entry = (-rank, uid, sampled)
        if len(self.reservoir) < self.reservoir_size:
            heapq.heappush(self.reservoir, entry)
        elif rank < -self.reservoir[0][0]:
            heapq.heapreplace(self.reservoir, entry)

    def finish(self, *, repo_id: str | None = None) -> dict[str, Any]:
        distributions: dict[str, Any] = {}
        for name in DISTRIBUTION_METRICS:
            metric = self.exact[name]
            values = sorted(
                float(row[name])
                for _, _, row in self.reservoir
                if row.get(name) is not None
            )

            def quantile(fraction: float) -> float | None:
                if not values:
                    return None
                position = fraction * (len(values) - 1)
                lower = int(math.floor(position))
                upper = int(math.ceil(position))
                if lower == upper:
                    return values[lower]
                weight = position - lower
                return values[lower] * (1.0 - weight) + values[upper] * weight

            distributions[name] = {
                "count": metric.count,
                "min": metric.minimum,
                "mean": metric.total / metric.count if metric.count else None,
                "p10_approx": quantile(0.10),
                "p50_approx": quantile(0.50),
                "p90_approx": quantile(0.90),
                "p99_approx": quantile(0.99),
                "max": metric.maximum,
                "quantile_sample_documents": len(values),
            }
        document_counts = {
            name: int(self.counters.get(name, 0)) for name in DOCUMENT_COUNTERS
        }
        result: dict[str, Any] = {
            "documents": self.rows,
            "characters": self.characters,
            "bytes_utf8": self.bytes_utf8,
            "source_datasets": sorted(self.datasets),
            "document_counts": dict(sorted(document_counts.items())),
            "document_rates": {
                name.removesuffix("_documents") + "_rate": count / self.rows
                if self.rows
                else 0.0
                for name, count in sorted(document_counts.items())
            },
            "distributions": distributions,
            "template_concentration": {
                "documents_with_template": sum(self.templates.values()),
                "unique_templates": len(self.templates),
                "top_1_fraction": (
                    max(self.templates.values()) / self.rows
                    if self.rows and self.templates
                    else 0.0
                ),
                "top_10_fraction": (
                    sum(count for _, count in self.templates.most_common(10))
                    / self.rows
                    if self.rows
                    else 0.0
                ),
            },
        }
        if repo_id is not None:
            result["repo_id"] = repo_id
        return result


def consolidate_batches(
    batch_receipts: Sequence[Mapping[str, Any]],
    *,
    output_root: Path,
    reservoir_size: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    import pyarrow.parquet as pq

    final = output_root / f"{DOCUMENT_SCHEMA}.parquet"
    temporary = output_root / f".{final.name}.partial-{os.getpid()}"
    temporary.unlink(missing_ok=True)
    writer = None
    groups: dict[str, GroupStats] = defaultdict(lambda: GroupStats(reservoir_size))
    global_group = GroupStats(reservoir_size)
    rows = 0
    try:
        for receipt in sorted(
            batch_receipts,
            key=lambda row: (
                str(row["input_shard"]["source_id"]),
                str(row["input_shard"]["path"]),
                int(row["batch_index"]),
            ),
        ):
            receipt_path = Path(str(receipt["receipt"]["path"])).resolve()
            data_path = receipt_path.parent / "documents.parquet"
            validate_file_receipt(data_path, receipt["output"])
            parquet = pq.ParquetFile(data_path)
            for batch in parquet.iter_batches(batch_size=8192):
                import pyarrow as pa

                table = pa.Table.from_batches([batch], schema=batch.schema)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary, table.schema, compression="zstd"
                    )
                writer.write_table(table, row_group_size=min(8192, table.num_rows))
                for row in table.to_pylist():
                    groups[str(row["source_repo_id"])].add(row)
                    global_group.add(row)
                    rows += 1
        if writer is None:
            raise ValueError("no batch documents to consolidate")
        writer.close()
        writer = None
        os.replace(temporary, final)
    except BaseException:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise
    repositories = [groups[name].finish(repo_id=name) for name in sorted(groups)]
    return file_receipt(final, rows=rows), global_group.finish(), repositories


def validate_completed_summary(
    path: Path, output_root: Path, contract_sha256: str
) -> dict[str, Any]:
    value = read_json(path)
    if value.get("schema_version") != SUMMARY_SCHEMA or value.get("status") != "passed":
        raise ValueError(f"{path}: unsupported or incomplete summary")
    if value.get("contract_sha256") != contract_sha256:
        raise ValueError(f"{path}: completed summary contract drift")
    output = value.get("document_output")
    if not isinstance(output, dict):
        raise ValueError(f"{path}: missing document output receipt")
    output_path = output_root / str(output.get("path", ""))
    validate_file_receipt(output_path, output)
    return value


def diagnostics_contract(
    args: argparse.Namespace,
    *,
    shards: Sequence[ShardBinding],
    excluded: Sequence[str],
    sample_input_shards: Sequence[Mapping[str, Any]] | None,
    sample_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": CONTRACT_SCHEMA,
        "scan_mode": args.scan_mode,
        "normalization_manifest": file_receipt(args.normalization_manifest),
        "canonical_root": str(args.canonical_root.resolve()),
        "selected_shards": (
            [dict(row) for row in sample_input_shards]
            if sample_input_shards is not None
            else [shard.receipt() for shard in shards]
        ),
        "review_sample": dict(sample_contract) if sample_contract is not None else None,
        "excluded_source_ids": list(excluded),
        "build_receipt": file_receipt(args.build_receipt),
        "expected_glossapi_commit": args.expected_commit,
        "profiler_script_sha256": sha256_file(Path(__file__).resolve()),
        "document_schema": DOCUMENT_SCHEMA,
        "batch_size": args.batch_size,
        "threads": args.threads,
        "quantile_sample_size": args.quantile_sample_size,
        "scripts_to_keep": ["greek", "latin"],
        "write_cleaned_files": False,
        "zero_badness_zero_greek_guard": True,
    }


def run_diagnostics(args: argparse.Namespace) -> int:
    if args.batch_size < 1 or args.threads < 1 or args.quantile_sample_size < 100:
        raise ValueError(
            "batch size/threads must be positive and quantile sample size >= 100"
        )
    if args.scan_mode == "review_sample":
        required = {
            "--review-sample-packet": args.review_sample_packet,
            "--review-sample-receipt": args.review_sample_receipt,
            "--review-sample-attestation": args.review_sample_attestation,
            "--review-requests": args.review_requests,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"review_sample mode requires {', '.join(missing)}")
    # Snapshot the manifest before parsing it to discover its receipt closure.
    # A manifest swap during discovery is then detected before publication.
    input_snapshots = snapshot_inputs(
        (
            args.normalization_manifest,
            args.build_receipt,
            args.review_sample_packet,
            args.review_sample_receipt,
            args.review_sample_attestation,
            args.review_requests,
        )
    )
    input_snapshots.update(
        snapshot_inputs(
            normalization_dependency_receipt_paths(args.normalization_manifest)
        )
    )
    runtime = validate_runtime_receipt(args.build_receipt, args.expected_commit)
    manifest, shards, excluded = load_normalized_shards(
        args.normalization_manifest,
        args.canonical_root,
        include_source_ids=set(args.source_id or []),
        include_base=args.include_base,
    )
    sample_rows: list[dict[str, Any]] | None = None
    sample_input_shards: list[dict[str, Any]] | None = None
    sample_contract: dict[str, Any] | None = None
    if args.scan_mode == "review_sample":
        if args.include_base or args.source_id:
            raise ValueError(
                "review_sample mode uses the exact packet and cannot alter source coverage"
            )
        assert args.review_sample_packet is not None
        assert args.review_sample_receipt is not None
        assert args.review_sample_attestation is not None
        assert args.review_requests is not None
        sample_rows, sample_input_shards = load_review_sample_packet(
            packet_path=args.review_sample_packet,
            receipt_path=args.review_sample_receipt,
            attestation_path=args.review_sample_attestation,
            requests_path=args.review_requests,
            normalization_manifest=args.normalization_manifest,
        )
        normalized_inventory = sorted(
            (shard.receipt() for shard in shards),
            key=lambda row: (str(row["source_id"]), str(row["path"])),
        )
        received_inventory = sorted(
            sample_input_shards,
            key=lambda row: (str(row.get("source_id", "")), str(row.get("path", ""))),
        )
        if received_inventory != normalized_inventory:
            raise ValueError(
                "review sample input-shard receipt differs from the exact normalization manifest"
            )
        declared_inputs = {
            (str(row.get("path", "")), str(row.get("sha256", "")))
            for row in sample_input_shards
        }
        for row in sample_rows:
            binding = (str(row["input_shard_path"]), str(row["input_shard_sha256"]))
            if binding not in declared_inputs:
                raise ValueError(
                    f"{row['stable_uid']}: review sample references an undeclared canonical shard"
                )
        sample_contract = {
            "review_sample_packet": file_receipt(args.review_sample_packet),
            "review_sample_receipt": file_receipt(args.review_sample_receipt),
            "review_sample_attestation": file_receipt(args.review_sample_attestation),
            "review_requests": file_receipt(args.review_requests),
            "documents": len(sample_rows),
            "text_variant": "high_precision_identifier_masked_review_sample",
        }
    contract = diagnostics_contract(
        args,
        shards=shards,
        excluded=excluded,
        sample_input_shards=sample_input_shards,
        sample_contract=sample_contract,
    )
    contract_sha256 = sha256_json(contract)
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    contract_path = output_root / "contract.json"
    if contract_path.exists():
        input_snapshots.update(snapshot_inputs((contract_path,)))
        if read_json(contract_path) != contract:
            raise ValueError(f"{contract_path}: resume contract drift")
        if not args.resume:
            raise FileExistsError(
                f"existing quality run requires --resume: {output_root}"
            )
    else:
        if any(output_root.iterdir()):
            raise ValueError(
                f"refusing non-empty output without a matching contract: {output_root}"
            )
        write_json_atomic(contract_path, contract, immutable=True)
        input_snapshots.update(snapshot_inputs((contract_path,)))

    summary_path = output_root / f"{SUMMARY_SCHEMA}.json"
    if summary_path.exists():
        input_snapshots.update(snapshot_inputs((summary_path,)))
        value = validate_completed_summary(summary_path, output_root, contract_sha256)
        input_snapshots.update(
            snapshot_quality_handoff_outputs(summary_path, output_root)
        )
        verify_input_snapshots(input_snapshots)
        handoff_payload = build_quality_site_handoff(
            summary_path=summary_path,
            output_root=output_root,
            normalization_manifest=args.normalization_manifest,
            build_receipt=args.build_receipt,
            contract_path=contract_path,
        )
        if args.site_handoff.exists():
            input_snapshots.update(snapshot_inputs((args.site_handoff,)))
            existing_handoff = read_json(args.site_handoff)
            # Creation time is the only intentionally non-contractual field.
            if {
                key: value
                for key, value in existing_handoff.items()
                if key != "created_at"
            } != {
                key: value
                for key, value in handoff_payload.items()
                if key != "created_at"
            }:
                raise ValueError(f"{args.site_handoff}: completed handoff drift")
            verify_input_snapshots(input_snapshots)
        else:
            verify_input_snapshots(input_snapshots)
            write_json_atomic(args.site_handoff, handoff_payload, immutable=True)
        print(
            canonical_json(
                {"ok": True, "already_complete": True, "summary": str(summary_path)}
            )
        )
        return 0 if value["status"] == "passed" else 1

    canonical_shard_snapshots: dict[Path, InputSnapshot] = {}
    if args.scan_mode == "full_scan":
        canonical_shard_snapshots = snapshot_inputs(shard.path for shard in shards)
        for shard in shards:
            snapshot = canonical_shard_snapshots[shard.path.resolve()]
            if snapshot.bytes != shard.bytes or snapshot.sha256 != shard.sha256:
                raise ValueError(f"canonical shard receipt drift: {shard.path}")
        input_snapshots.update(canonical_shard_snapshots)

    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    batch_receipts: list[dict[str, Any]] = []
    shard_inventory: list[dict[str, Any]] = []
    if sample_rows is not None:
        assert args.review_sample_packet is not None
        sample_binding = ShardBinding(
            source_id="exact_source_review_sample",
            path=args.review_sample_packet.resolve(),
            relative_path=f"review-sample/{args.review_sample_packet.name}",
            bytes=args.review_sample_packet.stat().st_size,
            sha256=sha256_file(args.review_sample_packet),
            rows=len(sample_rows),
        )
        for batch_index, row_start in enumerate(
            range(0, len(sample_rows), args.batch_size)
        ):
            rows = sample_rows[row_start : row_start + args.batch_size]
            receipt = process_batch(
                rows=rows,
                shard=sample_binding,
                batch_index=batch_index,
                row_start=row_start,
                output_root=output_root,
                scratch_root=args.scratch_dir,
                contract_sha256=contract_sha256,
                runtime=runtime,
                threads=args.threads,
            )
            batch_receipts.append(receipt)
        shard_inventory = [dict(row) for row in (sample_input_shards or [])]
    else:
        required_columns = [
            "source_id",
            "source_dataset",
            "source_repo_id",
            "source_revision",
            "stable_uid",
            "normalized_text_sha256",
            "source_metadata_json",
            "text",
        ]
        import pyarrow.parquet as pq

        for shard in shards:
            parquet = pq.ParquetFile(shard.path)
            missing = sorted(set(required_columns) - set(parquet.schema_arrow.names))
            if missing:
                raise ValueError(f"{shard.path}: missing canonical columns {missing}")
            row_start = 0
            batches = 0
            for batch_index, batch in enumerate(
                parquet.iter_batches(
                    batch_size=args.batch_size,
                    columns=required_columns,
                    use_threads=False,
                )
            ):
                rows = batch.to_pylist()
                if not rows:
                    continue
                for row in rows:
                    private, corrected = metadata_flags(row.get("source_metadata_json"))
                    row["private_data_true"] = private
                    row["corrected_version_present"] = corrected
                receipt = process_batch(
                    rows=rows,
                    shard=shard,
                    batch_index=batch_index,
                    row_start=row_start,
                    output_root=output_root,
                    scratch_root=args.scratch_dir,
                    contract_sha256=contract_sha256,
                    runtime=runtime,
                    threads=args.threads,
                )
                batch_receipts.append(receipt)
                row_start += len(rows)
                batches += 1
            if row_start != shard.rows:
                raise ValueError(
                    f"{shard.path}: processed {row_start} rows, receipt declares {shard.rows}"
                )
            shard_inventory.append({**shard.receipt(), "batches": batches})

    document_output, global_summary, repository_summaries = consolidate_batches(
        batch_receipts,
        output_root=output_root,
        reservoir_size=args.quantile_sample_size,
    )
    document_output["path"] = Path(str(document_output["path"])).name
    checkpoint_inventory = [
        {
            "receipt_path": Path(str(row["receipt"]["path"]))
            .resolve()
            .relative_to(output_root)
            .as_posix(),
            "receipt_sha256": str(row["receipt"]["sha256"]),
            "output_sha256": str(row["output"]["sha256"]),
            "rows": int(row["output"]["rows"]),
            "input_shard_sha256": str(row["input_shard"]["sha256"]),
            "batch_index": int(row["batch_index"]),
        }
        for row in sorted(
            batch_receipts,
            key=lambda value: (
                str(value["input_shard"]["source_id"]),
                str(value["input_shard"]["path"]),
                int(value["batch_index"]),
            ),
        )
    ]
    payload = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "passed",
        "created_at": utc_now(),
        "mode": "diagnostic_only_no_cleaned_text_persisted",
        "scan_mode": args.scan_mode,
        "contract_sha256": contract_sha256,
        "contract": file_receipt(contract_path, relative_to=output_root),
        "normalization_manifest": file_receipt(args.normalization_manifest),
        "normalization_schema_version": manifest["schema_version"],
        "glossapi_build_receipt": file_receipt(args.build_receipt),
        "glossapi_commit": args.expected_commit,
        "batch_size": args.batch_size,
        "threads": args.threads,
        "quantile_sample_size": args.quantile_sample_size,
        "selected_source_ids": sorted(
            {str(row["source_id"]) for row in sample_rows}
            if sample_rows is not None
            else {shard.source_id for shard in shards}
        ),
        "excluded_source_ids": excluded,
        "input_shards": shard_inventory,
        "batch_checkpoints": {
            "count": len(batch_receipts),
            "rows": sum(int(row["rows"]) for row in checkpoint_inventory),
            "inventory_sha256": sha256_json(checkpoint_inventory),
            "inventory": checkpoint_inventory,
        },
        "document_output": document_output,
        "global": global_summary,
        "repositories": repository_summaries,
        "metric_notes": {
            "rust_noise_badness_score": "Raw glossapi_rs_noise score on the canonical Markdown adapter.",
            "cleaner_removed_character_fraction": (
                "Diagnostic ratio from cleaner final non-whitespace/no-comment characters versus raw "
                "non-whitespace characters; it does not authorize corpus deletion."
            ),
            "approximate_quantiles": (
                f"Deterministic min-hash reservoir capped at {args.quantile_sample_size} documents per group."
            ),
            "zero_badness_zero_greek_guard": (
                "A zero noise score with zero Greek characters is explicitly guarded and must not be read as clean."
            ),
            "profile_scope": (
                "Exact source-review sample after high-precision identifier-pattern masking; use full_scan "
                "for selected raw-population estimates. Generic names and addresses may remain. Identifier "
                "counts in this mode are residual post-masking signals, not source prevalence."
                if args.scan_mode == "review_sample"
                else "All selected canonical documents; nanochat_base is excluded unless explicitly requested."
            ),
        },
    }
    verify_input_snapshots(input_snapshots)
    write_json_atomic(summary_path, payload, immutable=True)
    # The summary is now the terminal receipt for the full-corpus read.  The
    # compact handoff derives only from that summary and generated checkpoints,
    # so do not rehash terabytes of canonical shards a second time.
    for path in canonical_shard_snapshots:
        input_snapshots.pop(path)
    input_snapshots.update(snapshot_inputs((summary_path,)))
    input_snapshots.update(snapshot_quality_handoff_outputs(summary_path, output_root))
    handoff_payload = build_quality_site_handoff(
        summary_path=summary_path,
        output_root=output_root,
        normalization_manifest=args.normalization_manifest,
        build_receipt=args.build_receipt,
        contract_path=contract_path,
    )
    verify_input_snapshots(input_snapshots)
    write_json_atomic(args.site_handoff, handoff_payload, immutable=True)
    print(
        canonical_json(
            {
                "ok": True,
                "summary": str(summary_path),
                "documents": global_summary["documents"],
            }
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build-receipt", help="attest pinned Rust extension modules"
    )
    build.add_argument("--glossapi-root", type=Path, required=True)
    build.add_argument("--expected-commit", default=PINNED_GLOSSAPI_COMMIT)
    build.add_argument(
        "--module-root",
        type=Path,
        help="actual staging root containing imported extension modules",
    )
    build.add_argument(
        "--published-module-root",
        type=Path,
        help="future atomic publication root recorded in the receipt",
    )
    build.add_argument("--maturin-version", required=True)
    build.add_argument("--output", type=Path, required=True)
    build.set_defaults(function=build_runtime_receipt)

    validate_build = subparsers.add_parser(
        "validate-build-receipt", help="rehash and import a published Rust runtime"
    )
    validate_build.add_argument("--receipt", type=Path, required=True)
    validate_build.add_argument("--expected-commit", default=PINNED_GLOSSAPI_COMMIT)
    validate_build.set_defaults(function=validate_runtime_receipt_command)

    run = subparsers.add_parser("run", help="profile receipt-bound canonical Parquet")
    run.add_argument("--normalization-manifest", type=Path, required=True)
    run.add_argument("--canonical-root", type=Path, required=True)
    run.add_argument("--build-receipt", type=Path, required=True)
    run.add_argument("--expected-commit", default=PINNED_GLOSSAPI_COMMIT)
    run.add_argument(
        "--scan-mode",
        choices=("review_sample", "full_scan"),
        default="review_sample",
        help="fast exact review sample (default) or resumable selected-corpus scan",
    )
    run.add_argument("--review-sample-packet", type=Path)
    run.add_argument("--review-sample-receipt", type=Path)
    run.add_argument("--review-sample-attestation", type=Path)
    run.add_argument("--review-requests", type=Path)
    run.add_argument(
        "--source-id",
        action="append",
        help="limit to normalized source_id (repeatable)",
    )
    run.add_argument(
        "--include-base", action="store_true", help="also profile nanochat_base"
    )
    run.add_argument("--batch-size", type=int, default=4096)
    run.add_argument("--threads", type=int, default=256)
    run.add_argument("--quantile-sample-size", type=int, default=8192)
    run.add_argument("--scratch-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--site-handoff", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    run.set_defaults(function=run_diagnostics)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
