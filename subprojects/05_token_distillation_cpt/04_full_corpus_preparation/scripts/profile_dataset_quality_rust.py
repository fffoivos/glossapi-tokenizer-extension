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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
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

    requested: dict[str, dict[str, str]] = {}
    with requests_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
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
            row = json.loads(line)
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
        required = {
            "--review-sample-packet": args.review_sample_packet,
            "--review-sample-receipt": args.review_sample_receipt,
            "--review-requests": args.review_requests,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"review_sample mode requires {', '.join(missing)}")
        sample_rows, sample_input_shards = load_review_sample_packet(
            packet_path=args.review_sample_packet,
            receipt_path=args.review_sample_receipt,
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

    summary_path = output_root / f"{SUMMARY_SCHEMA}.json"
    if summary_path.exists():
        value = validate_completed_summary(summary_path, output_root, contract_sha256)
        print(
            canonical_json(
                {"ok": True, "already_complete": True, "summary": str(summary_path)}
            )
        )
        return 0 if value["status"] == "passed" else 1

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
            validate_file_receipt(shard.path, shard.receipt(), rows=shard.rows)
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
    write_json_atomic(summary_path, payload, immutable=True)
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
    run.add_argument("--resume", action="store_true")
    run.set_defaults(function=run_diagnostics)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
