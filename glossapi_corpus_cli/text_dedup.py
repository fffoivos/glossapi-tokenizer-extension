from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import re
import shlex
import shutil
import sqlite3
import time
import unicodedata
from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
try:
    from datetime import UTC, datetime
except ImportError:  # Python 3.10 compatibility for remote workers
    from datetime import datetime, timezone

    UTC = timezone.utc
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import regex
from blake3 import blake3

try:
    import duckdb
except ImportError:  # pragma: no cover - dependency is required in production/test envs
    duckdb = None


CODE_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = Path(os.environ.get("GLOSSAPI_WORK_ROOT", str(CODE_ROOT)))
DEFAULT_INPUT_ROOT = WORK_ROOT / "hf_release_publish" / "data"
DEFAULT_STATE_ROOT = WORK_ROOT / "analysis" / "dedup" / "text_publish" / "state" / "v1"
DEFAULT_RUNS_ROOT = WORK_ROOT / "analysis" / "dedup" / "text_publish" / "runs"

EXACT_STRICT_VERSION = "exact_strict_norm_v1"
EXACT_RELAXED_VERSION_BASE = "exact_relaxed_norm_v5"
NEAR_NORM_VERSION_BASE = "near_norm_v5"
TOKENIZATION_VERSION = "tokenization_v2"
MINHASH_VERSION = "minhash_v2"
LSH_VERSION = "lsh_v1"
REPRESENTATIVE_SCORE_VERSION = "representative_score_v1_len_greek_badness10"
EXACT_RESULT_INSERT_BATCH_SIZE = 65536
SELECTION_VERSION = "selection_v2"
SURVIVOR_EXPORT_VERSION = "survivor_export_v2"
NEAR_INCREMENTAL_VERSION = "near_incremental_v1"
OVERSIZED_BUCKET_STRATEGY_VERSION = "oversized_bucket_v3"
BUILDER_METADATA_VERSION = "builder_metadata_v2"

STRICT_STAGE = "strict_exact"
RELAXED_STAGE = "relaxed_exact"
EXACT_SURVIVOR_STAGE = "exact_survivors"
NEAR_SIGNATURE_STAGE = "near_signatures"
NEAR_CANDIDATE_STAGE = "near_candidates"
NEAR_CLUSTER_STAGE = "near_clusters"
FINAL_EXPORT_STAGE = "final_export"

ZERO_WIDTH_TRANSLATION = str.maketrans("", "", "\u200b\u200c\u200d\u2060\ufeff")
WHITESPACE_RE = re.compile(r"\s+")
PUNCT_RUN_RE = regex.compile(r"[\p{P}\p{Z}]+")
WORD_TOKEN_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+")
LINE_WRAP_DEHYPHEN_RE = regex.compile(r"(?<=\p{L})-\s*\n\s*(?=\p{L})")
DEFAULT_NEAR_THRESHOLD = 0.85
DEFAULT_NUM_PERM = 128
DEFAULT_BANDS = 32
DEFAULT_ROWS_PER_BAND = 4
DEFAULT_SHINGLE_MODE = "token"
DEFAULT_SHINGLE_SIZE = 5
DEFAULT_TEST_ROWS_PER_FILE = 500
DEFAULT_EXACT_CHUNK_MAX_WORKERS = 1
DEFAULT_RUN_MAX_WORKERS = max(1, os.cpu_count() or 1)
EXACT_STAGE_PARTITION_PREFIX_LEN = 2
EXACT_STAGE_REBUILD_DUCKDB_THREADS = 8
DEFAULT_LARGE_COMPONENT_THRESHOLD = 50
DEFAULT_MAX_BUCKET_SIZE = 5000
DEFAULT_NEAR_CLUSTER_CHUNK_DOCS = 4096
DEFAULT_NEAR_CLUSTER_MAX_COMPONENTS = 256
DEFAULT_NEAR_CLUSTER_MAX_WORKERS = 8
DEFAULT_NEAR_CANDIDATE_MAX_WORKERS = 8
DEFAULT_NEAR_CANDIDATE_PARTITION_MAX_WORKERS = 8
NEAR_CANDIDATE_BUCKET_PREFIX_LEN = 1
SHORT_DOC_TOKEN_THRESHOLD = 20
DEFAULT_EDGE_AUDIT_SAMPLE_SIZE = 256
DEFAULT_CLUSTER_AUDIT_SAMPLE_SIZE = 128
MINHASH_BLOCK_SIZE = 2048
MINHASH_MODULUS = (1 << 61) - 1
FILE_FINGERPRINT_SAMPLE_BYTES = 65536
OVERSIZED_BUCKET_TOKEN_BIN = 32
DEFAULT_GREEK_DIACRITIC_POLICY = "preserve"
GREEK_DIACRITIC_POLICIES = {"preserve", "strip"}
PROCESS_POOL_CONTEXT = mp.get_context("spawn")
PARQUET_COMPRESSION = "zstd"
NEAR_CANDIDATE_FLUSH_ROWS = 4096
NEAR_BUCKET_SUMMARY_FLUSH_ROWS = 4096
NEAR_TOUCHED_DOC_FLUSH_ROWS = 4096


def near_candidate_worker_cap() -> int:
    raw = os.environ.get("GLOSSAPI_NEAR_CANDIDATE_MAX_WORKERS", "").strip()
    if not raw:
        return DEFAULT_NEAR_CANDIDATE_MAX_WORKERS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("GLOSSAPI_NEAR_CANDIDATE_MAX_WORKERS must be an integer") from exc
    if value < 1:
        raise ValueError("GLOSSAPI_NEAR_CANDIDATE_MAX_WORKERS must be >= 1")
    return value


def near_candidate_partition_worker_cap() -> int:
    raw = os.environ.get("GLOSSAPI_NEAR_CANDIDATE_PARTITION_MAX_WORKERS", "").strip()
    if not raw:
        return DEFAULT_NEAR_CANDIDATE_PARTITION_MAX_WORKERS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("GLOSSAPI_NEAR_CANDIDATE_PARTITION_MAX_WORKERS must be an integer") from exc
    if value < 1:
        raise ValueError("GLOSSAPI_NEAR_CANDIDATE_PARTITION_MAX_WORKERS must be >= 1")
    return value


def near_cluster_worker_cap() -> int:
    raw = os.environ.get("GLOSSAPI_NEAR_CLUSTER_MAX_WORKERS", "").strip()
    if not raw:
        return DEFAULT_NEAR_CLUSTER_MAX_WORKERS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("GLOSSAPI_NEAR_CLUSTER_MAX_WORKERS must be an integer") from exc
    if value < 1:
        raise ValueError("GLOSSAPI_NEAR_CLUSTER_MAX_WORKERS must be >= 1")
    return value

RELAXED_TRANSLATION = str.maketrans(
    {
        "’": "'",
        "‘": "'",
        "ʼ": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‐": "-",
        "‑": "-",
        "–": "-",
        "—": "-",
        "…": "...",
        "·": "·",
    }
)

INVENTORY_COLUMNS = [
    "source_dataset",
    "source_doc_id",
    "text",
    "title",
    "author",
    "greek_badness_score",
    "mojibake_badness_score",
    "needs_ocr",
    "is_empty",
    "ocr_success",
    "is_historical_or_polytonic",
]

OPTIONAL_INVENTORY_COLUMNS = [
    "len_greek",
]

SNAPSHOT_MANIFEST_SCHEMA = pa.schema(
    [
        ("source_dataset", pa.string()),
        ("source_doc_id", pa.string()),
        ("doc_key", pa.string()),
        ("content_hash_raw", pa.string()),
        ("file_path", pa.string()),
        ("row_group_index", pa.int64()),
        ("row_index_in_file", pa.int64()),
        ("raw_text_bytes", pa.int64()),
        ("raw_text_chars", pa.int64()),
    ]
)

SNAPSHOT_FILE_SCHEMA = pa.schema(
    [
        ("file_path", pa.string()),
        ("size_bytes", pa.int64()),
        ("mtime_ns", pa.int64()),
        ("fingerprint", pa.string()),
        ("row_groups", pa.int64()),
        ("total_rows", pa.int64()),
    ]
)

EXACT_GROUP_SCHEMA = pa.schema(
    [
        ("group_hash", pa.string()),
        ("group_size", pa.int64()),
        ("kept_doc_key", pa.string()),
        ("member_doc_key", pa.string()),
        ("member_source_dataset", pa.string()),
        ("member_source_doc_id", pa.string()),
        ("dropped", pa.bool_()),
    ]
)

EXACT_DROP_SCHEMA = pa.schema(
    [
        ("doc_key", pa.string()),
        ("source_dataset", pa.string()),
        ("source_doc_id", pa.string()),
        ("kept_doc_key", pa.string()),
        ("group_hash", pa.string()),
        ("reason", pa.string()),
    ]
)

DOCS_EXACT_SCHEMA = pa.schema(
    [
        ("source_dataset", pa.string()),
        ("source_doc_id", pa.string()),
        ("doc_key", pa.string()),
        ("file_path", pa.string()),
        ("row_group_index", pa.int64()),
        ("row_index_in_file", pa.int64()),
        ("content_hash_raw", pa.string()),
        ("exact_strict_hash", pa.string()),
        ("exact_relaxed_hash", pa.string()),
        ("raw_text_bytes", pa.int64()),
        ("raw_text_chars", pa.int64()),
        ("strict_text_chars", pa.int64()),
        ("relaxed_text_chars", pa.int64()),
        ("title", pa.string()),
        ("author", pa.string()),
        ("greek_badness_score", pa.float64()),
        ("len_greek", pa.int64()),
        ("mojibake_badness_score", pa.float64()),
        ("needs_ocr", pa.int64()),
        ("is_empty", pa.int64()),
        ("ocr_success", pa.int64()),
        ("is_historical_or_polytonic", pa.int64()),
        ("reused_exact", pa.int64()),
        ("strict_group_size", pa.int64()),
        ("strict_kept_doc_key", pa.string()),
        ("strict_dropped", pa.int64()),
        ("relaxed_group_size", pa.int64()),
        ("relaxed_kept_doc_key", pa.string()),
        ("relaxed_dropped", pa.int64()),
        ("kept_after_exact", pa.int64()),
    ]
)

RUN_DOC_SCHEMA = pa.schema(
    [
        ("doc_key", pa.string()),
        ("source_dataset", pa.string()),
        ("source_doc_id", pa.string()),
        ("file_path", pa.string()),
        ("row_group_index", pa.int64()),
        ("row_index_in_file", pa.int64()),
        ("content_hash_raw", pa.string()),
        ("exact_strict_hash", pa.string()),
        ("exact_relaxed_hash", pa.string()),
        ("raw_text_bytes", pa.int64()),
        ("raw_text_chars", pa.int64()),
        ("strict_text_chars", pa.int64()),
        ("relaxed_text_chars", pa.int64()),
        ("title", pa.string()),
        ("author", pa.string()),
        ("greek_badness_score", pa.float64()),
        ("len_greek", pa.int64()),
        ("mojibake_badness_score", pa.float64()),
        ("needs_ocr", pa.int64()),
        ("is_empty", pa.int64()),
        ("ocr_success", pa.int64()),
        ("is_historical_or_polytonic", pa.int64()),
        ("reused_exact", pa.int64()),
    ]
)

EXACT_RESULT_SCHEMA = pa.schema(
    [
        ("doc_key", pa.string()),
        ("group_hash", pa.string()),
        ("group_size", pa.int64()),
        ("kept_doc_key", pa.string()),
        ("dropped", pa.int64()),
    ]
)

EXACT_SURVIVOR_SCHEMA = pa.schema(
    [
        ("source_dataset", pa.string()),
        ("source_doc_id", pa.string()),
        ("doc_key", pa.string()),
        ("file_path", pa.string()),
        ("row_group_index", pa.int64()),
        ("row_index_in_file", pa.int64()),
        ("content_hash_raw", pa.string()),
        ("exact_strict_hash", pa.string()),
        ("exact_relaxed_hash", pa.string()),
        ("strict_text_chars", pa.int64()),
        ("relaxed_text_chars", pa.int64()),
        ("title", pa.string()),
        ("author", pa.string()),
        ("greek_badness_score", pa.float64()),
        ("len_greek", pa.int64()),
        ("mojibake_badness_score", pa.float64()),
        ("needs_ocr", pa.int64()),
        ("is_empty", pa.int64()),
        ("ocr_success", pa.int64()),
        ("is_historical_or_polytonic", pa.int64()),
        ("text", pa.string()),
    ]
)

EXACT_SURVIVOR_MANIFEST_SCHEMA = pa.schema(
    [
        ("chunk_key", pa.string()),
        ("source_file_path", pa.string()),
        ("row_group_index", pa.int64()),
        ("row_count", pa.int64()),
        ("survivor_digest", pa.string()),
        ("shard_path", pa.string()),
    ]
)

SIGNATURE_SCHEMA = pa.schema(
    [
        ("doc_key", pa.string()),
        ("source_dataset", pa.string()),
        ("source_doc_id", pa.string()),
        ("file_path", pa.string()),
        ("row_group_index", pa.int64()),
        ("row_index_in_file", pa.int64()),
        ("token_count", pa.int64()),
        ("char_count", pa.int64()),
        ("near_text_chars", pa.int64()),
        ("shingle_count", pa.int64()),
        ("shingle_mode", pa.string()),
        ("shingle_size", pa.int64()),
        ("signature", pa.list_(pa.uint64())),
    ]
)

LSH_BUCKET_SCHEMA = pa.schema(
    [
        ("doc_key", pa.string()),
        ("band_index", pa.int64()),
        ("bucket_hash", pa.string()),
        ("token_count", pa.int64()),
        ("char_count", pa.int64()),
        ("shingle_mode", pa.string()),
    ]
)

CANDIDATE_PAIR_SCHEMA = pa.schema(
    [
        ("doc_key_left", pa.string()),
        ("doc_key_right", pa.string()),
        ("estimated_jaccard", pa.float64()),
        ("shingle_mode", pa.string()),
        ("token_count_left", pa.int64()),
        ("token_count_right", pa.int64()),
        ("length_ratio", pa.float64()),
        ("likely_containment_flag", pa.bool_()),
        ("accepted_reason", pa.string()),
        ("bucket_match_bands", pa.int64()),
    ]
)

BUCKET_SUMMARY_SCHEMA = pa.schema(
    [
        ("band_index", pa.int64()),
        ("bucket_hash", pa.string()),
        ("member_count", pa.int64()),
        ("member_digest", pa.string()),
        ("candidate_row_count", pa.int64()),
    ]
)

TOUCHED_DOC_SCHEMA = pa.schema([("doc_key", pa.string())])

NEAR_CLUSTER_SCHEMA = pa.schema(
    [
        ("cluster_id", pa.string()),
        ("kept_doc_key", pa.string()),
        ("member_doc_key", pa.string()),
        ("member_source_dataset", pa.string()),
        ("member_source_doc_id", pa.string()),
        ("dropped", pa.bool_()),
        ("estimated_jaccard", pa.float64()),
        ("shingle_mode", pa.string()),
        ("token_count", pa.int64()),
        ("char_count", pa.int64()),
        ("length_ratio", pa.float64()),
        ("likely_containment_flag", pa.bool_()),
        ("accepted_reason", pa.string()),
        ("cluster_size", pa.int64()),
        ("component_size", pa.int64()),
        ("large_component_audit_flag", pa.bool_()),
    ]
)

NEAR_DROP_SCHEMA = pa.schema(
    [
        ("doc_key", pa.string()),
        ("source_dataset", pa.string()),
        ("source_doc_id", pa.string()),
        ("kept_doc_key", pa.string()),
        ("cluster_id", pa.string()),
        ("estimated_jaccard", pa.float64()),
        ("shingle_mode", pa.string()),
        ("token_count", pa.int64()),
        ("char_count", pa.int64()),
        ("length_ratio", pa.float64()),
        ("likely_containment_flag", pa.bool_()),
        ("reason", pa.string()),
    ]
)

FINAL_DECISION_SCHEMA = pa.schema(
    [
        ("doc_key", pa.string()),
        ("source_dataset", pa.string()),
        ("source_doc_id", pa.string()),
        ("decision", pa.string()),
        ("decision_stage", pa.string()),
        ("cluster_id", pa.string()),
        ("kept_doc_key", pa.string()),
        ("reason", pa.string()),
        ("exact_strict_version", pa.string()),
        ("exact_relaxed_version", pa.string()),
        ("near_norm_version", pa.string()),
        ("tokenization_version", pa.string()),
        ("shingle_version", pa.string()),
        ("minhash_version", pa.string()),
        ("lsh_version", pa.string()),
        ("selection_version", pa.string()),
    ]
)

CLUSTER_SUMMARY_SCHEMA = pa.schema(
    [
        ("cluster_id", pa.string()),
        ("decision_stage", pa.string()),
        ("cluster_size", pa.int64()),
        ("dropped_count", pa.int64()),
        ("kept_doc_key", pa.string()),
        ("mixed_source", pa.bool_()),
        ("large_component_audit_flag", pa.bool_()),
        ("narrow_margin_flag", pa.bool_()),
    ]
)

BUILDER_DOC_METADATA_SCHEMA = pa.schema(
    [
        ("doc_key", pa.string()),
        ("source_dataset", pa.string()),
        ("source_doc_id", pa.string()),
        ("strict_exact_group_hash", pa.string()),
        ("strict_exact_group_size", pa.int64()),
        ("strict_exact_mixed_source", pa.bool_()),
        ("strict_exact_kept_doc_key", pa.string()),
        ("relaxed_exact_group_hash", pa.string()),
        ("relaxed_exact_group_size", pa.int64()),
        ("relaxed_exact_mixed_source", pa.bool_()),
        ("relaxed_exact_kept_doc_key", pa.string()),
        ("near_candidate_count", pa.int64()),
        ("near_best_match_doc_key", pa.string()),
        ("near_best_match_source_dataset", pa.string()),
        ("near_best_estimated_jaccard", pa.float64()),
        ("near_best_length_ratio", pa.float64()),
        ("near_cross_dataset_candidate_count", pa.int64()),
        ("near_same_dataset_candidate_count", pa.int64()),
        ("needs_ocr", pa.int64()),
        ("greek_badness_score", pa.float64()),
        ("len_greek", pa.int64()),
        ("mojibake_badness_score", pa.float64()),
        ("ocr_success", pa.int64()),
        ("text_length_for_selection", pa.int64()),
        ("representative_score", pa.float64()),
        ("representative_score_version", pa.string()),
        ("has_title", pa.bool_()),
        ("has_author", pa.bool_()),
        ("dedup_run_id", pa.string()),
        ("greek_diacritic_policy", pa.string()),
        ("exact_strict_version", pa.string()),
        ("exact_relaxed_version", pa.string()),
        ("near_norm_version", pa.string()),
        ("tokenization_version", pa.string()),
        ("shingle_version", pa.string()),
        ("minhash_version", pa.string()),
        ("lsh_version", pa.string()),
        ("selection_version", pa.string()),
    ]
)

BUILDER_EXACT_MEMBERSHIP_SCHEMA = pa.schema(
    [
        ("doc_key", pa.string()),
        ("source_dataset", pa.string()),
        ("source_doc_id", pa.string()),
        ("stage", pa.string()),
        ("group_hash", pa.string()),
        ("group_size", pa.int64()),
        ("mixed_source", pa.bool_()),
        ("kept_doc_key", pa.string()),
    ]
)

BUILDER_NEAR_PAIR_SCHEMA = pa.schema(
    [
        ("doc_key_left", pa.string()),
        ("source_dataset_left", pa.string()),
        ("source_doc_id_left", pa.string()),
        ("doc_key_right", pa.string()),
        ("source_dataset_right", pa.string()),
        ("source_doc_id_right", pa.string()),
        ("estimated_jaccard", pa.float64()),
        ("length_ratio", pa.float64()),
        ("likely_containment_flag", pa.bool_()),
        ("accepted_reason", pa.string()),
        ("bucket_match_bands", pa.int64()),
        ("shingle_mode", pa.string()),
        ("shingle_size", pa.int64()),
        ("num_perm", pa.int64()),
        ("bands", pa.int64()),
        ("rows_per_band", pa.int64()),
        ("candidate_score_floor", pa.float64()),
    ]
)

BUILDER_FAMILY_MEMBERSHIP_SCHEMA = pa.schema(
    [
        ("doc_key", pa.string()),
        ("source_dataset", pa.string()),
        ("source_doc_id", pa.string()),
        ("family_id", pa.string()),
        ("family_size", pa.int64()),
        ("family_source_count", pa.int64()),
        ("family_mixed_source", pa.bool_()),
        ("canonical_kept_doc_key", pa.string()),
        ("canonical_decision", pa.string()),
        ("canonical_decision_stage", pa.string()),
        ("dedup_run_id", pa.string()),
        ("selection_version", pa.string()),
        ("representative_score_version", pa.string()),
    ]
)

EDGE_AUDIT_SAMPLE_SCHEMA = pa.schema(
    [
        ("sample_reason", pa.string()),
        ("doc_key_left", pa.string()),
        ("doc_key_right", pa.string()),
        ("estimated_jaccard", pa.float64()),
        ("shingle_mode", pa.string()),
        ("token_count_left", pa.int64()),
        ("token_count_right", pa.int64()),
        ("length_ratio", pa.float64()),
        ("likely_containment_flag", pa.bool_()),
        ("accepted_reason", pa.string()),
        ("bucket_match_bands", pa.int64()),
    ]
)

CLUSTER_AUDIT_SAMPLE_SCHEMA = pa.schema(
    [
        ("sample_reason", pa.string()),
        ("cluster_id", pa.string()),
        ("decision_stage", pa.string()),
        ("cluster_size", pa.int64()),
        ("dropped_count", pa.int64()),
        ("kept_doc_key", pa.string()),
        ("mixed_source", pa.bool_()),
        ("large_component_audit_flag", pa.bool_()),
        ("narrow_margin_flag", pa.bool_()),
    ]
)


@dataclass(frozen=True)
class InputFile:
    path: Path
    size_bytes: int
    mtime_ns: int
    fingerprint: str
    row_groups: int
    total_rows: int


@dataclass(frozen=True)
class RowGroupChunk:
    file_path: Path
    row_group_index: int
    row_group_rows: int
    row_offset: int


@dataclass(frozen=True)
class SignatureLookup:
    row_by_doc_key: dict[str, int]
    matrix: np.ndarray

    def __getitem__(self, doc_key: str) -> np.ndarray:
        return self.matrix[self.row_by_doc_key[doc_key]]

    def __iter__(self) -> Any:
        return iter(self.row_by_doc_key)

    def __len__(self) -> int:
        return len(self.row_by_doc_key)


@dataclass(frozen=True)
class SignatureMetadataLookup:
    row_by_doc_key: dict[str, int]
    values: np.ndarray

    def __len__(self) -> int:
        return len(self.row_by_doc_key)

    def _row_index(self, doc_key: str) -> int:
        return self.row_by_doc_key[doc_key]

    def token_count(self, doc_key: str) -> int:
        return int(self.values["token_count"][self._row_index(doc_key)])

    def char_count(self, doc_key: str) -> int:
        return int(self.values["char_count"][self._row_index(doc_key)])

    def near_text_chars(self, doc_key: str) -> int:
        return int(self.values["near_text_chars"][self._row_index(doc_key)])

    def shingle_count(self, doc_key: str) -> int:
        return int(self.values["shingle_count"][self._row_index(doc_key)])

    def shingle_mode(self, doc_key: str) -> str:
        return str(self.values["shingle_mode"][self._row_index(doc_key)])

    def shingle_size(self, doc_key: str) -> int:
        return int(self.values["shingle_size"][self._row_index(doc_key)])


def now_utc_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def temp_output_path(path: Path) -> Path:
    ensure_dir(path.parent)
    return path.parent / f".{path.name}.{uuid4().hex}.tmp"


def archival_path(path: Path, *, archive_root: Path) -> Path:
    ensure_dir(archive_root)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return archive_root / f"{path.name}.{stamp}.{uuid4().hex}"


def archive_existing_path(path: Path, *, archive_root: Path) -> Path | None:
    if not path.exists():
        return None
    destination = archival_path(path, archive_root=archive_root)
    path.replace(destination)
    return destination


def atomic_replace(src: Path, dest: Path) -> None:
    ensure_dir(dest.parent)
    src.replace(dest)


def atomic_copy(src: Path, dest: Path) -> None:
    temp_path = temp_output_path(dest)
    shutil.copy2(src, temp_path)
    atomic_replace(temp_path, dest)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp_path = temp_output_path(path)
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    atomic_replace(temp_path, path)


def append_debug_trace(path: Path, message: str) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"{now_utc_iso()} {message}\n")


def write_text_atomic(path: Path, payload: str, *, executable: bool = False) -> None:
    temp_path = temp_output_path(path)
    temp_path.write_text(payload)
    if executable:
        temp_path.chmod(0o755)
    atomic_replace(temp_path, path)
    if executable:
        path.chmod(0o755)


def write_table_atomic(path: Path, table: pa.Table) -> None:
    temp_path = temp_output_path(path)
    pq.write_table(table, temp_path)
    atomic_replace(temp_path, path)


def progress_dir(run_root: Path) -> Path:
    return run_root / "progress"


def progress_file_path(run_root: Path, stage: str) -> Path:
    return progress_dir(run_root) / f"{stage}.json"


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def all_paths_exist(paths: list[Path]) -> bool:
    return all(path.exists() for path in paths)


def parquet_num_rows(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def hashed_filename(stem: str) -> str:
    return hash_bytes(stem.encode("utf-8"))


def text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text or None


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return float(value)
    try:
        parsed = float(value)
    except Exception:
        return None
    if math.isnan(parsed):
        return None
    return parsed


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        return int(value)
    except Exception:
        return None


def optional_bool_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return int(bool(value))


def row_value(row: Any, key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except Exception:
        pass
    if isinstance(row, dict):
        return row.get(key, default)
    return default


def selection_length_value(row: Any, *, text_length_field: str) -> int:
    len_greek = optional_int(row_value(row, "len_greek"))
    if len_greek is not None:
        return len_greek
    return int(row_value(row, text_length_field, 0) or 0)


def representative_score(length_value: int, greek_badness_score: float | None) -> float:
    if length_value <= 0:
        return 0.0
    if greek_badness_score is None:
        return 0.0
    return max(0.0, float(length_value) * (1.0 - (float(greek_badness_score) / 10.0)))


def representative_score_from_row(row: Any, *, text_length_field: str) -> float:
    return representative_score(
        selection_length_value(row, text_length_field=text_length_field),
        optional_float(row_value(row, "greek_badness_score")),
    )


def hash_bytes(data: bytes) -> str:
    return blake3(data).hexdigest()


def stable_doc_key(source_dataset: str, source_doc_id: str) -> str:
    return hash_bytes(f"{source_dataset}\0{source_doc_id}".encode("utf-8"))


def file_resume_fingerprint(path: Path, *, size_bytes: int, mtime_ns: int) -> str:
    hasher = blake3()
    hasher.update(str(path).encode("utf-8"))
    hasher.update(str(size_bytes).encode("ascii"))
    hasher.update(str(mtime_ns).encode("ascii"))
    with path.open("rb") as handle:
        head = handle.read(FILE_FINGERPRINT_SAMPLE_BYTES)
        hasher.update(head)
        if size_bytes > FILE_FINGERPRINT_SAMPLE_BYTES:
            tail_offset = max(0, size_bytes - FILE_FINGERPRINT_SAMPLE_BYTES)
            if tail_offset > len(head):
                handle.seek(tail_offset)
                hasher.update(handle.read(FILE_FINGERPRINT_SAMPLE_BYTES))
    return hasher.hexdigest()


def digest_doc_keys(doc_keys: list[str] | set[str]) -> str:
    hasher = blake3()
    for doc_key in sorted(doc_keys):
        hasher.update(doc_key.encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def exact_relaxed_version(*, greek_diacritic_policy: str) -> str:
    return f"{EXACT_RELAXED_VERSION_BASE}_{greek_diacritic_policy}"


def near_norm_version(*, greek_diacritic_policy: str) -> str:
    return f"{NEAR_NORM_VERSION_BASE}_{greek_diacritic_policy}"


def validate_greek_diacritic_policy(greek_diacritic_policy: str) -> str:
    if greek_diacritic_policy not in GREEK_DIACRITIC_POLICIES:
        allowed = ", ".join(sorted(GREEK_DIACRITIC_POLICIES))
        raise ValueError(f"greek_diacritic_policy must be one of: {allowed}")
    return greek_diacritic_policy


def normalize_exact_strict_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.translate(ZERO_WIDTH_TRANSLATION)
    normalized = unicodedata.normalize("NFC", normalized)
    normalized = WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def strip_greek_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    output: list[str] = []
    previous_base_is_greek = False
    for char in decomposed:
        category = unicodedata.category(char)
        if category.startswith("M"):
            if previous_base_is_greek:
                continue
            output.append(char)
            continue
        codepoint = ord(char)
        previous_base_is_greek = 0x0370 <= codepoint <= 0x03FF or 0x1F00 <= codepoint <= 0x1FFF
        output.append(char)
    return unicodedata.normalize("NFC", "".join(output))


def normalize_exact_relaxed_text(text: str, *, greek_diacritic_policy: str = DEFAULT_GREEK_DIACRITIC_POLICY) -> str:
    greek_diacritic_policy = validate_greek_diacritic_policy(greek_diacritic_policy)
    normalized = text.casefold()
    if greek_diacritic_policy == "strip":
        normalized = strip_greek_diacritics(normalized)
    normalized = normalized.replace("\u00ad", "")
    normalized = normalized.translate(RELAXED_TRANSLATION)
    normalized = PUNCT_RUN_RE.sub(" ", normalized)
    normalized = WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def config_payload(
    *,
    input_root: Path,
    state_root: Path,
    run_root: Path,
    greek_diacritic_policy: str,
) -> dict[str, Any]:
    greek_diacritic_policy = validate_greek_diacritic_policy(greek_diacritic_policy)
    return {
        "input_root": str(input_root),
        "state_root": str(state_root),
        "run_root": str(run_root),
        "exact_strict_version": EXACT_STRICT_VERSION,
        "exact_relaxed_version": exact_relaxed_version(greek_diacritic_policy=greek_diacritic_policy),
        "near_norm_version": near_norm_version(greek_diacritic_policy=greek_diacritic_policy),
        "greek_diacritic_policy": greek_diacritic_policy,
        "stages": ["inventory", "strict_exact", "relaxed_exact"],
    }


def config_hash(payload: dict[str, Any]) -> str:
    return hash_bytes(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))


def default_run_root() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_RUNS_ROOT / f"exact_stage_{stamp}"


def run_docs_inventory_path(run_root: Path) -> Path:
    return run_root / "run_docs_inventory.parquet"


def exact_membership_path(run_root: Path, stage: str) -> Path:
    stage_root = run_root / "stage_01_exact"
    if stage == STRICT_STAGE:
        return stage_root / "strict_exact_memberships.parquet"
    if stage == RELAXED_STAGE:
        return stage_root / "relaxed_exact_memberships.parquet"
    raise ValueError(f"unsupported exact membership stage: {stage}")


def require_duckdb() -> Any:
    if duckdb is None:
        raise RuntimeError("duckdb is required for parquet-backed exact dedup execution")
    return duckdb


def connect_duckdb(*, threads: int | None = None) -> Any:
    db = require_duckdb()
    conn = db.connect()
    if threads is None:
        threads = max(1, os.cpu_count() or 1)
    conn.execute(f"PRAGMA threads={int(max(1, threads))}")
    return conn


def db_path(state_root: Path) -> Path:
    return state_root / "state.sqlite"


def connect_db(state_root: Path) -> sqlite3.Connection:
    ensure_dir(state_root)
    conn = sqlite3.connect(db_path(state_root))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA foreign_keys=ON")
    ensure_schema(conn)
    return conn


def connect_db_reader(state_root: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path(state_root)}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            status TEXT NOT NULL,
            input_root TEXT NOT NULL,
            run_root TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            config_json TEXT NOT NULL,
            completed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS run_input_files (
            run_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            fingerprint TEXT,
            row_groups INTEGER NOT NULL,
            total_rows INTEGER NOT NULL,
            PRIMARY KEY (run_id, file_path)
        );

        CREATE TABLE IF NOT EXISTS run_chunks (
            run_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            row_group_index INTEGER NOT NULL,
            row_group_rows INTEGER NOT NULL,
            status TEXT NOT NULL,
            processed_at TEXT,
            PRIMARY KEY (run_id, file_path, row_group_index)
        );

        CREATE TABLE IF NOT EXISTS run_stage_chunks (
            run_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            chunk_key TEXT NOT NULL,
            status TEXT NOT NULL,
            processed_at TEXT,
            artifact_path TEXT,
            row_count INTEGER,
            PRIMARY KEY (run_id, stage, chunk_key)
        );

        CREATE TABLE IF NOT EXISTS stage_progress (
            run_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            status TEXT NOT NULL,
            total_chunks INTEGER NOT NULL,
            completed_chunks INTEGER NOT NULL,
            progress_path TEXT,
            progress_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (run_id, stage)
        );

        CREATE TABLE IF NOT EXISTS doc_cache (
            doc_key TEXT PRIMARY KEY,
            source_dataset TEXT NOT NULL,
            source_doc_id TEXT NOT NULL,
            content_hash_raw TEXT NOT NULL,
            exact_strict_hash TEXT NOT NULL,
            exact_relaxed_hash TEXT NOT NULL,
            exact_strict_version TEXT NOT NULL,
            exact_relaxed_version TEXT NOT NULL,
            raw_text_bytes INTEGER NOT NULL,
            raw_text_chars INTEGER NOT NULL,
            strict_text_chars INTEGER NOT NULL,
            relaxed_text_chars INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS run_docs (
            run_id TEXT NOT NULL,
            doc_key TEXT NOT NULL,
            source_dataset TEXT NOT NULL,
            source_doc_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            row_group_index INTEGER NOT NULL,
            row_index_in_file INTEGER NOT NULL,
            content_hash_raw TEXT NOT NULL,
            exact_strict_hash TEXT NOT NULL,
            exact_relaxed_hash TEXT NOT NULL,
            raw_text_bytes INTEGER NOT NULL,
            raw_text_chars INTEGER NOT NULL,
            strict_text_chars INTEGER NOT NULL,
            relaxed_text_chars INTEGER NOT NULL,
            title TEXT,
            author TEXT,
            greek_badness_score REAL,
            len_greek INTEGER,
            mojibake_badness_score REAL,
            needs_ocr INTEGER,
            is_empty INTEGER,
            ocr_success INTEGER,
            is_historical_or_polytonic INTEGER,
            reused_exact INTEGER NOT NULL,
            PRIMARY KEY (run_id, doc_key)
        );

        CREATE TABLE IF NOT EXISTS run_exact_results (
            run_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            doc_key TEXT NOT NULL,
            group_hash TEXT NOT NULL,
            group_size INTEGER NOT NULL,
            kept_doc_key TEXT NOT NULL,
            dropped INTEGER NOT NULL,
            PRIMARY KEY (run_id, stage, doc_key)
        );

        CREATE TABLE IF NOT EXISTS run_near_results (
            run_id TEXT NOT NULL,
            doc_key TEXT NOT NULL,
            cluster_id TEXT NOT NULL,
            kept_doc_key TEXT NOT NULL,
            dropped INTEGER NOT NULL,
            estimated_jaccard REAL,
            shingle_mode TEXT,
            token_count INTEGER,
            char_count INTEGER,
            length_ratio REAL,
            likely_containment_flag INTEGER,
            accepted_reason TEXT,
            component_size INTEGER NOT NULL,
            cluster_size INTEGER NOT NULL,
            large_component_audit_flag INTEGER NOT NULL,
            PRIMARY KEY (run_id, doc_key)
        );

        CREATE INDEX IF NOT EXISTS idx_run_docs_run_id ON run_docs (run_id);
        CREATE INDEX IF NOT EXISTS idx_run_docs_strict_hash ON run_docs (run_id, exact_strict_hash);
        CREATE INDEX IF NOT EXISTS idx_run_docs_relaxed_hash ON run_docs (run_id, exact_relaxed_hash);
        CREATE INDEX IF NOT EXISTS idx_run_exact_results_stage ON run_exact_results (run_id, stage, dropped);
        CREATE INDEX IF NOT EXISTS idx_run_stage_chunks_stage ON run_stage_chunks (run_id, stage, status);
        CREATE INDEX IF NOT EXISTS idx_run_near_results_run_id ON run_near_results (run_id, dropped);
        """
    )
    existing_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(run_input_files)")}
    if "fingerprint" not in existing_columns:
        conn.execute("ALTER TABLE run_input_files ADD COLUMN fingerprint TEXT")
    run_docs_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(run_docs)")}
    if "len_greek" not in run_docs_columns:
        conn.execute("ALTER TABLE run_docs ADD COLUMN len_greek INTEGER")
    conn.commit()


def discover_input_files(input_root: Path) -> list[InputFile]:
    files: list[InputFile] = []
    for path in sorted(input_root.rglob("*.parquet")):
        parquet_file = pq.ParquetFile(path)
        metadata = parquet_file.metadata
        stat = path.stat()
        files.append(
            InputFile(
                path=path,
                size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                fingerprint=file_resume_fingerprint(path, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns),
                row_groups=metadata.num_row_groups,
                total_rows=metadata.num_rows,
            )
        )
    return files


def validate_input_files(files: list[InputFile]) -> None:
    missing_by_file: dict[str, list[str]] = {}
    for input_file in files:
        schema_names = set(pq.ParquetFile(input_file.path).schema_arrow.names)
        missing = [column for column in INVENTORY_COLUMNS if column not in schema_names]
        if missing:
            missing_by_file[str(input_file.path)] = missing
    if missing_by_file:
        details = "; ".join(f"{path}: missing {', '.join(columns)}" for path, columns in missing_by_file.items())
        raise ValueError(f"input snapshot schema validation failed: {details}")


def _copy_parquet_head(src: Path, dest: Path, *, row_limit: int) -> int:
    parquet_file = pq.ParquetFile(src)
    total_rows = int(parquet_file.metadata.num_rows)
    if row_limit <= 0:
        write_table_atomic(dest, parquet_file.schema_arrow.empty_table())
        return 0
    if total_rows <= row_limit:
        atomic_copy(src, dest)
        return total_rows
    remaining = row_limit
    batches: list[pa.Table] = []
    for row_group_index in range(parquet_file.metadata.num_row_groups):
        if remaining <= 0:
            break
        table = parquet_file.read_row_group(row_group_index)
        if table.num_rows > remaining:
            table = table.slice(0, remaining)
        batches.append(table)
        remaining -= int(table.num_rows)
    combined = pa.concat_tables(batches) if batches else parquet_file.schema_arrow.empty_table()
    write_table_atomic(dest, combined)
    return row_limit - remaining


def build_dedup_run_command(
    *,
    input_root: Path,
    state_root: Path,
    run_root: Path,
    max_workers: int,
    greek_diacritic_policy: str,
    exact_only: bool,
    minhash_threshold: float,
    num_perm: int,
    bands: int,
    rows_per_band: int,
    shingle_mode: str,
    shingle_size: int,
    max_bucket_size: int,
) -> list[str]:
    command = [
        "glossapi-corpus",
        "dedup-text",
        "run",
        "--input-root",
        str(input_root),
        "--state-root",
        str(state_root),
        "--run-root",
        str(run_root),
        "--max-workers",
        str(max_workers),
        "--greek-diacritic-policy",
        greek_diacritic_policy,
    ]
    if exact_only:
        command.append("--exact-only")
    command.extend(
        [
            "--minhash-threshold",
            str(minhash_threshold),
            "--num-perm",
            str(num_perm),
            "--bands",
            str(bands),
            "--rows-per-band",
            str(rows_per_band),
            "--shingle-mode",
            shingle_mode,
            "--shingle-size",
            str(shingle_size),
            "--max-bucket-size",
            str(max_bucket_size),
        ]
    )
    return command


def prepare_test_dedup_run(
    *,
    experiment_root: Path,
    input_root: Path = DEFAULT_INPUT_ROOT,
    rows_per_file: int = DEFAULT_TEST_ROWS_PER_FILE,
    max_files: int | None = None,
    max_workers: int = DEFAULT_RUN_MAX_WORKERS,
    greek_diacritic_policy: str = DEFAULT_GREEK_DIACRITIC_POLICY,
    exact_only: bool = False,
    minhash_threshold: float = DEFAULT_NEAR_THRESHOLD,
    num_perm: int = DEFAULT_NUM_PERM,
    bands: int = DEFAULT_BANDS,
    rows_per_band: int = DEFAULT_ROWS_PER_BAND,
    shingle_mode: str = DEFAULT_SHINGLE_MODE,
    shingle_size: int = DEFAULT_SHINGLE_SIZE,
    max_bucket_size: int = DEFAULT_MAX_BUCKET_SIZE,
) -> dict[str, Any]:
    if rows_per_file < 1:
        raise ValueError("rows_per_file must be >= 1")
    if max_files is not None and max_files < 1:
        raise ValueError("max_files must be >= 1 when provided")
    input_root = input_root.resolve()
    experiment_root = experiment_root.resolve()
    try:
        experiment_root.relative_to(input_root)
    except ValueError:
        pass
    else:
        raise ValueError("experiment_root must not be inside input_root")
    greek_diacritic_policy = validate_greek_diacritic_policy(greek_diacritic_policy)
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    if max_bucket_size < 2:
        raise ValueError("max_bucket_size must be >= 2")
    if bands <= 0 or rows_per_band <= 0 or num_perm <= 0:
        raise ValueError("num_perm, bands, and rows_per_band must be positive")
    if bands * rows_per_band != num_perm:
        raise ValueError("bands * rows_per_band must equal num_perm")
    if shingle_mode not in {"token", "char"}:
        raise ValueError("shingle_mode must be one of: token, char")
    if shingle_size < 2:
        raise ValueError("shingle_size must be >= 2")
    files = discover_input_files(input_root)
    if not files:
        raise ValueError(f"no parquet files found under {input_root}")
    validate_input_files(files)
    selected_files = files if max_files is None else files[:max_files]
    prepared_input_root = experiment_root / "input"
    state_root = experiment_root / "state"
    run_root = experiment_root / "run_current"
    launch_script_path = experiment_root / "launch_test_dedup.sh"
    summary_path = experiment_root / "prepare_test_run_summary.json"
    if prepared_input_root.exists():
        shutil.rmtree(prepared_input_root)
    ensure_dir(prepared_input_root)
    sampled_files: list[dict[str, Any]] = []
    sampled_rows = 0
    source_rows = 0
    for input_file in selected_files:
        relative_path = input_file.path.relative_to(input_root)
        output_path = prepared_input_root / relative_path
        copied_rows = _copy_parquet_head(input_file.path, output_path, row_limit=rows_per_file)
        sampled_rows += copied_rows
        source_rows += int(input_file.total_rows)
        sampled_files.append(
            {
                "source_file_path": str(input_file.path),
                "relative_path": str(relative_path),
                "prepared_file_path": str(output_path),
                "source_rows": int(input_file.total_rows),
                "prepared_rows": int(copied_rows),
                "truncated": bool(copied_rows < int(input_file.total_rows)),
            }
        )
    command = build_dedup_run_command(
        input_root=prepared_input_root,
        state_root=state_root,
        run_root=run_root,
        max_workers=max_workers,
        greek_diacritic_policy=greek_diacritic_policy,
        exact_only=exact_only,
        minhash_threshold=minhash_threshold,
        num_perm=num_perm,
        bands=bands,
        rows_per_band=rows_per_band,
        shingle_mode=shingle_mode,
        shingle_size=shingle_size,
        max_bucket_size=max_bucket_size,
    )
    write_text_atomic(
        launch_script_path,
        "#!/usr/bin/env bash\nset -euo pipefail\n\n" + shlex.join(command) + "\n",
        executable=True,
    )
    payload = {
        "prepared_at": now_utc_iso(),
        "experiment_root": str(experiment_root),
        "input_root": str(input_root),
        "prepared_input_root": str(prepared_input_root),
        "state_root": str(state_root),
        "run_root": str(run_root),
        "launch_script_path": str(launch_script_path),
        "summary_path": str(summary_path),
        "run_command": shlex.join(command),
        "sampling": {
            "rows_per_file": int(rows_per_file),
            "max_files": None if max_files is None else int(max_files),
            "selected_file_count": int(len(selected_files)),
            "total_source_file_count": int(len(files)),
            "prepared_rows": int(sampled_rows),
            "selected_source_rows": int(source_rows),
        },
        "sampled_files": sampled_files,
    }
    write_json_atomic(summary_path, payload)
    return payload


def init_run(conn: sqlite3.Connection, *, run_id: str, config: dict[str, Any], config_digest: str) -> None:
    existing = conn.execute("SELECT config_hash FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    timestamp = now_utc_iso()
    if existing is None:
        conn.execute(
            """
            INSERT INTO runs (run_id, created_at, updated_at, status, input_root, run_root, config_hash, config_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                timestamp,
                timestamp,
                "running",
                config["input_root"],
                config["run_root"],
                config_digest,
                json.dumps(config, ensure_ascii=False, sort_keys=True),
            ),
        )
        conn.commit()
        return
    if existing["config_hash"] != config_digest:
        raise ValueError(f"run {run_id} already exists with a different config hash")
    conn.execute("UPDATE runs SET updated_at = ?, status = ? WHERE run_id = ?", (timestamp, "running", run_id))
    conn.commit()


def register_input_files(conn: sqlite3.Connection, *, run_id: str, files: list[InputFile]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO run_input_files (run_id, file_path, size_bytes, mtime_ns, fingerprint, row_groups, total_rows)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (run_id, str(item.path), item.size_bytes, item.mtime_ns, item.fingerprint, item.row_groups, item.total_rows)
            for item in files
        ],
    )
    conn.commit()


def validate_registered_input_snapshot(conn: sqlite3.Connection, *, run_id: str, files: list[InputFile], resume: bool) -> None:
    if not resume:
        return
    stored_rows = list(
        conn.execute(
            """
            SELECT file_path, size_bytes, mtime_ns, fingerprint, row_groups, total_rows
            FROM run_input_files
            WHERE run_id = ?
            ORDER BY file_path
            """,
            (run_id,),
        )
    )
    if not stored_rows:
        existing_activity = conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM run_chunks WHERE run_id = ?) AS chunk_rows,
                (SELECT COUNT(*) FROM run_docs WHERE run_id = ?) AS doc_rows
            """,
            (run_id, run_id),
        ).fetchone()
        if int(existing_activity["chunk_rows"] or 0) > 0 or int(existing_activity["doc_rows"] or 0) > 0:
            raise ValueError(
                f"cannot resume run {run_id}: stored input snapshot metadata is missing under the existing run_root"
            )
        return
    current_by_path = {str(item.path): item for item in files}
    stored_by_path = {str(row["file_path"]): row for row in stored_rows}
    added_paths = sorted(set(current_by_path) - set(stored_by_path))
    missing_paths = sorted(set(stored_by_path) - set(current_by_path))
    mismatch_details: list[str] = []
    if added_paths:
        mismatch_details.append(f"added files: {', '.join(added_paths[:3])}")
    if missing_paths:
        mismatch_details.append(f"missing files: {', '.join(missing_paths[:3])}")
    for file_path, current in current_by_path.items():
        stored = stored_by_path.get(file_path)
        if stored is None:
            continue
        differences: list[str] = []
        if int(stored["size_bytes"]) != int(current.size_bytes):
            differences.append("size_bytes")
        if int(stored["mtime_ns"]) != int(current.mtime_ns):
            differences.append("mtime_ns")
        if int(stored["row_groups"]) != int(current.row_groups):
            differences.append("row_groups")
        if int(stored["total_rows"]) != int(current.total_rows):
            differences.append("total_rows")
        stored_fingerprint = stored["fingerprint"]
        if stored_fingerprint is None:
            differences.append("stored fingerprint missing")
        elif str(stored_fingerprint) != current.fingerprint:
            differences.append("fingerprint")
        if differences:
            mismatch_details.append(f"{file_path}: {', '.join(differences)}")
    if mismatch_details:
        preview = "; ".join(mismatch_details[:5])
        raise ValueError(
            f"cannot resume run {run_id}: input snapshot changed under the existing run_root ({preview})"
        )


def register_chunks(conn: sqlite3.Connection, *, run_id: str, files: list[InputFile]) -> None:
    rows: list[tuple[Any, ...]] = []
    for input_file in files:
        parquet_file = pq.ParquetFile(input_file.path)
        for row_group_index in range(input_file.row_groups):
            rows.append(
                (
                    run_id,
                    str(input_file.path),
                    row_group_index,
                    parquet_file.metadata.row_group(row_group_index).num_rows,
                    "pending",
                )
            )
    conn.executemany(
        """
        INSERT OR IGNORE INTO run_chunks (run_id, file_path, row_group_index, row_group_rows, status)
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def upsert_stage_progress(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    stage: str,
    status: str,
    total_chunks: int,
    completed_chunks: int,
    progress_path: Path,
    payload: dict[str, Any],
) -> None:
    updated_at = now_utc_iso()
    conn.execute(
        """
        INSERT INTO stage_progress (run_id, stage, status, total_chunks, completed_chunks, progress_path, progress_json, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, stage) DO UPDATE SET
            status = excluded.status,
            total_chunks = excluded.total_chunks,
            completed_chunks = excluded.completed_chunks,
            progress_path = excluded.progress_path,
            progress_json = excluded.progress_json,
            updated_at = excluded.updated_at
        """,
        (
            run_id,
            stage,
            status,
            total_chunks,
            completed_chunks,
            str(progress_path),
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            updated_at,
        ),
    )
    conn.commit()
    write_json_atomic(progress_path, payload)


def register_stage_chunks(conn: sqlite3.Connection, *, run_id: str, stage: str, chunk_keys: list[str]) -> None:
    conn.executemany(
        """
        INSERT OR IGNORE INTO run_stage_chunks (run_id, stage, chunk_key, status)
        VALUES (?, ?, ?, ?)
        """,
        [(run_id, stage, chunk_key, "pending") for chunk_key in chunk_keys],
    )
    conn.commit()


def stage_chunk_status(conn: sqlite3.Connection, *, run_id: str, stage: str, chunk_key: str) -> str | None:
    row = conn.execute(
        "SELECT status FROM run_stage_chunks WHERE run_id = ? AND stage = ? AND chunk_key = ?",
        (run_id, stage, chunk_key),
    ).fetchone()
    return None if row is None else str(row["status"])


def mark_stage_chunk_complete(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    stage: str,
    chunk_key: str,
    artifact_path: Path | None = None,
    row_count: int | None = None,
) -> None:
    conn.execute(
        """
        UPDATE run_stage_chunks
        SET status = ?, processed_at = ?, artifact_path = ?, row_count = ?
        WHERE run_id = ? AND stage = ? AND chunk_key = ?
        """,
        ("completed", now_utc_iso(), None if artifact_path is None else str(artifact_path), row_count, run_id, stage, chunk_key),
    )
    conn.commit()


def row_group_status(conn: sqlite3.Connection, *, run_id: str, file_path: Path, row_group_index: int) -> str | None:
    row = conn.execute(
        """
        SELECT status
        FROM run_chunks
        WHERE run_id = ? AND file_path = ? AND row_group_index = ?
        """,
        (run_id, str(file_path), row_group_index),
    ).fetchone()
    return None if row is None else str(row["status"])


def enumerate_row_group_chunks(files: list[InputFile]) -> list[RowGroupChunk]:
    tasks: list[RowGroupChunk] = []
    for input_file in files:
        parquet_file = pq.ParquetFile(input_file.path)
        row_offset = 0
        for row_group_index in range(input_file.row_groups):
            row_group_rows = parquet_file.metadata.row_group(row_group_index).num_rows
            tasks.append(
                RowGroupChunk(
                    file_path=input_file.path,
                    row_group_index=row_group_index,
                    row_group_rows=row_group_rows,
                    row_offset=row_offset,
                )
            )
            row_offset += row_group_rows
    return tasks


def effective_worker_count(max_workers: int, task_count: int) -> int:
    if task_count <= 0:
        return 1
    return max(1, min(max_workers, task_count))


def fetch_cache_map(conn: sqlite3.Connection, doc_keys: list[str]) -> dict[str, sqlite3.Row]:
    if not doc_keys:
        return {}
    rows: dict[str, sqlite3.Row] = {}
    chunk_size = 500
    for offset in range(0, len(doc_keys), chunk_size):
        chunk = doc_keys[offset : offset + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        query = f"SELECT * FROM doc_cache WHERE doc_key IN ({placeholders})"
        for row in conn.execute(query, chunk):
            rows[str(row["doc_key"])] = row
    return rows


def selection_priority(row: sqlite3.Row, *, text_length_field: str) -> tuple[float, float, float, float, float, int, int, str, str]:
    priority = selection_priority_tuple(row, text_length_field=text_length_field)
    return (
        float(priority[0]),
        float(priority[1]),
        float(priority[2]),
        float(priority[3]),
        float(priority[4]),
        int(priority[5]) + int(priority[6]) + int(priority[7]),
        str(priority[8]),
        str(priority[9]),
    )


def selection_priority_tuple(row: sqlite3.Row, *, text_length_field: str) -> tuple[Any, ...]:
    needs_ocr = row["needs_ocr"]
    if needs_ocr == 0:
        needs_ocr_rank = 0.0
    elif needs_ocr == 1:
        needs_ocr_rank = 1.0
    else:
        needs_ocr_rank = 0.5
    ocr_success = row["ocr_success"]
    invalid_ocr_rank = 1.0 if needs_ocr == 1 and ocr_success != 1 else 0.0
    greek_badness = optional_float(row["greek_badness_score"])
    representative_rank = -representative_score_from_row(row, text_length_field=text_length_field)
    greek_rank = float(greek_badness) if greek_badness is not None else float("inf")
    mojibake_badness = row["mojibake_badness_score"]
    mojibake_rank = float(mojibake_badness) if mojibake_badness is not None else float("inf")
    if ocr_success == 1:
        ocr_rank = 0.0
    elif ocr_success == 0:
        ocr_rank = 1.0
    else:
        ocr_rank = 0.5
    title_rank = 0 if row["title"] else 1
    author_rank = 0 if row["author"] else 1
    return (
        invalid_ocr_rank,
        needs_ocr_rank,
        representative_rank,
        greek_rank,
        mojibake_rank,
        ocr_rank,
        title_rank,
        author_rank,
        str(row["source_dataset"]),
        str(row["source_doc_id"]),
    )


def arrow_field_value(value: Any, field_type: pa.DataType) -> Any:
    if value is None:
        return None
    if pa.types.is_boolean(field_type):
        return bool(value)
    if pa.types.is_integer(field_type):
        return int(value)
    if pa.types.is_floating(field_type):
        return float(value)
    if pa.types.is_string(field_type):
        return str(value)
    return value


def table_from_pylist(payload: list[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    arrays = [
        pa.array([arrow_field_value(row.get(field.name), field.type) for row in payload], type=field.type)
        for field in schema
    ]
    return pa.Table.from_arrays(arrays, schema=schema)


def _stream_query_to_parquet(
    conn: sqlite3.Connection,
    query: str,
    params: tuple[Any, ...],
    path: Path,
    *,
    schema: pa.Schema,
) -> int:
    cursor = conn.execute(query, params)
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    temp_path = temp_output_path(path)
    try:
        while True:
            batch = cursor.fetchmany(2048)
            if not batch:
                break
            payload = [dict(row) for row in batch]
            table = table_from_pylist(payload, schema)
            if writer is None:
                writer = pq.ParquetWriter(temp_path, schema)
            writer.write_table(table)
            rows_written += len(payload)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        pq.write_table(schema.empty_table(), temp_path)
    atomic_replace(temp_path, path)
    return rows_written


def _stream_duckdb_query_to_parquet(
    query: str,
    params: list[Any],
    path: Path,
    *,
    schema: pa.Schema,
    threads: int | None = None,
) -> int:
    duck = connect_duckdb(threads=threads)
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    temp_path = temp_output_path(path)
    try:
        reader = duck.execute(query, params).fetch_record_batch(rows_per_batch=2048)
        try:
            for batch in reader:
                table = pa.Table.from_batches([batch], schema=batch.schema).cast(schema)
                if writer is None:
                    writer = pq.ParquetWriter(temp_path, schema)
                writer.write_table(table)
                rows_written += table.num_rows
        finally:
            if writer is not None:
                writer.close()
        if writer is None:
            pq.write_table(schema.empty_table(), temp_path)
        atomic_replace(temp_path, path)
        return rows_written
    finally:
        duck.close()


def write_run_docs_inventory(conn: sqlite3.Connection, *, run_id: str, path: Path) -> int:
    return _stream_query_to_parquet(
        conn,
        """
        SELECT
            doc_key,
            source_dataset,
            source_doc_id,
            file_path,
            row_group_index,
            row_index_in_file,
            content_hash_raw,
            exact_strict_hash,
            exact_relaxed_hash,
            raw_text_bytes,
            raw_text_chars,
            strict_text_chars,
            relaxed_text_chars,
            title,
            author,
            greek_badness_score,
            len_greek,
            mojibake_badness_score,
            needs_ocr,
            is_empty,
            ocr_success,
            is_historical_or_polytonic,
            reused_exact
        FROM run_docs
        WHERE run_id = ?
        ORDER BY doc_key
        """,
        (run_id,),
        path,
        schema=RUN_DOC_SCHEMA,
    )


def load_existing_exact_stage_summary(*, run_root: Path, emit_survivor_export: bool) -> dict[str, Any] | None:
    stage_root = run_root / "stage_01_exact"
    summary_path = stage_root / "summary.json"
    required_paths = [
        run_docs_inventory_path(run_root),
        run_root / "snapshot_manifest.parquet",
        stage_root / "strict_exact_groups.parquet",
        stage_root / "strict_exact_drop_list.parquet",
        stage_root / "relaxed_exact_groups.parquet",
        stage_root / "relaxed_exact_drop_list.parquet",
        stage_root / "docs_exact.parquet",
        summary_path,
    ]
    if emit_survivor_export:
        required_paths.append(stage_root / "exact_survivor_manifest.parquet")
    if any(not path.exists() for path in required_paths):
        return None
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(summary, dict) or not summary.get("run_id"):
        return None
    return summary


def export_exact_membership_from_sqlite(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    stage: str,
    destination: Path,
) -> int:
    return _stream_query_to_parquet(
        conn,
        """
        SELECT
            doc_key,
            group_hash,
            group_size,
            kept_doc_key,
            dropped
        FROM run_exact_results
        WHERE run_id = ? AND stage = ?
        ORDER BY doc_key
        """,
        (run_id, stage),
        destination,
        schema=EXACT_RESULT_SCHEMA,
    )


def summarize_exact_stage_artifacts(*, stage: str, membership_path: Path, groups_path: Path, drops_path: Path) -> dict[str, Any]:
    membership_rows = 0
    kept_rows = 0
    duplicate_groups = 0
    duplicate_rows = 0
    drop_rows = 0
    if membership_path.exists():
        for batch_rows in iter_parquet_batches(membership_path, columns=["dropped"]):
            membership_rows += len(batch_rows)
            kept_rows += sum(1 for row in batch_rows if int(row["dropped"] or 0) == 0)
    if groups_path.exists():
        for batch_rows in iter_parquet_batches(groups_path, columns=["dropped"]):
            duplicate_rows += len(batch_rows)
            duplicate_groups += sum(1 for row in batch_rows if not bool(row["dropped"]))
    if drops_path.exists():
        for batch_rows in iter_parquet_batches(drops_path, columns=["doc_key"]):
            drop_rows += len(batch_rows)
    return {
        "stage": stage,
        "duplicate_groups": int(duplicate_groups),
        "duplicate_rows": int(duplicate_rows),
        "dropped_rows": int(membership_rows - kept_rows),
        "kept_rows": int(kept_rows),
        "group_membership_rows": int(duplicate_rows),
        "drop_rows": int(drop_rows),
    }


def exact_stage_hash_prefixes(stage: str) -> list[str | None]:
    if stage == RELAXED_STAGE:
        width = EXACT_STAGE_PARTITION_PREFIX_LEN
        return [f"{value:0{width}x}" for value in range(16**width)]
    return [None]


def iter_exact_stage_rows(
    *,
    run_root: Path,
    stage: str,
    threads: int | None = None,
    hash_prefix: str | None = None,
) -> Any:
    inventory_path = run_docs_inventory_path(run_root)
    if not inventory_path.exists():
        raise FileNotFoundError(f"run docs inventory missing under {inventory_path}")
    if stage == STRICT_STAGE:
        if hash_prefix is None:
            query = """
                SELECT *
                FROM read_parquet(?)
                ORDER BY exact_strict_hash, doc_key
            """
            params = [str(inventory_path)]
        else:
            query = """
                SELECT *
                FROM read_parquet(?)
                WHERE substr(exact_strict_hash, 1, ?) = ?
                ORDER BY exact_strict_hash, doc_key
            """
            params = [str(inventory_path), len(hash_prefix), hash_prefix]
    elif stage == RELAXED_STAGE:
        strict_membership = exact_membership_path(run_root, STRICT_STAGE)
        if not strict_membership.exists():
            raise FileNotFoundError(f"strict exact membership parquet missing under {strict_membership}")
        if hash_prefix is None:
            query = """
                SELECT d.*
                FROM read_parquet(?) AS d
                JOIN read_parquet(?) AS s
                  ON s.doc_key = d.doc_key
                WHERE s.dropped = 0
                ORDER BY d.exact_relaxed_hash, d.doc_key
            """
            params = [str(inventory_path), str(strict_membership)]
        else:
            query = """
                SELECT d.*
                FROM read_parquet(?) AS d
                JOIN read_parquet(?) AS s
                  ON s.doc_key = d.doc_key
                WHERE s.dropped = 0
                  AND substr(d.exact_relaxed_hash, 1, ?) = ?
                ORDER BY d.exact_relaxed_hash, d.doc_key
            """
            params = [str(inventory_path), str(strict_membership), len(hash_prefix), hash_prefix]
    else:
        raise ValueError(f"unsupported exact-stage query stage: {stage}")
    duck = connect_duckdb(threads=threads)
    try:
        reader = duck.execute(query, params).fetch_record_batch(rows_per_batch=2048)
        for batch in reader:
            table = pa.Table.from_batches([batch], schema=batch.schema)
            payload = table.to_pydict()
            for idx in range(table.num_rows):
                yield {name: payload[name][idx] for name in payload}
    finally:
        duck.close()


def prepare_exact_row_group_result(
    *,
    state_root: Path,
    run_id: str,
    chunk: RowGroupChunk,
    greek_diacritic_policy: str,
) -> dict[str, Any]:
    reader = connect_db_reader(state_root)
    try:
        parquet_file = pq.ParquetFile(chunk.file_path)
        selected_columns = [
            column
            for column in [*INVENTORY_COLUMNS, *OPTIONAL_INVENTORY_COLUMNS]
            if column in parquet_file.schema_arrow.names
        ]
        table = parquet_file.read_row_group(
            chunk.row_group_index,
            columns=selected_columns,
            use_threads=False,
        )
        columns = {name: table.column(name).to_pylist() for name in selected_columns}
        for column in OPTIONAL_INVENTORY_COLUMNS:
            if column not in columns:
                columns[column] = [None] * table.num_rows
        doc_keys = [
            stable_doc_key(str(columns["source_dataset"][i]), str(columns["source_doc_id"][i]))
            for i in range(table.num_rows)
        ]
        cache_map = fetch_cache_map(reader, doc_keys)
    finally:
        reader.close()
    processed_at = now_utc_iso()
    exact_relaxed_version_value = exact_relaxed_version(greek_diacritic_policy=greek_diacritic_policy)
    run_doc_rows: list[tuple[Any, ...]] = []
    cache_rows: list[tuple[Any, ...]] = []
    reused_rows = 0
    for idx in range(table.num_rows):
        source_dataset = str(columns["source_dataset"][idx])
        source_doc_id = str(columns["source_doc_id"][idx])
        doc_key = doc_keys[idx]
        raw_text = text_value(columns["text"][idx])
        raw_bytes = raw_text.encode("utf-8")
        content_hash_raw = hash_bytes(raw_bytes)
        cache_row = cache_map.get(doc_key)
        reused_exact = False
        if (
            cache_row is not None
            and str(cache_row["content_hash_raw"]) == content_hash_raw
            and str(cache_row["exact_strict_version"]) == EXACT_STRICT_VERSION
            and str(cache_row["exact_relaxed_version"]) == exact_relaxed_version_value
        ):
            exact_strict_hash = str(cache_row["exact_strict_hash"])
            exact_relaxed_hash = str(cache_row["exact_relaxed_hash"])
            strict_text_chars = int(cache_row["strict_text_chars"])
            relaxed_text_chars = int(cache_row["relaxed_text_chars"])
            reused_exact = True
            reused_rows += 1
        else:
            strict_text = normalize_exact_strict_text(raw_text)
            relaxed_text = normalize_exact_relaxed_text(
                strict_text,
                greek_diacritic_policy=greek_diacritic_policy,
            )
            exact_strict_hash = hash_bytes(strict_text.encode("utf-8"))
            exact_relaxed_hash = hash_bytes(relaxed_text.encode("utf-8"))
            strict_text_chars = len(strict_text)
            relaxed_text_chars = len(relaxed_text)
        raw_text_bytes = len(raw_bytes)
        raw_text_chars = len(raw_text)
        title = optional_text(columns["title"][idx])
        author = optional_text(columns["author"][idx])
        greek_badness_score = optional_float(columns["greek_badness_score"][idx])
        len_greek = optional_int(columns["len_greek"][idx])
        mojibake_badness_score = optional_float(columns["mojibake_badness_score"][idx])
        needs_ocr = optional_bool_int(columns["needs_ocr"][idx])
        is_empty = optional_bool_int(columns["is_empty"][idx])
        ocr_success = optional_bool_int(columns["ocr_success"][idx])
        is_historical_or_polytonic = optional_bool_int(columns["is_historical_or_polytonic"][idx])
        run_doc_rows.append(
            (
                run_id,
                doc_key,
                source_dataset,
                source_doc_id,
                str(chunk.file_path),
                chunk.row_group_index,
                chunk.row_offset + idx,
                content_hash_raw,
                exact_strict_hash,
                exact_relaxed_hash,
                raw_text_bytes,
                raw_text_chars,
                strict_text_chars,
                relaxed_text_chars,
                title,
                author,
                greek_badness_score,
                len_greek,
                mojibake_badness_score,
                needs_ocr,
                is_empty,
                ocr_success,
                is_historical_or_polytonic,
                int(reused_exact),
            )
        )
        cache_rows.append(
            (
                doc_key,
                source_dataset,
                source_doc_id,
                content_hash_raw,
                exact_strict_hash,
                exact_relaxed_hash,
                EXACT_STRICT_VERSION,
                exact_relaxed_version_value,
                raw_text_bytes,
                raw_text_chars,
                strict_text_chars,
                relaxed_text_chars,
                processed_at,
            )
        )
    return {
        "file_path": str(chunk.file_path),
        "row_group_index": int(chunk.row_group_index),
        "rows": int(table.num_rows),
        "reused_rows": int(reused_rows),
        "processed_at": processed_at,
        "run_doc_rows": run_doc_rows,
        "cache_rows": cache_rows,
    }


def commit_exact_row_group_result(conn: sqlite3.Connection, *, run_id: str, result: dict[str, Any]) -> dict[str, int]:
    file_path = str(result["file_path"])
    row_group_index = int(result["row_group_index"])
    processed_at = str(result["processed_at"])
    run_doc_rows = list(result["run_doc_rows"])
    cache_rows = list(result["cache_rows"])
    try:
        with conn:
            # Some upstream source parquets contain repeated (source_dataset, source_doc_id)
            # pairs. Keep the first observed doc_key for this run and ignore later duplicates
            # so the dedup pipeline can continue over a noisy snapshot.
            conn.executemany(
                """
                INSERT OR IGNORE INTO run_docs (
                    run_id,
                    doc_key,
                    source_dataset,
                    source_doc_id,
                    file_path,
                    row_group_index,
                    row_index_in_file,
                    content_hash_raw,
                    exact_strict_hash,
                    exact_relaxed_hash,
                    raw_text_bytes,
                    raw_text_chars,
                    strict_text_chars,
                    relaxed_text_chars,
                    title,
                    author,
                    greek_badness_score,
                    len_greek,
                    mojibake_badness_score,
                    needs_ocr,
                    is_empty,
                    ocr_success,
                    is_historical_or_polytonic,
                    reused_exact
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                run_doc_rows,
            )
            conn.executemany(
                """
                INSERT INTO doc_cache (
                    doc_key,
                    source_dataset,
                    source_doc_id,
                    content_hash_raw,
                    exact_strict_hash,
                    exact_relaxed_hash,
                    exact_strict_version,
                    exact_relaxed_version,
                    raw_text_bytes,
                    raw_text_chars,
                    strict_text_chars,
                    relaxed_text_chars,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_key) DO UPDATE SET
                    source_dataset = excluded.source_dataset,
                    source_doc_id = excluded.source_doc_id,
                    content_hash_raw = excluded.content_hash_raw,
                    exact_strict_hash = excluded.exact_strict_hash,
                    exact_relaxed_hash = excluded.exact_relaxed_hash,
                    exact_strict_version = excluded.exact_strict_version,
                    exact_relaxed_version = excluded.exact_relaxed_version,
                    raw_text_bytes = excluded.raw_text_bytes,
                    raw_text_chars = excluded.raw_text_chars,
                    strict_text_chars = excluded.strict_text_chars,
                    relaxed_text_chars = excluded.relaxed_text_chars,
                    updated_at = excluded.updated_at
                """,
                cache_rows,
            )
            conn.execute(
                """
                UPDATE run_chunks
                SET status = ?, processed_at = ?
                WHERE run_id = ? AND file_path = ? AND row_group_index = ?
                """,
                ("completed", processed_at, run_id, file_path, row_group_index),
            )
            conn.execute(
                "UPDATE runs SET updated_at = ? WHERE run_id = ?",
                (processed_at, run_id),
            )
    except sqlite3.IntegrityError as exc:
        raise ValueError(
            f"duplicate source_dataset + source_doc_id detected within the snapshot while processing {file_path} row group {row_group_index}"
        ) from exc
    return {"rows": int(result["rows"]), "reused_rows": int(result["reused_rows"])}


def write_group_parquet(rows: list[dict[str, Any]], path: Path, *, schema: pa.Schema) -> int:
    temp_path = temp_output_path(path)
    if not rows:
        pq.write_table(schema.empty_table(), temp_path)
        atomic_replace(temp_path, path)
        return 0
    pq.write_table(table_from_pylist(rows, schema), temp_path)
    atomic_replace(temp_path, path)
    return len(rows)


def append_rows_to_parquet_writer(
    writer: pq.ParquetWriter | None,
    *,
    rows: list[dict[str, Any]],
    temp_path: Path,
    schema: pa.Schema,
    compression: str = PARQUET_COMPRESSION,
    row_group_size: int | None = None,
) -> tuple[pq.ParquetWriter | None, int]:
    if not rows:
        return writer, 0
    table = table_from_pylist(rows, schema)
    if writer is None:
        writer = pq.ParquetWriter(temp_path, schema, compression=compression)
    writer.write_table(table, row_group_size=row_group_size)
    return writer, len(rows)


def finalize_parquet_writer(
    writer: pq.ParquetWriter | None,
    *,
    temp_path: Path,
    destination: Path,
    schema: pa.Schema,
) -> None:
    if writer is not None:
        writer.close()
    else:
        pq.write_table(schema.empty_table(), temp_path, compression=PARQUET_COMPRESSION)
    atomic_replace(temp_path, destination)


def checkpoint_wal(conn: sqlite3.Connection) -> None:
    """Best-effort WAL truncation after large stage writes.

    This does not change any dedup decision semantics. It only keeps the SQLite
    state from carrying a pathological WAL across exact/near stage boundaries.
    """

    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.OperationalError:
        return


def build_stage_results(
    conn: sqlite3.Connection,
    *,
    run_root: Path,
    run_id: str,
    stage: str,
    hash_column: str,
    text_length_field: str,
    groups_path: Path,
    drops_path: Path,
) -> dict[str, Any]:
    if hash_column not in {"exact_strict_hash", "exact_relaxed_hash"}:
        raise ValueError(f"unsupported exact-stage hash column: {hash_column}")
    trace_path = progress_dir(run_root) / "exact_stage_trace.log"
    membership_path = exact_membership_path(run_root, stage)
    existing_stage_rows = conn.execute(
        "SELECT COUNT(*) FROM run_exact_results WHERE run_id = ? AND stage = ?",
        (run_id, stage),
    ).fetchone()[0]
    if membership_path.exists() and groups_path.exists() and drops_path.exists():
        append_debug_trace(trace_path, f"{stage}:reuse_existing_parquet")
        return summarize_exact_stage_artifacts(
            stage=stage,
            membership_path=membership_path,
            groups_path=groups_path,
            drops_path=drops_path,
        )
    if existing_stage_rows and groups_path.exists() and drops_path.exists():
        append_debug_trace(trace_path, f"{stage}:export_membership_from_sqlite:start rows={existing_stage_rows}")
        export_exact_membership_from_sqlite(conn, run_id=run_id, stage=stage, destination=membership_path)
        append_debug_trace(trace_path, f"{stage}:export_membership_from_sqlite:done")
        return summarize_exact_stage_artifacts(
            stage=stage,
            membership_path=membership_path,
            groups_path=groups_path,
            drops_path=drops_path,
        )
    append_debug_trace(trace_path, f"{stage}:rebuild:start rows={existing_stage_rows}")
    group_temp_path = temp_output_path(groups_path)
    drop_temp_path = temp_output_path(drops_path)
    membership_temp_path = temp_output_path(membership_path)
    group_writer: pq.ParquetWriter | None = None
    drop_writer: pq.ParquetWriter | None = None
    membership_writer: pq.ParquetWriter | None = None
    group_count = 0
    drop_count = 0
    duplicate_groups = 0
    duplicate_rows = 0
    dropped_rows = 0
    kept_rows = 0
    current_hash: str | None = None
    current_group: list[dict[str, Any]] = []
    partition_prefixes = exact_stage_hash_prefixes(stage)
    append_debug_trace(trace_path, f"{stage}:rebuild:partitions={len(partition_prefixes)}")

    def flush_group(group: list[dict[str, Any]], group_hash: str | None) -> None:
        nonlocal group_writer, drop_writer, membership_writer
        nonlocal group_count, drop_count, duplicate_groups, duplicate_rows, dropped_rows, kept_rows
        if not group or group_hash is None:
            return
        kept = min(group, key=lambda row: selection_priority_tuple(row, text_length_field=text_length_field))
        group_size = len(group)
        group_rows: list[dict[str, Any]] = []
        drop_rows: list[dict[str, Any]] = []
        membership_rows: list[dict[str, Any]] = []
        if group_size > 1:
            duplicate_groups += 1
            duplicate_rows += group_size
        for row in group:
            dropped = int(row["doc_key"] != kept["doc_key"])
            if dropped:
                dropped_rows += 1
            else:
                kept_rows += 1
            membership_rows.append(
                (
                    {
                        "doc_key": str(row["doc_key"]),
                        "group_hash": group_hash,
                        "group_size": group_size,
                        "kept_doc_key": str(kept["doc_key"]),
                        "dropped": dropped,
                    }
                )
            )
            if group_size > 1:
                group_rows.append(
                    {
                        "group_hash": group_hash,
                        "group_size": group_size,
                        "kept_doc_key": str(kept["doc_key"]),
                        "member_doc_key": str(row["doc_key"]),
                        "member_source_dataset": str(row["source_dataset"]),
                        "member_source_doc_id": str(row["source_doc_id"]),
                        "dropped": bool(dropped),
                    }
                )
                if dropped:
                    drop_rows.append(
                        {
                            "doc_key": str(row["doc_key"]),
                            "source_dataset": str(row["source_dataset"]),
                            "source_doc_id": str(row["source_doc_id"]),
                            "kept_doc_key": str(kept["doc_key"]),
                            "group_hash": group_hash,
                            "reason": stage,
                        }
                    )
        group_writer, written_group_rows = append_rows_to_parquet_writer(
            group_writer,
            rows=group_rows,
            temp_path=group_temp_path,
            schema=EXACT_GROUP_SCHEMA,
        )
        group_count += written_group_rows
        drop_writer, written_drop_rows = append_rows_to_parquet_writer(
            drop_writer,
            rows=drop_rows,
            temp_path=drop_temp_path,
            schema=EXACT_DROP_SCHEMA,
        )
        drop_count += written_drop_rows
        membership_writer, _ = append_rows_to_parquet_writer(
            membership_writer,
            rows=membership_rows,
            temp_path=membership_temp_path,
            schema=EXACT_RESULT_SCHEMA,
        )

    try:
        for idx, hash_prefix in enumerate(partition_prefixes):
            if hash_prefix is not None and (idx == 0 or idx == len(partition_prefixes) - 1 or idx % 32 == 0):
                append_debug_trace(trace_path, f"{stage}:rebuild:partition_start prefix={hash_prefix} index={idx+1}/{len(partition_prefixes)}")
            for row in iter_exact_stage_rows(
                run_root=run_root,
                stage=stage,
                threads=EXACT_STAGE_REBUILD_DUCKDB_THREADS,
                hash_prefix=hash_prefix,
            ):
                row_hash = str(row[hash_column])
                if current_hash is None:
                    current_hash = row_hash
                if row_hash != current_hash:
                    flush_group(current_group, current_hash)
                    current_group = []
                    current_hash = row_hash
                current_group.append(row)
            flush_group(current_group, current_hash)
            current_group = []
            current_hash = None
    finally:
        finalize_parquet_writer(group_writer, temp_path=group_temp_path, destination=groups_path, schema=EXACT_GROUP_SCHEMA)
        finalize_parquet_writer(drop_writer, temp_path=drop_temp_path, destination=drops_path, schema=EXACT_DROP_SCHEMA)
        finalize_parquet_writer(
            membership_writer,
            temp_path=membership_temp_path,
            destination=membership_path,
            schema=EXACT_RESULT_SCHEMA,
        )
    result = {
        "stage": stage,
        "duplicate_groups": duplicate_groups,
        "duplicate_rows": duplicate_rows,
        "dropped_rows": dropped_rows,
        "kept_rows": kept_rows,
        "group_membership_rows": group_count,
        "drop_rows": drop_count,
    }
    append_debug_trace(
        trace_path,
        f"{stage}:rebuild:done duplicate_groups={duplicate_groups} duplicate_rows={duplicate_rows} kept_rows={kept_rows} dropped_rows={dropped_rows}",
    )
    return result


def write_snapshot_manifest(conn: sqlite3.Connection, *, run_id: str, path: Path) -> int:
    return _stream_query_to_parquet(
        conn,
        """
        SELECT
            source_dataset,
            source_doc_id,
            doc_key,
            content_hash_raw,
            file_path,
            row_group_index,
            row_index_in_file,
            raw_text_bytes,
            raw_text_chars
        FROM run_docs
        WHERE run_id = ?
        ORDER BY file_path, row_index_in_file
        """,
        (run_id,),
        path,
        schema=SNAPSHOT_MANIFEST_SCHEMA,
    )


def write_docs_exact_export(*, run_root: Path, path: Path) -> int:
    return _stream_duckdb_query_to_parquet(
        """
        SELECT
            d.source_dataset,
            d.source_doc_id,
            d.doc_key,
            d.file_path,
            d.row_group_index,
            d.row_index_in_file,
            d.content_hash_raw,
            d.exact_strict_hash,
            d.exact_relaxed_hash,
            d.raw_text_bytes,
            d.raw_text_chars,
            d.strict_text_chars,
            d.relaxed_text_chars,
            d.title,
            d.author,
            d.greek_badness_score,
            d.len_greek,
            d.mojibake_badness_score,
            d.needs_ocr,
            d.is_empty,
            d.ocr_success,
            d.is_historical_or_polytonic,
            d.reused_exact,
            s.group_size AS strict_group_size,
            s.kept_doc_key AS strict_kept_doc_key,
            s.dropped AS strict_dropped,
            r.group_size AS relaxed_group_size,
            r.kept_doc_key AS relaxed_kept_doc_key,
            r.dropped AS relaxed_dropped,
            CASE
                WHEN s.dropped = 1 THEN 0
                WHEN COALESCE(r.dropped, 0) = 1 THEN 0
                ELSE 1
            END AS kept_after_exact
        FROM read_parquet(?) AS d
        JOIN read_parquet(?) AS s
          ON s.doc_key = d.doc_key
        LEFT JOIN read_parquet(?) AS r
          ON r.doc_key = d.doc_key
        ORDER BY d.file_path, d.row_index_in_file
        """,
        [
            str(run_docs_inventory_path(run_root)),
            str(exact_membership_path(run_root, STRICT_STAGE)),
            str(exact_membership_path(run_root, RELAXED_STAGE)),
        ],
        path,
        schema=DOCS_EXACT_SCHEMA,
    )


def exact_survivor_chunk_key(file_path: Path, row_group_index: int) -> str:
    return f"{file_path}::{row_group_index}"


def exact_survivor_shard_path(stage_root: Path, file_path: Path, row_group_index: int) -> Path:
    shard_root = stage_root / "shards" / "exact_survivors"
    ensure_dir(shard_root)
    return shard_root / f"{file_path.stem}__rg{row_group_index:05d}.parquet"


def exact_survivor_manifest_path(stage_root: Path) -> Path:
    return stage_root / "exact_survivor_manifest.parquet"


def survivor_digest_from_payload(payload: list[dict[str, Any]]) -> str:
    hasher = blake3()
    for row in payload:
        hasher.update(str(row["doc_key"]).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(row["content_hash_raw"]).encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def compute_exact_survivor_shard(
    *,
    docs_exact_path: Path,
    file_path: Path,
    row_group_index: int,
    row_offset: int,
    stage_root: Path,
) -> dict[str, Any]:
    survivor_rows = exact_survivor_rows_for_group(
        docs_exact_path=docs_exact_path,
        file_path=file_path,
        row_group_index=row_group_index,
    )
    shard_path = exact_survivor_shard_path(stage_root, file_path, row_group_index)
    if not survivor_rows:
        if shard_path.exists():
            shard_path.unlink()
        return {
            "chunk_key": exact_survivor_chunk_key(file_path, row_group_index),
            "artifact_path": None,
            "row_count": 0,
            "survivor_digest": None,
            "source_file_path": str(file_path),
            "row_group_index": int(row_group_index),
        }
    parquet_file = pq.ParquetFile(file_path)
    text_table = parquet_file.read_row_group(row_group_index, columns=["text"], use_threads=False)
    texts = text_table.column("text").to_pylist()
    payload: list[dict[str, Any]] = []
    for row in survivor_rows:
        local_idx = int(row["row_index_in_file"]) - row_offset
        payload.append(
            {
                "source_dataset": str(row["source_dataset"]),
                "source_doc_id": str(row["source_doc_id"]),
                "doc_key": str(row["doc_key"]),
                "file_path": str(row["file_path"]),
                "row_group_index": int(row["row_group_index"]),
                "row_index_in_file": int(row["row_index_in_file"]),
                "content_hash_raw": str(row["content_hash_raw"]),
                "exact_strict_hash": str(row["exact_strict_hash"]),
                "exact_relaxed_hash": str(row["exact_relaxed_hash"]),
                "strict_text_chars": int(row["strict_text_chars"]),
                "relaxed_text_chars": int(row["relaxed_text_chars"]),
                "title": row["title"],
                "author": row["author"],
                "greek_badness_score": row["greek_badness_score"],
                "len_greek": row["len_greek"],
                "mojibake_badness_score": row["mojibake_badness_score"],
                "needs_ocr": row["needs_ocr"],
                "is_empty": row["is_empty"],
                "ocr_success": row["ocr_success"],
                "is_historical_or_polytonic": row["is_historical_or_polytonic"],
                "text": text_value(texts[local_idx]),
            }
        )
    write_group_parquet(payload, shard_path, schema=EXACT_SURVIVOR_SCHEMA)
    return {
        "chunk_key": exact_survivor_chunk_key(file_path, row_group_index),
        "artifact_path": str(shard_path),
        "row_count": int(len(payload)),
        "survivor_digest": survivor_digest_from_payload(payload),
        "source_file_path": str(file_path),
        "row_group_index": int(row_group_index),
    }


def load_exact_survivor_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row["chunk_key"]): row for row in iter_parquet_rows(path)}


def write_exact_survivor_manifest(stage_root: Path) -> tuple[int, Path]:
    rows: list[dict[str, Any]] = []
    for shard_path in sorted((stage_root / "shards" / "exact_survivors").glob("*.parquet")):
        parquet_file = pq.ParquetFile(shard_path)
        if parquet_file.metadata.num_rows == 0:
            continue
        first_row = iter_parquet_rows(
            shard_path,
            columns=["file_path", "row_group_index", "doc_key", "content_hash_raw"],
            batch_size=4096,
        )
        if not first_row:
            continue
        source_file_path = Path(str(first_row[0]["file_path"]))
        row_group_index = int(first_row[0]["row_group_index"])
        rows.append(
            {
                "chunk_key": exact_survivor_chunk_key(source_file_path, row_group_index),
                "source_file_path": str(source_file_path),
                "row_group_index": int(row_group_index),
                "row_count": int(parquet_file.metadata.num_rows),
                "survivor_digest": survivor_digest_from_payload(first_row),
                "shard_path": str(shard_path),
            }
        )
    manifest_path = exact_survivor_manifest_path(stage_root)
    write_group_parquet(rows, manifest_path, schema=EXACT_SURVIVOR_MANIFEST_SCHEMA)
    return len(rows), manifest_path


def _run_exact_survivor_export_stage(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    run_root: Path,
    files: list[InputFile],
    max_workers: int,
) -> tuple[int, Path]:
    stage_root = run_root / "stage_01_exact"
    docs_exact_path = stage_root / "docs_exact.parquet"
    progress_path = progress_file_path(run_root, EXACT_SURVIVOR_STAGE)
    chunk_specs = [(input_file.path, row_group_index) for input_file in files for row_group_index in range(input_file.row_groups)]
    row_group_chunks = enumerate_row_group_chunks(files)
    chunk_keys = [exact_survivor_chunk_key(chunk.file_path, chunk.row_group_index) for chunk in row_group_chunks]
    register_stage_chunks(conn, run_id=run_id, stage=EXACT_SURVIVOR_STAGE, chunk_keys=chunk_keys)
    total_chunks, completed_chunks = stage_chunk_counts(conn, run_id=run_id, stage=EXACT_SURVIVOR_STAGE)
    upsert_stage_progress(
        conn,
        run_id=run_id,
        stage=EXACT_SURVIVOR_STAGE,
        status="running",
        total_chunks=total_chunks,
        completed_chunks=completed_chunks,
        progress_path=progress_path,
        payload={
            "run_id": run_id,
            "stage": EXACT_SURVIVOR_STAGE,
            "status": "running",
            "total_chunks": total_chunks,
            "completed_chunks": completed_chunks,
        },
    )
    pending_specs = [
        chunk
        for chunk in row_group_chunks
        if stage_chunk_status(
            conn,
            run_id=run_id,
            stage=EXACT_SURVIVOR_STAGE,
            chunk_key=exact_survivor_chunk_key(chunk.file_path, chunk.row_group_index),
        )
        != "completed"
    ]
    worker_count = effective_worker_count(max_workers, len(pending_specs))
    if worker_count == 1:
        for chunk in pending_specs:
            result = compute_exact_survivor_shard(
                docs_exact_path=docs_exact_path,
                file_path=chunk.file_path,
                row_group_index=chunk.row_group_index,
                row_offset=chunk.row_offset,
                stage_root=stage_root,
            )
            mark_stage_chunk_complete(
                conn,
                run_id=run_id,
                stage=EXACT_SURVIVOR_STAGE,
                chunk_key=str(result["chunk_key"]),
                artifact_path=None if result["artifact_path"] is None else Path(str(result["artifact_path"])),
                row_count=int(result["row_count"]),
            )
            total_chunks, completed_chunks = stage_chunk_counts(conn, run_id=run_id, stage=EXACT_SURVIVOR_STAGE)
            upsert_stage_progress(
                conn,
                run_id=run_id,
                stage=EXACT_SURVIVOR_STAGE,
                status="running",
                total_chunks=total_chunks,
                completed_chunks=completed_chunks,
                progress_path=progress_path,
                payload={
                    "run_id": run_id,
                    "stage": EXACT_SURVIVOR_STAGE,
                    "status": "running",
                    "total_chunks": total_chunks,
                    "completed_chunks": completed_chunks,
                },
            )
    elif pending_specs:
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=PROCESS_POOL_CONTEXT) as executor:
            pending_iter = iter(pending_specs)
            seed_specs = [next(pending_iter) for _ in range(min(worker_count, len(pending_specs)))]
            in_flight = {
                executor.submit(
                    compute_exact_survivor_shard,
                    docs_exact_path=docs_exact_path,
                    file_path=chunk.file_path,
                    row_group_index=chunk.row_group_index,
                    row_offset=chunk.row_offset,
                    stage_root=stage_root,
                ): chunk
                for chunk in seed_specs
            }
            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    in_flight.pop(future)
                    result = future.result()
                    mark_stage_chunk_complete(
                        conn,
                        run_id=run_id,
                        stage=EXACT_SURVIVOR_STAGE,
                        chunk_key=str(result["chunk_key"]),
                        artifact_path=None if result["artifact_path"] is None else Path(str(result["artifact_path"])),
                        row_count=int(result["row_count"]),
                    )
                    total_chunks, completed_chunks = stage_chunk_counts(conn, run_id=run_id, stage=EXACT_SURVIVOR_STAGE)
                    upsert_stage_progress(
                        conn,
                        run_id=run_id,
                        stage=EXACT_SURVIVOR_STAGE,
                        status="running",
                        total_chunks=total_chunks,
                        completed_chunks=completed_chunks,
                        progress_path=progress_path,
                        payload={
                            "run_id": run_id,
                            "stage": EXACT_SURVIVOR_STAGE,
                            "status": "running",
                            "total_chunks": total_chunks,
                            "completed_chunks": completed_chunks,
                        },
                    )
                    try:
                        next_chunk = next(pending_iter)
                    except StopIteration:
                        continue
                    in_flight[
                        executor.submit(
                            compute_exact_survivor_shard,
                            docs_exact_path=docs_exact_path,
                            file_path=next_chunk.file_path,
                            row_group_index=next_chunk.row_group_index,
                            row_offset=next_chunk.row_offset,
                            stage_root=stage_root,
                        )
                    ] = next_chunk
    manifest_rows, manifest_path = write_exact_survivor_manifest(stage_root)
    survivor_rows = int(
        conn.execute(
            """
            SELECT COALESCE(SUM(row_count), 0)
            FROM run_stage_chunks
            WHERE run_id = ? AND stage = ?
            """,
            (run_id, EXACT_SURVIVOR_STAGE),
        ).fetchone()[0]
        or 0
    )
    upsert_stage_progress(
        conn,
        run_id=run_id,
        stage=EXACT_SURVIVOR_STAGE,
        status="completed",
        total_chunks=total_chunks,
        completed_chunks=total_chunks,
        progress_path=progress_path,
        payload={
            "run_id": run_id,
            "stage": EXACT_SURVIVOR_STAGE,
            "status": "completed",
            "total_chunks": int(total_chunks),
            "completed_chunks": int(total_chunks),
            "manifest_path": str(manifest_path),
            "manifest_rows": int(manifest_rows),
            "survivor_rows": int(survivor_rows),
        },
    )
    return survivor_rows, manifest_path


def write_latest_snapshot(state_root: Path, *, run_id: str, input_root: Path, files: list[InputFile]) -> None:
    payload = {
        "run_id": run_id,
        "input_root": str(input_root),
        "generated_at": now_utc_iso(),
        "file_count": len(files),
        "total_rows": int(sum(item.total_rows for item in files)),
        "files": [
            {
                "file_path": str(item.path),
                "size_bytes": int(item.size_bytes),
                "mtime_ns": int(item.mtime_ns),
                "fingerprint": str(item.fingerprint),
                "row_groups": int(item.row_groups),
                "total_rows": int(item.total_rows),
            }
            for item in files
        ],
    }
    write_json_atomic(state_root / "latest_snapshot.json", payload)


def combine_parquet_files(paths: list[Path], destination: Path, *, schema: pa.Schema) -> int:
    temp_path = temp_output_path(destination)
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    try:
        for path in sorted(paths):
            if not path.exists():
                continue
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(batch_size=8192):
                table = pa.Table.from_batches([batch]).cast(schema)
                if writer is None:
                    writer = pq.ParquetWriter(temp_path, schema, compression=PARQUET_COMPRESSION)
                writer.write_table(table)
                rows_written += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        pq.write_table(schema.empty_table(), temp_path, compression=PARQUET_COMPRESSION)
    atomic_replace(temp_path, destination)
    return rows_written


def iter_parquet_batches(path: Path, *, columns: list[str] | None = None, batch_size: int = 2048) -> Any:
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(columns=columns, batch_size=batch_size):
        table = pa.Table.from_batches([batch])
        payload = table.to_pydict()
        row_count = table.num_rows
        yield [{name: payload[name][idx] for name in payload} for idx in range(row_count)]


def iter_parquet_rows(path: Path, *, columns: list[str] | None = None, batch_size: int = 2048) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch_rows in iter_parquet_batches(path, columns=columns, batch_size=batch_size):
        rows.extend(batch_rows)
    return rows


def iter_duckdb_query_rows(query: str, params: list[Any], *, batch_size: int = 2048, threads: int | None = None) -> Any:
    duck = connect_duckdb(threads=threads)
    try:
        reader = duck.execute(query, params).fetch_record_batch(rows_per_batch=batch_size)
        for batch in reader:
            table = pa.Table.from_batches([batch], schema=batch.schema)
            payload = table.to_pydict()
            for idx in range(table.num_rows):
                yield {name: payload[name][idx] for name in payload}
    finally:
        duck.close()


def exact_cluster_summaries_from_groups(source_path: Path, *, stage: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current_hash: str | None = None
    current_kept: str | None = None
    current_size = 0
    current_dropped = 0
    current_sources: set[str] = set()

    def flush() -> None:
        nonlocal current_hash, current_kept, current_size, current_dropped, current_sources
        if current_hash is None:
            return
        rows.append(
            {
                "cluster_id": f"{'strict' if stage == STRICT_STAGE else 'relaxed'}:{current_hash}",
                "decision_stage": stage,
                "cluster_size": int(current_size),
                "dropped_count": int(current_dropped),
                "kept_doc_key": str(current_kept),
                "mixed_source": bool(len(current_sources) > 1),
                "large_component_audit_flag": False,
                "narrow_margin_flag": False,
            }
        )
        current_hash = None
        current_kept = None
        current_size = 0
        current_dropped = 0
        current_sources = set()

    for batch_rows in iter_parquet_batches(
        source_path,
        columns=["group_hash", "group_size", "kept_doc_key", "member_source_dataset", "dropped"],
    ):
        for row in batch_rows:
            group_hash = str(row["group_hash"])
            if current_hash is None:
                current_hash = group_hash
            if group_hash != current_hash:
                flush()
                current_hash = group_hash
            current_kept = str(row["kept_doc_key"])
            current_size = int(row["group_size"])
            current_sources.add(str(row["member_source_dataset"]))
            current_dropped += int(bool(row["dropped"]))
    flush()
    return rows


def exact_survivor_clause() -> str:
    return """
        s.dropped = 0
        AND COALESCE(r.dropped, 0) = 0
    """


def exact_survivor_rows_for_group(
    *,
    docs_exact_path: Path,
    file_path: Path,
    row_group_index: int,
) -> list[dict[str, Any]]:
    table = ds.dataset(str(docs_exact_path), format="parquet").to_table(
        columns=[
            "source_dataset",
            "source_doc_id",
            "doc_key",
            "file_path",
            "row_group_index",
            "row_index_in_file",
            "content_hash_raw",
            "exact_strict_hash",
            "exact_relaxed_hash",
            "strict_text_chars",
            "relaxed_text_chars",
            "title",
            "author",
            "greek_badness_score",
            "len_greek",
            "mojibake_badness_score",
            "needs_ocr",
            "is_empty",
            "ocr_success",
            "is_historical_or_polytonic",
            "kept_after_exact",
        ],
        filter=(
            (ds.field("file_path") == str(file_path))
            & (ds.field("row_group_index") == int(row_group_index))
            & (ds.field("kept_after_exact") == 1)
        ),
    )
    rows = table.to_pylist()
    rows.sort(key=lambda row: int(row["row_index_in_file"]))
    return rows


def unicode_word_tokens(text: str) -> list[str]:
    return WORD_TOKEN_RE.findall(text)


def normalize_near_text(text: str, *, greek_diacritic_policy: str = DEFAULT_GREEK_DIACRITIC_POLICY) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.translate(ZERO_WIDTH_TRANSLATION)
    normalized = unicodedata.normalize("NFC", normalized)
    normalized = LINE_WRAP_DEHYPHEN_RE.sub("", normalized)
    return normalize_exact_relaxed_text(normalized, greek_diacritic_policy=greek_diacritic_policy)


def shingle_version(*, shingle_mode: str, shingle_size: int) -> str:
    return f"shingle_{shingle_mode}_{shingle_size}_v1"


_PERMUTATION_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def permutation_params(num_perm: int) -> tuple[np.ndarray, np.ndarray]:
    cached = _PERMUTATION_CACHE.get(num_perm)
    if cached is not None:
        return cached
    a_values: list[int] = []
    b_values: list[int] = []
    for idx in range(num_perm):
        seed = blake3(f"minhash-perm:{idx}".encode("utf-8")).digest(length=16)
        a_value = int.from_bytes(seed[:8], "little") % MINHASH_MODULUS
        if a_value == 0:
            a_value = 1
        if a_value % 2 == 0:
            a_value += 1
            if a_value >= MINHASH_MODULUS:
                a_value = 1
        b_value = int.from_bytes(seed[8:], "little") % MINHASH_MODULUS
        a_values.append(a_value)
        b_values.append(b_value)
    params = (np.array(a_values, dtype=np.uint64), np.array(b_values, dtype=np.uint64))
    _PERMUTATION_CACHE[num_perm] = params
    return params


def shingle_hashes_from_text(*, near_text: str, shingle_mode: str, shingle_size: int) -> tuple[list[int], int, int]:
    tokens = unicode_word_tokens(near_text)
    token_count = len(tokens)
    char_count = len(near_text)
    shingle_hashes: set[int] = set()
    if shingle_mode == "token":
        if token_count < SHORT_DOC_TOKEN_THRESHOLD:
            return [], token_count, char_count
        if token_count < shingle_size:
            return [], token_count, char_count
        for idx in range(token_count - shingle_size + 1):
            shingle = "\x1f".join(tokens[idx : idx + shingle_size])
            shingle_hashes.add(int.from_bytes(blake3(shingle.encode("utf-8")).digest(length=8), "little"))
    elif shingle_mode == "char":
        if char_count < shingle_size:
            return [], token_count, char_count
        for idx in range(char_count - shingle_size + 1):
            shingle = near_text[idx : idx + shingle_size]
            shingle_hashes.add(int.from_bytes(blake3(shingle.encode("utf-8")).digest(length=8), "little"))
    else:
        raise ValueError(f"unsupported shingle_mode: {shingle_mode}")
    return sorted(shingle_hashes), token_count, char_count


def _mersenne_reduce(values: np.ndarray) -> np.ndarray:
    reduced = (values & np.uint64(MINHASH_MODULUS)) + (values >> 61)
    reduced = (reduced & np.uint64(MINHASH_MODULUS)) + (reduced >> 61)
    return np.where(
        reduced >= np.uint64(MINHASH_MODULUS),
        reduced - np.uint64(MINHASH_MODULUS),
        reduced,
    ).astype(np.uint64, copy=False)


def _minhash_permutations_for_block(
    shingle_hashes: np.ndarray,
    *,
    a_values: np.ndarray,
    b_values: np.ndarray,
) -> np.ndarray:
    shingle_values = _mersenne_reduce(np.ascontiguousarray(shingle_hashes, dtype=np.uint64).reshape(-1, 1))
    a_view = a_values.reshape(1, -1)
    b_view = b_values.reshape(1, -1)
    mask32 = np.uint64(0xFFFFFFFF)
    x_lo = shingle_values & mask32
    x_hi = shingle_values >> 32
    a_lo = a_view & mask32
    a_hi = a_view >> 32
    lo_lo = a_lo * x_lo
    cross = (a_hi * x_lo) + (a_lo * x_hi)
    low = lo_lo + ((cross & mask32) << 32)
    carry = (low < lo_lo).astype(np.uint64)
    high = (a_hi * x_hi) + (cross >> 32) + carry
    product_mod = _mersenne_reduce((low & np.uint64(MINHASH_MODULUS)) + (high << 3) + (low >> 61))
    return _mersenne_reduce(product_mod + b_view)


def minhash_signature(shingle_hashes: list[int], *, num_perm: int) -> np.ndarray:
    if not shingle_hashes:
        return np.zeros(num_perm, dtype=np.uint64)
    a_values, b_values = permutation_params(num_perm)
    signature = np.full(num_perm, np.uint64(MINHASH_MODULUS), dtype=np.uint64)
    shingle_array = np.ascontiguousarray(shingle_hashes, dtype=np.uint64)
    for start in range(0, shingle_array.size, MINHASH_BLOCK_SIZE):
        block = shingle_array[start : start + MINHASH_BLOCK_SIZE]
        permuted = _minhash_permutations_for_block(block, a_values=a_values, b_values=b_values)
        signature = np.minimum(signature, np.min(permuted, axis=0))
    return signature


def signature_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    return float(np.mean(left == right))


def band_bucket_hash(signature: np.ndarray, *, band_index: int, rows_per_band: int) -> str:
    start = band_index * rows_per_band
    end = start + rows_per_band
    band_values = np.ascontiguousarray(signature[start:end], dtype=np.uint64)
    return blake3(band_values.tobytes()).hexdigest()


def deterministic_sample(rows: list[dict[str, Any]], *, key_fields: list[str], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or not rows:
        return []
    ranked = sorted(
        rows,
        key=lambda row: blake3(
            "\0".join(str(row[field]) for field in key_fields).encode("utf-8")
        ).hexdigest(),
    )
    return ranked[: min(limit, len(ranked))]


def deterministic_sample_rank(row: dict[str, Any], *, key_fields: list[str]) -> str:
    return blake3("\0".join(str(row[field]) for field in key_fields).encode("utf-8")).hexdigest()


def update_streaming_deterministic_sample(
    state: list[tuple[str, dict[str, Any]]],
    *,
    row: dict[str, Any],
    key_fields: list[str],
    limit: int,
) -> None:
    if limit <= 0:
        return
    state.append((deterministic_sample_rank(row, key_fields=key_fields), dict(row)))
    state.sort(key=lambda item: item[0])
    del state[limit:]


def update_streaming_ranked_top_rows(
    state: list[tuple[tuple[Any, ...], dict[str, Any]]],
    *,
    row: dict[str, Any],
    rank: tuple[Any, ...],
    limit: int,
) -> None:
    if limit <= 0:
        return
    state.append((rank, dict(row)))
    state.sort(key=lambda item: item[0])
    del state[limit:]


def signature_metadata_dtype() -> np.dtype[Any]:
    return np.dtype(
        [
            ("token_count", np.int32),
            ("char_count", np.int32),
            ("near_text_chars", np.int32),
            ("shingle_count", np.int32),
            ("shingle_size", np.int32),
            ("shingle_mode", "U8"),
        ]
    )


def signature_matrix_path(signatures_path: Path) -> Path:
    return signatures_path.with_suffix(".npy")


def signature_doc_keys_path(signatures_path: Path) -> Path:
    return signatures_path.with_name("signature_doc_keys.txt")


def signature_metadata_path(signatures_path: Path) -> Path:
    return signatures_path.with_name("signature_metadata.npy")


def build_signature_matrix_artifacts(signatures_path: Path, *, num_perm: int) -> tuple[Path, Path, Path]:
    matrix_path = signature_matrix_path(signatures_path)
    doc_keys_path = signature_doc_keys_path(signatures_path)
    metadata_path = signature_metadata_path(signatures_path)
    parquet_file = pq.ParquetFile(signatures_path)
    matrix_temp_path = temp_output_path(matrix_path)
    doc_keys_temp_path = temp_output_path(doc_keys_path)
    metadata_temp_path = temp_output_path(metadata_path)
    metadata_dtype = signature_metadata_dtype()
    if parquet_file.metadata.num_rows == 0:
        with matrix_temp_path.open("wb") as handle:
            np.save(handle, np.zeros((0, num_perm), dtype=np.uint64))
        with metadata_temp_path.open("wb") as handle:
            np.save(handle, np.zeros((0,), dtype=metadata_dtype))
        doc_keys_temp_path.write_text("", encoding="utf-8")
    else:
        matrix = np.lib.format.open_memmap(
            matrix_temp_path,
            mode="w+",
            dtype=np.uint64,
            shape=(parquet_file.metadata.num_rows, num_perm),
        )
        metadata_values = np.lib.format.open_memmap(
            metadata_temp_path,
            mode="w+",
            dtype=metadata_dtype,
            shape=(parquet_file.metadata.num_rows,),
        )
        row_index = 0
        try:
            with doc_keys_temp_path.open("w", encoding="utf-8") as handle:
                for batch in parquet_file.iter_batches(
                    columns=[
                        "doc_key",
                        "token_count",
                        "char_count",
                        "near_text_chars",
                        "shingle_count",
                        "shingle_mode",
                        "shingle_size",
                        "signature",
                    ],
                    batch_size=2048,
                ):
                    table = pa.Table.from_batches([batch])
                    payload = table.to_pydict()
                    for idx in range(table.num_rows):
                        signature_values = np.asarray(payload["signature"][idx], dtype=np.uint64)
                        if signature_values.size != num_perm:
                            raise ValueError(
                                f"signature width mismatch while building matrix for {signatures_path}: "
                                f"expected {num_perm}, saw {signature_values.size}"
                            )
                        matrix[row_index, :] = signature_values
                        metadata_values[row_index] = (
                            int(payload["token_count"][idx]),
                            int(payload["char_count"][idx]),
                            int(payload["near_text_chars"][idx]),
                            int(payload["shingle_count"][idx]),
                            int(payload["shingle_size"][idx]),
                            str(payload["shingle_mode"][idx]),
                        )
                        handle.write(f"{payload['doc_key'][idx]}\n")
                        row_index += 1
            matrix.flush()
            metadata_values.flush()
        finally:
            del matrix
            del metadata_values
    atomic_replace(matrix_temp_path, matrix_path)
    atomic_replace(metadata_temp_path, metadata_path)
    atomic_replace(doc_keys_temp_path, doc_keys_path)
    return matrix_path, doc_keys_path, metadata_path


def load_signature_index(signatures_path: Path) -> tuple[SignatureLookup, SignatureMetadataLookup]:
    row_by_doc_key: dict[str, int] = {}
    parquet_file = pq.ParquetFile(signatures_path)
    matrix_path = signature_matrix_path(signatures_path)
    doc_keys_path = signature_doc_keys_path(signatures_path)
    metadata_path = signature_metadata_path(signatures_path)
    if matrix_path.exists() and doc_keys_path.exists() and metadata_path.exists():
        matrix = np.load(matrix_path, mmap_mode="r")
        metadata_values = np.load(metadata_path, mmap_mode="r")
        with doc_keys_path.open("r", encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                row_by_doc_key[line.rstrip("\n")] = row_index
        return (
            SignatureLookup(row_by_doc_key=row_by_doc_key, matrix=matrix),
            SignatureMetadataLookup(row_by_doc_key=row_by_doc_key, values=metadata_values),
        )
    signature_rows: list[np.ndarray] = []
    metadata_rows: list[tuple[int, int, int, int, int, str]] = []
    for batch in parquet_file.iter_batches(batch_size=2048):
        table = pa.Table.from_batches([batch])
        payload = table.to_pydict()
        for idx in range(table.num_rows):
            doc_key = str(payload["doc_key"][idx])
            row_by_doc_key[doc_key] = len(signature_rows)
            signature_rows.append(np.asarray(payload["signature"][idx], dtype=np.uint64))
            metadata_rows.append(
                (
                    int(payload["token_count"][idx]),
                    int(payload["char_count"][idx]),
                    int(payload["near_text_chars"][idx]),
                    int(payload["shingle_count"][idx]),
                    int(payload["shingle_size"][idx]),
                    str(payload["shingle_mode"][idx]),
                )
            )
    if signature_rows:
        matrix = np.vstack(signature_rows)
        metadata_values = np.array(metadata_rows, dtype=signature_metadata_dtype())
    else:
        matrix = np.zeros((0, 0), dtype=np.uint64)
        metadata_values = np.zeros((0,), dtype=signature_metadata_dtype())
    return (
        SignatureLookup(row_by_doc_key=row_by_doc_key, matrix=matrix),
        SignatureMetadataLookup(row_by_doc_key=row_by_doc_key, values=metadata_values),
    )


_CANDIDATE_WORKER_SIGNATURE_MAP: SignatureLookup | None = None
_CANDIDATE_WORKER_SIGNATURE_META: SignatureMetadataLookup | None = None
_CANDIDATE_WORKER_SIGNATURE_PATH: Path | None = None
_CLUSTER_WORKER_SIGNATURE_MAP: SignatureLookup | None = None
_CLUSTER_WORKER_SIGNATURE_META: SignatureMetadataLookup | None = None
_CLUSTER_WORKER_ADJACENCY: dict[str, set[str]] | None = None


def candidate_process_pool_context():
    raw = os.environ.get("GLOSSAPI_NEAR_CANDIDATE_START_METHOD", "").strip().lower()
    if raw:
        return mp.get_context(raw)
    if os.name == "posix" and "fork" in mp.get_all_start_methods():
        return mp.get_context("fork")
    return PROCESS_POOL_CONTEXT


def preload_candidate_worker_state(signatures_path: Path) -> None:
    global _CANDIDATE_WORKER_SIGNATURE_MAP, _CANDIDATE_WORKER_SIGNATURE_META, _CANDIDATE_WORKER_SIGNATURE_PATH
    if _CANDIDATE_WORKER_SIGNATURE_PATH == signatures_path and _CANDIDATE_WORKER_SIGNATURE_MAP is not None:
        return
    _CANDIDATE_WORKER_SIGNATURE_MAP, _CANDIDATE_WORKER_SIGNATURE_META = load_signature_index(signatures_path)
    _CANDIDATE_WORKER_SIGNATURE_PATH = signatures_path


def candidate_pool_initializer_config(signatures_path: Path):
    context = candidate_process_pool_context()
    if context.get_start_method() == "fork":
        preload_candidate_worker_state(signatures_path)
        return context, None, ()
    return context, init_candidate_worker, (str(signatures_path),)


def init_candidate_worker(signatures_path: str) -> None:
    global _CANDIDATE_WORKER_SIGNATURE_MAP, _CANDIDATE_WORKER_SIGNATURE_META, _CANDIDATE_WORKER_SIGNATURE_PATH
    signatures_path_obj = Path(signatures_path)
    _CANDIDATE_WORKER_SIGNATURE_MAP, _CANDIDATE_WORKER_SIGNATURE_META = load_signature_index(signatures_path_obj)
    _CANDIDATE_WORKER_SIGNATURE_PATH = signatures_path_obj


def near_cluster_process_pool_context():
    raw = os.environ.get("GLOSSAPI_NEAR_CLUSTER_START_METHOD", "").strip().lower()
    if raw:
        return mp.get_context(raw)
    if os.name == "posix" and "fork" in mp.get_all_start_methods():
        return mp.get_context("fork")
    return PROCESS_POOL_CONTEXT


def preload_cluster_worker_state(
    *,
    signature_map: SignatureLookup,
    signature_meta: SignatureMetadataLookup,
    adjacency: dict[str, set[str]],
) -> None:
    global _CLUSTER_WORKER_SIGNATURE_MAP, _CLUSTER_WORKER_SIGNATURE_META, _CLUSTER_WORKER_ADJACENCY
    _CLUSTER_WORKER_SIGNATURE_MAP = signature_map
    _CLUSTER_WORKER_SIGNATURE_META = signature_meta
    _CLUSTER_WORKER_ADJACENCY = adjacency


def cluster_pool_initializer_config(
    *,
    signature_map: SignatureLookup,
    signature_meta: SignatureMetadataLookup,
    adjacency: dict[str, set[str]],
):
    context = near_cluster_process_pool_context()
    if context.get_start_method() == "fork":
        preload_cluster_worker_state(
            signature_map=signature_map,
            signature_meta=signature_meta,
            adjacency=adjacency,
        )
        return context, None, ()
    return context, None, ()


def near_component_subgraphs(nodes: set[str], adjacency: dict[str, set[str]]) -> list[set[str]]:
    pending = set(nodes)
    components: list[set[str]] = []
    while pending:
        start = pending.pop()
        component = {start}
        queue: deque[str] = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency.get(current, set()):
                if neighbor not in pending:
                    continue
                pending.remove(neighbor)
                component.add(neighbor)
                queue.append(neighbor)
        components.append(component)
    return components


def cluster_id_for_doc_keys(prefix: str, doc_keys: list[str]) -> str:
    payload = "\0".join(sorted(doc_keys))
    return f"{prefix}:{blake3(payload.encode('utf-8')).hexdigest()}"


def connect_candidate_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        PRAGMA journal_mode=DELETE;
        PRAGMA synchronous=NORMAL;

        CREATE TABLE IF NOT EXISTS candidate_pairs (
            doc_key_left TEXT NOT NULL,
            doc_key_right TEXT NOT NULL,
            estimated_jaccard REAL NOT NULL,
            shingle_mode TEXT NOT NULL,
            token_count_left INTEGER NOT NULL,
            token_count_right INTEGER NOT NULL,
            length_ratio REAL NOT NULL,
            likely_containment_flag INTEGER NOT NULL,
            accepted_reason TEXT NOT NULL,
            bucket_match_bands INTEGER NOT NULL,
            PRIMARY KEY (doc_key_left, doc_key_right)
        );
        """
    )
    conn.commit()
    return conn


def aggregate_candidate_shards_to_parquet(stage_root: Path) -> tuple[int, Path]:
    candidate_db_path = stage_root / "candidate_pairs.sqlite"
    if candidate_db_path.exists():
        candidate_db_path.unlink()
    conn = connect_candidate_db(candidate_db_path)
    try:
        for path in sorted((stage_root / "shards" / "candidate_pairs").glob("band_*/*.parquet")):
            for batch_rows in iter_parquet_batches(path):
                if not batch_rows:
                    continue
                conn.executemany(
                    """
                    INSERT INTO candidate_pairs (
                        doc_key_left,
                        doc_key_right,
                        estimated_jaccard,
                        shingle_mode,
                        token_count_left,
                        token_count_right,
                        length_ratio,
                        likely_containment_flag,
                        accepted_reason,
                        bucket_match_bands
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(doc_key_left, doc_key_right) DO UPDATE SET
                        bucket_match_bands = candidate_pairs.bucket_match_bands + excluded.bucket_match_bands
                    """,
                    [
                        (
                            str(row["doc_key_left"]),
                            str(row["doc_key_right"]),
                            float(row["estimated_jaccard"]),
                            str(row["shingle_mode"]),
                            int(row["token_count_left"]),
                            int(row["token_count_right"]),
                            float(row["length_ratio"]),
                            int(bool(row["likely_containment_flag"])),
                            str(row["accepted_reason"]),
                            int(row["bucket_match_bands"]),
                        )
                        for row in batch_rows
                    ],
                )
                conn.commit()
        candidate_pairs_path = stage_root / "candidate_pairs.parquet"
        candidate_pair_rows = _stream_query_to_parquet(
            conn,
            """
            SELECT
                doc_key_left,
                doc_key_right,
                estimated_jaccard,
                shingle_mode,
                token_count_left,
                token_count_right,
                length_ratio,
                likely_containment_flag,
                accepted_reason,
                bucket_match_bands
            FROM candidate_pairs
            ORDER BY doc_key_left, doc_key_right
            """,
            (),
            candidate_pairs_path,
            schema=CANDIDATE_PAIR_SCHEMA,
        )
    finally:
        conn.close()
    return candidate_pair_rows, candidate_db_path


def aggregate_bucket_summary_shards(stage_root: Path) -> tuple[int, Path]:
    bucket_summary_path = stage_root / "bucket_summary.parquet"
    bucket_summary_rows = combine_parquet_files(
        sorted((stage_root / "shards" / "bucket_summaries").glob("*.parquet")),
        bucket_summary_path,
        schema=BUCKET_SUMMARY_SCHEMA,
    )
    return bucket_summary_rows, bucket_summary_path


def summarize_bucket_summary_shards(stage_root: Path) -> tuple[int, Path]:
    bucket_summary_path = stage_root / "bucket_summary.parquet"
    if bucket_summary_path.exists():
        return parquet_num_rows(bucket_summary_path), bucket_summary_path
    shard_paths = sorted((stage_root / "shards" / "bucket_summaries").glob("*.parquet"))
    return (
        sum(parquet_num_rows(path) for path in shard_paths),
        stage_root / "shards" / "bucket_summaries",
    )


def aggregate_touched_doc_shards(stage_root: Path) -> tuple[int, Path]:
    touched_doc_keys: set[str] = set()
    for path in sorted((stage_root / "shards" / "touched_doc_keys").glob("*.parquet")):
        for batch_rows in iter_parquet_batches(path):
            for row in batch_rows:
                touched_doc_keys.add(str(row["doc_key"]))
    touched_doc_path = stage_root / "touched_doc_keys.parquet"
    write_group_parquet(
        [{"doc_key": doc_key} for doc_key in sorted(touched_doc_keys)],
        touched_doc_path,
        schema=TOUCHED_DOC_SCHEMA,
    )
    return len(touched_doc_keys), touched_doc_path


def build_near_cluster_chunk_specs(
    components: list[set[str]],
    *,
    target_doc_count: int = DEFAULT_NEAR_CLUSTER_CHUNK_DOCS,
    max_components: int = DEFAULT_NEAR_CLUSTER_MAX_COMPONENTS,
) -> list[dict[str, Any]]:
    ordered_components = [
        {
            "component_hash": blake3("\0".join(sorted(component)).encode("utf-8")).hexdigest(),
            "doc_keys": sorted(component),
        }
        for component in components
    ]
    ordered_components.sort(key=lambda item: (item["doc_keys"][0], item["component_hash"]))
    chunk_specs: list[dict[str, Any]] = []
    current_components: list[dict[str, Any]] = []
    current_doc_count = 0

    def flush_chunk() -> None:
        nonlocal current_components, current_doc_count
        if not current_components:
            return
        chunk_index = len(chunk_specs)
        first_hash = str(current_components[0]["component_hash"])
        chunk_specs.append(
            {
                "chunk_key": f"component_chunk:{chunk_index:05d}:{first_hash}",
                "chunk_index": chunk_index,
                "components": current_components,
                "doc_count": current_doc_count,
            }
        )
        current_components = []
        current_doc_count = 0

    for component in ordered_components:
        component_doc_count = len(component["doc_keys"])
        if current_components and (
            current_doc_count + component_doc_count > target_doc_count or len(current_components) >= max_components
        ):
            flush_chunk()
        current_components.append(component)
        current_doc_count += component_doc_count
    flush_chunk()
    return chunk_specs


def build_near_cluster_singleton_chunk_specs(
    doc_keys: list[str],
    *,
    start_index: int = 0,
    target_doc_count: int = DEFAULT_NEAR_CLUSTER_CHUNK_DOCS,
) -> list[dict[str, Any]]:
    if len(doc_keys) <= max(target_doc_count, DEFAULT_NEAR_CLUSTER_MAX_COMPONENTS):
        base_specs = build_near_cluster_chunk_specs([{doc_key} for doc_key in doc_keys])
        adjusted_specs: list[dict[str, Any]] = []
        for offset, spec in enumerate(base_specs):
            singleton_keys = [str(component["doc_keys"][0]) for component in list(spec["components"])]
            chunk_index = start_index + offset
            first_doc_key = singleton_keys[0]
            adjusted_specs.append(
                {
                    "chunk_key": f"singleton_chunk:{chunk_index:05d}:{blake3(first_doc_key.encode('utf-8')).hexdigest()[:16]}",
                    "chunk_index": chunk_index,
                    "singleton_doc_keys": singleton_keys,
                    "doc_count": len(singleton_keys),
                }
            )
        return adjusted_specs
    chunk_specs: list[dict[str, Any]] = []
    for offset in range(0, len(doc_keys), target_doc_count):
        chunk_doc_keys = list(doc_keys[offset : offset + target_doc_count])
        if not chunk_doc_keys:
            continue
        chunk_index = start_index + len(chunk_specs)
        first_doc_key = str(chunk_doc_keys[0])
        chunk_specs.append(
            {
                "chunk_key": f"singleton_chunk:{chunk_index:05d}:{blake3(first_doc_key.encode('utf-8')).hexdigest()[:16]}",
                "chunk_index": chunk_index,
                "singleton_doc_keys": chunk_doc_keys,
                "doc_count": len(chunk_doc_keys),
            }
        )
    return chunk_specs


def uses_legacy_near_cluster_component_only_layout(chunk_keys: list[str]) -> bool:
    data_chunk_keys = [str(chunk_key) for chunk_key in chunk_keys if str(chunk_key) != "reuse:previous_components"]
    return bool(data_chunk_keys) and all(
        chunk_key.startswith("component_chunk:") and not chunk_key.startswith("singleton_chunk:")
        for chunk_key in data_chunk_keys
    )


def load_run_config_payload(run_root: Path) -> dict[str, Any] | None:
    config_path = run_root / "run_config.json"
    if not config_path.exists():
        return None
    return json.loads(config_path.read_text()).get("config", {})


def config_matches_keys(
    previous_config: dict[str, Any] | None,
    current_config: dict[str, Any],
    *,
    keys: list[str],
) -> bool:
    if previous_config is None:
        return False
    return all(previous_config.get(key) == current_config.get(key) for key in keys)


def latest_success_run_root(state_root: Path) -> Path | None:
    latest_path = state_root / "latest_success.json"
    if not latest_path.exists():
        return None
    payload = json.loads(latest_path.read_text())
    run_root = payload.get("run_root")
    return None if not run_root else Path(str(run_root))


def latest_compatible_run_root(
    *,
    state_root: Path,
    current_run_root: Path,
    current_config: dict[str, Any],
    keys: list[str],
    required_relative_paths: list[Path],
) -> Path | None:
    previous_run_root = latest_success_run_root(state_root)
    if previous_run_root is None or previous_run_root == current_run_root:
        return None
    previous_config = load_run_config_payload(previous_run_root)
    if not config_matches_keys(previous_config, current_config, keys=keys):
        return None
    if not all((previous_run_root / relative_path).exists() for relative_path in required_relative_paths):
        return None
    return previous_run_root


def near_cluster_chunk_shard_path(stage_root: Path, *, chunk_index: int) -> Path:
    shard_root = stage_root / "shards" / "near_clusters"
    ensure_dir(shard_root)
    return shard_root / f"chunk_{chunk_index:05d}.parquet"


def near_cluster_summary_shard_path(stage_root: Path, *, chunk_index: int) -> Path:
    shard_root = stage_root / "shards" / "cluster_summaries"
    ensure_dir(shard_root)
    return shard_root / f"chunk_{chunk_index:05d}.parquet"


def stage_chunk_counts(conn: sqlite3.Connection, *, run_id: str, stage: str) -> tuple[int, int]:
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS total_chunks,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed_chunks
        FROM run_stage_chunks
        WHERE run_id = ? AND stage = ?
        """,
        (run_id, stage),
    ).fetchone()
    return int(row["total_chunks"] or 0), int(row["completed_chunks"] or 0)


def ensure_run_config(*, run_root: Path, config: dict[str, Any], resume: bool) -> tuple[str, str]:
    run_id = run_root.name
    digest = config_hash(config)
    config_path = run_root / "run_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous.get("config_hash") != digest:
            raise ValueError(f"existing run_root has a different config hash: {run_root}")
        if not resume:
            raise ValueError(f"run_root already exists; pass --resume to continue: {run_root}")
    elif resume:
        raise ValueError(f"cannot resume; run_config.json is missing under {run_root}")
    else:
        write_json_atomic(
            config_path,
            {
                "run_id": run_id,
                "created_at": now_utc_iso(),
                "config_hash": digest,
                "config": config,
            },
        )
    return run_id, digest


def pipeline_config_payload(
    *,
    input_root: Path,
    state_root: Path,
    run_root: Path,
    greek_diacritic_policy: str,
    exact_only: bool,
    max_workers: int,
    minhash_threshold: float,
    num_perm: int,
    bands: int,
    rows_per_band: int,
    shingle_mode: str,
    shingle_size: int,
    large_component_threshold: int,
    max_bucket_size: int,
) -> dict[str, Any]:
    greek_diacritic_policy = validate_greek_diacritic_policy(greek_diacritic_policy)
    payload = config_payload(
        input_root=input_root,
        state_root=state_root,
        run_root=run_root,
        greek_diacritic_policy=greek_diacritic_policy,
    )
    payload.update(
        {
            "exact_only": bool(exact_only),
            "max_workers": int(max_workers),
            "tokenization_version": TOKENIZATION_VERSION,
            "minhash_version": MINHASH_VERSION,
            "lsh_version": LSH_VERSION,
            "selection_version": SELECTION_VERSION,
            "survivor_export_version": SURVIVOR_EXPORT_VERSION,
            "near_incremental_version": NEAR_INCREMENTAL_VERSION,
            "oversized_bucket_strategy_version": OVERSIZED_BUCKET_STRATEGY_VERSION,
            "minhash_threshold": float(minhash_threshold),
            "num_perm": int(num_perm),
            "bands": int(bands),
            "rows_per_band": int(rows_per_band),
            "shingle_mode": shingle_mode,
            "shingle_size": int(shingle_size),
            "large_component_threshold": int(large_component_threshold),
            "max_bucket_size": int(max_bucket_size),
        }
    )
    return payload


def _run_exact_stage_core(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    input_root: Path,
    state_root: Path,
    run_root: Path,
    files: list[InputFile],
    greek_diacritic_policy: str,
    max_workers: int,
    emit_survivor_export: bool,
    resume: bool,
) -> dict[str, Any]:
    stage_root = run_root / "stage_01_exact"
    ensure_dir(stage_root)
    summary_path = stage_root / "summary.json"
    exact_required_paths = [
        run_docs_inventory_path(run_root),
        run_root / "snapshot_manifest.parquet",
        stage_root / "strict_exact_groups.parquet",
        stage_root / "strict_exact_drop_list.parquet",
        stage_root / "relaxed_exact_groups.parquet",
        stage_root / "relaxed_exact_drop_list.parquet",
        exact_membership_path(run_root, STRICT_STAGE),
        exact_membership_path(run_root, RELAXED_STAGE),
        stage_root / "docs_exact.parquet",
    ]
    if emit_survivor_export:
        exact_required_paths.append(stage_root / "exact_survivor_manifest.parquet")
    validate_registered_input_snapshot(conn, run_id=run_id, files=files, resume=resume)
    register_input_files(conn, run_id=run_id, files=files)
    write_latest_snapshot(state_root, run_id=run_id, input_root=input_root, files=files)
    register_chunks(conn, run_id=run_id, files=files)
    total_chunks = sum(item.row_groups for item in files)
    progress_path = progress_file_path(run_root, "stage_01_exact")
    completed_chunks = conn.execute(
        """
        SELECT COALESCE(SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END), 0)
        FROM run_chunks
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()[0]
    existing_summary = read_json_if_exists(summary_path)
    if existing_summary is not None and all_paths_exist(exact_required_paths):
        upsert_stage_progress(
            conn,
            run_id=run_id,
            stage="stage_01_exact",
            status="completed",
            total_chunks=total_chunks,
            completed_chunks=total_chunks,
            progress_path=progress_path,
            payload={
                "run_id": run_id,
                "stage": "stage_01_exact",
                "status": "completed",
                "total_chunks": int(total_chunks),
                "completed_chunks": int(total_chunks),
                "summary_path": str(summary_path),
            },
        )
        return existing_summary
    upsert_stage_progress(
        conn,
        run_id=run_id,
        stage="stage_01_exact",
        status="running",
        total_chunks=total_chunks,
        completed_chunks=int(completed_chunks or 0),
        progress_path=progress_path,
        payload={
            "run_id": run_id,
            "stage": "stage_01_exact",
            "status": "running",
            "total_chunks": int(total_chunks),
            "completed_chunks": int(completed_chunks or 0),
        },
    )
    start = time.perf_counter()
    pending_chunks = [
        chunk
        for chunk in enumerate_row_group_chunks(files)
        if row_group_status(conn, run_id=run_id, file_path=chunk.file_path, row_group_index=chunk.row_group_index) != "completed"
    ]
    worker_count = effective_worker_count(max_workers, len(pending_chunks))
    if worker_count == 1:
        for chunk in pending_chunks:
            result = prepare_exact_row_group_result(
                state_root=state_root,
                run_id=run_id,
                chunk=chunk,
                greek_diacritic_policy=greek_diacritic_policy,
            )
            commit_exact_row_group_result(conn, run_id=run_id, result=result)
            completed_chunks += 1
            upsert_stage_progress(
                conn,
                run_id=run_id,
                stage="stage_01_exact",
                status="running",
                total_chunks=total_chunks,
                completed_chunks=int(completed_chunks or 0),
                progress_path=progress_path,
                payload={
                    "run_id": run_id,
                    "stage": "stage_01_exact",
                    "status": "running",
                    "total_chunks": int(total_chunks),
                    "completed_chunks": int(completed_chunks or 0),
                },
            )
    elif pending_chunks:
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=PROCESS_POOL_CONTEXT) as executor:
            pending_iter = iter(pending_chunks)
            in_flight = {
                executor.submit(
                    prepare_exact_row_group_result,
                    state_root=state_root,
                    run_id=run_id,
                    chunk=chunk,
                    greek_diacritic_policy=greek_diacritic_policy,
                ): chunk
                for chunk in [next(pending_iter) for _ in range(min(worker_count, len(pending_chunks)))]
            }
            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    in_flight.pop(future)
                    result = future.result()
                    commit_exact_row_group_result(conn, run_id=run_id, result=result)
                    completed_chunks += 1
                    upsert_stage_progress(
                        conn,
                        run_id=run_id,
                        stage="stage_01_exact",
                        status="running",
                        total_chunks=total_chunks,
                        completed_chunks=int(completed_chunks or 0),
                        progress_path=progress_path,
                        payload={
                            "run_id": run_id,
                            "stage": "stage_01_exact",
                            "status": "running",
                            "total_chunks": int(total_chunks),
                            "completed_chunks": int(completed_chunks or 0),
                        },
                    )
                    try:
                        next_chunk = next(pending_iter)
                    except StopIteration:
                        continue
                    in_flight[
                        executor.submit(
                            prepare_exact_row_group_result,
                            state_root=state_root,
                            run_id=run_id,
                            chunk=next_chunk,
                            greek_diacritic_policy=greek_diacritic_policy,
                        )
                    ] = next_chunk
    trace_path = progress_dir(run_root) / "exact_stage_trace.log"
    if resume:
        existing_summary = load_existing_exact_stage_summary(
            run_root=run_root,
            emit_survivor_export=emit_survivor_export,
        )
        if existing_summary is not None:
            append_debug_trace(trace_path, "exact_stage:reuse_existing_summary")
            upsert_stage_progress(
                conn,
                run_id=run_id,
                stage="stage_01_exact",
                status="completed",
                total_chunks=total_chunks,
                completed_chunks=total_chunks,
                progress_path=progress_path,
                payload={
                    "run_id": run_id,
                    "stage": "stage_01_exact",
                    "status": "completed",
                    "total_chunks": int(total_chunks),
                    "completed_chunks": int(total_chunks),
                    "summary_path": str(stage_root / "summary.json"),
                },
            )
            return existing_summary
    append_debug_trace(trace_path, "exact_stage:run_docs_inventory:start")
    inventory_rows = write_run_docs_inventory(conn, run_id=run_id, path=run_docs_inventory_path(run_root))
    append_debug_trace(trace_path, f"exact_stage:run_docs_inventory:done rows={inventory_rows}")
    append_debug_trace(trace_path, "exact_stage:strict:start")
    strict_summary = build_stage_results(
        conn,
        run_root=run_root,
        run_id=run_id,
        stage=STRICT_STAGE,
        hash_column="exact_strict_hash",
        text_length_field="strict_text_chars",
        groups_path=stage_root / "strict_exact_groups.parquet",
        drops_path=stage_root / "strict_exact_drop_list.parquet",
    )
    append_debug_trace(trace_path, f"exact_stage:strict:done duplicate_groups={strict_summary['duplicate_groups']} duplicate_rows={strict_summary['duplicate_rows']}")
    append_debug_trace(trace_path, "exact_stage:relaxed:start")
    relaxed_summary = build_stage_results(
        conn,
        run_root=run_root,
        run_id=run_id,
        stage=RELAXED_STAGE,
        hash_column="exact_relaxed_hash",
        text_length_field="relaxed_text_chars",
        groups_path=stage_root / "relaxed_exact_groups.parquet",
        drops_path=stage_root / "relaxed_exact_drop_list.parquet",
    )
    append_debug_trace(trace_path, f"exact_stage:relaxed:done duplicate_groups={relaxed_summary['duplicate_groups']} duplicate_rows={relaxed_summary['duplicate_rows']}")
    append_debug_trace(trace_path, "exact_stage:snapshot_manifest:start")
    manifest_rows = write_snapshot_manifest(conn, run_id=run_id, path=run_root / "snapshot_manifest.parquet")
    append_debug_trace(trace_path, f"exact_stage:snapshot_manifest:done rows={manifest_rows}")
    append_debug_trace(trace_path, "exact_stage:docs_exact:start")
    docs_exact_rows = write_docs_exact_export(run_root=run_root, path=stage_root / "docs_exact.parquet")
    append_debug_trace(trace_path, f"exact_stage:docs_exact:done rows={docs_exact_rows}")
    exact_survivor_rows = 0
    exact_survivor_manifest = None
    exact_survivor_shards_root = None
    if emit_survivor_export:
        exact_survivor_rows, exact_survivor_manifest = _run_exact_survivor_export_stage(
            conn,
            run_id=run_id,
            run_root=run_root,
            files=files,
            max_workers=max_workers,
        )
        exact_survivor_shards_root = stage_root / "shards" / "exact_survivors"
    final_kept_rows = 0
    for batch_rows in iter_parquet_batches(stage_root / "docs_exact.parquet", columns=["kept_after_exact"]):
        final_kept_rows += sum(1 for row in batch_rows if int(row["kept_after_exact"] or 0) == 1)
    duration_seconds = round(time.perf_counter() - start, 3)
    total_docs = conn.execute("SELECT COUNT(*) FROM run_docs WHERE run_id = ?", (run_id,)).fetchone()[0]
    reused_exact_rows = conn.execute(
        "SELECT COALESCE(SUM(reused_exact), 0) FROM run_docs WHERE run_id = ?",
        (run_id,),
    ).fetchone()[0]
    summary = {
        "run_id": run_id,
        "input_root": str(input_root),
        "state_root": str(state_root),
        "run_root": str(run_root),
        "run_docs_inventory_path": str(run_docs_inventory_path(run_root)),
        "run_docs_inventory_rows": int(inventory_rows),
        "snapshot_manifest_path": str(run_root / "snapshot_manifest.parquet"),
        "docs_exact_path": str(stage_root / "docs_exact.parquet"),
        "exact_survivor_manifest_path": None if exact_survivor_manifest is None else str(exact_survivor_manifest),
        "exact_survivor_shards_root": None if exact_survivor_shards_root is None else str(exact_survivor_shards_root),
        "strict_exact_groups_path": str(stage_root / "strict_exact_groups.parquet"),
        "strict_exact_drop_list_path": str(stage_root / "strict_exact_drop_list.parquet"),
        "relaxed_exact_groups_path": str(stage_root / "relaxed_exact_groups.parquet"),
        "relaxed_exact_drop_list_path": str(stage_root / "relaxed_exact_drop_list.parquet"),
        "exact_strict_version": EXACT_STRICT_VERSION,
        "exact_relaxed_version": exact_relaxed_version(greek_diacritic_policy=greek_diacritic_policy),
        "greek_diacritic_policy": greek_diacritic_policy,
        "total_rows": int(total_docs),
        "snapshot_manifest_rows": int(manifest_rows),
        "docs_exact_rows": int(docs_exact_rows),
        "exact_survivor_rows": int(exact_survivor_rows),
        "reused_exact_rows": int(reused_exact_rows or 0),
        "computed_exact_rows": int(total_docs - (reused_exact_rows or 0)),
        "kept_after_exact_rows": int(final_kept_rows),
        "strict": strict_summary,
        "relaxed": relaxed_summary,
        "duration_seconds": duration_seconds,
    }
    write_json_atomic(stage_root / "summary.json", summary)
    upsert_stage_progress(
        conn,
        run_id=run_id,
        stage="stage_01_exact",
        status="completed",
        total_chunks=total_chunks,
        completed_chunks=total_chunks,
        progress_path=progress_path,
        payload={
            "run_id": run_id,
            "stage": "stage_01_exact",
            "status": "completed",
            "total_chunks": int(total_chunks),
            "completed_chunks": int(total_chunks),
            "summary_path": str(stage_root / "summary.json"),
        },
    )
    return summary


def near_signature_chunk_key(file_path: Path, row_group_index: int) -> str:
    return f"{file_path}::{row_group_index}"


def near_signature_shard_path(stage_root: Path, file_path: Path, row_group_index: int) -> Path:
    shard_root = stage_root / "shards" / "signatures"
    ensure_dir(shard_root)
    return shard_root / f"{file_path.stem}__rg{row_group_index:05d}.parquet"


def near_band_shard_path(stage_root: Path, *, band_index: int, file_path: Path, row_group_index: int) -> Path:
    band_root = stage_root / "shards" / "lsh_buckets" / f"band_{band_index:02d}"
    ensure_dir(band_root)
    return band_root / f"{file_path.stem}__rg{row_group_index:05d}.parquet"


def candidate_band_shard_root(stage_root: Path, *, band_index: int) -> Path:
    shard_root = stage_root / "shards" / "candidate_pairs" / f"band_{band_index:02d}"
    ensure_dir(shard_root)
    return shard_root


def candidate_bucket_partition_root(stage_root: Path, *, band_index: int) -> Path:
    partition_root = stage_root / "shards" / "bucket_members" / f"band_{band_index:02d}"
    ensure_dir(partition_root)
    return partition_root


def candidate_bucket_partition_path(stage_root: Path, *, band_index: int, bucket_prefix: str) -> Path:
    return candidate_bucket_partition_root(stage_root, band_index=band_index) / f"prefix_{bucket_prefix}.parquet"


def candidate_bucket_partition_complete_path(stage_root: Path, *, band_index: int) -> Path:
    return candidate_bucket_partition_root(stage_root, band_index=band_index) / "_COMPLETE"


def candidate_bucket_shard_path(stage_root: Path, *, band_index: int, bucket_hash: str) -> Path:
    return candidate_band_shard_root(stage_root, band_index=band_index) / f"{hashed_filename(bucket_hash)}.parquet"


def candidate_chunk_shard_path(stage_root: Path, *, band_index: int, chunk_suffix: str | None = None) -> Path:
    shard_root = candidate_band_shard_root(stage_root, band_index=band_index)
    if chunk_suffix:
        return shard_root / f"{chunk_suffix}.parquet"
    return shard_root / "band.parquet"


def bucket_summary_shard_path(stage_root: Path, *, band_index: int, chunk_suffix: str | None = None) -> Path:
    shard_root = stage_root / "shards" / "bucket_summaries"
    ensure_dir(shard_root)
    suffix = "" if not chunk_suffix else f"__{chunk_suffix}"
    return shard_root / f"band_{band_index:02d}{suffix}.parquet"


def touched_doc_shard_path(stage_root: Path, *, band_index: int, chunk_suffix: str | None = None) -> Path:
    shard_root = stage_root / "shards" / "touched_doc_keys"
    ensure_dir(shard_root)
    suffix = "" if not chunk_suffix else f"__{chunk_suffix}"
    return shard_root / f"band_{band_index:02d}{suffix}.parquet"


def compute_near_signature_chunk(
    *,
    survivor_shard_path: Path,
    source_file_path: Path,
    row_group_index: int,
    stage_root: Path,
    num_perm: int,
    bands: int,
    rows_per_band: int,
    shingle_mode: str,
    shingle_size: int,
    greek_diacritic_policy: str,
) -> dict[str, Any]:
    survivor_file = pq.ParquetFile(survivor_shard_path)
    table = survivor_file.read(use_threads=False)
    payload = table.to_pydict()
    signature_rows: list[dict[str, Any]] = []
    bucket_rows_by_band: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for idx in range(table.num_rows):
        near_text = normalize_near_text(
            text_value(payload["text"][idx]),
            greek_diacritic_policy=greek_diacritic_policy,
        )
        shingle_hashes, token_count, char_count = shingle_hashes_from_text(
            near_text=near_text,
            shingle_mode=shingle_mode,
            shingle_size=shingle_size,
        )
        if not shingle_hashes:
            continue
        signature = minhash_signature(shingle_hashes, num_perm=num_perm)
        signature_rows.append(
            {
                "doc_key": str(payload["doc_key"][idx]),
                "source_dataset": str(payload["source_dataset"][idx]),
                "source_doc_id": str(payload["source_doc_id"][idx]),
                "file_path": str(payload["file_path"][idx]),
                "row_group_index": int(payload["row_group_index"][idx]),
                "row_index_in_file": int(payload["row_index_in_file"][idx]),
                "token_count": int(token_count),
                "char_count": int(char_count),
                "near_text_chars": int(len(near_text)),
                "shingle_count": int(len(shingle_hashes)),
                "shingle_mode": shingle_mode,
                "shingle_size": int(shingle_size),
                "signature": [int(value) for value in signature.tolist()],
            }
        )
        for band_index in range(bands):
            bucket_rows_by_band[band_index].append(
                {
                    "doc_key": str(payload["doc_key"][idx]),
                    "band_index": int(band_index),
                    "bucket_hash": band_bucket_hash(signature, band_index=band_index, rows_per_band=rows_per_band),
                    "token_count": int(token_count),
                    "char_count": int(char_count),
                    "shingle_mode": shingle_mode,
                }
            )
    signature_path = near_signature_shard_path(stage_root, source_file_path, row_group_index)
    write_group_parquet(signature_rows, signature_path, schema=SIGNATURE_SCHEMA)
    for band_index, bucket_rows in bucket_rows_by_band.items():
        bucket_path = near_band_shard_path(
            stage_root,
            band_index=band_index,
            file_path=source_file_path,
            row_group_index=row_group_index,
        )
        write_group_parquet(bucket_rows, bucket_path, schema=LSH_BUCKET_SCHEMA)
    return {
        "chunk_key": near_signature_chunk_key(source_file_path, row_group_index),
        "artifact_path": str(signature_path),
        "row_count": int(len(signature_rows)),
    }


def copy_near_signature_chunk_artifacts(
    *,
    previous_stage_root: Path,
    current_stage_root: Path,
    source_file_path: Path,
    row_group_index: int,
    bands: int,
) -> int | None:
    previous_signature_path = near_signature_shard_path(previous_stage_root, source_file_path, row_group_index)
    current_signature_path = near_signature_shard_path(current_stage_root, source_file_path, row_group_index)
    if not previous_signature_path.exists():
        return None
    atomic_copy(previous_signature_path, current_signature_path)
    for band_index in range(bands):
        previous_bucket_path = near_band_shard_path(
            previous_stage_root,
            band_index=band_index,
            file_path=source_file_path,
            row_group_index=row_group_index,
        )
        current_bucket_path = near_band_shard_path(
            current_stage_root,
            band_index=band_index,
            file_path=source_file_path,
            row_group_index=row_group_index,
        )
        if previous_bucket_path.exists():
            atomic_copy(previous_bucket_path, current_bucket_path)
    return int(pq.ParquetFile(previous_signature_path).metadata.num_rows)


def _run_near_signature_stage(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    run_root: Path,
    survivor_manifest_path: Path,
    state_root: Path,
    config: dict[str, Any],
    greek_diacritic_policy: str,
    max_workers: int,
    num_perm: int,
    bands: int,
    rows_per_band: int,
    shingle_mode: str,
    shingle_size: int,
) -> dict[str, Any]:
    stage_root = run_root / "stage_02_near"
    ensure_dir(stage_root)
    survivor_manifest_rows = sorted(
        iter_parquet_rows(survivor_manifest_path),
        key=lambda row: (str(row["source_file_path"]), int(row["row_group_index"])),
    )
    chunk_keys = [
        near_signature_chunk_key(Path(str(row["source_file_path"])), int(row["row_group_index"]))
        for row in survivor_manifest_rows
    ]
    register_stage_chunks(conn, run_id=run_id, stage=NEAR_SIGNATURE_STAGE, chunk_keys=chunk_keys)
    total_chunks, completed_chunks = stage_chunk_counts(conn, run_id=run_id, stage=NEAR_SIGNATURE_STAGE)
    progress_path = progress_file_path(run_root, NEAR_SIGNATURE_STAGE)
    existing_summary = read_json_if_exists(progress_path)
    near_signature_required_paths = [
        stage_root / "signatures.parquet",
        signature_matrix_path(stage_root / "signatures.parquet"),
        signature_doc_keys_path(stage_root / "signatures.parquet"),
        signature_metadata_path(stage_root / "signatures.parquet"),
        stage_root / "lsh_buckets.parquet",
    ]
    if existing_summary is not None and completed_chunks == total_chunks and total_chunks > 0 and all_paths_exist(near_signature_required_paths):
        upsert_stage_progress(
            conn,
            run_id=run_id,
            stage=NEAR_SIGNATURE_STAGE,
            status="completed",
            total_chunks=total_chunks,
            completed_chunks=total_chunks,
            progress_path=progress_path,
            payload=existing_summary,
        )
        return existing_summary
    upsert_stage_progress(
        conn,
        run_id=run_id,
        stage=NEAR_SIGNATURE_STAGE,
        status="running",
        total_chunks=total_chunks,
        completed_chunks=completed_chunks,
        progress_path=progress_path,
        payload={
            "run_id": run_id,
            "stage": NEAR_SIGNATURE_STAGE,
            "status": "running",
            "total_chunks": total_chunks,
            "completed_chunks": completed_chunks,
        },
    )
    exact_survivor_rows = int(sum(int(row["row_count"]) for row in survivor_manifest_rows))
    previous_run_root = latest_compatible_run_root(
        state_root=state_root,
        current_run_root=run_root,
        current_config=config,
        keys=[
            "greek_diacritic_policy",
            "near_norm_version",
            "tokenization_version",
            "minhash_version",
            "lsh_version",
            "near_incremental_version",
            "num_perm",
            "bands",
            "rows_per_band",
            "shingle_mode",
            "shingle_size",
        ],
        required_relative_paths=[
            Path("stage_01_exact") / "exact_survivor_manifest.parquet",
            Path("stage_02_near") / "signatures.parquet",
        ],
    )
    previous_manifest = (
        load_exact_survivor_manifest(previous_run_root / "stage_01_exact" / "exact_survivor_manifest.parquet")
        if previous_run_root is not None
        else {}
    )
    reused_chunks = 0
    pending_manifest_rows: list[dict[str, Any]] = []
    for row in survivor_manifest_rows:
        source_file_path = Path(str(row["source_file_path"]))
        row_group_index = int(row["row_group_index"])
        chunk_key = near_signature_chunk_key(source_file_path, row_group_index)
        if stage_chunk_status(conn, run_id=run_id, stage=NEAR_SIGNATURE_STAGE, chunk_key=chunk_key) == "completed":
            continue
        previous_row = previous_manifest.get(chunk_key)
        reused_row_count: int | None = None
        if (
            previous_run_root is not None
            and previous_row is not None
            and str(previous_row["survivor_digest"]) == str(row["survivor_digest"])
        ):
            reused_row_count = copy_near_signature_chunk_artifacts(
                previous_stage_root=previous_run_root / "stage_02_near",
                current_stage_root=stage_root,
                source_file_path=source_file_path,
                row_group_index=row_group_index,
                bands=bands,
            )
        if reused_row_count is not None:
            reused_chunks += 1
            mark_stage_chunk_complete(
                conn,
                run_id=run_id,
                stage=NEAR_SIGNATURE_STAGE,
                chunk_key=chunk_key,
                artifact_path=near_signature_shard_path(stage_root, source_file_path, row_group_index),
                row_count=int(reused_row_count),
            )
        else:
            pending_manifest_rows.append(row)
    worker_count = effective_worker_count(max_workers, len(pending_manifest_rows))
    if worker_count == 1:
        for row in pending_manifest_rows:
            source_file_path = Path(str(row["source_file_path"]))
            row_group_index = int(row["row_group_index"])
            result = compute_near_signature_chunk(
                survivor_shard_path=Path(str(row["shard_path"])),
                source_file_path=source_file_path,
                row_group_index=row_group_index,
                stage_root=stage_root,
                num_perm=num_perm,
                bands=bands,
                rows_per_band=rows_per_band,
                shingle_mode=shingle_mode,
                shingle_size=shingle_size,
                greek_diacritic_policy=greek_diacritic_policy,
            )
            mark_stage_chunk_complete(
                conn,
                run_id=run_id,
                stage=NEAR_SIGNATURE_STAGE,
                chunk_key=str(result["chunk_key"]),
                artifact_path=Path(str(result["artifact_path"])),
                row_count=int(result["row_count"]),
            )
            total_chunks, completed_chunks = stage_chunk_counts(conn, run_id=run_id, stage=NEAR_SIGNATURE_STAGE)
            upsert_stage_progress(
                conn,
                run_id=run_id,
                stage=NEAR_SIGNATURE_STAGE,
                status="running",
                total_chunks=total_chunks,
                completed_chunks=completed_chunks,
                progress_path=progress_path,
                payload={
                    "run_id": run_id,
                    "stage": NEAR_SIGNATURE_STAGE,
                    "status": "running",
                    "total_chunks": total_chunks,
                    "completed_chunks": completed_chunks,
                },
            )
    elif pending_manifest_rows:
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=PROCESS_POOL_CONTEXT) as executor:
            pending_iter = iter(pending_manifest_rows)
            in_flight = {
                executor.submit(
                    compute_near_signature_chunk,
                    survivor_shard_path=Path(str(row["shard_path"])),
                    source_file_path=Path(str(row["source_file_path"])),
                    row_group_index=int(row["row_group_index"]),
                    stage_root=stage_root,
                    num_perm=num_perm,
                    bands=bands,
                    rows_per_band=rows_per_band,
                    shingle_mode=shingle_mode,
                    shingle_size=shingle_size,
                    greek_diacritic_policy=greek_diacritic_policy,
                ): row
                for row in [next(pending_iter) for _ in range(min(worker_count, len(pending_manifest_rows)))]
            }
            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    in_flight.pop(future)
                    result = future.result()
                    mark_stage_chunk_complete(
                        conn,
                        run_id=run_id,
                        stage=NEAR_SIGNATURE_STAGE,
                        chunk_key=str(result["chunk_key"]),
                        artifact_path=Path(str(result["artifact_path"])),
                        row_count=int(result["row_count"]),
                    )
                    total_chunks, completed_chunks = stage_chunk_counts(conn, run_id=run_id, stage=NEAR_SIGNATURE_STAGE)
                    upsert_stage_progress(
                        conn,
                        run_id=run_id,
                        stage=NEAR_SIGNATURE_STAGE,
                        status="running",
                        total_chunks=total_chunks,
                        completed_chunks=completed_chunks,
                        progress_path=progress_path,
                        payload={
                            "run_id": run_id,
                            "stage": NEAR_SIGNATURE_STAGE,
                            "status": "running",
                            "total_chunks": total_chunks,
                            "completed_chunks": completed_chunks,
                        },
                    )
                    try:
                        next_row = next(pending_iter)
                    except StopIteration:
                        continue
                    in_flight[
                        executor.submit(
                            compute_near_signature_chunk,
                            survivor_shard_path=Path(str(next_row["shard_path"])),
                            source_file_path=Path(str(next_row["source_file_path"])),
                            row_group_index=int(next_row["row_group_index"]),
                            stage_root=stage_root,
                            num_perm=num_perm,
                            bands=bands,
                            rows_per_band=rows_per_band,
                            shingle_mode=shingle_mode,
                            shingle_size=shingle_size,
                            greek_diacritic_policy=greek_diacritic_policy,
                        )
                    ] = next_row
    signature_paths = sorted((stage_root / "shards" / "signatures").glob("*.parquet"))
    signatures_rows = combine_parquet_files(signature_paths, stage_root / "signatures.parquet", schema=SIGNATURE_SCHEMA)
    signature_matrix, signature_doc_keys, signature_metadata = build_signature_matrix_artifacts(
        stage_root / "signatures.parquet",
        num_perm=num_perm,
    )
    bucket_paths = sorted((stage_root / "shards" / "lsh_buckets").glob("band_*/*.parquet"))
    lsh_bucket_rows = combine_parquet_files(bucket_paths, stage_root / "lsh_buckets.parquet", schema=LSH_BUCKET_SCHEMA)
    summary = {
        "run_id": run_id,
        "stage": NEAR_SIGNATURE_STAGE,
        "survivor_manifest_path": str(survivor_manifest_path),
        "signatures_path": str(stage_root / "signatures.parquet"),
        "signature_matrix_path": str(signature_matrix),
        "signature_doc_keys_path": str(signature_doc_keys),
        "signature_metadata_path": str(signature_metadata),
        "lsh_buckets_path": str(stage_root / "lsh_buckets.parquet"),
        "shingle_mode": shingle_mode,
        "shingle_size": int(shingle_size),
        "num_perm": int(num_perm),
        "bands": int(bands),
        "rows_per_band": int(rows_per_band),
        "exact_survivor_rows": int(exact_survivor_rows),
        "stage2_input_rows": int(signatures_rows),
        "short_or_skipped_rows": int(exact_survivor_rows - signatures_rows),
        "reused_signature_chunks": int(reused_chunks),
        "computed_signature_chunks": int(len(pending_manifest_rows)),
        "lsh_bucket_rows": int(lsh_bucket_rows),
    }
    upsert_stage_progress(
        conn,
        run_id=run_id,
        stage=NEAR_SIGNATURE_STAGE,
        status="completed",
        total_chunks=total_chunks,
        completed_chunks=total_chunks,
        progress_path=progress_path,
        payload=summary,
    )
    return summary


def split_bucket_members_by_length_bin(members: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in members:
        grouped[int(row["token_count"]) // OVERSIZED_BUCKET_TOKEN_BIN].append(row)
    return [group for _, group in sorted(grouped.items())]


def split_bucket_members_by_secondary_band(
    members: list[dict[str, Any]],
    *,
    signature_map: SignatureLookup | dict[str, np.ndarray],
    band_index: int,
    bands: int,
    rows_per_band: int,
    band_offset: int,
) -> list[list[dict[str, Any]]]:
    secondary_band_index = (band_index + band_offset) % bands
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in members:
        doc_key = str(row["doc_key"])
        grouped[
            band_bucket_hash(
                signature_map[doc_key],
                band_index=secondary_band_index,
                rows_per_band=rows_per_band,
            )
        ].append(row)
    return [group for _, group in sorted(grouped.items())]


def subdivide_oversized_bucket_members(
    members: list[dict[str, Any]],
    *,
    signature_map: SignatureLookup | dict[str, np.ndarray],
    band_index: int,
    bands: int,
    rows_per_band: int,
    max_bucket_size: int,
) -> tuple[list[list[dict[str, Any]]], dict[str, int]]:
    pending = [members]
    accepted: list[list[dict[str, Any]]] = []
    subdivided_bucket_count = 0
    fallback_chunked_bucket_count = 0
    fallback_chunked_member_rows = 0
    while pending:
        current = pending.pop()
        if len(current) <= max_bucket_size:
            accepted.append(current)
            continue
        length_groups = split_bucket_members_by_length_bin(current)
        if len(length_groups) > 1 and max(len(group) for group in length_groups) < len(current):
            subdivided_bucket_count += 1
            pending.extend(length_groups)
            continue
        split_groups: list[list[dict[str, Any]]] | None = None
        for band_offset in range(1, min(4, bands)):
            candidate_groups = split_bucket_members_by_secondary_band(
                current,
                signature_map=signature_map,
                band_index=band_index,
                bands=bands,
                rows_per_band=rows_per_band,
                band_offset=band_offset,
            )
            if len(candidate_groups) > 1 and max(len(group) for group in candidate_groups) < len(current):
                split_groups = candidate_groups
                break
        if split_groups is not None:
            subdivided_bucket_count += 1
            pending.extend(split_groups)
            continue
        ordered = sorted(current, key=lambda row: str(row["doc_key"]))
        fallback_chunked_bucket_count += 1
        fallback_chunked_member_rows += len(ordered)
        window_step = max(1, max_bucket_size // 2)
        start_offsets = list(range(0, max(1, len(ordered) - max_bucket_size + 1), window_step))
        last_offset = max(0, len(ordered) - max_bucket_size)
        if not start_offsets or start_offsets[-1] != last_offset:
            start_offsets.append(last_offset)
        accepted.extend(ordered[start : start + max_bucket_size] for start in start_offsets)
    return accepted, {
        "subdivided_bucket_count": int(subdivided_bucket_count),
        "fallback_chunked_bucket_count": int(fallback_chunked_bucket_count),
        "fallback_chunked_member_rows": int(fallback_chunked_member_rows),
    }


def bucket_member_digest(
    members: list[dict[str, Any]],
    *,
    signature_map: SignatureLookup | dict[str, np.ndarray],
    signature_meta: SignatureMetadataLookup,
) -> str:
    hasher = blake3()
    for row in sorted(members, key=lambda item: str(item["doc_key"])):
        doc_key = str(row["doc_key"])
        hasher.update(doc_key.encode("utf-8"))
        hasher.update(b"\0")
        signature_values = np.asarray(signature_map[doc_key], dtype=np.uint64)
        hasher.update(signature_values.tobytes())
        hasher.update(b"\0")
        hasher.update(str(signature_meta.token_count(doc_key)).encode("ascii"))
        hasher.update(b"\0")
        hasher.update(str(signature_meta.char_count(doc_key)).encode("ascii"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def load_bucket_summary(path: Path) -> dict[int, dict[str, dict[str, Any]]]:
    paths: list[Path]
    if path.exists():
        if path.is_dir():
            paths = sorted(path.glob("*.parquet"))
        else:
            paths = [path]
    elif path.name == "bucket_summary.parquet":
        shard_root = path.parent / "shards" / "bucket_summaries"
        paths = sorted(shard_root.glob("*.parquet")) if shard_root.exists() else []
    else:
        paths = []
    if not paths:
        return {}
    summary: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for source_path in paths:
        for row in iter_parquet_rows(source_path):
            summary[int(row["band_index"])][str(row["bucket_hash"])] = row
    return summary


def near_candidate_bucket_prefixes() -> list[str]:
    width = max(1, NEAR_CANDIDATE_BUCKET_PREFIX_LEN)
    return [format(index, f"0{width}x") for index in range(16**width)]


def near_candidate_chunk_key(*, band_index: int, bucket_prefix: str) -> str:
    return f"band:{band_index:02d}:prefix:{bucket_prefix}"


def prepare_candidate_bucket_partitions_for_band(stage_root: Path, *, band_index: int) -> list[str]:
    partition_root = candidate_bucket_partition_root(stage_root, band_index=band_index)
    complete_path = candidate_bucket_partition_complete_path(stage_root, band_index=band_index)
    prefixes = near_candidate_bucket_prefixes()
    if complete_path.exists() and all(
        candidate_bucket_partition_path(stage_root, band_index=band_index, bucket_prefix=prefix).exists()
        for prefix in prefixes
    ):
        return prefixes
    if partition_root.exists():
        shutil.rmtree(partition_root)
    ensure_dir(partition_root)
    writers: dict[str, pq.ParquetWriter | None] = {prefix: None for prefix in prefixes}
    temp_paths = {
        prefix: temp_output_path(candidate_bucket_partition_path(stage_root, band_index=band_index, bucket_prefix=prefix))
        for prefix in prefixes
    }
    buffers: dict[str, list[dict[str, Any]]] = {prefix: [] for prefix in prefixes}
    try:
        band_dir = stage_root / "shards" / "lsh_buckets" / f"band_{band_index:02d}"
        for path in sorted(band_dir.glob("*.parquet")):
            for batch_rows in iter_parquet_batches(path):
                for row in batch_rows:
                    bucket_hash = str(row["bucket_hash"])
                    prefix = bucket_hash[:NEAR_CANDIDATE_BUCKET_PREFIX_LEN]
                    buffers[prefix].append(
                        {
                            "doc_key": str(row["doc_key"]),
                            "band_index": int(row["band_index"]),
                            "bucket_hash": bucket_hash,
                            "token_count": int(row["token_count"]),
                            "char_count": int(row["char_count"]),
                            "shingle_mode": str(row["shingle_mode"]),
                        }
                    )
                for prefix, rows in buffers.items():
                    if len(rows) >= 4096:
                        writers[prefix], _ = append_rows_to_parquet_writer(
                            writers[prefix],
                            rows=rows,
                            temp_path=temp_paths[prefix],
                            schema=LSH_BUCKET_SCHEMA,
                        )
                        buffers[prefix] = []
        for prefix, rows in buffers.items():
            if rows:
                writers[prefix], _ = append_rows_to_parquet_writer(
                    writers[prefix],
                    rows=rows,
                    temp_path=temp_paths[prefix],
                    schema=LSH_BUCKET_SCHEMA,
                )
        for prefix in prefixes:
            finalize_parquet_writer(
                writers[prefix],
                temp_path=temp_paths[prefix],
                destination=candidate_bucket_partition_path(stage_root, band_index=band_index, bucket_prefix=prefix),
                schema=LSH_BUCKET_SCHEMA,
            )
        complete_path.write_text(json.dumps({"band_index": int(band_index), "prefixes": prefixes}, indent=2))
    except Exception:
        shutil.rmtree(partition_root, ignore_errors=True)
        raise
    return prefixes


def iter_band_bucket_members(stage_root: Path, *, band_index: int) -> Any:
    band_dir = stage_root / "shards" / "lsh_buckets" / f"band_{band_index:02d}"
    band_paths = sorted(band_dir.glob("*.parquet"))
    if not band_paths:
        return
    current_bucket_hash: str | None = None
    current_members: list[dict[str, Any]] = []
    for row in iter_duckdb_query_rows(
        """
        SELECT doc_key, bucket_hash, token_count, char_count, shingle_mode
        FROM read_parquet(?)
        ORDER BY bucket_hash, doc_key
        """,
        [str(band_dir / "*.parquet")],
        batch_size=4096,
        threads=1,
    ):
        bucket_hash = str(row["bucket_hash"])
        if current_bucket_hash is not None and bucket_hash != current_bucket_hash:
            yield current_bucket_hash, current_members
            current_members = []
        current_bucket_hash = bucket_hash
        current_members.append(
            {
                "doc_key": str(row["doc_key"]),
                "bucket_hash": bucket_hash,
                "token_count": int(row["token_count"]),
                "char_count": int(row["char_count"]),
                "shingle_mode": str(row["shingle_mode"]),
            }
        )
    if current_bucket_hash is not None:
        yield current_bucket_hash, current_members


def iter_partition_bucket_members(stage_root: Path, *, band_index: int, bucket_prefix: str) -> Any:
    partition_path = candidate_bucket_partition_path(stage_root, band_index=band_index, bucket_prefix=bucket_prefix)
    if not partition_path.exists():
        return
    current_bucket_hash: str | None = None
    current_members: list[dict[str, Any]] = []
    for row in iter_duckdb_query_rows(
        """
        SELECT doc_key, bucket_hash, token_count, char_count, shingle_mode
        FROM read_parquet(?)
        ORDER BY bucket_hash, doc_key
        """,
        [str(partition_path)],
        batch_size=4096,
        threads=1,
    ):
        bucket_hash = str(row["bucket_hash"])
        if current_bucket_hash is not None and bucket_hash != current_bucket_hash:
            yield current_bucket_hash, current_members
            current_members = []
        current_bucket_hash = bucket_hash
        current_members.append(
            {
                "doc_key": str(row["doc_key"]),
                "bucket_hash": bucket_hash,
                "token_count": int(row["token_count"]),
                "char_count": int(row["char_count"]),
                "shingle_mode": str(row["shingle_mode"]),
            }
        )
    if current_bucket_hash is not None:
        yield current_bucket_hash, current_members


def append_candidate_rows_for_bucket(
    *,
    candidate_groups: list[list[dict[str, Any]]],
    writer: pq.ParquetWriter | None,
    temp_path: Path,
    row_buffer: list[dict[str, Any]],
    minhash_threshold: float,
    signature_map: SignatureLookup | dict[str, np.ndarray],
    signature_meta: SignatureMetadataLookup,
) -> tuple[pq.ParquetWriter | None, list[dict[str, Any]], int]:
    rows_written = 0
    seen_pairs: set[tuple[str, str]] = set()
    for candidate_group in candidate_groups:
        if len(candidate_group) < 2:
            continue
        ordered = sorted(candidate_group, key=lambda row: str(row["doc_key"]))
        for left_index in range(len(ordered)):
            for right_index in range(left_index + 1, len(ordered)):
                left_key = str(ordered[left_index]["doc_key"])
                right_key = str(ordered[right_index]["doc_key"])
                pair_key = (left_key, right_key)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                left_token_count = signature_meta.token_count(left_key)
                right_token_count = signature_meta.token_count(right_key)
                shorter = min(left_token_count, right_token_count)
                longer = max(left_token_count, right_token_count)
                length_ratio = 0.0 if longer == 0 else float(shorter / longer)
                estimated_jaccard = signature_jaccard(signature_map[left_key], signature_map[right_key])
                if estimated_jaccard < minhash_threshold:
                    continue
                row_buffer.append(
                    {
                        "doc_key_left": left_key,
                        "doc_key_right": right_key,
                        "estimated_jaccard": float(estimated_jaccard),
                        "shingle_mode": signature_meta.shingle_mode(left_key),
                        "token_count_left": int(left_token_count),
                        "token_count_right": int(right_token_count),
                        "length_ratio": float(length_ratio),
                        "likely_containment_flag": bool(length_ratio < 0.85),
                        "accepted_reason": "lsh_threshold",
                        "bucket_match_bands": 1,
                    }
                )
                if len(row_buffer) >= NEAR_CANDIDATE_FLUSH_ROWS:
                    writer, appended = append_rows_to_parquet_writer(
                        writer,
                        rows=row_buffer,
                        temp_path=temp_path,
                        schema=CANDIDATE_PAIR_SCHEMA,
                        row_group_size=NEAR_CANDIDATE_FLUSH_ROWS,
                    )
                    rows_written += appended
                    row_buffer = []
    return writer, row_buffer, rows_written


def build_candidate_band_chunk(
    *,
    stage_root: Path,
    band_index: int,
    bucket_prefix: str | None = None,
    bands: int,
    rows_per_band: int,
    minhash_threshold: float,
    max_bucket_size: int,
    signature_map: SignatureLookup | dict[str, np.ndarray] | None = None,
    signature_meta: SignatureMetadataLookup | None = None,
    previous_band_summary: dict[str, dict[str, Any]] | None = None,
    previous_stage_root: Path | None = None,
) -> dict[str, Any]:
    if signature_map is None or signature_meta is None:
        if _CANDIDATE_WORKER_SIGNATURE_MAP is None or _CANDIDATE_WORKER_SIGNATURE_META is None:
            raise ValueError("candidate worker signature index is not initialized")
        signature_map = _CANDIDATE_WORKER_SIGNATURE_MAP
        signature_meta = _CANDIDATE_WORKER_SIGNATURE_META
    previous_band_summary = previous_band_summary or {}
    chunk_suffix = None if bucket_prefix is None else f"prefix_{bucket_prefix}"
    candidate_path = candidate_chunk_shard_path(stage_root, band_index=band_index, chunk_suffix=chunk_suffix)
    bucket_summary_path = bucket_summary_shard_path(stage_root, band_index=band_index, chunk_suffix=chunk_suffix)
    touched_doc_path = touched_doc_shard_path(stage_root, band_index=band_index, chunk_suffix=chunk_suffix)
    candidate_writer: pq.ParquetWriter | None = None
    bucket_summary_writer: pq.ParquetWriter | None = None
    touched_doc_writer: pq.ParquetWriter | None = None
    candidate_temp_path = temp_output_path(candidate_path)
    bucket_summary_temp_path = temp_output_path(bucket_summary_path)
    touched_doc_temp_path = temp_output_path(touched_doc_path)
    candidate_buffer: list[dict[str, Any]] = []
    bucket_summary_buffer: list[dict[str, Any]] = []
    touched_doc_buffer: list[dict[str, Any]] = []
    total_candidate_rows = 0
    oversized_bucket_count = 0
    oversized_bucket_member_rows = 0
    subdivided_bucket_count = 0
    fallback_chunked_bucket_count = 0
    fallback_chunked_member_rows = 0
    reused_bucket_count = 0
    recomputed_bucket_count = 0

    def flush_candidate_buffer() -> None:
        nonlocal candidate_writer
        nonlocal candidate_buffer
        if not candidate_buffer:
            return
        candidate_writer, _ = append_rows_to_parquet_writer(
            candidate_writer,
            rows=candidate_buffer,
            temp_path=candidate_temp_path,
            schema=CANDIDATE_PAIR_SCHEMA,
            row_group_size=NEAR_CANDIDATE_FLUSH_ROWS,
        )
        candidate_buffer = []

    def flush_bucket_summary_buffer() -> None:
        nonlocal bucket_summary_writer
        nonlocal bucket_summary_buffer
        if not bucket_summary_buffer:
            return
        bucket_summary_writer, _ = append_rows_to_parquet_writer(
            bucket_summary_writer,
            rows=bucket_summary_buffer,
            temp_path=bucket_summary_temp_path,
            schema=BUCKET_SUMMARY_SCHEMA,
            row_group_size=NEAR_BUCKET_SUMMARY_FLUSH_ROWS,
        )
        bucket_summary_buffer = []

    def flush_touched_doc_buffer() -> None:
        nonlocal touched_doc_writer
        nonlocal touched_doc_buffer
        if not touched_doc_buffer:
            return
        touched_doc_writer, _ = append_rows_to_parquet_writer(
            touched_doc_writer,
            rows=touched_doc_buffer,
            temp_path=touched_doc_temp_path,
            schema=TOUCHED_DOC_SCHEMA,
            row_group_size=NEAR_TOUCHED_DOC_FLUSH_ROWS,
        )
        touched_doc_buffer = []

    try:
        member_iter = (
            iter_band_bucket_members(stage_root, band_index=band_index)
            if bucket_prefix is None
            else iter_partition_bucket_members(stage_root, band_index=band_index, bucket_prefix=bucket_prefix)
        )
        for bucket_hash, members in member_iter or []:
            bucket_summary_row: dict[str, Any]
            candidate_row_count = 0
            member_digest = bucket_member_digest(members, signature_map=signature_map, signature_meta=signature_meta)
            previous_row = previous_band_summary.get(bucket_hash)
            previous_candidate_path = None
            if previous_stage_root is not None:
                previous_candidate_path = candidate_bucket_shard_path(previous_stage_root, band_index=band_index, bucket_hash=bucket_hash)
            if len(members) < 2:
                bucket_summary_row = {
                    "band_index": int(band_index),
                    "bucket_hash": bucket_hash,
                    "member_count": int(len(members)),
                    "member_digest": member_digest,
                    "candidate_row_count": 0,
                }
                bucket_summary_buffer.append(bucket_summary_row)
                if len(bucket_summary_buffer) >= NEAR_BUCKET_SUMMARY_FLUSH_ROWS:
                    flush_bucket_summary_buffer()
                continue
            elif (
                previous_row is not None
                and str(previous_row["member_digest"]) == member_digest
            ):
                if previous_candidate_path is None or not previous_candidate_path.exists():
                    if int(previous_row.get("candidate_row_count", 0) or 0) == 0:
                        reused_bucket_count += 1
                        bucket_summary_row = {
                            "band_index": int(band_index),
                            "bucket_hash": bucket_hash,
                            "member_count": int(len(members)),
                            "member_digest": member_digest,
                            "candidate_row_count": 0,
                        }
                    else:
                        previous_candidate_path = None
                else:
                    reused_bucket_count += 1
                    candidate_row_count = 0
                    for batch_rows in iter_parquet_batches(previous_candidate_path):
                        candidate_buffer.extend(batch_rows)
                        candidate_row_count += len(batch_rows)
                        if len(candidate_buffer) >= NEAR_CANDIDATE_FLUSH_ROWS:
                            flush_candidate_buffer()
                    bucket_summary_row = {
                        "band_index": int(band_index),
                        "bucket_hash": bucket_hash,
                        "member_count": int(len(members)),
                        "member_digest": member_digest,
                        "candidate_row_count": candidate_row_count,
                    }
                if previous_candidate_path is None and int(previous_row.get("candidate_row_count", 0) or 0) != 0:
                    recomputed_bucket_count += 1
                else:
                    total_candidate_rows += int(bucket_summary_row["candidate_row_count"])
                    bucket_summary_buffer.append(bucket_summary_row)
                    if len(bucket_summary_buffer) >= NEAR_BUCKET_SUMMARY_FLUSH_ROWS:
                        flush_bucket_summary_buffer()
                    continue
            else:
                recomputed_bucket_count += 1
            touched_doc_buffer.extend({"doc_key": str(row["doc_key"])} for row in members)
            if len(touched_doc_buffer) >= NEAR_TOUCHED_DOC_FLUSH_ROWS:
                flush_touched_doc_buffer()
            candidate_groups = [members]
            if len(members) > max_bucket_size:
                oversized_bucket_count += 1
                oversized_bucket_member_rows += len(members)
                candidate_groups, subdivision_stats = subdivide_oversized_bucket_members(
                    members,
                    signature_map=signature_map,
                    band_index=band_index,
                    bands=bands,
                    rows_per_band=rows_per_band,
                    max_bucket_size=max_bucket_size,
                )
                subdivided_bucket_count += int(subdivision_stats["subdivided_bucket_count"])
                fallback_chunked_bucket_count += int(subdivision_stats["fallback_chunked_bucket_count"])
                fallback_chunked_member_rows += int(subdivision_stats["fallback_chunked_member_rows"])
            candidate_writer, candidate_buffer, candidate_row_count = append_candidate_rows_for_bucket(
                candidate_groups=candidate_groups,
                writer=candidate_writer,
                temp_path=candidate_temp_path,
                row_buffer=candidate_buffer,
                minhash_threshold=minhash_threshold,
                signature_map=signature_map,
                signature_meta=signature_meta,
            )
            bucket_summary_row = {
                "band_index": int(band_index),
                "bucket_hash": bucket_hash,
                "member_count": int(len(members)),
                "member_digest": member_digest,
                "candidate_row_count": int(candidate_row_count),
            }
            total_candidate_rows += int(candidate_row_count)
            bucket_summary_buffer.append(bucket_summary_row)
            if len(bucket_summary_buffer) >= NEAR_BUCKET_SUMMARY_FLUSH_ROWS:
                flush_bucket_summary_buffer()
        flush_candidate_buffer()
        flush_bucket_summary_buffer()
        flush_touched_doc_buffer()
        finalize_parquet_writer(
            candidate_writer,
            temp_path=candidate_temp_path,
            destination=candidate_path,
            schema=CANDIDATE_PAIR_SCHEMA,
        )
        finalize_parquet_writer(
            bucket_summary_writer,
            temp_path=bucket_summary_temp_path,
            destination=bucket_summary_path,
            schema=BUCKET_SUMMARY_SCHEMA,
        )
        finalize_parquet_writer(
            touched_doc_writer,
            temp_path=touched_doc_temp_path,
            destination=touched_doc_path,
            schema=TOUCHED_DOC_SCHEMA,
        )
    except Exception:
        for temp_path in (candidate_temp_path, bucket_summary_temp_path, touched_doc_temp_path):
            if temp_path.exists():
                temp_path.unlink()
        raise
    return {
        "chunk_key": (
            f"band:{band_index:02d}"
            if bucket_prefix is None
            else near_candidate_chunk_key(band_index=band_index, bucket_prefix=bucket_prefix)
        ),
        "artifact_path": str(bucket_summary_path),
        "row_count": int(total_candidate_rows),
        "oversized_bucket_count": int(oversized_bucket_count),
        "oversized_bucket_member_rows": int(oversized_bucket_member_rows),
        "subdivided_bucket_count": int(subdivided_bucket_count),
        "fallback_chunked_bucket_count": int(fallback_chunked_bucket_count),
        "fallback_chunked_member_rows": int(fallback_chunked_member_rows),
        "reused_bucket_count": int(reused_bucket_count),
        "recomputed_bucket_count": int(recomputed_bucket_count),
        "touched_doc_path": str(touched_doc_path),
    }


def _run_near_candidate_stage(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    run_root: Path,
    state_root: Path,
    config: dict[str, Any],
    minhash_threshold: float,
    bands: int,
    rows_per_band: int,
    max_workers: int,
    max_bucket_size: int,
) -> dict[str, Any]:
    stage_root = run_root / "stage_02_near"
    signatures_path = stage_root / "signatures.parquet"
    progress_path = progress_file_path(run_root, NEAR_CANDIDATE_STAGE)
    trace_path = progress_dir(run_root) / "near_candidate_trace.log"
    existing_chunk_keys = [
        str(row["chunk_key"])
        for row in conn.execute(
            "SELECT chunk_key FROM run_stage_chunks WHERE run_id = ? AND stage = ?",
            (run_id, NEAR_CANDIDATE_STAGE),
        ).fetchall()
    ]
    uses_legacy_band_only_chunks = bool(existing_chunk_keys) and all(":prefix:" not in key for key in existing_chunk_keys)
    if uses_legacy_band_only_chunks:
        archive_root = stage_root / "legacy_reset"
        append_debug_trace(
            trace_path,
            "near_candidates:legacy_chunk_migration:start mode=archive",
        )
        conn.execute("DELETE FROM run_stage_chunks WHERE run_id = ? AND stage = ?", (run_id, NEAR_CANDIDATE_STAGE))
        conn.execute("DELETE FROM stage_progress WHERE run_id = ? AND stage = ?", (run_id, NEAR_CANDIDATE_STAGE))
        for relative in [
            Path("candidate_pairs.parquet"),
            Path("candidate_pairs.sqlite"),
            Path("bucket_summary.parquet"),
            Path("touched_doc_keys.parquet"),
        ]:
            path = stage_root / relative
            archived = archive_existing_path(path, archive_root=archive_root)
            if archived is not None:
                append_debug_trace(
                    trace_path,
                    f"near_candidates:legacy_chunk_migration:archived path={archived.relative_to(stage_root)}",
                )
        for relative_dir in [
            stage_root / "shards" / "candidate_pairs",
            stage_root / "shards" / "bucket_summaries",
            stage_root / "shards" / "touched_doc_keys",
        ]:
            archived = archive_existing_path(relative_dir, archive_root=archive_root)
            if archived is not None:
                append_debug_trace(
                    trace_path,
                    f"near_candidates:legacy_chunk_migration:archived path={archived.relative_to(stage_root)}",
                )
        append_debug_trace(
            trace_path,
            "near_candidates:legacy_chunk_migration:done mode=archive",
        )
    bucket_prefixes = near_candidate_bucket_prefixes()
    chunk_specs = [
        {"band_index": band_index, "bucket_prefix": bucket_prefix}
        for band_index in range(bands)
        for bucket_prefix in bucket_prefixes
    ]
    chunk_keys = [
        near_candidate_chunk_key(band_index=int(spec["band_index"]), bucket_prefix=str(spec["bucket_prefix"]))
        for spec in chunk_specs
    ]
    register_stage_chunks(conn, run_id=run_id, stage=NEAR_CANDIDATE_STAGE, chunk_keys=chunk_keys)
    total_chunks, completed_chunks = stage_chunk_counts(conn, run_id=run_id, stage=NEAR_CANDIDATE_STAGE)
    existing_summary = read_json_if_exists(progress_path)
    near_candidate_required_paths = [
        stage_root / "candidate_pairs.parquet",
        stage_root / "touched_doc_keys.parquet",
    ]
    if existing_summary is not None and completed_chunks == total_chunks and total_chunks > 0 and all_paths_exist(near_candidate_required_paths):
        upsert_stage_progress(
            conn,
            run_id=run_id,
            stage=NEAR_CANDIDATE_STAGE,
            status="completed",
            total_chunks=total_chunks,
            completed_chunks=total_chunks,
            progress_path=progress_path,
            payload=existing_summary,
        )
        return existing_summary
    upsert_stage_progress(
        conn,
        run_id=run_id,
        stage=NEAR_CANDIDATE_STAGE,
        status="running",
        total_chunks=total_chunks,
        completed_chunks=completed_chunks,
        progress_path=progress_path,
        payload={
            "run_id": run_id,
            "stage": NEAR_CANDIDATE_STAGE,
            "status": "running",
            "total_chunks": total_chunks,
            "completed_chunks": completed_chunks,
        },
    )
    pending_chunks = [
        spec
        for spec in chunk_specs
        if stage_chunk_status(
            conn,
            run_id=run_id,
            stage=NEAR_CANDIDATE_STAGE,
            chunk_key=near_candidate_chunk_key(
                band_index=int(spec["band_index"]),
                bucket_prefix=str(spec["bucket_prefix"]),
            ),
        )
        != "completed"
    ]
    pending_chunks_by_band: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for chunk_spec in pending_chunks:
        pending_chunks_by_band[int(chunk_spec["band_index"])].append(chunk_spec)
    oversized_bucket_count = 0
    oversized_bucket_member_rows = 0
    subdivided_bucket_count = 0
    fallback_chunked_bucket_count = 0
    fallback_chunked_member_rows = 0
    reused_bucket_count = 0
    recomputed_bucket_count = 0
    previous_run_root = latest_compatible_run_root(
        state_root=state_root,
        current_run_root=run_root,
        current_config=config,
        keys=[
            "greek_diacritic_policy",
            "near_norm_version",
            "tokenization_version",
            "minhash_version",
            "lsh_version",
            "near_incremental_version",
            "oversized_bucket_strategy_version",
            "minhash_threshold",
            "num_perm",
            "bands",
            "rows_per_band",
            "shingle_mode",
            "shingle_size",
            "max_bucket_size",
        ],
        required_relative_paths=[
            Path("stage_02_near") / "candidate_pairs.parquet",
            Path("stage_02_near") / "shards" / "bucket_summaries",
        ],
    )
    previous_bucket_summary = (
        load_bucket_summary(previous_run_root / "stage_02_near" / "bucket_summary.parquet")
        if previous_run_root is not None
        else {}
    )
    worker_count = effective_worker_count(min(max_workers, near_candidate_worker_cap()), len(pending_chunks))
    pending_band_indexes = sorted(pending_chunks_by_band)
    append_debug_trace(
        trace_path,
        f"near_candidates:start pending_chunks={len(pending_chunks)} worker_count={worker_count} max_bucket_size={max_bucket_size}",
    )
    if worker_count == 1:
        signature_map, signature_meta = load_signature_index(signatures_path)
        for band_index in pending_band_indexes:
            prefixes = prepare_candidate_bucket_partitions_for_band(stage_root, band_index=band_index)
            append_debug_trace(
                trace_path,
                f"near_candidates:partitions_ready band={band_index:02d} prefixes={len(prefixes)}",
            )
            for chunk_spec in pending_chunks_by_band[band_index]:
                bucket_prefix = str(chunk_spec["bucket_prefix"])
                result = build_candidate_band_chunk(
                    stage_root=stage_root,
                    band_index=band_index,
                    bucket_prefix=bucket_prefix,
                    bands=bands,
                    rows_per_band=rows_per_band,
                    minhash_threshold=minhash_threshold,
                    signature_map=signature_map,
                    signature_meta=signature_meta,
                    previous_band_summary=previous_bucket_summary.get(band_index, {}),
                    previous_stage_root=None if previous_run_root is None else previous_run_root / "stage_02_near",
                    max_bucket_size=max_bucket_size,
                )
                oversized_bucket_count += int(result["oversized_bucket_count"])
                oversized_bucket_member_rows += int(result["oversized_bucket_member_rows"])
                subdivided_bucket_count += int(result["subdivided_bucket_count"])
                fallback_chunked_bucket_count += int(result["fallback_chunked_bucket_count"])
                fallback_chunked_member_rows += int(result["fallback_chunked_member_rows"])
                reused_bucket_count += int(result["reused_bucket_count"])
                recomputed_bucket_count += int(result["recomputed_bucket_count"])
                mark_stage_chunk_complete(
                    conn,
                    run_id=run_id,
                    stage=NEAR_CANDIDATE_STAGE,
                    chunk_key=str(result["chunk_key"]),
                    artifact_path=Path(str(result["artifact_path"])),
                    row_count=int(result["row_count"]),
                )
                total_chunks, completed_chunks = stage_chunk_counts(conn, run_id=run_id, stage=NEAR_CANDIDATE_STAGE)
                upsert_stage_progress(
                    conn,
                    run_id=run_id,
                    stage=NEAR_CANDIDATE_STAGE,
                    status="running",
                    total_chunks=total_chunks,
                    completed_chunks=completed_chunks,
                    progress_path=progress_path,
                    payload={
                        "run_id": run_id,
                        "stage": NEAR_CANDIDATE_STAGE,
                        "status": "running",
                        "total_chunks": total_chunks,
                        "completed_chunks": completed_chunks,
                    },
                )
    elif pending_chunks:
        try:
            pool_context, initializer, initargs = candidate_pool_initializer_config(signatures_path)
            partition_worker_count = effective_worker_count(
                min(max_workers, near_candidate_partition_worker_cap()),
                len(pending_band_indexes),
            )
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=pool_context,
                initializer=initializer,
                initargs=initargs,
            ) as executor:
                future_map: dict[Any, str] = {}

                def drain_completed(*, wait_mode: Any) -> None:
                    nonlocal oversized_bucket_count
                    nonlocal oversized_bucket_member_rows
                    nonlocal subdivided_bucket_count
                    nonlocal fallback_chunked_bucket_count
                    nonlocal fallback_chunked_member_rows
                    nonlocal reused_bucket_count
                    nonlocal recomputed_bucket_count
                    if not future_map:
                        return
                    done, _ = wait(set(future_map), return_when=wait_mode)
                    for future in done:
                        future_map.pop(future)
                        result = future.result()
                        oversized_bucket_count += int(result["oversized_bucket_count"])
                        oversized_bucket_member_rows += int(result["oversized_bucket_member_rows"])
                        subdivided_bucket_count += int(result["subdivided_bucket_count"])
                        fallback_chunked_bucket_count += int(result["fallback_chunked_bucket_count"])
                        fallback_chunked_member_rows += int(result["fallback_chunked_member_rows"])
                        reused_bucket_count += int(result["reused_bucket_count"])
                        recomputed_bucket_count += int(result["recomputed_bucket_count"])
                        mark_stage_chunk_complete(
                            conn,
                            run_id=run_id,
                            stage=NEAR_CANDIDATE_STAGE,
                            chunk_key=str(result["chunk_key"]),
                            artifact_path=Path(str(result["artifact_path"])),
                            row_count=int(result["row_count"]),
                        )
                        total_chunks, completed_chunks = stage_chunk_counts(conn, run_id=run_id, stage=NEAR_CANDIDATE_STAGE)
                        upsert_stage_progress(
                            conn,
                            run_id=run_id,
                            stage=NEAR_CANDIDATE_STAGE,
                            status="running",
                            total_chunks=total_chunks,
                            completed_chunks=completed_chunks,
                            progress_path=progress_path,
                            payload={
                                "run_id": run_id,
                                "stage": NEAR_CANDIDATE_STAGE,
                                "status": "running",
                                "total_chunks": total_chunks,
                                "completed_chunks": completed_chunks,
                            },
                        )

                submit_buffer = max(worker_count * 2, worker_count)
                if partition_worker_count <= 1:
                    for band_index in pending_band_indexes:
                        prefixes = prepare_candidate_bucket_partitions_for_band(stage_root, band_index=band_index)
                        append_debug_trace(
                            trace_path,
                            f"near_candidates:partitions_ready band={band_index:02d} prefixes={len(prefixes)}",
                        )
                        for chunk_spec in pending_chunks_by_band[band_index]:
                            while len(future_map) >= submit_buffer:
                                drain_completed(wait_mode=FIRST_COMPLETED)
                            future = executor.submit(
                                build_candidate_band_chunk,
                                stage_root=stage_root,
                                band_index=band_index,
                                bucket_prefix=str(chunk_spec["bucket_prefix"]),
                                bands=bands,
                                rows_per_band=rows_per_band,
                                minhash_threshold=minhash_threshold,
                                previous_band_summary=previous_bucket_summary.get(band_index, {}),
                                previous_stage_root=None if previous_run_root is None else previous_run_root / "stage_02_near",
                                max_bucket_size=max_bucket_size,
                            )
                            future_map[future] = near_candidate_chunk_key(
                                band_index=band_index,
                                bucket_prefix=str(chunk_spec["bucket_prefix"]),
                            )
                else:
                    with ProcessPoolExecutor(max_workers=partition_worker_count, mp_context=PROCESS_POOL_CONTEXT) as partition_executor:
                        partition_futures = {
                            partition_executor.submit(
                                prepare_candidate_bucket_partitions_for_band,
                                stage_root,
                                band_index=band_index,
                            ): band_index
                            for band_index in pending_band_indexes
                        }
                        while partition_futures or future_map:
                            ready = wait(
                                set(partition_futures) | set(future_map),
                                return_when=FIRST_COMPLETED,
                            )[0]
                            for future in ready:
                                if future in partition_futures:
                                    band_index = partition_futures.pop(future)
                                    prefixes = future.result()
                                    append_debug_trace(
                                        trace_path,
                                        f"near_candidates:partitions_ready band={band_index:02d} prefixes={len(prefixes)}",
                                    )
                                    for chunk_spec in pending_chunks_by_band[band_index]:
                                        while len(future_map) >= submit_buffer:
                                            drain_completed(wait_mode=FIRST_COMPLETED)
                                        chunk_future = executor.submit(
                                            build_candidate_band_chunk,
                                            stage_root=stage_root,
                                            band_index=band_index,
                                            bucket_prefix=str(chunk_spec["bucket_prefix"]),
                                            bands=bands,
                                            rows_per_band=rows_per_band,
                                            minhash_threshold=minhash_threshold,
                                            previous_band_summary=previous_bucket_summary.get(band_index, {}),
                                            previous_stage_root=None if previous_run_root is None else previous_run_root / "stage_02_near",
                                            max_bucket_size=max_bucket_size,
                                        )
                                        future_map[chunk_future] = near_candidate_chunk_key(
                                            band_index=band_index,
                                            bucket_prefix=str(chunk_spec["bucket_prefix"]),
                                        )
                                else:
                                    if future not in future_map:
                                        continue
                                    future_map.pop(future)
                                    result = future.result()
                                    oversized_bucket_count += int(result["oversized_bucket_count"])
                                    oversized_bucket_member_rows += int(result["oversized_bucket_member_rows"])
                                    subdivided_bucket_count += int(result["subdivided_bucket_count"])
                                    fallback_chunked_bucket_count += int(result["fallback_chunked_bucket_count"])
                                    fallback_chunked_member_rows += int(result["fallback_chunked_member_rows"])
                                    reused_bucket_count += int(result["reused_bucket_count"])
                                    recomputed_bucket_count += int(result["recomputed_bucket_count"])
                                    mark_stage_chunk_complete(
                                        conn,
                                        run_id=run_id,
                                        stage=NEAR_CANDIDATE_STAGE,
                                        chunk_key=str(result["chunk_key"]),
                                        artifact_path=Path(str(result["artifact_path"])),
                                        row_count=int(result["row_count"]),
                                    )
                                    total_chunks, completed_chunks = stage_chunk_counts(conn, run_id=run_id, stage=NEAR_CANDIDATE_STAGE)
                                    upsert_stage_progress(
                                        conn,
                                        run_id=run_id,
                                        stage=NEAR_CANDIDATE_STAGE,
                                        status="running",
                                        total_chunks=total_chunks,
                                        completed_chunks=completed_chunks,
                                        progress_path=progress_path,
                                        payload={
                                            "run_id": run_id,
                                            "stage": NEAR_CANDIDATE_STAGE,
                                            "status": "running",
                                            "total_chunks": total_chunks,
                                            "completed_chunks": completed_chunks,
                                        },
                                    )
                while future_map:
                    drain_completed(wait_mode=FIRST_COMPLETED)
        except BrokenProcessPool:
            append_debug_trace(trace_path, f"near_candidates:broken_process_pool worker_count={worker_count} retry=serial")
            return _run_near_candidate_stage(
                conn,
                run_id=run_id,
                run_root=run_root,
                state_root=state_root,
                config=config,
                minhash_threshold=minhash_threshold,
                bands=bands,
                rows_per_band=rows_per_band,
                max_workers=1,
                max_bucket_size=max_bucket_size,
            )
    touched_doc_count, touched_doc_path = aggregate_touched_doc_shards(stage_root)
    candidate_pair_rows, candidate_pairs_db_path = aggregate_candidate_shards_to_parquet(stage_root)
    bucket_summary_rows, bucket_summary_path = summarize_bucket_summary_shards(stage_root)
    summary = {
        "run_id": run_id,
        "stage": NEAR_CANDIDATE_STAGE,
        "candidate_pairs_path": str(stage_root / "candidate_pairs.parquet"),
        "candidate_pairs_db_path": str(candidate_pairs_db_path),
        "bucket_summary_path": str(bucket_summary_path),
        "bucket_summary_rows": int(bucket_summary_rows),
        "touched_doc_path": str(touched_doc_path),
        "touched_doc_count": int(touched_doc_count),
        "candidate_pair_rows": int(candidate_pair_rows),
        "threshold": float(minhash_threshold),
        "max_bucket_size": int(max_bucket_size),
        "oversized_bucket_count": int(oversized_bucket_count),
        "oversized_bucket_member_rows": int(oversized_bucket_member_rows),
        "subdivided_bucket_count": int(subdivided_bucket_count),
        "fallback_chunked_bucket_count": int(fallback_chunked_bucket_count),
        "fallback_chunked_member_rows": int(fallback_chunked_member_rows),
        "reused_bucket_count": int(reused_bucket_count),
        "recomputed_bucket_count": int(recomputed_bucket_count),
    }
    upsert_stage_progress(
        conn,
        run_id=run_id,
        stage=NEAR_CANDIDATE_STAGE,
        status="completed",
        total_chunks=total_chunks,
        completed_chunks=total_chunks,
        progress_path=progress_path,
        payload=summary,
    )
    return summary


def load_run_doc_metadata(conn: sqlite3.Connection, *, run_id: str, doc_keys: list[str]) -> dict[str, dict[str, Any]]:
    if not doc_keys:
        return {}
    rows: dict[str, dict[str, Any]] = {}
    chunk_size = 500
    for offset in range(0, len(doc_keys), chunk_size):
        chunk = doc_keys[offset : offset + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        query = f"SELECT * FROM run_docs WHERE run_id = ? AND doc_key IN ({placeholders})"
        for row in conn.execute(query, (run_id, *chunk)):
            rows[str(row["doc_key"])] = dict(row)
    return rows


def _resolve_near_component(
    component_doc_keys: set[str],
    *,
    adjacency: dict[str, set[str]],
    signature_map: SignatureLookup,
    signature_meta: SignatureMetadataLookup,
    doc_meta: dict[str, dict[str, Any]],
    minhash_threshold: float,
    large_component_threshold: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    component_size = len(component_doc_keys)
    if component_size == 0:
        return [], []
    if component_size == 1:
        doc_key = next(iter(component_doc_keys))
        source = doc_meta[doc_key]
        cluster_id = cluster_id_for_doc_keys("near", [doc_key])
        return (
            [
                {
                    "cluster_id": cluster_id,
                    "kept_doc_key": doc_key,
                    "member_doc_key": doc_key,
                    "member_source_dataset": str(source["source_dataset"]),
                    "member_source_doc_id": str(source["source_doc_id"]),
                    "dropped": False,
                    "estimated_jaccard": 1.0,
                    "shingle_mode": signature_meta.shingle_mode(doc_key),
                    "token_count": signature_meta.token_count(doc_key),
                    "char_count": signature_meta.char_count(doc_key),
                    "length_ratio": 1.0,
                    "likely_containment_flag": False,
                    "accepted_reason": "singleton",
                    "cluster_size": 1,
                    "component_size": 1,
                    "large_component_audit_flag": False,
                }
            ],
            [
                {
                    "cluster_id": cluster_id,
                    "decision_stage": "near_duplicate",
                    "cluster_size": 1,
                    "dropped_count": 0,
                    "kept_doc_key": doc_key,
                    "mixed_source": False,
                    "large_component_audit_flag": False,
                    "narrow_margin_flag": False,
                }
            ],
        )
    ranked = sorted(
        component_doc_keys,
        key=lambda key: selection_priority_tuple(
            {**doc_meta[key], "near_text_chars": signature_meta.near_text_chars(key)},
            text_length_field="near_text_chars",
        ),
    )
    representative = ranked[0]
    accepted = [representative]
    weak_members: list[str] = []
    for doc_key in ranked[1:]:
        left_tokens = signature_meta.token_count(representative)
        right_tokens = signature_meta.token_count(doc_key)
        longer = max(left_tokens, right_tokens)
        shorter = min(left_tokens, right_tokens)
        length_ratio = 0.0 if longer == 0 else float(shorter / longer)
        estimated_jaccard = signature_jaccard(signature_map[representative], signature_map[doc_key])
        if estimated_jaccard >= minhash_threshold:
            accepted.append(doc_key)
        else:
            weak_members.append(doc_key)
    large_component_flag = component_size > large_component_threshold
    accepted_cluster_id = cluster_id_for_doc_keys("near", accepted)
    accepted_rows: list[dict[str, Any]] = []
    for doc_key in accepted:
        source = doc_meta[doc_key]
        if doc_key == representative:
            estimated_jaccard = 1.0
            length_ratio = 1.0
            accepted_reason = "representative"
        else:
            longer = max(signature_meta.token_count(representative), signature_meta.token_count(doc_key))
            shorter = min(signature_meta.token_count(representative), signature_meta.token_count(doc_key))
            length_ratio = 0.0 if longer == 0 else float(shorter / longer)
            estimated_jaccard = signature_jaccard(signature_map[representative], signature_map[doc_key])
            accepted_reason = "representative_validation"
        accepted_rows.append(
            {
                "cluster_id": accepted_cluster_id,
                "kept_doc_key": representative,
                "member_doc_key": doc_key,
                "member_source_dataset": str(source["source_dataset"]),
                "member_source_doc_id": str(source["source_doc_id"]),
                "dropped": bool(doc_key != representative),
                "estimated_jaccard": float(estimated_jaccard),
                "shingle_mode": signature_meta.shingle_mode(doc_key),
                "token_count": signature_meta.token_count(doc_key),
                "char_count": signature_meta.char_count(doc_key),
                "length_ratio": float(length_ratio),
                "likely_containment_flag": bool(length_ratio < 0.85 and doc_key != representative),
                "accepted_reason": accepted_reason,
                "cluster_size": int(len(accepted)),
                "component_size": int(component_size),
                "large_component_audit_flag": bool(large_component_flag),
            }
        )
    priority_values = [
        selection_priority_tuple(
            {**doc_meta[key], "near_text_chars": signature_meta.near_text_chars(key)},
            text_length_field="near_text_chars",
        )
        for key in accepted
    ]
    narrow_margin_flag = bool(len(priority_values) > 1 and priority_values[0][:7] == priority_values[1][:7])
    summary_rows = [
        {
            "cluster_id": accepted_cluster_id,
            "decision_stage": "near_duplicate",
            "cluster_size": int(len(accepted)),
            "dropped_count": int(max(0, len(accepted) - 1)),
            "kept_doc_key": representative,
            "mixed_source": bool(len({str(doc_meta[key]["source_dataset"]) for key in accepted}) > 1),
            "large_component_audit_flag": bool(large_component_flag),
            "narrow_margin_flag": bool(narrow_margin_flag),
        }
    ]
    if not weak_members:
        return accepted_rows, summary_rows
    weak_components = near_component_subgraphs(
        set(weak_members),
        {key: adjacency.get(key, set()) & set(weak_members) for key in weak_members},
    )
    all_rows = list(accepted_rows)
    all_summaries = list(summary_rows)
    for weak_component in weak_components:
        resolved_rows, resolved_summaries = _resolve_near_component(
            weak_component,
            adjacency=adjacency,
            signature_map=signature_map,
            signature_meta=signature_meta,
            doc_meta=doc_meta,
            minhash_threshold=minhash_threshold,
            large_component_threshold=large_component_threshold,
        )
        all_rows.extend(resolved_rows)
        all_summaries.extend(resolved_summaries)
    return all_rows, all_summaries


def singleton_near_component_rows(
    doc_key: str,
    *,
    signature_meta: SignatureMetadataLookup,
    doc_meta: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = doc_meta[doc_key]
    cluster_id = cluster_id_for_doc_keys("near", [doc_key])
    return (
        [
            {
                "cluster_id": cluster_id,
                "kept_doc_key": doc_key,
                "member_doc_key": doc_key,
                "member_source_dataset": str(source["source_dataset"]),
                "member_source_doc_id": str(source["source_doc_id"]),
                "dropped": False,
                "estimated_jaccard": 1.0,
                "shingle_mode": signature_meta.shingle_mode(doc_key),
                "token_count": signature_meta.token_count(doc_key),
                "char_count": signature_meta.char_count(doc_key),
                "length_ratio": 1.0,
                "likely_containment_flag": False,
                "accepted_reason": "singleton",
                "cluster_size": 1,
                "component_size": 1,
                "large_component_audit_flag": False,
            }
        ],
        [
            {
                "cluster_id": cluster_id,
                "decision_stage": "near_duplicate",
                "cluster_size": 1,
                "dropped_count": 0,
                "kept_doc_key": doc_key,
                "mixed_source": False,
                "large_component_audit_flag": False,
                "narrow_margin_flag": False,
            }
        ],
    )


def resolve_near_cluster_chunk(
    *,
    stage_root: Path,
    chunk_spec: dict[str, Any],
    adjacency: dict[str, set[str]] | None = None,
    signature_map: SignatureLookup | None = None,
    signature_meta: SignatureMetadataLookup | None = None,
    doc_meta: dict[str, dict[str, Any]] | None = None,
    state_root: Path | None = None,
    run_id: str | None = None,
    minhash_threshold: float,
    large_component_threshold: int,
) -> dict[str, Any]:
    if adjacency is None:
        adjacency = _CLUSTER_WORKER_ADJACENCY
    if signature_map is None:
        signature_map = _CLUSTER_WORKER_SIGNATURE_MAP
    if signature_meta is None:
        signature_meta = _CLUSTER_WORKER_SIGNATURE_META
    if adjacency is None or signature_map is None or signature_meta is None:
        raise ValueError("near cluster worker state is not initialized")
    if doc_meta is None:
        if state_root is None or run_id is None:
            raise ValueError("state_root and run_id are required when doc_meta is not provided")
        doc_keys = set(chunk_spec.get("singleton_doc_keys", []))
        for component_payload in list(chunk_spec.get("components", [])):
            doc_keys.update(component_payload["doc_keys"])
        reader = connect_db_reader(state_root)
        try:
            doc_meta = load_run_doc_metadata(reader, run_id=run_id, doc_keys=sorted(doc_keys))
        finally:
            reader.close()
    chunk_rows: list[dict[str, Any]] = []
    chunk_summaries: list[dict[str, Any]] = []
    for doc_key in list(chunk_spec.get("singleton_doc_keys", [])):
        resolved_rows, resolved_summaries = singleton_near_component_rows(
            str(doc_key),
            signature_meta=signature_meta,
            doc_meta=doc_meta,
        )
        chunk_rows.extend(resolved_rows)
        chunk_summaries.extend(resolved_summaries)
    for component_payload in list(chunk_spec.get("components", [])):
        resolved_rows, resolved_summaries = _resolve_near_component(
            set(component_payload["doc_keys"]),
            adjacency=adjacency,
            signature_map=signature_map,
            signature_meta=signature_meta,
            doc_meta=doc_meta,
            minhash_threshold=minhash_threshold,
            large_component_threshold=large_component_threshold,
        )
        chunk_rows.extend(resolved_rows)
        chunk_summaries.extend(resolved_summaries)
    chunk_rows = sorted(chunk_rows, key=lambda row: (str(row["cluster_id"]), str(row["member_doc_key"])))
    chunk_summaries = sorted(chunk_summaries, key=lambda row: str(row["cluster_id"]))
    cluster_path = near_cluster_chunk_shard_path(stage_root, chunk_index=int(chunk_spec["chunk_index"]))
    summary_path = near_cluster_summary_shard_path(stage_root, chunk_index=int(chunk_spec["chunk_index"]))
    write_group_parquet(chunk_rows, cluster_path, schema=NEAR_CLUSTER_SCHEMA)
    write_group_parquet(chunk_summaries, summary_path, schema=CLUSTER_SUMMARY_SCHEMA)
    return {
        "chunk_key": str(chunk_spec["chunk_key"]),
        "artifact_path": str(cluster_path),
        "row_count": int(len(chunk_rows)),
        "summary_path": str(summary_path),
        "summary_row_count": int(len(chunk_summaries)),
    }


def load_touched_doc_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    touched: set[str] = set()
    for batch_rows in iter_parquet_batches(path):
        for row in batch_rows:
            touched.add(str(row["doc_key"]))
    return touched


def load_previous_cluster_membership(path: Path) -> tuple[dict[str, str], dict[str, str], dict[str, int]]:
    cluster_doc_keys: dict[str, list[str]] = defaultdict(list)
    cluster_id_by_doc: dict[str, str] = {}
    for batch_rows in iter_parquet_batches(path, columns=["cluster_id", "member_doc_key"]):
        for row in batch_rows:
            cluster_id = str(row["cluster_id"])
            doc_key = str(row["member_doc_key"])
            cluster_id_by_doc[doc_key] = cluster_id
            cluster_doc_keys[cluster_id].append(doc_key)
    cluster_digest_by_id = {cluster_id: digest_doc_keys(doc_keys) for cluster_id, doc_keys in cluster_doc_keys.items()}
    cluster_size_by_id = {cluster_id: len(doc_keys) for cluster_id, doc_keys in cluster_doc_keys.items()}
    return cluster_id_by_doc, cluster_digest_by_id, cluster_size_by_id


def write_reused_near_cluster_shards(
    *,
    previous_stage_root: Path,
    current_stage_root: Path,
    reusable_cluster_ids: set[str],
) -> tuple[int, int]:
    cluster_path = current_stage_root / "shards" / "near_clusters" / "reused_previous.parquet"
    summary_path = current_stage_root / "shards" / "cluster_summaries" / "reused_previous.parquet"
    cluster_writer: pq.ParquetWriter | None = None
    summary_writer: pq.ParquetWriter | None = None
    cluster_temp_path = temp_output_path(cluster_path)
    summary_temp_path = temp_output_path(summary_path)
    cluster_rows = 0
    summary_rows = 0
    try:
        for batch_rows in iter_parquet_batches(previous_stage_root / "near_clusters.parquet"):
            filtered = [row for row in batch_rows if str(row["cluster_id"]) in reusable_cluster_ids]
            cluster_writer, written_rows = append_rows_to_parquet_writer(
                cluster_writer,
                rows=filtered,
                temp_path=cluster_temp_path,
                schema=NEAR_CLUSTER_SCHEMA,
            )
            cluster_rows += written_rows
        for batch_rows in iter_parquet_batches(previous_stage_root / "cluster_summary.parquet"):
            filtered = [row for row in batch_rows if str(row["cluster_id"]) in reusable_cluster_ids]
            summary_writer, written_rows = append_rows_to_parquet_writer(
                summary_writer,
                rows=filtered,
                temp_path=summary_temp_path,
                schema=CLUSTER_SUMMARY_SCHEMA,
            )
            summary_rows += written_rows
    finally:
        finalize_parquet_writer(
            cluster_writer,
            temp_path=cluster_temp_path,
            destination=cluster_path,
            schema=NEAR_CLUSTER_SCHEMA,
        )
        finalize_parquet_writer(
            summary_writer,
            temp_path=summary_temp_path,
            destination=summary_path,
            schema=CLUSTER_SUMMARY_SCHEMA,
        )
    return cluster_rows, summary_rows


def _run_near_cluster_stage(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    run_root: Path,
    state_root: Path,
    config: dict[str, Any],
    minhash_threshold: float,
    large_component_threshold: int,
) -> dict[str, Any]:
    stage_root = run_root / "stage_02_near"
    progress_path = progress_file_path(run_root, NEAR_CLUSTER_STAGE)
    trace_path = progress_dir(run_root) / "near_cluster_trace.log"
    existing_summary = read_json_if_exists(progress_path)
    near_cluster_required_paths = [
        stage_root / "near_clusters.parquet",
        stage_root / "cluster_summary.parquet",
        stage_root / "near_drop_list.parquet",
    ]
    if existing_summary is not None and all_paths_exist(near_cluster_required_paths):
        total_chunks = int(existing_summary.get("total_chunks", 1))
        completed_chunks = int(existing_summary.get("completed_chunks", total_chunks))
        upsert_stage_progress(
            conn,
            run_id=run_id,
            stage=NEAR_CLUSTER_STAGE,
            status="completed",
            total_chunks=total_chunks,
            completed_chunks=completed_chunks,
            progress_path=progress_path,
            payload=existing_summary,
        )
        return existing_summary
    upsert_stage_progress(
        conn,
        run_id=run_id,
        stage=NEAR_CLUSTER_STAGE,
        status="running",
        total_chunks=1,
        completed_chunks=0,
        progress_path=progress_path,
        payload={
            "run_id": run_id,
            "stage": NEAR_CLUSTER_STAGE,
            "status": "running",
            "phase": "prelude",
            "total_chunks": 1,
            "completed_chunks": 0,
        },
    )
    append_debug_trace(trace_path, "near_clusters:start")
    signature_map, signature_meta = load_signature_index(stage_root / "signatures.parquet")
    append_debug_trace(
        trace_path,
        f"near_clusters:signatures_loaded docs={len(signature_map.row_by_doc_key)}",
    )
    adjacency: dict[str, set[str]] = defaultdict(set)
    candidate_pair_rows = 0
    for batch_rows in iter_parquet_batches(stage_root / "candidate_pairs.parquet"):
        for row in batch_rows:
            left = str(row["doc_key_left"])
            right = str(row["doc_key_right"])
            adjacency[left].add(right)
            adjacency[right].add(left)
            candidate_pair_rows += 1
    append_debug_trace(
        trace_path,
        f"near_clusters:adjacency_loaded candidate_pair_rows={candidate_pair_rows} nodes={len(adjacency)}",
    )
    existing_cluster_chunk_keys = [
        str(row["chunk_key"])
        for row in conn.execute(
            "SELECT chunk_key FROM run_stage_chunks WHERE run_id = ? AND stage = ?",
            (run_id, NEAR_CLUSTER_STAGE),
        )
    ]
    uses_legacy_component_only_layout = uses_legacy_near_cluster_component_only_layout(existing_cluster_chunk_keys)
    if uses_legacy_component_only_layout:
        components = near_component_subgraphs(set(signature_map), adjacency)
        connected_components = components
        singleton_doc_keys: list[str] = []
        append_debug_trace(
            trace_path,
            f"near_clusters:components legacy_component_only=true total={len(components)}",
        )
    else:
        connected_doc_keys = set(adjacency)
        connected_components = near_component_subgraphs(set(connected_doc_keys), adjacency) if connected_doc_keys else []
        singleton_doc_keys = [str(doc_key) for doc_key in signature_map.row_by_doc_key if doc_key not in connected_doc_keys]
        components = connected_components
        append_debug_trace(
            trace_path,
            f"near_clusters:components connected={len(connected_components)} singletons={len(singleton_doc_keys)}",
        )
    touched_doc_keys = load_touched_doc_keys(stage_root / "touched_doc_keys.parquet")
    previous_run_root = latest_compatible_run_root(
        state_root=state_root,
        current_run_root=run_root,
        current_config=config,
        keys=[
            "greek_diacritic_policy",
            "near_norm_version",
            "tokenization_version",
            "minhash_version",
            "lsh_version",
            "selection_version",
            "near_incremental_version",
            "minhash_threshold",
            "num_perm",
            "bands",
            "rows_per_band",
            "shingle_mode",
            "shingle_size",
            "large_component_threshold",
        ],
        required_relative_paths=[
            Path("stage_02_near") / "near_clusters.parquet",
            Path("stage_02_near") / "cluster_summary.parquet",
            Path("stage_02_near") / "touched_doc_keys.parquet",
        ],
    )
    reusable_cluster_ids: set[str] = set()
    recompute_components: list[set[str]] = components
    if previous_run_root is not None:
        previous_cluster_id_by_doc, previous_digest_by_id, previous_size_by_id = load_previous_cluster_membership(
            previous_run_root / "stage_02_near" / "near_clusters.parquet"
        )
        recompute_components = []
        for component in components:
            if component & touched_doc_keys:
                recompute_components.append(component)
                continue
            sample_doc_key = next(iter(component))
            previous_cluster_id = previous_cluster_id_by_doc.get(sample_doc_key)
            if previous_cluster_id is None:
                recompute_components.append(component)
                continue
            if previous_size_by_id.get(previous_cluster_id) != len(component):
                recompute_components.append(component)
                continue
            if previous_digest_by_id.get(previous_cluster_id) != digest_doc_keys(component):
                recompute_components.append(component)
                continue
            reusable_cluster_ids.add(previous_cluster_id)
    component_chunk_specs = build_near_cluster_chunk_specs(recompute_components)
    singleton_chunk_specs: list[dict[str, Any]] = []
    if not uses_legacy_component_only_layout:
        singleton_chunk_specs = build_near_cluster_singleton_chunk_specs(
            singleton_doc_keys,
            start_index=len(component_chunk_specs),
        )
    cluster_chunk_specs = [*component_chunk_specs, *singleton_chunk_specs]
    reusable_chunk_key = "reuse:previous_components"
    cluster_chunk_keys = [str(chunk_spec["chunk_key"]) for chunk_spec in cluster_chunk_specs]
    if reusable_cluster_ids:
        cluster_chunk_keys = [reusable_chunk_key, *cluster_chunk_keys]
    register_stage_chunks(
        conn,
        run_id=run_id,
        stage=NEAR_CLUSTER_STAGE,
        chunk_keys=cluster_chunk_keys,
    )
    total_chunks, completed_chunks = stage_chunk_counts(conn, run_id=run_id, stage=NEAR_CLUSTER_STAGE)
    upsert_stage_progress(
        conn,
        run_id=run_id,
        stage=NEAR_CLUSTER_STAGE,
        status="running",
        total_chunks=total_chunks,
        completed_chunks=completed_chunks,
        progress_path=progress_path,
        payload={
            "run_id": run_id,
            "stage": NEAR_CLUSTER_STAGE,
            "status": "running",
            "total_chunks": total_chunks,
            "completed_chunks": completed_chunks,
            },
        )
    append_debug_trace(
        trace_path,
        f"near_clusters:chunks_registered total={total_chunks} reusable={len(reusable_cluster_ids)} component_chunks={len(component_chunk_specs)} singleton_chunks={len(singleton_chunk_specs)}",
    )
    reused_cluster_count = len(reusable_cluster_ids)
    if reusable_cluster_ids and stage_chunk_status(conn, run_id=run_id, stage=NEAR_CLUSTER_STAGE, chunk_key=reusable_chunk_key) != "completed":
        reused_cluster_rows, reused_summary_rows = write_reused_near_cluster_shards(
            previous_stage_root=previous_run_root / "stage_02_near",
            current_stage_root=stage_root,
            reusable_cluster_ids=reusable_cluster_ids,
        )
        mark_stage_chunk_complete(
            conn,
            run_id=run_id,
            stage=NEAR_CLUSTER_STAGE,
            chunk_key=reusable_chunk_key,
            artifact_path=stage_root / "shards" / "near_clusters" / "reused_previous.parquet",
            row_count=int(reused_cluster_rows),
        )
        total_chunks, completed_chunks = stage_chunk_counts(conn, run_id=run_id, stage=NEAR_CLUSTER_STAGE)
        upsert_stage_progress(
            conn,
            run_id=run_id,
            stage=NEAR_CLUSTER_STAGE,
            status="running",
            total_chunks=total_chunks,
            completed_chunks=completed_chunks,
            progress_path=progress_path,
            payload={
                "run_id": run_id,
                "stage": NEAR_CLUSTER_STAGE,
                "status": "running",
                "total_chunks": total_chunks,
                "completed_chunks": completed_chunks,
                "reused_cluster_rows": int(reused_cluster_rows),
                "reused_summary_rows": int(reused_summary_rows),
            },
        )
    pending_cluster_chunk_specs = [
        chunk_spec
        for chunk_spec in cluster_chunk_specs
        if stage_chunk_status(conn, run_id=run_id, stage=NEAR_CLUSTER_STAGE, chunk_key=str(chunk_spec["chunk_key"])) != "completed"
    ]
    cluster_worker_count = 1
    if pending_cluster_chunk_specs:
        pool_context = near_cluster_process_pool_context()
        if pool_context.get_start_method() == "fork":
            cluster_worker_count = effective_worker_count(
                min(max_workers, near_cluster_worker_cap()),
                len(pending_cluster_chunk_specs),
            )
        append_debug_trace(
            trace_path,
            f"near_clusters:chunk_resolution_start pending_chunks={len(pending_cluster_chunk_specs)} worker_count={cluster_worker_count}",
        )
    if cluster_worker_count == 1:
        for chunk_spec in pending_cluster_chunk_specs:
            result = resolve_near_cluster_chunk(
                stage_root=stage_root,
                chunk_spec=chunk_spec,
                adjacency=adjacency,
                signature_map=signature_map,
                signature_meta=signature_meta,
                state_root=state_root,
                run_id=run_id,
                minhash_threshold=minhash_threshold,
                large_component_threshold=large_component_threshold,
            )
            mark_stage_chunk_complete(
                conn,
                run_id=run_id,
                stage=NEAR_CLUSTER_STAGE,
                chunk_key=str(result["chunk_key"]),
                artifact_path=Path(str(result["artifact_path"])),
                row_count=int(result["row_count"]),
            )
            total_chunks, completed_chunks = stage_chunk_counts(conn, run_id=run_id, stage=NEAR_CLUSTER_STAGE)
            upsert_stage_progress(
                conn,
                run_id=run_id,
                stage=NEAR_CLUSTER_STAGE,
                status="running",
                total_chunks=total_chunks,
                completed_chunks=completed_chunks,
                progress_path=progress_path,
                payload={
                    "run_id": run_id,
                    "stage": NEAR_CLUSTER_STAGE,
                    "status": "running",
                    "total_chunks": total_chunks,
                    "completed_chunks": completed_chunks,
                },
            )
    elif pending_cluster_chunk_specs:
        try:
            pool_context, initializer, initargs = cluster_pool_initializer_config(
                signature_map=signature_map,
                signature_meta=signature_meta,
                adjacency=adjacency,
            )
            with ProcessPoolExecutor(
                max_workers=cluster_worker_count,
                mp_context=pool_context,
                initializer=initializer,
                initargs=initargs,
            ) as executor:
                future_map = {
                    executor.submit(
                        resolve_near_cluster_chunk,
                        stage_root=stage_root,
                        chunk_spec=chunk_spec,
                        state_root=state_root,
                        run_id=run_id,
                        minhash_threshold=minhash_threshold,
                        large_component_threshold=large_component_threshold,
                    ): str(chunk_spec["chunk_key"])
                    for chunk_spec in pending_cluster_chunk_specs
                }
                while future_map:
                    done, _ = wait(set(future_map), return_when=FIRST_COMPLETED)
                    for future in done:
                        chunk_key = future_map.pop(future)
                        result = future.result()
                        mark_stage_chunk_complete(
                            conn,
                            run_id=run_id,
                            stage=NEAR_CLUSTER_STAGE,
                            chunk_key=chunk_key,
                            artifact_path=Path(str(result["artifact_path"])),
                            row_count=int(result["row_count"]),
                        )
                        total_chunks, completed_chunks = stage_chunk_counts(conn, run_id=run_id, stage=NEAR_CLUSTER_STAGE)
                        upsert_stage_progress(
                            conn,
                            run_id=run_id,
                            stage=NEAR_CLUSTER_STAGE,
                            status="running",
                            total_chunks=total_chunks,
                            completed_chunks=completed_chunks,
                            progress_path=progress_path,
                            payload={
                                "run_id": run_id,
                                "stage": NEAR_CLUSTER_STAGE,
                                "status": "running",
                                "total_chunks": total_chunks,
                                "completed_chunks": completed_chunks,
                            },
                        )
        except BrokenProcessPool:
            append_debug_trace(trace_path, f"near_clusters:broken_process_pool worker_count={cluster_worker_count} retry=serial")
            for chunk_spec in pending_cluster_chunk_specs:
                if stage_chunk_status(conn, run_id=run_id, stage=NEAR_CLUSTER_STAGE, chunk_key=str(chunk_spec["chunk_key"])) == "completed":
                    continue
                result = resolve_near_cluster_chunk(
                    stage_root=stage_root,
                    chunk_spec=chunk_spec,
                    adjacency=adjacency,
                    signature_map=signature_map,
                    signature_meta=signature_meta,
                    state_root=state_root,
                    run_id=run_id,
                    minhash_threshold=minhash_threshold,
                    large_component_threshold=large_component_threshold,
                )
                mark_stage_chunk_complete(
                    conn,
                    run_id=run_id,
                    stage=NEAR_CLUSTER_STAGE,
                    chunk_key=str(result["chunk_key"]),
                    artifact_path=Path(str(result["artifact_path"])),
                    row_count=int(result["row_count"]),
                )
                total_chunks, completed_chunks = stage_chunk_counts(conn, run_id=run_id, stage=NEAR_CLUSTER_STAGE)
                upsert_stage_progress(
                    conn,
                    run_id=run_id,
                    stage=NEAR_CLUSTER_STAGE,
                    status="running",
                    total_chunks=total_chunks,
                    completed_chunks=completed_chunks,
                    progress_path=progress_path,
                    payload={
                        "run_id": run_id,
                        "stage": NEAR_CLUSTER_STAGE,
                        "status": "running",
                        "total_chunks": total_chunks,
                        "completed_chunks": completed_chunks,
                    },
                )
    cluster_paths = sorted((stage_root / "shards" / "near_clusters").glob("*.parquet"))
    cluster_summary_paths = sorted((stage_root / "shards" / "cluster_summaries").glob("*.parquet"))
    cluster_rows = combine_parquet_files(cluster_paths, stage_root / "near_clusters.parquet", schema=NEAR_CLUSTER_SCHEMA)
    cluster_count = combine_parquet_files(cluster_summary_paths, stage_root / "cluster_summary.parquet", schema=CLUSTER_SUMMARY_SCHEMA)
    near_drop_path = stage_root / "near_drop_list.parquet"
    near_drop_temp_path = temp_output_path(near_drop_path)
    near_drop_writer: pq.ParquetWriter | None = None
    near_drop_rows = 0
    conn.execute("DELETE FROM run_near_results WHERE run_id = ?", (run_id,))
    try:
        for batch_rows in iter_parquet_batches(stage_root / "near_clusters.parquet"):
            conn.executemany(
                """
                INSERT INTO run_near_results (
                    run_id,
                    doc_key,
                    cluster_id,
                    kept_doc_key,
                    dropped,
                    estimated_jaccard,
                    shingle_mode,
                    token_count,
                    char_count,
                    length_ratio,
                    likely_containment_flag,
                    accepted_reason,
                    component_size,
                    cluster_size,
                    large_component_audit_flag
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        str(row["member_doc_key"]),
                        str(row["cluster_id"]),
                        str(row["kept_doc_key"]),
                        int(bool(row["dropped"])),
                        float(row["estimated_jaccard"]),
                        str(row["shingle_mode"]),
                        int(row["token_count"]),
                        int(row["char_count"]),
                        float(row["length_ratio"]),
                        int(bool(row["likely_containment_flag"])),
                        str(row["accepted_reason"]),
                        int(row["component_size"]),
                        int(row["cluster_size"]),
                        int(bool(row["large_component_audit_flag"])),
                    )
                    for row in batch_rows
                ],
            )
            conn.commit()
            drop_batch = [
                {
                    "doc_key": str(row["member_doc_key"]),
                    "source_dataset": str(row["member_source_dataset"]),
                    "source_doc_id": str(row["member_source_doc_id"]),
                    "kept_doc_key": str(row["kept_doc_key"]),
                    "cluster_id": str(row["cluster_id"]),
                    "estimated_jaccard": float(row["estimated_jaccard"]),
                    "shingle_mode": str(row["shingle_mode"]),
                    "token_count": int(row["token_count"]),
                    "char_count": int(row["char_count"]),
                    "length_ratio": float(row["length_ratio"]),
                    "likely_containment_flag": bool(row["likely_containment_flag"]),
                    "reason": "near_duplicate",
                }
                for row in batch_rows
                if bool(row["dropped"])
            ]
            near_drop_writer, written_drop_rows = append_rows_to_parquet_writer(
                near_drop_writer,
                rows=drop_batch,
                temp_path=near_drop_temp_path,
                schema=NEAR_DROP_SCHEMA,
            )
            near_drop_rows += written_drop_rows
    finally:
        finalize_parquet_writer(
            near_drop_writer,
            temp_path=near_drop_temp_path,
            destination=near_drop_path,
            schema=NEAR_DROP_SCHEMA,
        )
    summary = {
        "run_id": run_id,
        "stage": NEAR_CLUSTER_STAGE,
        "status": "completed",
        "total_chunks": int(total_chunks),
        "completed_chunks": int(total_chunks),
        "near_clusters_path": str(stage_root / "near_clusters.parquet"),
        "cluster_summary_path": str(stage_root / "cluster_summary.parquet"),
        "near_drop_list_path": str(near_drop_path),
        "candidate_pair_rows": int(candidate_pair_rows),
        "cluster_rows": int(cluster_rows),
        "near_dropped_rows": int(near_drop_rows),
        "cluster_count": int(cluster_count),
        "touched_doc_count": int(len(touched_doc_keys)),
        "reused_cluster_count": int(reused_cluster_count),
        "recomputed_cluster_count": int(len(recompute_components)),
        "large_component_threshold": int(large_component_threshold),
    }
    upsert_stage_progress(
        conn,
        run_id=run_id,
        stage=NEAR_CLUSTER_STAGE,
        status="completed",
        total_chunks=total_chunks,
        completed_chunks=total_chunks,
        progress_path=progress_path,
        payload=summary,
    )
    return summary


def build_final_kept_doc_lookup(*, run_id: str, run_root: Path) -> dict[str, str]:
    parent_by_doc_key: dict[str, str] = {}
    docs_exact_path = run_root / "stage_01_exact" / "docs_exact.parquet"
    near_clusters_path = run_root / "stage_02_near" / "near_clusters.parquet"
    if near_clusters_path.exists():
        query = """
            SELECT
                d.doc_key,
                d.strict_kept_doc_key,
                d.strict_dropped,
                d.relaxed_kept_doc_key,
                d.relaxed_dropped,
                n.kept_doc_key AS near_kept_doc_key,
                n.dropped AS near_dropped
            FROM read_parquet(?) AS d
            LEFT JOIN read_parquet(?) AS n
              ON n.member_doc_key = d.doc_key
        """
        params = [str(docs_exact_path), str(near_clusters_path)]
    else:
        query = """
            SELECT
                d.doc_key,
                d.strict_kept_doc_key,
                d.strict_dropped,
                d.relaxed_kept_doc_key,
                d.relaxed_dropped,
                NULL AS near_kept_doc_key,
                NULL AS near_dropped
            FROM read_parquet(?) AS d
        """
        params = [str(docs_exact_path)]
    doc_keys: list[str] = []
    for row in iter_duckdb_query_rows(query, params):
        doc_key = str(row["doc_key"])
        doc_keys.append(doc_key)
        if int(row["strict_dropped"] or 0) == 1 and row["strict_kept_doc_key"] is not None:
            parent_by_doc_key[doc_key] = str(row["strict_kept_doc_key"])
        if int(row["relaxed_dropped"] or 0) == 1 and row["relaxed_kept_doc_key"] is not None:
            parent_by_doc_key[doc_key] = str(row["relaxed_kept_doc_key"])
        if int(row["near_dropped"] or 0) == 1 and row["near_kept_doc_key"] is not None:
            parent_by_doc_key[doc_key] = str(row["near_kept_doc_key"])

    resolved: dict[str, str] = {}

    def resolve(doc_key: str) -> str:
        cached = resolved.get(doc_key)
        if cached is not None:
            return cached
        seen: set[str] = set()
        current = doc_key
        while True:
            if current in seen:
                raise ValueError(f"cycle detected while resolving final kept_doc_key lineage for run {run_id}: {doc_key}")
            seen.add(current)
            parent = parent_by_doc_key.get(current)
            if parent is None or parent == current:
                resolved[doc_key] = current
                return current
            current = parent

    for doc_key in doc_keys:
        resolve(doc_key)
    return resolved


def _build_final_exports(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    run_root: Path,
    greek_diacritic_policy: str,
    minhash_threshold: float,
    shingle_mode: str,
    shingle_size: int,
    near_cluster_summary_path: Path | None = None,
) -> dict[str, Any]:
    final_root = run_root / "final"
    ensure_dir(final_root)
    progress_path = progress_file_path(run_root, FINAL_EXPORT_STAGE)
    summary_path = final_root / "run_summary.json"
    existing_summary = read_json_if_exists(summary_path)
    final_required_paths = [
        summary_path,
        final_root / "dedup_decisions.parquet",
        final_root / "kept_docs.parquet",
        final_root / "dropped_docs.parquet",
        final_root / "cluster_summary.parquet",
        final_root / "edge_audit_sample.parquet",
        final_root / "cluster_audit_sample.parquet",
    ]
    if existing_summary is not None and all_paths_exist(final_required_paths):
        upsert_stage_progress(
            conn,
            run_id=run_id,
            stage=FINAL_EXPORT_STAGE,
            status="completed",
            total_chunks=1,
            completed_chunks=1,
            progress_path=progress_path,
            payload=existing_summary,
        )
        return existing_summary
    final_kept_lookup = build_final_kept_doc_lookup(run_id=run_id, run_root=run_root)
    docs_exact_path = run_root / "stage_01_exact" / "docs_exact.parquet"
    near_clusters_path = run_root / "stage_02_near" / "near_clusters.parquet"
    if near_clusters_path.exists():
        decision_query = """
            SELECT
                d.doc_key,
                d.source_dataset,
                d.source_doc_id,
                d.strict_group_size,
                d.strict_kept_doc_key,
                d.strict_dropped,
                d.relaxed_group_size,
                d.relaxed_kept_doc_key,
                d.relaxed_dropped,
                d.exact_strict_hash AS strict_group_hash,
                d.exact_relaxed_hash AS relaxed_group_hash,
                n.cluster_id AS near_cluster_id,
                n.kept_doc_key AS near_kept_doc_key,
                n.dropped AS near_dropped
            FROM read_parquet(?) AS d
            LEFT JOIN read_parquet(?) AS n
              ON n.member_doc_key = d.doc_key
            ORDER BY d.file_path, d.row_index_in_file
        """
        decision_params = [str(docs_exact_path), str(near_clusters_path)]
    else:
        decision_query = """
            SELECT
                d.doc_key,
                d.source_dataset,
                d.source_doc_id,
                d.strict_group_size,
                d.strict_kept_doc_key,
                d.strict_dropped,
                d.relaxed_group_size,
                d.relaxed_kept_doc_key,
                d.relaxed_dropped,
                d.exact_strict_hash AS strict_group_hash,
                d.exact_relaxed_hash AS relaxed_group_hash,
                NULL AS near_cluster_id,
                NULL AS near_kept_doc_key,
                NULL AS near_dropped
            FROM read_parquet(?) AS d
            ORDER BY d.file_path, d.row_index_in_file
        """
        decision_params = [str(docs_exact_path)]
    exact_relaxed_version_value = exact_relaxed_version(greek_diacritic_policy=greek_diacritic_policy)
    near_norm_version_value = near_norm_version(greek_diacritic_policy=greek_diacritic_policy)
    shingle_version_value = shingle_version(shingle_mode=shingle_mode, shingle_size=shingle_size)
    decisions_temp_path = temp_output_path(final_root / "dedup_decisions.parquet")
    kept_temp_path = temp_output_path(final_root / "kept_docs.parquet")
    dropped_temp_path = temp_output_path(final_root / "dropped_docs.parquet")
    decisions_writer: pq.ParquetWriter | None = None
    kept_writer: pq.ParquetWriter | None = None
    dropped_writer: pq.ParquetWriter | None = None
    decision_rows = 0
    kept_rows = 0
    dropped_rows = 0
    decision_batch: list[dict[str, Any]] = []
    kept_batch: list[dict[str, Any]] = []
    dropped_batch: list[dict[str, Any]] = []

    def flush_batches() -> None:
        nonlocal decisions_writer, kept_writer, dropped_writer
        nonlocal decision_rows, kept_rows, dropped_rows
        nonlocal decision_batch, kept_batch, dropped_batch
        if not decision_batch and not kept_batch and not dropped_batch:
            return
        decisions_writer, written_decisions = append_rows_to_parquet_writer(
            decisions_writer,
            rows=decision_batch,
            temp_path=decisions_temp_path,
            schema=FINAL_DECISION_SCHEMA,
        )
        kept_writer, written_kept = append_rows_to_parquet_writer(
            kept_writer,
            rows=kept_batch,
            temp_path=kept_temp_path,
            schema=FINAL_DECISION_SCHEMA,
        )
        dropped_writer, written_dropped = append_rows_to_parquet_writer(
            dropped_writer,
            rows=dropped_batch,
            temp_path=dropped_temp_path,
            schema=FINAL_DECISION_SCHEMA,
        )
        decision_rows += written_decisions
        kept_rows += written_kept
        dropped_rows += written_dropped
        decision_batch = []
        kept_batch = []
        dropped_batch = []

    try:
        for row in iter_duckdb_query_rows(decision_query, decision_params):
            if int(row["strict_dropped"] or 0) == 1:
                decision = {
                    "doc_key": str(row["doc_key"]),
                    "source_dataset": str(row["source_dataset"]),
                    "source_doc_id": str(row["source_doc_id"]),
                    "decision": "drop",
                    "decision_stage": STRICT_STAGE,
                    "cluster_id": f"strict:{row['strict_group_hash']}",
                    "kept_doc_key": final_kept_lookup[str(row["doc_key"])],
                    "reason": STRICT_STAGE,
                    "exact_strict_version": EXACT_STRICT_VERSION,
                    "exact_relaxed_version": exact_relaxed_version_value,
                    "near_norm_version": near_norm_version_value,
                    "tokenization_version": TOKENIZATION_VERSION,
                    "shingle_version": shingle_version_value,
                    "minhash_version": MINHASH_VERSION,
                    "lsh_version": LSH_VERSION,
                    "selection_version": SELECTION_VERSION,
                }
            elif int(row["relaxed_dropped"] or 0) == 1:
                decision = {
                    "doc_key": str(row["doc_key"]),
                    "source_dataset": str(row["source_dataset"]),
                    "source_doc_id": str(row["source_doc_id"]),
                    "decision": "drop",
                    "decision_stage": RELAXED_STAGE,
                    "cluster_id": f"relaxed:{row['relaxed_group_hash']}",
                    "kept_doc_key": final_kept_lookup[str(row["doc_key"])],
                    "reason": RELAXED_STAGE,
                    "exact_strict_version": EXACT_STRICT_VERSION,
                    "exact_relaxed_version": exact_relaxed_version_value,
                    "near_norm_version": near_norm_version_value,
                    "tokenization_version": TOKENIZATION_VERSION,
                    "shingle_version": shingle_version_value,
                    "minhash_version": MINHASH_VERSION,
                    "lsh_version": LSH_VERSION,
                    "selection_version": SELECTION_VERSION,
                }
            elif row["near_cluster_id"] is not None and int(row["near_dropped"] or 0) == 1:
                decision = {
                    "doc_key": str(row["doc_key"]),
                    "source_dataset": str(row["source_dataset"]),
                    "source_doc_id": str(row["source_doc_id"]),
                    "decision": "drop",
                    "decision_stage": "near_duplicate",
                    "cluster_id": str(row["near_cluster_id"]),
                    "kept_doc_key": final_kept_lookup[str(row["doc_key"])],
                    "reason": "near_duplicate",
                    "exact_strict_version": EXACT_STRICT_VERSION,
                    "exact_relaxed_version": exact_relaxed_version_value,
                    "near_norm_version": near_norm_version_value,
                    "tokenization_version": TOKENIZATION_VERSION,
                    "shingle_version": shingle_version_value,
                    "minhash_version": MINHASH_VERSION,
                    "lsh_version": LSH_VERSION,
                    "selection_version": SELECTION_VERSION,
                }
            else:
                decision = {
                    "doc_key": str(row["doc_key"]),
                    "source_dataset": str(row["source_dataset"]),
                    "source_doc_id": str(row["source_doc_id"]),
                    "decision": "keep",
                    "decision_stage": "kept_after_near" if row["near_cluster_id"] is not None else "kept_after_exact",
                    "cluster_id": None if row["near_cluster_id"] is None else str(row["near_cluster_id"]),
                    "kept_doc_key": final_kept_lookup[str(row["doc_key"])],
                    "reason": "kept",
                    "exact_strict_version": EXACT_STRICT_VERSION,
                    "exact_relaxed_version": exact_relaxed_version_value,
                    "near_norm_version": near_norm_version_value,
                    "tokenization_version": TOKENIZATION_VERSION,
                    "shingle_version": shingle_version_value,
                    "minhash_version": MINHASH_VERSION,
                    "lsh_version": LSH_VERSION,
                    "selection_version": SELECTION_VERSION,
                }
            decision_batch.append(decision)
            if decision["decision"] == "keep":
                kept_batch.append(decision)
            else:
                dropped_batch.append(decision)
            if len(decision_batch) >= 2048:
                flush_batches()
        flush_batches()
    finally:
        finalize_parquet_writer(
            decisions_writer,
            temp_path=decisions_temp_path,
            destination=final_root / "dedup_decisions.parquet",
            schema=FINAL_DECISION_SCHEMA,
        )
        finalize_parquet_writer(
            kept_writer,
            temp_path=kept_temp_path,
            destination=final_root / "kept_docs.parquet",
            schema=FINAL_DECISION_SCHEMA,
        )
        finalize_parquet_writer(
            dropped_writer,
            temp_path=dropped_temp_path,
            destination=final_root / "dropped_docs.parquet",
            schema=FINAL_DECISION_SCHEMA,
        )
    exact_cluster_summary_path = final_root / "_exact_cluster_summary.parquet"
    write_group_parquet(
        [
            *exact_cluster_summaries_from_groups(run_root / "stage_01_exact" / "strict_exact_groups.parquet", stage=STRICT_STAGE),
            *exact_cluster_summaries_from_groups(run_root / "stage_01_exact" / "relaxed_exact_groups.parquet", stage=RELAXED_STAGE),
        ],
        exact_cluster_summary_path,
        schema=CLUSTER_SUMMARY_SCHEMA,
    )
    combined_cluster_summary_inputs = [exact_cluster_summary_path]
    if near_cluster_summary_path is not None and near_cluster_summary_path.exists():
        combined_cluster_summary_inputs.append(near_cluster_summary_path)
    else:
        near_clusters_path = run_root / "stage_02_near" / "near_clusters.parquet"
        if near_clusters_path.exists():
            synthesized_near_summary_path = final_root / "_near_cluster_summary.parquet"
            write_group_parquet(
                near_cluster_summaries_from_parquet(near_clusters_path),
                synthesized_near_summary_path,
                schema=CLUSTER_SUMMARY_SCHEMA,
            )
            combined_cluster_summary_inputs.append(synthesized_near_summary_path)
    cluster_summary_rows = combine_parquet_files(
        combined_cluster_summary_inputs,
        final_root / "cluster_summary.parquet",
        schema=CLUSTER_SUMMARY_SCHEMA,
    )
    edge_limit = DEFAULT_EDGE_AUDIT_SAMPLE_SIZE // 3
    edge_threshold_state: list[tuple[str, dict[str, Any]]] = []
    edge_low_ratio_state: list[tuple[str, dict[str, Any]]] = []
    edge_containment_state: list[tuple[str, dict[str, Any]]] = []
    for batch_rows in iter_parquet_batches(run_root / "stage_02_near" / "candidate_pairs.parquet"):
        for row in batch_rows:
            if abs(float(row["estimated_jaccard"]) - minhash_threshold) <= 0.02:
                update_streaming_deterministic_sample(
                    edge_threshold_state,
                    row=row,
                    key_fields=["doc_key_left", "doc_key_right"],
                    limit=edge_limit,
                )
            if float(row["length_ratio"]) < 0.80:
                update_streaming_deterministic_sample(
                    edge_low_ratio_state,
                    row=row,
                    key_fields=["doc_key_left", "doc_key_right"],
                    limit=edge_limit,
                )
            if bool(row["likely_containment_flag"]):
                update_streaming_deterministic_sample(
                    edge_containment_state,
                    row=row,
                    key_fields=["doc_key_left", "doc_key_right"],
                    limit=edge_limit,
                )
    edge_samples: list[dict[str, Any]] = []
    edge_samples.extend({"sample_reason": "near_threshold", **row} for _, row in edge_threshold_state)
    edge_samples.extend({"sample_reason": "low_length_ratio", **row} for _, row in edge_low_ratio_state)
    edge_samples.extend({"sample_reason": "containment_hint", **row} for _, row in edge_containment_state)
    write_group_parquet(edge_samples, final_root / "edge_audit_sample.parquet", schema=EDGE_AUDIT_SAMPLE_SCHEMA)
    cluster_limit = DEFAULT_CLUSTER_AUDIT_SAMPLE_SIZE // 3
    cluster_largest_state: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    cluster_mixed_state: list[tuple[str, dict[str, Any]]] = []
    cluster_narrow_state: list[tuple[str, dict[str, Any]]] = []
    for batch_rows in iter_parquet_batches(final_root / "cluster_summary.parquet"):
        for row in batch_rows:
            update_streaming_ranked_top_rows(
                cluster_largest_state,
                row=row,
                rank=(-int(row["cluster_size"]), str(row["cluster_id"])),
                limit=cluster_limit,
            )
            if bool(row["mixed_source"]):
                update_streaming_deterministic_sample(
                    cluster_mixed_state,
                    row=row,
                    key_fields=["cluster_id"],
                    limit=cluster_limit,
                )
            if bool(row["narrow_margin_flag"]):
                update_streaming_deterministic_sample(
                    cluster_narrow_state,
                    row=row,
                    key_fields=["cluster_id"],
                    limit=cluster_limit,
                )
    cluster_audit_rows: list[dict[str, Any]] = []
    cluster_audit_rows.extend({"sample_reason": "largest_cluster", **row} for _, row in cluster_largest_state)
    cluster_audit_rows.extend({"sample_reason": "mixed_source", **row} for _, row in cluster_mixed_state)
    cluster_audit_rows.extend({"sample_reason": "narrow_margin", **row} for _, row in cluster_narrow_state)
    write_group_parquet(cluster_audit_rows, final_root / "cluster_audit_sample.parquet", schema=CLUSTER_AUDIT_SAMPLE_SCHEMA)
    summary = {
        "run_id": run_id,
        "decision_rows": int(decision_rows),
        "kept_rows": int(kept_rows),
        "dropped_rows": int(dropped_rows),
        "cluster_summary_rows": int(cluster_summary_rows),
        "edge_audit_sample_rows": int(len(edge_samples)),
        "cluster_audit_sample_rows": int(len(cluster_audit_rows)),
        "decision_path": str(final_root / "dedup_decisions.parquet"),
        "kept_path": str(final_root / "kept_docs.parquet"),
        "dropped_path": str(final_root / "dropped_docs.parquet"),
        "cluster_summary_path": str(final_root / "cluster_summary.parquet"),
        "edge_audit_sample_path": str(final_root / "edge_audit_sample.parquet"),
        "cluster_audit_sample_path": str(final_root / "cluster_audit_sample.parquet"),
    }
    write_json_atomic(final_root / "run_summary.json", summary)
    upsert_stage_progress(
        conn,
        run_id=run_id,
        stage=FINAL_EXPORT_STAGE,
        status="completed",
        total_chunks=1,
        completed_chunks=1,
        progress_path=progress_path,
        payload=summary,
    )
    return summary


def builder_metadata_root(run_root: Path) -> Path:
    return run_root / "builder_metadata"


def _group_mixed_source_map(path: Path) -> dict[str, bool]:
    mixed_source: dict[str, set[str]] = defaultdict(set)
    if not path.exists():
        return {}
    for batch_rows in iter_parquet_batches(path, columns=["group_hash", "member_source_dataset"]):
        for row in batch_rows:
            mixed_source[str(row["group_hash"])].add(str(row["member_source_dataset"]))
    return {group_hash: len(source_sets) > 1 for group_hash, source_sets in mixed_source.items()}


def _stream_builder_exact_memberships(
    *,
    source_path: Path,
    stage: str,
    mixed_source_map: dict[str, bool],
    destination: Path,
) -> int:
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    temp_path = temp_output_path(destination)
    try:
        for batch_rows in iter_parquet_batches(source_path):
            payload = [
                {
                    "doc_key": str(row["member_doc_key"]),
                    "source_dataset": str(row["member_source_dataset"]),
                    "source_doc_id": str(row["member_source_doc_id"]),
                    "stage": stage,
                    "group_hash": str(row["group_hash"]),
                    "group_size": int(row["group_size"]),
                    "mixed_source": bool(mixed_source_map.get(str(row["group_hash"]), False)),
                    "kept_doc_key": str(row["kept_doc_key"]),
                }
                for row in batch_rows
            ]
            writer, written = append_rows_to_parquet_writer(
                writer,
                rows=payload,
                temp_path=temp_path,
                schema=BUILDER_EXACT_MEMBERSHIP_SCHEMA,
            )
            rows_written += written
    finally:
        finalize_parquet_writer(
            writer,
            temp_path=temp_path,
            destination=destination,
            schema=BUILDER_EXACT_MEMBERSHIP_SCHEMA,
    )
    return rows_written


def _export_builder_family_membership(
    *,
    final_decisions_path: Path,
    destination: Path,
    run_id: str,
) -> int:
    family_sizes: dict[str, int] = defaultdict(int)
    family_sources: dict[str, set[str]] = defaultdict(set)
    if not final_decisions_path.exists():
        raise FileNotFoundError(f"final decisions missing under {final_decisions_path}")
    for batch_rows in iter_parquet_batches(
        final_decisions_path,
        columns=["kept_doc_key", "source_dataset"],
    ):
        for row in batch_rows:
            kept_doc_key = str(row["kept_doc_key"])
            family_sizes[kept_doc_key] += 1
            family_sources[kept_doc_key].add(str(row["source_dataset"]))
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    temp_path = temp_output_path(destination)
    try:
        for batch_rows in iter_parquet_batches(
            final_decisions_path,
            columns=["doc_key", "source_dataset", "source_doc_id", "decision", "decision_stage", "kept_doc_key"],
        ):
            payload = []
            for row in batch_rows:
                kept_doc_key = str(row["kept_doc_key"])
                source_sets = family_sources[kept_doc_key]
                payload.append(
                    {
                        "doc_key": str(row["doc_key"]),
                        "source_dataset": str(row["source_dataset"]),
                        "source_doc_id": str(row["source_doc_id"]),
                        "family_id": stable_doc_key("family", kept_doc_key),
                        "family_size": int(family_sizes[kept_doc_key]),
                        "family_source_count": int(len(source_sets)),
                        "family_mixed_source": bool(len(source_sets) > 1),
                        "canonical_kept_doc_key": kept_doc_key,
                        "canonical_decision": str(row["decision"]),
                        "canonical_decision_stage": str(row["decision_stage"]),
                        "dedup_run_id": run_id,
                        "selection_version": SELECTION_VERSION,
                        "representative_score_version": REPRESENTATIVE_SCORE_VERSION,
                    }
                )
            writer, written = append_rows_to_parquet_writer(
                writer,
                rows=payload,
                temp_path=temp_path,
                schema=BUILDER_FAMILY_MEMBERSHIP_SCHEMA,
            )
            rows_written += written
    finally:
        finalize_parquet_writer(
            writer,
            temp_path=temp_path,
            destination=destination,
            schema=BUILDER_FAMILY_MEMBERSHIP_SCHEMA,
        )
    return rows_written


def export_builder_metadata_bundle(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    run_root: Path,
    greek_diacritic_policy: str,
    minhash_threshold: float,
    shingle_mode: str,
    shingle_size: int,
    num_perm: int,
    bands: int,
    rows_per_band: int,
) -> dict[str, Any]:
    bundle_root = builder_metadata_root(run_root)
    ensure_dir(bundle_root)
    manifest_path = bundle_root / "manifest.json"
    existing_manifest = read_json_if_exists(manifest_path)
    if existing_manifest is not None:
        files_payload = dict(existing_manifest.get("files", {}))
        # near_candidate_pairs.parquet remains part of the required exported bundle even
        # though the modern builder path prefers family membership, because we still want
        # the raw near-pair evidence available for audit/debugging.
        required_paths = [
            bundle_root / "doc_dedup_metadata.parquet",
            bundle_root / "strict_exact_memberships.parquet",
            bundle_root / "relaxed_exact_memberships.parquet",
            bundle_root / "exact_group_memberships.parquet",
            bundle_root / "dedup_family_membership.parquet",
            bundle_root / "near_candidate_pairs.parquet",
            bundle_root.parent / str(files_payload.get("run_summary", "final/run_summary.json")),
        ]
        if all_paths_exist(required_paths):
            return {
                "bundle_root": str(bundle_root),
                "manifest_path": str(manifest_path),
                "doc_metadata_path": str(bundle_root / "doc_dedup_metadata.parquet"),
                "strict_exact_membership_rows": parquet_num_rows(bundle_root / "strict_exact_memberships.parquet"),
                "relaxed_exact_membership_rows": parquet_num_rows(bundle_root / "relaxed_exact_memberships.parquet"),
                "exact_membership_rows": parquet_num_rows(bundle_root / "exact_group_memberships.parquet"),
                "doc_metadata_rows": parquet_num_rows(bundle_root / "doc_dedup_metadata.parquet"),
                "family_membership_rows": parquet_num_rows(bundle_root / "dedup_family_membership.parquet"),
                "near_candidate_pair_rows": parquet_num_rows(bundle_root / "near_candidate_pairs.parquet"),
            }
    strict_groups_path = run_root / "stage_01_exact" / "strict_exact_groups.parquet"
    relaxed_groups_path = run_root / "stage_01_exact" / "relaxed_exact_groups.parquet"
    candidate_pairs_path = run_root / "stage_02_near" / "candidate_pairs.parquet"
    strict_mixed_source_map = _group_mixed_source_map(strict_groups_path)
    relaxed_mixed_source_map = _group_mixed_source_map(relaxed_groups_path)
    strict_exact_membership_path = bundle_root / "strict_exact_memberships.parquet"
    relaxed_exact_membership_path = bundle_root / "relaxed_exact_memberships.parquet"
    exact_memberships_path = bundle_root / "exact_group_memberships.parquet"
    strict_membership_rows = _stream_builder_exact_memberships(
        source_path=strict_groups_path,
        stage=STRICT_STAGE,
        mixed_source_map=strict_mixed_source_map,
        destination=strict_exact_membership_path,
    )
    relaxed_membership_rows = _stream_builder_exact_memberships(
        source_path=relaxed_groups_path,
        stage=RELAXED_STAGE,
        mixed_source_map=relaxed_mixed_source_map,
        destination=relaxed_exact_membership_path,
    )
    exact_membership_rows = combine_parquet_files(
        [strict_exact_membership_path, relaxed_exact_membership_path],
        exact_memberships_path,
        schema=BUILDER_EXACT_MEMBERSHIP_SCHEMA,
    )

    doc_lookup = {
        str(row["doc_key"]): {
            "source_dataset": str(row["source_dataset"]),
            "source_doc_id": str(row["source_doc_id"]),
        }
        for row in conn.execute(
            "SELECT doc_key, source_dataset, source_doc_id FROM run_docs WHERE run_id = ?",
            (run_id,),
        )
    }
    near_stats: dict[str, dict[str, Any]] = {}
    near_pairs_temp_path = temp_output_path(bundle_root / "near_candidate_pairs.parquet")
    near_pairs_writer: pq.ParquetWriter | None = None
    near_pair_rows = 0
    try:
        if candidate_pairs_path.exists():
            for batch_rows in iter_parquet_batches(candidate_pairs_path):
                enriched_rows: list[dict[str, Any]] = []
                for row in batch_rows:
                    left_key = str(row["doc_key_left"])
                    right_key = str(row["doc_key_right"])
                    left_lookup = doc_lookup[left_key]
                    right_lookup = doc_lookup[right_key]
                    estimated_jaccard = float(row["estimated_jaccard"])
                    length_ratio = float(row["length_ratio"])
                    enriched_rows.append(
                        {
                            "doc_key_left": left_key,
                            "source_dataset_left": left_lookup["source_dataset"],
                            "source_doc_id_left": left_lookup["source_doc_id"],
                            "doc_key_right": right_key,
                            "source_dataset_right": right_lookup["source_dataset"],
                            "source_doc_id_right": right_lookup["source_doc_id"],
                            "estimated_jaccard": estimated_jaccard,
                            "length_ratio": length_ratio,
                            "likely_containment_flag": bool(row["likely_containment_flag"]),
                            "accepted_reason": str(row["accepted_reason"]),
                            "bucket_match_bands": int(row["bucket_match_bands"]),
                            "shingle_mode": str(row["shingle_mode"]),
                            "shingle_size": int(shingle_size),
                            "num_perm": int(num_perm),
                            "bands": int(bands),
                            "rows_per_band": int(rows_per_band),
                            "candidate_score_floor": float(minhash_threshold),
                        }
                    )
                    for doc_key, other_key, doc_dataset, other_dataset in (
                        (left_key, right_key, left_lookup["source_dataset"], right_lookup["source_dataset"]),
                        (right_key, left_key, right_lookup["source_dataset"], left_lookup["source_dataset"]),
                    ):
                        stats = near_stats.setdefault(
                            doc_key,
                            {
                                "near_candidate_count": 0,
                                "near_best_match_doc_key": None,
                                "near_best_match_source_dataset": None,
                                "near_best_estimated_jaccard": None,
                                "near_best_length_ratio": None,
                                "near_cross_dataset_candidate_count": 0,
                                "near_same_dataset_candidate_count": 0,
                            },
                        )
                        stats["near_candidate_count"] += 1
                        if doc_dataset == other_dataset:
                            stats["near_same_dataset_candidate_count"] += 1
                        else:
                            stats["near_cross_dataset_candidate_count"] += 1
                        best_score = stats["near_best_estimated_jaccard"]
                        best_other = stats["near_best_match_doc_key"]
                        if (
                            best_score is None
                            or estimated_jaccard > float(best_score)
                            or (
                                estimated_jaccard == float(best_score)
                                and (
                                    best_other is None
                                    or other_key < str(best_other)
                                )
                            )
                        ):
                            stats["near_best_match_doc_key"] = other_key
                            stats["near_best_match_source_dataset"] = other_dataset
                            stats["near_best_estimated_jaccard"] = estimated_jaccard
                            stats["near_best_length_ratio"] = length_ratio
                near_pairs_writer, written_pairs = append_rows_to_parquet_writer(
                    near_pairs_writer,
                    rows=enriched_rows,
                    temp_path=near_pairs_temp_path,
                    schema=BUILDER_NEAR_PAIR_SCHEMA,
                )
                near_pair_rows += written_pairs
    finally:
        finalize_parquet_writer(
            near_pairs_writer,
            temp_path=near_pairs_temp_path,
            destination=bundle_root / "near_candidate_pairs.parquet",
            schema=BUILDER_NEAR_PAIR_SCHEMA,
        )

    exact_relaxed_version_value = exact_relaxed_version(greek_diacritic_policy=greek_diacritic_policy)
    near_norm_version_value = near_norm_version(greek_diacritic_policy=greek_diacritic_policy)
    shingle_version_value = shingle_version(shingle_mode=shingle_mode, shingle_size=shingle_size)
    docs_exact_path = run_root / "stage_01_exact" / "docs_exact.parquet"
    doc_metadata_path = bundle_root / "doc_dedup_metadata.parquet"
    doc_writer: pq.ParquetWriter | None = None
    doc_rows = 0
    doc_temp_path = temp_output_path(doc_metadata_path)
    try:
        for batch in iter_parquet_batches(
            docs_exact_path,
            columns=[
                "doc_key",
                "source_dataset",
                "source_doc_id",
                "title",
                "author",
                "raw_text_chars",
                "greek_badness_score",
                "len_greek",
                "mojibake_badness_score",
                "needs_ocr",
                "ocr_success",
                "exact_strict_hash",
                "strict_group_size",
                "strict_kept_doc_key",
                "exact_relaxed_hash",
                "relaxed_group_size",
                "relaxed_kept_doc_key",
            ],
        ):
            payload: list[dict[str, Any]] = []
            for row in batch:
                doc_key = str(row["doc_key"])
                stats = near_stats.get(doc_key, {})
                strict_group_hash = str(row["exact_strict_hash"])
                relaxed_group_hash = None if row["exact_relaxed_hash"] is None else str(row["exact_relaxed_hash"])
                payload.append(
                    {
                        "doc_key": doc_key,
                        "source_dataset": str(row["source_dataset"]),
                        "source_doc_id": str(row["source_doc_id"]),
                        "strict_exact_group_hash": strict_group_hash,
                        "strict_exact_group_size": int(row["strict_group_size"]),
                        "strict_exact_mixed_source": bool(strict_mixed_source_map.get(strict_group_hash, False)),
                        "strict_exact_kept_doc_key": str(row["strict_kept_doc_key"]),
                        "relaxed_exact_group_hash": relaxed_group_hash,
                        "relaxed_exact_group_size": 0 if row["relaxed_group_size"] is None else int(row["relaxed_group_size"]),
                        "relaxed_exact_mixed_source": bool(
                            relaxed_mixed_source_map.get(relaxed_group_hash, False)
                        )
                        if relaxed_group_hash is not None
                        else False,
                        "relaxed_exact_kept_doc_key": None
                        if row["relaxed_kept_doc_key"] is None
                        else str(row["relaxed_kept_doc_key"]),
                        "near_candidate_count": int(stats.get("near_candidate_count", 0)),
                        "near_best_match_doc_key": stats.get("near_best_match_doc_key"),
                        "near_best_match_source_dataset": stats.get("near_best_match_source_dataset"),
                        "near_best_estimated_jaccard": stats.get("near_best_estimated_jaccard"),
                        "near_best_length_ratio": stats.get("near_best_length_ratio"),
                        "near_cross_dataset_candidate_count": int(stats.get("near_cross_dataset_candidate_count", 0)),
                        "near_same_dataset_candidate_count": int(stats.get("near_same_dataset_candidate_count", 0)),
                        "needs_ocr": optional_bool_int(row["needs_ocr"]),
                        "greek_badness_score": optional_float(row["greek_badness_score"]),
                        "len_greek": optional_int(row["len_greek"]),
                        "mojibake_badness_score": optional_float(row["mojibake_badness_score"]),
                        "ocr_success": optional_bool_int(row["ocr_success"]),
                        "text_length_for_selection": int(row["raw_text_chars"] or 0),
                        "representative_score": float(
                            representative_score(
                                selection_length_value(dict(row), text_length_field="raw_text_chars"),
                                optional_float(row["greek_badness_score"]),
                            )
                        ),
                        "representative_score_version": REPRESENTATIVE_SCORE_VERSION,
                        "has_title": bool(row["title"]),
                        "has_author": bool(row["author"]),
                        "dedup_run_id": run_id,
                        "greek_diacritic_policy": greek_diacritic_policy,
                        "exact_strict_version": EXACT_STRICT_VERSION,
                        "exact_relaxed_version": exact_relaxed_version_value,
                        "near_norm_version": near_norm_version_value,
                        "tokenization_version": TOKENIZATION_VERSION,
                        "shingle_version": shingle_version_value,
                        "minhash_version": MINHASH_VERSION,
                        "lsh_version": LSH_VERSION,
                        "selection_version": SELECTION_VERSION,
                    }
                )
            doc_writer, written_doc_rows = append_rows_to_parquet_writer(
                doc_writer,
                rows=payload,
                temp_path=doc_temp_path,
                schema=BUILDER_DOC_METADATA_SCHEMA,
            )
            doc_rows += written_doc_rows
    finally:
        finalize_parquet_writer(
            doc_writer,
            temp_path=doc_temp_path,
            destination=doc_metadata_path,
            schema=BUILDER_DOC_METADATA_SCHEMA,
        )

    family_membership_path = bundle_root / "dedup_family_membership.parquet"
    family_membership_rows = _export_builder_family_membership(
        final_decisions_path=run_root / "final" / "dedup_decisions.parquet",
        destination=family_membership_path,
        run_id=run_id,
    )

    manifest = {
        "builder_metadata_version": BUILDER_METADATA_VERSION,
        "run_id": run_id,
        "run_root": str(run_root),
        "candidate_score_floor": float(minhash_threshold),
        "builder_default_threshold": float(minhash_threshold),
        "builder_exact_stage": "strict_and_relaxed",
        "near_cluster_mode": "representative_validation",
        "greek_diacritic_policy": greek_diacritic_policy,
        "exact_strict_version": EXACT_STRICT_VERSION,
        "exact_relaxed_version": exact_relaxed_version_value,
        "near_norm_version": near_norm_version_value,
        "tokenization_version": TOKENIZATION_VERSION,
        "shingle_version": shingle_version_value,
        "minhash_version": MINHASH_VERSION,
        "lsh_version": LSH_VERSION,
        "selection_version": SELECTION_VERSION,
        "representative_score_version": REPRESENTATIVE_SCORE_VERSION,
        "shingle_mode": shingle_mode,
        "shingle_size": int(shingle_size),
        "num_perm": int(num_perm),
        "bands": int(bands),
        "rows_per_band": int(rows_per_band),
        "files": {
            "doc_metadata": "doc_dedup_metadata.parquet",
            "strict_exact_memberships": "strict_exact_memberships.parquet",
            "relaxed_exact_memberships": "relaxed_exact_memberships.parquet",
            "exact_group_memberships": "exact_group_memberships.parquet",
            "family_membership": "dedup_family_membership.parquet",
            "near_candidate_pairs": "near_candidate_pairs.parquet",
            "run_summary": str((run_root / "final" / "run_summary.json").relative_to(bundle_root.parent)),
        },
    }
    write_json_atomic(bundle_root / "manifest.json", manifest)
    return {
        "bundle_root": str(bundle_root),
        "manifest_path": str(bundle_root / "manifest.json"),
        "doc_metadata_path": str(doc_metadata_path),
        "strict_exact_membership_rows": int(strict_membership_rows),
        "relaxed_exact_membership_rows": int(relaxed_membership_rows),
        "exact_membership_rows": int(exact_membership_rows),
        "doc_metadata_rows": int(doc_rows),
        "family_membership_rows": int(family_membership_rows),
        "near_candidate_pair_rows": int(near_pair_rows),
    }


def near_cluster_summaries_from_parquet(path: Path) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in iter_parquet_rows(path):
        cluster_id = str(row["cluster_id"])
        entry = grouped.setdefault(
            cluster_id,
            {
                "cluster_id": cluster_id,
                "decision_stage": "near_duplicate",
                "cluster_size": int(row["cluster_size"]),
                "dropped_count": 0,
                "kept_doc_key": str(row["kept_doc_key"]),
                "source_datasets": set(),
                "large_component_audit_flag": bool(row["large_component_audit_flag"]),
                "narrow_margin_flag": False,
            },
        )
        entry["dropped_count"] += int(bool(row["dropped"]))
        entry["source_datasets"].add(str(row["member_source_dataset"]))
    return [
        {
            "cluster_id": cluster_id,
            "decision_stage": str(payload["decision_stage"]),
            "cluster_size": int(payload["cluster_size"]),
            "dropped_count": int(payload["dropped_count"]),
            "kept_doc_key": str(payload["kept_doc_key"]),
            "mixed_source": bool(len(payload["source_datasets"]) > 1),
            "large_component_audit_flag": bool(payload["large_component_audit_flag"]),
            "narrow_margin_flag": bool(payload["narrow_margin_flag"]),
        }
        for cluster_id, payload in sorted(grouped.items())
    ]

def run_status(conn: sqlite3.Connection, *, run_id: str) -> dict[str, Any]:
    run_row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if run_row is None:
        raise ValueError(f"unknown run_id: {run_id}")
    chunk_row = conn.execute(
        """
        SELECT
            COUNT(*) AS total_chunks,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed_chunks,
            SUM(row_group_rows) AS total_rows,
            SUM(CASE WHEN status = 'completed' THEN row_group_rows ELSE 0 END) AS completed_rows
        FROM run_chunks
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()
    docs_row = conn.execute(
        """
        SELECT
            COUNT(*) AS total_docs,
            SUM(reused_exact) AS reused_exact_rows
        FROM run_docs
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()
    near_row = conn.execute(
        """
        SELECT
            COUNT(*) AS near_rows,
            SUM(dropped) AS near_dropped_rows
        FROM run_near_results
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()
    stage_rows = [
        {
            "stage": str(row["stage"]),
            "status": str(row["status"]),
            "total_chunks": int(row["total_chunks"]),
            "completed_chunks": int(row["completed_chunks"]),
            "progress_path": str(row["progress_path"]) if row["progress_path"] is not None else None,
            "updated_at": str(row["updated_at"]),
        }
        for row in conn.execute(
            "SELECT stage, status, total_chunks, completed_chunks, progress_path, updated_at FROM stage_progress WHERE run_id = ? ORDER BY stage",
            (run_id,),
        )
    ]
    return {
        "run_id": run_id,
        "status": str(run_row["status"]),
        "input_root": str(run_row["input_root"]),
        "run_root": str(run_row["run_root"]),
        "created_at": str(run_row["created_at"]),
        "updated_at": str(run_row["updated_at"]),
        "completed_at": None if run_row["completed_at"] is None else str(run_row["completed_at"]),
        "total_chunks": int(chunk_row["total_chunks"] or 0),
        "completed_chunks": int(chunk_row["completed_chunks"] or 0),
        "total_rows": int(chunk_row["total_rows"] or 0),
        "completed_rows": int(chunk_row["completed_rows"] or 0),
        "materialized_docs": int(docs_row["total_docs"] or 0),
        "reused_exact_rows": int(docs_row["reused_exact_rows"] or 0),
        "near_materialized_docs": int(near_row["near_rows"] or 0),
        "near_dropped_rows": int(near_row["near_dropped_rows"] or 0),
        "stage_progress": stage_rows,
    }


def latest_run_id(state_root: Path) -> str | None:
    latest_path = state_root / "latest_success.json"
    if not latest_path.exists():
        return None
    payload = json.loads(latest_path.read_text())
    run_id = payload.get("run_id")
    return None if not run_id else str(run_id)


def run_exact_dedup(
    *,
    input_root: Path = DEFAULT_INPUT_ROOT,
    state_root: Path = DEFAULT_STATE_ROOT,
    run_root: Path | None = None,
    resume: bool = False,
    max_workers: int = DEFAULT_RUN_MAX_WORKERS,
    greek_diacritic_policy: str = DEFAULT_GREEK_DIACRITIC_POLICY,
) -> dict[str, Any]:
    result = run_dedup_pipeline(
        input_root=input_root,
        state_root=state_root,
        run_root=run_root,
        resume=resume,
        max_workers=max_workers,
        greek_diacritic_policy=greek_diacritic_policy,
        exact_only=True,
    )
    return dict(result["exact"])


def run_dedup_pipeline(
    *,
    input_root: Path = DEFAULT_INPUT_ROOT,
    state_root: Path = DEFAULT_STATE_ROOT,
    run_root: Path | None = None,
    resume: bool = False,
    max_workers: int = DEFAULT_RUN_MAX_WORKERS,
    greek_diacritic_policy: str = DEFAULT_GREEK_DIACRITIC_POLICY,
    exact_only: bool = False,
    minhash_threshold: float = DEFAULT_NEAR_THRESHOLD,
    num_perm: int = DEFAULT_NUM_PERM,
    bands: int = DEFAULT_BANDS,
    rows_per_band: int = DEFAULT_ROWS_PER_BAND,
    shingle_mode: str = DEFAULT_SHINGLE_MODE,
    shingle_size: int = DEFAULT_SHINGLE_SIZE,
    large_component_threshold: int = DEFAULT_LARGE_COMPONENT_THRESHOLD,
    max_bucket_size: int = DEFAULT_MAX_BUCKET_SIZE,
) -> dict[str, Any]:
    input_root = input_root.resolve()
    state_root = state_root.resolve()
    run_root = (run_root or default_run_root()).resolve()
    ensure_dir(run_root)
    greek_diacritic_policy = validate_greek_diacritic_policy(greek_diacritic_policy)
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    if max_bucket_size < 2:
        raise ValueError("max_bucket_size must be >= 2")
    if bands <= 0 or rows_per_band <= 0 or num_perm <= 0:
        raise ValueError("num_perm, bands, and rows_per_band must be positive")
    if bands * rows_per_band != num_perm:
        raise ValueError("bands * rows_per_band must equal num_perm")
    if shingle_mode not in {"token", "char"}:
        raise ValueError("shingle_mode must be one of: token, char")
    if shingle_size < 2:
        raise ValueError("shingle_size must be >= 2")
    config = pipeline_config_payload(
        input_root=input_root,
        state_root=state_root,
        run_root=run_root,
        greek_diacritic_policy=greek_diacritic_policy,
        exact_only=exact_only,
        max_workers=max_workers,
        minhash_threshold=minhash_threshold,
        num_perm=num_perm,
        bands=bands,
        rows_per_band=rows_per_band,
        shingle_mode=shingle_mode,
        shingle_size=shingle_size,
        large_component_threshold=large_component_threshold,
        max_bucket_size=max_bucket_size,
    )
    run_id, digest = ensure_run_config(run_root=run_root, config=config, resume=resume)
    files = discover_input_files(input_root)
    if not files:
        raise ValueError(f"no parquet files found under {input_root}")
    validate_input_files(files)
    conn = connect_db(state_root)
    try:
        init_run(conn, run_id=run_id, config=config, config_digest=digest)
        exact_summary = _run_exact_stage_core(
            conn,
            run_id=run_id,
            input_root=input_root,
            state_root=state_root,
            run_root=run_root,
            files=files,
            greek_diacritic_policy=greek_diacritic_policy,
            max_workers=max_workers,
            emit_survivor_export=not exact_only,
            resume=resume,
        )
        if exact_only:
            completed_at = now_utc_iso()
            with conn:
                conn.execute(
                    "UPDATE runs SET status = ?, updated_at = ?, completed_at = ? WHERE run_id = ?",
                    ("completed", completed_at, completed_at, run_id),
                )
            write_json_atomic(
                state_root / "latest_success.json",
                {
                    "run_id": run_id,
                    "completed_at": completed_at,
                    "run_root": str(run_root),
                    "summary_path": str(run_root / "stage_01_exact" / "summary.json"),
                },
            )
            return {"run_id": run_id, "exact": exact_summary, "exact_only": True}
        near_signature_summary = _run_near_signature_stage(
            conn,
            run_id=run_id,
            run_root=run_root,
            survivor_manifest_path=run_root / "stage_01_exact" / "exact_survivor_manifest.parquet",
            state_root=state_root,
            config=config,
            greek_diacritic_policy=greek_diacritic_policy,
            max_workers=max_workers,
            num_perm=num_perm,
            bands=bands,
            rows_per_band=rows_per_band,
            shingle_mode=shingle_mode,
            shingle_size=shingle_size,
        )
        near_candidate_summary = _run_near_candidate_stage(
            conn,
            run_id=run_id,
            run_root=run_root,
            state_root=state_root,
            config=config,
            minhash_threshold=minhash_threshold,
            bands=bands,
            rows_per_band=rows_per_band,
            max_workers=max_workers,
            max_bucket_size=max_bucket_size,
        )
        near_cluster_summary = _run_near_cluster_stage(
            conn,
            run_id=run_id,
            run_root=run_root,
            state_root=state_root,
            config=config,
            minhash_threshold=minhash_threshold,
            large_component_threshold=large_component_threshold,
        )
        near_summary = {
            "run_id": run_id,
            "signatures": near_signature_summary,
            "candidates": near_candidate_summary,
            "clusters": near_cluster_summary,
            "summary_path": str(run_root / "stage_02_near" / "summary.json"),
        }
        write_json_atomic(run_root / "stage_02_near" / "summary.json", near_summary)
        final_summary = _build_final_exports(
            conn,
            run_id=run_id,
            run_root=run_root,
            greek_diacritic_policy=greek_diacritic_policy,
            minhash_threshold=minhash_threshold,
            shingle_mode=shingle_mode,
            shingle_size=shingle_size,
            near_cluster_summary_path=run_root / "stage_02_near" / "cluster_summary.parquet",
        )
        builder_metadata_summary = export_builder_metadata_bundle(
            conn,
            run_id=run_id,
            run_root=run_root,
            greek_diacritic_policy=greek_diacritic_policy,
            minhash_threshold=minhash_threshold,
            shingle_mode=shingle_mode,
            shingle_size=shingle_size,
            num_perm=num_perm,
            bands=bands,
            rows_per_band=rows_per_band,
        )
        completed_at = now_utc_iso()
        with conn:
            conn.execute(
                "UPDATE runs SET status = ?, updated_at = ?, completed_at = ? WHERE run_id = ?",
                ("completed", completed_at, completed_at, run_id),
            )
        write_json_atomic(
            state_root / "latest_success.json",
            {
                "run_id": run_id,
                "completed_at": completed_at,
                "run_root": str(run_root),
                "summary_path": str(run_root / "final" / "run_summary.json"),
            },
        )
        return {
            "run_id": run_id,
            "exact": exact_summary,
            "near": near_summary,
            "final": final_summary,
            "builder_metadata": builder_metadata_summary,
        }
    finally:
        conn.close()


def export_dedup_run(*, state_root: Path = DEFAULT_STATE_ROOT, run_root: Path | None = None) -> dict[str, Any]:
    state_root = state_root.resolve()
    conn = connect_db(state_root)
    try:
        run_id = run_root.name if run_root is not None else latest_run_id(state_root)
        if run_id is None:
            raise ValueError(f"no completed run found under {state_root}")
        run_row = conn.execute("SELECT run_root FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if run_row is None:
            raise ValueError(f"unknown run_id: {run_id}")
        resolved_run_root = Path(str(run_row["run_root"]))
        config_path = resolved_run_root / "run_config.json"
        if not config_path.exists():
            raise ValueError(f"run_config.json is missing under {resolved_run_root}")
        config_payload = json.loads(config_path.read_text()).get("config", {})
        shingle_mode = str(config_payload.get("shingle_mode", DEFAULT_SHINGLE_MODE))
        shingle_size = int(config_payload.get("shingle_size", DEFAULT_SHINGLE_SIZE))
        minhash_threshold = float(config_payload.get("minhash_threshold", DEFAULT_NEAR_THRESHOLD))
        num_perm = int(config_payload.get("num_perm", DEFAULT_NUM_PERM))
        bands = int(config_payload.get("bands", DEFAULT_BANDS))
        rows_per_band = int(config_payload.get("rows_per_band", DEFAULT_ROWS_PER_BAND))
        greek_diacritic_policy = validate_greek_diacritic_policy(
            str(config_payload.get("greek_diacritic_policy", DEFAULT_GREEK_DIACRITIC_POLICY))
        )
        near_clusters_path = resolved_run_root / "stage_02_near" / "near_clusters.parquet"
        if not near_clusters_path.exists():
            raise ValueError(f"near cluster artifacts are missing under {resolved_run_root / 'stage_02_near'}")
        final_summary = _build_final_exports(
            conn,
            run_id=run_id,
            run_root=resolved_run_root,
            greek_diacritic_policy=greek_diacritic_policy,
            minhash_threshold=minhash_threshold,
            shingle_mode=shingle_mode,
            shingle_size=shingle_size,
            near_cluster_summary_path=resolved_run_root / "stage_02_near" / "cluster_summary.parquet",
        )
        builder_metadata_summary = export_builder_metadata_bundle(
            conn,
            run_id=run_id,
            run_root=resolved_run_root,
            greek_diacritic_policy=greek_diacritic_policy,
            minhash_threshold=minhash_threshold,
            shingle_mode=shingle_mode,
            shingle_size=shingle_size,
            num_perm=num_perm,
            bands=bands,
            rows_per_band=rows_per_band,
        )
        return {"final": final_summary, "builder_metadata": builder_metadata_summary}
    finally:
        conn.close()


def export_builder_metadata_run(*, state_root: Path = DEFAULT_STATE_ROOT, run_root: Path | None = None) -> dict[str, Any]:
    state_root = state_root.resolve()
    conn = connect_db(state_root)
    try:
        run_id = run_root.name if run_root is not None else latest_run_id(state_root)
        if run_id is None:
            raise ValueError(f"no completed run found under {state_root}")
        run_row = conn.execute("SELECT run_root FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if run_row is None:
            raise ValueError(f"unknown run_id: {run_id}")
        resolved_run_root = Path(str(run_row["run_root"]))
        config_path = resolved_run_root / "run_config.json"
        if not config_path.exists():
            raise ValueError(f"run_config.json is missing under {resolved_run_root}")
        config_payload = json.loads(config_path.read_text()).get("config", {})
        return export_builder_metadata_bundle(
            conn,
            run_id=run_id,
            run_root=resolved_run_root,
            greek_diacritic_policy=validate_greek_diacritic_policy(
                str(config_payload.get("greek_diacritic_policy", DEFAULT_GREEK_DIACRITIC_POLICY))
            ),
            minhash_threshold=float(config_payload.get("minhash_threshold", DEFAULT_NEAR_THRESHOLD)),
            shingle_mode=str(config_payload.get("shingle_mode", DEFAULT_SHINGLE_MODE)),
            shingle_size=int(config_payload.get("shingle_size", DEFAULT_SHINGLE_SIZE)),
            num_perm=int(config_payload.get("num_perm", DEFAULT_NUM_PERM)),
            bands=int(config_payload.get("bands", DEFAULT_BANDS)),
            rows_per_band=int(config_payload.get("rows_per_band", DEFAULT_ROWS_PER_BAND)),
        )
    finally:
        conn.close()


def dedup_status(*, state_root: Path = DEFAULT_STATE_ROOT, run_root: Path | None = None) -> dict[str, Any]:
    state_root = state_root.resolve()
    conn = connect_db(state_root)
    try:
        run_id = run_root.name if run_root is not None else latest_run_id(state_root)
        if run_id is None:
            raise ValueError(f"no completed run found under {state_root}")
        return run_status(conn, run_id=run_id)
    finally:
        conn.close()
