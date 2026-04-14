#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from blake3 import blake3
import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _resolve_pipeline_import_root(script_path: Path) -> Path:
    for parent in script_path.parents:
        if (parent / "glossapi_corpus_cli").exists():
            return parent
    for parent in script_path.parents:
        latest_pointer = parent / "dedup_metadata" / "latest.json"
        if not latest_pointer.exists():
            continue
        try:
            payload = json.loads(latest_pointer.read_text(encoding="utf-8"))
        except Exception:
            continue
        code_root = payload.get("code_root")
        if not code_root:
            continue
        candidate = (parent / str(code_root)).resolve()
        if (candidate / "glossapi_corpus_cli").exists():
            return candidate
    return script_path.parents[1]


REPO_ROOT = _resolve_pipeline_import_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glossapi_corpus_cli import pipeline


DEFAULT_INPUT_ROOT = Path("/home/foivos/data/glossapi_work/hf_release_publish/data")
DEFAULT_OUTPUT_ROOT = Path("/home/foivos/data/glossapi_work/nanochat_glossapi_en_vs_el")
DEFAULT_EXCLUDED = ["HuggingFaceFW/finepdfs-edu"]
DEFAULT_BADNESS = 10.0
DEFAULT_MOJIBAKE = None
DEFAULT_TRAIN_CHARS = 2_000_000_000
DEFAULT_VAL_CHARS = 251_485_449
DEFAULT_TEST_CHARS = 251_485_449
DEFAULT_TRAIN_SHARD_CHARS = 250_000_000
DEFAULT_ROW_GROUP_ROWS = 2048
DEFAULT_SEED = 20260322
DEFAULT_NAMESPACE = "nanochat-glossapi-en-vs-el"
DEFAULT_CHUNK_TARGET_CHARS = 3200
DEFAULT_CHUNK_MIN_CHARS = 1800
DEFAULT_CHUNK_MAX_CHARS = 5200
DEFAULT_FULLY_INCLUDE_BELOW_SHARE = None
DEFAULT_DEDUP_ACTION = "ignore"
DEFAULT_DEDUP_EXACT_STAGE = "strict_and_relaxed"
DEFAULT_DEDUP_INTER_DATASET_POLICY = "share_aware"

DEDUP_ACTIONS = {"ignore", "annotate", "drop_intra", "drop_intra_and_inter"}
DEDUP_EXACT_STAGE_OPTIONS = {"strict_only", "strict_and_relaxed"}
DEDUP_INTER_DATASET_POLICIES = {"quality_first", "share_aware"}

HEADER_RE = re.compile(r"(?m)^(#{1,6})[ \t]+.+$")
PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?;·…])\s+")


@dataclass(frozen=True)
class ResolvedInputLayout:
    input_root: Path
    repo_root: Path | None
    data_root: Path
    data_glob: str
    latest_dedup_pointer_path: Path | None
    has_chars_column: bool
    has_needs_ocr_column: bool
    has_ocr_success_column: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Greek train/val/test manifests and parquet exports for nanochat.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--badness-lt", type=float, default=DEFAULT_BADNESS)
    parser.add_argument(
        "--mojibake-lte",
        type=float,
        default=DEFAULT_MOJIBAKE,
        help="Optional inclusive mojibake threshold. Null mojibake values are kept.",
    )
    parser.add_argument("--train-chars", type=int, default=DEFAULT_TRAIN_CHARS)
    parser.add_argument("--val-chars", type=int, default=DEFAULT_VAL_CHARS)
    parser.add_argument("--test-chars", type=int, default=DEFAULT_TEST_CHARS)
    parser.add_argument("--train-shard-target-chars", type=int, default=DEFAULT_TRAIN_SHARD_CHARS)
    parser.add_argument("--row-group-rows", type=int, default=DEFAULT_ROW_GROUP_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--salt", default=None, help="Optional explicit split salt override. If omitted, a deterministic salt is derived from namespace + seed.")
    parser.add_argument("--exclude-dataset", action="append", default=[], help="Dataset(s) to exclude.")
    parser.add_argument("--chunk-markdown", action="store_true", help="Rechunk exported markdown texts by nearest header, with paragraph fallback for oversized sections.")
    parser.add_argument("--chunk-target-chars", type=int, default=DEFAULT_CHUNK_TARGET_CHARS)
    parser.add_argument("--chunk-min-chars", type=int, default=DEFAULT_CHUNK_MIN_CHARS)
    parser.add_argument("--chunk-max-chars", type=int, default=DEFAULT_CHUNK_MAX_CHARS)
    parser.add_argument(
        "--fully-include-below-share",
        type=float,
        default=DEFAULT_FULLY_INCLUDE_BELOW_SHARE,
        help=(
            "If set, fully include any retained allocation pool whose character share is at or below this "
            "fraction of the retained corpus, then allocate the remaining budget proportionally across larger pools."
        ),
    )
    parser.add_argument("--shuffle-chunks", dest="shuffle_chunks", action="store_true", default=True, help="Deterministically shuffle exported chunks within each split before writing shards.")
    parser.add_argument("--no-shuffle-chunks", dest="shuffle_chunks", action="store_false", help="Disable deterministic chunk-level shuffling.")
    parser.add_argument("--dedup-action", choices=sorted(DEDUP_ACTIONS), default=DEFAULT_DEDUP_ACTION)
    parser.add_argument("--dedup-exact-stage", choices=sorted(DEDUP_EXACT_STAGE_OPTIONS), default=DEFAULT_DEDUP_EXACT_STAGE)
    parser.add_argument(
        "--dedup-similarity-threshold",
        type=float,
        default=None,
        help="Builder-time near-duplicate threshold. If omitted, the bundle default is used.",
    )
    parser.add_argument(
        "--dedup-inter-dataset-policy",
        choices=sorted(DEDUP_INTER_DATASET_POLICIES),
        default=DEFAULT_DEDUP_INTER_DATASET_POLICY,
    )
    parser.add_argument(
        "--dedup-source-weights-path",
        type=Path,
        default=None,
        help="Optional JSON mapping source_dataset to a positive builder share weight.",
    )
    parser.add_argument(
        "--dedup-metadata-root",
        type=Path,
        default=None,
        help="Explicit builder_metadata directory. If omitted, the script resolves dedup_metadata/latest.json under the dataset repo root.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def hash_bytes(data: bytes) -> str:
    return blake3(data).hexdigest()


def stable_doc_key(source_dataset: str, source_doc_id: str) -> str:
    return hash_bytes(f"{source_dataset}\0{source_doc_id}".encode("utf-8"))


def stable_chunk_key(*parts: str) -> str:
    return hashlib.md5("::".join(parts).encode("utf-8", errors="replace")).hexdigest()


def write_text_parquet(path: Path, texts: list[str], row_group_rows: int) -> None:
    table = pa.Table.from_pydict({"text": texts}, schema=pa.schema([("text", pa.string())]))
    pq.write_table(table, path, compression="zstd", row_group_size=max(1, int(row_group_rows)))


def has_any_parquet(root: Path) -> bool:
    return root.exists() and any(root.glob("*.parquet"))


def detect_has_column(data_root: Path, column_name: str) -> bool:
    for path in data_root.glob("*.parquet"):
        if column_name in pq.read_schema(path).names:
            return True
    return False


def resolve_input_layout(input_root: Path) -> ResolvedInputLayout:
    input_root = input_root.resolve()
    if has_any_parquet(input_root / "data"):
        repo_root = input_root
        data_root = input_root / "data"
    elif has_any_parquet(input_root):
        data_root = input_root
        repo_root = input_root.parent if (input_root.parent / "dedup_metadata" / "latest.json").exists() else None
    else:
        raise FileNotFoundError(f"Could not find parquet data under {input_root}")
    latest_path = None
    if repo_root is not None:
        candidate = repo_root / "dedup_metadata" / "latest.json"
        if candidate.exists():
            latest_path = candidate
    has_chars_column = detect_has_column(data_root, "chars")
    has_needs_ocr_column = detect_has_column(data_root, "needs_ocr")
    has_ocr_success_column = detect_has_column(data_root, "ocr_success")
    return ResolvedInputLayout(
        input_root=input_root,
        repo_root=repo_root,
        data_root=data_root,
        data_glob=str((data_root / "*.parquet").resolve()),
        latest_dedup_pointer_path=latest_path,
        has_chars_column=has_chars_column,
        has_needs_ocr_column=has_needs_ocr_column,
        has_ocr_success_column=has_ocr_success_column,
    )


def load_latest_dedup_pointer(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_dedup_metadata_root(layout: ResolvedInputLayout, explicit_root: Path | None) -> tuple[Path | None, Path | None]:
    if explicit_root is not None:
        dedup_root = explicit_root.resolve()
        return dedup_root, None
    if layout.latest_dedup_pointer_path is None or layout.repo_root is None:
        return None, None
    pointer = load_latest_dedup_pointer(layout.latest_dedup_pointer_path)
    dedup_root = (layout.repo_root / pointer["builder_metadata_root"]).resolve()
    code_root = (layout.repo_root / pointer["code_root"]).resolve()
    return dedup_root, code_root


def dataset_name_sql(column_name: str = "filename") -> str:
    return f"""
    replace(
      regexp_replace(
        regexp_replace({column_name}, '^.*/', ''),
        '(\\.part-\\d+)?\\.parquet$',
        ''
      ),
      '__',
      '/'
    )
    """.strip()


def build_quality_where_sql(
    *,
    source_dataset_sql: str,
    excluded_sql: str,
    badness_lt: float,
    mojibake_lte: float | None,
    has_needs_ocr_column: bool,
    has_ocr_success_column: bool,
    greek_badness_sql: str = "greek_badness_score",
    mojibake_sql: str = "mojibake_badness_score",
    needs_ocr_sql: str = "needs_ocr",
    ocr_success_sql: str = "ocr_success",
) -> str:
    predicates = [
        f"{greek_badness_sql} < {float(badness_lt)}",
        f"{source_dataset_sql} NOT IN ({excluded_sql})",
    ]
    if mojibake_lte is not None:
        predicates.append(f"({mojibake_sql} IS NULL OR {mojibake_sql} <= {float(mojibake_lte)})")
    if has_needs_ocr_column:
        if has_ocr_success_column:
            predicates.append(
                f"(coalesce({needs_ocr_sql}, false) = FALSE OR coalesce({ocr_success_sql}, false) = TRUE)"
            )
        else:
            predicates.append(f"coalesce({needs_ocr_sql}, false) = FALSE")
    return "\n      AND ".join(predicates)


def normalize_chunk_whitespace(text: str) -> str:
    return text.strip()


def split_markdown_sections(text: str) -> list[str]:
    matches = list(HEADER_RE.finditer(text))
    if not matches:
        return [normalize_chunk_whitespace(text)] if text.strip() else []

    sections: list[str] = []
    first_start = matches[0].start()
    if text[:first_start].strip():
        sections.append(normalize_chunk_whitespace(text[:first_start]))

    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        chunk = normalize_chunk_whitespace(text[start:end])
        if chunk:
            sections.append(chunk)
    return sections


def split_paragraphs(text: str) -> list[str]:
    parts = [normalize_chunk_whitespace(part) for part in PARAGRAPH_SPLIT_RE.split(text)]
    return [part for part in parts if part]


def split_sentences(text: str) -> list[str]:
    parts = [normalize_chunk_whitespace(part) for part in SENTENCE_BREAK_RE.split(text)]
    return [part for part in parts if part]


def hard_wrap_text(text: str, max_chars: int) -> list[str]:
    text = normalize_chunk_whitespace(text)
    if not text:
        return []
    out: list[str] = []
    cursor = 0
    while cursor < len(text):
        end = min(len(text), cursor + max_chars)
        if end < len(text):
            window = text[cursor:end]
            split_at = max(window.rfind("\n"), window.rfind(" "), window.rfind("\t"))
            if split_at > max_chars // 2:
                end = cursor + split_at
        piece = normalize_chunk_whitespace(text[cursor:end])
        if piece:
            out.append(piece)
        cursor = max(end, cursor + 1)
    return out


def pack_units(units: list[str], target_chars: int, min_chars: int, max_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_chars = 0

    def flush() -> None:
        nonlocal current, current_chars
        if not current:
            return
        text = normalize_chunk_whitespace("\n\n".join(current))
        if text:
            chunks.append(text)
        current = []
        current_chars = 0

    for unit in units:
        unit = normalize_chunk_whitespace(unit)
        if not unit:
            continue
        unit_chars = len(unit)
        if current_chars == 0:
            current = [unit]
            current_chars = unit_chars
            if current_chars >= max_chars:
                flush()
            continue

        proposed_chars = current_chars + 2 + unit_chars
        if proposed_chars <= max_chars:
            if current_chars < min_chars or abs(target_chars - proposed_chars) <= abs(target_chars - current_chars):
                current.append(unit)
                current_chars = proposed_chars
            else:
                flush()
                current = [unit]
                current_chars = unit_chars
        else:
            if current_chars >= min_chars:
                flush()
                current = [unit]
                current_chars = unit_chars
                if current_chars >= max_chars:
                    flush()
            else:
                current.append(unit)
                current_chars = proposed_chars
                flush()
    flush()
    return chunks


def chunk_oversized_section(text: str, target_chars: int, min_chars: int, max_chars: int) -> list[str]:
    paragraphs = split_paragraphs(text)
    if len(paragraphs) > 1:
        packed_paragraphs = pack_units(paragraphs, target_chars, min_chars, max_chars)
        if len(packed_paragraphs) == 1 and packed_paragraphs[0] == text:
            packed_paragraphs = []
        out: list[str] = []
        for chunk in packed_paragraphs:
            if len(chunk) > max_chars:
                out.extend(chunk_oversized_section(chunk, target_chars, min_chars, max_chars))
            else:
                out.append(chunk)
        if out:
            return out

    sentences = split_sentences(text)
    if len(sentences) > 1:
        packed_sentences = pack_units(sentences, target_chars, min_chars, max_chars)
        if len(packed_sentences) == 1 and packed_sentences[0] == text:
            packed_sentences = []
        out = []
        for chunk in packed_sentences:
            if len(chunk) > max_chars:
                out.extend(hard_wrap_text(chunk, max_chars))
            else:
                out.append(chunk)
        if out:
            return out

    return hard_wrap_text(text, max_chars)


def maybe_chunk_text(text: str, *, enabled: bool, target_chars: int, min_chars: int, max_chars: int) -> list[str]:
    text = normalize_chunk_whitespace(text)
    if not text:
        return []
    if not enabled:
        return [text]

    sections = split_markdown_sections(text)
    packed = pack_units(sections, target_chars, min_chars, max_chars)
    out: list[str] = []
    for chunk in packed:
        if len(chunk) <= max_chars:
            out.append(chunk)
        else:
            out.extend(chunk_oversized_section(chunk, target_chars, min_chars, max_chars))
    return [chunk for chunk in out if chunk]


def materialize_chunk_records(
    reader: Any,
    *,
    chunk_markdown: bool,
    chunk_target_chars: int,
    chunk_min_chars: int,
    chunk_max_chars: int,
    split_salt: str,
    shuffle_chunks: bool,
) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    append = records.append
    for batch in reader:
        datasets = batch.column("source_dataset").to_pylist()
        doc_ids = batch.column("source_doc_id").to_pylist()
        stable_keys = batch.column("stable_key").to_pylist()
        texts = batch.column("text").to_pylist()
        for source_dataset, source_doc_id, base_key, text in zip(datasets, doc_ids, stable_keys, texts, strict=True):
            chunks = maybe_chunk_text(
                str(text),
                enabled=chunk_markdown,
                target_chars=chunk_target_chars,
                min_chars=chunk_min_chars,
                max_chars=chunk_max_chars,
            )
            for idx, chunk in enumerate(chunks):
                chunk_key = stable_chunk_key(str(base_key), str(source_dataset), str(source_doc_id), str(idx), split_salt)
                append((chunk_key, chunk))
    if shuffle_chunks:
        records.sort(key=lambda item: item[0])
    return records


def export_split(
    con: duckdb.DuckDBPyConnection,
    *,
    export_root: Path,
    split: str,
    query: str,
    row_group_rows: int,
    train_shard_target_chars: int,
    chunk_markdown: bool,
    chunk_target_chars: int,
    chunk_min_chars: int,
    chunk_max_chars: int,
    split_salt: str,
    shuffle_chunks: bool,
) -> list[dict[str, int | str]]:
    reader = con.execute(query).fetch_record_batch(rows_per_batch=2048)
    records = materialize_chunk_records(
        reader,
        chunk_markdown=chunk_markdown,
        chunk_target_chars=chunk_target_chars,
        chunk_min_chars=chunk_min_chars,
        chunk_max_chars=chunk_max_chars,
        split_salt=split_salt,
        shuffle_chunks=shuffle_chunks,
    )
    shard_rows: list[str] = []
    shard_chars = 0
    shard_idx = 0
    outputs: list[dict[str, int | str]] = []

    def flush_train_shard() -> None:
        nonlocal shard_rows, shard_chars, shard_idx
        if not shard_rows:
            return
        path = export_root / f"shard_{shard_idx:05d}.parquet"
        write_text_parquet(path, shard_rows, row_group_rows)
        outputs.append({"path": str(path), "rows": len(shard_rows), "chars": shard_chars})
        shard_idx += 1
        shard_rows = []
        shard_chars = 0

    if split == "train":
        for _, chunk in records:
            shard_rows.append(chunk)
            shard_chars += len(chunk)
            if shard_chars >= train_shard_target_chars:
                flush_train_shard()
        flush_train_shard()
        return outputs

    texts: list[str] = []
    total_chars = 0
    for _, chunk in records:
        texts.append(chunk)
        total_chars += len(chunk)
    filename = "shard_06542.parquet" if split == "val" else "test.parquet"
    path = export_root / filename
    write_text_parquet(path, texts, row_group_rows)
    outputs.append({"path": str(path), "rows": len(texts), "chars": total_chars})
    return outputs


def validate_dedup_action(action: str) -> str:
    if action not in DEDUP_ACTIONS:
        raise ValueError(f"dedup_action must be one of: {', '.join(sorted(DEDUP_ACTIONS))}")
    return action


def validate_dedup_exact_stage(exact_stage: str) -> str:
    if exact_stage not in DEDUP_EXACT_STAGE_OPTIONS:
        raise ValueError(f"dedup_exact_stage must be one of: {', '.join(sorted(DEDUP_EXACT_STAGE_OPTIONS))}")
    return exact_stage


def validate_dedup_inter_dataset_policy(policy: str) -> str:
    if policy not in DEDUP_INTER_DATASET_POLICIES:
        raise ValueError(
            f"dedup_inter_dataset_policy must be one of: {', '.join(sorted(DEDUP_INTER_DATASET_POLICIES))}"
        )
    return policy


def build_dedup_annotations(
    filtered_docs: pd.DataFrame,
    *,
    dedup_metadata_root: Path,
    dedup_action: str,
    dedup_exact_stage: str,
    dedup_similarity_threshold: float | None,
    dedup_inter_dataset_policy: str,
    dedup_source_weights_path: Path | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dedup_action = validate_dedup_action(dedup_action)
    dedup_exact_stage = validate_dedup_exact_stage(dedup_exact_stage)
    dedup_inter_dataset_policy = validate_dedup_inter_dataset_policy(dedup_inter_dataset_policy)
    if filtered_docs.empty:
        return filtered_docs.iloc[0:0].copy(), {"dedup_action": dedup_action, "rows_before": 0, "rows_after": 0}

    annotated, summary = pipeline.apply_builder_dedup(
        filtered_docs,
        dedup_metadata_root=dedup_metadata_root,
        dedup_action=dedup_action,
        dedup_exact_stage=dedup_exact_stage,
        dedup_similarity_threshold=dedup_similarity_threshold,
        dedup_inter_dataset_policy=dedup_inter_dataset_policy,
        dedup_source_weights_path=dedup_source_weights_path,
    )
    return annotated, summary


def build_filtered_source_docs_sql(
    *,
    layout: ResolvedInputLayout,
    data_glob_sql: str,
    excluded_sql: str,
    badness_lt: float,
    mojibake_lte: float | None,
    salt: str,
    chars_expr: str,
) -> str:
    source_dataset_expr = dataset_name_sql("src.filename")
    quality_where = build_quality_where_sql(
        source_dataset_sql=source_dataset_expr,
        excluded_sql=excluded_sql,
        badness_lt=badness_lt,
        mojibake_lte=mojibake_lte,
        has_needs_ocr_column=layout.has_needs_ocr_column,
        has_ocr_success_column=layout.has_ocr_success_column,
        greek_badness_sql="src.greek_badness_score",
        mojibake_sql="src.mojibake_badness_score",
        needs_ocr_sql="src.needs_ocr",
        ocr_success_sql="src.ocr_success",
    )
    return f"""
    CREATE OR REPLACE TEMP VIEW filtered_source_docs AS
    SELECT
      {source_dataset_expr} AS source_dataset,
      src.source_doc_id,
      {chars_expr} AS chars,
      md5({source_dataset_expr} || ':' || src.source_doc_id || ':{salt}') AS stable_key
    FROM read_parquet({data_glob_sql}, filename=true, union_by_name=true) AS src
    WHERE {quality_where}
    """


def build_docs_cte_sql(
    *,
    layout: ResolvedInputLayout,
    data_glob_sql: str,
    excluded_sql: str,
    badness_lt: float,
    mojibake_lte: float | None,
    chars_expr: str,
) -> str:
    source_dataset_expr = dataset_name_sql("src.filename")
    quality_where = build_quality_where_sql(
        source_dataset_sql=source_dataset_expr,
        excluded_sql=excluded_sql,
        badness_lt=badness_lt,
        mojibake_lte=mojibake_lte,
        has_needs_ocr_column=layout.has_needs_ocr_column,
        has_ocr_success_column=layout.has_ocr_success_column,
        greek_badness_sql="src.greek_badness_score",
        mojibake_sql="src.mojibake_badness_score",
        needs_ocr_sql="src.needs_ocr",
        ocr_success_sql="src.ocr_success",
    )
    return f"""
      WITH docs AS (
        SELECT
          {source_dataset_expr} AS source_dataset,
          src.source_doc_id,
          src.text,
          {chars_expr} AS chars
        FROM read_parquet({data_glob_sql}, filename=true, union_by_name=true) AS src
        WHERE {quality_where}
      )
    """


def build_assigned_view_sql(*, requested_total_target_chars: int, effective_total_target_chars: int, val_chars: int, test_chars: int, train_chars: int, fully_include_below_share: float | None) -> str:
    if fully_include_below_share is None:
        allocation_ctes = ""
        allocation_budget_expr = f"gt.allocation_chars * {effective_total_target_chars}::DOUBLE / ct.total_chars"
        fully_include_expr = "FALSE"
        fully_include_cutoff_expr = "NULL::DOUBLE"
    else:
        allocation_ctes = f"""
    ,
    allocation AS (
      SELECT
        gt.allocation_group,
        gt.allocation_chars,
        ct.total_chars,
        ({fully_include_below_share}) * ct.total_chars AS fully_include_cutoff_chars,
        gt.allocation_chars <= ({fully_include_below_share}) * ct.total_chars AS fully_include_group
      FROM group_totals gt
      CROSS JOIN corpus_total ct
    ),
    allocation_totals AS (
      SELECT
        coalesce(sum(CASE WHEN fully_include_group THEN allocation_chars ELSE 0 END), 0) AS fully_include_total_chars,
        coalesce(sum(CASE WHEN NOT fully_include_group THEN allocation_chars ELSE 0 END), 0) AS sampled_total_chars
      FROM allocation
    )
        """
        allocation_budget_expr = f"""
        CASE
          WHEN alloc.fully_include_group THEN gt.allocation_chars::DOUBLE
          WHEN alloc_totals.sampled_total_chars > 0 THEN
            gt.allocation_chars * greatest(0, {effective_total_target_chars} - alloc_totals.fully_include_total_chars)::DOUBLE / alloc_totals.sampled_total_chars
          ELSE 0::DOUBLE
        END
        """
        fully_include_expr = "alloc.fully_include_group"
        fully_include_cutoff_expr = "alloc.fully_include_cutoff_chars"
    return f"""
    CREATE OR REPLACE TEMP TABLE assigned AS
    WITH group_totals AS (
      SELECT allocation_group, sum(chars) AS allocation_chars
      FROM builder_input
      GROUP BY allocation_group
    ),
    corpus_total AS (
      SELECT coalesce(sum(allocation_chars), 0) AS total_chars
      FROM group_totals
    ){allocation_ctes},
    ranked AS (
      SELECT
        b.*,
        gt.allocation_chars,
        ct.total_chars,
        {allocation_budget_expr} AS allocation_budget_chars,
        {fully_include_expr} AS fully_include_group,
        {fully_include_cutoff_expr} AS fully_include_cutoff_chars,
        ({allocation_budget_expr}) * {val_chars}::DOUBLE / {effective_total_target_chars} AS val_quota_chars,
        ({allocation_budget_expr}) * {test_chars}::DOUBLE / {effective_total_target_chars} AS test_quota_chars,
        ({allocation_budget_expr}) * {train_chars}::DOUBLE / {effective_total_target_chars} AS train_quota_chars,
        sum(chars) OVER (
          PARTITION BY b.allocation_group
          ORDER BY stable_key, source_doc_id
          ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cum_chars
      FROM builder_input b
      JOIN group_totals gt USING(allocation_group)
      CROSS JOIN corpus_total ct
      {"JOIN allocation alloc USING(allocation_group)" if fully_include_below_share is not None else ""}
      {"CROSS JOIN allocation_totals alloc_totals" if fully_include_below_share is not None else ""}
    )
    SELECT
      source_dataset,
      source_doc_id,
      chars,
      stable_key,
      allocation_group,
      allocation_chars,
      total_chars,
      allocation_budget_chars,
      fully_include_group,
      fully_include_cutoff_chars,
      val_quota_chars,
      test_quota_chars,
      train_quota_chars,
      dedup_pool_key,
      dedup_pool_source_count,
      dedup_is_shared_pool,
      dedup_family_size,
      dedup_family_role,
      CASE
        WHEN cum_chars <= val_quota_chars THEN 'val'
        WHEN cum_chars <= val_quota_chars + test_quota_chars THEN 'test'
        WHEN cum_chars <= val_quota_chars + test_quota_chars + train_quota_chars THEN 'train'
        ELSE 'drop'
      END AS split
    FROM ranked
    """


def export_manifest_summary(export_manifest: dict[str, list[dict[str, int | str]]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for split, items in export_manifest.items():
        summary[split] = {
            "rows": int(sum(int(item["rows"]) for item in items)),
            "chars": int(sum(int(item["chars"]) for item in items)),
        }
    return summary


def main() -> None:
    args = parse_args()
    if args.fully_include_below_share is not None and not (0.0 < args.fully_include_below_share < 1.0):
        raise ValueError("--fully-include-below-share must be between 0 and 1")

    excluded = sorted(set(DEFAULT_EXCLUDED + list(args.exclude_dataset)))
    layout = resolve_input_layout(args.input_root)
    dedup_metadata_root, dedup_code_root = resolve_dedup_metadata_root(layout, args.dedup_metadata_root)
    if args.dedup_action != "ignore" and dedup_metadata_root is None:
        raise FileNotFoundError("dedup_action requires an in-repo dedup_metadata/latest.json or --dedup-metadata-root")

    output_root = args.output_root.resolve()
    manifests_dir = output_root / "manifests"
    export_root = output_root / "exports"
    scripts_dir = output_root / "scripts"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    export_root.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={max(1, int(args.threads))}")
    con.execute("PRAGMA preserve_insertion_order=false")

    excluded_sql = ", ".join(sql_quote(item) for item in excluded) or "''"
    data_glob_sql = sql_quote(layout.data_glob)
    effective_salt = args.salt or f"{args.split_namespace}:seed={args.seed}"
    salt = effective_salt.replace("'", "''")
    chars_expr = "coalesce(src.chars, length(src.text))" if layout.has_chars_column else "length(src.text)"

    script_path = Path(__file__).resolve()
    stored_script_path = scripts_dir / script_path.name
    shutil.copy2(script_path, stored_script_path)

    con.execute(
        build_filtered_source_docs_sql(
            layout=layout,
            data_glob_sql=data_glob_sql,
            excluded_sql=excluded_sql,
            badness_lt=args.badness_lt,
            mojibake_lte=args.mojibake_lte,
            salt=salt,
            chars_expr=chars_expr,
        )
    )

    filtered_doc_rows = con.execute(
        """
        SELECT source_dataset, source_doc_id, chars, stable_key
        FROM filtered_source_docs
        """
    ).fetchdf()

    dedup_summary: dict[str, Any] | None = None
    if args.dedup_action != "ignore":
        dedup_docs, dedup_summary = build_dedup_annotations(
            filtered_doc_rows,
            dedup_metadata_root=dedup_metadata_root,
            dedup_action=args.dedup_action,
            dedup_exact_stage=args.dedup_exact_stage,
            dedup_similarity_threshold=args.dedup_similarity_threshold,
            dedup_inter_dataset_policy=args.dedup_inter_dataset_policy,
            dedup_source_weights_path=args.dedup_source_weights_path,
        )
        con.register("builder_dedup_docs_df", dedup_docs)
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW builder_input AS
            SELECT
              source_dataset,
              source_doc_id,
              chars,
              stable_key,
              coalesce(dedup_pool_key, concat('unique:', source_dataset)) AS allocation_group,
              dedup_pool_key,
              dedup_pool_source_count,
              dedup_is_shared_pool,
              dedup_family_size,
              dedup_family_role
            FROM builder_dedup_docs_df
            """
        )
    else:
        con.execute(
            """
            CREATE OR REPLACE TEMP VIEW builder_input AS
            SELECT
              source_dataset,
              source_doc_id,
              chars,
              stable_key,
              concat('unique:', source_dataset) AS allocation_group,
              NULL::VARCHAR AS dedup_pool_key,
              NULL::BIGINT AS dedup_pool_source_count,
              NULL::BOOLEAN AS dedup_is_shared_pool,
              NULL::BIGINT AS dedup_family_size,
              NULL::VARCHAR AS dedup_family_role
            FROM filtered_source_docs
            """
        )

    requested_total_target_chars = int(args.train_chars + args.val_chars + args.test_chars)
    effective_total_target_chars = requested_total_target_chars
    if args.fully_include_below_share is not None:
        allocation_check = con.execute(
            f"""
            WITH group_totals AS (
              SELECT allocation_group, sum(chars) AS allocation_chars
              FROM builder_input
              GROUP BY allocation_group
            ),
            corpus_total AS (
              SELECT coalesce(sum(allocation_chars), 0) AS total_chars
              FROM group_totals
            ),
            allocation AS (
              SELECT
                gt.allocation_group,
                gt.allocation_chars,
                ct.total_chars,
                gt.allocation_chars <= ({args.fully_include_below_share}) * ct.total_chars AS fully_include_group
              FROM group_totals gt
              CROSS JOIN corpus_total ct
            )
            SELECT
              coalesce(sum(CASE WHEN fully_include_group THEN allocation_chars ELSE 0 END), 0) AS fully_include_total_chars,
              coalesce(max(total_chars), 0) AS total_chars
            FROM allocation
            """
        ).fetchone()
        fully_include_total_chars = int(allocation_check[0] or 0)
        retained_total_chars = int(allocation_check[1] or 0)
        if retained_total_chars <= 0:
            raise ValueError("No retained source rows matched the requested filters")
        effective_total_target_chars = max(requested_total_target_chars, fully_include_total_chars)

    con.execute(
        build_assigned_view_sql(
            requested_total_target_chars=requested_total_target_chars,
            effective_total_target_chars=effective_total_target_chars,
            val_chars=args.val_chars,
            test_chars=args.test_chars,
            train_chars=args.train_chars,
            fully_include_below_share=args.fully_include_below_share,
        )
    )

    for split in ("train", "val", "test"):
        con.execute(
            f"""
            COPY (
              SELECT source_dataset, source_doc_id, chars, stable_key, allocation_group
              FROM assigned
              WHERE split = '{split}'
              ORDER BY allocation_group, stable_key, source_doc_id
            ) TO '{(manifests_dir / f"{split}_manifest.csv").as_posix()}'
            (HEADER, DELIMITER ',')
            """
        )

    docs_cte = build_docs_cte_sql(
        layout=layout,
        data_glob_sql=data_glob_sql,
        excluded_sql=excluded_sql,
        badness_lt=args.badness_lt,
        mojibake_lte=args.mojibake_lte,
        chars_expr=chars_expr,
    )
    export_manifest = {
        "train": export_split(
            con,
            export_root=export_root,
            split="train",
            query=f"""
              {docs_cte}
              SELECT d.source_dataset, d.source_doc_id, a.stable_key, d.text
              FROM docs d
              JOIN assigned a
                ON d.source_dataset = a.source_dataset
               AND d.source_doc_id = a.source_doc_id
              WHERE a.split = 'train'
              ORDER BY a.allocation_group, a.stable_key, a.source_doc_id
            """,
            row_group_rows=args.row_group_rows,
            train_shard_target_chars=args.train_shard_target_chars,
            chunk_markdown=args.chunk_markdown,
            chunk_target_chars=args.chunk_target_chars,
            chunk_min_chars=args.chunk_min_chars,
            chunk_max_chars=args.chunk_max_chars,
            split_salt=f"{effective_salt}:train",
            shuffle_chunks=args.shuffle_chunks,
        ),
        "val": export_split(
            con,
            export_root=export_root,
            split="val",
            query=f"""
              {docs_cte}
              SELECT d.source_dataset, d.source_doc_id, a.stable_key, d.text
              FROM docs d
              JOIN assigned a
                ON d.source_dataset = a.source_dataset
               AND d.source_doc_id = a.source_doc_id
              WHERE a.split = 'val'
              ORDER BY a.allocation_group, a.stable_key, a.source_doc_id
            """,
            row_group_rows=args.row_group_rows,
            train_shard_target_chars=args.train_shard_target_chars,
            chunk_markdown=args.chunk_markdown,
            chunk_target_chars=args.chunk_target_chars,
            chunk_min_chars=args.chunk_min_chars,
            chunk_max_chars=args.chunk_max_chars,
            split_salt=f"{effective_salt}:val",
            shuffle_chunks=args.shuffle_chunks,
        ),
        "test": export_split(
            con,
            export_root=export_root,
            split="test",
            query=f"""
              {docs_cte}
              SELECT d.source_dataset, d.source_doc_id, a.stable_key, d.text
              FROM docs d
              JOIN assigned a
                ON d.source_dataset = a.source_dataset
               AND d.source_doc_id = a.source_doc_id
              WHERE a.split = 'test'
              ORDER BY a.allocation_group, a.stable_key, a.source_doc_id
            """,
            row_group_rows=args.row_group_rows,
            train_shard_target_chars=args.train_shard_target_chars,
            chunk_markdown=args.chunk_markdown,
            chunk_target_chars=args.chunk_target_chars,
            chunk_min_chars=args.chunk_min_chars,
            chunk_max_chars=args.chunk_max_chars,
            split_salt=f"{effective_salt}:test",
            shuffle_chunks=args.shuffle_chunks,
        ),
    }

    split_summary_source_docs: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        row = con.execute(
            f"""
            SELECT count(*) AS rows, coalesce(sum(chars), 0) AS chars
            FROM assigned
            WHERE split = '{split}'
            """
        ).fetchone()
        split_summary_source_docs[split] = {"rows": int(row[0]), "chars": int(row[1])}

    split_summary_exported_chunks = export_manifest_summary(export_manifest)

    by_dataset = con.execute(
        """
        SELECT split, source_dataset, count(*) AS rows, sum(chars) AS chars
        FROM assigned
        WHERE split IN ('train', 'val', 'test')
        GROUP BY split, source_dataset
        ORDER BY split, source_dataset
        """
    ).fetchdf()
    by_dataset.to_csv(output_root / "split_dataset_summary.csv", index=False)

    quota_rows = con.execute(
        """
        SELECT
          allocation_group,
          max(allocation_chars) AS allocation_chars,
          max(total_chars) AS corpus_total_chars,
          max(allocation_budget_chars) AS allocation_budget_chars,
          max(fully_include_group) AS fully_include_group,
          max(fully_include_cutoff_chars) AS fully_include_cutoff_chars,
          max(train_quota_chars) AS train_target_chars,
          max(val_quota_chars) AS val_target_chars,
          max(test_quota_chars) AS test_target_chars
        FROM assigned
        GROUP BY allocation_group
        ORDER BY allocation_chars DESC, allocation_group
        """
    ).fetchdf()
    quota_rows.to_csv(output_root / "dataset_quota_summary.csv", index=False)

    if args.dedup_action != "ignore":
        pool_rows = con.execute(
            """
            SELECT split, allocation_group, count(*) AS rows, sum(chars) AS chars
            FROM assigned
            WHERE split IN ('train', 'val', 'test')
            GROUP BY split, allocation_group
            ORDER BY split, allocation_group
            """
        ).fetchdf()
        pool_rows.to_csv(output_root / "split_pool_summary.csv", index=False)

    payload: dict[str, Any] = {
        "input_root": str(layout.input_root),
        "data_root": str(layout.data_root),
        "repo_root": str(layout.repo_root) if layout.repo_root is not None else None,
        "output_root": str(output_root),
        "excluded_datasets": excluded,
        "badness_lt": args.badness_lt,
        "mojibake_lte": args.mojibake_lte,
        "train_target_chars": args.train_chars,
        "val_target_chars": args.val_chars,
        "test_target_chars": args.test_chars,
        "requested_total_target_chars": requested_total_target_chars,
        "effective_total_target_chars": effective_total_target_chars,
        "train_shard_target_chars": args.train_shard_target_chars,
        "row_group_rows": args.row_group_rows,
        "fully_include_below_share": args.fully_include_below_share,
        "chunk_markdown": bool(args.chunk_markdown),
        "chunk_target_chars": args.chunk_target_chars,
        "chunk_min_chars": args.chunk_min_chars,
        "chunk_max_chars": args.chunk_max_chars,
        "shuffle_chunks": bool(args.shuffle_chunks),
        "seed": args.seed,
        "split_namespace": args.split_namespace,
        "salt": effective_salt,
        "script_snapshot_path": str(stored_script_path.resolve()),
        "script_sha256": file_sha256(stored_script_path),
        "split_summary_source_docs": split_summary_source_docs,
        "split_summary_exported_chunks": split_summary_exported_chunks,
        "export_manifest": export_manifest,
        "filtered_rows_before_dedup": int(len(filtered_doc_rows)),
        "filtered_chars_before_dedup": int(filtered_doc_rows["chars"].sum()) if not filtered_doc_rows.empty else 0,
        "dedup_action": args.dedup_action,
        "dedup_exact_stage": args.dedup_exact_stage,
        "dedup_inter_dataset_policy": args.dedup_inter_dataset_policy,
        "dedup_metadata_root": str(dedup_metadata_root) if dedup_metadata_root is not None else None,
        "dedup_code_root": str(dedup_code_root) if dedup_code_root is not None else None,
        "dedup_source_weights_path": str(args.dedup_source_weights_path.resolve()) if args.dedup_source_weights_path else None,
    }
    if dedup_summary is not None:
        payload["dedup_summary"] = dedup_summary

    (output_root / "prep_config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme_lines = [
        "# GlossAPI Greek Experiment Build",
        "",
        f"- seed: {args.seed}",
        f"- namespace: {args.split_namespace}",
        f"- badness_lt: {args.badness_lt}",
        f"- mojibake_lte: {args.mojibake_lte}",
        f"- dedup_action: {args.dedup_action}",
        f"- dedup_exact_stage: {args.dedup_exact_stage}",
        f"- dedup_similarity_threshold: {payload.get('dedup_summary', {}).get('dedup_similarity_threshold') if dedup_summary else None}",
        f"- fully_include_below_share: {args.fully_include_below_share}",
        f"- chunk_markdown: {args.chunk_markdown}",
        f"- shuffle_chunks: {args.shuffle_chunks}",
        f"- input_root: {layout.input_root}",
        f"- output_root: {output_root}",
        f"- script_snapshot: {stored_script_path.resolve()}",
        f"- excluded_datasets: {', '.join(excluded)}",
        "",
        "This build is deterministic for the same input snapshot, seed, and CLI arguments.",
        "",
    ]
    (output_root / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
