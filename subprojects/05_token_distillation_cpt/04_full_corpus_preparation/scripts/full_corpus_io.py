#!/usr/bin/env python3
"""Shared streaming contracts for Phase-04 corpus materialization.

This module deliberately contains no policy decisions.  It turns a completed
acquisition receipt plus the tracked source registry into deterministic input
records and a fixed canonical Parquet schema.  Cleaning, admission and dedup
remain separate, receipt-bound stages.
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import re
import sqlite3
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from source_lineage import (
    choose_alias_id,
    normalize_work_identifier,
    representation_generation as lineage_representation_generation,
    sha256_parts,
)


ZERO_WIDTH = str.maketrans("", "", "\u200b\u200c\u200d\u2060\ufeff")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MULTI_BLANK = re.compile(r"\n{4,}")
SPACE = re.compile(r"\s+")
URL_PREFIX = re.compile(r"(?i)^https?://(?:www\.)?")
URL_TRAILING = re.compile(r"[/?#].*$")


@dataclass(frozen=True)
class SourceArtifact:
    source_id: str
    repo_id: str
    revision: str
    role: str
    source_family_id: str
    files: tuple[Path, ...]
    config: Mapping[str, Any]


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return value


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def stable_hash(*values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value if value is not None else "").encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def normalize_text(value: object) -> str:
    """Perform representation-safe canonical normalization.

    This is intentionally narrower than a source cleaner: newline/control/NFC
    normalization is safe for identity and token accounting, while HTML, OCR
    and structural removal require later reason-coded policy stages.
    """

    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text.translate(ZERO_WIDTH))
    text = CONTROL.sub("", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = MULTI_BLANK.sub("\n\n\n", text)
    return text.strip()


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"bytes_sha256": sha256_bytes(value), "bytes": len(value)}
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if hasattr(value, "as_py"):
        return jsonable(value.as_py())
    return str(value)


def compact_metadata(row: Mapping[str, Any], excluded: set[str]) -> str:
    payload = {str(key): jsonable(value) for key, value in row.items() if key not in excluded and value is not None}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def first_nonempty(row: Mapping[str, Any], columns: Sequence[str]) -> tuple[str, str] | None:
    for column in columns:
        value = row.get(column)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return column, str(value)
    return None


def canonical_work_value(value: str) -> str:
    result = unicodedata.normalize("NFKC", value).casefold().strip()
    result = URL_PREFIX.sub("", result)
    result = URL_TRAILING.sub("", result)
    return SPACE.sub(" ", result)


def artifact_relative_path(source: SourceArtifact, path: Path) -> str:
    parts = path.resolve().parts
    try:
        revision_index = len(parts) - 1 - list(reversed(parts)).index(source.revision)
    except ValueError:
        return path.name
    return Path(*parts[revision_index + 1 :]).as_posix()


def source_config_map(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {"source_id": "nanochat_base", **dict(config["base"])},
        *[dict(row) for row in config.get("sources", [])],
    ]
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_id = str(row["source_id"])
        if source_id in result:
            raise ValueError(f"duplicate source_id in source registry: {source_id}")
        result[source_id] = row
    return result


def base_family_map(
    config: Mapping[str, Any], aliases: Mapping[str, Any]
) -> dict[str, str]:
    """Map Nanochat names to reviewed cross-snapshot families.

    Only tracked aliases may merge a base name with a candidate family.  Names
    without such evidence remain their own family, which is safer than a fuzzy
    repository-name match.
    """

    family_by_repo = {
        str(row["repo_id"]): str(row["source_family_id"])
        for row in config.get("sources", [])
        if isinstance(row, Mapping) and row.get("repo_id") and row.get("source_family_id")
    }
    result: dict[str, str] = {}
    for alias in aliases.get("aliases", []):
        if not isinstance(alias, Mapping):
            continue
        family = family_by_repo.get(str(alias.get("current_repo_id") or ""))
        if not family:
            continue
        for name in alias.get("initial_source_datasets", []):
            source_dataset = str(name)
            previous = result.get(source_dataset)
            if previous is not None and previous != family:
                raise ValueError(
                    f"conflicting reviewed families for base source_dataset {source_dataset!r}: "
                    f"{previous!r} vs {family!r}"
                )
            result[source_dataset] = family
    return result


def artifacts_from_receipt(config_path: Path, receipt_path: Path, selected: set[str] | None = None) -> list[SourceArtifact]:
    config = read_json_object(config_path)
    receipt = read_json_object(receipt_path)
    if receipt.get("schema_version") != "full_cpt_acquisition_receipt_v1":
        raise ValueError(f"{receipt_path}: unsupported acquisition receipt schema")
    if receipt.get("status") != "passed":
        raise ValueError(f"{receipt_path}: acquisition receipt is not complete")
    configs = source_config_map(config)
    artifacts: list[SourceArtifact] = []
    seen: set[str] = set()
    for receipt_row in receipt.get("sources", []):
        source_id = str(receipt_row["source_id"])
        if source_id not in configs or (selected and source_id not in selected):
            continue
        source = configs[source_id]
        files = tuple(Path(str(row["local_path"])).resolve() for row in receipt_row.get("files", []))
        if not files or any(not path.is_file() for path in files):
            raise FileNotFoundError(f"{source_id}: receipt-bound files are missing")
        if str(receipt_row.get("revision")) != str(source["revision"]):
            raise ValueError(f"{source_id}: receipt revision differs from registry")
        artifacts.append(
            SourceArtifact(
                source_id=source_id,
                repo_id=str(source["repo_id"]),
                revision=str(source["revision"]),
                role=str(source.get("role", "base" if source_id == "nanochat_base" else "candidate")),
                source_family_id=str(source.get("source_family_id", "nanochat_base")),
                files=files,
                config=source,
            )
        )
        seen.add(source_id)
    requested = selected or set(configs)
    missing = requested - seen
    if missing:
        raise ValueError(f"acquisition receipt does not contain requested data sources: {sorted(missing)}")
    return sorted(artifacts, key=lambda item: item.source_id)


def iter_parquet_rows(path: Path, columns: Sequence[str] | None = None, batch_size: int = 1024) -> Iterator[tuple[int, dict[str, Any]]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    available = set(parquet.schema_arrow.names)
    selected = None if columns is None else [column for column in columns if column in available]
    row_index = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=selected, use_threads=False):
        payload = batch.to_pydict()
        for index in range(batch.num_rows):
            yield row_index, {column: values[index] for column, values in payload.items()}
            row_index += 1


def _zstd_text_reader(path: Path) -> io.TextIOBase:
    try:
        import zstandard
    except ImportError as exc:  # pragma: no cover - Clariden runtime contract
        raise RuntimeError("zstandard is required for .jsonl.zst inputs") from exc
    raw = path.open("rb")
    reader = zstandard.ZstdDecompressor().stream_reader(raw)
    return io.TextIOWrapper(reader, encoding="utf-8")


def iter_jsonl_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    opener = _zstd_text_reader if path.name.endswith(".zst") else lambda value: value.open(encoding="utf-8")
    with opener(path) as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{index + 1}: row must be a JSON object")
            yield index, row


def iter_artifact_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    if path.suffix == ".parquet":
        yield from iter_parquet_rows(path)
        return
    if path.name.endswith(".jsonl") or path.name.endswith(".jsonl.zst"):
        yield from iter_jsonl_rows(path)
        return
    raise ValueError(f"unsupported source artifact: {path}")


def iter_grouped_section_rows(
    source: SourceArtifact,
    path: Path,
    *,
    temporary_root: Path,
) -> Iterator[tuple[int, dict[str, Any], str, str]]:
    """Group section rows at canonical work granularity using a disk spool.

    The selected Kallipos/Pergamos artifacts can be much larger than memory.
    SQLite preserves source order per work without assuming that equal work IDs
    are adjacent in the Parquet file.
    """

    id_columns = list(source.config.get("id_columns", []))
    text_columns = list(source.config.get("text_columns", []))
    if not id_columns or not text_columns:
        raise ValueError(f"{source.source_id}: section grouping requires id_columns and text_columns")
    temporary_root.mkdir(parents=True, exist_ok=True)
    descriptor, db_name = tempfile.mkstemp(prefix=f".{source.source_id}.", suffix=".sections.sqlite", dir=temporary_root)
    import os

    os.close(descriptor)
    Path(db_name).unlink(missing_ok=True)
    try:
        conn = sqlite3.connect(db_name)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                "CREATE TABLE sections (work_id TEXT NOT NULL, row_index INTEGER NOT NULL, "
                "text_field TEXT NOT NULL, text_value TEXT NOT NULL, row_json TEXT NOT NULL)"
            )
            pending: list[tuple[str, int, str, str, str]] = []
            for row_index, row in iter_artifact_rows(path):
                identity = first_nonempty(row, id_columns)
                selected = first_nonempty(row, text_columns)
                if identity is None or selected is None or not selected[1].strip():
                    continue
                pending.append(
                    (
                        identity[1],
                        row_index,
                        selected[0],
                        selected[1],
                        json.dumps(jsonable(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                    )
                )
                if len(pending) >= 4096:
                    conn.executemany("INSERT INTO sections VALUES (?, ?, ?, ?, ?)", pending)
                    conn.commit()
                    pending = []
            if pending:
                conn.executemany("INSERT INTO sections VALUES (?, ?, ?, ?, ?)", pending)
                conn.commit()
            conn.execute("CREATE INDEX sections_work_order ON sections(work_id, row_index)")
            cursor = conn.execute(
                "SELECT work_id, row_index, text_field, text_value, row_json "
                "FROM sections ORDER BY work_id, row_index"
            )
            current_id: str | None = None
            first_index = 0
            first_row: dict[str, Any] = {}
            fields: list[str] = []
            sections: list[str] = []
            for work_id, row_index, text_field, text_value, row_json in cursor:
                if current_id is not None and work_id != current_id:
                    first_row["_section_count"] = len(sections)
                    first_row["_section_text_fields"] = sorted(set(fields))
                    yield first_index, first_row, "section_grouped", "\n\n".join(sections)
                    fields, sections = [], []
                if work_id != current_id:
                    current_id = str(work_id)
                    first_index = int(row_index)
                    first_row = json.loads(str(row_json))
                fields.append(str(text_field))
                sections.append(str(text_value).strip())
            if current_id is not None:
                first_row["_section_count"] = len(sections)
                first_row["_section_text_fields"] = sorted(set(fields))
                yield first_index, first_row, "section_grouped", "\n\n".join(sections)
        finally:
            conn.close()
    finally:
        Path(db_name).unlink(missing_ok=True)
        Path(db_name + "-wal").unlink(missing_ok=True)
        Path(db_name + "-shm").unlink(missing_ok=True)


def _nested_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, Mapping):
        return ""
    preferred = ("text", "content", "body", "comment", "description", "title")
    pieces = [str(value[key]) for key in preferred if value.get(key) not in (None, "") and not isinstance(value[key], (list, dict))]
    return "\n\n".join(dict.fromkeys(pieces))


def expand_nested_row(row: Mapping[str, Any], source: SourceArtifact) -> Iterator[tuple[str, str, str]]:
    """Yield `(suffix, text_field, text)` without repeating parent context."""

    if source.source_id != "opengov_deliberations_v2":
        selected = first_nonempty(row, list(source.config.get("text_columns", [])) + list(source.config.get("alternate_text_columns", [])))
        if selected:
            yield "0", selected[0], selected[1]
        return
    emitted = False
    for field in source.config.get("text_columns", []):
        value = row.get(field)
        members = value if isinstance(value, list) else [value] if isinstance(value, Mapping) else []
        for index, member in enumerate(members):
            text = _nested_text(member)
            if text.strip():
                emitted = True
                member_id = member.get("id") if isinstance(member, Mapping) else None
                yield f"{field}:{member_id if member_id is not None else index}", field, text
            if isinstance(member, Mapping):
                for child_field in ("comments", "responses"):
                    children = member.get(child_field)
                    if not isinstance(children, list):
                        continue
                    for child_index, child in enumerate(children):
                        child_text = _nested_text(child)
                        if child_text.strip():
                            emitted = True
                            yield f"{field}:{index}:{child_field}:{child_index}", child_field, child_text
    if not emitted:
        selected = first_nonempty(row, list(source.config.get("text_columns", [])))
        if selected and selected[1].strip():
            yield "0", selected[0], selected[1]


def canonical_row(
    *,
    source: SourceArtifact,
    artifact_path: Path,
    artifact_row_index: int,
    raw_row: Mapping[str, Any],
    representation_suffix: str,
    text_field: str,
    raw_text: str,
    lineage_aliases: Mapping[str, Any],
    base_families: Mapping[str, str],
) -> dict[str, Any]:
    id_columns = list(source.config.get("id_columns", []))
    id_values = [str(raw_row[column]) for column in id_columns if raw_row.get(column) not in (None, "")]
    upstream_id = "|".join(id_values) if id_values else f"row:{artifact_row_index}"
    source_column = str(source.config.get("source_column", "source_dataset"))
    exact_source = str(raw_row.get(source_column) or source.repo_id)
    original = str(raw_text)
    normalized = normalize_text(original)
    original_hash = sha256_text(original)
    normalized_hash = sha256_text(normalized)
    relative_artifact = artifact_relative_path(source, artifact_path)
    source_row_id = f"{relative_artifact}:{artifact_row_index}:{representation_suffix}"
    source_doc_id = upstream_id if representation_suffix == "0" else f"{upstream_id}#{representation_suffix}"
    origin = "base" if source.source_id == "nanochat_base" or source.role == "base" else "candidate"
    source_family_id = (
        str(base_families.get(exact_source, exact_source))
        if origin == "base"
        else source.source_family_id
    )
    work_id = normalize_work_identifier(id_values[0] if id_values else source_doc_id)
    work_key = sha256_parts("full_cpt_work_key_v1", source_family_id, work_id)
    stable_uid = sha256_parts(
        "full_cpt_stable_uid_v1",
        source.repo_id,
        source.revision,
        relative_artifact,
        source_row_id,
        exact_source,
        text_field,
    )
    alias_id = (
        None
        if origin == "base"
        else choose_alias_id(raw_row, source.repo_id, exact_source, lineage_aliases)
    )
    excluded = {text_field, *source.config.get("text_columns", []), *source.config.get("alternate_text_columns", [])}
    metadata = compact_metadata(raw_row, excluded)
    title = raw_row.get("title") or raw_row.get("titlos")
    author = raw_row.get("author") or raw_row.get("creator")
    return {
        "source_id": source.source_id,
        "source_dataset": exact_source,
        "source_doc_id": source_doc_id,
        "text": normalized,
        "title": None if title is None else str(title),
        "author": None if author is None else str(author),
        "greek_badness_score": _optional_float(raw_row.get("greek_badness_score")),
        "mojibake_badness_score": _optional_float(raw_row.get("mojibake_badness_score")),
        "needs_ocr": _optional_bool(raw_row.get("needs_ocr")),
        "is_empty": not bool(normalized),
        "ocr_success": _optional_bool(raw_row.get("ocr_success")),
        "is_historical_or_polytonic": _optional_bool(raw_row.get("is_historical_or_polytonic")),
        "source_family_id": source_family_id,
        "acquisition_source_id": source.source_id,
        "source_repo_id": source.repo_id,
        "source_revision": source.revision,
        "source_artifact_path": relative_artifact,
        "source_row_id": source_row_id,
        "source_text_field": text_field,
        "original_text_sha256": original_hash,
        "normalized_text_sha256": normalized_hash,
        "stable_uid": stable_uid,
        "work_key": work_key,
        "work_id": work_id,
        "representation_generation": lineage_representation_generation(origin, source.config),
        "lineage_alias_id": alias_id,
        "source_metadata_json": metadata,
        "cleaning_profile": str(source.config.get("cleaning_profile", "base_canonical")),
        "structural_policy": str(source.config.get("structural_policy", "source_routed")),
        "training_eligibility": str(source.config.get("training_eligibility", "inherited_base")),
        "source_role": source.role,
    }


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    folded = str(value).strip().casefold()
    if folded in {"true", "1", "yes", "y"}:
        return True
    if folded in {"false", "0", "no", "n"}:
        return False
    return None


def canonical_schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("source_id", pa.string()),
            ("source_dataset", pa.string()),
            ("source_doc_id", pa.string()),
            ("text", pa.string()),
            ("title", pa.string()),
            ("author", pa.string()),
            ("greek_badness_score", pa.float64()),
            ("mojibake_badness_score", pa.float64()),
            ("needs_ocr", pa.bool_()),
            ("is_empty", pa.bool_()),
            ("ocr_success", pa.bool_()),
            ("is_historical_or_polytonic", pa.bool_()),
            ("source_family_id", pa.string()),
            ("acquisition_source_id", pa.string()),
            ("source_repo_id", pa.string()),
            ("source_revision", pa.string()),
            ("source_artifact_path", pa.string()),
            ("source_row_id", pa.string()),
            ("source_text_field", pa.string()),
            ("original_text_sha256", pa.string()),
            ("normalized_text_sha256", pa.string()),
            ("stable_uid", pa.string()),
            ("work_key", pa.string()),
            ("work_id", pa.string()),
            ("representation_generation", pa.string()),
            ("lineage_alias_id", pa.string()),
            ("source_metadata_json", pa.string()),
            ("cleaning_profile", pa.string()),
            ("structural_policy", pa.string()),
            ("training_eligibility", pa.string()),
            ("source_role", pa.string()),
        ]
    )


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def strip_html_markup(text: str) -> tuple[str, int]:
    """Conservative markup removal for explicitly web-routed sources."""

    script = re.compile(r"(?is)<(?:script|style|noscript)\b[^>]*>.*?</(?:script|style|noscript)>")
    tags = re.compile(r"(?s)<[^>]{1,1000}>")
    without_script, script_count = script.subn("\n", text)
    without_tags, tag_count = tags.subn(" ", without_script)
    decoded = html.unescape(without_tags)
    decoded = "\n".join(SPACE.sub(" ", line).strip() for line in decoded.splitlines())
    decoded = MULTI_BLANK.sub("\n\n\n", decoded).strip()
    return decoded, script_count + tag_count
