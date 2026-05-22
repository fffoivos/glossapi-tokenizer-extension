from __future__ import annotations

import concurrent.futures as cf
import gzip
import io
import importlib.util
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import sqlite3
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd
from blake3 import blake3
from huggingface_hub import hf_hub_download, list_repo_files

import glossapi_rs_noise


def _duckdb_connect_streaming() -> "duckdb.DuckDBPyConnection":
    """Open a duckdb connection that streams ORDER BY results to disk.

    Default duckdb behaviour (`preserve_insertion_order = true`) materializes
    the full result set in memory before writing — this OOMs on the 129 GB
    nanochat corpus during the apertus-drop COPY (verified by jobs 2334358 /
    2334476 on Clariden, both OOMed at the same `materialize_doc_key_excluded_mix_input`
    join+sort, at 305 GiB and 610 GiB respectively). With
    `preserve_insertion_order = false` and an explicit `ORDER BY` in the
    query, duckdb uses external sort (spills to `temp_directory`) and
    streams in bounded memory.

    Every duckdb connection in this module is opened via this helper so the
    streaming behaviour is uniform.
    """
    # NB: must use duckdb.connect() directly — calling _duckdb_connect_streaming
    # here would recurse forever (caught by job 2334826 RecursionError).
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")
    memory_limit = os.environ.get("DUCKDB_MEMORY_LIMIT")
    if memory_limit:
        con.execute(f"SET memory_limit = {sql_quote(memory_limit)}")
    temp_directory = os.environ.get("DUCKDB_TEMP_DIRECTORY")
    if temp_directory:
        Path(temp_directory).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = {sql_quote(temp_directory)}")
    threads = os.environ.get("DUCKDB_THREADS")
    if threads:
        con.execute(f"SET threads = {int(threads)}")
    return con


CODE_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = Path(os.environ.get("GLOSSAPI_RAW_ROOT", "/home/foivos/data/glossapi_raw"))
WORK_ROOT = Path(os.environ.get("GLOSSAPI_WORK_ROOT", str(CODE_ROOT)))
REEVAL_ROOT = WORK_ROOT / "reeval"
EXTERNAL_ROOT = RAW_ROOT / "external_hf"
DEFAULT_OUTPUT_ROOT = WORK_ROOT / "unified_corpus"
NANOCHAT_ROW_GROUP_ROWS = 2048
DEFAULT_DEDUP_POOL_FULL_INCLUDE_THRESHOLD = 0.05
DEDUP_ACTIONS = {"ignore", "annotate", "drop_intra", "drop_intra_and_inter"}
DEDUP_EXACT_STAGE_OPTIONS = {"strict_only", "strict_and_relaxed"}
DEDUP_INTER_DATASET_POLICIES = {"quality_first", "share_aware"}
SOURCE_MIX_FRACTION_MODES = {"of_group", "of_total"}
DEFAULT_STANDARD_BADNESS_LT = 60.0
DEFAULT_STANDARD_MOJIBAKE_LTE = 0.1
DEFAULT_STANDARD_GREEK_RATIO_GTE = 0.5
OPENSUBTITLES_EL_DATASET = "OPUS/OpenSubtitles-el-v2018"
OPENSUBTITLES_EL_URL = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/xml/el.zip"
OPENSUBTITLES_EL_FILENAME = "el.zip"


def load_reeval_module() -> Any:
    module_path = WORK_ROOT / "rust_reevaluate_pdf_datasets.py"
    spec = importlib.util.spec_from_file_location("glossapi_local_reeval", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load reevaluation module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reeval = load_reeval_module()

GLOSSAPI_INCLUDED_DATASETS = [
    "Sxolika_vivlia",
    "dimodis_logotexnia",
    "Ellinika_Keimena_Project_Gutenberg",
    "1000_prwta_xronia_ellhnikhs",
    "klasikh_arx_ell_grammateia",
    "Wikisource_Greek_texts",
    "Ekklisiastika_Keimena",
    "Apothetirio_Pergamos",
    "Apothetirio_Kallipos",
    "ellinika_dedomena_europaikou_koinovouliou",
    "eurlex-greek-legislation",
    "openarchives.gr",
    "opengov.gr-diaboyleuseis",
    "openbook_gr",
    "greek_phd",
]

EXTERNAL_DATASETS = [
    # Wave-2 (2026-04-26) external scope per user direction:
    # - keep finewiki (Greek Wikipedia from the FineWeb pipeline) and
    #   greek_legal_code (AI-team-UoA).
    # - drop HuggingFaceFW/finepdfs-edu — PDF residue post-cleaning is
    #   not trustworthy enough for tokenizer training input.
    # - drop OPUS OpenSubtitles — subtitle prose is not representative
    #   for a tokenizer training distribution.
    # The drop affects EXTERNAL_DATASETS only; the per-dataset
    # iter_*_rows / STREAM_DATASET_BUILDERS / FRAME_DATASET_BUILDERS
    # entries below stay available for ad-hoc / discovery use but are
    # never invoked during a `build` run as long as they're not in
    # this whitelist.
    "HuggingFaceFW/finewiki",
    "AI-team-UoA/greek_legal_code",
]

CANONICAL_COLUMNS = [
    "source_dataset",
    "source_doc_id",
    "text",
    "title",
    "author",
    "source_metadata_json",
    "is_historical_or_polytonic",
    "contains_math",
    "contains_latex",
    "greek_percentage",
    "latin_percentage",
    "polytonic_ratio",
    "table_ratio",
    "greek_badness_score",
    "len_greek",
    "mojibake_badness_score",
    "needs_ocr",
    "is_empty",
    "filter",
    "ocr_success",
    "quality_method",
    "reevaluated_at",
]

QUALITY_COLUMNS = [
    "greek_percentage",
    "latin_percentage",
    "polytonic_ratio",
    "table_ratio",
    "greek_badness_score",
    "len_greek",
    "mojibake_badness_score",
    "needs_ocr",
    "is_empty",
    "filter",
    "ocr_success",
    "quality_method",
    "reevaluated_at",
]

FLOAT_COLUMNS = [
    "greek_percentage",
    "latin_percentage",
    "polytonic_ratio",
    "table_ratio",
    "greek_badness_score",
    "mojibake_badness_score",
]

INTEGER_COLUMNS = [
    "len_greek",
]

BOOL_COLUMNS = [
    "is_historical_or_polytonic",
    "contains_math",
    "contains_latex",
    "needs_ocr",
    "is_empty",
    "ocr_success",
]

STRING_COLUMNS = [
    "source_dataset",
    "source_doc_id",
    "text",
    "title",
    "author",
    "source_metadata_json",
    "filter",
    "quality_method",
]

TIMESTAMP_COLUMNS = ["reevaluated_at"]

FORBIDDEN_SOURCE_METADATA_KEYS = {
    "filename",
    "filepath",
    "file_path",
    "original_filepath",
    "md_filename",
    "md_path",
    "source_doc_id",
    "source_jsonl",
    "row_id",
    "id",
}

URLISH_SOURCE_METADATA_KEYS = {
    "url",
    "link_url",
    "preferred_url",
    "document_url",
    "permanent_url",
    "handle_url",
    "el_html_link",
    "pdf_links_json",
}

URL_PLACEHOLDER_VALUES = {"na", "n/a", "none", "null", "nan", "-", "_"}

SAFE_EXACT_METADATA_DEDUP_FIELDS = {
    "Apothetirio_Pergamos": ("permanent_url",),
    "Wikisource_Greek_texts": ("url",),
    "openbook_gr": ("url",),
    "greek_phd": ("handle_url",),
    "HuggingFaceFW/finewiki": ("url", "__title__"),
    "HuggingFaceFW/finepdfs-edu": ("url",),
}

HISTORICAL_DATASETS = {
    "1000_prwta_xronia_ellhnikhs",
    "dimodis_logotexnia",
    "Ekklisiastika_Keimena",
    "klasikh_arx_ell_grammateia",
}

RUST_BADNESS_RELIABLE_DATASETS = {
    "Sxolika_vivlia",
    "Apothetirio_Pergamos",
    "Apothetirio_Kallipos",
    "ellinika_dedomena_europaikou_koinovouliou",
    "eurlex-greek-legislation",
    "openarchives.gr",
    "opengov.gr-diaboyleuseis",
    "openbook_gr",
    "greek_phd",
    "HuggingFaceFW/finewiki",
    "HuggingFaceFW/finepdfs-edu",
    "AI-team-UoA/greek_legal_code",
    OPENSUBTITLES_EL_DATASET,
}


@dataclass(frozen=True)
class SourceBuildResult:
    dataset_name: str
    path: Path
    row_count: int


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).replace("\x00", "").strip()


def clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return clean_scalar(value.tolist())
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return [clean_scalar(item) for item in value]
    if isinstance(value, Mapping):
        return {str(k): clean_scalar(v) for k, v in value.items() if clean_scalar(v) is not None}
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return value


def normalize_whitespace(value: str | None) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text:
        return None
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return text or None


def normalize_title(value: Any) -> str | None:
    if isinstance(value, list):
        value = " - ".join(str(item) for item in value if normalize_whitespace(str(item)))
    return normalize_whitespace(clean_scalar(value))


def normalize_author(value: Any) -> str | None:
    if isinstance(value, list):
        parts = [normalize_whitespace(str(item)) for item in value]
        parts = [item for item in parts if item]
        return "; ".join(parts) if parts else None
    return normalize_whitespace(clean_scalar(value))


def maybe_json_loads(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return {str(k): clean_scalar(v) for k, v in value.items() if clean_scalar(v) is not None}
    try:
        parsed = json.loads(str(value))
    except Exception:
        return {}
    if isinstance(parsed, dict):
        return {str(k): clean_scalar(v) for k, v in parsed.items() if clean_scalar(v) is not None}
    return {}


def select_metadata_fields(payload: Mapping[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    return {key: payload.get(key) for key in keys if key in payload}


def is_urlish_source_metadata_key(key: str) -> bool:
    lowered = str(key).lower()
    return lowered in URLISH_SOURCE_METADATA_KEYS or lowered.endswith("_url")


def normalize_placeholder_url(value: Any) -> Any:
    cleaned = clean_scalar(value)
    if cleaned is None:
        return None
    if isinstance(cleaned, str):
        normalized = normalize_whitespace(cleaned)
        if normalized is None:
            return None
        if normalized.lower() in URL_PLACEHOLDER_VALUES:
            return None
        return normalized
    if isinstance(cleaned, list):
        items = [normalize_placeholder_url(item) for item in cleaned]
        items = [item for item in items if item is not None]
        return items or None
    if isinstance(cleaned, Mapping):
        parsed = {str(key): normalize_placeholder_url(item) for key, item in cleaned.items()}
        parsed = {key: item for key, item in parsed.items() if item is not None}
        return parsed or None
    return cleaned


def metadata_json(payload: Mapping[str, Any]) -> str | None:
    cleaned = {}
    for key, value in payload.items():
        if key in FORBIDDEN_SOURCE_METADATA_KEYS:
            continue
        if is_urlish_source_metadata_key(str(key)):
            cleaned_value = normalize_placeholder_url(value)
        else:
            cleaned_value = clean_scalar(value)
        if cleaned_value is not None:
            cleaned[str(key)] = cleaned_value
    if not cleaned:
        return None
    return json.dumps(cleaned, ensure_ascii=False, sort_keys=True)


def normalize_dedup_token(value: Any) -> str | None:
    cleaned = clean_scalar(value)
    if cleaned is None:
        return None
    if not isinstance(cleaned, str):
        cleaned = json.dumps(cleaned, ensure_ascii=False, sort_keys=True)
    normalized = normalize_whitespace(cleaned)
    if normalized is None:
        return None
    return normalized.lower()


def exact_metadata_dedup_key(row: Mapping[str, Any]) -> tuple[str, ...] | None:
    dataset_name = str(row.get("source_dataset") or "")
    dedup_fields = SAFE_EXACT_METADATA_DEDUP_FIELDS.get(dataset_name)
    if not dedup_fields:
        return None
    payload = maybe_json_loads(row.get("source_metadata_json"))
    parts: list[str] = [dataset_name]
    for field in dedup_fields:
        if field == "__title__":
            value = normalize_dedup_token(row.get("title"))
        elif field == "__author__":
            value = normalize_dedup_token(row.get("author"))
        else:
            value = normalize_dedup_token(payload.get(field))
        if value is None:
            return None
        parts.append(value)
    return tuple(parts)


def dedup_row_priority(row: Mapping[str, Any]) -> tuple[float, float, float, int, str]:
    needs_ocr = row.get("needs_ocr")
    if needs_ocr is False:
        needs_ocr_rank = 0.0
    elif needs_ocr is True:
        needs_ocr_rank = 1.0
    else:
        needs_ocr_rank = 0.5
    is_empty = row.get("is_empty")
    if is_empty is False:
        empty_rank = 0.0
    elif is_empty is True:
        empty_rank = 1.0
    else:
        empty_rank = 0.5
    try:
        badness_rank = float(row.get("greek_badness_score"))
    except Exception:
        badness_rank = float("inf")
    text_length_rank = -len(clean_text(row.get("text")))
    source_doc_id_rank = normalize_dedup_token(row.get("source_doc_id")) or ""
    return (needs_ocr_rank, empty_rank, badness_rank, text_length_rank, source_doc_id_rank)


def iter_exact_metadata_dedup(row_iter: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    passthrough_rows: list[tuple[int, dict[str, Any]]] = []
    keyed_rows: dict[tuple[str, ...], tuple[int, tuple[float, float, float, int, str], dict[str, Any]]] = {}
    for row_index, row in enumerate(row_iter):
        dedup_key = exact_metadata_dedup_key(row)
        if dedup_key is None:
            passthrough_rows.append((row_index, row))
            continue
        priority = dedup_row_priority(row)
        previous = keyed_rows.get(dedup_key)
        if previous is None or priority < previous[1]:
            first_seen_index = row_index if previous is None else previous[0]
            keyed_rows[dedup_key] = (first_seen_index, priority, row)
    ordered_rows = passthrough_rows + [(row_index, row) for row_index, _, row in keyed_rows.values()]
    for _, row in sorted(ordered_rows, key=lambda item: item[0]):
        yield row


def maybe_iter_exact_metadata_dedup(dataset_name: str, row_iter: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    if dataset_name not in SAFE_EXACT_METADATA_DEDUP_FIELDS:
        return row_iter
    return iter_exact_metadata_dedup(row_iter)


LATEX_RE = re.compile(
    r"(\\begin\{(?:equation|align|gather|multline|matrix|bmatrix|pmatrix)\}|\\[a-zA-Z]+(?:\[[^\]]+\])?(?:\{[^}]+\})?|\\\(|\\\)|\\\[|\\\]|\$\$)",
    re.MULTILINE,
)
MATH_RE = re.compile(r"[∑∫√∞≈≠≤≥±×÷∂∇∈∉∩∪⊂⊆⊕⊗≃≅∀∃]")


def contains_latex(text: str) -> bool:
    return bool(LATEX_RE.search(text))


def contains_math(text: str, *, source_hint: bool = False, formula_total: Any = None) -> bool:
    if source_hint:
        return True
    if formula_total not in (None, 0, 0.0):
        return True
    if contains_latex(text):
        return True
    return bool(MATH_RE.search(text))


def derive_historical_flag(dataset_name: str, *, polytonic_ratio: Any = None, author_year: Any = None) -> bool:
    if dataset_name in HISTORICAL_DATASETS:
        return True
    try:
        if polytonic_ratio is not None and float(polytonic_ratio) >= 0.02:
            return True
    except Exception:
        pass
    if author_year not in (None, ""):
        try:
            return int(float(author_year)) < 1900
        except Exception:
            return False
    return False


def derive_greek_percentage(existing_value: Any = None, latin_percentage: Any = None) -> float | None:
    if existing_value not in (None, ""):
        try:
            return float(existing_value)
        except Exception:
            pass
    if latin_percentage not in (None, ""):
        try:
            return max(0.0, 100.0 - float(latin_percentage))
        except Exception:
            return None
    return None


def extract_first_meaningful_line(text: str) -> str | None:
    for raw_line in text.splitlines():
        line = normalize_whitespace(raw_line.strip("#*| -\t"))
        if not line:
            continue
        if line in {"European flag", "EL Σειρά L", "ΚΕΙΜΕΝΑ ΠΟΥ ΕΓΚΡΙΘΗΚΑΝ"}:
            continue
        if re.fullmatch(r"P\d+_TA\(\d{4}\)\d{4}", line):
            continue
        if re.fullmatch(r"\d{4}/\d+", line):
            continue
        if line.startswith("PE") and any(ch.isdigit() for ch in line):
            continue
        return line
    return None


def extract_europarl_code(text: str) -> str | None:
    match = re.search(r"P(\d+)_TA\((\d{4})\)(\d{4})", text[:2500])
    if not match:
        return None
    plen, year, seq = match.groups()
    return f"TA-{plen}-{year}-{seq}"


def extract_europarl_title(text: str) -> str | None:
    match = re.search(r"(?m)^Τίτλος:\s*(.+)$", text)
    if match:
        return normalize_title(match.group(1))
    lines = [normalize_whitespace(line) for line in text.splitlines()[:30]]
    lines = [line for line in lines if line]
    skip_prefixes = (
        "ΚΕΙΜΕΝΑ ΠΟΥ ΕΓΚΡΙΘΗΚΑΝ",
        "Επιτροπή:",
        "Αναφορά Διαδικασίας:",
        "Τύπος Διαδικασίας:",
        "Ημερομηνία:",
        "Αναφορά Εγγράφου:",
        "Εσωτερική Επιτροπή:",
        "Τύπος Τίτλου:",
        "Υπεύθυνη Επιτροπή:",
    )
    for line in lines:
        if any(line.startswith(prefix) for prefix in skip_prefixes):
            continue
        if re.fullmatch(r"P\d+_TA\(\d{4}\)\d{4}", line):
            continue
        if line.startswith("PE") and any(ch.isdigit() for ch in line):
            continue
        if len(line) >= 16:
            return line
    return None


def extract_eurlex_title(text: str) -> str | None:
    lines = [normalize_whitespace(line) for line in text.splitlines()[:20]]
    lines = [line for line in lines if line]
    collected: list[str] = []
    for line in lines:
        if line.startswith("|"):
            continue
        if line in {"---"}:
            continue
        collected.append(line)
        if len(collected) >= 4:
            break
    for line in collected:
        if any(ch.islower() for ch in line):
            return line
    if len(collected) >= 3:
        return " ".join(collected[2:4]).strip()
    return collected[0] if collected else None


def download_selected_external_sources(force: bool = False) -> dict[str, list[Path]]:
    ensure_dir(EXTERNAL_ROOT)
    # Wave-2: gate every download by EXTERNAL_DATASETS so dropping a
    # dataset from the whitelist also stops it from being downloaded.
    # Pre-wave-2 the download list was hardcoded and downloaded
    # finepdfs-edu / OpenSubtitles even after they were dropped from
    # the build whitelist.
    repo_patterns = {
        "HuggingFaceFW/finewiki": ["data/elwiki/*.parquet", "README.md"],
        "HuggingFaceFW/finepdfs-edu": ["data/ell_Grek/train/*.parquet", "README.md"],
        "AI-team-UoA/greek_legal_code": ["volume/*.parquet", "chapter/*.parquet", "subject/*.parquet", "README.md"],
    }
    downloaded: dict[str, list[Path]] = {}
    for repo_id, patterns in repo_patterns.items():
        if repo_id not in EXTERNAL_DATASETS:
            continue
        repo_root = EXTERNAL_ROOT / repo_id.replace("/", "__")
        ensure_dir(repo_root)
        files = list_repo_files(repo_id, repo_type="dataset")
        matched = []
        for pattern in patterns:
            regex = re.compile("^" + pattern.replace(".", r"\.").replace("*", ".*") + "$")
            matched.extend(file for file in files if regex.match(file))
        paths = []
        for file in sorted(set(matched)):
            local_path = repo_root / file
            if force and local_path.exists():
                local_path.unlink()
            if not local_path.exists():
                ensure_dir(local_path.parent)
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=file,
                    local_dir=repo_root,
                    local_dir_use_symlinks=False,
                )
            paths.append(local_path)
        downloaded[repo_id] = paths
    if OPENSUBTITLES_EL_DATASET in EXTERNAL_DATASETS:
        opensubtitles_path = ensure_opensubtitles_el_source(force=force)
        downloaded[OPENSUBTITLES_EL_DATASET] = [opensubtitles_path]
    return downloaded


def ensure_opensubtitles_el_source(*, force: bool = False) -> Path:
    root = external_repo_root(OPENSUBTITLES_EL_DATASET)
    ensure_dir(root)
    path = root / OPENSUBTITLES_EL_FILENAME
    if force and path.exists():
        path.unlink()
    if not path.exists():
        with urllib.request.urlopen(OPENSUBTITLES_EL_URL) as response, path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    return path


def strip_known_suffixes(value: str, suffixes: tuple[str, ...]) -> str:
    result = value
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if result.endswith(suffix):
                result = result[: -len(suffix)]
                changed = True
    return result


def normalize_opus_sentence_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def xml_local_name(tag: Any) -> str:
    return str(tag).rsplit("}", 1)[-1].lower()


def opus_tokens_to_text(tokens: list[str]) -> str:
    closers = {".", ",", ";", ":", "!", "?", ")", "]", "}", "%", "…", "...", "»"}
    apostrophes = {"'", "’"}
    openers = {"(", "[", "{", "«"}
    text = ""
    for token in tokens:
        piece = clean_text(token)
        if not piece:
            continue
        if piece in closers or piece in apostrophes:
            text = text.rstrip() + piece
            continue
        if piece in openers:
            text = f"{text} {piece}".strip()
            continue
        if text.endswith(tuple(openers)):
            text += piece
            continue
        text = f"{text} {piece}".strip()
    return normalize_opus_sentence_text(text)


def opus_xml_document_text(root: ET.Element) -> str:
    sentences: list[str] = []
    for element in root.iter():
        tag = xml_local_name(element.tag)
        if tag != "s":
            continue
        tokens = [clean_text(token.text) for token in element if xml_local_name(token.tag) == "w"]
        text = opus_tokens_to_text(tokens) if any(tokens) else normalize_opus_sentence_text(" ".join(part for part in element.itertext()))
        if text:
            sentences.append(text)
    if sentences:
        return "\n".join(sentences)
    return normalize_opus_sentence_text(" ".join(part for part in root.itertext()))


def maybe_int(value: Any) -> int | None:
    text = normalize_whitespace(clean_scalar(value))
    if text is None or not re.fullmatch(r"[+-]?\d+", text):
        return None
    try:
        return int(text)
    except Exception:
        return None


def maybe_float(value: Any) -> float | None:
    text = normalize_whitespace(clean_scalar(value))
    if text is None:
        return None
    try:
        return float(text)
    except Exception:
        return None


def quality_value_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def decode_noise_score_row(score_row: Any) -> dict[str, Any]:
    row = list(score_row)
    if len(row) < 6:
        raise ValueError(f"unexpected glossapi_rs_noise detailed row length: {len(row)}")
    return {
        "md_path": str(row[0]),
        "greek_badness_score": float(row[1]),
        "latin_percentage": float(row[2]),
        "table_ratio": float(row[3]),
        "polytonic_ratio": float(row[4]),
        "len_greek": int(row[5]),
    }


def opus_xml_metadata(root: ET.Element, filename: str, base_metadata: Mapping[str, Any]) -> dict[str, Any]:
    metadata = {
        "language": base_metadata.get("language"),
        "zip_member": filename,
    }

    document_id = normalize_whitespace(clean_scalar(root.attrib.get("id")))
    if document_id is not None:
        metadata["document_id"] = document_id

    int_fields = {
        "subtitle_blocks",
        "source_duration",
        "source_year",
    }
    float_fields = {"subtitle_confidence"}
    allowed_fields = {
        "subtitle_language",
        "subtitle_confidence",
        "subtitle_blocks",
        "subtitle_duration",
        "subtitle_date",
        "source_year",
        "source_duration",
        "source_genre",
        "source_country",
        "source_original",
    }

    meta = next((child for child in root if xml_local_name(child.tag) == "meta"), None)
    if meta is None:
        return metadata
    for section in meta:
        section_name = xml_local_name(section.tag)
        if not section_name:
            continue
        for field in section:
            field_name = xml_local_name(field.tag)
            if not field_name:
                continue
            key = f"{section_name}_{field_name}"
            if key not in allowed_fields:
                continue
            raw_value = normalize_whitespace(clean_scalar(field.text))
            if raw_value is None:
                continue
            if key in int_fields:
                parsed_int = maybe_int(raw_value)
                metadata[key] = parsed_int if parsed_int is not None else raw_value
                continue
            if key in float_fields:
                parsed_float = maybe_float(raw_value)
                metadata[key] = parsed_float if parsed_float is not None else raw_value
                continue
            metadata[key] = raw_value
    return metadata


def load_quality(dataset_name: str) -> pd.DataFrame:
    path = REEVAL_ROOT / dataset_name / "document_quality.parquet"
    df = pd.read_parquet(path).copy()
    keep = [col for col in ["source_doc_id", *QUALITY_COLUMNS] if col in df.columns]
    return df[keep]


def attach_quality(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    quality = load_quality(dataset_name)
    merged = df.merge(quality, on="source_doc_id", how="left")
    return merged


def load_local_docs(dataset_name: str, columns: list[str] | None = None) -> pd.DataFrame:
    path = REEVAL_ROOT / dataset_name / "document_level.parquet"
    return pd.read_parquet(path, columns=columns).copy()


def finalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["text"] = frame["text"].fillna("").astype(str)
    frame["title"] = frame["title"].map(normalize_title)
    frame["author"] = frame["author"].map(normalize_author)
    frame["contains_latex"] = frame["contains_latex"].fillna(False).astype(bool)
    frame["contains_math"] = frame["contains_math"].fillna(False).astype(bool)
    frame["is_historical_or_polytonic"] = frame["is_historical_or_polytonic"].fillna(False).astype(bool)
    frame["greek_percentage"] = frame.apply(
        lambda row: derive_greek_percentage(row.get("greek_percentage"), row.get("latin_percentage")),
        axis=1,
    )
    for col in CANONICAL_COLUMNS:
        if col not in frame.columns:
            frame[col] = None
    for col in STRING_COLUMNS:
        frame[col] = frame[col].where(frame[col].notna(), None)
    for col in FLOAT_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    for col in INTEGER_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("Int64")
    for col in BOOL_COLUMNS:
        if col in {"is_historical_or_polytonic", "contains_math", "contains_latex"}:
            frame[col] = frame[col].fillna(False).astype(bool)
        else:
            frame[col] = frame[col].astype("boolean")
    for col in TIMESTAMP_COLUMNS:
        frame[col] = pd.to_datetime(frame[col], errors="coerce", utc=True)
    return frame[CANONICAL_COLUMNS]


CANONICAL_ARROW_SCHEMA = pa.schema(
    [
        pa.field("source_dataset", pa.large_string()),
        pa.field("source_doc_id", pa.large_string()),
        pa.field("text", pa.large_string()),
        pa.field("title", pa.large_string()),
        pa.field("author", pa.large_string()),
        pa.field("source_metadata_json", pa.large_string()),
        pa.field("is_historical_or_polytonic", pa.bool_()),
        pa.field("contains_math", pa.bool_()),
        pa.field("contains_latex", pa.bool_()),
        pa.field("greek_percentage", pa.float64()),
        pa.field("latin_percentage", pa.float64()),
        pa.field("polytonic_ratio", pa.float64()),
        pa.field("table_ratio", pa.float64()),
        pa.field("greek_badness_score", pa.float64()),
        pa.field("len_greek", pa.int64()),
        pa.field("mojibake_badness_score", pa.float64()),
        pa.field("needs_ocr", pa.bool_()),
        pa.field("is_empty", pa.bool_()),
        pa.field("filter", pa.large_string()),
        pa.field("ocr_success", pa.bool_()),
        pa.field("quality_method", pa.large_string()),
        pa.field("reevaluated_at", pa.timestamp("us", tz="UTC")),
    ]
)


def batch_score_missing_quality(rows: list[dict[str, Any]], dataset_name: str) -> list[dict[str, Any]]:
    missing = [
        row
        for row in rows
        if quality_value_missing(row.get("greek_badness_score"))
    ]
    if not missing:
        return rows
    with tempfile.TemporaryDirectory(prefix=f"score_{dataset_name.replace('/', '_')}_") as tmpdir:
        tmp_root = Path(tmpdir)
        filename_map: dict[str, dict[str, Any]] = {}
        for idx, row in enumerate(missing):
            name = f"doc_{idx:06d}.md"
            (tmp_root / name).write_text(row["text"].strip() + "\n", encoding="utf-8")
            filename_map[name] = row
        scored = glossapi_rs_noise.score_markdown_directory_detailed(str(tmp_root), None)
        reevaluated_at = pd.Timestamp.now("UTC")
        for score_row in scored:
            metrics = decode_noise_score_row(score_row)
            name = Path(str(metrics["md_path"])).name
            row = filename_map[name]
            if quality_value_missing(row.get("greek_badness_score")):
                row["greek_badness_score"] = float(metrics["greek_badness_score"])
            if quality_value_missing(row.get("latin_percentage")):
                row["latin_percentage"] = float(metrics["latin_percentage"])
            if quality_value_missing(row.get("table_ratio")):
                row["table_ratio"] = float(metrics["table_ratio"])
            if quality_value_missing(row.get("polytonic_ratio")):
                row["polytonic_ratio"] = float(metrics["polytonic_ratio"])
            if quality_value_missing(row.get("len_greek")):
                row["len_greek"] = int(metrics["len_greek"])
            if quality_value_missing(row.get("greek_percentage")):
                row["greek_percentage"] = derive_greek_percentage(None, row["latin_percentage"])
            if quality_value_missing(row.get("quality_method")):
                row["quality_method"] = "glossapi_rs_noise"
            if quality_value_missing(row.get("reevaluated_at")):
                row["reevaluated_at"] = reevaluated_at
    return rows


def write_parquet_rows(path: Path, row_iter: Iterable[dict[str, Any]], *, batch_size: int, score_missing_quality: bool) -> int:
    ensure_dir(path.parent)
    writer: pq.ParquetWriter | None = None
    batch: list[dict[str, Any]] = []
    row_count = 0
    for row in row_iter:
        batch.append(row)
        if len(batch) >= batch_size:
            if score_missing_quality:
                batch = batch_score_missing_quality(batch, row["source_dataset"])
            frame = finalize_frame(pd.DataFrame(batch))
            table = pa.Table.from_pandas(frame, schema=CANONICAL_ARROW_SCHEMA, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
            row_count += len(batch)
            batch = []
    if batch:
        if score_missing_quality:
            batch = batch_score_missing_quality(batch, batch[0]["source_dataset"])
        frame = finalize_frame(pd.DataFrame(batch))
        table = pa.Table.from_pandas(frame, schema=CANONICAL_ARROW_SCHEMA, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema, compression="zstd")
        writer.write_table(table)
        row_count += len(batch)
    if writer is not None:
        writer.close()
    return row_count


def write_nanochat_text_shard(path: Path, rows: list[dict[str, str]], *, row_group_rows: int = NANOCHAT_ROW_GROUP_ROWS) -> None:
    ensure_dir(path.parent)
    table = pa.Table.from_pydict(
        {"text": [str(row["text"]) for row in rows]},
        schema=pa.schema([("text", pa.string())]),
    )
    pq.write_table(
        table,
        path,
        compression="zstd",
        row_group_size=max(1, int(row_group_rows)),
    )


def dataframe_rows(df: pd.DataFrame) -> Iterator[dict[str, Any]]:
    yield from df.to_dict("records")


def build_sxolika() -> pd.DataFrame:
    df = load_local_docs("Sxolika_vivlia")
    df["title"] = df["titlos"]
    df["author"] = None
    df["source_metadata_json"] = [
        metadata_json({"mathima": row["mathima"], "taxis": row["taxis"], "kateuthinsi": row["kateuthinsi"]})
        for _, row in df.iterrows()
    ]
    df["contains_latex"] = df["text"].map(contains_latex)
    df["contains_math"] = df.apply(lambda row: contains_math(row["text"]), axis=1)
    df["is_historical_or_polytonic"] = False
    return finalize_frame(df)


def build_dimodis() -> pd.DataFrame:
    df = load_local_docs("dimodis_logotexnia")
    df["title"] = df["title"]
    df["author"] = None
    df["source_metadata_json"] = None
    df["contains_latex"] = df["text"].map(contains_latex)
    df["contains_math"] = df["text"].map(contains_math)
    df["is_historical_or_polytonic"] = True
    return finalize_frame(df)


def build_gutenberg() -> pd.DataFrame:
    df = load_local_docs("Ellinika_Keimena_Project_Gutenberg")
    df["title"] = df["Title"]
    df["author"] = df["Author"]
    df["source_metadata_json"] = [
        metadata_json(
            {
                "Author Year": row["Author Year"],
                "Translator": row["Translator"],
                "Translation Year": row["Translation Year"],
                "Variety": row["Variety"],
            }
        )
        for _, row in df.iterrows()
    ]
    df["contains_latex"] = df["text"].map(contains_latex)
    df["contains_math"] = df["text"].map(contains_math)
    df["is_historical_or_polytonic"] = False
    return finalize_frame(df)


def build_first_1k() -> pd.DataFrame:
    df = load_local_docs("1000_prwta_xronia_ellhnikhs")
    df["title"] = df["title"]
    df["author"] = df["author"]
    df["source_metadata_json"] = None
    df["contains_latex"] = df["text"].map(contains_latex)
    df["contains_math"] = df["text"].map(contains_math)
    df["is_historical_or_polytonic"] = True
    return finalize_frame(df)


def build_klasikh() -> pd.DataFrame:
    df = load_local_docs("klasikh_arx_ell_grammateia")
    df["title"] = df["title"]
    df["author"] = df["author"]
    df["source_metadata_json"] = None
    df["contains_latex"] = df["text"].map(contains_latex)
    df["contains_math"] = df["text"].map(contains_math)
    df["is_historical_or_polytonic"] = True
    return finalize_frame(df)


def build_wikisource() -> pd.DataFrame:
    df = load_local_docs("Wikisource_Greek_texts")
    df["title"] = df["title"]
    df["author"] = df["author"]
    df["source_metadata_json"] = [
        metadata_json({"author_year": row["author_year"], "url": row["url"]})
        for _, row in df.iterrows()
    ]
    df["contains_latex"] = df["text"].map(contains_latex)
    df["contains_math"] = df["text"].map(contains_math)
    df["is_historical_or_polytonic"] = df.apply(
        lambda row: derive_historical_flag(
            "Wikisource_Greek_texts",
            polytonic_ratio=row.get("polytonic_ratio"),
            author_year=row.get("author_year"),
        ),
        axis=1,
    )
    return finalize_frame(df)


def build_ekklisiastika() -> pd.DataFrame:
    df = load_local_docs("Ekklisiastika_Keimena")
    df["title"] = df["titlos"]
    df["author"] = None
    df["source_metadata_json"] = [
        metadata_json({"katigoria": row["katigoria"], "ypokatigoria": row["ypokatigoria"]})
        for _, row in df.iterrows()
    ]
    df["contains_latex"] = df["text"].map(contains_latex)
    df["contains_math"] = df["text"].map(contains_math)
    df["is_historical_or_polytonic"] = True
    return finalize_frame(df)


def build_pergamos() -> pd.DataFrame:
    df = load_local_docs("Apothetirio_Pergamos")
    df["title"] = df["title"]
    df["author"] = df["author"]
    df["source_metadata_json"] = [
        metadata_json(
            {
                "department": row["department"],
                "submission_date": row["submission_date"],
                "year": row["year"],
                "language": row["language"],
                "abstract": row["abstract"],
                "permanent_url": row["permanent_url"],
                "supervisors": row["supervisors"],
            }
        )
        for _, row in df.iterrows()
    ]
    df["contains_latex"] = df["text"].map(contains_latex)
    df["contains_math"] = df["text"].map(contains_math)
    df["is_historical_or_polytonic"] = False
    return finalize_frame(df)


def kallipos_metadata_payload(raw_json: Any, document_type: Any) -> tuple[str | None, str | None, str | None]:
    payload = maybe_json_loads(raw_json)
    title = None
    bib = payload.get("Βιβλιογραφική Αναφορά")
    if isinstance(bib, list) and len(bib) >= 2:
        title = normalize_title(bib[1])
    if not title:
        title = normalize_title(payload.get("Τίτλος"))
    author = normalize_author(payload.get("Συγγραφείς"))
    meta_payload = select_metadata_fields(
        payload,
        [
            "Υπότιτλος",
            "Θεματικές Κατηγορίες",
            "Λέξεις-κλειδιά",
            "Περίληψη",
            "Τύπος",
            "Γλώσσα",
            "DOI",
            "Βιβλιογραφική Αναφορά",
        ],
    )
    if document_type is not None:
        meta_payload.setdefault("document_type", clean_scalar(document_type))
    return title, author, metadata_json(meta_payload)


def build_kallipos() -> pd.DataFrame:
    df = load_local_docs("Apothetirio_Kallipos")
    titles = []
    authors = []
    metadata_rows = []
    for _, row in df.iterrows():
        title, author, payload = kallipos_metadata_payload(row["metadata_extracted"], row["document_type"])
        titles.append(title)
        authors.append(author)
        metadata_rows.append(payload)
    df["title"] = titles
    df["author"] = authors
    df["source_metadata_json"] = metadata_rows
    df["contains_latex"] = df["text"].map(contains_latex)
    df["contains_math"] = df["text"].map(contains_math)
    df["is_historical_or_polytonic"] = False
    return finalize_frame(df)


def build_europarl() -> pd.DataFrame:
    df = load_local_docs("ellinika_dedomena_europaikou_koinovouliou")
    meta = pd.read_parquet(RAW_ROOT / "hf" / "ellinika_dedomena_europaikou_koinovouliou" / "metadata.parquet").copy()
    meta["europarl_code"] = meta["preferred_url"].str.extract(r"/(TA-\d-\d{4}-\d{4})/")
    df["europarl_code"] = df["text"].map(extract_europarl_code)
    df = df.merge(
        meta[
            [
                "europarl_code",
                "title",
                "title_dcterms",
                "doc_type",
                "year",
                "link_url",
                "format",
                "available_formats",
                "preferred_format",
                "preferred_url",
            ]
        ],
        on="europarl_code",
        how="left",
    )
    df["title"] = df["title"].fillna(df["text"].map(extract_europarl_title))
    df["author"] = None
    df["source_metadata_json"] = [
        metadata_json(
            {
                "doc_type": row["doc_type"],
                "year": row["year"],
                "link_url": row["link_url"],
                "preferred_format": row["preferred_format"],
                "preferred_url": row["preferred_url"],
            }
        )
        for _, row in df.iterrows()
    ]
    df["contains_latex"] = df["text"].map(contains_latex)
    df["contains_math"] = df["text"].map(contains_math)
    df["is_historical_or_polytonic"] = False
    return finalize_frame(df)


def build_eurlex() -> pd.DataFrame:
    df = load_local_docs("eurlex-greek-legislation")
    meta = pd.read_parquet(RAW_ROOT / "hf" / "eurlex-greek-legislation" / "eurlex_legislation_data.parquet").copy()
    meta["row_index"] = meta.index.astype(int)
    df["row_index"] = df["source_doc_id"].str.extract(r"(\d+)").astype(int)
    df = df.merge(meta, on="row_index", how="left")
    df["title"] = df["text"].map(extract_eurlex_title).fillna(df["title"])
    df["author"] = None
    df["source_metadata_json"] = [
        metadata_json(
            {
                "document_url": row["document_url"],
                "el_html_link": row["el_html_link"],
                "category_id": row["category_id"],
                "category_title": row["category_title"],
                "category_acts_count": row["category_acts_count"],
            }
        )
        for _, row in df.iterrows()
    ]
    df["contains_latex"] = df["text"].map(contains_latex)
    df["contains_math"] = df["text"].map(contains_math)
    df["is_historical_or_polytonic"] = False
    return finalize_frame(df)


def iter_europarl_rows() -> Iterator[dict[str, Any]]:
    meta = pd.read_parquet(RAW_ROOT / "hf" / "ellinika_dedomena_europaikou_koinovouliou" / "metadata.parquet").copy()
    meta["europarl_code"] = meta["preferred_url"].str.extract(r"/(TA-\d-\d{4}-\d{4})/")
    meta = meta.dropna(subset=["europarl_code"]).drop_duplicates("europarl_code", keep="first")
    meta_by_code = meta.set_index("europarl_code").to_dict("index")
    parquet = pq.ParquetFile(REEVAL_ROOT / "ellinika_dedomena_europaikou_koinovouliou" / "document_level.parquet")
    for batch in parquet.iter_batches(batch_size=128):
        frame = batch.to_pandas()
        for row in frame.to_dict("records"):
            text = clean_text(row.get("text"))
            europarl_code = extract_europarl_code(text)
            meta_row = meta_by_code.get(europarl_code, {})
            title = normalize_title(meta_row.get("title")) or extract_europarl_title(text)
            yield {
                "source_dataset": row["source_dataset"],
                "source_doc_id": row["source_doc_id"],
                "text": text,
                "title": title,
                "author": None,
                "source_metadata_json": metadata_json(
                    {
                        "doc_type": meta_row.get("doc_type"),
                        "year": meta_row.get("year"),
                        "link_url": meta_row.get("link_url"),
                        "preferred_format": meta_row.get("preferred_format"),
                        "preferred_url": meta_row.get("preferred_url"),
                    }
                ),
                "is_historical_or_polytonic": False,
                "contains_latex": contains_latex(text),
                "contains_math": contains_math(text),
                "greek_percentage": derive_greek_percentage(None, row.get("latin_percentage")),
                "latin_percentage": row.get("latin_percentage"),
                "polytonic_ratio": row.get("polytonic_ratio"),
                "table_ratio": row.get("table_ratio"),
                "greek_badness_score": row.get("greek_badness_score"),
                "mojibake_badness_score": row.get("mojibake_badness_score"),
                "needs_ocr": row.get("needs_ocr"),
                "is_empty": row.get("is_empty"),
                "filter": row.get("filter"),
                "ocr_success": row.get("ocr_success"),
                "quality_method": row.get("quality_method"),
                "reevaluated_at": row.get("reevaluated_at"),
            }


def iter_eurlex_rows() -> Iterator[dict[str, Any]]:
    meta = pd.read_parquet(RAW_ROOT / "hf" / "eurlex-greek-legislation" / "eurlex_legislation_data.parquet").copy()
    meta["row_index"] = meta.index.astype(int)
    meta_by_index = meta.set_index("row_index").to_dict("index")
    parquet = pq.ParquetFile(REEVAL_ROOT / "eurlex-greek-legislation" / "document_level.parquet")
    for batch in parquet.iter_batches(batch_size=128):
        frame = batch.to_pandas()
        for row in frame.to_dict("records"):
            text = clean_text(row.get("text"))
            match = re.search(r"(\d+)$", str(row.get("source_doc_id")))
            row_index = int(match.group(1)) if match else None
            meta_row = meta_by_index.get(row_index, {})
            yield {
                "source_dataset": row["source_dataset"],
                "source_doc_id": row["source_doc_id"],
                "text": text,
                "title": extract_eurlex_title(text) or normalize_title(meta_row.get("title")),
                "author": None,
                "source_metadata_json": metadata_json(
                    {
                        "document_url": meta_row.get("document_url"),
                        "el_html_link": meta_row.get("el_html_link"),
                        "category_id": meta_row.get("category_id"),
                        "category_title": meta_row.get("category_title"),
                        "category_acts_count": meta_row.get("category_acts_count"),
                    }
                ),
                "is_historical_or_polytonic": False,
                "contains_latex": contains_latex(text),
                "contains_math": contains_math(text),
                "greek_percentage": derive_greek_percentage(None, row.get("latin_percentage")),
                "latin_percentage": row.get("latin_percentage"),
                "polytonic_ratio": row.get("polytonic_ratio"),
                "table_ratio": row.get("table_ratio"),
                "greek_badness_score": row.get("greek_badness_score"),
                "mojibake_badness_score": row.get("mojibake_badness_score"),
                "needs_ocr": row.get("needs_ocr"),
                "is_empty": row.get("is_empty"),
                "filter": row.get("filter"),
                "ocr_success": row.get("ocr_success"),
                "quality_method": row.get("quality_method"),
                "reevaluated_at": row.get("reevaluated_at"),
            }


def build_opengov() -> pd.DataFrame:
    df = load_local_docs("opengov.gr-diaboyleuseis")
    db_path = RAW_ROOT / "hf" / "opengov.gr-diaboyleuseis" / "deliberation_data_gr_MIGRATED_FRESH_20250602170747.db"
    query = """
        SELECT
            d.id,
            d.consultation_id,
            c.title AS consultation_title,
            c.start_date,
            c.end_date,
            c.total_comments,
            c.accepted_comments,
            m.name AS ministry_name
        FROM documents d
        LEFT JOIN consultations c ON c.id = d.consultation_id
        LEFT JOIN ministries m ON m.id = c.ministry_id
    """
    with sqlite3.connect(db_path) as con:
        meta = pd.read_sql_query(query, con)
    df = df.merge(meta, on="id", how="left")
    df["title"] = df["title"]
    df["author"] = None
    df["source_metadata_json"] = [
        metadata_json(
            {
                "type": row["type"],
                "url": row["url"],
                "consultations.title": row["consultation_title"],
                "consultations.start_date": row["start_date"],
                "consultations.end_date": row["end_date"],
                "consultations.total_comments": row["total_comments"],
                "consultations.accepted_comments": row["accepted_comments"],
                "ministries.name": row["ministry_name"],
            }
        )
        for _, row in df.iterrows()
    ]
    df["contains_latex"] = df["text"].map(contains_latex)
    df["contains_math"] = df["text"].map(contains_math)
    df["is_historical_or_polytonic"] = False
    return finalize_frame(df)


def build_openbook() -> pd.DataFrame:
    df = load_local_docs("openbook_gr")
    df["title"] = None
    df["author"] = None
    df["source_metadata_json"] = [
        metadata_json(
            {
                "url": row["url"],
                "created_at": row["created_at"],
                "modified_at": row["modified_at"],
                "issues": row["issues"],
            }
        )
        for _, row in df.iterrows()
    ]
    df["contains_latex"] = df["text"].map(contains_latex)
    df["contains_math"] = df["text"].map(contains_math)
    df["is_historical_or_polytonic"] = False
    return finalize_frame(df)


def iter_openarchives_rows() -> Iterator[dict[str, Any]]:
    reevaluated_at = pd.read_parquet(REEVAL_ROOT / "openarchives.gr" / "document_quality.parquet", columns=["reevaluated_at"]).iloc[0]["reevaluated_at"]
    root = RAW_ROOT / "hf" / "openarchives.gr" / "data" / "openarchives"
    for jsonl_path in sorted(root.glob("shard_*/*.jsonl.zst")):
        dctx = zstd.ZstdDecompressor()
        with jsonl_path.open("rb") as fh, dctx.stream_reader(fh) as reader:
            text_reader = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_reader:
                record = json.loads(line)
                source = record.get("source_metadata") or {}
                pipeline = record.get("pipeline_metadata") or {}
                nested = maybe_json_loads(source.get("metadata_json"))
                title = normalize_title(nested.get("Τίτλος") or nested.get("Title"))
                author = normalize_author(nested.get("Δημιουργός") or nested.get("Creator"))
                metadata_payload = select_metadata_fields(
                    nested,
                    [
                        "Θέμα",
                        "Περιγραφή",
                        "Επιστημονικό πεδίο",
                        "Σχολή/Τμήμα/Ινστιτούτο",
                        "Τύπος",
                        "Χρονολογία",
                        "Ημερομηνία έκδοσης",
                        "Συντελεστής",
                        "Πάροχος",
                        "Δικαιώματα",
                        "Αποθετήριο / συλλογή",
                        "Επιμέρους συλλογή",
                    ],
                )
                metadata_payload.update(
                    select_metadata_fields(
                        source,
                        ["type", "collection_slug", "language_code"],
                    )
                )
                text = clean_text(record.get("text"))
                yield {
                    "source_dataset": "openarchives.gr",
                    "source_doc_id": clean_text(record.get("doc_id") or record.get("filename")),
                    "text": text,
                    "title": title,
                    "author": author,
                    "source_metadata_json": metadata_json(metadata_payload),
                    "is_historical_or_polytonic": derive_historical_flag(
                        "openarchives.gr",
                        polytonic_ratio=pipeline.get("polytonic_ratio"),
                        author_year=None,
                    ),
                    "contains_latex": contains_latex(text),
                    "contains_math": contains_math(
                        text,
                        source_hint=bool(pipeline.get("math_enriched")),
                        formula_total=pipeline.get("formula_total"),
                    ),
                    "greek_percentage": derive_greek_percentage(
                        pipeline.get("percentage_greek"),
                        pipeline.get("latin_percentage"),
                    ),
                    "latin_percentage": pipeline.get("latin_percentage"),
                    "polytonic_ratio": pipeline.get("polytonic_ratio"),
                    "table_ratio": None,
                    "greek_badness_score": pipeline.get("greek_badness_score"),
                    "mojibake_badness_score": pipeline.get("mojibake_badness_score"),
                    "needs_ocr": pipeline.get("needs_ocr"),
                    "is_empty": pipeline.get("is_empty"),
                    "filter": pipeline.get("filter"),
                    "ocr_success": pipeline.get("ocr_success"),
                    "quality_method": "existing_pipeline_exact",
                    "reevaluated_at": reevaluated_at,
                }


def iter_greek_phd_rows() -> Iterator[dict[str, Any]]:
    reevaluated_at = pd.read_parquet(REEVAL_ROOT / "greek_phd" / "document_quality.parquet", columns=["reevaluated_at"]).iloc[0]["reevaluated_at"]
    root = RAW_ROOT / "mozilla" / "greek_phd" / "phd-theses-corpus" / "contents"
    for jsonl_path in sorted(root.glob("*.jsonl.zst")):
        dctx = zstd.ZstdDecompressor()
        with jsonl_path.open("rb") as fh, dctx.stream_reader(fh) as reader:
            text_reader = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_reader:
                record = json.loads(line)
                source = record.get("source_metadata") or {}
                metadata_payload = select_metadata_fields(
                    source,
                    [
                        "year",
                        "department",
                        "university",
                        "abstract_el",
                        "scientific_field",
                        "scientific_field_level1",
                        "scientific_field_level2",
                        "scientific_field_level3",
                        "keywords",
                        "doi",
                        "language",
                        "title_en",
                        "date_accepted",
                        "handle_url",
                        "license",
                    ],
                )
                text = clean_text(record.get("document"))
                yield {
                    "source_dataset": "greek_phd",
                    "source_doc_id": clean_text(record.get("doc_id") or record.get("filename")),
                    "text": text,
                    "title": normalize_title(source.get("title")),
                    "author": normalize_author(source.get("author")),
                    "source_metadata_json": metadata_json(metadata_payload),
                    "is_historical_or_polytonic": False,
                    "contains_latex": contains_latex(text),
                    "contains_math": contains_math(
                        text,
                        source_hint=bool(record.get("math_enriched")),
                        formula_total=record.get("formula_total"),
                    ),
                    "greek_percentage": derive_greek_percentage(record.get("percentage_greek"), record.get("latin_percentage")),
                    "latin_percentage": record.get("latin_percentage"),
                    "polytonic_ratio": record.get("polytonic_ratio"),
                    "table_ratio": None,
                    "greek_badness_score": record.get("greek_badness_score"),
                    "mojibake_badness_score": record.get("mojibake_badness_score"),
                    "needs_ocr": record.get("needs_ocr"),
                    "is_empty": record.get("is_empty"),
                    "filter": record.get("filter"),
                    "ocr_success": record.get("ocr_success"),
                    "quality_method": "existing_pipeline_exact",
                    "reevaluated_at": reevaluated_at,
                }


def external_repo_root(repo_id: str) -> Path:
    return EXTERNAL_ROOT / repo_id.replace("/", "__")


def iter_finewiki_rows() -> Iterator[dict[str, Any]]:
    root = external_repo_root("HuggingFaceFW/finewiki")
    path = root / "data" / "elwiki" / "000_00000.parquet"
    parquet = pq.ParquetFile(path)
    row_offset = 0
    for batch in parquet.iter_batches(batch_size=2000):
        frame = batch.to_pandas()
        for batch_index, row in enumerate(frame.to_dict("records")):
            text = clean_text(row.get("text"))
            original_id = clean_text(row.get("id") or row.get("page_id"))
            yield {
                "source_dataset": "HuggingFaceFW/finewiki",
                "source_doc_id": f"elwiki::{path.name}::{row_offset + batch_index:07d}::{original_id}",
                "text": text,
                "title": normalize_title(row.get("title")),
                "author": None,
                "source_metadata_json": metadata_json(
                    {
                        "url": row.get("url"),
                        "date_modified": row.get("date_modified"),
                        "wikidata_id": row.get("wikidata_id"),
                        "page_id": row.get("page_id"),
                        "wikiname": row.get("wikiname"),
                        "in_language": row.get("in_language"),
                        "version": row.get("version"),
                    }
                ),
                "is_historical_or_polytonic": False,
                "contains_latex": contains_latex(text),
                "contains_math": contains_math(text, source_hint=bool(row.get("has_math"))),
                "greek_percentage": None,
                "latin_percentage": None,
                "polytonic_ratio": None,
                "table_ratio": None,
                "greek_badness_score": None,
                "mojibake_badness_score": None,
                "needs_ocr": None,
                "is_empty": None,
                "filter": None,
                "ocr_success": None,
                "quality_method": None,
                "reevaluated_at": None,
            }
        row_offset += len(frame)


def iter_finepdfs_edu_rows() -> Iterator[dict[str, Any]]:
    root = external_repo_root("HuggingFaceFW/finepdfs-edu")
    for path in sorted((root / "data" / "ell_Grek" / "train").glob("*.parquet")):
        parquet = pq.ParquetFile(path)
        row_offset = 0
        for batch in parquet.iter_batches(batch_size=1000):
            frame = batch.to_pandas()
            for batch_index, row in enumerate(frame.to_dict("records")):
                text = clean_text(row.get("text"))
                original_id = clean_text(row.get("id"))
                yield {
                    "source_dataset": "HuggingFaceFW/finepdfs-edu",
                    "source_doc_id": f"ell_Grek::{path.name}::{row_offset + batch_index:07d}::{original_id}",
                    "text": text,
                    "title": None,
                    "author": None,
                    "source_metadata_json": metadata_json(
                        {
                            "url": row.get("url"),
                            "date": row.get("date"),
                            "dump": row.get("dump"),
                            "language": row.get("language"),
                            "full_doc_lid": row.get("full_doc_lid"),
                            "full_doc_lid_score": row.get("full_doc_lid_score"),
                            "page_average_lid": row.get("page_average_lid"),
                            "page_average_lid_score": row.get("page_average_lid_score"),
                            "token_count": row.get("token_count"),
                            "fw_edu_scores": row.get("fw_edu_scores"),
                        }
                    ),
                    "is_historical_or_polytonic": False,
                    "contains_latex": contains_latex(text),
                    "contains_math": contains_math(text),
                    "greek_percentage": None,
                    "latin_percentage": None,
                    "polytonic_ratio": None,
                    "table_ratio": None,
                    "greek_badness_score": None,
                    "mojibake_badness_score": None,
                    "needs_ocr": None,
                    "is_empty": None,
                    "filter": None,
                    "ocr_success": None,
                    "quality_method": None,
                    "reevaluated_at": None,
                }
        row_offset += len(frame)


def iter_opensubtitles_el_rows() -> Iterator[dict[str, Any]]:
    path = ensure_opensubtitles_el_source()
    base_metadata = {
        "corpus": "OpenSubtitles",
        "language": "el",
        "version": "v2018",
        "format": "xml_zip",
        "source_url": OPENSUBTITLES_EL_URL,
    }
    with zipfile.ZipFile(path) as archive:
        members = sorted(
            (
                info
                for info in archive.infolist()
                if not info.is_dir() and (info.filename.endswith(".xml") or info.filename.endswith(".xml.gz"))
            ),
            key=lambda info: info.filename,
        )
        for info in members:
            with archive.open(info) as raw_handle:
                if info.filename.endswith(".gz"):
                    xml_handle = gzip.GzipFile(fileobj=raw_handle)
                else:
                    xml_handle = raw_handle
                tree = ET.parse(xml_handle)
            text = clean_text(opus_xml_document_text(tree.getroot()))
            if not text:
                continue
            source_doc_id = strip_known_suffixes(info.filename, (".xml.gz", ".xml")).replace("/", "::")
            metadata = opus_xml_metadata(tree.getroot(), info.filename, base_metadata)
            yield {
                "source_dataset": OPENSUBTITLES_EL_DATASET,
                "source_doc_id": source_doc_id,
                "text": text,
                "title": None,
                "author": None,
                "source_metadata_json": metadata_json(metadata),
                "is_historical_or_polytonic": False,
                "contains_latex": contains_latex(text),
                "contains_math": contains_math(text),
                "greek_percentage": None,
                "latin_percentage": None,
                "polytonic_ratio": None,
                "table_ratio": None,
                "greek_badness_score": None,
                "mojibake_badness_score": None,
                "needs_ocr": None,
                "is_empty": None,
                "filter": None,
                "ocr_success": None,
                "quality_method": None,
                "reevaluated_at": None,
            }


def load_greek_legal_split(root: Path, config: str, split: str) -> pd.DataFrame:
    return pd.read_parquet(root / config / f"{split}-00000-of-00001.parquet")


def build_greek_legal() -> pd.DataFrame:
    root = external_repo_root("AI-team-UoA/greek_legal_code")
    frames = []
    for split in ["train", "validation", "test"]:
        volume = load_greek_legal_split(root, "volume", split)
        chapter = load_greek_legal_split(root, "chapter", split)
        subject = load_greek_legal_split(root, "subject", split)
        if not volume["text"].equals(chapter["text"]) or not volume["text"].equals(subject["text"]):
            raise ValueError(f"greek_legal_code text misalignment in split {split}")
        merged = pd.DataFrame(
            {
                "source_dataset": "AI-team-UoA/greek_legal_code",
                "source_doc_id": [f"{split}_{idx:06d}" for idx in range(len(volume))],
                "text": volume["text"].astype(str),
                "title": None,
                "author": None,
                "source_metadata_json": [
                    metadata_json({"volume": vol, "chapter": ch, "subject": sub})
                    for vol, ch, sub in zip(volume["label"], chapter["label"], subject["label"], strict=True)
                ],
                "is_historical_or_polytonic": False,
                "contains_latex": volume["text"].astype(str).map(contains_latex),
                "contains_math": volume["text"].astype(str).map(contains_math),
                "greek_percentage": None,
                "latin_percentage": None,
                "polytonic_ratio": None,
                "table_ratio": None,
                "greek_badness_score": None,
                "mojibake_badness_score": None,
                "needs_ocr": None,
                "is_empty": None,
                "filter": None,
                "ocr_success": None,
                "quality_method": None,
                "reevaluated_at": None,
            }
        )
        frames.append(merged)
    return finalize_frame(pd.concat(frames, ignore_index=True))


FRAME_DATASET_BUILDERS = {
    "Sxolika_vivlia": build_sxolika,
    "dimodis_logotexnia": build_dimodis,
    "Ellinika_Keimena_Project_Gutenberg": build_gutenberg,
    "1000_prwta_xronia_ellhnikhs": build_first_1k,
    "klasikh_arx_ell_grammateia": build_klasikh,
    "Wikisource_Greek_texts": build_wikisource,
    "Ekklisiastika_Keimena": build_ekklisiastika,
    "Apothetirio_Pergamos": build_pergamos,
    "Apothetirio_Kallipos": build_kallipos,
    "opengov.gr-diaboyleuseis": build_opengov,
    "openbook_gr": build_openbook,
    "AI-team-UoA/greek_legal_code": build_greek_legal,
}

STREAM_DATASET_BUILDERS = {
    "ellinika_dedomena_europaikou_koinovouliou": iter_europarl_rows,
    "eurlex-greek-legislation": iter_eurlex_rows,
    "openarchives.gr": iter_openarchives_rows,
    "greek_phd": iter_greek_phd_rows,
    "HuggingFaceFW/finewiki": iter_finewiki_rows,
    "HuggingFaceFW/finepdfs-edu": iter_finepdfs_edu_rows,
    OPENSUBTITLES_EL_DATASET: iter_opensubtitles_el_rows,
}


def default_build_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(4, cpu_count))


def dataset_output_path(output_root: Path, dataset_name: str) -> Path:
    return output_root / "data" / f"{dataset_name.replace('/', '__')}.parquet"


def build_dataset_to_parquet(dataset_name: str, output_root: str, score_external_quality: bool = False) -> dict[str, Any]:
    output_root_path = Path(output_root)
    ensure_dir(output_root_path / "data")
    path = dataset_output_path(output_root_path, dataset_name)
    if dataset_name in FRAME_DATASET_BUILDERS:
        frame = FRAME_DATASET_BUILDERS[dataset_name]()
        row_iter = dataframe_rows(frame)
    else:
        row_iter = STREAM_DATASET_BUILDERS[dataset_name]()
    row_count = write_parquet_rows(
        path,
        maybe_iter_exact_metadata_dedup(dataset_name, row_iter),
        batch_size=1000 if "finepdfs" in dataset_name else 2000,
        score_missing_quality=score_external_quality and dataset_name in EXTERNAL_DATASETS,
    )
    return {"source_dataset": dataset_name, "path": str(path), "row_count": row_count}


def build_canonical_corpus(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    include_external: bool = True,
    score_external_quality: bool = False,
    force_download_external: bool = False,
    workers: int | None = None,
) -> list[SourceBuildResult]:
    data_root = output_root / "data"
    ensure_dir(data_root)
    results: list[SourceBuildResult] = []
    for path in data_root.glob("*.parquet"):
        path.unlink()

    if include_external:
        download_selected_external_sources(force=force_download_external)
    if workers is None:
        workers = default_build_workers()
    frame_datasets = [name for name in GLOSSAPI_INCLUDED_DATASETS if name in FRAME_DATASET_BUILDERS]
    if include_external:
        frame_datasets.append("AI-team-UoA/greek_legal_code")
    stream_datasets = [name for name in GLOSSAPI_INCLUDED_DATASETS if name in STREAM_DATASET_BUILDERS]
    if include_external:
        stream_datasets.extend([name for name in EXTERNAL_DATASETS if name in STREAM_DATASET_BUILDERS])

    built: list[dict[str, Any]] = []
    frame_worker_count = max(1, min(workers, len(frame_datasets))) if frame_datasets else 0
    if frame_worker_count == 1:
        built.extend(build_dataset_to_parquet(name, str(output_root), score_external_quality) for name in frame_datasets)
    elif frame_worker_count > 1:
        with cf.ProcessPoolExecutor(max_workers=frame_worker_count) as executor:
            futures = [
                executor.submit(build_dataset_to_parquet, name, str(output_root), score_external_quality)
                for name in frame_datasets
            ]
            built.extend(future.result() for future in cf.as_completed(futures))

    stream_worker_count = max(1, min(2, workers, len(stream_datasets))) if stream_datasets else 0
    if stream_worker_count == 1:
        built.extend(build_dataset_to_parquet(name, str(output_root), score_external_quality) for name in stream_datasets)
    elif stream_worker_count > 1:
        with cf.ProcessPoolExecutor(max_workers=stream_worker_count) as executor:
            futures = [
                executor.submit(build_dataset_to_parquet, name, str(output_root), score_external_quality)
                for name in stream_datasets
            ]
            built.extend(future.result() for future in cf.as_completed(futures))

    for item in built:
        results.append(SourceBuildResult(dataset_name=item["source_dataset"], path=Path(item["path"]), row_count=int(item["row_count"])))

    manifest = pd.DataFrame(
        [{"source_dataset": result.dataset_name, "path": str(result.path), "row_count": result.row_count} for result in results]
    ).sort_values("source_dataset")
    manifest.to_csv(output_root / "row_counts.csv", index=False)
    validation = validate_canonical_corpus(output_root)
    pd.DataFrame(validation).to_csv(output_root / "validation_summary.csv", index=False)
    return sorted(results, key=lambda item: item.dataset_name)


def read_all_canonical_frames(output_root: Path = DEFAULT_OUTPUT_ROOT) -> pd.DataFrame:
    data_root = output_root / "data"
    frames = [pd.read_parquet(path) for path in sorted(data_root.glob("*.parquet"))]
    if not frames:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def validate_canonical_corpus(output_root: Path = DEFAULT_OUTPUT_ROOT) -> list[dict[str, Any]]:
    rows = []
    data_root = output_root / "data"
    for path in sorted(data_root.glob("*.parquet")):
        parquet = pq.ParquetFile(path)
        source_dataset = path.stem
        leaks = 0
        empty_text = 0
        duplicate_ids = 0
        seen_ids: set[str] = set()
        for batch in parquet.iter_batches(
            batch_size=2048,
            columns=["source_dataset", "source_doc_id", "text", "source_metadata_json"],
        ):
            frame = batch.to_pandas()
            if source_dataset == path.stem and not frame.empty:
                source_dataset = str(frame["source_dataset"].iloc[0])
            texts = frame["text"].fillna("").astype(str).str.strip()
            empty_text += int((texts == "").sum())
            for source_doc_id in frame["source_doc_id"].fillna("").astype(str):
                if source_doc_id in seen_ids:
                    duplicate_ids += 1
                else:
                    seen_ids.add(source_doc_id)
            for payload in frame["source_metadata_json"].dropna():
                try:
                    parsed = json.loads(payload)
                except Exception:
                    leaks += 1
                    continue
                if any(key in FORBIDDEN_SOURCE_METADATA_KEYS for key in parsed):
                    leaks += 1
        rows.append(
            {
                "source_dataset": source_dataset,
                "row_count": parquet.metadata.num_rows,
                "empty_text_rows": empty_text,
                "duplicate_source_doc_id_rows": duplicate_ids,
                "source_metadata_leaks": leaks,
            }
        )
    return rows


def rust_badness_is_reliable(dataset_name: str) -> bool:
    return dataset_name in RUST_BADNESS_RELIABLE_DATASETS


def apply_quality_preset(df: pd.DataFrame, preset: str) -> pd.DataFrame:
    frame = df.copy()
    if preset == "none":
        return frame
    if "is_empty" in frame:
        frame = frame[frame["is_empty"].fillna(False) == False]  # noqa: E712
    if "filter" in frame:
        frame = frame[frame["filter"].fillna("ok").isin(["ok", "keep", ""])]
    if "needs_ocr" in frame:
        needs_ocr = frame["needs_ocr"].fillna(False)
        if "ocr_success" in frame:
            ocr_success = frame["ocr_success"].fillna(False)
        else:
            ocr_success = pd.Series(False, index=frame.index)
        frame = frame[(needs_ocr == False) | (ocr_success == True)]  # noqa: E712
    reliable = frame["source_dataset"].map(rust_badness_is_reliable)
    if preset == "modern_strict":
        mask = (~reliable) | frame["greek_badness_score"].isna() | (frame["greek_badness_score"] < 25)
        return frame[mask]
    if preset == "modern_relaxed":
        mask = (~reliable) | frame["greek_badness_score"].isna() | (frame["greek_badness_score"] < 30)
        return frame[mask]
    if preset == "historical_tolerant":
        mask = frame["is_historical_or_polytonic"].fillna(False) | frame["greek_badness_score"].isna() | (frame["greek_badness_score"] < 35)
        return frame[mask]
    raise ValueError(f"unknown quality preset: {preset}")


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def hash_bytes(data: bytes) -> str:
    return blake3(data).hexdigest()


def stable_doc_key(source_dataset: str, source_doc_id: str) -> str:
    return hash_bytes(f"{source_dataset}\0{source_doc_id}".encode("utf-8"))


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


def validate_source_mix_fraction_mode(mode: str) -> str:
    if mode not in SOURCE_MIX_FRACTION_MODES:
        raise ValueError(f"source mix fraction_mode must be one of: {', '.join(sorted(SOURCE_MIX_FRACTION_MODES))}")
    return mode


def _coerce_optional_source_list(value: Any, *, field_name: str) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            raise ValueError(f"{field_name} entries must be non-empty strings")
        return [candidate]
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a string or list of strings")
    normalized: list[str] = []
    for item in value:
        candidate = str(item).strip()
        if not candidate:
            raise ValueError(f"{field_name} entries must be non-empty strings")
        normalized.append(candidate)
    return normalized


def load_source_mix_config(source_mix_config_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(source_mix_config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("source mix config must be a JSON object")
    raw_entries = payload.get("entries", payload.get("components"))
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("source mix config must define a non-empty 'entries' list")
    normalized_entries: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for idx, raw in enumerate(raw_entries):
        if not isinstance(raw, dict):
            raise ValueError("each source mix entry must be a JSON object")
        include_sources = _coerce_optional_source_list(raw.get("include_sources"), field_name="include_sources")
        if include_sources is None:
            include_sources = _coerce_optional_source_list(raw.get("source_datasets"), field_name="source_datasets")
        if include_sources is None:
            include_sources = _coerce_optional_source_list(raw.get("sources"), field_name="sources")
        source_dataset = raw.get("source_dataset")
        if include_sources is None and source_dataset is not None:
            include_sources = _coerce_optional_source_list(source_dataset, field_name="source_dataset")
        exclude_sources = _coerce_optional_source_list(raw.get("exclude_sources"), field_name="exclude_sources")
        if include_sources is None and exclude_sources is None:
            raise ValueError("each source mix entry must define at least include_sources/source_dataset or exclude_sources")
        try:
            fraction = float(raw.get("fraction"))
        except Exception as exc:
            raise ValueError("each source mix entry must define a numeric fraction") from exc
        if not math.isfinite(fraction) or fraction < 0 or fraction > 1:
            raise ValueError("source mix entry fractions must be finite numbers in [0, 1]")
        fraction_mode = validate_source_mix_fraction_mode(str(raw.get("fraction_mode", "of_group")))
        name = str(
            raw.get("name")
            or raw.get("group")
            or source_dataset
            or f"entry_{idx:02d}"
        ).strip()
        if not name:
            raise ValueError("source mix entry names must be non-empty")
        if name in seen_names:
            raise ValueError(f"duplicate source mix entry name: {name}")
        seen_names.add(name)
        normalized_entries.append(
            {
                "name": name,
                "include_sources": include_sources,
                "exclude_sources": exclude_sources,
                "fraction": fraction,
                "fraction_mode": fraction_mode,
            }
        )
    return normalized_entries


def frame_char_mass(frame: pd.DataFrame) -> pd.Series:
    fallback = frame["text"].fillna("").astype(str).str.len().astype("int64")
    if "chars" not in frame.columns:
        return fallback
    chars = pd.to_numeric(frame["chars"], errors="coerce")
    return chars.fillna(fallback).astype("int64").clip(lower=0)


def select_rows_for_char_budget(frame: pd.DataFrame, *, target_chars: int, chars_column: str) -> pd.DataFrame:
    if frame.empty or target_chars <= 0:
        return frame.iloc[0:0].copy()
    total_chars = int(frame[chars_column].sum())
    if total_chars <= target_chars:
        return frame.copy()
    ordered = frame.sort_values(
        ["_source_mix_doc_key", "source_dataset", "source_doc_id"],
        ascending=[True, True, True],
    )
    selected_indices: list[int] = []
    running_chars = 0
    for idx, row in ordered.iterrows():
        row_chars = int(row[chars_column])
        if running_chars + row_chars >= target_chars:
            before_diff = abs(target_chars - running_chars)
            after_diff = abs(target_chars - (running_chars + row_chars))
            if not selected_indices or after_diff < before_diff:
                selected_indices.append(idx)
            break
        selected_indices.append(idx)
        running_chars += row_chars
    return frame.loc[selected_indices].copy()


def apply_source_mix_config(
    frame: pd.DataFrame,
    *,
    source_mix_config_path: Path,
    annotate_component: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    entries = load_source_mix_config(source_mix_config_path)
    if frame.empty:
        return frame.copy(), {
            "config_path": str(source_mix_config_path),
            "rows_before": 0,
            "rows_after": 0,
            "chars_before": 0,
            "chars_after": 0,
            "entries": [],
        }
    working = frame.copy()
    working["_source_mix_chars"] = frame_char_mass(working)
    working["_source_mix_doc_key"] = [
        stable_doc_key(str(source_dataset), str(source_doc_id))
        for source_dataset, source_doc_id in zip(working["source_dataset"], working["source_doc_id"], strict=True)
    ]
    used_indices: set[int] = set()
    resolved_entries: list[dict[str, Any]] = []
    for entry in entries:
        mask = pd.Series(True, index=working.index)
        if entry["include_sources"] is not None:
            mask &= working["source_dataset"].isin(entry["include_sources"])
        if entry["exclude_sources"] is not None:
            mask &= ~working["source_dataset"].isin(entry["exclude_sources"])
        entry_frame = working[mask].copy()
        if entry_frame.empty:
            raise ValueError(f"source mix entry '{entry['name']}' matched zero rows")
        overlap = used_indices.intersection(entry_frame.index.tolist())
        if overlap:
            raise ValueError(f"source mix entry '{entry['name']}' overlaps with a previous entry; entries must be disjoint")
        used_indices.update(entry_frame.index.tolist())
        resolved_entries.append(
            {
                **entry,
                "frame": entry_frame,
                "available_rows": int(len(entry_frame)),
                "available_chars": int(entry_frame["_source_mix_chars"].sum()),
            }
        )
    unmatched = working.loc[~working.index.isin(used_indices)].copy()
    fixed_entries = [entry for entry in resolved_entries if entry["fraction_mode"] == "of_group"]
    share_entries = [entry for entry in resolved_entries if entry["fraction_mode"] == "of_total"]
    fixed_target_chars_total = 0.0
    for entry in fixed_entries:
        entry["requested_chars"] = float(entry["available_chars"]) * float(entry["fraction"])
        fixed_target_chars_total += entry["requested_chars"]
    share_fraction_total = float(sum(float(entry["fraction"]) for entry in share_entries))
    target_mix_chars_total: float | None = None
    if share_entries:
        if fixed_target_chars_total > 0:
            if share_fraction_total >= 1.0:
                raise ValueError("source mix config is infeasible: of_total fractions must sum to < 1 when of_group entries are present")
            target_mix_chars_total = fixed_target_chars_total / (1.0 - share_fraction_total)
        else:
            if not math.isclose(share_fraction_total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(
                    "source mix config with only of_total entries must have fractions summing to exactly 1.0"
                )
            target_mix_chars_total = min(
                float(entry["available_chars"]) / float(entry["fraction"])
                for entry in share_entries
                if float(entry["fraction"]) > 0
            )
        for entry in share_entries:
            requested_chars = float(target_mix_chars_total) * float(entry["fraction"])
            if requested_chars > float(entry["available_chars"]) + 1e-6:
                raise ValueError(
                    f"source mix entry '{entry['name']}' cannot satisfy requested of_total share with available chars"
                )
            entry["requested_chars"] = requested_chars
    selected_parts: list[pd.DataFrame] = []
    entry_summaries: list[dict[str, Any]] = []
    for entry in resolved_entries:
        requested_chars_int = int(math.floor(float(entry["requested_chars"]) + 1e-9))
        entry_frame = entry["frame"]
        if float(entry["fraction"]) >= 1.0 or requested_chars_int >= int(entry["available_chars"]):
            selected = entry_frame.copy()
        else:
            selected = select_rows_for_char_budget(
                entry_frame,
                target_chars=requested_chars_int,
                chars_column="_source_mix_chars",
            )
        if annotate_component:
            selected["source_mix_component"] = entry["name"]
        selected_parts.append(selected)
        entry_summaries.append(
            {
                "name": entry["name"],
                "fraction_mode": entry["fraction_mode"],
                "fraction": float(entry["fraction"]),
                "include_sources": entry["include_sources"],
                "exclude_sources": entry["exclude_sources"],
                "available_rows": int(entry["available_rows"]),
                "available_chars": int(entry["available_chars"]),
                "requested_chars": float(entry["requested_chars"]),
                "selected_rows": int(len(selected)),
                "selected_chars": int(selected["_source_mix_chars"].sum()),
            }
        )
    selected_frame = pd.concat(selected_parts, ignore_index=False) if selected_parts else working.iloc[0:0].copy()
    selected_frame = selected_frame.sort_values(["_source_mix_doc_key", "source_dataset", "source_doc_id"]).reset_index(drop=True)
    selected_frame = selected_frame.drop(columns=["_source_mix_chars", "_source_mix_doc_key"], errors="ignore")
    summary = {
        "config_path": str(source_mix_config_path),
        "rows_before": int(len(frame)),
        "rows_after": int(len(selected_frame)),
        "chars_before": int(working["_source_mix_chars"].sum()),
        "chars_after": int(sum(item["selected_chars"] for item in entry_summaries)),
        "unmatched_rows": int(len(unmatched)),
        "unmatched_chars": int(unmatched["_source_mix_chars"].sum()) if not unmatched.empty else 0,
        "target_mix_chars_total": None if target_mix_chars_total is None else float(target_mix_chars_total),
        "entries": entry_summaries,
    }
    return selected_frame, summary


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def mix_intermediate_schema(include_component: bool = False) -> pa.Schema:
    fields = list(CANONICAL_ARROW_SCHEMA)
    fields.extend(
        [
            ("doc_key", pa.string()),
            ("source_mix_chars", pa.int64()),
        ]
    )
    if include_component:
        fields.append(("source_mix_component", pa.string()))
    return pa.schema(fields)


def mix_output_schema(include_component: bool = False) -> pa.Schema:
    fields = list(CANONICAL_ARROW_SCHEMA)
    if include_component:
        fields.append(("source_mix_component", pa.string()))
    return pa.schema(fields)


def builder_bundle_paths(dedup_metadata_root: Path) -> tuple[dict[str, Any], Path, Path | None]:
    manifest_path = dedup_metadata_root / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"builder dedup manifest missing under {dedup_metadata_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    doc_metadata_path = dedup_metadata_root / str(files.get("doc_metadata", "doc_dedup_metadata.parquet"))
    family_membership_rel = files.get("family_membership", "dedup_family_membership.parquet")
    family_membership_path = dedup_metadata_root / str(family_membership_rel)
    return manifest, doc_metadata_path, family_membership_path if family_membership_path.exists() else None


def build_mix_should_stream(
    *,
    mix_output_path: Path | None,
    dedup_metadata_root: Path | None,
    dedup_action: str,
    source_mix_config_path: Path | None,
    exclude_doc_keys_path: Path | None = None,
) -> bool:
    if exclude_doc_keys_path is not None:
        return True
    if mix_output_path is None:
        return False
    if source_mix_config_path is not None:
        return True
    if dedup_metadata_root is None:
        return False
    if validate_dedup_action(dedup_action) not in {"drop_intra", "drop_intra_and_inter"}:
        return False
    try:
        _, _, family_membership_path = builder_bundle_paths(dedup_metadata_root)
    except Exception:
        return False
    return family_membership_path is not None


def default_mix_prepare_workers() -> int:
    override = os.environ.get("GLOSSAPI_MIX_PREPARE_WORKERS")
    if override:
        try:
            value = int(override)
        except ValueError:
            value = 1
        return max(1, value)
    cpu_count = os.cpu_count() or 1
    return max(1, min(16, cpu_count))


def _iter_filtered_mix_frames_from_parquet(
    path: Path,
    *,
    include_sources: list[str] | None = None,
    exclude_sources: list[str] | None = None,
    exclude_needs_ocr_sources: list[str] | None = None,
    quality_preset: str = "none",
    historical_mode: str = "include",
    math_mode: str = "include",
    latex_mode: str = "include",
    batch_size: int = 2048,
) -> Iterator[pd.DataFrame]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        frame = batch.to_pandas()
        frame = finalize_frame(frame)
        if include_sources:
            frame = frame[frame["source_dataset"].isin(include_sources)]
        if exclude_sources:
            frame = frame[~frame["source_dataset"].isin(exclude_sources)]
        if exclude_needs_ocr_sources:
            frame = frame[
                ~(
                    frame["source_dataset"].isin(exclude_needs_ocr_sources)
                    & frame["needs_ocr"].fillna(False)
                )
            ]
        if frame.empty:
            continue
        frame = apply_quality_preset(frame, quality_preset)
        if historical_mode == "exclude":
            frame = frame[frame["is_historical_or_polytonic"].fillna(False) == False]  # noqa: E712
        elif historical_mode == "only":
            frame = frame[frame["is_historical_or_polytonic"].fillna(False)]
        if math_mode == "exclude":
            frame = frame[frame["contains_math"].fillna(False) == False]  # noqa: E712
        elif math_mode == "only":
            frame = frame[frame["contains_math"].fillna(False)]
        if latex_mode == "exclude":
            frame = frame[frame["contains_latex"].fillna(False) == False]  # noqa: E712
        elif latex_mode == "only":
            frame = frame[frame["contains_latex"].fillna(False)]
        if frame.empty:
            continue
        frame = frame.reset_index(drop=True)
        frame["doc_key"] = [
            stable_doc_key(str(source_dataset), str(source_doc_id))
            for source_dataset, source_doc_id in zip(frame["source_dataset"], frame["source_doc_id"], strict=True)
        ]
        frame["source_mix_chars"] = frame_char_mass(frame)
        yield frame[[*CANONICAL_COLUMNS, "doc_key", "source_mix_chars"]]


def iter_filtered_mix_frames(
    output_root: Path,
    *,
    include_sources: list[str] | None = None,
    exclude_sources: list[str] | None = None,
    exclude_needs_ocr_sources: list[str] | None = None,
    quality_preset: str = "none",
    historical_mode: str = "include",
    math_mode: str = "include",
    latex_mode: str = "include",
    batch_size: int = 2048,
) -> Iterator[pd.DataFrame]:
    data_root = output_root / "data"
    for path in sorted(data_root.glob("*.parquet")):
        yield from _iter_filtered_mix_frames_from_parquet(
            path,
            include_sources=include_sources,
            exclude_sources=exclude_sources,
            exclude_needs_ocr_sources=exclude_needs_ocr_sources,
            quality_preset=quality_preset,
            historical_mode=historical_mode,
            math_mode=math_mode,
            latex_mode=latex_mode,
            batch_size=batch_size,
        )


def filter_mix_parquet_file_to_shard(
    path_str: str,
    destination_str: str,
    *,
    include_sources: list[str] | None = None,
    exclude_sources: list[str] | None = None,
    exclude_needs_ocr_sources: list[str] | None = None,
    quality_preset: str = "none",
    historical_mode: str = "include",
    math_mode: str = "include",
    latex_mode: str = "include",
    batch_size: int = 2048,
) -> dict[str, Any]:
    path = Path(path_str)
    destination = Path(destination_str)
    rows_written = write_mix_frames_to_parquet(
        destination,
        _iter_filtered_mix_frames_from_parquet(
            path,
            include_sources=include_sources,
            exclude_sources=exclude_sources,
            exclude_needs_ocr_sources=exclude_needs_ocr_sources,
            quality_preset=quality_preset,
            historical_mode=historical_mode,
            math_mode=math_mode,
            latex_mode=latex_mode,
            batch_size=batch_size,
        ),
    )
    return {
        "source_path": str(path),
        "destination_path": str(destination),
        "rows_written": int(rows_written),
    }


def write_mix_frames_to_parquet(
    destination: Path,
    frames: Iterable[pd.DataFrame],
    *,
    include_component: bool = False,
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f".{destination.name}.tmp")
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    schema = mix_intermediate_schema(include_component=include_component)
    try:
        for frame in frames:
            if frame.empty:
                continue
            table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temp_path, schema, compression="zstd")
            writer.write_table(table)
            rows_written += int(len(frame))
    finally:
        if writer is not None:
            writer.close()
        else:
            pq.write_table(schema.empty_table(), temp_path, compression="zstd")
    temp_path.replace(destination)
    return rows_written


def materialize_filtered_mix_input(
    output_root: Path,
    *,
    destination: Path,
    include_sources: list[str] | None,
    exclude_sources: list[str] | None,
    exclude_needs_ocr_sources: list[str] | None,
    quality_preset: str,
    historical_mode: str,
    math_mode: str,
    latex_mode: str,
    workers: int | None = None,
) -> dict[str, Any]:
    data_root = output_root / "data"
    source_paths = sorted(data_root.glob("*.parquet"))
    if not source_paths:
        pq.write_table(mix_intermediate_schema().empty_table(), destination, compression="zstd")
        return {"rows_written": 0, "path": str(destination), "worker_count": 0, "source_file_count": 0}
    if workers is None:
        workers = default_mix_prepare_workers()
    worker_count = max(1, min(workers, len(source_paths)))
    if worker_count == 1:
        rows_written = write_mix_frames_to_parquet(
            destination,
            iter_filtered_mix_frames(
                output_root,
                include_sources=include_sources,
                exclude_sources=exclude_sources,
                exclude_needs_ocr_sources=exclude_needs_ocr_sources,
                quality_preset=quality_preset,
                historical_mode=historical_mode,
                math_mode=math_mode,
                latex_mode=latex_mode,
            ),
        )
        return {
            "rows_written": rows_written,
            "path": str(destination),
            "worker_count": 1,
            "source_file_count": len(source_paths),
        }
    with tempfile.TemporaryDirectory(prefix="glossapi_mix_filter_shards_", dir=str(destination.parent)) as temp_dir:
        shard_root = Path(temp_dir)
        shard_paths = [shard_root / f"filtered_{idx:05d}.parquet" for idx in range(len(source_paths))]
        rows_written = 0
        non_empty_shards: list[Path] = []
        with cf.ProcessPoolExecutor(max_workers=worker_count, mp_context=mp.get_context("spawn")) as executor:
            futures = [
                executor.submit(
                    filter_mix_parquet_file_to_shard,
                    str(source_path),
                    str(shard_path),
                    include_sources=include_sources,
                    exclude_sources=exclude_sources,
                    exclude_needs_ocr_sources=exclude_needs_ocr_sources,
                    quality_preset=quality_preset,
                    historical_mode=historical_mode,
                    math_mode=math_mode,
                    latex_mode=latex_mode,
                )
                for source_path, shard_path in zip(source_paths, shard_paths, strict=True)
            ]
            for future in cf.as_completed(futures):
                item = future.result()
                rows_written += int(item["rows_written"])
                if int(item["rows_written"]) > 0:
                    non_empty_shards.append(Path(item["destination_path"]))
        if not non_empty_shards:
            pq.write_table(mix_intermediate_schema().empty_table(), destination, compression="zstd")
        elif len(non_empty_shards) == 1:
            shutil.copy2(non_empty_shards[0], destination)
        else:
            con = _duckdb_connect_streaming()
            try:
                con.execute(
                    f"""
                    COPY (
                        SELECT *
                        FROM read_parquet({sql_quote(str((shard_root / '*.parquet').resolve()))})
                    ) TO {sql_quote(str(destination))} (FORMAT parquet, COMPRESSION zstd)
                    """
                )
            finally:
                con.close()
    return {
        "rows_written": rows_written,
        "path": str(destination),
        "worker_count": worker_count,
        "source_file_count": len(source_paths),
    }


def materialize_doc_key_excluded_mix_input(
    input_path: Path,
    *,
    exclude_doc_keys_path: Path,
    destination: Path,
) -> dict[str, Any]:
    """Anti-join a selected-input parquet against an external doc_key drop list."""
    if not exclude_doc_keys_path.exists():
        raise ValueError(f"exclude_doc_keys_path does not exist: {exclude_doc_keys_path}")
    ensure_dir(destination.parent)
    if destination.exists():
        destination.unlink()
    con = _duckdb_connect_streaming()
    try:
        rows_before = con.execute(
            f"SELECT count(*) FROM read_parquet({sql_quote(str(input_path))})"
        ).fetchone()[0]
        excluded_rows = con.execute(
            f"""
            SELECT count(*)
            FROM read_parquet({sql_quote(str(input_path))}) AS src
            JOIN (
                SELECT DISTINCT doc_key
                FROM read_parquet({sql_quote(str(exclude_doc_keys_path))})
            ) AS drop_docs USING (doc_key)
            """
        ).fetchone()[0]
        con.execute(
            f"""
            COPY (
                SELECT src.*
                FROM read_parquet({sql_quote(str(input_path))}) AS src
                LEFT JOIN (
                    SELECT DISTINCT doc_key
                    FROM read_parquet({sql_quote(str(exclude_doc_keys_path))})
                ) AS drop_docs USING (doc_key)
                WHERE drop_docs.doc_key IS NULL
                ORDER BY src.doc_key, src.source_dataset, src.source_doc_id
            ) TO {sql_quote(str(destination))} (FORMAT parquet, COMPRESSION zstd)
            """
        )
        rows_after = con.execute(
            f"SELECT count(*) FROM read_parquet({sql_quote(str(destination))})"
        ).fetchone()[0]
    finally:
        con.close()
    return {
        "input_path": str(input_path),
        "exclude_doc_keys_path": str(exclude_doc_keys_path),
        "output_path": str(destination),
        "rows_before": int(rows_before or 0),
        "excluded_rows": int(excluded_rows or 0),
        "rows_after": int(rows_after or 0),
    }


def prepare_reduced_builder_bundle(
    *,
    dedup_metadata_root: Path,
    duplicate_rows_path: Path,
    reduced_bundle_root: Path,
) -> Path:
    manifest, doc_metadata_path, family_membership_path = builder_bundle_paths(dedup_metadata_root)
    if family_membership_path is None:
        raise ValueError("family membership parquet missing; reduced builder bundle requires builder_metadata_v2")
    reduced_bundle_root.mkdir(parents=True, exist_ok=True)
    reduced_doc_metadata_path = reduced_bundle_root / "doc_dedup_metadata.parquet"
    reduced_family_membership_path = reduced_bundle_root / "dedup_family_membership.parquet"
    con = _duckdb_connect_streaming()
    try:
        con.execute(
            f"""
            COPY (
                SELECT d.*
                FROM read_parquet({sql_quote(str(doc_metadata_path))}) AS d
                JOIN (
                    SELECT DISTINCT doc_key
                    FROM read_parquet({sql_quote(str(duplicate_rows_path))})
                ) AS k USING (doc_key)
            ) TO {sql_quote(str(reduced_doc_metadata_path))} (FORMAT parquet, COMPRESSION zstd)
            """
        )
        con.execute(
            f"""
            COPY (
                SELECT d.*
                FROM read_parquet({sql_quote(str(family_membership_path))}) AS d
                JOIN (
                    SELECT DISTINCT doc_key
                    FROM read_parquet({sql_quote(str(duplicate_rows_path))})
                ) AS k USING (doc_key)
            ) TO {sql_quote(str(reduced_family_membership_path))} (FORMAT parquet, COMPRESSION zstd)
            """
        )
    finally:
        con.close()
    reduced_manifest = dict(manifest)
    reduced_manifest["files"] = dict(manifest.get("files", {}))
    reduced_manifest["files"]["doc_metadata"] = "doc_dedup_metadata.parquet"
    reduced_manifest["files"]["family_membership"] = "dedup_family_membership.parquet"
    (reduced_bundle_root / "manifest.json").write_text(
        json.dumps(reduced_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return reduced_bundle_root


def materialize_drop_action_deduped_mix(
    filtered_input_path: Path,
    *,
    dedup_metadata_root: Path,
    dedup_action: str,
    dedup_exact_stage: str,
    dedup_similarity_threshold: float | None,
    dedup_inter_dataset_policy: str,
    dedup_source_weights_path: Path | None,
    temp_root: Path,
    deduped_output_path: Path | None = None,
) -> tuple[Path, dict[str, Any] | None]:
    dedup_action = validate_dedup_action(dedup_action)
    if dedup_action not in {"drop_intra", "drop_intra_and_inter"}:
        return filtered_input_path, None
    _, _, family_membership_path = builder_bundle_paths(dedup_metadata_root)
    if family_membership_path is None:
        return filtered_input_path, None
    duplicate_rows_path = temp_root / "duplicate_rows.parquet"
    duplicate_doc_keys_path = temp_root / "duplicate_doc_keys.parquet"
    selected_duplicate_doc_keys_path = temp_root / "selected_duplicate_doc_keys.parquet"
    deduped_output_path = deduped_output_path or (temp_root / "deduped_mix_input.parquet")
    reduced_bundle_root = temp_root / "reduced_builder_bundle"
    con = _duckdb_connect_streaming()
    try:
        con.execute(
            f"""
            COPY (
                SELECT src.*
                FROM read_parquet({sql_quote(str(filtered_input_path))}) AS src
                JOIN read_parquet({sql_quote(str(family_membership_path))}) AS fam USING (doc_key)
                WHERE fam.family_size > 1
            ) TO {sql_quote(str(duplicate_rows_path))} (FORMAT parquet, COMPRESSION zstd)
            """
        )
    finally:
        con.close()
    duplicate_frame = pd.read_parquet(duplicate_rows_path)
    if duplicate_frame.empty:
        shutil.copy2(filtered_input_path, deduped_output_path)
        return deduped_output_path, {
            "dedup_action": dedup_action,
            "rows_before": 0,
            "rows_after": 0,
            "family_count": 0,
            "shared_family_count": 0,
            "bundle_root": str(dedup_metadata_root),
            "bundle_replay_mode": "family_membership",
        }
    prepare_reduced_builder_bundle(
        dedup_metadata_root=dedup_metadata_root,
        duplicate_rows_path=duplicate_rows_path,
        reduced_bundle_root=reduced_bundle_root,
    )
    duplicate_frame = duplicate_frame[CANONICAL_COLUMNS].copy()
    deduped_duplicate_frame, dedup_summary = apply_builder_dedup(
        duplicate_frame,
        dedup_metadata_root=reduced_bundle_root,
        dedup_action=dedup_action,
        dedup_exact_stage=dedup_exact_stage,
        dedup_similarity_threshold=dedup_similarity_threshold,
        dedup_inter_dataset_policy=dedup_inter_dataset_policy,
        dedup_source_weights_path=dedup_source_weights_path,
    )
    if "doc_key" in duplicate_frame.columns:
        duplicate_doc_keys = duplicate_frame[["doc_key"]].copy()
    else:
        duplicate_doc_keys = pd.DataFrame(
            {
                "doc_key": [
                    stable_doc_key(str(source_dataset), str(source_doc_id))
                    for source_dataset, source_doc_id in zip(
                        duplicate_frame["source_dataset"],
                        duplicate_frame["source_doc_id"],
                        strict=True,
                    )
                ]
            }
        )
    duplicate_doc_keys.drop_duplicates().to_parquet(duplicate_doc_keys_path, index=False)
    deduped_duplicate_frame[["doc_key"]].drop_duplicates().to_parquet(selected_duplicate_doc_keys_path, index=False)
    con = _duckdb_connect_streaming()
    try:
        con.execute(
            f"""
            COPY (
                SELECT src.*
                FROM read_parquet({sql_quote(str(filtered_input_path))}) AS src
                LEFT JOIN read_parquet({sql_quote(str(duplicate_doc_keys_path))}) AS dup USING (doc_key)
                LEFT JOIN read_parquet({sql_quote(str(selected_duplicate_doc_keys_path))}) AS keep USING (doc_key)
                WHERE dup.doc_key IS NULL OR keep.doc_key IS NOT NULL
                ORDER BY src.doc_key, src.source_dataset, src.source_doc_id
            ) TO {sql_quote(str(deduped_output_path))} (FORMAT parquet, COMPRESSION zstd)
            """
        )
    finally:
        con.close()
    return deduped_output_path, dedup_summary


def summarize_mix_intermediate_path(input_path: Path) -> dict[str, Any]:
    con = _duckdb_connect_streaming()
    try:
        rows, chars = con.execute(
            f"""
            SELECT count(*), coalesce(sum(source_mix_chars), 0)
            FROM read_parquet({sql_quote(str(input_path))})
            """
        ).fetchone()
    finally:
        con.close()
    return {
        "path": str(input_path),
        "rows": int(rows or 0),
        "chars": int(chars or 0),
    }


def materialize_streaming_mix_selected_input(
    output_root: Path,
    *,
    destination: Path,
    include_sources: list[str] | None,
    exclude_sources: list[str] | None,
    exclude_needs_ocr_sources: list[str] | None,
    quality_preset: str,
    historical_mode: str,
    math_mode: str,
    latex_mode: str,
    dedup_metadata_root: Path | None,
    dedup_action: str,
    dedup_exact_stage: str,
    dedup_similarity_threshold: float | None,
    dedup_inter_dataset_policy: str,
    dedup_source_weights_path: Path | None,
    exclude_doc_keys_path: Path | None = None,
) -> dict[str, Any]:
    ensure_dir(destination.parent)
    if destination.exists():
        destination.unlink()
    with tempfile.TemporaryDirectory(prefix="glossapi_mix_prelude_", dir=str(destination.parent)) as temp_dir:
        temp_root = Path(temp_dir)
        filtered_input_path = temp_root / "filtered_input.parquet"
        filtered_summary = materialize_filtered_mix_input(
            output_root,
            destination=filtered_input_path,
            include_sources=include_sources,
            exclude_sources=exclude_sources,
            exclude_needs_ocr_sources=exclude_needs_ocr_sources,
            quality_preset=quality_preset,
            historical_mode=historical_mode,
            math_mode=math_mode,
            latex_mode=latex_mode,
        )
        external_drop_summary: dict[str, Any] | None = None
        selected_for_dedup_path = filtered_input_path
        if exclude_doc_keys_path is not None:
            external_filtered_input_path = temp_root / "external_drop_filtered_input.parquet"
            external_drop_summary = materialize_doc_key_excluded_mix_input(
                filtered_input_path,
                exclude_doc_keys_path=exclude_doc_keys_path,
                destination=external_filtered_input_path,
            )
            selected_for_dedup_path = external_filtered_input_path
        dedup_summary: dict[str, Any] | None = None
        if dedup_metadata_root is not None and validate_dedup_action(dedup_action) in {"drop_intra", "drop_intra_and_inter"}:
            _, dedup_summary = materialize_drop_action_deduped_mix(
                selected_for_dedup_path,
                dedup_metadata_root=dedup_metadata_root,
                dedup_action=dedup_action,
                dedup_exact_stage=dedup_exact_stage,
                dedup_similarity_threshold=dedup_similarity_threshold,
                dedup_inter_dataset_policy=dedup_inter_dataset_policy,
                dedup_source_weights_path=dedup_source_weights_path,
                temp_root=temp_root,
                deduped_output_path=destination,
            )
        else:
            selected_for_dedup_path.replace(destination)
        selected_input_summary = summarize_mix_intermediate_path(destination)
        return {
            "filtered_input": filtered_summary,
            "external_drop_summary": external_drop_summary,
            "selected_input": selected_input_summary,
            "dedup_action": dedup_action,
            "dedup_summary": dedup_summary,
        }


def source_mix_entry_condition_sql(entry: Mapping[str, Any]) -> str:
    clauses: list[str] = []
    include_sources = entry.get("include_sources")
    exclude_sources = entry.get("exclude_sources")
    if include_sources:
        quoted = ", ".join(sql_quote(str(item)) for item in include_sources)
        clauses.append(f"source_dataset IN ({quoted})")
    if exclude_sources:
        quoted = ", ".join(sql_quote(str(item)) for item in exclude_sources)
        clauses.append(f"source_dataset NOT IN ({quoted})")
    if not clauses:
        return "TRUE"
    return " AND ".join(f"({clause})" for clause in clauses)


def combine_sql_conditions(*conditions: str | None) -> str:
    normalized = [condition for condition in conditions if condition and condition != "TRUE"]
    if not normalized:
        return "TRUE"
    return " AND ".join(f"({condition})" for condition in normalized)


def standard_training_filter_sql(
    column_names: set[str],
    *,
    table_alias: str | None = "src",
    badness_lt: float = DEFAULT_STANDARD_BADNESS_LT,
    mojibake_lte: float | None = DEFAULT_STANDARD_MOJIBAKE_LTE,
    allow_missing_badness_scores: bool = False,
    greek_ratio_gte: float | None = DEFAULT_STANDARD_GREEK_RATIO_GTE,
    require_non_empty_content: bool = True,
) -> tuple[str, dict[str, Any]]:
    predicates: list[str] = []
    column_prefix = f"{table_alias}." if table_alias else ""
    summary = {
        "badness_lt": badness_lt,
        "mojibake_lte": mojibake_lte,
        "allow_missing_badness_scores": allow_missing_badness_scores,
        "greek_ratio_gte": greek_ratio_gte,
        "require_non_empty_content": require_non_empty_content,
        "has_greek_badness": "greek_badness_score" in column_names,
        "has_mojibake_badness": "mojibake_badness_score" in column_names,
        "has_charset_greek": "charset_greek_ratio" in column_names,
        "has_content_chars_kept": "content_chars_kept" in column_names,
        "has_needs_ocr": "needs_ocr" in column_names,
        "has_ocr_success": "ocr_success" in column_names,
    }

    if "greek_badness_score" in column_names:
        greek_score = f"try_cast({column_prefix}greek_badness_score as DOUBLE)"
        if allow_missing_badness_scores:
            predicates.append(f"({greek_score} IS NULL OR {greek_score} < {float(badness_lt)})")
        else:
            predicates.append(f"({greek_score} IS NOT NULL AND {greek_score} < {float(badness_lt)})")
    elif not allow_missing_badness_scores:
        raise ValueError("input is missing greek_badness_score; cannot apply production badness filter")

    if mojibake_lte is not None:
        if "mojibake_badness_score" in column_names:
            mojibake_score = f"try_cast({column_prefix}mojibake_badness_score as DOUBLE)"
            if allow_missing_badness_scores:
                predicates.append(f"({mojibake_score} IS NULL OR {mojibake_score} <= {float(mojibake_lte)})")
            else:
                predicates.append(f"({mojibake_score} IS NOT NULL AND {mojibake_score} <= {float(mojibake_lte)})")
        elif not allow_missing_badness_scores:
            raise ValueError("input is missing mojibake_badness_score; cannot apply production badness filter")

    if greek_ratio_gte is not None and "charset_greek_ratio" in column_names:
        predicates.append(
            f"({column_prefix}charset_greek_ratio IS NULL OR {column_prefix}charset_greek_ratio >= {float(greek_ratio_gte)})"
        )

    if require_non_empty_content and "content_chars_kept" in column_names:
        predicates.append(f"({column_prefix}content_chars_kept IS NULL OR {column_prefix}content_chars_kept > 0)")

    if "needs_ocr" in column_names and "ocr_success" in column_names:
        predicates.append(
            f"CASE WHEN {column_prefix}source_dataset='openarchives.gr' "
            f"THEN coalesce({column_prefix}needs_ocr,false)=FALSE "
            f"ELSE (coalesce({column_prefix}needs_ocr,false)=FALSE OR coalesce({column_prefix}ocr_success,false)=TRUE) END"
        )
    elif "needs_ocr" in column_names:
        predicates.append(f"coalesce({column_prefix}needs_ocr,false)=FALSE")

    return " AND ".join(predicates) if predicates else "TRUE", summary


def materialize_standard_training_filtered_selected_input(
    input_path: Path,
    *,
    output_path: Path,
    badness_lt: float = DEFAULT_STANDARD_BADNESS_LT,
    mojibake_lte: float | None = DEFAULT_STANDARD_MOJIBAKE_LTE,
    allow_missing_badness_scores: bool = False,
    greek_ratio_gte: float | None = DEFAULT_STANDARD_GREEK_RATIO_GTE,
    require_non_empty_content: bool = True,
) -> dict[str, Any]:
    column_names = set(pq.read_schema(input_path).names)
    predicate_sql, filter_summary = standard_training_filter_sql(
        column_names,
        table_alias="src",
        badness_lt=badness_lt,
        mojibake_lte=mojibake_lte,
        allow_missing_badness_scores=allow_missing_badness_scores,
        greek_ratio_gte=greek_ratio_gte,
        require_non_empty_content=require_non_empty_content,
    )
    con = _duckdb_connect_streaming()
    try:
        rows_before, chars_before = con.execute(
            f"""
            SELECT count(*), coalesce(sum(source_mix_chars), 0)
            FROM read_parquet({sql_quote(str(input_path))})
            """
        ).fetchone()
        con.execute(
            f"""
            COPY (
                SELECT src.*
                FROM read_parquet({sql_quote(str(input_path))}) AS src
                WHERE {predicate_sql}
                ORDER BY src.doc_key, src.source_dataset, src.source_doc_id
            ) TO {sql_quote(str(output_path))} (FORMAT parquet, COMPRESSION zstd)
            """
        )
        rows_after, chars_after = con.execute(
            f"""
            SELECT count(*), coalesce(sum(source_mix_chars), 0)
            FROM read_parquet({sql_quote(str(output_path))})
            """
        ).fetchone()
    finally:
        con.close()
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows_before": int(rows_before or 0),
        "chars_before": int(chars_before or 0),
        "rows_after": int(rows_after or 0),
        "chars_after": int(chars_after or 0),
        **filter_summary,
    }


def summarize_standard_training_filter_for_selected_input(
    input_path: Path,
    *,
    badness_lt: float = DEFAULT_STANDARD_BADNESS_LT,
    mojibake_lte: float | None = DEFAULT_STANDARD_MOJIBAKE_LTE,
    allow_missing_badness_scores: bool = False,
    greek_ratio_gte: float | None = DEFAULT_STANDARD_GREEK_RATIO_GTE,
    require_non_empty_content: bool = True,
) -> tuple[str, dict[str, Any]]:
    column_names = set(pq.read_schema(input_path).names)
    predicate_sql, filter_summary = standard_training_filter_sql(
        column_names,
        table_alias=None,
        badness_lt=badness_lt,
        mojibake_lte=mojibake_lte,
        allow_missing_badness_scores=allow_missing_badness_scores,
        greek_ratio_gte=greek_ratio_gte,
        require_non_empty_content=require_non_empty_content,
    )
    con = _duckdb_connect_streaming()
    try:
        rows_before, chars_before = con.execute(
            f"""
            SELECT count(*), coalesce(sum(source_mix_chars), 0)
            FROM read_parquet({sql_quote(str(input_path))})
            """
        ).fetchone()
        rows_after, chars_after = con.execute(
            f"""
            SELECT count(*), coalesce(sum(source_mix_chars), 0)
            FROM read_parquet({sql_quote(str(input_path))})
            WHERE {predicate_sql}
            """
        ).fetchone()
    finally:
        con.close()
    return predicate_sql, {
        "input_path": str(input_path),
        "rows_before": int(rows_before or 0),
        "chars_before": int(chars_before or 0),
        "rows_after": int(rows_after or 0),
        "chars_after": int(chars_after or 0),
        **filter_summary,
    }


def select_source_mix_rows_to_parquet(
    *,
    input_path: Path,
    output_path: Path,
    entry: Mapping[str, Any],
    requested_chars_int: int,
    input_filter_sql: str | None = None,
) -> dict[str, Any]:
    condition = combine_sql_conditions(input_filter_sql, source_mix_entry_condition_sql(entry))
    con = _duckdb_connect_streaming()
    try:
        available_rows, available_chars = con.execute(
            f"""
            SELECT count(*), coalesce(sum(source_mix_chars), 0)
            FROM read_parquet({sql_quote(str(input_path))})
            WHERE {condition}
            """
        ).fetchone()
        available_rows = int(available_rows or 0)
        available_chars = int(available_chars or 0)
        if available_rows == 0:
            raise ValueError(f"source mix entry '{entry['name']}' matched zero rows")
        if requested_chars_int <= 0:
            pq.write_table(mix_intermediate_schema(include_component=True).empty_table(), output_path, compression="zstd")
            return {
                "name": str(entry["name"]),
                "fraction_mode": str(entry["fraction_mode"]),
                "fraction": float(entry["fraction"]),
                "include_sources": entry.get("include_sources"),
                "exclude_sources": entry.get("exclude_sources"),
                "available_rows": available_rows,
                "available_chars": available_chars,
                "requested_chars": float(requested_chars_int),
                "selected_rows": 0,
                "selected_chars": 0,
            }
        if float(entry["fraction"]) >= 1.0 or requested_chars_int >= available_chars:
            con.execute(
                f"""
                COPY (
                    SELECT {", ".join(CANONICAL_COLUMNS)}, doc_key, source_mix_chars, {sql_quote(str(entry['name']))} AS source_mix_component
                    FROM read_parquet({sql_quote(str(input_path))})
                    WHERE {condition}
                    ORDER BY doc_key, source_dataset, source_doc_id
                ) TO {sql_quote(str(output_path))} (FORMAT parquet, COMPRESSION zstd)
                """
            )
        else:
            crossing = con.execute(
                f"""
                WITH ordered AS (
                    SELECT
                        row_number() OVER (ORDER BY doc_key, source_dataset, source_doc_id) AS rn,
                        source_mix_chars
                    FROM read_parquet({sql_quote(str(input_path))})
                    WHERE {condition}
                ),
                scored AS (
                    SELECT
                        rn,
                        sum(source_mix_chars) OVER (ORDER BY rn ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_chars
                    FROM ordered
                ),
                boundary AS (
                    SELECT
                        rn,
                        cum_chars,
                        coalesce(lag(cum_chars, 1, 0) OVER (ORDER BY rn), 0) AS prev_cum
                    FROM scored
                )
                SELECT rn, cum_chars, prev_cum
                FROM boundary
                WHERE cum_chars >= ?
                ORDER BY rn
                LIMIT 1
                """,
                [requested_chars_int],
            ).fetchone()
            if crossing is None:
                boundary_rn = available_rows
            else:
                boundary_rn = int(crossing[0])
                cum_chars = int(crossing[1] or 0)
                prev_cum = int(crossing[2] or 0)
                before_diff = abs(requested_chars_int - prev_cum)
                after_diff = abs(requested_chars_int - cum_chars)
                if boundary_rn > 1 and after_diff >= before_diff:
                    boundary_rn -= 1
            con.execute(
                f"""
                COPY (
                    WITH ordered AS (
                        SELECT
                            {", ".join(CANONICAL_COLUMNS)},
                            doc_key,
                            source_mix_chars,
                            row_number() OVER (ORDER BY doc_key, source_dataset, source_doc_id) AS rn
                        FROM read_parquet({sql_quote(str(input_path))})
                        WHERE {condition}
                    )
                    SELECT
                        {", ".join(CANONICAL_COLUMNS)},
                        doc_key,
                        source_mix_chars,
                        {sql_quote(str(entry['name']))} AS source_mix_component
                    FROM ordered
                    WHERE rn <= {boundary_rn}
                    ORDER BY rn
                ) TO {sql_quote(str(output_path))} (FORMAT parquet, COMPRESSION zstd)
                """
            )
        selected_rows, selected_chars = con.execute(
            f"""
            SELECT count(*), coalesce(sum(source_mix_chars), 0)
            FROM read_parquet({sql_quote(str(output_path))})
            """
        ).fetchone()
    finally:
        con.close()
    return {
        "name": str(entry["name"]),
        "fraction_mode": str(entry["fraction_mode"]),
        "fraction": float(entry["fraction"]),
        "include_sources": entry.get("include_sources"),
        "exclude_sources": entry.get("exclude_sources"),
        "available_rows": available_rows,
        "available_chars": available_chars,
        "requested_chars": float(requested_chars_int),
        "selected_rows": int(selected_rows or 0),
        "selected_chars": int(selected_chars or 0),
    }


def apply_source_mix_to_parquet(
    input_path: Path,
    *,
    output_path: Path,
    source_mix_config_path: Path,
    temp_root: Path,
    input_filter_sql: str | None = None,
) -> dict[str, Any]:
    entries = load_source_mix_config(source_mix_config_path)
    base_condition = input_filter_sql or "TRUE"
    con = _duckdb_connect_streaming()
    try:
        rows_before, chars_before = con.execute(
            f"""
            SELECT count(*), coalesce(sum(source_mix_chars), 0)
            FROM read_parquet({sql_quote(str(input_path))})
            """
        ).fetchone()
        eligible_rows, eligible_chars = con.execute(
            f"""
            SELECT count(*), coalesce(sum(source_mix_chars), 0)
            FROM read_parquet({sql_quote(str(input_path))})
            WHERE {base_condition}
            """
        ).fetchone()
        rows_before = int(rows_before or 0)
        chars_before = int(chars_before or 0)
        eligible_rows = int(eligible_rows or 0)
        eligible_chars = int(eligible_chars or 0)
        overlap_sql = " UNION ALL ".join(
            [
                f"SELECT doc_key, {sql_quote(str(entry['name']))} AS entry_name FROM read_parquet({sql_quote(str(input_path))}) WHERE {combine_sql_conditions(base_condition, source_mix_entry_condition_sql(entry))}"
                for entry in entries
            ]
        )
        overlap = con.execute(
            f"""
            SELECT doc_key, count(DISTINCT entry_name) AS overlap_count
            FROM ({overlap_sql})
            GROUP BY doc_key
            HAVING overlap_count > 1
            LIMIT 1
            """
        ).fetchone()
    finally:
        con.close()
    if overlap is not None:
        raise ValueError("source mix entries overlap with one another; entries must be disjoint")
    resolved_entries: list[dict[str, Any]] = []
    con = _duckdb_connect_streaming()
    try:
        used_rows = 0
        used_chars = 0
        for entry in entries:
            condition = combine_sql_conditions(base_condition, source_mix_entry_condition_sql(entry))
            available_rows, available_chars = con.execute(
                f"""
                SELECT count(*), coalesce(sum(source_mix_chars), 0)
                FROM read_parquet({sql_quote(str(input_path))})
                WHERE {condition}
                """
            ).fetchone()
            available_rows = int(available_rows or 0)
            available_chars = int(available_chars or 0)
            if available_rows == 0:
                raise ValueError(f"source mix entry '{entry['name']}' matched zero rows")
            used_rows += available_rows
            used_chars += available_chars
            resolved_entries.append(
                {
                    **entry,
                    "available_rows": available_rows,
                    "available_chars": available_chars,
                }
            )
    finally:
        con.close()
    fixed_entries = [entry for entry in resolved_entries if entry["fraction_mode"] == "of_group"]
    share_entries = [entry for entry in resolved_entries if entry["fraction_mode"] == "of_total"]
    fixed_target_chars_total = 0.0
    for entry in fixed_entries:
        entry["requested_chars"] = float(entry["available_chars"]) * float(entry["fraction"])
        fixed_target_chars_total += float(entry["requested_chars"])
    target_mix_chars_total: float | None = None
    share_fraction_total = float(sum(float(entry["fraction"]) for entry in share_entries))
    if share_entries:
        if fixed_target_chars_total > 0:
            if share_fraction_total >= 1.0:
                raise ValueError("source mix config is infeasible: of_total fractions must sum to < 1 when of_group entries are present")
            target_mix_chars_total = fixed_target_chars_total / (1.0 - share_fraction_total)
        else:
            if not math.isclose(share_fraction_total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("source mix config with only of_total entries must have fractions summing to exactly 1.0")
            target_mix_chars_total = min(
                float(entry["available_chars"]) / float(entry["fraction"])
                for entry in share_entries
                if float(entry["fraction"]) > 0
            )
        for entry in share_entries:
            requested_chars = float(target_mix_chars_total) * float(entry["fraction"])
            if requested_chars > float(entry["available_chars"]) + 1e-6:
                raise ValueError(
                    f"source mix entry '{entry['name']}' cannot satisfy requested of_total share with available chars"
                )
            entry["requested_chars"] = requested_chars
    component_paths: list[Path] = []
    entry_summaries: list[dict[str, Any]] = []
    for entry in resolved_entries:
        component_path = temp_root / f"component_{entry['name']}.parquet"
        entry_summary = select_source_mix_rows_to_parquet(
            input_path=input_path,
            output_path=component_path,
            entry=entry,
            requested_chars_int=int(math.floor(float(entry["requested_chars"]) + 1e-9)),
            input_filter_sql=base_condition,
        )
        component_paths.append(component_path)
        entry_summaries.append(entry_summary)
    if component_paths:
        component_glob = str((temp_root / "component_*.parquet").resolve())
        con = _duckdb_connect_streaming()
        try:
            con.execute(
                f"""
                COPY (
                    SELECT {", ".join(CANONICAL_COLUMNS)}, source_mix_component
                    FROM read_parquet({sql_quote(component_glob)})
                    ORDER BY source_mix_component, doc_key, source_dataset, source_doc_id
                ) TO {sql_quote(str(output_path))} (FORMAT parquet, COMPRESSION zstd)
                """
            )
        finally:
            con.close()
    else:
        pq.write_table(mix_output_schema(include_component=True).empty_table(), output_path, compression="zstd")
    rows_after = int(sum(item["selected_rows"] for item in entry_summaries))
    chars_after = int(sum(item["selected_chars"] for item in entry_summaries))
    return {
        "config_path": str(source_mix_config_path),
        "rows_before": rows_before,
        "rows_after": rows_after,
        "chars_before": chars_before,
        "chars_after": chars_after,
        "eligible_rows_before_source_mix": eligible_rows,
        "eligible_chars_before_source_mix": eligible_chars,
        "filtered_out_rows_before_source_mix": int(rows_before - eligible_rows),
        "filtered_out_chars_before_source_mix": int(chars_before - eligible_chars),
        "unmatched_rows": int(eligible_rows - sum(entry["available_rows"] for entry in resolved_entries)),
        "unmatched_chars": int(eligible_chars - sum(entry["available_chars"] for entry in resolved_entries)),
        "target_mix_chars_total": None if target_mix_chars_total is None else float(target_mix_chars_total),
        "entries": entry_summaries,
    }


def summarize_mix_output(mix_output_path: Path) -> tuple[int, int, list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        return summarize_mix_output_duckdb(mix_output_path)
    except Exception:
        # Keep the old PyArrow/Pandas summarizer as a compatibility fallback
        # for older DuckDB builds that lack the regex/list functions used by
        # the parallel SQL path.
        return summarize_mix_output_pyarrow(mix_output_path)


def summarize_mix_output_duckdb(mix_output_path: Path) -> tuple[int, int, list[dict[str, Any]], list[dict[str, Any]]]:
    if os.environ.get("GLOSSAPI_FAST_MIX_SUMMARY") == "1":
        token_expr = (
            "CASE "
            "WHEN length(coalesce(CAST(text AS VARCHAR), '')) = 0 THEN 0 "
            "ELSE CAST(ceil(length(coalesce(CAST(text AS VARCHAR), '')) / 4.0) AS BIGINT) "
            "END"
        )
    else:
        token_expr = (
            "list_count(regexp_extract_all("
            "coalesce(CAST(text AS VARCHAR), ''), "
            r"'[\p{L}\p{N}_]+|[^\p{L}\p{N}_\s]'"
            "))"
        )
    con = _duckdb_connect_streaming()
    try:
        threads = max(1, os.cpu_count() or 1)
        con.execute(f"PRAGMA threads={threads}")
        path_sql = sql_quote(str(mix_output_path))
        columns = {
            str(row[0])
            for row in con.execute(
                f"DESCRIBE SELECT * FROM read_parquet({path_sql})"
            ).fetchall()
        }
        pool_enabled = {"dedup_pool_key", "dedup_is_shared_pool", "dedup_pool_source_count"}.issubset(columns)
        summary_columns = [
            "source_dataset",
            f"{token_expr} AS estimated_tokens",
        ]
        if pool_enabled:
            summary_columns.extend(
                [
                    "dedup_pool_key",
                    "dedup_is_shared_pool",
                    "dedup_pool_source_count",
                ]
            )
        con.execute(
            f"""
            CREATE TEMP TABLE mix_summary AS
            SELECT {", ".join(summary_columns)}
            FROM read_parquet({path_sql})
            """
        )
        total_row = con.execute(
            """
            SELECT count(*) AS rows, coalesce(sum(estimated_tokens), 0) AS estimated_tokens
            FROM mix_summary
            """
        ).fetchone()
        rows_kept = int(total_row[0] or 0)
        estimated_tokens_total = int(total_row[1] or 0)
        per_source_records = [
            {
                "source_dataset": str(row[0]),
                "rows": int(row[1] or 0),
                "estimated_tokens": int(row[2] or 0),
            }
            for row in con.execute(
                f"""
                SELECT
                    source_dataset,
                    count(*) AS rows,
                    coalesce(sum(estimated_tokens), 0) AS estimated_tokens
                FROM mix_summary
                GROUP BY source_dataset
                ORDER BY estimated_tokens DESC, rows DESC, source_dataset
                """
            ).fetchall()
        ]
        per_pool_records: list[dict[str, Any]] = []
        if pool_enabled:
            per_pool_records = [
                {
                    "dedup_pool_key": row[0],
                    "dedup_is_shared_pool": row[1],
                    "dedup_pool_source_count": row[2],
                    "rows": int(row[3] or 0),
                    "estimated_tokens": int(row[4] or 0),
                }
                for row in con.execute(
                    f"""
                    SELECT
                        dedup_pool_key,
                        dedup_is_shared_pool,
                        dedup_pool_source_count,
                        count(*) AS rows,
                        coalesce(sum(estimated_tokens), 0) AS estimated_tokens
                    FROM mix_summary
                    GROUP BY dedup_pool_key, dedup_is_shared_pool, dedup_pool_source_count
                    ORDER BY estimated_tokens DESC, rows DESC, dedup_pool_key
                    """
                ).fetchall()
            ]
        return rows_kept, estimated_tokens_total, per_source_records, per_pool_records
    finally:
        con.close()


def summarize_mix_output_pyarrow(mix_output_path: Path) -> tuple[int, int, list[dict[str, Any]], list[dict[str, Any]]]:
    parquet = pq.ParquetFile(mix_output_path)
    columns = set(parquet.schema_arrow.names)
    per_source: dict[str, dict[str, int]] = defaultdict(lambda: {"rows": 0, "estimated_tokens": 0})
    per_pool: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    rows_kept = 0
    estimated_tokens_total = 0
    summary_columns = ["source_dataset", "text"]
    pool_enabled = {"dedup_pool_key", "dedup_is_shared_pool", "dedup_pool_source_count"}.issubset(columns)
    if pool_enabled:
        summary_columns.extend(["dedup_pool_key", "dedup_is_shared_pool", "dedup_pool_source_count"])
    for batch in parquet.iter_batches(batch_size=2048, columns=summary_columns):
        frame = batch.to_pandas()
        if frame.empty:
            continue
        frame["estimated_tokens"] = frame["text"].fillna("").astype(str).map(estimate_text_tokens)
        rows_kept += int(len(frame))
        estimated_tokens_total += int(frame["estimated_tokens"].sum())
        grouped_source = frame.groupby("source_dataset", as_index=False).agg(
            rows=("text", "count"),
            estimated_tokens=("estimated_tokens", "sum"),
        )
        for row in grouped_source.to_dict("records"):
            stats = per_source[str(row["source_dataset"])]
            stats["rows"] += int(row["rows"])
            stats["estimated_tokens"] += int(row["estimated_tokens"])
        if pool_enabled:
            grouped_pool = frame.groupby(
                ["dedup_pool_key", "dedup_is_shared_pool", "dedup_pool_source_count"],
                dropna=False,
                as_index=False,
            ).agg(rows=("text", "count"), estimated_tokens=("estimated_tokens", "sum"))
            for row in grouped_pool.to_dict("records"):
                key = (
                    row["dedup_pool_key"],
                    row["dedup_is_shared_pool"],
                    row["dedup_pool_source_count"],
                )
                stats = per_pool.setdefault(
                    key,
                    {
                        "dedup_pool_key": row["dedup_pool_key"],
                        "dedup_is_shared_pool": row["dedup_is_shared_pool"],
                        "dedup_pool_source_count": row["dedup_pool_source_count"],
                        "rows": 0,
                        "estimated_tokens": 0,
                    },
                )
                stats["rows"] += int(row["rows"])
                stats["estimated_tokens"] += int(row["estimated_tokens"])
    per_source_records = [
        {"source_dataset": dataset, **stats}
        for dataset, stats in per_source.items()
    ]
    per_source_records.sort(key=lambda row: (-int(row["estimated_tokens"]), -int(row["rows"]), str(row["source_dataset"])))
    per_pool_records = list(per_pool.values())
    per_pool_records.sort(
        key=lambda row: (-int(row["estimated_tokens"]), -int(row["rows"]), str(row["dedup_pool_key"]))
    )
    return rows_kept, estimated_tokens_total, per_source_records, per_pool_records


def build_mix_output_from_selected_input(
    selected_input_path: Path,
    *,
    mix_output_path: Path,
    source_mix_config_path: Path | None,
    apply_standard_split_filters: bool = False,
    badness_lt: float = DEFAULT_STANDARD_BADNESS_LT,
    mojibake_lte: float | None = DEFAULT_STANDARD_MOJIBAKE_LTE,
    allow_missing_badness_scores: bool = False,
    greek_ratio_gte: float | None = DEFAULT_STANDARD_GREEK_RATIO_GTE,
    require_non_empty_content: bool = True,
) -> dict[str, Any]:
    ensure_dir(mix_output_path.parent)
    with tempfile.TemporaryDirectory(prefix="glossapi_mix_finalize_", dir=str(mix_output_path.parent)) as temp_dir:
        temp_root = Path(temp_dir)
        temp_mix_output_path = temp_root / mix_output_path.name
        source_input_path = selected_input_path
        input_filter_sql: str | None = None
        standard_split_filter_summary: dict[str, Any] | None = None
        if apply_standard_split_filters:
            input_filter_sql, standard_split_filter_summary = summarize_standard_training_filter_for_selected_input(
                selected_input_path,
                badness_lt=badness_lt,
                mojibake_lte=mojibake_lte,
                allow_missing_badness_scores=allow_missing_badness_scores,
                greek_ratio_gte=greek_ratio_gte,
                require_non_empty_content=require_non_empty_content,
            )
        source_mix_summary: dict[str, Any] | None = None
        if source_mix_config_path is not None:
            source_mix_summary = apply_source_mix_to_parquet(
                source_input_path,
                output_path=temp_mix_output_path,
                source_mix_config_path=source_mix_config_path,
                temp_root=temp_root,
                input_filter_sql=input_filter_sql,
            )
        else:
            con = _duckdb_connect_streaming()
            try:
                con.execute(
                    f"""
                    COPY (
                        SELECT {", ".join(CANONICAL_COLUMNS)}
                        FROM read_parquet({sql_quote(str(source_input_path))})
                        WHERE {input_filter_sql or "TRUE"}
                        ORDER BY doc_key, source_dataset, source_doc_id
                    ) TO {sql_quote(str(temp_mix_output_path))} (FORMAT parquet, COMPRESSION zstd)
                    """
                )
            finally:
                con.close()
        rows_kept, estimated_tokens_total, per_source_records, per_pool_records = summarize_mix_output(temp_mix_output_path)
        temp_mix_output_path.replace(mix_output_path)
        pd.DataFrame(per_source_records).to_csv(mix_output_path.with_suffix(".summary.csv"), index=False)
        if per_pool_records:
            pd.DataFrame(per_pool_records).to_csv(mix_output_path.with_suffix(".pool_summary.csv"), index=False)
        if source_mix_summary is not None:
            mix_output_path.with_suffix(".source_mix_summary.json").write_text(
                json.dumps(source_mix_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return {
            "rows_kept": int(rows_kept),
            "estimated_tokens": int(estimated_tokens_total),
            "per_source": per_source_records,
            "per_pool": per_pool_records,
            "standard_split_filter": standard_split_filter_summary,
            "source_mix": source_mix_summary,
        }


def build_mix_export_streaming(
    output_root: Path,
    mix_output_path: Path,
    *,
    include_sources: list[str] | None,
    exclude_sources: list[str] | None,
    exclude_needs_ocr_sources: list[str] | None,
    quality_preset: str,
    historical_mode: str,
    math_mode: str,
    latex_mode: str,
    dedup_metadata_root: Path | None,
    dedup_action: str,
    dedup_exact_stage: str,
    dedup_similarity_threshold: float | None,
    dedup_inter_dataset_policy: str,
    dedup_source_weights_path: Path | None,
    source_mix_config_path: Path | None,
    exclude_doc_keys_path: Path | None = None,
) -> dict[str, Any]:
    ensure_dir(mix_output_path.parent)
    with tempfile.TemporaryDirectory(prefix="glossapi_mix_", dir=str(mix_output_path.parent)) as temp_dir:
        selected_input_path = Path(temp_dir) / "selected_input.parquet"
        selected_input_payload = materialize_streaming_mix_selected_input(
            output_root,
            destination=selected_input_path,
            include_sources=include_sources,
            exclude_sources=exclude_sources,
            exclude_needs_ocr_sources=exclude_needs_ocr_sources,
            quality_preset=quality_preset,
            historical_mode=historical_mode,
            math_mode=math_mode,
            latex_mode=latex_mode,
            dedup_metadata_root=dedup_metadata_root,
            dedup_action=dedup_action,
            dedup_exact_stage=dedup_exact_stage,
            dedup_similarity_threshold=dedup_similarity_threshold,
            dedup_inter_dataset_policy=dedup_inter_dataset_policy,
            dedup_source_weights_path=dedup_source_weights_path,
            exclude_doc_keys_path=exclude_doc_keys_path,
        )
        output_payload = build_mix_output_from_selected_input(
            selected_input_path,
            mix_output_path=mix_output_path,
            source_mix_config_path=source_mix_config_path,
        )
        return {
            **output_payload,
            "dedup_action": dedup_action,
            "dedup_summary": selected_input_payload["dedup_summary"],
            "external_drop_summary": selected_input_payload["external_drop_summary"],
            "selected_input": selected_input_payload["selected_input"],
        }


class UnionFind:
    def __init__(self, members: Iterable[str]) -> None:
        self.parent = {member: member for member in members}

    def find(self, member: str) -> str:
        parent = self.parent[member]
        if parent != member:
            self.parent[member] = self.find(parent)
        return self.parent[member]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def load_dedup_source_weights(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dedup_source_weights_path must point to a JSON object")
    weights: dict[str, float] = {}
    for key, value in payload.items():
        weight = float(value)
        if weight <= 0:
            raise ValueError("all dedup source weights must be > 0")
        weights[str(key)] = weight
    return weights


def load_builder_dedup_bundle(dedup_metadata_root: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest_path = dedup_metadata_root / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"builder dedup manifest missing under {dedup_metadata_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    doc_metadata_path = dedup_metadata_root / str(files.get("doc_metadata", "doc_dedup_metadata.parquet"))
    family_membership_rel = files.get("family_membership", "dedup_family_membership.parquet")
    family_membership_path = dedup_metadata_root / str(family_membership_rel)
    near_pairs_path = dedup_metadata_root / str(files.get("near_candidate_pairs", "near_candidate_pairs.parquet"))
    if not doc_metadata_path.exists():
        raise ValueError(f"builder doc metadata missing under {dedup_metadata_root}")
    family_membership = pd.read_parquet(family_membership_path) if family_membership_path.exists() else pd.DataFrame()
    near_pairs = pd.DataFrame()
    # The builder_metadata_v2 path prefers exported family membership. We still keep
    # near_candidate_pairs.parquet in the bundle as an evidence/audit artifact, but
    # only load it here for legacy bundles or empty membership exports.
    if family_membership.empty and near_pairs_path.exists():
        near_pairs = pd.read_parquet(near_pairs_path)
    return manifest, pd.read_parquet(doc_metadata_path), family_membership, near_pairs


def dedup_optional_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def dedup_optional_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return None


def dedup_selection_length_value(row: Mapping[str, Any]) -> int:
    len_greek = dedup_optional_int(row.get("dedup_len_greek", row.get("len_greek")))
    if len_greek is not None:
        return len_greek
    text_length_value = row.get("dedup_text_length_for_selection")
    parsed_text_length = dedup_optional_int(text_length_value)
    if parsed_text_length is not None:
        return parsed_text_length
    return len(clean_text(row.get("text")))


def dedup_representative_score(row: Mapping[str, Any]) -> float:
    exported = dedup_optional_float(row.get("dedup_representative_score"))
    if exported is not None:
        return exported
    badness = dedup_optional_float(row.get("dedup_greek_badness_score", row.get("greek_badness_score")))
    if badness is None:
        return 0.0
    return max(0.0, float(dedup_selection_length_value(row)) * (1.0 - (badness / 10.0)))


def dedup_selection_priority(row: Mapping[str, Any]) -> tuple[float, float, float, float, float, float, int, int, str, str]:
    needs_ocr = row.get("dedup_needs_ocr", row.get("needs_ocr"))
    if needs_ocr is False or needs_ocr == 0:
        needs_ocr_rank = 0.0
    elif needs_ocr is True or needs_ocr == 1:
        needs_ocr_rank = 1.0
    else:
        needs_ocr_rank = 0.5
    ocr_success = row.get("dedup_ocr_success", row.get("ocr_success"))
    invalid_ocr_rank = 1.0 if (needs_ocr is True or needs_ocr == 1) and not (ocr_success is True or ocr_success == 1) else 0.0
    greek_rank = dedup_optional_float(row.get("dedup_greek_badness_score", row.get("greek_badness_score")))
    if greek_rank is None:
        greek_rank = float("inf")
    mojibake_rank = dedup_optional_float(row.get("dedup_mojibake_badness_score", row.get("mojibake_badness_score")))
    if mojibake_rank is None:
        mojibake_rank = float("inf")
    if ocr_success is True or ocr_success == 1:
        ocr_rank = 0.0
    elif ocr_success is False or ocr_success == 0:
        ocr_rank = 1.0
    else:
        ocr_rank = 0.5
    has_title = row.get("dedup_has_title")
    if has_title is None:
        title_rank = 0 if normalize_title(row.get("title")) else 1
    else:
        title_rank = 0 if bool(has_title) else 1
    has_author = row.get("dedup_has_author")
    if has_author is None:
        author_rank = 0 if normalize_author(row.get("author")) else 1
    else:
        author_rank = 0 if bool(has_author) else 1
    return (
        invalid_ocr_rank,
        needs_ocr_rank,
        -dedup_representative_score(row),
        greek_rank,
        mojibake_rank,
        ocr_rank,
        title_rank,
        author_rank,
        str(row.get("source_dataset") or ""),
        str(row.get("source_doc_id") or ""),
    )


def dedup_mass_chars(row: Mapping[str, Any]) -> int:
    chars_value = row.get("chars")
    try:
        if chars_value is not None and not pd.isna(chars_value):
            return max(0, int(chars_value))
    except Exception:
        pass
    text_value = row.get("text")
    if text_value is None:
        return 0
    return len(str(text_value))


def select_best_doc_key(member_doc_keys: list[str], rows_by_doc_key: Mapping[str, Mapping[str, Any]]) -> str:
    return min(member_doc_keys, key=lambda doc_key: dedup_selection_priority(rows_by_doc_key[doc_key]))


def _resolve_builder_thresholds(
    manifest: Mapping[str, Any],
    dedup_similarity_threshold: float | None,
) -> tuple[float, float]:
    candidate_score_floor = float(manifest.get("candidate_score_floor", manifest.get("builder_default_threshold", 0.0)))
    if dedup_similarity_threshold is None:
        dedup_similarity_threshold = float(manifest.get("builder_default_threshold", candidate_score_floor))
    if float(dedup_similarity_threshold) < candidate_score_floor:
        raise ValueError(
            f"dedup_similarity_threshold={dedup_similarity_threshold} is below the exported candidate floor {candidate_score_floor}"
        )
    return candidate_score_floor, float(dedup_similarity_threshold)


def _family_groups_from_membership(
    *,
    family_membership: pd.DataFrame,
    doc_keys_in_frame: set[str],
    rows_by_doc_key: Mapping[str, Mapping[str, Any]],
    same_source_only: bool,
) -> list[list[str]]:
    grouped_members: dict[str, list[str]] = defaultdict(list)
    if not family_membership.empty:
        subset = family_membership[family_membership["doc_key"].isin(doc_keys_in_frame)]
        for row in subset[["family_id", "doc_key"]].to_dict("records"):
            family_id = str(row["family_id"])
            grouped_members[family_id].append(str(row["doc_key"]))
    covered_doc_keys: set[str] = set()
    family_groups: list[list[str]] = []
    for family_id in sorted(grouped_members):
        member_doc_keys = sorted(set(grouped_members[family_id]))
        covered_doc_keys.update(member_doc_keys)
        if same_source_only:
            per_source: dict[str, list[str]] = defaultdict(list)
            for doc_key in member_doc_keys:
                per_source[str(rows_by_doc_key[doc_key]["source_dataset"])].append(doc_key)
            for source_dataset in sorted(per_source):
                family_groups.append(sorted(per_source[source_dataset]))
        else:
            family_groups.append(member_doc_keys)
    for doc_key in sorted(doc_keys_in_frame - covered_doc_keys):
        family_groups.append([doc_key])
    return family_groups


def _legacy_family_groups_from_bundle(
    *,
    enriched: pd.DataFrame,
    near_pairs: pd.DataFrame,
    doc_keys_in_frame: set[str],
    dedup_exact_stage: str,
    dedup_similarity_threshold: float,
    same_source_only: bool,
) -> list[list[str]]:
    union_find = UnionFind(doc_keys_in_frame)
    rows_by_doc_key = {str(row["doc_key"]): row for row in enriched.to_dict("records")}
    metadata_by_doc_key = {
        str(row["doc_key"]): row
        for row in enriched[
            [
                "doc_key",
                "source_dataset",
                "source_doc_id",
                "dedup_strict_exact_group_hash",
                "dedup_strict_exact_group_size",
                "dedup_relaxed_exact_group_hash",
                "dedup_relaxed_exact_group_size",
            ]
        ].to_dict("records")
    }
    group_specs = [("dedup_strict_exact_group_hash", "dedup_strict_exact_group_size")]
    if dedup_exact_stage == "strict_and_relaxed":
        group_specs.append(("dedup_relaxed_exact_group_hash", "dedup_relaxed_exact_group_size"))
    for hash_column, size_column in group_specs:
        grouped_members: dict[str, list[str]] = defaultdict(list)
        for doc_key, metadata in metadata_by_doc_key.items():
            group_hash = metadata.get(hash_column)
            group_size = metadata.get(size_column)
            if group_hash in (None, ""):
                continue
            try:
                if int(group_size or 0) <= 1:
                    continue
            except Exception:
                continue
            grouped_members[str(group_hash)].append(doc_key)
        for member_doc_keys in grouped_members.values():
            if len(member_doc_keys) <= 1:
                continue
            if same_source_only:
                per_source: dict[str, list[str]] = defaultdict(list)
                for doc_key in member_doc_keys:
                    per_source[str(rows_by_doc_key[doc_key]["source_dataset"])].append(doc_key)
                groups_to_union = per_source.values()
            else:
                groups_to_union = [member_doc_keys]
            for group_doc_keys in groups_to_union:
                if len(group_doc_keys) <= 1:
                    continue
                anchor = group_doc_keys[0]
                for doc_key in group_doc_keys[1:]:
                    union_find.union(anchor, doc_key)
    if not near_pairs.empty:
        filtered_pairs = near_pairs[
            (near_pairs["estimated_jaccard"] >= float(dedup_similarity_threshold))
            & (near_pairs["doc_key_left"].isin(doc_keys_in_frame))
            & (near_pairs["doc_key_right"].isin(doc_keys_in_frame))
        ]
        if same_source_only:
            filtered_pairs = filtered_pairs[filtered_pairs["source_dataset_left"] == filtered_pairs["source_dataset_right"]]
        for row in filtered_pairs[["doc_key_left", "doc_key_right"]].itertuples(index=False):
            union_find.union(str(row.doc_key_left), str(row.doc_key_right))
    members_by_root: dict[str, list[str]] = defaultdict(list)
    for doc_key in sorted(doc_keys_in_frame):
        members_by_root[union_find.find(doc_key)].append(doc_key)
    return [sorted(member_doc_keys) for member_doc_keys in members_by_root.values()]


def _annotate_builder_families(
    *,
    enriched: pd.DataFrame,
    family_groups: list[list[str]],
    dedup_inter_dataset_policy: str,
    source_weights: Mapping[str, float],
    dedup_similarity_threshold: float,
    candidate_score_floor: float,
) -> tuple[pd.DataFrame, int]:
    rows_by_doc_key = {str(row["doc_key"]): row for row in enriched.to_dict("records")}
    family_infos: dict[str, dict[str, Any]] = {}
    for member_doc_keys in family_groups:
        normalized_doc_keys = sorted({str(doc_key) for doc_key in member_doc_keys if str(doc_key) in rows_by_doc_key})
        if not normalized_doc_keys:
            continue
        source_datasets = sorted({str(rows_by_doc_key[doc_key]["source_dataset"]) for doc_key in normalized_doc_keys})
        family_id = stable_doc_key("family", "\0".join(normalized_doc_keys))
        family_infos[family_id] = {
            "family_id": family_id,
            "member_doc_keys": normalized_doc_keys,
            "pool_key": f"unique:{source_datasets[0]}" if len(source_datasets) == 1 else f"shared:{'+'.join(source_datasets)}",
            "source_datasets": source_datasets,
        }
    representative_by_family: dict[str, str] = {}
    for family_id, family in family_infos.items():
        if len(family["source_datasets"]) == 1 or dedup_inter_dataset_policy == "quality_first":
            representative_by_family[family_id] = select_best_doc_key(family["member_doc_keys"], rows_by_doc_key)
    if dedup_inter_dataset_policy == "share_aware":
        families_by_pool: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
        for family_id, family in family_infos.items():
            if len(family["source_datasets"]) > 1:
                families_by_pool[family["pool_key"]].append((family_id, family))
        for pool_key, pool_families in sorted(families_by_pool.items()):
            del pool_key
            pool_datasets = sorted({dataset for _, family in pool_families for dataset in family["source_datasets"]})
            if source_weights:
                pool_weights = {dataset: float(source_weights.get(dataset, 1.0)) for dataset in pool_datasets}
            else:
                pool_weights = {dataset: 1.0 for dataset in pool_datasets}
            pool_chars: dict[str, int] = defaultdict(int)
            for family_id, family in sorted(pool_families, key=lambda item: item[1]["family_id"]):
                candidates_by_dataset: dict[str, list[str]] = defaultdict(list)
                for doc_key in family["member_doc_keys"]:
                    candidates_by_dataset[str(rows_by_doc_key[doc_key]["source_dataset"])].append(doc_key)
                best_doc_key_by_dataset = {
                    dataset: select_best_doc_key(member_doc_keys, rows_by_doc_key)
                    for dataset, member_doc_keys in candidates_by_dataset.items()
                }
                chosen_dataset = min(
                    sorted(best_doc_key_by_dataset),
                    key=lambda dataset: (
                        (pool_chars[dataset] + dedup_mass_chars(rows_by_doc_key[best_doc_key_by_dataset[dataset]]))
                        / pool_weights[dataset],
                        pool_chars[dataset] / pool_weights[dataset],
                        dataset,
                    ),
                )
                representative_doc_key = best_doc_key_by_dataset[chosen_dataset]
                representative_by_family[family_id] = representative_doc_key
                pool_chars[chosen_dataset] += dedup_mass_chars(rows_by_doc_key[representative_doc_key])
    annotation_rows: list[dict[str, Any]] = []
    for family_id in sorted(family_infos):
        family = family_infos[family_id]
        representative_doc_key = representative_by_family[family_id]
        for doc_key in family["member_doc_keys"]:
            annotation_rows.append(
                {
                    "doc_key": doc_key,
                    "dedup_family_id": family["family_id"],
                    "dedup_family_size": len(family["member_doc_keys"]),
                    "dedup_pool_key": family["pool_key"],
                    "dedup_pool_source_count": len(family["source_datasets"]),
                    "dedup_is_shared_pool": len(family["source_datasets"]) > 1,
                    "dedup_representative_doc_key": representative_doc_key,
                    "dedup_family_role": "representative" if doc_key == representative_doc_key else "member",
                    "dedup_similarity_threshold": float(dedup_similarity_threshold),
                    "dedup_candidate_score_floor": float(candidate_score_floor),
                }
            )
    annotation_frame = pd.DataFrame(annotation_rows)
    if annotation_frame.empty:
        annotation_frame = pd.DataFrame(
            columns=[
                "doc_key",
                "dedup_family_id",
                "dedup_family_size",
                "dedup_pool_key",
                "dedup_pool_source_count",
                "dedup_is_shared_pool",
                "dedup_representative_doc_key",
                "dedup_family_role",
                "dedup_similarity_threshold",
                "dedup_candidate_score_floor",
            ]
        )
    enriched = enriched.merge(annotation_frame, on="doc_key", how="left")
    shared_family_count = int(sum(1 for family in family_infos.values() if len(family["source_datasets"]) > 1))
    return enriched, shared_family_count


def _annotate_builder_families_from_membership_fast(
    *,
    enriched: pd.DataFrame,
    family_membership: pd.DataFrame,
    dedup_inter_dataset_policy: str,
    source_weights: Mapping[str, float],
    dedup_similarity_threshold: float,
    candidate_score_floor: float,
) -> tuple[pd.DataFrame, int]:
    """Annotate builder families without materializing every retained row as a dict.

    The builder metadata v2 family-membership file already contains one row per
    source document. Most families are singletons, so the old all-row
    ``to_dict("records")`` path was paying Python-object cost for tens of
    millions of rows that only need vectorized defaults. We keep the original
    representative selection semantics for non-singleton families and only
    materialize those rows.
    """

    if enriched.empty:
        return enriched.copy(), 0

    membership_cols = [
        "doc_key",
        "family_id",
        "family_size",
        "family_source_count",
        "family_mixed_source",
        "canonical_kept_doc_key",
    ]
    available_membership_cols = [column for column in membership_cols if column in family_membership.columns]
    membership = family_membership[available_membership_cols].copy()
    membership["doc_key"] = membership["doc_key"].astype(str)
    enriched = enriched.merge(membership, on="doc_key", how="left", sort=False)

    enriched["family_id"] = enriched["family_id"].where(
        enriched["family_id"].notna() & (enriched["family_id"].astype(str) != ""),
        enriched["doc_key"],
    )
    enriched["family_id"] = enriched["family_id"].astype(str)
    enriched["family_size"] = enriched["family_size"].fillna(1).astype("int64")
    enriched["family_source_count"] = enriched["family_source_count"].fillna(1).astype("int64")
    enriched["family_mixed_source"] = enriched["family_mixed_source"].fillna(False).astype(bool)

    enriched["dedup_family_id"] = enriched["family_id"]
    enriched["dedup_family_size"] = enriched["family_size"]
    enriched["dedup_pool_source_count"] = enriched["family_source_count"]
    enriched["dedup_is_shared_pool"] = enriched["family_source_count"] > 1
    enriched["dedup_pool_key"] = "unique:" + enriched["source_dataset"].astype(str)
    enriched["dedup_representative_doc_key"] = enriched["canonical_kept_doc_key"].where(
        enriched["canonical_kept_doc_key"].notna() & (enriched["canonical_kept_doc_key"].astype(str) != ""),
        enriched["doc_key"],
    )
    enriched["dedup_representative_doc_key"] = enriched["dedup_representative_doc_key"].astype(str)
    enriched["dedup_similarity_threshold"] = float(dedup_similarity_threshold)
    enriched["dedup_candidate_score_floor"] = float(candidate_score_floor)

    non_singleton_mask = enriched["family_size"] > 1
    shared_family_count = int(enriched.loc[enriched["family_source_count"] > 1, "family_id"].nunique())

    if non_singleton_mask.any():
        non_singleton_family_ids = set(enriched.loc[non_singleton_mask, "family_id"].astype(str))
        non_singleton_docs = set(enriched.loc[non_singleton_mask, "doc_key"].astype(str))
        membership_non_singleton = family_membership[
            family_membership["family_id"].astype(str).isin(non_singleton_family_ids)
            & family_membership["doc_key"].astype(str).isin(non_singleton_docs)
        ][["family_id", "doc_key"]].copy()
        membership_non_singleton["family_id"] = membership_non_singleton["family_id"].astype(str)
        membership_non_singleton["doc_key"] = membership_non_singleton["doc_key"].astype(str)

        needed_columns = [
            "doc_key",
            "source_dataset",
            "source_doc_id",
            "chars",
            "dedup_needs_ocr",
            "dedup_ocr_success",
            "dedup_greek_badness_score",
            "dedup_mojibake_badness_score",
            "dedup_has_title",
            "dedup_has_author",
            "dedup_representative_score",
            "dedup_text_length_for_selection",
            "needs_ocr",
            "ocr_success",
            "greek_badness_score",
            "mojibake_badness_score",
            "title",
            "author",
        ]
        compact_columns = [column for column in needed_columns if column in enriched.columns]
        compact = enriched.loc[non_singleton_mask, compact_columns].copy()
        rows_by_doc_key = {str(row["doc_key"]): row for row in compact.to_dict("records")}

        representative_by_family: dict[str, str] = {}
        stable_family_id_by_family: dict[str, str] = {}
        pool_key_by_family: dict[str, str] = {}
        families_by_pool: dict[str, list[tuple[str, str, list[str], list[str]]]] = defaultdict(list)

        for family_id, member_frame in membership_non_singleton.groupby("family_id", sort=True):
            member_doc_keys = sorted({str(doc_key) for doc_key in member_frame["doc_key"] if str(doc_key) in rows_by_doc_key})
            if not member_doc_keys:
                continue
            source_datasets = sorted({str(rows_by_doc_key[doc_key]["source_dataset"]) for doc_key in member_doc_keys})
            stable_family_id = stable_doc_key("family", "\0".join(member_doc_keys))
            pool_key = f"unique:{source_datasets[0]}" if len(source_datasets) == 1 else f"shared:{'+'.join(source_datasets)}"
            stable_family_id_by_family[family_id] = stable_family_id
            pool_key_by_family[family_id] = pool_key
            if len(source_datasets) == 1 or dedup_inter_dataset_policy == "quality_first":
                representative_by_family[family_id] = select_best_doc_key(member_doc_keys, rows_by_doc_key)
            else:
                families_by_pool[pool_key].append((stable_family_id, family_id, member_doc_keys, source_datasets))

        if dedup_inter_dataset_policy == "share_aware":
            for pool_key, pool_families in sorted(families_by_pool.items()):
                del pool_key
                pool_datasets = sorted({dataset for _, _, _, source_datasets in pool_families for dataset in source_datasets})
                if source_weights:
                    pool_weights = {dataset: float(source_weights.get(dataset, 1.0)) for dataset in pool_datasets}
                else:
                    pool_weights = {dataset: 1.0 for dataset in pool_datasets}
                pool_chars: dict[str, int] = defaultdict(int)
                for _, family_id, member_doc_keys, _ in sorted(pool_families, key=lambda item: item[0]):
                    candidates_by_dataset: dict[str, list[str]] = defaultdict(list)
                    for doc_key in member_doc_keys:
                        candidates_by_dataset[str(rows_by_doc_key[doc_key]["source_dataset"])].append(doc_key)
                    best_doc_key_by_dataset = {
                        dataset: select_best_doc_key(dataset_doc_keys, rows_by_doc_key)
                        for dataset, dataset_doc_keys in candidates_by_dataset.items()
                    }
                    chosen_dataset = min(
                        sorted(best_doc_key_by_dataset),
                        key=lambda dataset: (
                            (pool_chars[dataset] + dedup_mass_chars(rows_by_doc_key[best_doc_key_by_dataset[dataset]]))
                            / pool_weights[dataset],
                            pool_chars[dataset] / pool_weights[dataset],
                            dataset,
                        ),
                    )
                    representative_doc_key = best_doc_key_by_dataset[chosen_dataset]
                    representative_by_family[family_id] = representative_doc_key
                    pool_chars[chosen_dataset] += dedup_mass_chars(rows_by_doc_key[representative_doc_key])

        if stable_family_id_by_family:
            mapped_family_id = enriched.loc[non_singleton_mask, "family_id"].map(stable_family_id_by_family)
            enriched.loc[non_singleton_mask, "dedup_family_id"] = mapped_family_id.fillna(
                enriched.loc[non_singleton_mask, "dedup_family_id"]
            )
        if pool_key_by_family:
            mapped_pool_key = enriched.loc[non_singleton_mask, "family_id"].map(pool_key_by_family)
            enriched.loc[non_singleton_mask, "dedup_pool_key"] = mapped_pool_key.fillna(
                enriched.loc[non_singleton_mask, "dedup_pool_key"]
            )
        if representative_by_family:
            mapped_representative = enriched.loc[non_singleton_mask, "family_id"].map(representative_by_family)
            enriched.loc[non_singleton_mask, "dedup_representative_doc_key"] = mapped_representative.fillna(
                enriched.loc[non_singleton_mask, "dedup_representative_doc_key"]
            )

    enriched["dedup_family_role"] = "member"
    enriched.loc[enriched["doc_key"].astype(str) == enriched["dedup_representative_doc_key"].astype(str), "dedup_family_role"] = "representative"

    return enriched.drop(
        columns=[
            "family_id",
            "family_size",
            "family_source_count",
            "family_mixed_source",
            "canonical_kept_doc_key",
        ],
        errors="ignore",
    ), shared_family_count


def apply_builder_dedup(
    frame: pd.DataFrame,
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
    if frame.empty:
        return frame.copy(), {"dedup_action": dedup_action, "rows_before": 0, "rows_after": 0}
    manifest, doc_metadata, family_membership, near_pairs = load_builder_dedup_bundle(dedup_metadata_root)
    candidate_score_floor, dedup_similarity_threshold = _resolve_builder_thresholds(
        manifest,
        dedup_similarity_threshold,
    )
    source_weights = load_dedup_source_weights(dedup_source_weights_path)
    enriched = frame.copy()
    if "doc_key" in enriched.columns and not enriched["doc_key"].isna().any():
        enriched["doc_key"] = enriched["doc_key"].astype(str)
        doc_columns = [column for column in doc_metadata.columns if column not in {"doc_key", "source_dataset", "source_doc_id"}]
        enriched = enriched.merge(
            doc_metadata.rename(columns={column: f"dedup_{column}" for column in doc_columns}),
            on=["doc_key", "source_dataset", "source_doc_id"],
            how="left",
        )
    else:
        # The builder input normally carries source ids but not doc_key. The
        # dedup bundle already has doc_key for every source id, so join it in
        # directly instead of hashing tens of millions of ids in Python.
        doc_columns = [column for column in doc_metadata.columns if column not in {"source_dataset", "source_doc_id"}]
        enriched = enriched.merge(
            doc_metadata.rename(columns={column: f"dedup_{column}" for column in doc_columns if column != "doc_key"}),
            on=["source_dataset", "source_doc_id"],
            how="left",
        )
        if "doc_key" not in enriched.columns:
            raise KeyError("dedup metadata merge did not provide doc_key")
        missing_doc_key = enriched["doc_key"].isna()
        if missing_doc_key.any():
            enriched.loc[missing_doc_key, "doc_key"] = [
                stable_doc_key(str(source_dataset), str(source_doc_id))
                for source_dataset, source_doc_id in zip(
                    enriched.loc[missing_doc_key, "source_dataset"],
                    enriched.loc[missing_doc_key, "source_doc_id"],
                    strict=True,
                )
            ]
        enriched["doc_key"] = enriched["doc_key"].astype(str)
    same_source_only = dedup_action == "drop_intra"
    requested_exact_stage = str(manifest.get("builder_exact_stage", dedup_exact_stage))
    requested_threshold = float(manifest.get("builder_default_threshold", candidate_score_floor))
    use_family_membership = (
        not family_membership.empty
        and requested_exact_stage == dedup_exact_stage
        and abs(requested_threshold - float(dedup_similarity_threshold)) < 1e-9
    )
    if use_family_membership and not same_source_only:
        enriched, shared_family_count = _annotate_builder_families_from_membership_fast(
            enriched=enriched,
            family_membership=family_membership,
            dedup_inter_dataset_policy=dedup_inter_dataset_policy,
            source_weights=source_weights,
            dedup_similarity_threshold=float(dedup_similarity_threshold),
            candidate_score_floor=float(candidate_score_floor),
        )
        rows_before = len(enriched)
        if dedup_action in {"drop_intra", "drop_intra_and_inter"}:
            enriched = enriched[enriched["doc_key"] == enriched["dedup_representative_doc_key"]].copy()
        enriched = enriched.reset_index(drop=True)
        return enriched, {
            "dedup_action": dedup_action,
            "dedup_exact_stage": dedup_exact_stage,
            "dedup_similarity_threshold": float(dedup_similarity_threshold),
            "dedup_inter_dataset_policy": dedup_inter_dataset_policy,
            "rows_before": int(rows_before),
            "rows_after": int(len(enriched)),
            "family_count": int(family_membership["family_id"].nunique()),
            "shared_family_count": int(shared_family_count),
            "bundle_root": str(dedup_metadata_root),
            "bundle_replay_mode": "family_membership",
        }

    doc_keys_in_frame = set(enriched["doc_key"].astype(str))
    rows_by_doc_key = {str(row["doc_key"]): row for row in enriched.to_dict("records")}
    if use_family_membership:
        family_groups = _family_groups_from_membership(
            family_membership=family_membership,
            doc_keys_in_frame=doc_keys_in_frame,
            rows_by_doc_key=rows_by_doc_key,
            same_source_only=same_source_only,
        )
        replay_mode = "family_membership"
    else:
        family_groups = _legacy_family_groups_from_bundle(
            enriched=enriched,
            near_pairs=near_pairs,
            doc_keys_in_frame=doc_keys_in_frame,
            dedup_exact_stage=dedup_exact_stage,
            dedup_similarity_threshold=float(dedup_similarity_threshold),
            same_source_only=same_source_only,
        )
        replay_mode = "legacy_pairs"
    enriched, shared_family_count = _annotate_builder_families(
        enriched=enriched,
        family_groups=family_groups,
        dedup_inter_dataset_policy=dedup_inter_dataset_policy,
        source_weights=source_weights,
        dedup_similarity_threshold=float(dedup_similarity_threshold),
        candidate_score_floor=float(candidate_score_floor),
    )
    rows_before = len(enriched)
    if dedup_action in {"drop_intra", "drop_intra_and_inter"}:
        enriched = enriched[enriched["doc_key"] == enriched["dedup_representative_doc_key"]].copy()
    enriched = enriched.reset_index(drop=True)
    return enriched, {
        "dedup_action": dedup_action,
        "dedup_exact_stage": dedup_exact_stage,
        "dedup_similarity_threshold": float(dedup_similarity_threshold),
        "dedup_inter_dataset_policy": dedup_inter_dataset_policy,
        "rows_before": int(rows_before),
        "rows_after": int(len(enriched)),
        "family_count": int(len(family_groups)),
        "shared_family_count": int(shared_family_count),
        "bundle_root": str(dedup_metadata_root),
        "bundle_replay_mode": replay_mode,
    }


def load_filtered_mix(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    include_sources: list[str] | None = None,
    exclude_sources: list[str] | None = None,
    exclude_needs_ocr_sources: list[str] | None = None,
    quality_preset: str = "none",
    historical_mode: str = "include",
    math_mode: str = "include",
    latex_mode: str = "include",
    dedup_metadata_root: Path | None = None,
    dedup_action: str = "ignore",
    dedup_exact_stage: str = "strict_and_relaxed",
    dedup_similarity_threshold: float | None = None,
    dedup_inter_dataset_policy: str = "share_aware",
    dedup_source_weights_path: Path | None = None,
) -> pd.DataFrame:
    frame = read_all_canonical_frames(output_root)
    if include_sources:
        frame = frame[frame["source_dataset"].isin(include_sources)]
    if exclude_sources:
        frame = frame[~frame["source_dataset"].isin(exclude_sources)]
    if exclude_needs_ocr_sources:
        frame = frame[
            ~(
                frame["source_dataset"].isin(exclude_needs_ocr_sources)
                & frame["needs_ocr"].fillna(False)
            )
        ]
    frame = apply_quality_preset(frame, quality_preset)
    if historical_mode == "exclude":
        frame = frame[frame["is_historical_or_polytonic"].fillna(False) == False]  # noqa: E712
    elif historical_mode == "only":
        frame = frame[frame["is_historical_or_polytonic"].fillna(False)]
    if math_mode == "exclude":
        frame = frame[frame["contains_math"].fillna(False) == False]  # noqa: E712
    elif math_mode == "only":
        frame = frame[frame["contains_math"].fillna(False)]
    if latex_mode == "exclude":
        frame = frame[frame["contains_latex"].fillna(False) == False]  # noqa: E712
    elif latex_mode == "only":
        frame = frame[frame["contains_latex"].fillna(False)]
    frame = frame.reset_index(drop=True)
    if dedup_metadata_root is not None and validate_dedup_action(dedup_action) != "ignore":
        frame, _ = apply_builder_dedup(
            frame,
            dedup_metadata_root=dedup_metadata_root,
            dedup_action=dedup_action,
            dedup_exact_stage=dedup_exact_stage,
            dedup_similarity_threshold=dedup_similarity_threshold,
            dedup_inter_dataset_policy=dedup_inter_dataset_policy,
            dedup_source_weights_path=dedup_source_weights_path,
        )
    return frame


def build_mix_export(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    mix_output_path: Path | None = None,
    *,
    include_sources: list[str] | None = None,
    exclude_sources: list[str] | None = None,
    exclude_needs_ocr_sources: list[str] | None = None,
    quality_preset: str = "none",
    historical_mode: str = "include",
    math_mode: str = "include",
    latex_mode: str = "include",
    dedup_metadata_root: Path | None = None,
    dedup_action: str = "ignore",
    dedup_exact_stage: str = "strict_and_relaxed",
    dedup_similarity_threshold: float | None = None,
    dedup_inter_dataset_policy: str = "share_aware",
    dedup_source_weights_path: Path | None = None,
    source_mix_config_path: Path | None = None,
    exclude_doc_keys_path: Path | None = None,
) -> dict[str, Any]:
    if build_mix_should_stream(
        mix_output_path=mix_output_path,
        dedup_metadata_root=dedup_metadata_root,
        dedup_action=dedup_action,
        source_mix_config_path=source_mix_config_path,
        exclude_doc_keys_path=exclude_doc_keys_path,
    ):
        return build_mix_export_streaming(
            output_root=output_root,
            mix_output_path=mix_output_path if mix_output_path is not None else Path(tempfile.mkstemp(suffix=".parquet")[1]),
            include_sources=include_sources,
            exclude_sources=exclude_sources,
            exclude_needs_ocr_sources=exclude_needs_ocr_sources,
            quality_preset=quality_preset,
            historical_mode=historical_mode,
            math_mode=math_mode,
            latex_mode=latex_mode,
            dedup_metadata_root=dedup_metadata_root,
            dedup_action=dedup_action,
            dedup_exact_stage=dedup_exact_stage,
            dedup_similarity_threshold=dedup_similarity_threshold,
            dedup_inter_dataset_policy=dedup_inter_dataset_policy,
            dedup_source_weights_path=dedup_source_weights_path,
            source_mix_config_path=source_mix_config_path,
            exclude_doc_keys_path=exclude_doc_keys_path,
        )
    frame = load_filtered_mix(
        output_root,
        include_sources=include_sources,
        exclude_sources=exclude_sources,
        exclude_needs_ocr_sources=exclude_needs_ocr_sources,
        quality_preset=quality_preset,
        historical_mode=historical_mode,
        math_mode=math_mode,
        latex_mode=latex_mode,
        dedup_metadata_root=dedup_metadata_root,
        dedup_action=dedup_action,
        dedup_exact_stage=dedup_exact_stage,
        dedup_similarity_threshold=dedup_similarity_threshold,
        dedup_inter_dataset_policy=dedup_inter_dataset_policy,
        dedup_source_weights_path=dedup_source_weights_path,
    )
    source_mix_summary: dict[str, Any] | None = None
    if source_mix_config_path is not None:
        frame, source_mix_summary = apply_source_mix_config(
            frame,
            source_mix_config_path=source_mix_config_path,
            annotate_component=True,
        )
    frame["estimated_tokens"] = frame["text"].map(estimate_text_tokens)
    summary = (
        frame.groupby("source_dataset", as_index=False)
        .agg(rows=("source_doc_id", "count"), estimated_tokens=("estimated_tokens", "sum"))
        .sort_values(["estimated_tokens", "rows"], ascending=False)
    )
    per_pool_records: list[dict[str, Any]] = []
    if "dedup_pool_key" in frame.columns:
        per_pool = (
            frame.groupby(
                ["dedup_pool_key", "dedup_is_shared_pool", "dedup_pool_source_count"],
                dropna=False,
                as_index=False,
            )
            .agg(rows=("source_doc_id", "count"), estimated_tokens=("estimated_tokens", "sum"))
            .sort_values(["estimated_tokens", "rows", "dedup_pool_key"], ascending=[False, False, True])
        )
        per_pool_records = per_pool.to_dict("records")
    if mix_output_path is not None:
        ensure_dir(mix_output_path.parent)
        frame.drop(columns=["estimated_tokens"]).to_parquet(mix_output_path, index=False)
        summary.to_csv(mix_output_path.with_suffix(".summary.csv"), index=False)
        if per_pool_records:
            pd.DataFrame(per_pool_records).to_csv(mix_output_path.with_suffix(".pool_summary.csv"), index=False)
        if source_mix_summary is not None:
            mix_output_path.with_suffix(".source_mix_summary.json").write_text(
                json.dumps(source_mix_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    return {
        "rows_kept": int(len(frame)),
        "estimated_tokens": int(frame["estimated_tokens"].sum()),
        "per_source": summary.to_dict("records"),
        "per_pool": per_pool_records,
        "dedup_action": dedup_action,
        "source_mix": source_mix_summary,
    }


def nanochat_model_dim(depth: int, aspect_ratio: int = 64, head_dim: int = 128) -> int:
    base_dim = depth * aspect_ratio
    return ((base_dim + head_dim - 1) // head_dim) * head_dim


def estimate_nanochat_scaling_params(depth: int, aspect_ratio: int = 64, head_dim: int = 128) -> int:
    model_dim = nanochat_model_dim(depth, aspect_ratio=aspect_ratio, head_dim=head_dim)
    return 15 * depth * model_dim * model_dim


def target_nanochat_tokens(depth: int, *, target_param_data_ratio: float = 10.5, target_tokens: int | None = None) -> int:
    if target_tokens is not None:
        return int(target_tokens)
    return int(target_param_data_ratio * estimate_nanochat_scaling_params(depth))


def select_rows_for_token_budget(frame: pd.DataFrame, *, target_tokens: int) -> pd.DataFrame:
    if frame.empty or target_tokens <= 0:
        return frame.iloc[0:0].copy()
    ordered = frame.sort_values(
        ["estimated_tokens", "source_dataset", "source_doc_id"],
        ascending=[False, True, True],
    )
    selected_indices: list[int] = []
    running_tokens = 0
    for idx, row in ordered.iterrows():
        if running_tokens >= target_tokens:
            break
        selected_indices.append(idx)
        running_tokens += int(row["estimated_tokens"])
    return frame.loc[selected_indices].copy()


def thresholded_proportional_select_rows(
    frame: pd.DataFrame,
    *,
    target_tokens: int,
    group_column: str,
    full_group_threshold: float,
) -> pd.DataFrame:
    if frame.empty or target_tokens <= 0:
        return frame.iloc[0:0].copy()
    total_tokens = int(frame["estimated_tokens"].sum())
    if total_tokens <= target_tokens:
        return frame.copy()
    grouped_tokens = (
        frame.groupby(group_column, as_index=False)["estimated_tokens"]
        .sum()
        .rename(columns={"estimated_tokens": "group_tokens"})
    )
    grouped_tokens["share"] = grouped_tokens["group_tokens"] / float(total_tokens)
    small_groups = grouped_tokens[grouped_tokens["share"] <= float(full_group_threshold)].copy()
    selected_parts: list[pd.DataFrame] = []
    selected_indices: set[int] = set()
    remaining_tokens = int(target_tokens)
    if not small_groups.empty:
        for row in small_groups.sort_values(["group_tokens", group_column], ascending=[True, True]).to_dict("records"):
            group_frame = frame[frame[group_column] == row[group_column]]
            group_tokens = int(row["group_tokens"])
            if remaining_tokens <= 0:
                break
            if group_tokens <= remaining_tokens:
                selected_parts.append(group_frame)
                selected_indices.update(group_frame.index.tolist())
                remaining_tokens -= group_tokens
            else:
                partial = select_rows_for_token_budget(group_frame, target_tokens=remaining_tokens)
                selected_parts.append(partial)
                selected_indices.update(partial.index.tolist())
                remaining_tokens = 0
                break
    large_groups = grouped_tokens[~grouped_tokens[group_column].isin(set(small_groups[group_column]))].copy()
    if remaining_tokens > 0 and not large_groups.empty:
        large_total = float(large_groups["group_tokens"].sum())
        if large_total > 0:
            for row in large_groups.to_dict("records"):
                quota = int(math.floor(remaining_tokens * (float(row["group_tokens"]) / large_total)))
                if quota <= 0:
                    continue
                group_frame = frame[(frame[group_column] == row[group_column]) & (~frame.index.isin(selected_indices))]
                partial = select_rows_for_token_budget(group_frame, target_tokens=quota)
                if not partial.empty:
                    selected_parts.append(partial)
                    selected_indices.update(partial.index.tolist())
    selected = pd.concat(selected_parts, ignore_index=False) if selected_parts else frame.iloc[0:0].copy()
    selected_tokens = int(selected["estimated_tokens"].sum()) if not selected.empty else 0
    if selected_tokens < target_tokens:
        remainder = frame.loc[~frame.index.isin(selected_indices)]
        top_up = select_rows_for_token_budget(remainder, target_tokens=target_tokens - selected_tokens)
        if not top_up.empty:
            selected = pd.concat([selected, top_up], ignore_index=False)
    return selected.reset_index(drop=True)


def export_nanochat_shards(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    export_root: Path | None = None,
    *,
    nanochat_depth: int,
    target_param_data_ratio: float = 10.5,
    target_tokens: int | None = None,
    include_sources: list[str] | None = None,
    exclude_sources: list[str] | None = None,
    exclude_needs_ocr_sources: list[str] | None = None,
    quality_preset: str = "none",
    historical_mode: str = "include",
    math_mode: str = "include",
    latex_mode: str = "include",
    dedup_metadata_root: Path | None = None,
    dedup_action: str = "ignore",
    dedup_exact_stage: str = "strict_and_relaxed",
    dedup_similarity_threshold: float | None = None,
    dedup_inter_dataset_policy: str = "share_aware",
    dedup_source_weights_path: Path | None = None,
    dedup_pool_full_include_threshold: float = DEFAULT_DEDUP_POOL_FULL_INCLUDE_THRESHOLD,
    source_mix_config_path: Path | None = None,
    shard_target_tokens: int = 2_000_000,
    row_group_rows: int = NANOCHAT_ROW_GROUP_ROWS,
) -> dict[str, Any]:
    frame = load_filtered_mix(
        output_root,
        include_sources=include_sources,
        exclude_sources=exclude_sources,
        exclude_needs_ocr_sources=exclude_needs_ocr_sources,
        quality_preset=quality_preset,
        historical_mode=historical_mode,
        math_mode=math_mode,
        latex_mode=latex_mode,
        dedup_metadata_root=dedup_metadata_root,
        dedup_action=dedup_action,
        dedup_exact_stage=dedup_exact_stage,
        dedup_similarity_threshold=dedup_similarity_threshold,
        dedup_inter_dataset_policy=dedup_inter_dataset_policy,
        dedup_source_weights_path=dedup_source_weights_path,
    )
    source_mix_summary: dict[str, Any] | None = None
    if source_mix_config_path is not None:
        frame, source_mix_summary = apply_source_mix_config(
            frame,
            source_mix_config_path=source_mix_config_path,
            annotate_component=False,
        )
    frame["estimated_tokens"] = frame["text"].map(estimate_text_tokens)
    frame = frame[frame["estimated_tokens"] > 0].copy()
    wanted_tokens = target_nanochat_tokens(
        nanochat_depth,
        target_param_data_ratio=target_param_data_ratio,
        target_tokens=target_tokens,
    )
    allocation_group = "dedup_pool_key" if "dedup_pool_key" in frame.columns else "source_dataset"
    selected_columns = ["text", "source_dataset", "estimated_tokens"]
    if allocation_group not in {"text", "source_dataset", "estimated_tokens"}:
        selected_columns.append(allocation_group)
    selected = thresholded_proportional_select_rows(
        frame,
        target_tokens=int(wanted_tokens),
        group_column=allocation_group,
        full_group_threshold=float(dedup_pool_full_include_threshold),
    )[selected_columns].copy()
    if export_root is None:
        export_root = output_root / "nanochat_shards"
    ensure_dir(export_root)
    for existing in export_root.glob("*.parquet"):
        existing.unlink()
    shard_rows: list[dict[str, Any]] = []
    shard_tokens = 0
    shard_idx = 0
    for row in selected.to_dict("records"):
        shard_rows.append({"text": row["text"]})
        shard_tokens += int(row["estimated_tokens"])
        if shard_tokens >= shard_target_tokens:
            write_nanochat_text_shard(
                export_root / f"shard_{shard_idx:05d}.parquet",
                shard_rows,
                row_group_rows=row_group_rows,
            )
            shard_idx += 1
            shard_rows = []
            shard_tokens = 0
    if shard_rows:
        write_nanochat_text_shard(
            export_root / f"shard_{shard_idx:05d}.parquet",
            shard_rows,
            row_group_rows=row_group_rows,
        )
    per_source = (
        selected.groupby("source_dataset", as_index=False)
        .agg(rows=("text", "count"), estimated_tokens=("estimated_tokens", "sum"))
        .sort_values("estimated_tokens", ascending=False)
    )
    per_source.to_csv(export_root / "summary.csv", index=False)
    per_pool_records: list[dict[str, Any]] = []
    if allocation_group in selected.columns:
        per_pool = (
            selected.groupby(allocation_group, as_index=False)
            .agg(rows=("text", "count"), estimated_tokens=("estimated_tokens", "sum"))
            .sort_values(["estimated_tokens", "rows", allocation_group], ascending=[False, False, True])
        )
        per_pool.to_csv(export_root / "pool_summary.csv", index=False)
        per_pool_records = per_pool.to_dict("records")
    return {
        "target_tokens": int(wanted_tokens),
        "selected_tokens": int(selected["estimated_tokens"].sum()),
        "selected_rows": int(len(selected)),
        "num_shards": len(list(export_root.glob("shard_*.parquet"))),
        "estimated_scaling_params": estimate_nanochat_scaling_params(nanochat_depth),
        "per_source": per_source.to_dict("records"),
        "per_pool": per_pool_records,
        "allocation_group": allocation_group,
        "dedup_action": dedup_action,
        "source_mix": source_mix_summary,
    }
