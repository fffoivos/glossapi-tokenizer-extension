#!/usr/bin/env python3
"""Prepare polytonic Greek source parquets for the existing dedup pipeline.

This script does not deduplicate by itself. It creates canonical input
parquets that can be fed to:

    python -m glossapi_corpus_cli.cli dedup-text run --input-root <out>/data

The important policy choice here is that Wikisource is filtered to
polytonic rows only. Older/author-year heuristics are intentionally not
used as inclusion criteria for Wikisource in this polytonic lane.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata as ud
try:
    from datetime import UTC, datetime
except ImportError:  # Python 3.10 on existing GCP workers
    from datetime import datetime, timezone

    UTC = timezone.utc
from pathlib import Path
from typing import Any


ROOT = Path("/home/foivos")
DEFAULT_CANONICAL_DATA = ROOT / "data/glossapi_work/hf_release/data"
DEFAULT_SCHOLARIOS_TEXT = (
    ROOT / "data/glossapi_raw/web/scholarios_graeca_patristic/data/http_text_greek"
)

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

BOOLEAN_COLUMNS = {
    "is_historical_or_polytonic",
    "contains_math",
    "contains_latex",
    "needs_ocr",
    "is_empty",
    "ocr_success",
}

FLOAT_COLUMNS = {
    "greek_percentage",
    "latin_percentage",
    "polytonic_ratio",
    "table_ratio",
    "greek_badness_score",
    "mojibake_badness_score",
}

INTEGER_COLUMNS = {"len_greek"}

TEXT_SOURCES = {
    "first1k_open_greek_latin": "1000_prwta_xronia_ellhnikhs.parquet",
    "perseus_canonical_greek_lit": "klasikh_arx_ell_grammateia.parquet",
    "goarch_liturgical": "Ekklisiastika_Keimena.parquet",
}

DISTINCTIVE_POLYTONIC_COMBINING_MARKS = {
    0x0300,  # grave/varia
    0x0313,  # smooth breathing/psili
    0x0314,  # rough breathing/dasia
    0x0342,  # perispomeni
    0x0345,  # ypogegrammeni/iota subscript
}

LATEX_RE = re.compile(
    r"(\\begin\{(?:equation|align|gather|multline|matrix|bmatrix|pmatrix)\}|"
    r"\\[a-zA-Z]+(?:\[[^\]]+\])?(?:\{[^}]+\})?|\\\(|\\\)|\\\[|\\\]|\$\$)",
    re.MULTILINE,
)
MATH_RE = re.compile(r"[∑∫√∞≈≠≤≥±×÷∂∇∈∉∩∪⊂⊆⊕⊗≃≅∀∃]")


def load_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("prepare_polytonic_dedup_inputs.py requires pandas") from exc
    return pd


def pct(n: int | float, d: int | float) -> float | None:
    return (100.0 * n / d) if d else None


def ratio(n: int | float, d: int | float) -> float | None:
    return (float(n) / float(d)) if d else None


def metric_value(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except Exception:
        return default


def has_distinctive_polytonic_signal(ch: str) -> bool:
    """Return true for polytonic-only marks, not plain tonos/oxia."""
    cp = ord(ch)
    if cp in DISTINCTIVE_POLYTONIC_COMBINING_MARKS:
        return True
    if 0x1F00 <= cp <= 0x1FFF:
        return any(
            ord(mark) in DISTINCTIVE_POLYTONIC_COMBINING_MARKS
            for mark in ud.normalize("NFD", ch)
        )
    return False


def is_greek_codepoint(cp: int) -> bool:
    return 0x0370 <= cp <= 0x03FF or 0x1F00 <= cp <= 0x1FFF


def text_stats(text: str) -> dict[str, Any]:
    greek = 0
    distinctive_poly = 0
    greek_words = 0
    distinctive_poly_words = 0
    latin = 0
    table_chars = 0
    mojibake = 0
    controls = 0

    for ch in text:
        cp = ord(ch)
        if is_greek_codepoint(cp):
            greek += 1
        if has_distinctive_polytonic_signal(ch):
            distinctive_poly += 1
        elif (
            0x0041 <= cp <= 0x005A
            or 0x0061 <= cp <= 0x007A
            or 0x00C0 <= cp <= 0x024F
        ):
            latin += 1
        if ch in "|┌┐└┘├┤┬┴┼─═":
            table_chars += 1
        if ch == "\ufffd":
            mojibake += 1
        if ud.category(ch).startswith("C") and ch not in "\n\r\t":
            controls += 1

    for token in text.split():
        has_greek = False
        has_poly = False
        for ch in token:
            cp = ord(ch)
            if is_greek_codepoint(cp):
                has_greek = True
            if has_distinctive_polytonic_signal(ch):
                has_poly = True
        if has_greek:
            greek_words += 1
            if has_poly:
                distinctive_poly_words += 1

    chars = len(text)
    return {
        "greek_percentage": pct(greek, chars),
        "latin_percentage": pct(latin, chars),
        "polytonic_ratio": ratio(distinctive_poly_words, greek_words),
        "distinctive_polytonic_char_ratio": ratio(distinctive_poly, greek),
        "table_ratio": ratio(table_chars, chars),
        "len_greek": greek,
        "mojibake_badness_score": float(mojibake + controls),
    }


def contains_latex(text: str) -> bool:
    return bool(LATEX_RE.search(text))


def contains_math(text: str) -> bool:
    return contains_latex(text) or bool(MATH_RE.search(text))


def first_title_line(text: str) -> str | None:
    for raw in text.splitlines()[:40]:
        line = raw.strip(" \t\r\n#*|-=·")
        if 8 <= len(line) <= 180 and any("\u0370" <= ch <= "\u03ff" or "\u1f00" <= ch <= "\u1fff" for ch in line):
            return re.sub(r"\s+", " ", line)
    return None


def ensure_canonical(df: Any) -> Any:
    pd = load_pandas()
    frame = df.copy()
    for col in CANONICAL_COLUMNS:
        if col not in frame.columns:
            frame[col] = None
    for col in CANONICAL_COLUMNS:
        if col in BOOLEAN_COLUMNS:
            frame[col] = frame[col].astype("boolean")
        elif col in FLOAT_COLUMNS:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
        elif col in INTEGER_COLUMNS:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("Int64")
        elif col == "reevaluated_at":
            frame[col] = pd.to_datetime(frame[col], errors="coerce", utc=True)
        else:
            frame[col] = frame[col].where(frame[col].notna(), None).astype("string")
    return frame[CANONICAL_COLUMNS]


def write_frame(df: Any, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = ensure_canonical(df)
    df.to_parquet(path, compression="zstd", engine="pyarrow", row_group_size=2048, index=False)
    return len(df)


def copy_existing_source(
    *,
    input_dir: Path,
    output_dir: Path,
    source_key: str,
    filename: str,
    limit_rows: int | None,
) -> dict[str, Any]:
    pd = load_pandas()
    src = input_dir / filename
    if not src.exists():
        raise FileNotFoundError(src)
    df = pd.read_parquet(src)
    if limit_rows is not None:
        df = df.head(limit_rows)
    out = output_dir / filename
    rows = write_frame(df, out)
    return {"source": source_key, "rows": rows, "path": str(out), "mode": "copied_existing_canonical"}


def filter_wikisource_polytonic(
    *,
    input_dir: Path,
    output_dir: Path,
    polytonic_min: float,
    polytonic_char_min: float,
    greek_pct_min: float,
    latin_pct_max: float,
    limit_rows: int | None,
) -> dict[str, Any]:
    pd = load_pandas()
    src = input_dir / "Wikisource_Greek_texts.parquet"
    if not src.exists():
        raise FileNotFoundError(src)
    original = pd.read_parquet(src)
    original_rows = len(original)
    if limit_rows is not None:
        original = original.head(limit_rows)
    def strict_ratios(value: Any) -> tuple[float, float]:
        if value is None:
            return 0.0, 0.0
        try:
            if pd.isna(value):
                return 0.0, 0.0
        except Exception:
            pass
        stats = text_stats(str(value))
        return (
            metric_value(stats["polytonic_ratio"], 0.0),
            metric_value(stats["distinctive_polytonic_char_ratio"], 0.0),
        )

    strict_pairs = original["text"].map(strict_ratios)
    strict_polytonic_ratio = strict_pairs.map(lambda pair: pair[0])
    strict_polytonic_char_ratio = strict_pairs.map(lambda pair: pair[1])
    is_empty = original["is_empty"].fillna(False).astype(bool)
    mask = (
        strict_polytonic_ratio.ge(polytonic_min)
        & strict_polytonic_char_ratio.ge(polytonic_char_min)
        & pd.to_numeric(original["greek_percentage"], errors="coerce").fillna(0.0).ge(greek_pct_min)
        & pd.to_numeric(original["latin_percentage"], errors="coerce").fillna(100.0).le(latin_pct_max)
        & is_empty.eq(False)
    )
    df = original.loc[mask].copy()
    df["polytonic_ratio"] = strict_polytonic_ratio.loc[mask].astype("float64")
    df["is_historical_or_polytonic"] = True
    df["needs_ocr"] = False
    df["ocr_success"] = True
    df["quality_method"] = "polytonic_wikisource_distinctive_signal_v2"
    out = output_dir / "Wikisource_Greek_texts.polytonic_only.parquet"
    rows = write_frame(df, out)
    return {
        "source": "wikisource_greek_polytonic_only",
        "rows": rows,
        "input_rows": original_rows,
        "path": str(out),
        "mode": "filtered_existing_canonical",
        "filter": {
            "distinctive_polytonic_word_ratio_gte": polytonic_min,
            "distinctive_polytonic_char_ratio_gte": polytonic_char_min,
            "plain_tonos_or_oxia_counted": False,
            "greek_percentage_gte": greek_pct_min,
            "latin_percentage_lte": latin_pct_max,
            "is_empty": False,
        },
    }


def iter_scholarios_rows(text_dir: Path, limit_rows: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    now = datetime.now(UTC)
    for path in sorted(text_dir.glob("*.txt")):
        if limit_rows is not None and len(rows) >= limit_rows:
            break
        text = path.read_text(encoding="utf-8", errors="replace")
        stats = text_stats(text)
        stem = path.stem
        metadata = {
            "source_collection": "scholarios_graeca_patristic",
            "source_site": "scholarios.graeca.org",
            "record_id": stem,
            "source_modality": "web_scraped_text",
        }
        rows.append(
            {
                "source_dataset": "scholarios_graeca_patristic",
                "source_doc_id": f"scholarios::{stem}",
                "text": text,
                "title": first_title_line(text) or stem,
                "author": None,
                "source_metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                "is_historical_or_polytonic": True,
                "contains_math": contains_math(text),
                "contains_latex": contains_latex(text),
                "greek_percentage": stats["greek_percentage"],
                "latin_percentage": stats["latin_percentage"],
                "polytonic_ratio": stats["polytonic_ratio"],
                "_distinctive_polytonic_char_ratio": stats["distinctive_polytonic_char_ratio"],
                "table_ratio": stats["table_ratio"],
                "greek_badness_score": None,
                "len_greek": stats["len_greek"],
                "mojibake_badness_score": stats["mojibake_badness_score"],
                "needs_ocr": False,
                "is_empty": not bool(text.strip()),
                "filter": "scholarios_web_text_polytonic_source_v1",
                "ocr_success": True,
                "quality_method": "polytonic_source_distinctive_signal_v2",
                "reevaluated_at": now,
            }
        )
    return rows


def build_scholarios(
    *,
    output_dir: Path,
    text_dir: Path,
    polytonic_min: float,
    polytonic_char_min: float,
    greek_pct_min: float,
    latin_pct_max: float,
    limit_rows: int | None,
) -> dict[str, Any]:
    rows = iter_scholarios_rows(text_dir, limit_rows)
    original_rows = len(rows)
    kept = []
    for row in rows:
        if row["is_empty"]:
            continue
        if metric_value(row["polytonic_ratio"], 0.0) < polytonic_min:
            continue
        if metric_value(row.get("_distinctive_polytonic_char_ratio"), 0.0) < polytonic_char_min:
            continue
        if metric_value(row["greek_percentage"], 0.0) < greek_pct_min:
            continue
        if metric_value(row["latin_percentage"], 100.0) > latin_pct_max:
            continue
        kept.append(row)
    pd = load_pandas()
    df = pd.DataFrame(kept, columns=CANONICAL_COLUMNS)
    out = output_dir / "scholarios_graeca_patristic.parquet"
    written = write_frame(df, out)
    return {
        "source": "scholarios_graeca_patristic",
        "rows": written,
        "input_rows": original_rows,
        "path": str(out),
        "mode": "converted_txt_dir",
        "filter": {
            "distinctive_polytonic_word_ratio_gte": polytonic_min,
            "distinctive_polytonic_char_ratio_gte": polytonic_char_min,
            "plain_tonos_or_oxia_counted": False,
            "greek_percentage_gte": greek_pct_min,
            "latin_percentage_lte": latin_pct_max,
            "is_empty": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-data-dir", type=Path, default=DEFAULT_CANONICAL_DATA)
    parser.add_argument("--scholarios-text-dir", type=Path, default=DEFAULT_SCHOLARIOS_TEXT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--polytonic-min",
        type=float,
        default=0.50,
        help=(
            "Minimum distinctive-polytonic word ratio over Greek words. "
            "Plain tonos/oxia is not counted."
        ),
    )
    parser.add_argument(
        "--polytonic-char-min",
        type=float,
        default=0.10,
        help=(
            "Minimum distinctive-polytonic char ratio over Greek chars. "
            "Plain tonos/oxia is not counted."
        ),
    )
    parser.add_argument("--greek-pct-min", type=float, default=50.0)
    parser.add_argument("--latin-pct-max", type=float, default=10.0)
    parser.add_argument("--limit-rows", type=int, default=0, help="Smoke-test cap per source")
    parser.add_argument(
        "--source",
        action="append",
        choices=[
            *TEXT_SOURCES.keys(),
            "wikisource_greek_polytonic_only",
            "scholarios_graeca_patristic",
        ],
        help="Prepare only selected sources; repeatable. Defaults to all.",
    )
    args = parser.parse_args()

    output_data = args.output_root / "data"
    output_data.mkdir(parents=True, exist_ok=True)
    limit_rows = args.limit_rows or None
    selected = set(args.source or [*TEXT_SOURCES.keys(), "wikisource_greek_polytonic_only", "scholarios_graeca_patristic"])

    outputs: list[dict[str, Any]] = []
    for source_key, filename in TEXT_SOURCES.items():
        if source_key in selected:
            outputs.append(
                copy_existing_source(
                    input_dir=args.canonical_data_dir,
                    output_dir=output_data,
                    source_key=source_key,
                    filename=filename,
                    limit_rows=limit_rows,
                )
            )
    if "wikisource_greek_polytonic_only" in selected:
        outputs.append(
            filter_wikisource_polytonic(
                input_dir=args.canonical_data_dir,
                output_dir=output_data,
                polytonic_min=args.polytonic_min,
                polytonic_char_min=args.polytonic_char_min,
                greek_pct_min=args.greek_pct_min,
                latin_pct_max=args.latin_pct_max,
                limit_rows=limit_rows,
            )
        )
    if "scholarios_graeca_patristic" in selected:
        outputs.append(
            build_scholarios(
                output_dir=output_data,
                text_dir=args.scholarios_text_dir,
                polytonic_min=args.polytonic_min,
                polytonic_char_min=args.polytonic_char_min,
                greek_pct_min=args.greek_pct_min,
                latin_pct_max=args.latin_pct_max,
                limit_rows=limit_rows,
            )
        )

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": "polytonic_greek_dedup_input",
        "dedup_contract": "glossapi_corpus_cli dedup-text run input_root",
        "canonical_data_dir": str(args.canonical_data_dir),
        "scholarios_text_dir": str(args.scholarios_text_dir),
        "outputs": outputs,
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
