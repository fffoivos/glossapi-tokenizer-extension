#!/usr/bin/env python3
"""Audit local candidate corpora for a polytonic/ancient Greek tokenizer arm.

The script intentionally measures cheap, tokenizer-relevant signals:

- document and character counts
- Greek-script share
- distinctive polytonic signal share, excluding plain tonos/oxia
- Greek words whose spelling contains distinctive polytonic evidence
- combining polytonic mark counts for breathings, varia, perispomeni, and
  ypogegrammeni
- obvious contamination signals: Latin, digits, replacement chars, controls

It does not decide licensing or final inclusion by itself. The companion
decision document records those judgments.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import unicodedata as ud
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


ROOT = Path("/home/foivos")


DEFAULT_SOURCES = [
    {
        "key": "first1k_open_greek_latin",
        "kind": "parquet",
        "path": ROOT
        / "data/glossapi_raw/hf/1000_prwta_xronia_ellhnikhs/1k_texts.parquet",
        "text_column": "text",
    },
    {
        "key": "perseus_canonical_greek_lit",
        "kind": "parquet",
        "path": ROOT
        / "data/glossapi_raw/hf/klasikh_arx_ell_grammateia/Classic_AG_texts_v2.parquet",
        "text_column": "text",
    },
    {
        "key": "wikisource_greek",
        "kind": "parquet",
        "path": ROOT
        / "data/glossapi_raw/hf/Wikisource_Greek_texts/wikisource_greek_deduped.parquet",
        "text_column": "text",
    },
    {
        "key": "goarch_liturgical",
        "kind": "parquet",
        "path": ROOT
        / "data/glossapi_raw/hf/Ekklisiastika_Keimena/litourgical_texts.parquet",
        "text_column": "texts",
    },
    {
        "key": "scholarios_graeca_patristic_http_text_greek",
        "kind": "txt_dir",
        "path": ROOT
        / "data/glossapi_raw/web/scholarios_graeca_patristic/data/http_text_greek",
        "text_column": None,
    },
]


DISTINCTIVE_POLYTONIC_COMBINING_MARKS = {
    0x0300,  # grave/varia
    0x0313,  # smooth breathing/psili
    0x0314,  # rough breathing/dasia
    0x0342,  # perispomeni
    0x0345,  # ypogegrammeni/iota subscript
}


@dataclass
class SourceStats:
    key: str
    kind: str
    path: str
    docs: int = 0
    nonempty_docs: int = 0
    lines: int = 0
    chars: int = 0
    greek_chars: int = 0
    greek_modern_block_chars: int = 0
    greek_extended_chars: int = 0
    distinctive_polytonic_chars: int = 0
    combining_polytonic_marks: int = 0
    greek_words: int = 0
    distinctive_polytonic_words: int = 0
    latin_chars: int = 0
    cyrillic_chars: int = 0
    arabic_chars: int = 0
    digit_chars: int = 0
    replacement_chars: int = 0
    non_whitespace_control_chars: int = 0
    docs_poly_ge_0_5pct_greek: int = 0
    docs_poly_ge_2pct_greek: int = 0
    docs_greek_chars_ge_50pct: int = 0
    docs_latin_ge_10pct: int = 0
    docs_with_replacement: int = 0
    doc_chars: list[int] = field(default_factory=list)

    def derived(self) -> dict[str, float | int]:
        median_doc_chars = int(statistics.median(self.doc_chars)) if self.doc_chars else 0
        p95_doc_chars = 0
        if self.doc_chars:
            ordered = sorted(self.doc_chars)
            p95_doc_chars = ordered[int(0.95 * (len(ordered) - 1))]

        return {
            "greek_pct_chars": pct(self.greek_chars, self.chars),
            "polytonic_pct_greek": pct(
                self.distinctive_polytonic_chars + self.combining_polytonic_marks,
                self.greek_chars,
            ),
            "distinctive_polytonic_word_pct": pct(
                self.distinctive_polytonic_words,
                self.greek_words,
            ),
            "latin_pct_chars": pct(self.latin_chars, self.chars),
            "digit_pct_chars": pct(self.digit_chars, self.chars),
            "median_doc_chars": median_doc_chars,
            "p95_doc_chars": p95_doc_chars,
            "nonempty_doc_pct": pct(self.nonempty_docs, self.docs),
            "docs_poly_ge_0_5pct_greek_pct": pct(
                self.docs_poly_ge_0_5pct_greek, self.docs
            ),
            "docs_poly_ge_2pct_greek_pct": pct(self.docs_poly_ge_2pct_greek, self.docs),
            "docs_greek_chars_ge_50pct_pct": pct(
                self.docs_greek_chars_ge_50pct, self.docs
            ),
        }

    def public_dict(self) -> dict:
        out = asdict(self)
        out.pop("doc_chars", None)
        out["derived"] = self.derived()
        return out


def pct(n: int | float, d: int | float) -> float:
    return (100.0 * n / d) if d else 0.0


def is_greek(ch: str) -> bool:
    cp = ord(ch)
    return 0x0370 <= cp <= 0x03FF or 0x1F00 <= cp <= 0x1FFF


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


def scan_text(stats: SourceStats, text: str | None) -> None:
    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)

    chars = len(text)
    greek = 0
    greek_modern = 0
    greek_extended = 0
    distinctive_polytonic = 0
    combining_poly = 0
    greek_words = 0
    distinctive_polytonic_words = 0
    latin = 0
    cyrillic = 0
    arabic = 0
    digits = 0
    replacement = 0
    controls = 0

    for ch in text:
        cp = ord(ch)
        if 0x0370 <= cp <= 0x03FF:
            greek += 1
            greek_modern += 1
        elif 0x1F00 <= cp <= 0x1FFF:
            greek += 1
            greek_extended += 1
        if has_distinctive_polytonic_signal(ch):
            if cp in DISTINCTIVE_POLYTONIC_COMBINING_MARKS:
                combining_poly += 1
            else:
                distinctive_polytonic += 1
        elif (
            0x0041 <= cp <= 0x005A
            or 0x0061 <= cp <= 0x007A
            or 0x00C0 <= cp <= 0x024F
        ):
            latin += 1
        elif 0x0400 <= cp <= 0x052F:
            cyrillic += 1
        elif 0x0600 <= cp <= 0x06FF:
            arabic += 1

        if ch.isdigit():
            digits += 1
        if ch == "\ufffd":
            replacement += 1
        if ud.category(ch).startswith("C") and ch not in "\n\r\t":
            controls += 1

    poly_signal = distinctive_polytonic + combining_poly
    for token in text.split():
        has_greek = False
        has_poly = False
        for ch in token:
            if is_greek(ch):
                has_greek = True
            if has_distinctive_polytonic_signal(ch):
                has_poly = True
        if has_greek:
            greek_words += 1
            if has_poly:
                distinctive_polytonic_words += 1

    stats.docs += 1
    stats.lines += text.count("\n") + (1 if text else 0)
    stats.doc_chars.append(chars)
    stats.chars += chars
    stats.greek_chars += greek
    stats.greek_modern_block_chars += greek_modern
    stats.greek_extended_chars += greek_extended
    stats.distinctive_polytonic_chars += distinctive_polytonic
    stats.combining_polytonic_marks += combining_poly
    stats.greek_words += greek_words
    stats.distinctive_polytonic_words += distinctive_polytonic_words
    stats.latin_chars += latin
    stats.cyrillic_chars += cyrillic
    stats.arabic_chars += arabic
    stats.digit_chars += digits
    stats.replacement_chars += replacement
    stats.non_whitespace_control_chars += controls

    if text.strip():
        stats.nonempty_docs += 1
    if greek and poly_signal / greek >= 0.005:
        stats.docs_poly_ge_0_5pct_greek += 1
    if greek and poly_signal / greek >= 0.02:
        stats.docs_poly_ge_2pct_greek += 1
    if chars and greek / chars >= 0.5:
        stats.docs_greek_chars_ge_50pct += 1
    if chars and latin / chars >= 0.1:
        stats.docs_latin_ge_10pct += 1
    if replacement:
        stats.docs_with_replacement += 1


def iter_texts(source: dict) -> Iterable[str]:
    path = Path(source["path"])
    if not path.exists():
        raise FileNotFoundError(path)

    if source["kind"] == "txt_dir":
        for txt_path in sorted(path.glob("*.txt")):
            yield txt_path.read_text(encoding="utf-8", errors="replace")
        return

    if source["kind"] == "parquet":
        try:
            import polars as pl
        except ImportError as exc:
            raise SystemExit(
                "Reading parquet sources requires polars. Install polars or audit "
                "only text-directory sources."
            ) from exc

        column = source["text_column"]
        df = pl.read_parquet(path, columns=[column])
        for (text,) in df.iter_rows():
            yield text
        return

    raise ValueError(f"Unknown source kind: {source['kind']}")


def audit_source(source: dict, limit: int | None = None) -> SourceStats:
    stats = SourceStats(
        key=source["key"],
        kind=source["kind"],
        path=str(source["path"]),
    )
    for idx, text in enumerate(iter_texts(source), start=1):
        if limit is not None and idx > limit:
            break
        scan_text(stats, text)
    return stats


def render_markdown(rows: list[SourceStats]) -> str:
    lines = [
        "| source | docs | chars | Greek chars | poly % Greek | Latin % chars | notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        d = row.derived()
        notes = []
        if row.replacement_chars:
            notes.append(f"{row.replacement_chars} replacement chars")
        if row.docs_latin_ge_10pct:
            notes.append(f"{row.docs_latin_ge_10pct} docs >=10% Latin")
        if row.nonempty_docs < row.docs:
            notes.append(f"{row.docs - row.nonempty_docs} empty docs")
        note = "; ".join(notes) if notes else "clean by coarse counters"
        lines.append(
            "| `{key}` | {docs:,} | {chars:,} | {greek:,} | {poly:.2f} | "
            "{latin:.2f} | {note} |".format(
                key=row.key,
                docs=row.docs,
                chars=row.chars,
                greek=row.greek_chars,
                poly=d["polytonic_pct_greek"],
                latin=d["latin_pct_chars"],
                note=note,
            )
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=[s["key"] for s in DEFAULT_SOURCES],
        action="append",
        help="Audit only this source key; repeatable. Defaults to all sources.",
    )
    parser.add_argument(
        "--limit-docs",
        type=int,
        default=0,
        help="Optional per-source document limit for quick smoke runs.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format.",
    )
    args = parser.parse_args()

    selected = DEFAULT_SOURCES
    if args.source:
        wanted = set(args.source)
        selected = [s for s in DEFAULT_SOURCES if s["key"] in wanted]

    rows = [audit_source(s, limit=args.limit_docs or None) for s in selected]

    if args.format == "json":
        print(json.dumps([r.public_dict() for r in rows], ensure_ascii=False, indent=2))
    else:
        print(render_markdown(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
