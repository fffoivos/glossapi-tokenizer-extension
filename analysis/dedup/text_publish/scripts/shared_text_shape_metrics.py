from __future__ import annotations

import re
import unicodedata
from typing import Any


WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)
HEADER_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s+")
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", flags=re.DOTALL)
BLOCK_MATH_RE = re.compile(r"\$\$.*?\$\$", flags=re.DOTALL)
BEGIN_END_MATH_RE = re.compile(r"\\begin\{[^{}]+\}.*?\\end\{[^{}]+\}", flags=re.DOTALL)
INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", flags=re.DOTALL)
LATEX_MARKER_RE = re.compile(r"\$\$|\\begin\{|\\frac|\\sum|\\int|\\alpha|\\beta")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value))
    return text.replace("\r\n", "\n").replace("\r", "\n")


def collapse_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def strip_html_comments(text: str) -> str:
    return HTML_COMMENT_RE.sub(" ", text)


def strip_markdown_table_lines(text: str) -> str:
    kept_lines: list[str] = []
    for line in text.splitlines():
        if "|" in line:
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


def strip_latex_math(text: str) -> str:
    text = BLOCK_MATH_RE.sub(" ", text)
    text = BEGIN_END_MATH_RE.sub(" ", text)
    text = INLINE_MATH_RE.sub(" ", text)
    return text


def strip_header_lines(text: str) -> str:
    kept_lines: list[str] = []
    for line in text.splitlines():
        if HEADER_RE.match(line):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


def text_word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def table_line_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if "|" in line)


def image_markup_count(text: str) -> int:
    return text.count("<!-- image -->") + text.lower().count("<img")


def html_comment_count(text: str) -> int:
    return len(HTML_COMMENT_RE.findall(text))


def abstract_marker_count(text: str) -> int:
    lowered = text.casefold()
    markers = (
        "## abstract",
        "περίληψη",
        "abstract",
        "λέξεις κλειδιά",
        "keywords",
    )
    return sum(lowered.count(marker) for marker in markers)


def bibliography_marker_count(text: str) -> int:
    lowered = text.casefold()
    markers = (
        "βιβλιογραφ",
        "references",
        "bibliography",
        "βιβλιογρ",
    )
    return sum(lowered.count(marker) for marker in markers)


def alpha_char_count(text: str) -> int:
    return sum(1 for char in text if char.isalpha())


def build_shared_text_shape_metrics(text: str | None) -> dict[str, Any]:
    raw_text = normalize_text(text)
    no_comments = strip_html_comments(raw_text)
    no_comments_tables = strip_markdown_table_lines(no_comments)
    no_comments_tables_math = strip_latex_math(no_comments_tables)
    plain_body = strip_header_lines(no_comments_tables_math)

    raw_collapsed = collapse_whitespace(raw_text)
    no_comments_collapsed = collapse_whitespace(no_comments)
    no_comments_tables_collapsed = collapse_whitespace(no_comments_tables)
    no_comments_tables_math_collapsed = collapse_whitespace(no_comments_tables_math)
    plain_body_collapsed = collapse_whitespace(plain_body)

    raw_chars = len(raw_text)
    alpha_chars = alpha_char_count(plain_body_collapsed)
    return {
        "raw_chars": raw_chars,
        "raw_collapsed_chars": len(raw_collapsed),
        "word_count": text_word_count(raw_text),
        "header_count": len(HEADER_RE.findall(raw_text)),
        "table_line_count": table_line_count(raw_text),
        "latex_marker_count": len(LATEX_MARKER_RE.findall(raw_text)),
        "image_markup_count": image_markup_count(raw_text),
        "html_comment_count": html_comment_count(raw_text),
        "abstract_marker_count": abstract_marker_count(raw_text),
        "bibliography_marker_count": bibliography_marker_count(raw_text),
        "chars_no_comments": len(no_comments_collapsed),
        "chars_no_comments_tables": len(no_comments_tables_collapsed),
        "chars_no_comments_tables_math": len(no_comments_tables_math_collapsed),
        "chars_plain_body": len(plain_body_collapsed),
        "alpha_char_ratio_plain_body": round((alpha_chars / len(plain_body_collapsed)), 4) if plain_body_collapsed else 0.0,
        "plain_body_word_count": text_word_count(plain_body_collapsed),
        "removed_by_comments_chars": max(0, len(raw_collapsed) - len(no_comments_collapsed)),
        "removed_by_tables_chars": max(0, len(no_comments_collapsed) - len(no_comments_tables_collapsed)),
        "removed_by_math_chars": max(0, len(no_comments_tables_collapsed) - len(no_comments_tables_math_collapsed)),
        "removed_by_headers_chars": max(0, len(no_comments_tables_math_collapsed) - len(plain_body_collapsed)),
    }
