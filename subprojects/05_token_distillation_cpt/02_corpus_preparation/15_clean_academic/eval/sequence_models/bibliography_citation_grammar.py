#!/usr/bin/env python3
"""Intra-line citation-grammar features for bibliography line classification.

A bibliography entry has a field grammar — authors, then year, then title, then
journal/publisher, then pages, then DOI. Citation-heavy prose reuses the same
vocabulary in a different arrangement. These features encode that arrangement as
shallow independent binaries (never one composed schema), plus the two citation
ANTI-patterns that appear in prose and essentially never in an entry.

Measured on the 29,147 consensus-labelled Kallipos lines of
`bibliography_150_20260723_v2` (base rate 0.1105):

  author token present         P(bib)=0.988   rescues every confused cue
  (year) in first 40% of line  P(bib)=0.917  vs 0.335 when later
  author before year           P(bib)=0.992  vs 0.839 out of order
  Authors (year)   narrative   P(bib)=0.881
  (Authors, year)  in-text     P(bib)=0.016   ANTI-PATTERN
  "quote" (Authors, year)      P(bib)=0.000   ANTI-PATTERN (122 firings, zero bib)

Deliberately excluded: author-before-pages and author-before-DOI, which hold
~0.99 in BOTH directions and therefore add no separation over the elements
themselves.

This module is the single definition used by both the development feature table
and the label-blind unseen path, so the two cannot drift apart.
"""

from __future__ import annotations

import re

# --- element vocabulary -------------------------------------------------------
_NAME = r"[Α-ΩA-ZΆΈΉΊΌΎΏ][\wά-ώΆ-Ώ’'.-]+"
_YEAR = r"(?:1[5-9]\d{2}|20[0-4]\d)[a-zαβγ]?"

# "Surname, I." / "Surname, I. J." -- the canonical author token
_AUTHOR = re.compile(rf"{_NAME},\s*[Α-ΩA-Z]\.(?:\s*[Α-ΩA-Z]\.)?")
# two author tokens joined -> an author block
_AUTHOR_BLOCK = re.compile(
    rf"{_NAME},\s*[Α-ΩA-Z]\.(?:\s*[Α-ΩA-Z]\.)?(?:\s*[,&·]|\s+(?:και|and)\s+)\s*{_NAME},\s*[Α-ΩA-Z]\."
)
_YEAR_PAREN = re.compile(rf"\(\s*{_YEAR}\s*\)")
_QUOTED = re.compile(r"[«“\"][^«»“”\"]{5,300}[»”\"]")
_PAGES = re.compile(r"(?:σσ?\.|σελ\.?|pp?\.)\s*\d{1,4}\s*[-–—]\s*\d{1,4}")
_PUB_PLACE = re.compile(rf"{_NAME}\s*:\s*{_NAME}")

# --- the two citation forms, which are opposites ------------------------------
# in-text: author tokens INSIDE the parentheses -> prose
_IN_TEXT = re.compile(rf"\(\s*{_NAME}[^()]{{0,60}}?{_YEAR}[^()]{{0,25}}\)")
# narrative: author OUTSIDE, parentheses hold essentially only the year
_NARRATIVE = re.compile(rf"{_NAME}\s*\(\s*{_YEAR}\s*\)")
# quoted material immediately followed by an in-text citation -> prose, always
_QUOTE_THEN_CITE = re.compile(
    rf"[«“\"][^«»“”\"]{{10,400}}[»”\"]\s*[.,;:]?\s*\(\s*{_NAME}[^()]{{0,60}}?{_YEAR}[^()]{{0,25}}\)"
)
# running-prose connectives/verbs: absent from entries, present in 76% of confusers
_PROSE = re.compile(
    r"\b(είναι|όπως|καθώς|σύμφωνα|ωστόσο|επομένως|δηλαδή|αναφέρ|μελέτ|θεωρ|"
    r"μπορεί|έχει|έχουν|αυτό|αυτή|οποίο|οποία|επίσης|ενώ|στο|στη)\b",
    re.IGNORECASE,
)

CITATION_GRAMMAR_NAMES: tuple[str, ...] = (
    "grammar:author_token",
    "grammar:author_block",
    "grammar:narrative_citation",
    "grammar:in_text_citation",
    "grammar:quote_then_citation",
    "grammar:prose_marker",
    "grammar:year_paren_early",
    "grammar:author_first_fifth",
    "grammar:author_before_year",
    "grammar:author_before_quote",
    "grammar:author_before_publisher",
    "grammar:quoted_before_pages",
)


def citation_grammar(text: str) -> tuple[float, ...]:
    """Return the citation-grammar feature row for one line, in fixed order."""

    length = max(1, len(text))
    author = _AUTHOR.search(text)
    year_paren = _YEAR_PAREN.search(text)
    quoted = _QUOTED.search(text)
    pages = _PAGES.search(text)
    publisher = _PUB_PLACE.search(text)

    author_at = author.start() if author else None
    year_at = year_paren.start() if year_paren else None
    quote_at = quoted.start() if quoted else None
    pages_at = pages.start() if pages else None
    pub_at = publisher.start() if publisher else None

    def before(a: int | None, b: int | None) -> float:
        # only asserted when both elements are present; absent -> 0.0
        return 1.0 if a is not None and b is not None and a < b else 0.0

    return (
        1.0 if author else 0.0,
        1.0 if _AUTHOR_BLOCK.search(text) else 0.0,
        1.0 if _NARRATIVE.search(text) else 0.0,
        1.0 if _IN_TEXT.search(text) else 0.0,
        1.0 if _QUOTE_THEN_CITE.search(text) else 0.0,
        1.0 if _PROSE.search(text) else 0.0,
        1.0 if year_at is not None and year_at / length < 0.40 else 0.0,
        1.0 if author_at is not None and author_at / length < 0.20 else 0.0,
        before(author_at, year_at),
        before(author_at, quote_at),
        before(author_at, pub_at),
        before(quote_at, pages_at),
    )
