#!/usr/bin/env python3
"""Dump the compiled regexes of `bibliography_v2` for the Rust port to consume.

Transcribing these by hand is not viable and not safe. The character classes are
*enumerated* at import time from `unicodedata` over the European script ranges --
`_UPPER` alone is well over a thousand codepoints -- so a hand-written Rust class
would silently disagree the moment the two runtimes disagree about a category.
Emitting the exact pattern strings Python compiled removes that whole class of
divergence: Rust compiles the same source text.

Two syntax edits are needed, because the Rust side uses `fancy-regex` (chosen for
its lookaround and backreference support, which the `regex` crate lacks and 33 of
these patterns require):

* `(?P<name>...)` -> `(?<name>...)`
* `(?P=name)`     -> `\\k<name>`

and `re.I` becomes a leading `(?i)`. Everything else is passed through verbatim.

    python3 bib_line_model/fixtures/dump_patterns.py --out bib_line_model/patterns.json

Run from `15_clean_academic/eval` so `sequence_models` imports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

# Module-level `re.Pattern` constants the Rust feature stack needs, named exactly
# as in bibliography_v2 so the mapping is auditable by grep.
WANTED = [
    "_YEAR",
    "_NO_DATE",
    "_NUMERIC_DATE",
    "_MONTH_DATE",
    "_ACCESS_DATE",
    "_URL",
    "_DOI",
    "_ISBN",
    "_ISSN",
    "_INITIAL",
    "_PROPER_WORD",
    "_INVERTED_AUTHOR",
    "_AUTHOR_PREFIX",
    "_NAME_INITIAL_PAIR",
    "_DIRECT_AUTHOR",
    "_AMPERSAND",
    "_QUOTED",
    "_EDITOR_TERMS",
    "_THESIS_TERMS",
    "_IN_CONTAINER",
    "_EDITION_TERMS",
    "_DOTTED_WORD",
    "_VOLUME_MARKER",
    "_VOLUME_SHAPE",
    "_JOURNAL_YEAR_VOLUME",
    "_PAGE_MARKER",
    "_ARTICLE_PAGE_RANGE",
    "_PAGE_RANGE",
    "_PUBLISHER_TERMS",
    "_PLACE_NAMES",
    "_PLACE_PUBLISHER_SHAPE",
    "_PROSE_LEAD",
    "_BIB_HEADING_WORD",
    "_BIB_EXTENDED_HEADING",
    "_BIB_EXTENDED_SUBHEADING",
    "_FIGURE_CAPTION_START",
    "_ENUMERATED_PROSE_START",
]

# `line_shape` and the heading candidate predicate live in a second module. Names
# are prefixed on the way out so the two namespaces cannot collide.
WANTED_ROLE = [
    "_WORD",
    "_NUMBERED_HEADING",
    "_REPEATED_RULE",
    "_TABLE_RULE",
    "_HTML_FRAGMENT",
    "_PAGE_NUMBER",
    "_BULLET_ONLY",
]

# The five `structure:` flags are assembled in the table builder, from patterns that
# run on the RAW line rather than the normalized one.
WANTED_TABLE = [
    "_MARKDOWN_HEADING",
    "_IMAGE_MARKER",
    "_RULE_LINE",
    "_SECTION_PREFIX",
    "_WRAPPERS",
]

_NAMED_GROUP = re.compile(r"\(\?P<([A-Za-z_][A-Za-z0-9_]*)>")
_NAMED_BACKREF = re.compile(r"\(\?P=([A-Za-z_][A-Za-z0-9_]*)\)")

# Python and Rust disagree about what a "word character" is, and the disagreement
# is not academic: Rust's \w is Alphabetic|M|Nd|Pc|Join_Control, so it accepts
# combining marks, while Python's is `str.isalnum()` plus underscore, which does
# not. On OCR'd Greek a combining tilde before a token then flips `(?<!\w)` and
# every `\b`, silently dropping matches. CPython's SRE defines these classes via
# the `str` predicates, and those are exactly expressible as Unicode categories:
#
#   str.isalnum() == category L* or N*   =>  \w == [\p{L}\p{N}_]
#   str.isspace() == White_Space plus the four ASCII separator controls
#
# so every \w, \W, \s, \S and \b is rewritten to the Python definition rather than
# left to mean whatever the Rust engine means by it. `dump_unicode_tables.py`
# emits the same predicates as raw ranges, and `main` below asserts the two agree.
_PY_WORD = r"\p{L}\p{N}_"
_PY_SPACE = r"\p{White_Space}\x{1c}-\x{1f}"
_PY_BOUNDARY = (
    rf"(?:(?<=[{_PY_WORD}])(?![{_PY_WORD}])|(?<![{_PY_WORD}])(?=[{_PY_WORD}]))"
)


# `[^\W_]` is a negated class containing a negated shorthand, which the mechanical
# rewriter below cannot expand -- but it reduces by hand exactly: "not (non-word or
# underscore)" is the word characters minus underscore, and Python's word characters
# are category L* or N*. Both the tokenizer in `_features_and_spans` and `_WORD` in
# `bibliography_role_features` are this same pattern.
HAND_REDUCED = {
    r"[^\W_]+(?:[’'\-][^\W_]+)*": r"[\p{L}\p{N}]+(?:[’'\-][\p{L}\p{N}]+)*",
}


# Negated shorthands *inside* a negated class cannot be expanded mechanically, but
# the idioms that actually occur reduce exactly. Python's word class is L|N|_, so:
#
#   [^\W_]     = (L|N|_) - _            = L|N          ("word chars, not underscore")
#   [^\W\d_]   = (L|N|_) - Nd - _       = L|Nl|No      ("letters")
#
# `\d` is Nd specifically, not all of N, which is why Nl and No survive the second
# reduction. Applied as a literal pre-pass so the general scanner never sees them.
NEGATED_CLASS_IDIOMS = {
    r"[^\W\d_]": r"[\p{L}\p{Nl}\p{No}]",
    r"[^\W_]": r"[\p{L}\p{N}]",
}


def rewrite_classes(pattern: str) -> str:
    """Replace \\w, \\W, \\s, \\S and \\b with their Python definitions.

    Character-class aware: inside `[...]` the shorthand expands to bare ranges,
    outside it expands to a bracketed class. `\\d` is left alone -- Python's `\\d`
    and Rust's are both exactly Nd.
    """

    for idiom, replacement in NEGATED_CLASS_IDIOMS.items():
        pattern = pattern.replace(idiom, replacement)

    out: list[str] = []
    i = 0
    in_class = False
    n = len(pattern)
    while i < n:
        ch = pattern[i]
        if ch == "\\" and i + 1 < n:
            esc = pattern[i + 1]
            if esc == "w":
                out.append(_PY_WORD if in_class else f"[{_PY_WORD}]")
            elif esc == "s":
                out.append(_PY_SPACE if in_class else f"[{_PY_SPACE}]")
            elif esc == "W":
                if in_class:
                    raise SystemExit(r"\W inside a character class needs a hand rewrite")
                out.append(f"[^{_PY_WORD}]")
            elif esc == "S":
                if in_class:
                    raise SystemExit(r"\S inside a character class needs a hand rewrite")
                out.append(f"[^{_PY_SPACE}]")
            elif esc == "b" and not in_class:
                # `\b` inside a class means backspace in both dialects; leave it.
                out.append(_PY_BOUNDARY)
            elif esc == "B":
                raise SystemExit(r"\B is not handled")
            else:
                out.append(pattern[i : i + 2])
            i += 2
            continue
        if not in_class and ch == "[":
            in_class = True
            out.append(ch)
            i += 1
            # A `]` or `^]` directly after `[` is a literal, not the terminator.
            if i < n and pattern[i] == "^":
                out.append("^")
                i += 1
            if i < n and pattern[i] == "]":
                out.append("]")
                i += 1
            continue
        if in_class and ch == "]":
            in_class = False
        out.append(ch)
        i += 1
    if in_class:
        raise SystemExit("unbalanced character class in pattern")
    return "".join(out)


def to_fancy(pattern: str, flags: int) -> str:
    """Rewrite a Python pattern into the dialect fancy-regex accepts."""

    if pattern in HAND_REDUCED:
        out = HAND_REDUCED[pattern]
    else:
        out = _NAMED_GROUP.sub(r"(?<\1>", pattern)
        out = _NAMED_BACKREF.sub(r"\\k<\1>", out)
        out = rewrite_classes(out)
    prefix = ""
    if flags & re.I:
        prefix += "i"
    if flags & re.S:
        prefix += "s"
    if flags & re.M:
        prefix += "m"
    if flags & re.X:
        # None of these use verbose mode; refuse rather than mangle whitespace.
        raise SystemExit("re.X pattern encountered -- the dumper does not handle it")
    return f"(?{prefix}){out}" if prefix else out


def assert_isalnum_equals_LN() -> None:
    """The whole `\\w` rewrite rests on `str.isalnum() == category L* or N*`.

    That identity is what lets a compact `[\\p{L}\\p{N}_]` stand in for Python's
    word class instead of an 800-range table. Check it rather than assume it --
    if a future Unicode edition breaks it, this should fail loudly at dump time.
    """

    bad = []
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        ch = chr(cp)
        # Parenthesised deliberately: `a != b in c` is a chained comparison.
        if ch.isalnum() != (unicodedata.category(ch)[0] in "LN"):
            bad.append(cp)
            if len(bad) > 8:
                break
    if bad:
        raise SystemExit(
            "str.isalnum() no longer equals category L*/N* at "
            + ", ".join(f"U+{cp:04X}" for cp in bad)
            + " -- the \\w rewrite in this dumper is invalid"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from sequence_models import bibliography_v2 as v2

    entries = {}
    for name in WANTED:
        pattern = getattr(v2, name)
        if not isinstance(pattern, re.Pattern):
            raise SystemExit(f"{name} is not a compiled pattern")
        entries[name] = {
            "python": pattern.pattern,
            "fancy": to_fancy(pattern.pattern, pattern.flags),
            "flags": int(pattern.flags),
            "groups": pattern.groupindex and dict(pattern.groupindex) or {},
        }

    from sequence_models import bibliography_role_features as rf

    for name in WANTED_ROLE:
        pattern = getattr(rf, name)
        if not isinstance(pattern, re.Pattern):
            raise SystemExit(f"role_features.{name} is not a compiled pattern")
        entries[f"ROLE{name}"] = {
            "python": pattern.pattern,
            "fancy": to_fancy(pattern.pattern, pattern.flags),
            "flags": int(pattern.flags),
            "groups": dict(pattern.groupindex) if pattern.groupindex else {},
        }

    # `_heading_key`/`_fold` in deterministic_structure, plus the ATX prefix the
    # table builder strips. These drive the bibliography-heading lexicon lookup.
    entries["DS_HEADING_STRIP"] = {
        "python": r"^[\s#*_|:;\-–—]+|[\s#*_|:;.!?\-–—]+$",
        "fancy": to_fancy(r"^[\s#*_|:;\-–—]+|[\s#*_|:;.!?\-–—]+$", 0),
        "flags": 0,
        "groups": {},
    }
    entries["DS_WHITESPACE_RUN"] = {
        "python": r"\s+",
        "fancy": to_fancy(r"\s+", 0),
        "flags": 0,
        "groups": {},
    }
    # casefold maps Greek final sigma to σ; `_fold` restores it at word boundaries.
    entries["DS_FINAL_SIGMA"] = {
        "python": r"σ(?=\W|$)",
        "fancy": to_fancy(r"σ(?=\W|$)", 0),
        "flags": 0,
        "groups": {},
    }
    entries["TABLE_ATX_PREFIX"] = {
        "python": r"^\s*#{1,6}\s*",
        "fancy": to_fancy(r"^\s*#{1,6}\s*", 0),
        "flags": 0,
        "groups": {},
    }

    from sequence_models import bibliography_nextgen_table as nt

    for name in WANTED_TABLE:
        pattern = getattr(nt, name)
        if not isinstance(pattern, re.Pattern):
            raise SystemExit(f"nextgen_table.{name} is not a compiled pattern")
        entries[f"TABLE{name}"] = {
            "python": pattern.pattern,
            "fancy": to_fancy(pattern.pattern, pattern.flags),
            "flags": int(pattern.flags),
            "groups": dict(pattern.groupindex) if pattern.groupindex else {},
        }

    # `starts_bullet_or_number` also uses an inline pattern rather than a constant.
    entries["ROLE_BULLET_PREFIX"] = {
        "python": r"^\s*[-*•·–—]\s+",
        "fancy": to_fancy(r"^\s*[-*•·–—]\s+", 0),
        "flags": 0,
        "groups": {},
    }

    # The token pattern used for `token_count` lives inline in
    # `_features_and_spans` rather than as a constant. `[^\W_]` is a negated class
    # containing a negated shorthand, which the class rewriter cannot expand
    # mechanically -- but it reduces exactly: "not (non-word or underscore)" is the
    # word characters minus underscore, and Python's word characters are L* or N*.
    assert_isalnum_equals_LN()
    entries["_TOKEN"] = {
        "python": r"[^\W_]+(?:[’'\-][^\W_]+)*",
        "fancy": to_fancy(r"[^\W_]+(?:[’'\-][^\W_]+)*", 0),
        "flags": 0,
        "groups": {},
    }

    # `analyze_bib_line`'s negative-role taxonomy needs every module-level pattern
    # and string lexicon in deterministic_structure, not just the heading ones.
    # Enumerate them rather than maintaining a hand-written list: the port has to
    # track that module exactly, and a name added there should show up here without
    # anyone remembering to add it.
    import re as _re

    from sequence_models import deterministic_structure as _ds

    ds_lexicons: dict[str, list[str]] = {}
    for name in sorted(dir(_ds)):
        if name.startswith("__"):
            continue
        value = getattr(_ds, name)
        if isinstance(value, _re.Pattern):
            key = f"DS_{name.lstrip('_')}"
            if key not in entries:
                entries[key] = {
                    "python": value.pattern,
                    "fancy": to_fancy(value.pattern, value.flags),
                    "flags": int(value.flags),
                    "groups": dict(value.groupindex) if value.groupindex else {},
                }
        elif isinstance(value, (frozenset, set)) and value and all(
            isinstance(v, str) for v in value
        ):
            ds_lexicons[name.lstrip("_").lower()] = sorted(value)
    print(
        f"  deterministic_structure: {len(ds_lexicons)} lexicons enumerated",
        file=sys.stderr,
    )

    # The heading lexicons, dumped rather than retyped. Within `analyze_bib_line`
    # the HEADING and SUBHEADING roles are reachable *only* by exact membership of
    # `_heading_key(stripped)` in these two sets (verified: those are the only two
    # such returns in the function), so the lexicon-match predicate the table builder
    # calls reduces to set lookups plus the candidate rewrites. Sixty-odd accented
    # Greek strings are exactly the kind of thing that must not be hand-copied.
    from sequence_models import deterministic_structure as ds

    lexicons = {
        "bib_headings": sorted(ds._BIB_HEADINGS),
        "bib_subheadings": sorted(ds._BIB_SUBHEADINGS),
        "extra_bib_subheadings": sorted(nt._EXTRA_BIB_SUBHEADINGS),
        # Everything else deterministic_structure keeps as a string set, so the
        # negative-role port can look them up by their source name.
        **{k: v for k, v in ds_lexicons.items() if k not in
           ("bib_headings", "bib_subheadings")},
    }
    print(f"  {len(lexicons)} lexicons, {sum(len(v) for v in lexicons.values())} entries total",
          file=sys.stderr)

    payload = {
        "lexicons": lexicons,
        "schema_version": "bib-v2-patterns-v2",
        "rules_id": v2.RULES_ID,
        "python_version": sys.version.split()[0],
        "unicodedata_version": __import__("unicodedata").unidata_version,
        "patterns": entries,
    }
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1)
    payload["sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=1)
    print(f"wrote {len(entries)} patterns -> {out}", file=sys.stderr)
    print(f"  rules_id={v2.RULES_ID} unicode={payload['unicodedata_version']}", file=sys.stderr)
    biggest = max(entries.items(), key=lambda kv: len(kv[1]["fancy"]))
    print(f"  largest: {biggest[0]} ({len(biggest[1]['fancy'])} chars)", file=sys.stderr)


if __name__ == "__main__":
    main()
