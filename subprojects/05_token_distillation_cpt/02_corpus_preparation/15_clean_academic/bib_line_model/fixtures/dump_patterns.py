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

_NAMED_GROUP = re.compile(r"\(\?P<([A-Za-z_][A-Za-z0-9_]*)>")
_NAMED_BACKREF = re.compile(r"\(\?P=([A-Za-z_][A-Za-z0-9_]*)\)")


def to_fancy(pattern: str, flags: int) -> str:
    """Rewrite a Python pattern into the dialect fancy-regex accepts."""

    out = _NAMED_GROUP.sub(r"(?<\1>", pattern)
    out = _NAMED_BACKREF.sub(r"\\k<\1>", out)
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

    # Also emit the token pattern used for `token_count`; it lives inline in
    # `_features_and_spans` rather than as a constant.
    entries["_TOKEN"] = {
        "python": r"[^\W_]+(?:[’'\-][^\W_]+)*",
        "fancy": r"[^\W_]+(?:[’'\-][^\W_]+)*",
        "flags": 0,
        "groups": {},
    }

    payload = {
        "schema_version": "bib-v2-patterns-v1",
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
