#!/usr/bin/env python3
"""Dump the Unicode predicates the feature stack depends on, as codepoint ranges.

`line_shape` asks questions Rust's standard library answers *almost* the same way,
and "almost" is the problem:

* `category(ch).startswith("L")` is the general category, not `char::is_alphabetic`
  (which is the Alphabetic property -- it accepts U+0345, category Mn, and other
  non-L characters).
* `str.isdigit()` is Nd plus digit-valued No, not `char::is_numeric` (Nd|Nl|No).
* `_letters_by_script` classifies a letter by whether its *character name* contains
  "GREEK" or "LATIN". That is close to the Script property but not identical, and
  Rust has no character-name table at all.

Rather than approximate each one and hope the disagreements never land on a real
line, dump them from the same interpreter that runs the reference implementation
and compress to ranges. The Rust side then answers these questions from Python's
own tables, and `unicodedata_version` records which edition they came from.

    python3 bib_line_model/fixtures/dump_unicode_tables.py --out bib_line_model/unicode_tables.json
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

MAX_CODEPOINT = 0x110000

PREDICATES = {
    # general-category prefixes used by line_shape
    "cat_L": lambda ch: unicodedata.category(ch).startswith("L"),
    "cat_N": lambda ch: unicodedata.category(ch).startswith("N"),
    "cat_P": lambda ch: unicodedata.category(ch).startswith("P"),
    "cat_S": lambda ch: unicodedata.category(ch).startswith("S"),
    # script attribution, by character name, exactly as _letters_by_script does it
    "name_greek": lambda ch: "GREEK" in unicodedata.name(ch, ""),
    "name_latin": lambda ch: "LATIN" in unicodedata.name(ch, ""),
    # str methods -- each is a distinct Python predicate with its own edge cases
    "isupper": str.isupper,
    "islower": str.islower,
    "isspace": str.isspace,
    "isdigit": str.isdigit,
    "isalpha": str.isalpha,
    "isalnum": str.isalnum,
}


def ranges_for(predicate) -> list[tuple[int, int]]:
    """Run-length encode the codepoints satisfying `predicate`."""

    out: list[tuple[int, int]] = []
    start: int | None = None
    for cp in range(MAX_CODEPOINT):
        # Surrogates are not valid `char` values in Rust and never appear in a
        # Python str read from UTF-8; skip rather than emit an unusable range.
        if 0xD800 <= cp <= 0xDFFF:
            hit = False
        else:
            hit = bool(predicate(chr(cp)))
        if hit and start is None:
            start = cp
        elif not hit and start is not None:
            out.append((start, cp - 1))
            start = None
    if start is not None:
        out.append((start, MAX_CODEPOINT - 1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tables = {}
    for name, predicate in PREDICATES.items():
        rs = ranges_for(predicate)
        tables[name] = [[a, b] for a, b in rs]
        print(f"  {name:<12} {len(rs):>5} ranges", file=sys.stderr)

    payload = {
        "schema_version": "bib-unicode-tables-v1",
        "python_version": sys.version.split()[0],
        "unicodedata_version": unicodedata.unidata_version,
        "tables": tables,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    print(
        f"wrote {len(tables)} tables -> {out} "
        f"({out.stat().st_size / 1024:.0f} KiB, unicode {payload['unicodedata_version']})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
