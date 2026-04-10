#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from itertools import combinations
from pathlib import Path


FIELD_ORDER = ["has_greek", "has_latin", "has_polytonic", "no_accents", "is_uppercase"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a saved tokenizer variant with added Greek tokens.")
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def is_greek_letter(char: str) -> bool:
    if not char.isalpha():
        return False
    codepoint = ord(char)
    return 0x0370 <= codepoint <= 0x03FF or 0x1F00 <= codepoint <= 0x1FFF


def is_latin_letter(char: str) -> bool:
    if not char.isalpha():
        return False
    name = unicodedata.name(char, "")
    return "LATIN" in name


def has_polytonic_char(text: str) -> bool:
    return any(0x1F00 <= ord(char) <= 0x1FFF for char in text)


def has_any_accent(text: str) -> bool:
    return any(unicodedata.combining(char) for char in unicodedata.normalize("NFD", text))


def is_uppercase_token(text: str) -> bool:
    cased = [char for char in text if char.isalpha() and char.lower() != char.upper()]
    return bool(cased) and all(char == char.upper() for char in cased)


def token_flags(token: str) -> dict[str, bool]:
    has_greek = any(is_greek_letter(char) for char in token)
    has_latin = any(is_latin_letter(char) for char in token)
    return {
        "has_greek": has_greek,
        "has_latin": has_latin,
        "has_polytonic": has_polytonic_char(token),
        "no_accents": any(char.isalpha() for char in token) and not has_any_accent(token),
        "is_uppercase": is_uppercase_token(token),
    }


def combination_key(flags: dict[str, bool], fields: list[str]) -> str:
    active = [field for field in fields if flags[field]]
    return "+".join(active) if active else "(none)"


def venn_regions_three(records: list[dict], a: str, b: str, c: str) -> dict[str, int]:
    region_counts: Counter[str] = Counter()
    for record in records:
        active = []
        for field in (a, b, c):
            if record["flags"][field]:
                active.append(field)
        region_counts["+".join(active) if active else "(none)"] += 1
    return dict(sorted(region_counts.items()))


def pairwise_overlap_counts(records: list[dict], fields: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for left, right in combinations(fields, 2):
        counts[f"{left}&{right}"] = sum(1 for record in records if record["flags"][left] and record["flags"][right])
    return counts


def emit_simple_venn_svg(title: str, region_counts: dict[str, int], fields: tuple[str, str, str], output_path: Path) -> None:
    a, b, c = fields
    values = {
        a: region_counts.get(a, 0),
        b: region_counts.get(b, 0),
        c: region_counts.get(c, 0),
        f"{a}+{b}": region_counts.get(f"{a}+{b}", 0),
        f"{a}+{c}": region_counts.get(f"{a}+{c}", 0),
        f"{b}+{c}": region_counts.get(f"{b}+{c}", 0),
        f"{a}+{b}+{c}": region_counts.get(f"{a}+{b}+{c}", 0),
        "(none)": region_counts.get("(none)", 0),
    }
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="900" height="680" viewBox="0 0 900 680">
  <style>
    .title {{ font: 24px sans-serif; font-weight: bold; }}
    .label {{ font: 22px sans-serif; font-weight: bold; }}
    .count {{ font: 22px monospace; }}
    .note {{ font: 18px sans-serif; }}
  </style>
  <rect width="100%" height="100%" fill="white"/>
  <text x="40" y="50" class="title">{title}</text>
  <circle cx="320" cy="280" r="170" fill="#7dcfb6" fill-opacity="0.45" stroke="#3a7d6b" stroke-width="3"/>
  <circle cx="500" cy="280" r="170" fill="#f4d35e" fill-opacity="0.45" stroke="#a17a00" stroke-width="3"/>
  <circle cx="410" cy="420" r="170" fill="#ee964b" fill-opacity="0.45" stroke="#9c5417" stroke-width="3"/>
  <text x="190" y="120" class="label">{a}</text>
  <text x="560" y="120" class="label">{b}</text>
  <text x="385" y="610" class="label">{c}</text>
  <text x="210" y="290" class="count">{values[a]}</text>
  <text x="575" y="290" class="count">{values[b]}</text>
  <text x="405" y="500" class="count">{values[c]}</text>
  <text x="408" y="275" class="count">{values[f"{a}+{b}"]}</text>
  <text x="320" y="390" class="count">{values[f"{a}+{c}"]}</text>
  <text x="500" y="390" class="count">{values[f"{b}+{c}"]}</text>
  <text x="410" y="345" class="count">{values[f"{a}+{b}+{c}"]}</text>
  <text x="40" y="650" class="note">Outside all three sets: {values["(none)"]}</text>
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")


def main() -> None:
    args = parse_args()
    tokenizer = json.loads(args.tokenizer_json.read_text(encoding="utf-8"))
    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    base_vocab_size = int(summary["variant_results"]["base_vocab_size"])
    added_tokens = tokenizer["added_tokens"]
    base_vocab = tokenizer["model"]["vocab"]

    preexisting_non_special = [
        row for row in added_tokens if int(row["id"]) < base_vocab_size and not bool(row["special"])
    ]
    new_tokens = [
        row for row in added_tokens if int(row["id"]) >= base_vocab_size
    ]

    records: list[dict] = []
    for row in new_tokens:
        token = str(row["content"])
        flags = token_flags(token)
        records.append(
            {
                "id": int(row["id"]),
                "token": token,
                "flags": flags,
            }
        )

    base_vocab_keys = set(base_vocab)
    preexisting_non_special_tokens = {row["content"] for row in preexisting_non_special}

    exact_overlap_base_vocab = sorted(record["token"] for record in records if record["token"] in base_vocab_keys)
    exact_overlap_preexisting_added = sorted(record["token"] for record in records if record["token"] in preexisting_non_special_tokens)

    field_counts = {field: sum(1 for record in records if record["flags"][field]) for field in FIELD_ORDER}
    pairwise = pairwise_overlap_counts(records, FIELD_ORDER)
    combinations_all: Counter[str] = Counter(combination_key(record["flags"], FIELD_ORDER) for record in records)

    greek_poly_upper = venn_regions_three(records, "has_greek", "has_polytonic", "is_uppercase")
    greek_latin_upper = venn_regions_three(records, "has_greek", "has_latin", "is_uppercase")

    stats = {
        "tokenizer_json": str(args.tokenizer_json),
        "summary_json": str(args.summary_json),
        "base_vocab_size": base_vocab_size,
        "new_added_token_count": len(records),
        "preexisting_non_special_added_tokens": preexisting_non_special,
        "field_definitions": {
            "has_greek": "Token contains at least one Greek letter from Greek/Coptic or Greek Extended.",
            "has_latin": "Token contains at least one Latin alphabetic character.",
            "has_polytonic": "Token contains at least one codepoint in Greek Extended (U+1F00-U+1FFF).",
            "no_accents": "Token contains alphabetic characters and no combining diacritics after NFD normalization.",
            "is_uppercase": "Token has at least one cased letter and all cased letters are uppercase.",
        },
        "field_counts": field_counts,
        "pairwise_overlap_counts": pairwise,
        "combination_counts": dict(sorted(combinations_all.items())),
        "venn_regions": {
            "has_greek_has_polytonic_is_uppercase": greek_poly_upper,
            "has_greek_has_latin_is_uppercase": greek_latin_upper,
        },
        "exact_overlap_with_base_vocab": exact_overlap_base_vocab,
        "exact_overlap_with_preexisting_non_special_added_tokens": exact_overlap_preexisting_added,
    }

    (output_dir / "added_token_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "preexisting_non_special_added_tokens.json").write_text(
        json.dumps(preexisting_non_special, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "exact_overlap_with_base_vocab.txt").write_text(
        "\n".join(exact_overlap_base_vocab) + ("\n" if exact_overlap_base_vocab else ""),
        encoding="utf-8",
    )
    (output_dir / "exact_overlap_with_preexisting_non_special_added_tokens.txt").write_text(
        "\n".join(exact_overlap_preexisting_added) + ("\n" if exact_overlap_preexisting_added else ""),
        encoding="utf-8",
    )

    emit_simple_venn_svg(
        title="New Added Tokens: Greek vs Polytonic vs Uppercase",
        region_counts=greek_poly_upper,
        fields=("has_greek", "has_polytonic", "is_uppercase"),
        output_path=output_dir / "venn_greek_polytonic_uppercase.svg",
    )
    emit_simple_venn_svg(
        title="New Added Tokens: Greek vs Latin vs Uppercase",
        region_counts=greek_latin_upper,
        fields=("has_greek", "has_latin", "is_uppercase"),
        output_path=output_dir / "venn_greek_latin_uppercase.svg",
    )

    lines = [
        "# Added Token Statistics",
        "",
        f"- Base vocab size: `{base_vocab_size}`",
        f"- New added token count: `{len(records)}`",
        f"- Pre-existing non-special added tokens: `{len(preexisting_non_special)}`",
        "",
        "## Pre-existing Non-Special Added Tokens",
        "",
    ]
    for row in preexisting_non_special:
        lines.append(f"- `{row['content']}` (id `{row['id']}`)")
    lines.extend(
        [
            "",
            "## Field Counts",
            "",
        ]
    )
    for field in FIELD_ORDER:
        lines.append(f"- `{field}`: `{field_counts[field]}`")
    lines.extend(
        [
            "",
            "## Pairwise Overlaps",
            "",
        ]
    )
    for key, value in sorted(pairwise.items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Exact Overlap With Existing Token Strings",
            "",
            f"- base vocab overlap count: `{len(exact_overlap_base_vocab)}`",
            f"- pre-existing non-special added-token overlap count: `{len(exact_overlap_preexisting_added)}`",
            "",
            "## Combination Counts",
            "",
        ]
    )
    for key, value in sorted(combinations_all.items()):
        lines.append(f"- `{key}`: `{value}`")
    (output_dir / "added_token_stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
