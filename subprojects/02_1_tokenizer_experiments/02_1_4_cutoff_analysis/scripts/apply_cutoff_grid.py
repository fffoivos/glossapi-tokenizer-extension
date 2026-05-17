"""Apply the established tokenizer-experiment cutoff grid to C3.

Cutoff grid (frozen in
`subprojects/02_1_tokenizer_experiments/CONTINUOUS_BPE_EXTENSION_TODO.md`
§1.4): 10240, 15360, 20480, 25600 added merges on top of the Apertus base
vocab of 131072. Each cutoff corresponds to the merged variant of total
vocab size 131072 + N.

C3 = `C3_wave2_broad_glossapi_plus_hplt_50_50` (continuous BPE, glossapi+hplt
50/50 wave-2 broad cleaner mix). Its 25600 added tokens (ids 131072..156671)
are contiguous, so each cutoff is a prefix of the glossary by id.

This script slices the corrected glossary into the four prefixes and
emits, per cutoff:
  - by_category counts
  - by_language counts
  - by_greek_structure counts
  - by_greek_lexical counts
  - confidence_buckets

Plus a single comparison table across all four cutoffs in markdown.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


BASE_VOCAB = 131072
CUTOFFS = [n * 1024 for n in range(1, 26)]
GLOSS_PATH = Path(
    "/home/foivos/runs/c2_c3_analysis_20260506/c3_added_tokens_20260507/"
    "data/glossary/tokens_glossary.jsonl"
)
# Default to the sub-subproject's local artifacts/ dir. Override with
# CUTOFF_GRID_OUT_DIR env var if you want to publish back to the
# original ~/runs location.
import os
_HERE = Path(__file__).resolve().parent
OUT_DIR = Path(os.environ.get(
    "CUTOFF_GRID_OUT_DIR",
    str((_HERE / "../artifacts/cutoff_grid").resolve()),
))


def confidence_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 0.5:
        return "<0.5"
    if value < 0.7:
        return "0.5-0.7"
    if value < 0.9:
        return "0.7-0.9"
    return ">=0.9"


def distribution(rows: list[dict]) -> dict:
    by_category: Counter = Counter()
    by_language: Counter = Counter()
    by_struct: Counter = Counter()
    by_lex: Counter = Counter()
    by_conf: Counter = Counter()

    for r in rows:
        by_category[r.get("category", "unknown")] += 1
        lang = r.get("language")
        by_language[lang if lang is not None else "null"] += 1
        morph = r.get("greek_morphology") or {}
        struct = morph.get("structure")
        lex = morph.get("lexical")
        if struct:
            by_struct[struct] += 1
        if lex:
            by_lex[lex] += 1
        by_conf[confidence_bucket(r.get("confidence"))] += 1

    def sort_desc(c: Counter) -> dict[str, int]:
        return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))

    return {
        "total": len(rows),
        "by_category": sort_desc(by_category),
        "by_language": sort_desc(by_language),
        "by_greek_structure": sort_desc(by_struct),
        "by_greek_lexical": sort_desc(by_lex),
        "confidence_buckets": sort_desc(by_conf),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with GLOSS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r["id"]))

    assert rows[0]["id"] == BASE_VOCAB, (
        f"glossary does not start at base vocab: got {rows[0]['id']}"
    )

    per_cutoff: dict[int, dict] = {}
    for n in CUTOFFS:
        upper = BASE_VOCAB + n
        sliced = [r for r in rows if r["id"] < upper]
        if len(sliced) != n:
            raise RuntimeError(
                f"cutoff {n}: expected {n} rows, got {len(sliced)}"
            )
        dist = distribution(sliced)
        per_cutoff[n] = {
            "cutoff_added_units": n,
            "total_vocab_size": upper,
            "id_range": [BASE_VOCAB, upper - 1],
            **dist,
        }
        (OUT_DIR / f"distribution_at_{n}.json").write_text(
            json.dumps(per_cutoff[n], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    union = {
        "tokenizer": rows[0].get("tokenizer", "C3_wave2_broad_glossapi_plus_hplt_50_50"),
        "base_vocab": BASE_VOCAB,
        "source_glossary": str(GLOSS_PATH),
        "cutoffs": CUTOFFS,
        "per_cutoff": per_cutoff,
    }
    (OUT_DIR / "cutoff_grid_summary.json").write_text(
        json.dumps(union, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Markdown comparison table
    md_path = OUT_DIR / "cutoff_grid_summary.md"
    cats_union = sorted({k for c in per_cutoff.values() for k in c["by_category"]})
    structs_union = sorted({k for c in per_cutoff.values() for k in c["by_greek_structure"]})
    lex_union = sorted({k for c in per_cutoff.values() for k in c["by_greek_lexical"]})
    conf_order = [">=0.9", "0.7-0.9", "0.5-0.7", "<0.5", "unknown"]

    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("# C3 — cutoff-grid distribution\n\n")
        fh.write(f"Tokenizer: `{union['tokenizer']}`  \n")
        fh.write(f"Base vocab: `{BASE_VOCAB}` — cutoffs are added-merge counts on top of base.  \n")
        fh.write(f"Source glossary: `{GLOSS_PATH}` (corrected).\n\n")

        fh.write("## Totals\n\n")
        fh.write("| cutoff | total vocab | added rows |\n| ---: | ---: | ---: |\n")
        for n in CUTOFFS:
            c = per_cutoff[n]
            fh.write(f"| {n} | {c['total_vocab_size']} | {c['total']} |\n")
        fh.write("\n")

        fh.write("## Category × cutoff\n\n")
        fh.write("| category | " + " | ".join(str(n) for n in CUTOFFS) + " |\n")
        fh.write("| --- |" + "".join(" ---: |" for _ in CUTOFFS) + "\n")
        for cat in cats_union:
            row_vals = [str(per_cutoff[n]["by_category"].get(cat, 0)) for n in CUTOFFS]
            fh.write(f"| `{cat}` | " + " | ".join(row_vals) + " |\n")
        fh.write("\n")

        fh.write("## Greek structure × cutoff\n\n")
        fh.write("| structure | " + " | ".join(str(n) for n in CUTOFFS) + " |\n")
        fh.write("| --- |" + "".join(" ---: |" for _ in CUTOFFS) + "\n")
        for s in structs_union:
            row_vals = [str(per_cutoff[n]["by_greek_structure"].get(s, 0)) for n in CUTOFFS]
            fh.write(f"| `{s}` | " + " | ".join(row_vals) + " |\n")
        fh.write("\n")

        fh.write("## Greek lexical × cutoff\n\n")
        fh.write("| lexical | " + " | ".join(str(n) for n in CUTOFFS) + " |\n")
        fh.write("| --- |" + "".join(" ---: |" for _ in CUTOFFS) + "\n")
        for s in lex_union:
            row_vals = [str(per_cutoff[n]["by_greek_lexical"].get(s, 0)) for n in CUTOFFS]
            fh.write(f"| `{s}` | " + " | ".join(row_vals) + " |\n")
        fh.write("\n")

        fh.write("## Confidence × cutoff\n\n")
        fh.write("| bucket | " + " | ".join(str(n) for n in CUTOFFS) + " |\n")
        fh.write("| --- |" + "".join(" ---: |" for _ in CUTOFFS) + "\n")
        for b in conf_order:
            if not any(b in per_cutoff[n]["confidence_buckets"] for n in CUTOFFS):
                continue
            row_vals = [str(per_cutoff[n]["confidence_buckets"].get(b, 0)) for n in CUTOFFS]
            fh.write(f"| {b} | " + " | ".join(row_vals) + " |\n")
        fh.write("\n")

    print(f"wrote {OUT_DIR / 'cutoff_grid_summary.json'}")
    print(f"wrote {md_path}")
    for n in CUTOFFS:
        print(f"  per-cutoff: {OUT_DIR / f'distribution_at_{n}.json'}")


if __name__ == "__main__":
    main()
