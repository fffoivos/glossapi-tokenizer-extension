"""v3 — build the 11-language per-language groups file from PMI-attributed sets.

Loads each language's masked token file (PMI-distinctive tokens), assembles a
single groups.json with token-id lists per language. Also captures the
structural-substrate token set (uncovered categories 3 + 7) for totals-only
accounting.

Output:
  geometry/v3_perlang/groups.json
    {
      "languages": {
        "ell_Grek": [int, ...],
        "hin_Deva": [int, ...],
        ...
      },
      "structural_substrate_mass_only": [int, ...],   # cats 3+7 from uncovered_tokens.tsv
      "totals": {
        "scenario_A_no_additions": {
          "mass_with_anchor": int, "mass_without_anchor": int, "anchor": "ell_Grek"
        },
        "scenario_B_moderate_additions": {...}
      },
      "summary_source": "<path>"
    }
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path("/home/foivos/runs/apertus_embedding_init_test_20260512")
SP = Path(
    "/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/"
    "03_apertus_extension_and_embedding_adaptation/03_1_greek_embedding_diagnostic"
)
PMI_BASE = Path(
    "/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/"
    "02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution/"
    "analysis/main_token_sets_pmi"
)
OUT = SP / "artifacts" / "geometry" / "v3_perlang" / "groups.json"

LANGUAGES = [
    # tier 1 — Greek analogues (≈1:1 script-language, tight + high-mass)
    "ell_Grek", "hin_Deva", "hye_Armn", "heb_Hebr", "kat_Geor", "tha_Thai",
    # tier 2 — script-1:1 but wider footprint
    "kor_Hang", "fas_Arab",
    # top 3 — most popular by vocab footprint
    "eng_Latn_fineweb_hq", "fra_Latn", "deu_Latn",
]


def parse_masked(p: Path) -> list[int]:
    ids = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"\{(\d+):\s*(.+)\}", line)
        if m:
            ids.append(int(m.group(1)))
    return ids


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    summary = {r["target_key"]: r for r in csv.DictReader(
        open(PMI_BASE / "summary.tsv"), delimiter="\t")}

    languages = {}
    mass_per_lang = {}
    for k in LANGUAGES:
        ids = parse_masked(PMI_BASE / "tables" / f"{k}__masked.txt")
        languages[k] = sorted(ids)
        mass_per_lang[k] = int(summary[k]["masked_mass"])

    # Structural substrate for totals-only (uncovered cats 3 + 7)
    sub_ids = []
    sub_mass = 0
    with open(PMI_BASE / "uncovered_tokens.tsv", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            if row["category"] in ("3_substrate", "7_PMI_below_delta_for_every_lang"):
                sub_ids.append(int(row["token_id"]))
                sub_mass += int(row["fire_anywhere"] or 0)
    sub_ids = sorted(set(sub_ids))

    anchor = "ell_Grek"
    total_lang_mass = sum(mass_per_lang.values())
    total_without_anchor = total_lang_mass - mass_per_lang[anchor]
    totals = {
        "scenario_A_no_additions": {
            "anchor": anchor,
            "n_languages": len(LANGUAGES),
            "anchor_mass": mass_per_lang[anchor],
            "mass_with_anchor": total_lang_mass,
            "mass_without_anchor": total_without_anchor,
            "anchor_share_pct": 100.0 * mass_per_lang[anchor] / total_lang_mass,
        },
        "scenario_B_moderate_additions": {
            "anchor": anchor,
            "n_languages": len(LANGUAGES),
            "added_category": "structural_substrate (uncovered cats 3 + 7) — totals only",
            "added_mass": sub_mass,
            "added_token_count": len(sub_ids),
            "mass_with_anchor": total_lang_mass + sub_mass,
            "mass_without_anchor": total_without_anchor + sub_mass,
            "anchor_share_pct": 100.0 * mass_per_lang[anchor] / (total_lang_mass + sub_mass),
        },
    }

    out = {
        "languages": languages,
        "structural_substrate_mass_only": sub_ids,
        "mass_per_lang": mass_per_lang,
        "totals": totals,
        "summary_source": str(PMI_BASE / "summary.tsv"),
        "build_script": __file__,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")
    print(f"  languages: {len(LANGUAGES)}")
    for k in LANGUAGES:
        print(f"    {k:<22s} n={len(languages[k]):>6,d}  mass={mass_per_lang[k]:>14,d}")
    print(f"  structural_substrate (totals only): {len(sub_ids):,} tokens, mass={sub_mass:,}")
    print(f"  Scenario A: mass_with_anchor={totals['scenario_A_no_additions']['mass_with_anchor']:,}, "
          f"anchor_share={totals['scenario_A_no_additions']['anchor_share_pct']:.2f}%")
    print(f"  Scenario B: mass_with_anchor={totals['scenario_B_moderate_additions']['mass_with_anchor']:,}, "
          f"anchor_share={totals['scenario_B_moderate_additions']['anchor_share_pct']:.2f}%")


if __name__ == "__main__":
    main()
