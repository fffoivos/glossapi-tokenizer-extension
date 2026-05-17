"""v4 — assemble the canonical groups file with ALL well-sampled languages
from the PMI attribution corpus (those with non-empty masked sets).

Drops 7 of v3's 88 canonical keys that have empty masked sets (their scripts
are byte-fragmented by Apertus's tokenizer): amh_Ethi, khm_Khmr, sin_Sinh,
lao_Laoo, bod_Tibt, ory_Orya, div_Thaa.

Two English samples coexist in the source; we keep BOTH so the pipeline
can run on the FineWeb-HQ variant as the canonical English and the wiki
variant as a within-language domain-shift reference.

Structural-substrate mass is computed exactly as in v3 (uncovered cats 3+7)
for Scenario A / B totals.

Output:
  artifacts/geometry/v4_perlang/groups.json
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

SP = Path(
    "/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/"
    "03_apertus_extension_and_embedding_adaptation/03_1_greek_embedding_diagnostic"
)
PMI_BASE = Path(
    "/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/"
    "02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution/"
    "analysis/main_token_sets_pmi"
)
OUT = SP / "artifacts" / "geometry" / "v4_perlang" / "groups.json"

EMPTY_KEYS = {"amh_Ethi", "khm_Khmr", "sin_Sinh", "lao_Laoo",
                "bod_Tibt", "ory_Orya", "div_Thaa"}


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

    keys_all = sorted(summary.keys())
    languages = {}
    mass_per_lang = {}
    skipped = []
    for k in keys_all:
        if k in EMPTY_KEYS:
            skipped.append((k, "empty masked set per integration report"))
            continue
        masked_path = PMI_BASE / "tables" / f"{k}__masked.txt"
        if not masked_path.exists():
            skipped.append((k, "masked.txt not found"))
            continue
        ids = parse_masked(masked_path)
        if not ids:
            skipped.append((k, "empty after parse"))
            continue
        languages[k] = sorted(ids)
        mass_per_lang[k] = int(summary[k]["masked_mass"])

    # Structural substrate (cats 3+7) — totals only
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
    total_without_anchor = total_lang_mass - mass_per_lang.get(anchor, 0)
    totals = {
        "scenario_A_no_additions": {
            "anchor": anchor,
            "n_languages": len(languages),
            "anchor_mass": mass_per_lang.get(anchor, 0),
            "mass_with_anchor": total_lang_mass,
            "mass_without_anchor": total_without_anchor,
            "anchor_share_pct": (100 * mass_per_lang.get(anchor, 0) / total_lang_mass) if total_lang_mass else 0.0,
        },
        "scenario_B_moderate_additions": {
            "anchor": anchor,
            "n_languages": len(languages),
            "added_category": "structural_substrate (uncovered cats 3+7) — totals only",
            "added_mass": sub_mass,
            "added_token_count": len(sub_ids),
            "mass_with_anchor": total_lang_mass + sub_mass,
            "mass_without_anchor": total_without_anchor + sub_mass,
            "anchor_share_pct": (100 * mass_per_lang.get(anchor, 0) / (total_lang_mass + sub_mass)) if (total_lang_mass + sub_mass) else 0.0,
        },
    }

    out = {
        "languages": languages,
        "structural_substrate_mass_only": sub_ids,
        "mass_per_lang": mass_per_lang,
        "totals": totals,
        "summary_source": str(PMI_BASE / "summary.tsv"),
        "skipped_keys": skipped,
        "build_script": __file__,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")
    print(f"  languages: {len(languages)}")
    print(f"  skipped:   {len(skipped)}")
    for k, reason in skipped:
        print(f"    {k}: {reason}")
    print(f"  structural_substrate: {len(sub_ids):,} tokens, mass={sub_mass:,}")
    print(f"  Scenario A: total mass {totals['scenario_A_no_additions']['mass_with_anchor']:,}, "
          f"Greek share {totals['scenario_A_no_additions']['anchor_share_pct']:.2f}%")
    print(f"  Scenario B: total mass {totals['scenario_B_moderate_additions']['mass_with_anchor']:,}, "
          f"Greek share {totals['scenario_B_moderate_additions']['anchor_share_pct']:.2f}%")


if __name__ == "__main__":
    main()
