"""Emit the canonical added-token curation decision for the C3 tokenizer.

Applies the per-class keep/remove policy from CURATION_REPORT.md to the
corrected C3 glossary + char-mask classifier and writes:

  manifests/removal_list.jsonl    — one row per REMOVED token  (git-tracked)
  manifests/decision_summary.json — counts + per-cutoff impact (git-tracked)
  artifacts/keep_list.jsonl       — one row per KEPT token     (gitignored;
                                    bulky, regeneratable from the glossary)

The split is intentional: 02_2_tokenizer_implementation consumes the
manifests/ files, so they live under a git-tracked dir; the keep list
is only useful for ad-hoc inspection.

The script does NOT modify the tokenizer.json. It produces an
implementation manifest for 02_2_tokenizer_implementation to consume.

Removal rules (revised 2026-05-17 after audit-discussion):

  REMOVE if glossary.category == "mojibake"
    -> Latin-1-as-UTF-8 mojibake from misencoded Greek bytes (ÉÉ, Ø, ØØ,
       ÉÉÉÉ, ØØØØ, Ô).

  REMOVE if glossary.category == "mixed_script_token"     (WHOLE category)
    -> Greek-Latin lookalike substitution mojibake (τo, Tο, Tα, Oι,
       Ωστόσο) plus punctuation+Greek BPE-boundary fragments (.Ε, ,τι,
       /και). All of these are extraction/tokenization artifacts: the
       cleaner emits text where punctuation got fused with the next
       Greek letter, or where Latin lookalikes substituted for Greek
       letters. They are too infrequent in any single surface form for
       the model to learn anything useful, the same pattern repeats
       in many other surface forms not represented in the vocab, and
       byte-fallback composition handles all of them at input time
       without any single-token affordance.

  REMOVE if glossary.category == "postscript_glyph"       (WHOLE category)
    -> PDF/PostScript font-glyph-name leakage (slash-prefixed Greek
       glyph references like /Α, /η, /pi, /Σ, /Δ). Pure PDF-extraction
       residue.

  REMOVE if glossary.category == "code_identifier"        (WHOLE category)
    -> Cleaner-emitted line-boundary placeholders (LINENEWLINE,
       NEWLINENEWLINE). The cleaner should have replaced these with
       actual newlines; they're not real identifiers.

  REMOVE if glossary.category == "latin_acronym"
                AND decoded in REMOVE_LATIN_ACRONYM_LINENEW_FRAGS
    -> BPE fragments of LINENEWLINE that landed as their own tokens
       (LIN, ENEW, LINENEW). Same family as the code_identifier removal.
       Genuine acronyms like EURO and MEG stay.

  REMOVE if glossary.category == "latin_fragment"
                AND decoded in REMOVE_LATIN_FRAGMENT_TAGS
    -> Cleaner extraction-tag fragments (-missing, -decoded). Not real
       Latin words; placeholder fragments from HTML/XML-tag-attribute
       extraction that survived cleaning. Greek-surname transliteration
       fragments (opoulou, oulou, gean) stay.

  KEEP otherwise.

Output schema: each removal entry contains
  id, decoded, category, lang_bucket, reason, removal_class
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO = Path("/home/foivos/Projects/glossapi-tokenizer-extension")
GLOSSARY = Path(
    "/home/foivos/runs/c2_c3_analysis_20260506/c3_added_tokens_20260507/"
    "data/glossary/tokens_glossary.jsonl"
)
CLASSIFIED = (
    REPO
    / "subprojects/02_1_tokenizer_experiments/02_1_4_cutoff_analysis/"
    / "artifacts/classified_added_tokens.jsonl"
)
_SSP_ROOT = (
    REPO
    / "subprojects/02_1_tokenizer_experiments/02_1_5_added_token_curation"
)
# Canonical outputs the implementer (02_2) will read live under the
# git-tracked `manifests/` dir; the bulky keep_list is regeneratable
# from the glossary, so it can stay under the .gitignored `artifacts/`.
MANIFEST_DIR = _SSP_ROOT / "manifests"
ARTIFACTS_DIR = _SSP_ROOT / "artifacts"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

REMOVE_LATIN_FRAGMENT_TAGS = {"-missing", "-decoded"}
REMOVE_LATIN_ACRONYM_LINENEW_FRAGS = {"LIN", "ENEW", "LINENEW"}


def decide(glossary_row: dict, classified_row: dict) -> tuple[bool, str | None]:
    cat = glossary_row.get("category")
    decoded = glossary_row.get("decoded") or ""
    lang_bucket = classified_row.get("lang_bucket")

    if cat == "mojibake":
        return True, "latin1_utf8_mojibake"
    if cat == "mixed_script_token":
        # Whole category. Includes both Greek-Latin lookalike font-sub
        # mojibake (τo, Tο, Tα, Oι, Ωστόσο) and punctuation+Greek BPE-
        # boundary fragments (.Ε, ,τι, /και). Both are too infrequent in
        # any single surface form to deserve dedicated tokens.
        return True, "mixed_script_artifact"
    if cat == "postscript_glyph":
        return True, "pdf_postscript_glyph"
    if cat == "code_identifier":
        return True, "cleaner_linenewline_placeholder"
    if cat == "latin_acronym" and decoded in REMOVE_LATIN_ACRONYM_LINENEW_FRAGS:
        return True, "cleaner_linenewline_bpe_fragment"
    if cat == "latin_fragment" and decoded in REMOVE_LATIN_FRAGMENT_TAGS:
        return True, "cleaner_extraction_tag"
    return False, None


def main() -> None:
    glossary_rows = {}
    with GLOSSARY.open() as fh:
        for line in fh:
            r = json.loads(line)
            glossary_rows[int(r["id"])] = r

    classified_rows = {}
    with CLASSIFIED.open() as fh:
        for line in fh:
            r = json.loads(line)
            classified_rows[int(r["id"])] = r

    assert set(glossary_rows.keys()) == set(classified_rows.keys()), \
        "id sets differ between glossary and classified outputs"

    removals = []
    keeps = []
    for tok_id in sorted(glossary_rows.keys()):
        g = glossary_rows[tok_id]
        c = classified_rows[tok_id]
        is_remove, reason = decide(g, c)
        rec = {
            "id": tok_id,
            "decoded": g.get("decoded"),
            "category": g.get("category"),
            "lang_bucket": c.get("lang_bucket"),
            "removal_class": reason if is_remove else None,
        }
        if is_remove:
            rec["meaning_snippet"] = (g.get("meaning") or "")[:160]
            removals.append(rec)
        else:
            keeps.append(rec)

    # Canonical manifest (tracked in git, consumed by 02_2)
    with (MANIFEST_DIR / "removal_list.jsonl").open("w") as fh:
        for r in removals:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    # Bulky keep_list — regenerable; stays under .gitignored artifacts/
    with (ARTIFACTS_DIR / "keep_list.jsonl").open("w") as fh:
        for r in keeps:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary: counts per removal_class + per-cutoff impact
    by_class = Counter(r["removal_class"] for r in removals)
    cutoffs = [n * 1024 for n in range(1, 26)]
    BASE_VOCAB = 131_072
    per_cutoff = {}
    for n in cutoffs:
        upto_max_id = BASE_VOCAB + n - 1
        n_remove = sum(1 for r in removals if r["id"] <= upto_max_id)
        per_cutoff[n] = {
            "added_units": n,
            "removable_in_cutoff": n_remove,
            "removable_pct": (n_remove / n) * 100,
        }
    summary = {
        "removable_total_at_25600": len(removals),
        "keepable_total_at_25600": len(keeps),
        "removable_by_class": dict(by_class),
        "removable_per_cutoff": per_cutoff,
        "policy_version": "2026-05-17b",
        "rules": [
            {"class": "latin1_utf8_mojibake",
             "predicate": "glossary.category == 'mojibake'",
             "scope": "whole category"},
            {"class": "mixed_script_artifact",
             "predicate": "glossary.category == 'mixed_script_token'",
             "scope": "whole category — includes Greek-Latin lookalike font-sub mojibake AND punctuation+Greek BPE-boundary fragments"},
            {"class": "pdf_postscript_glyph",
             "predicate": "glossary.category == 'postscript_glyph'",
             "scope": "whole category"},
            {"class": "cleaner_linenewline_placeholder",
             "predicate": "glossary.category == 'code_identifier'",
             "scope": "whole category"},
            {"class": "cleaner_linenewline_bpe_fragment",
             "predicate": "glossary.category == 'latin_acronym' AND decoded in {'LIN','ENEW','LINENEW'}",
             "scope": "narrow subset (LINENEW family); keeps EURO, MEG"},
            {"class": "cleaner_extraction_tag",
             "predicate": "glossary.category == 'latin_fragment' AND decoded in {'-missing','-decoded'}",
             "scope": "narrow subset (extraction tags); keeps opoulou, oulou, gean, etc."},
        ],
    }
    (MANIFEST_DIR / "decision_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    print(f"removal_list.jsonl    {len(removals)} rows")
    print(f"keep_list.jsonl       {len(keeps)} rows")
    print(f"decision_summary.json written")
    print()
    print("By removal class:")
    for cls, n in sorted(by_class.items(), key=lambda kv: -kv[1]):
        print(f"  {cls:<30s}  {n:>4d}")
    print()
    print("Per-cutoff removable counts:")
    print(f"  {'cutoff':>6s}  {'removable':>10s}  {'%':>6s}")
    for n in cutoffs:
        c = per_cutoff[n]
        print(f"  {n:>6d}  {c['removable_in_cutoff']:>10d}  {c['removable_pct']:>5.2f}%")
    print()
    print("All removed tokens (id : category : decoded):")
    for r in removals:
        print(f"  {r['id']:>6d}  [{r['removal_class']:<30s}]  [{r['category']:<22s}]  {r['decoded']!r}")


if __name__ == "__main__":
    main()
