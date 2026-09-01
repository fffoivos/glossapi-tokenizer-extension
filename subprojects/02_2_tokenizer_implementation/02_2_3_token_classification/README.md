# 02.2.3 — Token classification (tiered, dataset-anchored)

> **In one line:** a design for a per-(token, dataset) tiered-label artifact that joins the strict char masks with the empirical firing histograms; it was specified, reviewed, and never built — the tier logic was instead inlined into one consumer script.
> **Period:** 2026-05-15 → 2026-05-15 (single commit `719d3834`). **Status:** abandoned as a standalone artifact; the tier vocabulary survived and was used inline by [`02_2_2`](../02_2_2_vocab_lang_attribution/) and [`02_2_4`](../02_2_4_language_category_promotion/).
> **Came from / led to:** [`02_2_1`](../02_2_1_char_language_membership/) + [`02_2_2`](../02_2_2_vocab_lang_attribution/) → this → [`02_2_4`](../02_2_4_language_category_promotion/)

## Why this existed

The char tool ([`02_2_1`](../02_2_1_char_language_membership/)) answers "which languages admit this codepoint?" under strict CLDR rules and deliberately refuses to look at data. The firing histograms ([`02_2_2`](../02_2_2_vocab_lang_attribution/)) count which tokens fire in which language corpus and deliberately refuse to interpret. Something had to combine them, and combining them requires a *defeasible* assumption — the **dataset-language premise**: in a corpus known to be predominantly language L, an L-admissible token defaults to L unless its chars rule L out. This directory was created to hold that premise explicitly, so the two upstream layers could stay pure and consumers would read one artifact instead of re-deriving the tiering each time.

## History

| Date | What happened | Result / decision | Evidence |
| --- | --- | --- | --- |
| 2026-05-15 | Directory created with a single design document; six tiers defined (T0 char-evidenced / T1 family-evidenced / T2 premise / T3 substrate / T4 excluded / T5 unknown-standalone), plus output schema, build pipeline, and validation plan for `artifacts/token_dataset_attribution.parquet` | Marked "**proposal — for review before implementation**" | [`PLAN.md`](PLAN.md), commit `719d3834` |
| 2026-05-15 | The same commit shipped `tiered_attribution.py` under the German review, whose docstring states it "Implements the tier policy from `02_2_3_token_classification/PLAN.md` **inline**… Once that sub-subproject is built out, this script becomes a consumer of its artifact" | The tier definitions ran, but only for German and only inside a consumer | [`../02_2_2_vocab_lang_attribution/analysis/german_review/tiered_attribution.py`](../02_2_2_vocab_lang_attribution/analysis/german_review/tiered_attribution.py) |
| 2026-05-15 | The parent checkpoint and the PMI spec adopted the tier vocabulary (T0/T2/T5, substrate-by-popcount) as working shorthand without the artifact existing | Tiers became a shared language, not a shared file | [`../CHECKPOINT_LANGUAGE_ATTRIBUTION_20260515.md`](../CHECKPOINT_LANGUAGE_ATTRIBUTION_20260515.md), [`../02_2_4_language_category_promotion/PMI_PROMOTION_SPEC.md`](../02_2_4_language_category_promotion/PMI_PROMOTION_SPEC.md) |
| after 2026-05-15 | No further commit touched this directory | Never implemented | pack `git_log_chronological.txt` — only `719d3834` touches it |

## Outcome

- **Nothing was built here.** The directory contains exactly one file, [`PLAN.md`](PLAN.md); no `scripts/`, no `artifacts/`, no manifest. `02_2_1` and `02_2_2` were never joined into the proposed Parquet.
- **The tier taxonomy was the real deliverable** and it was used: the substrate test (`popcount(bitmask_and) == N_LANG_BITS`), the T5 exclusion of `partial_utf8` / `byte_unmapped` / `special` statuses, and the T0/T2 distinction are all load-bearing in the PMI promotion that did run ([`../02_2_4_language_category_promotion/PMI_PROMOTION_SPEC.md`](../02_2_4_language_category_promotion/PMI_PROMOTION_SPEC.md) § Step D).
- **Two structural claims from the plan were confirmed elsewhere and stuck**: English T0 is *structurally* zero (its CLDR exemplar is a subset of every other Latin locale's), and German T0 is 103 tokens, all `ß`-bearing. Both are restated in [`../CHECKPOINT_LANGUAGE_ATTRIBUTION_20260515.md`](../CHECKPOINT_LANGUAGE_ATTRIBUTION_20260515.md) § 4.
- **Left open**: the four questions the plan itself flagged — collapsing T0/T1 for single-locale families, subdividing T4, emitting a "premise-doubt" flag, and renaming the directory. None was resolved.

## Where things are

| Artifact | Path | Note |
| --- | --- | --- |
| The design | [`PLAN.md`](PLAN.md) | tier table, premise text, output schema, validation plan, effort estimate (~3–4 h) |
| The only running implementation of the tiers | [`../02_2_2_vocab_lang_attribution/analysis/german_review/tiered_attribution.py`](../02_2_2_vocab_lang_attribution/analysis/german_review/tiered_attribution.py) | German only; inline, hardcoded to 55 language bits (the v4-era count) |

## Working documents

- [`PLAN.md`](PLAN.md) — historical proposal, never executed. It is the sole document in this directory.
