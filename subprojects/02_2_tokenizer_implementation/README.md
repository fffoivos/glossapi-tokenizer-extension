# 02.2 Tokenizer Implementation

## Scope

Implement the actual compatible Greek `BPE` discovery and merge-rule extension workflow.

## Already Decided

- do not ship `add_tokens(...)`
- patch Apertus through `model.vocab` and `model.merges`
- preserve all old token ids
- append only new ids
- emit a manifest of every newly added unit
- enforce final vocab divisibility by `128`

## Required Checks

- exact preservation of the first `1000` ids
- exact preservation of special-token behavior
- exact preservation of regex split and byte-level behavior
- non-Greek smoke test after extension

## Sub-subprojects — pipeline ordering

The four sub-subprojects form a linear pipeline. Stages 1 and 2 are
independent inputs; stages 3 and 4 consume both.

```
  02_2_1_char_language_membership  ─┐
       (strict-rule, char masks)    │
                                    ├→ 02_2_3_token_classification → 02_2_4_language_category_promotion → 03_1_greek_embedding_diagnostic
  02_2_2_vocab_lang_attribution  ───┘
       (empirical firing counts)
```

### `02_2_1_char_language_membership/`

Strict-rule char-level admissibility masks at three resolutions:
script (22 bits), family (31 bits), language (55 bits). Output is a
per-codepoint Parquet table plus per-token AND/OR aggregations across
the Apertus vocab. Derived purely from CLDR exemplars + Unicode-script
closures; no dataset signal. The reference layer everything downstream
joins against.

- Read first: [`02_2_1_char_language_membership/README.md`](02_2_1_char_language_membership/README.md)
- Current plan: [`02_2_1_char_language_membership/PLAN.md`](02_2_1_char_language_membership/PLAN.md) (canonical), [`PLAN_v3_HIERARCHICAL.md`](02_2_1_char_language_membership/PLAN_v3_HIERARCHICAL.md) (shipped, schema v4)
- Artifacts: `02_2_1_char_language_membership/artifacts/{char,token}_language_bitmask.parquet`

### `02_2_2_vocab_lang_attribution/`

Per-token firing histograms across 1,933 canonical language/dataset
keys (FineWeb-2 + Wikipedia + EuroParl + ParaDocs + FineWeb-Edu +
FineWeb-HQ + DCLM-Edu), ~1 B Apertus-tokens per key. Output is the
1,933 × 131,072 histogram_matrix plus token metadata. Empirical
observation layer; no char-tool dependency.

- Read first: [`02_2_2_vocab_lang_attribution/RUN_REPORT.md`](02_2_2_vocab_lang_attribution/RUN_REPORT.md)
- Scripts: [`02_2_2_vocab_lang_attribution/scripts/`](02_2_2_vocab_lang_attribution/scripts/)
- Final outputs: `02_2_2_vocab_lang_attribution/outputs/histogram_matrix.npz`, `token_metadata.parquet`
- Downstream analyses: `02_2_2_vocab_lang_attribution/analysis/{greek_review, english_review, german_review, script_family_composition, membership_rejection}/`

### `02_2_3_token_classification/` (proposal)

Per-(token, dataset) tiered labels: T0 char-evidenced / T1 family-evidenced
/ T2 premise / T3 substrate / T4 excluded / T5 unknown-standalone.
Deterministic transform over (char masks + histogram_matrix) producing one
Parquet artifact keyed by (token_id, dataset). The dataset-language premise
lives here — explicit, falsifiable, defeasible.

- Read first: [`02_2_3_token_classification/PLAN.md`](02_2_3_token_classification/PLAN.md)
- Status: design proposal; tiers computed inline by
  `02_2_2_vocab_lang_attribution/analysis/german_review/tiered_attribution.py`
  pending implementation of the artifact.

### `02_2_4_language_category_promotion/` (proposal)

Curated per-language token-id sets ("the canonical English tokens",
"the canonical German tokens", etc.) drop-in for the embedding diagnostic
([`03_1_greek_embedding_diagnostic/`](../../03_apertus_extension_and_embedding_adaptation/03_1_greek_embedding_diagnostic/)).
Replaces the legacy `base_greek_tokens.jsonl` interface with a uniform
`categories/<L>.jsonl` schema. Per-language regime (strong-T0 / empty-T0
/ aggregate-only) plus a per-token rate-distinctiveness test against
sister languages.

- Read first: [`02_2_4_language_category_promotion/PLAN.md`](02_2_4_language_category_promotion/PLAN.md)
- Status: design proposal; no artifacts yet.

