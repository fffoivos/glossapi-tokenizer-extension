# Token classification — dataset-anchored, tiered attribution

> Status: **proposal — for review before implementation.**

Sister to `02_2_1_char_language_membership/`. The char tool answers
*"which (scripts, families, languages) admit this codepoint?"* under
strict rules. This tool layers on the **dataset signal** and the
**dataset-language premise** to produce per-(token, dataset) labels.

## Why a separate sub-subproject

Char identification (`02_2_1_char_language_membership/`) is governed by CLDR
exemplars + Unicode-script closures. It is strict-rule, source-
authoritative, and never makes a defeasible call. The token-side
labels we want for downstream analysis (embedding-norm clusters,
training-coverage estimates, per-language vocabulary inventories)
require **two** ingredients that the char tool deliberately refuses
to mix in:

1. **The dataset signal** — which dataset(s) a token actually fired
   in, and with what mass. Lives in
   `02_2_2_vocab_lang_attribution/outputs/histogram_matrix.npz`.
2. **A dataset-language premise** — a defeasible working assumption
   that, in a corpus we know to be predominantly language L, an
   L-admissible token defaults to L unless its chars rule L out.

By isolating both ingredients here, the char tool stays a pure
reference and the consumer code (analyses, plots, dashboards) reads a
single artifact instead of re-implementing the tiering each time.

## Scope — and what this is *not*

In scope:

- For each in-scope language L and each fired token in L's dataset,
  emit a **tiered label**: *definitely-L (char-evidenced)* /
  *definitely-L-family (char-evidenced)* / *could-be-L (premise)* /
  *substrate* / *excluded* / *unknown-standalone*.
- The label includes a `basis` string explaining why the tier was
  assigned (which bits triggered, which closure applied).
- Output is one Parquet artifact keyed by (token_id, dataset). Schema
  is fixed; consumers slice it.

Explicitly **not** in scope:

- **Cross-dataset reconciliation.** A token fires in many datasets;
  this tool emits one row per (token, dataset) pair. It does *not*
  merge those rows into a single global "this token's language" label.
- **Firing-rate ratios across sister languages** (e.g. de-dominant
  ≥10× vs en). That comparison is a downstream consumer of this
  artifact, not part of it. The artifact provides the per-dataset
  tiering; the consumer can ratio across datasets as needed.
- **N-gram / morphological signals.** Out of scope. The char masks
  plus the premise give us a defensible coarse label; finer
  attribution that needs language-model evidence is a different
  project.
- **Threshold-based filtering.** This tool does not propose count
  thresholds. The artifact carries raw counts; downstream picks
  cutoffs.

## Inputs

| input | source | role |
| --- | --- | --- |
| `token_language_bitmask.parquet` | `02_2_1_char_language_membership/artifacts/` | per-token char-evidenced masks at script / family / language level (schema_version 4) |
| `manifest.json` (char tool) | same | bit indices for languages, families, scripts |
| `histogram_matrix.npz` | `02_2_2_vocab_lang_attribution/outputs/` | per-token × per-language-dataset firing counts (1,933 × 131,072) |
| `families.yaml` (or read from manifest) | char tool | language → family membership |

## The tier hierarchy (per dataset language L)

For each token `t` fired in L's dataset (`count_L(t) > 0`):

| tier | rule | premise needed? |
| --- | --- | --- |
| **T0 — definitely-L** | `bitmask_and(t) == (1 << bit_L)` exactly (popcount 1) — chars are admissible **only** in L among 55 in-scope locales | no — char evidence alone |
| **T1 — definitely-L-family** | `family_and(t)` has exactly L's family bit set, AND `bitmask_and(t)` has L-bit set | no for family certainty; **yes** to commit to L specifically over its family-siblings |
| **T2 — could-be-L (premise)** | `bitmask_and(t)` has L-bit set, `popcount(bitmask_and) < 55`, and the token is not in T0 or T1 | **yes** — dataset-premise call |
| **T3 — substrate** | `popcount(bitmask_and) == 55` (every in-scope locale admits) | n/a — not language-attributable from chars |
| **T4 — excluded (non-L char)** | `bitmask_and(t)` does **not** have L-bit set, token has decoded text (status is evaluable) | n/a — char evidence rules L out |
| **T5 — unknown standalone** | `status in {partial_utf8, byte_unmapped, special}` | n/a — char tool cannot evaluate |

Notes:

- Tiers are mutually exclusive and exhaustive for any token with
  `count_L > 0`.
- T0 is **strictly empty for English** by construction: en's CLDR
  exemplar is `[A-Za-z]`, a subset of every other Latin locale's
  exemplar, so no codepoint is `en`-exclusive among the 55 locales.
  This is a structural property, not a data artifact, and is one of
  the headline outputs of the framework.
- T0 for German is **103 vocab tokens** — all `ß`-bearing. T0 for
  Greek (`el`-only or `el ∩ el-polyton`-only) is large. T0 for
  most Latin locales is empty or near-empty.
- T1 lets us emit a weaker certainty: "Germanic-Latn-family-only"
  (171 tokens) — narrows to en/de/nl/da/nb/sv/is by char alone, then
  the German-dataset premise picks de. The basis string distinguishes
  T1-with-premise from T0-without.

## The premise — explicit, falsifiable, scoped

The premise is a single sentence, recorded in the output:

> "In the dataset for language L, a token whose chars are
> L-admissible (`bitmask_and` has L-bit set) is **provisionally**
> attributed to L. This is a working assumption that may be false for
> any individual token — common false attributions include
> loanwords from sister languages, code identifiers, proper names
> from other locales, and quoted text. The premise is not used for
> tiers T0, T1, T3, T4, or T5."

Concrete consequences:

- A pure-ASCII token like `the`, fired in both English and German
  datasets, lands at **T2 in English** (premise → en) and **T2 in
  German** (premise → de) — two rows, two different per-dataset
  labels, no contradiction. The artifact does not try to pick one.
- A `ß`-bearing token fired in both datasets lands at **T0 in German**
  (no premise) and **T4 in English** (char evidence rules en out).
- The downstream consumer is free to override T2 (premise) using
  firing-rate ratios across sister-language datasets — that's the
  natural place to peel English loanwords out of the German T2 bucket
  and vice versa.

## Output schema

`artifacts/token_dataset_attribution.parquet`:

| column | type | meaning |
| --- | --- | --- |
| `token_id` | uint32 | token id in Apertus vocab |
| `dataset` | string | canonical key like `eng_Latn`, `deu_Latn`, `ell_Grek` |
| `count` | uint64 | firing count in that dataset |
| `tier` | string | one of `T0_definitely_lang`, `T1_definitely_family`, `T2_premise_lang`, `T3_substrate`, `T4_excluded`, `T5_unknown` |
| `lang_code` | string | ISO 639-3 + script suffix matching the dataset (the language we attribute to) |
| `basis` | string | short rule label, e.g. `bitmask_and_only_de` / `family_and_only_germanic_latn_premise_de` / `bitmask_and_no_de_bit` |
| `script_and` | binary(4) | denormalised for downstream filtering (avoid re-joining) |
| `family_and` | binary(4) | same |
| `bitmask_and` | binary(16) | same |
| `popcount_language` | uint8 | popcount of `bitmask_and` |

`artifacts/manifest.json`:

| key | value |
| --- | --- |
| `schema_version` | 1 |
| `built_from_char_schema_version` | 4 (pin) |
| `histogram_matrix_md5` | pin |
| `premise` | the one-sentence premise text |
| `tiers` | tier name → rule (machine-readable) |
| `languages_processed` | list |
| `per_language_tier_counts` | { lang: { tier: token_types_fired } } |

## Build pipeline

`scripts/build_token_dataset_attribution.py`:

1. Load `token_language_bitmask.parquet`. Project bitmask_and /
   family_and / script_and to integer arrays per token.
2. Load `histogram_matrix.npz`. For each language L in
   `canonical_keys`:
   1. Read column `H[L]` (per-token firing count).
   2. Find `fired_idx = where(H[L] > 0)`.
   3. For each fired token, compute the tier using the table above.
   4. Append rows to the output dataframe.
3. Write Parquet (Polars), write manifest.

Pure deterministic transformation — no probabilities, no thresholds.
Re-runs from the same inputs produce byte-identical output.

## Validation

`scripts/validate.py`:

- For every (token, dataset) row in the output:
  - Recompute the tier from inputs; assert match.
  - Assert `count` matches `H[dataset][token_id]`.
- Per-language summary checks:
  - English T0 count == 0 (structural).
  - German T0 count == 103 (verified empirically against current
    artifacts; pin the number, error if it drifts).
  - For every language with a single-locale family bit, T1 count is
    consistent with `family_and` rows in `02_2_1_char_language_membership/`.
- Tier totals per language sum to `fired_token_types` in
  `02_2_2_vocab_lang_attribution/analysis/membership_rejection/`. (Cross-check
  against an independent counter built earlier.)

## Downstream consumers (illustrative, not in this project)

- `02_2_2_vocab_lang_attribution/analysis/english_review/` — replace the
  "Latin family" partition with the per-language T0/T1/T2/T3/T4/T5
  partition. Plots gain a tiered legend.
- `02_2_2_vocab_lang_attribution/analysis/german_review/` — same. The
  german-distinctive set (1,431 tokens, de-cap AND NOT en-cap) is
  superseded by T0+T1+T2-with-Germanic-only-family.
- `02_2_2_vocab_lang_attribution/analysis/membership_rejection/` —
  `rejected_decoded_mass` becomes T4 mass; `unknown_standalone_mass`
  becomes T5 mass; `compatible_mass` splits into T0+T1+T2+T3 (with
  the substrate broken out for the first time).
- Any embedding-norm analysis that wants "all tokens this language
  uses, char-certified": filter to T0+T1. For "all tokens this
  language dataset spends mass on": T0+T1+T2+T3.

## Open questions

1. **T1 family-tier policy for languages whose family has only one
   in-scope locale** (e.g. Hangul → ko; Devanagari → hi; el → Grek-modern).
   Should those languages collapse T0 and T1 (since family certainty
   is equivalent to language certainty for a 1-locale family)? Default
   in this draft: **yes, collapse** — emit T0 with `basis` noting the
   family is single-locale.
2. **Whether T4 (excluded) should subdivide** into "foreign-script"
   (different script_and entirely) vs "Latin-with-non-L-letter" (e.g.
   Polish `ć` in the German dataset). Probably yes, but a second
   pass. Default in this draft: single T4 bucket with `basis` carrying
   the subdivision.
3. **Whether to also emit a "premise-doubt" flag** for T2 tokens that
   fire far more in some sister language (e.g. `the` in the German
   dataset). The flag is just a derived ratio test, easy to add. It
   blurs the strict-rule character of this tool; arguably belongs in
   the consumer. Default: **don't emit; consumers compute**.
4. **Naming**: `02_2_3_token_classification` is the working dir name. Open
   to renaming to e.g. `token_lang_attribution` if that reads better
   against `02_2_2_vocab_lang_attribution`.

## Estimated work

- `build_token_dataset_attribution.py`: 1–2 h (single deterministic
  transform; main cost is mapping language codes ↔ bit positions
  ↔ family bit positions, plus the `lang_code` ↔ canonical-key map).
- `validate.py`: 1 h.
- Manifest + README: 0.5 h.
- First-run audit (per-language tier counts sanity-check against the
  membership_rejection table): 0.5 h.

Total ~3–4 h for implementation + validation. The downstream
consumer rewrites (English / German / Greek review) are a separate
step.
