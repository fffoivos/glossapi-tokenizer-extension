# Language-category promotion for the Apertus tokenizer

> Status: **proposal — for review before implementation.**

Sister to `02_2_1_char_language_membership/` (char identification) and
`02_2_3_token_classification/` (per-(token, dataset) tiered labels). This
sub-subproject **promotes**, per language category, a defensible set
of token ids that the embedding diagnostic
([`subprojects/03_apertus_extension_and_embedding_adaptation/03_1_greek_embedding_diagnostic/`](../../03_apertus_extension_and_embedding_adaptation/03_1_greek_embedding_diagnostic/))
can consume as `groups_<category>.json` files, in the same role that
`base_greek_tokens.jsonl` (1,494 strict-Greek ids) currently fills.

## What the embedding diagnostic needs

The Greek diagnostic at `03_1` runs an entire pipeline over a single
canonical set:

- `base_greek_tokens.jsonl` — 1,494 strict-Greek ids (decoded-as-Greek-only).
- `groups_greek_vs_not.json` — `{Greek: [...], not_Greek: [...]}` partitions
  the classified subset of the vocab.
- 17 scripts compute per-group centroids, spectra (Marchenko-Pastur edge,
  K_significant, anisotropy), within-group hulls + infiltrators, binary
  Greek-vs-¬Greek logistic classifier, k-means subspace clustering,
  morphological-family analyses, and cross-language semantic-cluster cosines.

The README explicitly states:

> *"Awaiting the user's per-language attribution before reopening the
> per-language version of this diagnostic for non-Greek script-aggregates
> (Cyrillic / CJK / Latin-script)."*

And the v1 multi-group classifier had `GROUPS_OF_INTEREST = ["English-
baseline", "Greek", "Cyrillic", "German", "French", "CJK",
"structural_non_linguistic"]`. So the consumer already expects
per-language and per-script-aggregate groups.

What this means concretely: for each category C the diagnostic wants to
analyse, this sub-subproject must emit a sorted list of token ids that
**we can defend as "the canonical tokens of C"** — knowing C has
embedding rows worth comparing against ¬C.

## Goal

Produce, for every language category we promote:

```
artifacts/categories/<category>.jsonl   # one row per promoted token id
artifacts/groups.json                   # global groups index
artifacts/manifest.json                 # provenance, knob values, summary
```

with a fixed schema (below) and explicit basis per id, so the diagnostic
can swap `base_greek_tokens.jsonl` for `categories/Greek.jsonl` and run
unchanged.

## Categories to promote

Three layers, mutually exclusive at the language level, mergeable at the
aggregate level:

1. **Single-language** — one promoted set per in-scope language whose
   data signal supports it. Examples: `English`, `German`, `Greek`,
   `French`, `Russian`, `Japanese`, `Hindi`. Roughly the 25–30 high-
   firing-mass languages in our histogram.
2. **Aggregate** — union of single-language sets sharing a script or
   family. Examples: `Cyrillic` = ru ∪ uk ∪ bg ∪ mk ∪ sr-Cyrl;
   `Germanic-Latn` = en ∪ de ∪ nl ∪ da ∪ nb ∪ sv ∪ is; `CJK` = zh-Hans
   ∪ zh-Hant ∪ ja. Useful when individual locales have weak signal but
   the script-aggregate has strong signal.
3. **Structural** — non-linguistic but vocab-grounded: `Substrate`
   (punctuation/digits/whitespace, popcount==55), `ByteFragment`
   (partial-utf8), `SpecialToken` (Apertus special ids). These mirror
   the Greek diagnostic's `structural_non_linguistic` bucket.

A residual category `Unattributable` collects tokens we cannot promote
to any single-language or aggregate set with confidence (firing-rate
too low, statistically ambiguous). Listed for traceability but not
loaded by the diagnostic.

## The promotion rule, per language regime

Different languages have structurally different evidence profiles. The
promotion logic adapts to the regime each language falls into.

### Regime A — strong T0 (char-evidenced) language

Examples: **Greek** (~1,500 T0 tokens in the script-distinctive pc=2
{el, el-polyton} bucket), **German** (103 T0 tokens, ß-bearing).

Rule:

```
promoted_L = T0_L  ∪  confident-T2_L
```

where `T0_L` is the char-evidenced tier from `02_2_3_token_classification/`
and `confident-T2_L` is the subset of T2 tokens whose firing
distribution in L's dataset is statistically distinguishable from L's
nearest sister languages (test defined below).

For Greek specifically, T0 is the entire current `base_greek_tokens.jsonl`
set (1,494 strict-Greek ids = pc-1 `{el}` ∪ pc-2 `{el, el-polyton}`)
plus a few additions char-evidence brings (`el-polyton`-only tokens
that weren't decoded-as-Greek-only by the old classifier).
**Backward-compatible** with the existing diagnostic: the strict-Greek
set should be a subset of `T0_Greek`.

### Regime B — empty or near-empty T0 language

Examples: **English** (T0 = 0 by structure — en's CLDR exemplar is a
subset of every other Latin locale's), every other Latin-script locale
without distinctive characters (Italian, Indonesian, Vietnamese w/o
tone marks, …).

Rule:

```
promoted_L = confident-T2_L only (no T0 to seed)
```

`confident-T2_L` here is the load-bearing concept: we **must** require
a statistical test against sister languages to call a token English /
French / Italian. Otherwise we'd promote substrate-adjacent tokens
(`the`, `of`, `and`) that fire in every Latin corpus and learn nothing
new from the embedding analysis.

### Regime C — aggregate-only

Examples: **Cyrillic-aggregate** when no single Cyrillic language has
enough mass on its own to support promotion. **CJK-aggregate** for
zh-Hans + zh-Hant + ja taken together when the ja-vs-zh distinction
is unreliable in our 1 B-per-language sample.

Rule:

```
promoted_aggregate_A = ⋃ promoted_L  for L in A.constituents,
       restricted to tokens whose firing is concentrated within A
       (i.e. mass_outside_A < ε * mass_inside_A).
```

The aggregate promotion is run **after** the single-language promotions
so that we don't double-count tokens (a Cyrillic token promoted under
ru is not also promoted "again" as Cyrillic — the Cyrillic aggregate
is built from the same ids with an aggregate-level basis label).

## The statistical test for confident-T2

For each token `t` and each language `L` under premise, define:

- `p_L(t) = (count_L(t) + α) / (Σ_t' count_L(t') + α·V)` — normalized
  firing rate. α = 0.5 Laplace smoothing.
- `competitor(L)` = the sister language with the second-highest
  normalized rate on `t` (could be empty if no other language fires
  on `t`).
- **Effect size**: `Δ(t, L) = log10(p_L(t) / p_competitor(t))`.
- **Evidence**: total count `N(t) = count_L(t) + count_competitor(t)`.
- **Credibility**: 95% Beta-Binomial CI on the proportion
  `p = count_L(t) / N(t)`; map back to log-ratio.

Promotion criteria for T2 token `t` under language `L`:

1. `N(t) ≥ min_count` — evidence floor.
2. **Lower bound of 95% CI on log-ratio ≥ δ** — effect size is at
   least δ with 95% credibility, ruling out the "shared" tier.

Suggested initial knob settings (for review):

- `min_count = 100` (orphans rare tail but covers >95% of mass).
- `δ = 0.5` (≥ 3× preference for L over the strongest sister).

These knobs are tunable per regime. For Regime A (strong T0) the knobs
can be looser because T0 carries the certainty; for Regime B (no T0)
the knobs should be stricter because we're entirely premise-dependent.

The competitor choice matters:

- For English: competitor is whichever sister Latin locale fires most
  on the token. So `the` (en=38M, de=400k, fr=…) competes against the
  highest non-en candidate; we promote `the` to English if its rate in
  English exceeds the best sister rate by ≥3× with 95% CI.
- For German: competitor is the highest sister Latin locale. `der` (de=15.8M,
  en=44k, nl=…) easily clears the bar; `Test` (de=high, en=high) may not.
- For Russian: competitor is the highest sister Cyrillic locale.

## Mass-coverage targets (sanity floor)

After promotion, the promoted set for L must cover **≥ 90% of L's
non-substrate, non-foreign-script mass** in L's dataset. If less,
either the knobs are too tight (lower `min_count` / `δ`) or L doesn't
have enough signal to support promotion (escalate to aggregate-only).

Per-language mass-coverage report is part of the artifact.

## Output schema

`artifacts/categories/<category>.jsonl` (one record per token):

```json
{
  "id":              12345,
  "decoded_text":    " Verbindung",
  "basis":           "T2_confident",
  "competitor":      "en",
  "count_L":         847291,
  "count_competitor": 1240,
  "log_ratio":       2.83,
  "log_ratio_ci_lo": 2.79,
  "log_ratio_ci_hi": 2.87,
  "ce_tier":         "T2"
}
```

`basis` ∈ `{T0, T1, T2_confident, T2_aggregate, special, substrate}`.
T0 / T1 / special / substrate rows skip the statistical fields
(they were promoted by char or structural evidence).

`artifacts/groups.json` is the diagnostic-consumer index:

```json
{
  "schema_version":      1,
  "categories": {
    "Greek":   { "n": 1612, "path": "categories/Greek.jsonl" },
    "German":  { "n": 8240, "path": "categories/German.jsonl" },
    "English": { "n": 7115, "path": "categories/English.jsonl" },
    "Cyrillic":{ "n": 5320, "path": "categories/Cyrillic.jsonl" },
    "CJK":     { "n": 4090, "path": "categories/CJK.jsonl" },
    "structural_non_linguistic": { "n": 7058, "path": "categories/structural_non_linguistic.jsonl" }
  },
  "promotion_knobs": { "min_count": 100, "delta_log_ratio": 0.5 },
  "premise_text":    "single sentence from 02_2_3_token_classification/PLAN.md",
  "char_membership_schema_version": 4
}
```

`artifacts/manifest.json` carries inputs' pins (histogram_matrix MD5,
char-tool schema version), the per-language statistical-test result
distributions, and per-category mass-coverage numbers.

## Compatibility with the existing Greek diagnostic

The Greek diagnostic loads `base_greek_tokens.jsonl` (1,494 ids). To
swap to the new infrastructure with **zero diagnostic-side changes**:

```
artifacts/categories/Greek.jsonl     # contains the 1,494 strict-Greek ids
                                       (plus ~100 additional confident-T2
                                        Greek tokens if our criteria add them)
```

Symlink `base_greek_tokens.jsonl` to `categories/Greek.jsonl` (renaming
the `decoded_text` field if needed; otherwise the schema is a superset).
Run the diagnostic; verify the Greek-vs-¬Greek classifier still hits
macro-F1 ≈ 0.99 / 1.00. If yes, we have a clean upgrade path.

For non-Greek diagnostics: re-run the same 17 scripts with
`categories/German.jsonl` in place of `categories/Greek.jsonl`. The
binary classifier, centroid + spectrum, hull + infiltrators, k-means
clustering, and cross-language cluster cosines all generalise; the
morphology-family analysis (`phase0_greek_families_and_analogies.py`)
needs a language-specific morpheme list and would be re-parameterised
per language.

## Pipeline

`scripts/promote_categories.py`:

1. Load `02_2_3_token_classification/artifacts/token_dataset_attribution.parquet`
   (once that artifact exists; for the bootstrap iteration, compute
   tiers inline as the German analysis does today).
2. Load `02_2_2_vocab_lang_attribution/outputs/histogram_matrix.npz` for raw
   counts.
3. For each promoted language `L`:
   1. Collect `T0_L`.
   2. For each `T2_L` candidate, identify the strongest sister
      competitor, compute `Δ`, `N`, 95% CI.
   3. Promote candidates meeting `min_count` and `CI_lo ≥ δ`.
   4. Write `categories/<L>.jsonl`.
4. For each aggregate `A`, union promoted sets and filter.
5. Write `categories/structural_non_linguistic.jsonl` from T3 + T5 +
   Apertus special ids.
6. Write `groups.json` + `manifest.json`.
7. Run `validate.py`.

`scripts/validate.py`:

- **Disjointness** — no token id appears in more than one
  single-language category.
- **Aggregate consistency** — every aggregate's id list ⊇ union of
  constituent single-language id lists (or ⊆ when filtered).
- **Backward compat** — `Greek.jsonl` ⊇ current
  `base_greek_tokens.jsonl` id set.
- **Mass coverage** — per-category fraction of L's non-substrate
  non-foreign mass on promoted tokens ≥ 90% (or report as escalation
  to aggregate).
- **No-cross-promotion** — for every promoted (L, token), the token's
  count in L exceeds the next-highest sister by `10^δ` (consistent
  with the CI test).
- **Provenance pin** — verify char-tool schema_version and
  histogram_matrix MD5 match recorded values.

## Open questions

1. **Knob settings** — `min_count` and `δ` defaults are sketched
   above; final values depend on per-language mass-coverage profiles.
   Recommend a calibration pass: run promotion at three knob settings
   per language and inspect the trade-off curves before committing.
2. **Aggregate-promotion policy** — do we always promote aggregates,
   or only when single-language promotion fails for some constituent?
   Default: emit aggregates for every multi-locale script (Cyrillic,
   CJK, Indic-script families) regardless, because the diagnostic
   needs them anyway for cross-script comparisons.
3. **Sister-language scope for the competitor test** — should English
   compete against all 27 other Latin locales, or just same-family
   (Germanic + Romance, i.e. nearest neighbours)? Default: full Latin
   family — most conservative, most defensible.
4. **What to do with non-distinctive tokens** — pure-substrate-adjacent
   tokens (`,`, `.`, digits) get attributed nowhere by the rate test.
   They land in `Substrate`. But what about pure ASCII letters that
   fire comparably in en/fr/it/es ("data", "test", "code")? Default:
   they go to `Unattributable` rather than being force-promoted to
   the largest-mass language.
5. **Backward-compat for the `base_greek_tokens.jsonl` interface** —
   keep the legacy filename or migrate the diagnostic to read
   `categories/Greek.jsonl` directly? Default: write
   `categories/Greek.jsonl` and add a stub `base_greek_tokens.jsonl`
   that's a symlink; remove the symlink once 03_1 is updated.
6. **Polytonic Greek** — promote as a separate category `Greek-polyton`
   or fold into `Greek`? The diagnostic currently treats them together;
   the char layer separates them. Default: emit both, let consumers
   pick.

## Estimated work

- `promote_categories.py` (core promotion loop): **2-3 h**
  (statistical test per token + I/O).
- `validate.py`: **1 h**.
- Manifest + README: **0.5 h**.
- Per-language calibration sweep + knob selection: **2 h**
  (small grid of `min_count × δ`, with per-language mass-coverage and
  promoted-set-size curves).
- Backward-compat validation against the existing Greek diagnostic
  (re-run the binary classifier on `categories/Greek.jsonl`, confirm
  macro-F1 ≈ 0.99 / 1.00): **1 h**.

Total: **~6-7 h** if `02_2_3_token_classification/` artifact exists; **~10 h**
if we have to compute tiers inline first. The bottleneck is the
calibration sweep — the rest is mechanical.
