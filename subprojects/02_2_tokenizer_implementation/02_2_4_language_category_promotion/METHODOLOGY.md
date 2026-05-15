# Language attribution methodology — filter vs weight, with empirical comparison

> **Companion to `PLAN.md` for this sub-subproject.** PLAN sketches the
> per-language category artifacts; this doc is the methodology
> reference that future agents apply when adding a new language or a
> new downstream consumer.

## Scope

This document is about deciding, **per token × per language**, what to
hand to a downstream embedding-analysis consumer (canonical example:
`subprojects/03_apertus_extension_and_embedding_adaptation/03_1_greek_embedding_diagnostic/`).

In scope:

- The two parallel candidate methods (distribution-filtering and
  distribution-weighting) and their sub-variants.
- The empirical comparison plan that decides, **per diagnostic**, which
  method wins on which metric.
- Per-language regime templates (strong-T0, empty-T0,
  aggregate-only) — placeholders until the comparison runs.

Out of scope:

- The embedding-analysis pipeline itself (lives in 03_1 and successors).
- Char-level admissibility derivation (lives in `02_2_1_char_language_membership/`).
- Per-token firing histograms (lives in `02_2_2_vocab_lang_attribution/`).
- Tier assignment per (token, dataset) (lives in `02_2_3_token_classification/`).

This doc consumes the three above and produces the
attribution artifacts the embedding diagnostic loads.

## The shared foundation

Regardless of which method we pick, every per-language attribution
**must** apply a methodology-agnostic hard mask first:

| condition | exclude? | reason |
| --- | --- | --- |
| `status ∈ {partial_utf8, byte_unmapped, special}` (T5) | yes | char tool cannot evaluate; not language-bearing |
| `bitmask_and` has no bit set AND has decoded text (T4-no-script) | yes | no in-scope locale admits any char |
| `bitmask_and` does **not** have L's bit (T4-non-L-char) | yes for L | at least one char is L-impossible |
| Apertus special tokens | yes | not vocabulary content |

After the hard mask, every fired token in L's universe falls into
one of:

- **T0** — `bitmask_and == {L}` (popcount 1, only L)
- **T1** — `family_and == {L's family}` only, AND L-bit set
- **T2** — L-bit set, popcount in `(1, N_LANG_BITS)`, not T1
- **T3** — substrate (`popcount == N_LANG_BITS`, where `N_LANG_BITS` is read from the char-tool manifest)

These four tiers are the input to BOTH methods below.

## Method F — distribution-filtering (categorical)

Output: a discrete `categories/<L>.jsonl` set. One token id is either
in the set or not. Downstream consumer treats the set as the canonical
token list for L.

### F sub-variants to evaluate

| variant | rule | knobs |
| --- | --- | --- |
| **F1** | T0 ∪ T1 ∪ (T2 with `count_L ≥ min_count`) | `min_count` |
| **F2** | T0 ∪ T1 ∪ (T2 with `PMI(t, L) ≥ δ_PMI`) | `δ_PMI` |
| **F3** | T0 ∪ T1 ∪ (T2 with Beta-Binomial 95% CI on `log_ratio(t, L vs strongest sister) ≥ δ_logratio`) | `δ_logratio`, `min_count` |
| **F4** | top-K by `p(t \| L) = count_L(t) / total_L` | `K` |

F3 is the test designed in `PLAN.md § "confident-T2"`. F4 is the
crudest baseline (per-language quantile cut). F1 is char-evidenced
floor only. F2 uses pointwise mutual information as a single-axis cut.

### F output schema

`artifacts/categories/<L>.jsonl`:

```json
{
  "id":             12345,
  "decoded_text":   " Verbindung",
  "tier":           "T2",
  "basis":          "F3",
  "count_L":        847291,
  "competitor":     "en",
  "count_competitor": 1240,
  "log_ratio":      2.83,
  "log_ratio_ci_lo": 2.79,
  "PMI":            3.41
}
```

The `basis` field carries the F-variant id so consumers can audit which
filter produced this set.

## Method W — distribution-weighting (continuous)

Output: a per-token continuous weight `w(t, L) ∈ [0, 1]` stored in a
`weights/<L>.parquet` table. Downstream consumer uses weights in
weighted centroids, weighted covariance, soft logistic regression,
etc.

### W sub-variants to evaluate

| variant | weight formula | knobs |
| --- | --- | --- |
| **W1** | `w(t, L) ∝ p(t \| L) = count_L(t) / total_L` | (none — pure normalisation) |
| **W2** | `w(t, L) ∝ count_L(t)^α` (sublinear power; equivalent to temperature softmax) | `α ∈ (0, 1)` |
| **W3** | `w(t, L) ∝ log(count_L(t) + 1)` | (none) |
| **W4** | `w(t, L) ∝ max(0, PMI(t, L))` | (none — PMI subtracts a corpus-wide baseline) |

All are normalised per-language to sum to 1 over L's hard-masked
universe.

PMI = `log(p(t \| L) / p(t))` where `p(t)` is the cross-language
marginal:
`p(t) = Σ_L' (count_{L'}(t)) / Σ_L' total_{L'}`.

### W output schema

`artifacts/weights/<L>.parquet`:

| column | type | meaning |
| --- | --- | --- |
| `token_id` | uint32 | |
| `decoded_text` | str | |
| `tier` | str | T0 / T1 / T2 / T3 |
| `count_L` | uint64 | raw firing count in L's dataset |
| `p_L` | float64 | `count_L / total_L` |
| `PMI` | float64 | `log(p_L / p_marginal)` |
| `w_W1` ... `w_W4` | float64 each | per-variant normalised weight |

Variants are stored side-by-side so the comparison harness reads one
file per language.

## The comparison plan — the spine of this document

The choice between F and W (and between sub-variants of each) is
**empirical**, not a priori. For each phase 3 diagnostic that can
consume either method, we run both and compare on a fixed metric.
Then we ship the winning variant **per diagnostic**, not as a
universal default.

### Diagnostic-by-diagnostic comparison table

| 03_1 diagnostic | input shape needed | F applicable? | W applicable? | comparison metric | tiebreaker if close |
| --- | --- | --- | --- | --- | --- |
| binary L-vs-¬L logistic classifier | per-token label | yes (hard labels) | yes (soft labels) | macro F1 on held-out fold, AUROC | F (simpler, debuggable) |
| centroid μ_L | per-token (set or weight) | mean over set | weighted mean | bootstrap-resample stability of `‖μ_L − μ_global‖` | F (more interpretable centroid) |
| within-L spectrum / K_significant | covariance over rows | covariance over set | weighted covariance | bootstrap-resample stability of K_sig and PR | W (uses more rows) |
| within-L hull occupancy | discrete L set | required | derive set from `w ≥ τ_w` | hull-volume stability + infiltrator-list overlap | F (set-shaped question) |
| infiltrators (¬L inside L hull) | discrete L set | required | derive set from threshold | requires F by construction | F |
| morphology-family clusters | tokens with optional weights | yes | weighted k-means | per-family cluster purity | W (weighting downweights outliers) |
| Mikolov analogies | discrete tokens | required | top-K from weights | analogy accuracy on held-out triples | F |
| en↔L semantic-cluster cosine | weighted centroid per cluster | unweighted per-token | weighted | cosine stability under resampling | W |

Two diagnostics (infiltrators, hull occupancy) **require** a discrete
set by construction. Everything else is genuinely comparable.

### Evaluation harness

A single harness script under `02_2_4_language_category_promotion/scripts/compare_F_vs_W.py`
that:

1. Loads `categories/<L>.jsonl` (F output, all variants) and
   `weights/<L>.parquet` (W output, all variants).
2. For each diagnostic that admits both, runs the diagnostic at
   each (variant, knob-setting) and records the comparison metric.
3. Emits `artifacts/comparison/<L>/<diagnostic>.csv` with the score
   per variant.
4. Aggregates to `artifacts/comparison/_summary.tsv` with one row per
   (language, diagnostic) and the winning variant.

Per-language and per-diagnostic decision criteria:

- **Significant win** (≥ 3 standard errors on the bootstrap): ship the
  winner.
- **Tie** (overlapping CIs): apply the tiebreaker column above.
- **Both inconclusive** (huge CI overlap): ship F because it's simpler
  to debug, and flag the language for source-confound or sample-size
  issues.

### Bootstrap resampling protocol

For each comparison: hold the histogram_matrix fixed; resample tokens
WITHIN the language pool (drawing token ids with replacement, weighted
by `count_L`). Compute the diagnostic on each resample. Standard error
is the bootstrap SE over 100 resamples. This isolates the methodology
choice from the firing-count noise.

## Per-language regime templates

The right F/W variants likely differ by regime. **Initial expectation
only — to be revised by the comparison harness**.

### Strong-T0 regime (Greek, German, Polish, Czech, Vietnamese, …)

T0 is large and char-certified. Sufficient as a categorical core.

| candidate | expected to work |
| --- | --- |
| F1 (T0 ∪ T1 ∪ count-thresholded T2) | yes — T0 anchors the set |
| F3 (T0 ∪ T1 ∪ CI-tested T2) | yes — gives strict + premise |
| W4 (PMI) | yes — substrate self-cancels |

### Empty-T0 regime (English)

T0 = 0 by structure. Everything rests on the rate test.

| candidate | expected to work |
| --- | --- |
| F1 | risky — counts alone don't separate from sister-Latin |
| F3 | required — CI vs strongest sister is the only honest discrete option |
| W4 (PMI) | natural fit — distinctive English tokens have positive PMI by definition |

### Aggregate-only regime (Cyrillic, CJK, Indic-script)

Per-language signal is weak; aggregate signal is strong.

| candidate | expected to work |
| --- | --- |
| F1 on the aggregate (union of constituents + script_and filter) | yes |
| W4 on the aggregate | yes — PMI against the cross-aggregate baseline |

## Validation and sanity checks (applicable to both methods)

Run after every promotion / weighting build, language-by-language:

1. **Char-mask consistency** — every token in F's output, and every
   token with `w > 0` in W's output, passes the hard-mask conditions.
   No char-excluded tokens leak through.
2. **Substrate behaviour** — for W4 (PMI), substrate tokens have
   PMI ≈ 0 in every language (sanity: cross-language baseline is
   close to per-language frequency for substrate).
3. **Mass coverage** — for F, `Σ count_L(t) for t in set / total_L ≥
   0.90` (less than 90 % covered = knobs too tight).
4. **Cross-language disjointness sanity** — for F: per-token,
   `count(L_with_F-membership)` should be 1 in expectation. If many
   tokens land in multiple languages' F sets, the rate test is
   under-discriminating.
5. **Backward compat for Greek** — `categories/Greek.jsonl` ⊇ current
   `base_greek_tokens.jsonl` (1,494 strict-Greek ids). Existing
   diagnostic numbers stay reproducible.
6. **Source-domain audit** — for every language with non-trivial
   attribution, log which source the histogram came from
   (`lang_metadata.json`'s `sources_contributed`). Flag any
   cross-language comparison where sources mismatch and the
   sister-language is from a different domain.

## Open issues

1. **Power-law normalisation choice (within W)** — W1/W2/W3/W4 all
   normalise the same firing distribution differently. W4 (PMI) is the
   information-theoretic default; W2 (sublinear power, α≈0.5) is the
   common practical fallback. **Decision deferred to the comparison
   harness output.**
2. **Threshold knob choice (within F)** — `min_count`, `δ_PMI`,
   `δ_logratio`, `K`. **Run F at a 3-point grid each and let the
   harness pick.**
3. **Source-domain confound** — see `REVIEW_ISSUES_20260514.md` in
   `02_2_2_vocab_lang_attribution/analysis/german_review/`. English
   rerun-from-FineWeb-HQ in progress; an audit of other-language
   sources is still TODO.
4. **Small-sample sensitivity for rare tokens** — both F and W are
   unreliable on tokens with `count_L < ~10`. F3's CI handles this
   by design (low-count → wide CI → not promoted). W needs an explicit
   smoothing constant; W4 PMI uses `count + 0.5` add-one already.
   **The orphan tail should be reported, not silently included.**
5. **Aggregate-vs-individual** — for Cyrillic / CJK / Indic, the
   methodology supports either path. Default: emit aggregate AND
   per-language attribution, let consumers pick.
6. **Whether the canonical artifact is F or W** — both ship. The
   comparison harness output tells us which one each phase 3 diagnostic
   loads. The legacy `base_greek_tokens.jsonl` interface stays
   compatible with F's `categories/Greek.jsonl`.

## What this doc obligates us to build

In order of dependency:

1. **F-variant builder** — `scripts/build_filter_categories.py`,
   emits `categories/<L>.jsonl` per variant.
2. **W-variant builder** — `scripts/build_weight_table.py`, emits
   `weights/<L>.parquet`.
3. **Comparison harness** — `scripts/compare_F_vs_W.py`, runs each
   03_1 diagnostic at each variant, emits `comparison/_summary.tsv`.
4. **Per-language ship decision** — manual: read the summary, pick
   the variant per (language, diagnostic).
5. **Doc the per-language choice** — update this file's
   "Per-language regime templates" section with the actual evidence.

Estimated work (rough): ~6 h for #1-#2 (deterministic transforms),
~2 days for #3 (running each variant through each diagnostic, with
bootstrap), ~1 day for #4-#5 (decision + writeup). Total ~4 days.
