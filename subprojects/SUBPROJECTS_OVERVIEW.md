# Subprojects Overview

One-paragraph summary of every sub-subproject under
`subprojects/` of the glossapi-tokenizer-extension repo. Generated
2026-05-18. Sorted by pipeline order within each parent subproject.

The parent subprojects themselves are:

- **`02_apertus_tokenizer_spec/`** — pins the exact Apertus-8B-2509
  tokenizer behaviour that any extension must reproduce (BPE, 131,072
  base vocab, fixed front block of first 1,000 ids, no normalizer,
  regex split + ByteLevel pre-tokeniser, `add_bos_token=true`,
  `add_prefix_space=false`, `model.ignore_merges=true`,
  `tie_word_embeddings=false`). No sub-subprojects — it is a single
  spec doc.
- **`02_1_tokenizer_experiments/`** — the live tokenizer-extension
  research arm. The C3 arm
  (`C3_wave2_broad_glossapi_plus_hplt_50_50`) is the chosen
  continuous-BPE arm; the chosen cutoff is **17,408 added units,
  curated + backfilled, total vocab 148,480**. Eight sub-subprojects
  (see below) form the training → cutoff-variant → fertility →
  cutoff-pick → curation pipeline, plus the policy-archaeology
  archive, the intrinsic-eval sweep that actually fixed the cutoff,
  and the parallel polytonic arm.
- **`02_2_tokenizer_implementation/`** — implements the
  Apertus-compatible merge-rule extension and the supporting
  per-token language-attribution machinery. Four sub-subprojects:
  strict char masks, empirical firing histograms, dataset-anchored
  tier classification, and language-category promotion for the
  embedding diagnostic.
- **`03_apertus_extension_and_embedding_adaptation/`** — model-side
  adaptation and historical bakeoff workspace: embedding resize,
  Token Distillation init, vanilla/retok/centroid comparisons, and
  reusable Clariden train/eval primitives. It is now mostly provenance
  plus shared low-level scripts, not the active Task-2 planning home.
- **`04_cpt_training_regime_on_vanilla/`** — completed Task-1 Vanilla
  CPT regime diagnostic. Owns the 5B Vanilla result, reports, run log,
  and vanilla-side scripts. It is the baseline/provenance package, not
  the active TD experiment home.
- **`05_token_distillation_cpt/`** — active Task-2 workspace. Current
  2026-06-02 scope is the CPU-only 5B-pool benchmark-decontamination
  pilot. TD layer tests and TD launch wrappers are parked planning
  material until the TD track is explicitly resumed.

---

## `02_1_tokenizer_experiments/`

### `02_1_1_tokenizer_training/`

Stage 1 of the tokenizer-experiments pipeline. Trains a continuous-BPE
tokenizer arm starting from the Apertus base snapshot on a chosen
training mix, while preserving Apertus front-end behaviour exactly
(special tokens, regex split, ByteLevel, first 1,000 ids). The
trainer runs in six phases — identity check, count segments, build
sequence shards, aggregate sequences, merge loop, write tokenizer —
and emits `tokenizer.json` at the target max vocab plus
`replication_check.json` / `front_end_contract_check.json`. The same
scripts produced the historical fresh-discovery arms (F1, F2,
archived) and remain reusable for future arms. Original execution
plan lives in `CONTINUOUS_BPE_EXTENSION_PLAN.md` (sections on cutoff
grid, mergeback, evaluation, acceptance still apply; multi-arm
sections are archived in-place).

### `02_1_2_cutoff_variant_builder/`

Stage 2. Given a fully-trained continuous-BPE arm at maximum target
vocab, derives Apertus-compatible **merged tokenizer variants** at
each requested cutoff `N` by truncating the added-vocab and
added-merges prefixes. Because continuous BPE preserves Apertus ids
`0..131,071` and appends new ids in merge order, every prefix is a
valid Apertus-compatible tokenizer. Each output dir
`<arm_prefix>_added_<N>/` carries `tokenizer.json`,
`tokenizer_config.json`, `special_tokens_map.json` (the latter two
byte-identical to the full arm so contract checks transfer); runtime
is seconds per cutoff. Drives the C3 sweep across 25 cutoffs at a
1,024 step.

### `02_1_3_fertility_evaluation/`

Stage 3. Measures intrinsic + fertility metrics
(`chars_per_token`, `tokens_per_byte`, `greek_word_space_fertility`,
`single_token_greek_word_share`, `added_token_rate`, added-token
utilization, unk / byte-fallback rates) for every (cutoff variant,
held-out slice) pair. Also hosts the held-out-cleaner helpers
(`clean_holdouts.py`, `build_virgin_hplt_eval.py`) needed because the
default C3 splitter partitions by row rather than by doc, leaking
~0.4 – 0.5 % of text-md5s between train and val/test; the C3 sweep
therefore evaluated on three verified-clean slices
(`virgin_hplt`, `C3_val_clean`, `C3_test_clean`). Runtime for the C3
sweep (26 tokenizers × 3 slices = 78 rows) was ~3 minutes on the
gcloud worker.

### `02_1_4_cutoff_analysis/`

Stage 4. Combines three independent evidence streams into a cutoff
recommendation: (1) comparable-language vocab sizes from
`02_2_2_vocab_lang_attribution`, (2) held-out fertility from
`02_1_3`, and (3) token-content composition (Gemini-pass glossary
categorisation + `02_2_1` char-language masks) to assign each added
token a function label (`GREEK`, `USEFUL_STRUCTURAL`, `NOISE`,
`AMBIGUOUS`). Headline first-draft recommendation was 11,264 added
units (vocab 142,336); this was an analytic anchor, later superseded
empirically by `02_1_7` at 17,408. Outputs `REPORT.md` plus per-cutoff
`distribution_at_<N>.json`, `classified_added_tokens.jsonl`, and the
plot suite the report cites.

### `02_1_5_added_token_curation/`

Stage 5, post-cutoff. Decides per-token which kept added units
should be **removed** because they are extraction / encoding
artefacts rather than content the model should learn. Six removal
classes: Latin-1-as-UTF-8 mojibake, mixed-script artefacts,
PostScript-glyph names, cleaner LINENEWLINE placeholders, their BPE
fragments, and `-missing` / `-decoded` cleaner extraction tags. The
rule set was originally authored against the 11,264 analytic anchor
from `02_1_4` (39 in-cutoff removals at that anchor; 104 at the full
25,600 vocab). The final ship cutoff is **17,408**, with **69
in-cutoff removals** — and they are not consumed at runtime: the
canonical ship tokenizer
(`02_1_7/variants/c3_added_17408_curated_padded/tokenizer.json`)
**structurally skips** the 69 ids during merge selection and
backfills with the next valid C3 merges, so the noise tokens are
not in the vocab at all (see
`02_1_7/CHOSEN_CUTOFF.md` § "Why the backfilled tokenizer is the
canonical ship"). This sub-subproject therefore emits the
machine-readable manifest only (`manifests/removal_list.jsonl`,
`manifests/decision_summary.json`); `02_1_7` consumes the manifest at
build time, and `02_2` then consumes the resulting tokenizer
verbatim. Detailed reasoning + keep justifications in
`CURATION_REPORT.md`.

### `02_1_6_representation_policy_analysis/`

Policy-archaeology investigation into Apertus's implicit
per-language representation policy — what rules actually produced
each language's vocab + pretraining-data share, separated into
*necessary* core and *accidental* periphery, with the goal of
deriving a principled Greek vocab budget. **Now in archive mode**
(2026-05-18): the cutoff decision was settled empirically downstream
in `02_1_7`, not by policy reasoning here. Net contributions: (1)
the provenance finding that Apertus inherited Mistral's BPE table
verbatim except for special-token-block changes + 486 trailing-BPE
truncations, reframing the question as Mistral-side, (2)
invalidation of the speaker-count hypothesis, (3) discovery of the
`swiss-ai/tokenizer-intrinsic-evals` (TokEval) suite that seeded
`02_1_7`, and (4) HPLT 3.0 / FLORES+ / classical-language reference
evidence. The Phase 3–4 synthesis and six unrun hypothesis stubs are
archived under `_deprecated_20260518/`.

### `02_1_7_intrinsic_eval_sweep/`

The cutoff-decision stage that actually shipped: applies the
swiss-ai TokEval intrinsic-evaluation suite (the same one Apertus
itself used) to a 1k-spaced grid of C3 cutoff variants (0 → 25,600
added tokens). **Decision frozen 2026-05-18 at 17,408 added units,
curated + backfilled, total vocab 148,480** (vocab `= 128 * 1160
= 256 * 580`, Apertus base ids `0..131,071` preserved verbatim, 69
in-cutoff noise tokens structurally skipped and backfilled with the
next valid C3 merges). Decision contract pinned in `CHOSEN_CUTOFF.md`,
evidence in `REPORT.md`. Post-decision, the canonical tokenizer was
run over the full C3 BPE training corpus (14.4 M rows / 99.26 B
chars / 24.89 B tokens) to count token firings by component and
source — see `FIRING_COUNT_RUN_20260518.md`,
`FIRING_COUNT_README.md`, and
`manifests/firing_count_20260518_run_summary_augmented.json` for
the completed-run provenance. Headlines: GlossAPI-nanochat = 49.79 %
of token mass, HPLT = 50.21 %; 0 zero-firings in the combined
corpus; 27 zero-firings in HPLT-only.

### `02_1_polytonic_greek_extension/`

Parallel **Ancient/Polytonic Greek** tokenizer-extension arm,
intentionally separate from the C3 modern-Greek pipeline because
polytonic Greek deserves its own orthographic lane. Selects
curated ancient/liturgical sources (First1KGreek, Perseus,
GOARCH liturgical) and filters mixed-collection sources
(Wikisource, Scholarios) by distinctive polytonic orthography
(grave/varia, breathings, perispomeni, ypogegrammeni — plain
tonos/oxia does **not** count). Strict candidate filter is
`distinctive_polytonic_word_ratio >= 0.50` AND
`distinctive_polytonic_char_ratio >= 0.10`. The current candidate
arm `c3p_poly_added_5120` continues C3 with +5,120 ancient/polytonic
tokens (final vocab 153,600 = 256 × 600) and improves balanced
polytonic-validation Greek-word fertility from 3.00 → 1.96 while
the modern-C3 polytonic-id firing on modern eval stays at 0.31 %.
**Local / not yet versioned**: as of 2026-05-18 the README and
`ARTIFACTS.md` updates plus the
`analysis/c3p_polytonic_20260518T_impl/` implementation bundle and
`scripts/render_polytonic_full_report.py` are still uncommitted; the
candidate-arm headline numbers above are reproduced from the local
tree, not from the committed repo.

---

## `02_2_tokenizer_implementation/`

### `02_2_1_char_language_membership/`

Strict-rule, source-authoritative char-level admissibility masks at
three resolutions — script, language-family, language — derived
from CLDR exemplars + Unicode-script closures with no dataset
signal. Emits `char_language_bitmask.parquet` (one row per
codepoint) plus `token_language_bitmask.parquet` (one row per
Apertus vocab token with AND / OR aggregations across the token's
chars). Current scope is 88 (language, script, encoding) triples,
47 families, 29 scripts (v3.3.x schema v5; consumers must read the
live bit counts from the manifest). Built **for rejection**: every
triple not in a token's mask is excluded with confidence; the
artifact never assigns a token to a single language. Apertus's
documented pretrain mix stands in as a proxy for Mistral-Nemo's
unpublished tokenizer-training language list. Reference layer
everything downstream joins against.

### `02_2_2_vocab_lang_attribution/`

Empirical per-token firing histograms across **1,934 canonical
(language, script) keys** drawn from FineWeb-2, FineWeb-2-HQ,
Wikipedia, EuroParl, ParaDocs, FineWeb-Edu, FineWeb-HQ, and
DCLM-Edu, with ~1 B Apertus tokens consumed per key. Per-key
documents are tokenised with `add_special_tokens=False`, token ids
counted via `np.bincount`, and aggregated into a `1,934 × 131,072`
histogram_matrix; post-hoc derivations include `P(L|t)`,
`primary[t] = argmax_L`, entropy, top-K signature, and confidence
flags. Empirical observation layer; no char-tool dependency. Final
artifacts at `outputs/histogram_matrix.npz` (58.4 MB compressed),
`token_metadata.parquet`, `lang_metadata.json`. Run completed
2026-05-13, 8 workers, 113.4 B tokens total. Downstream analyses
(German review, English review, Greek review, script/family
composition, membership rejection, PMI promotion) live under
`analysis/`.

### `02_2_3_token_classification/`

Proposed but not yet implemented. Joins the strict char masks
(`02_2_1`) with the empirical firing histograms (`02_2_2`) and a
defeasible *dataset-language premise* to emit per-(token, dataset)
**tiered labels**: `T0 char-evidenced`, `T1 family-evidenced`,
`T2 premise`, `T3 substrate`, `T4 excluded`, `T5 unknown-standalone`,
each carrying a `basis` string explaining which bits / closures
triggered. The output is one Parquet artifact keyed by
(token_id, dataset); cross-dataset reconciliation, sister-language
firing-rate ratios, n-gram signals, and threshold-based filtering
are explicitly out of scope (consumers slice the artifact
themselves). Tiers are currently computed inline by
`02_2_2/analysis/german_review/tiered_attribution.py` pending
canonical implementation. Spec at `PLAN.md`.

### `02_2_4_language_category_promotion/`

Proposed but not yet fully landed (currently implemented via the PMI
promotion analysis under
`02_2_2/analysis/main_token_sets_pmi/`). Produces, per language
category, a defensible **set of token ids** the embedding diagnostic
(`03_1_greek_embedding_diagnostic/`) can drop in as
`categories/<L>.jsonl`, replacing the legacy
`base_greek_tokens.jsonl` (1,494 strict-Greek ids) interface with a
uniform schema. Three category layers: single-language (English,
German, Greek, French, Russian, Japanese, Hindi…), aggregate
(Cyrillic, Germanic-Latn, CJK…), and structural (Substrate,
ByteFragment, SpecialToken). Residual `Unattributable` exists for
traceability only. Spec at `PLAN.md` + `PMI_PROMOTION_SPEC.md`,
methodology at `METHODOLOGY.md`.

---

## `03_apertus_extension_and_embedding_adaptation/`

### `03_1_greek_embedding_diagnostic/`

Pre-extension diagnostic characterising how Apertus-8B-2509
represents languages on its input (E) and output (U) embedding
matrices — centroid geometry, MP-edge spectrum, K_significant,
within-group hulls + infiltrators, morphological clustering, binary
Greek-vs-¬Greek logistic classifier, and cross-language semantic
clusters. Runs entirely on the base model's existing E / U rows; no
new-token init, no CPT, no LOO. Three pipeline iterations: **v2**
(single-anchor Greek vs ¬Greek), **v3** (11 PMI-attributed languages,
v3-corrected 2026-05-15 fixes an MP-edge formula bug and a
truncated-covariance bug), and **v4** (all 75 well-sampled
PMI-attributed languages; steps 1–3 only — pair-specific shared
subspace + L-discriminant directions deliberately not scaled to 75).
Headlines (v3-corrected, 11 langs): Greek centroid displaced
0.676 (E) / 0.733 (U) from classified-global; K_significant
123 / 83 on E / U; tightest pair-specific direction with Thai (2.23);
tokens cluster by morphology (`μέν*`, `ματ*`, `συν-` families
5–9 × tighter than random); en↔el cosine for Greek-origin concepts
(*democracy*, *philosophy*, *mathematics*) averages +0.05,
indistinguishable from non-Greek-origin baselines (+0.04).

---

## Pipeline shape

```
02_apertus_tokenizer_spec
        (spec — frozen)
            │
            ▼
02_1_tokenizer_experiments
  02_1_1 training  →  02_1_2 cutoff variants  →  02_1_3 fertility  →
  02_1_4 analysis  →  02_1_5 curation (manifest only)
                                          │
                                          ▼
  02_1_7 intrinsic eval sweep  →  chosen cutoff 17,408
       └─ builder consumes 02_1_5 manifest, skips 69 ids, backfills →
          variants/c3_added_17408_curated_padded/tokenizer.json
          + firing counts
  02_1_6 representation policy (archived; seeded 02_1_7)
  02_1_polytonic_greek_extension (parallel arm; ancient/polytonic)
            │
            ▼
02_2_tokenizer_implementation
  consumes the curated+backfilled tokenizer verbatim — no runtime mask
  02_2_1 char masks       ─┐
  02_2_2 firing histograms ─┤→ 02_2_3 tier classification (proposed)
                            │→ 02_2_4 category promotion (proposed)
                            │
                            ▼
03_apertus_extension_and_embedding_adaptation
  03_1 embedding diagnostic (pre-extension; v2 / v3-corrected / v4)
        │
        ▼
  embedding extension + CPT (not yet started; next after tokenizer
  handoff — cutoff is frozen at 17,408)
```
