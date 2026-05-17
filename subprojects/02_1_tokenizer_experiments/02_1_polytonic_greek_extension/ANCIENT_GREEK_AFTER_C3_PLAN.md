# Ancient/Polytonic Greek Extension After C3

Status: planning document.
Date: 2026-05-18.
Execution target: the tokenizer-extension GCP worker, not local `home`.

## 1. Decision Context

The modern-Greek C3 tokenizer cutoff is now fixed. The base artifact for
this arm is:

```text
../02_1_7_intrinsic_eval_sweep/variants/c3_added_17408_curated_padded/tokenizer.json
sha256: 358ae3f29ac17c99769d6d437339e28657d5fcaed3486f8550feed3d6adfc394
vocab: 148,480 = 128 x 1160 = 256 x 580
```

Contract inherited from C3:

- Apertus ids `0..131,071` are unchanged verbatim.
- C3 modern-Greek ids `131,072..148,479` are fixed.
- Ancient/polytonic Greek must be appended after C3, starting at id
  `148,480`.
- The final tokenizer must remain divisible by 256.

The C3 handoff manifest confirms:

- `17,408` accepted added merges.
- `69` curated noise tokens skipped at build time.
- `69` valid merges backfilled from the next C3 merge positions.
- `0` cascade skips.
- final vocab `148,480`.

Small process improvement for the Ancient/Polytonic arm: every final
manifest should include the tokenizer SHA-256 directly, not only the
path and structural counts.

## 2. Rationale

The earlier attribution work shows that Apertus already has many
Greek-script tokens, but it does not have a meaningful inventory of
distinctive polytonic tokens.

Important evidence:

- The modern-Greek review found 1,507 Greek-codepoint vocab entries,
  but `greek_script_poly_any = 0`.
- The Ancient Greek `grc_Grek` attribution run used 28,850 FineWeb-2
  documents and 128,396,354 sampled tokens.
- That run fired many base Greek fragments, but direct inspection found
  no token types containing distinctive polytonic codepoints or marks.
- The `grc_Grek` row had a very large unknown/byte-fragment component:
  about 35.88% of the sampled mass.

Interpretation:

- Existing Apertus/C3 tokens can support shared Greek substrate:
  alphabetic fragments, modern-compatible stems, punctuation, digits,
  and some unaccented ancient forms.
- True polytonic orthography is still poorly represented.
- Adding Ancient/Polytonic Greek as a layer after the now-fixed C3
  modern-Greek tokenizer is the cleanest path: it preserves the modern
  Greek decision while adding a distinct orthographic lane.

This arm should therefore optimize for:

1. Lower fertility on real Ancient/Polytonic Greek.
2. Lower byte-fallback and replacement-character pressure on polytonic
   text.
3. High utilization of newly added Ancient/polytonic ids.
4. No meaningful regression to modern Greek C3 or Apertus-55
   multilingual fairness.

## 3. Target Size And Cutoff Grid

Train a full Ancient/Polytonic continuation up to 5,120 new tokens.

```text
base after C3:       148,480
max polytonic added:   5,120
max final vocab:     153,600 = 256 x 600
```

Evaluate every 512 additions:

| polytonic added | final vocab | 256-aligned |
|---:|---:|:---:|
| 0 | 148,480 | yes |
| 512 | 148,992 | yes |
| 1,024 | 149,504 | yes |
| 1,536 | 150,016 | yes |
| 2,048 | 150,528 | yes |
| 2,560 | 151,040 | yes |
| 3,072 | 151,552 | yes |
| 3,584 | 152,064 | yes |
| 4,096 | 152,576 | yes |
| 4,608 | 153,088 | yes |
| 5,120 | 153,600 | yes |

The 512-step grid is close to the requested 500-token granularity while
preserving alignment at every candidate cutoff. The final selected
cutoff should be one of these grid points unless there is a strong
reason to add a narrower aligned follow-up sweep.

## 4. Training Corpus

Current local usable corpus:

```text
data/strict_w050_c010/20260517T131514Z/
  polytonic_greek_training_kept_strict_w050_c010_20260517T131514Z.parquet

rows: 18,726
compressed size: about 250 MiB
sha256: 2b89e098de95501734446b5f767205286eb709ac58c6fb7d2eca6ceb2d873001
```

Source composition after strict filtering and dedup:

| source | kept rows |
|---|---:|
| Scholarios graeca patristic | 12,991 |
| Wikisource Greek | 3,435 |
| First1KGreek | 983 |
| GOARCH liturgical | 673 |
| Perseus/classical Greek | 644 |

Training policy:

- Use the post-dedup kept-text parquet as the canonical source.
- Preserve Greek diacritics and combining marks.
- Do not count plain tonos/oxia as polytonic evidence.
- Do not reject curated ancient-source rows merely because they are
  under-accented.
- Do reject empty rows, one-letter rows, RTF/control payloads, and
  obvious extraction artifacts before split construction.
- Cap or weight Scholarios during training so it does not dominate all
  merge decisions merely because it is the largest source.

## 5. Held-Out Design

The eval suite needs a diverse held-out set before training starts. The
split should be source-stratified and deterministic.

Recommended splits:

| slice | purpose |
|---|---|
| `poly_train` | training rows only |
| `poly_val_balanced` | source-balanced validation for cutoff selection |
| `poly_test_balanced` | final source-balanced held-out test |
| `poly_first1k_test` | classical/public-domain prose regression |
| `poly_perseus_test` | classical corpus regression |
| `poly_wikisource_test` | web/Wikisource formatting and document-shape regression |
| `poly_scholarios_test` | patristic/church-heavy regression |
| `poly_goarch_test` | liturgical-domain regression |
| `poly_high_diacritic_test` | high distinctive-polytonic density |
| `poly_underaccented_test` | ancient Greek with low or absent polytonic density |
| `fineweb2_grc_reference` | out-of-training check against FineWeb-2 Ancient Greek |
| `modern_c3_val_clean` | modern Greek regression guard |
| `modern_c3_test_clean` | final modern Greek regression guard |
| `flores_ell_Grek` | paper-comparable Greek check |
| `apertus55_flores` | multilingual fairness guard |

Hold-out integrity requirements:

- Split before tokenizer training.
- Deduplicate train vs held-out by normalized exact text hash.
- Also check fragment containment where FineWeb/Wikisource sources may
  split documents differently.
- Store split manifests with row counts, source counts, text-char
  totals, token estimates, hash policy, and random seed.

## 6. Training Procedure

Run on:

```text
apertus-greek-tokenizer-20260408t160000z
zone: europe-west4-b
project: eellak-glossapi-20251008
```

Use the continuous-BPE trainer from `02_1_1_tokenizer_training`, but
point it at the C3 ship artifact as the base tokenizer directory.

Training target:

```text
target_vocab_size = 153600
base_vocab_size   = 148480
added_poly_tokens = 5120
```

Required checks:

- C3 base ids `0..148,479` preserved verbatim.
- Apertus front-end behavior remains unchanged for ids `0..131,071`.
- Special tokens unchanged.
- ByteLevel/pre-tokenizer settings unchanged.
- No normalization step strips polytonic marks.
- Training summary records source rows, source text chars, effective
  source weights, runtime, target vocab, final vocab, and SHA-256.

Suggested run name:

```text
c3p_polytonic_added_5120_<YYYYMMDD>
```

## 7. Variant Construction

From the full 5,120-token Ancient/Polytonic continuation, build cutoff
variants at the 512-token grid.

Suggested variant ids:

```text
c3p_poly_added_0000
c3p_poly_added_0512
c3p_poly_added_1024
...
c3p_poly_added_5120
```

Each variant should have:

- `tokenizer.json`
- `manifest.json`
- `sha256`
- `base_tokenizer_sha256`
- `c3_base_vocab_size = 148480`
- `polytonic_added_count`
- `final_vocab_size`
- `alignment_256 = true`
- front-end/base-contract check results

## 8. Curation And Backfill

Ancient/Polytonic curation should mirror the C3 structural curation
pattern: bad tokens are skipped during variant build, then valid later
merges are backfilled so alignment and append-only ids are preserved.

Expected curation classes:

| class | action |
|---|---|
| distinctive polytonic Greek | keep |
| ancient-skewed unaccented Greek | keep or review |
| liturgical/classical formulae | keep |
| modern-only Greek already covered by C3 | review, usually do not prioritize |
| byte-fragment or replacement-char artifacts | remove/backfill |
| mojibake | remove/backfill |
| RTF/control residues | remove/backfill |
| PDF/PostScript glyph residue | remove/backfill |
| mixed-script artifacts | remove/backfill |
| line/newline cleaner placeholders | remove/backfill |

The final chosen tokenizer should be:

```text
c3p_poly_added_<N>_curated_padded/tokenizer.json
```

where `N` is the chosen Ancient/Polytonic added-token count and
`148480 + N` is divisible by 256.

## 9. Evaluation Suite

Run the same evaluation family used by `02_1_7_intrinsic_eval_sweep`,
plus polytonic-specific diagnostics.

Mandatory metrics for every 512-step variant:

| metric | scope | purpose |
|---|---|---|
| Greek word fertility | polytonic held-outs, modern Greek held-outs, FLORES Greek | primary compression/fertility signal |
| chars per token / bytes per token | all held-outs | compression view |
| added-token utilization | all Greek/polytonic held-outs | detects wasted added vocab |
| byte fallback rate | all polytonic held-outs | direct pain-point metric |
| replacement-char token rate | all polytonic held-outs | detects bad decoding/artifact pressure |
| distinctive-polytonic word fertility | high-polytonic slices | measures the target orthographic lane |
| combining-mark integrity | high-polytonic and mixed-normalization slices | ensures marks are not split pathologically |
| UTF-8 integrity | all text slices | regression guard |
| Renyi-2.5 efficiency | all main slices | information-theoretic supplement |
| MorphScore Greek | modern Greek | modern-Greek morphology regression guard |
| TokEval TFG | Apertus-55 | multilingual fairness guard |

Evaluation axes:

- `polytonic_core`: balanced polytonic validation and test.
- `polytonic_by_source`: each source separately.
- `polytonic_by_orthography`: high-diacritic vs under-accented.
- `modern_regression`: C3 clean val/test and FLORES Greek.
- `multilingual_regression`: Apertus-55 FLORES TFG and per-language
  fertility deltas.

## 10. Plot Plan

The eval report should include polished plots, not only tables.

### 10.1 Cutoff Selection Dashboard

One multi-panel figure with cutoff on the x-axis:

- polytonic balanced fertility improvement vs C3 base
- byte-fallback reduction vs C3 base
- added-token utilization
- modern-Greek fertility regression
- Apertus-55 TFG delta

Purpose: one figure should make the cutoff tradeoff legible.

### 10.2 Knee Analysis

Line plot:

- y-axis: percent of maximum observed polytonic fertility improvement
- x-axis: added polytonic tokens
- annotate first point where the next 512 tokens buy less than a chosen
  marginal improvement threshold

Use the same visual logic as the C3 knee plot, but with 512-token
steps.

### 10.3 Source-Stratified Fertility

Small-multiple line plots, one per source:

- First1KGreek
- Perseus/classical
- Wikisource
- Scholarios
- GOARCH
- FineWeb-2 `grc_Grek` reference

Purpose: show whether one source dominates the apparent gain.

### 10.4 Orthography-Stratified Fertility

Line plot comparing:

- high distinctive-polytonic density
- medium polytonic density
- under-accented ancient Greek
- modern Greek regression

Purpose: verify that gains are specifically strongest where polytonic
orthography exists, while under-accented ancient Greek is not harmed.

### 10.5 Byte Fallback And Replacement-Char Collapse

Two-panel plot:

- byte fallback rate by cutoff
- replacement-char token rate by cutoff

Purpose: directly visualize the problem this arm is meant to solve.

### 10.6 Added-Token Utilization Heatmap

Heatmap:

- rows: held-out slices
- columns: cutoff variants
- cell: used added tokens / available added tokens

Purpose: identify wasted high cutoffs and source-specific overfitting.

### 10.7 Top Added Tokens By Cutoff Band

Stacked or faceted bar chart:

- cutoff bands: `1..512`, `513..1024`, ..., `4609..5120`
- token classes: distinctive polytonic, ancient unaccented,
  modern-compatible Greek, structural, artifact/review

Purpose: show what kinds of units each added band contributes.

### 10.8 Curation Waterfall

For the chosen cutoff:

- raw added tokens
- removed artifacts by class
- cascade skips
- backfilled valid merges
- final curated-padded vocab

Purpose: make curation auditable and comparable to C3.

### 10.9 Regression Plot

Bar or dot plot:

- modern C3 val/test fertility delta
- FLORES Greek fertility delta
- Apertus-55 TFG delta
- worst per-language fertility regression across Apertus-55

Purpose: show that Ancient/Polytonic specialization did not damage the
already-approved C3 tokenizer.

## 11. Selection Criteria

Choose the final Ancient/Polytonic cutoff when:

1. The balanced polytonic held-out fertility curve reaches its knee.
2. Byte fallback and replacement-char rates have materially fallen.
3. Added-token utilization remains healthy.
4. Source-stratified plots show broad gains, not only Scholarios gains.
5. Modern Greek C3 regression is negligible.
6. Apertus-55 TFG and worst-language regression are acceptable.
7. The final vocab is divisible by 256.
8. The final tokenizer is structurally curated and backfilled, if
   curation removes anything.

Expected final artifact pattern:

```text
variants/c3p_poly_added_<N>_curated_padded/tokenizer.json
```

with:

```text
final vocab = 148480 + N
N in {0, 512, 1024, ..., 5120}
final vocab % 256 == 0
```

## 12. Operational Rules

- Do not run tokenizer training, large eval sweeps, dedup, or bulk data
  transforms locally.
- Resume the tokenizer-extension GCP instance only when the command set
  is ready.
- Run training/eval in a durable session on the worker.
- Monitor until the process is confirmed healthy or complete.
- Pull back only manifests, reports, plots, and compact result tables.
- Suspend the tokenizer-extension instance after completion.
- Leave unrelated GCP instances untouched.

## 13. Deliverables

Minimum deliverables:

- split manifest for train/val/test/source/orthography held-outs
- full 5,120-token Ancient/Polytonic tokenizer
- 512-step cutoff variants
- per-variant structural manifests with SHA-256
- eval result parquet/json/csv across all slices
- plot bundle under `artifacts/plots/`
- final cutoff report
- final curated-padded tokenizer if curation is needed
- README update that points to the chosen final artifact
