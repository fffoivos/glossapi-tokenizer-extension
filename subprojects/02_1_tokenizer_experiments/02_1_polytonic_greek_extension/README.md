# 02_1 Polytonic Greek Extension

Parallel tokenizer-extension arm for Ancient/Polytonic Greek.

This project is intentionally separate from the C3 modern-Greek cutoff
pipeline. C3 remains the combined modern-Greek baseline. This arm starts
from a different source-selection premise: Apertus should get a distinct
orthographic lane for polytonic Greek rather than asking the modern-Greek
extension to cover it incidentally.

Current next-step plan: [`ANCIENT_GREEK_AFTER_C3_PLAN.md`](ANCIENT_GREEK_AFTER_C3_PLAN.md).
It pins the approved C3 tokenizer as the base, defines the 5,120-token
Ancient/Polytonic continuation ceiling, lays out the 512-step aligned
cutoff/eval grid, and specifies the report plots expected from the sweep.

## Current Pipeline

1. **Eligible source selection**
   - Include curated ancient/liturgical corpora by provenance:
     First1KGreek, Perseus/classical Greek, and GOARCH liturgical texts.
   - Treat Wikisource Greek and Scholarios as broad mixed collections that
     require orthographic filtering before they enter the arm.

2. **Polytonic filtering for mixed sources**
   - Filter Wikisource and Scholarios by actual distinctive polytonic
     orthography.
   - Plain tonos/oxia is **not** counted as polytonic evidence.
   - Count a Greek word as polytonic only when it contains one of:
     grave/varia, smooth or rough breathing, perispomeni, ypogegrammeni,
     or a Greek Extended codepoint whose NFD form contains one of those
     marks.
   - Current candidate filter:
     `distinctive_polytonic_word_ratio >= 0.50` and
     `distinctive_polytonic_char_ratio >= 0.10`.

3. **Curated-source hygiene**
   - Do not automatically reject unaccented rows from curated ancient
     sources just because the source chose unaccented text.
   - Do reject empty rows, one-letter rows, RTF/control payloads, and
     obvious non-Greek artifacts before training-mix construction.

4. **Deduplication**
   - Feed selected canonical parquets to the existing
     `glossapi_corpus_cli dedup-text run` pipeline with Greek diacritics
     preserved.
   - The dedup run is a source-selection artifact, not yet a tokenizer
     training mix. The next step is to join kept decisions back to text,
     apply hygiene, and produce train/eval splits.

5. **Tokenizer extension**
   - After source cutoff and dedup are frozen, train an Apertus-continuous
     BPE extension arm, build cutoff variants, run fertility/eval, and then
     choose the added-token cutoff.

## Completed Candidate Run

Run id:
`polytonic_strict_w050_c010_20260517T131514Z`

This is the first strict no-tonos/oxia filtering run for the mixed
Wikisource and Scholarios sources. It used:

- `distinctive_polytonic_word_ratio >= 0.50`
- `distinctive_polytonic_char_ratio >= 0.10`
- `greek_percentage >= 50`
- `latin_percentage <= 10`
- Greek diacritics preserved during dedup
- MinHash near-duplicate threshold `0.85`

Selected rows before dedup:

| Source | Selected rows | Input rows | Selection mode |
| --- | ---: | ---: | --- |
| First1KGreek | 1,016 | 1,016 | curated source copied |
| Perseus/classical Greek | 815 | 815 | curated source copied |
| GOARCH liturgical | 675 | 675 | curated source copied |
| Wikisource Greek | 3,435 | 5,394 | strict polytonic filter |
| Scholarios graeca patristic | 13,419 | 14,118 | strict polytonic filter |

Dedup result:

| Source | Kept | Dropped |
| --- | ---: | ---: |
| Scholarios graeca patristic | 12,991 | 428 |
| Wikisource Greek | 3,435 | 0 |
| First1KGreek | 983 | 33 |
| Perseus/classical Greek | 644 | 171 |
| GOARCH liturgical | 673 | 2 |

Overall final decision surface:

- `19,360` dedup decisions
- `18,726` kept rows
- `634` dropped rows
- `18,944` exact-stage survivors
- `223` near-duplicate candidate pairs
- `218` near-duplicate drops

Local training data:

```text
data/strict_w050_c010/20260517T131514Z/
  polytonic_greek_training_kept_strict_w050_c010_20260517T131514Z.parquet
```

That parquet is the post-dedup kept-text corpus: `18,726` rows, about
`250 MiB` compressed, with source fields and dedup metadata columns.

See [`ARTIFACTS.md`](ARTIFACTS.md) for the storage boundary between
subproject-local data and large worker/data-root state.

## Scripts

- `scripts/audit_polytonic_sources.py` — read-only source inventory and
  coarse signal audit.
- `scripts/prepare_polytonic_dedup_inputs.py` — produces canonical input
  parquets for the existing dedup pipeline.
- `scripts/summarize_polytonic_dedup_run.py` — summarizes kept/dropped
  decisions from a dedup run.

## Current Worker

Use the existing tokenizer-extension GCP instance:

```text
apertus-greek-tokenizer-20260408t160000z
zone: europe-west4-b
project: eellak-glossapi-20251008
```

Pause/suspend the instance when the run is verified.
