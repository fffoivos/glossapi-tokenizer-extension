# Production polytonic cutoff decision — 2026-07-29

## Outcome

Freeze **+512 polytonic merges** as the tokenizer for the next cleaned Greek
CPT dataset:

```text
ship/apertus_greek_modern_polytonic_148992/
vocab: 148,992 = 256 x 582
tokenizer.json sha256:
bbb08e71929b519c5c2362338b0fc6a0e99955cb8fdbf0729ae1311117e6561b
```

The +1,024 arm passed the modern-Greek guard but did not beat +512 on
ancient-Greek BPB after identical adaptation. The old +5,120 / 153,600 bundle
remains a historical specialization artifact and is not the tokenizer for the
next corpus materialization.

No new-dataset tokenization was run as part of this decision.

## Frozen gate and result

| Metric | Modern-148480 | +512 | +1,024 |
| --- | ---: | ---: | ---: |
| Ancient tokens, 311 docs | 221,695 | 204,809 | 198,148 |
| Ancient tokens vs baseline | — | **-7.62%** | -10.62% |
| Ancient single-token word rate | 20.13% | **41.05%** | 43.23% |
| Ancient BPB after bounded adaptation | 0.8533 | **0.9124** | 0.9283 |
| Modern BPB, 1,000 docs | 0.5178 | **0.5185** | 0.5190 |
| Modern BPB ratio to baseline | 1.0000 | **1.00138** | 1.00231 |

Both candidates passed the precommitted maximum 0.5% modern-BPB regression.
+1,024 modeled ancient text 1.74% worse than +512, so the larger cutoff did
not earn its extra 512 rows.

The ancient BPB values are a short adaptation probe, not a claim that the
untrained extension already beats the continued ModernGreek model. The next
CPT run must train the selected rows on the new corpus. The decision here is
that +512 gives a meaningful efficiency/word-recovery gain with the lower
adaptation burden and the better of the two extended-model BPB results.

## Suspicious-token review

All historically flagged IDs inside the selected cutoff were reviewed and
kept:

| ID | Raw bytes | Valid surfaces in the 30M-token scan | Firings | Decision |
| ---: | --- | --- | ---: | --- |
| 148924 | `b7 cf 82` | `ἷς` | 3,568 | keep structural ByteLevel component |
| 148979 | `93 cf 82` | `ὓς` | 3,619 | keep structural ByteLevel component |
| 148987 | `ae cf 87 ce bf cf 82` | `Ἦχος`, `Ὦχος` | 438 | keep structural ByteLevel component |

They are not mojibake and none is unresolved. Each is a partial UTF-8
ByteLevel merge component that fires inside a valid polytonic surface. Removing
one would break a live merge dependency. They stay in the tokenizer, use
merge-chain initialization, and are excluded only from standalone
token-distillation targets.

Machine-readable review:
`production_cutoff_candidates/suspicious_token_review.json`.

## Runtime issue found and fixed

The first model probe failed both modern guards by about 26%, despite almost
unchanged modern token counts. Positive-only token distillation had made the
new output rows overconfident, inflating the expanded softmax denominator.

The corrected pass:

1. kept the full model frozen;
2. alternated disjoint ancient and modern calibration blocks;
3. updated only IDs at or above 148,480;
4. exact-checked every old input and output row;
5. used 2,000 calibration documents with zero eval-text overlap.

After calibration, the +512 modern regression fell from 26.29% to 0.138%.
Both the failed first pass and the corrected selection are retained under
`production_cutoff_candidates/model_probe/`.

## Remote receipts

- Clariden run root:
  `/iopsstor/scratch/cscs/fffoivos/tokenizer_finalization/20260729T094000Z-poly512-1024`
- Asset/coverage job: `2922887`
- Initial packed probe: `2922926` (selection gate intentionally failed)
- Calibration-data job: `2924310`
- Corrected calibrated probe: `2924312`
- Selected probe-only adapted checkpoint:
  `results/calibrated_0512/`

The adapted checkpoint is evidence for tokenizer selection, not the release
model for the next CPT run.
