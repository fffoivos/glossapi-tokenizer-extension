# Bibliography next-generation reviewer fixes — development run (2026-07-22)

## Result

The reviewer-derived corrections were implemented and evaluated only on the
1,118-document grouped development OOF corpus. The best corrected strict-gate
candidate is the direct linear scope model:

| Candidate | Line precision | Line recall | Character precision | Character recall | Spurious blocks / zero-BIB doc |
|---|---:|---:|---:|---:|---:|
| Previous strict selected scope | 0.997449 | 0.789067 | 0.997249 | 0.842593 | 0.000000 |
| Corrected direct linear scope | 0.996135 | **0.812486** | 0.995728 | **0.863567** | 0.015038 |

This is a development improvement of +2.34 percentage points line recall and
+2.10 points character recall, with a 0.13-point line-precision reduction. It
still clears the configured 0.98 line/character precision gates and the 0.02
spurious-block gate.

The high-recall proposal pool increased scope negatives from 152 to 485, but
did not improve the operating point. Its best strict candidate was histogram:
line P/R 0.996102/0.748849. The pooled variants are retained as negative
results, not promoted.

## Implemented corrections

- Gated bibliography heading emission, with a two-line attachment window and
  exact bibliography-heading lexicon guard.
- Subheaders may be emitted only when sandwiched inside a component; the final
  guard also requires the bibliography lexicon.
- Conditioned long-line bridging/expansion: long lines require year, pages,
  DOI, URL, or numbered-entry evidence.
- Scope features now include image markers, separator/rule lines, and exact
  bibliography-heading lexicon signals.
- Heading-supported components can be rescued above a configurable component
  probability floor.
- Image-heavy and rule-heavy components receive the reviewer-proposed hard
  vetoes.
- Scope targets now require at least 50% gold BIB purity; mixed components
  below that threshold are excluded from fitting.
- Optional scope training pool from eight deliberately loose decoders.

## Heading-label finding

The probability-only heading rule admitted 29 non-BIB Markdown headings on
development. Twenty-eight were false subheader-model predictions such as CV
and chapter headings; the exact lexicon guard removes them. The sole remaining
crossing is `## ΒΙΒΛΙΟΓΡΑΦΙΑ` in document `abb102c19a0f...`, which the
development silver data labels `O`. This is a label-policy disagreement, not a
false heading. The final gated-heading decoder therefore reports one crossing;
the strict selected model retains the no-heading policy until that label policy
is reconciled.

## Local code

- `sequence_models/bibliography_nextgen_table.py`
- `sequence_models/bibliography_nextgen_decode.py`
- `sequence_models/bibliography_nextgen_scope.py`
- `sequence_models/clariden/train_bibliography_nextgen_scope.sbatch`
- `sequence_models/tests/test_bibliography_nextgen.py`

All 15 targeted bibliography-nextgen tests pass locally, including the new
heading, conditioned-expansion, component-purity, rescue, and structural-veto
cases.

## Clariden artifacts

Root:

`/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/experiments/bib_nextgen_devfix_20260722`

- Corrected feature table: `full_table_v2/`
- Final decoder sweep: `decode_v3_lexicon_subheaders/` (job 2863808)
- Best corrected strict scope candidate: `scope_linear_direct_v3/` (job 2863694)
- Direct histogram ablation: `scope_hist_direct_v3/` (job 2863695)
- Pooled linear negative result: `scope_linear_pool_v3/` (job 2863580)
- Pooled histogram negative result: `scope_hist_pool_v3/` (job 2863581)
- Final immutable code bundle:
  `/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/code_bundles/bib_nextgen_devfix_aa8b477/`

The final decoder's deployment-near-miss prediction is byte-identical to the
decoder prediction consumed by `scope_linear_direct_v3` (SHA-256
`c392b12d6f74fd06d9761ac65ba162fb2513a1ad3862a102c8f477295154dfc4`).

## Test-set status

At the corpus owner's request, the corrected direct-linear scope model was
subsequently evaluated on the existing 143-document test cohort:

| Candidate | Line precision | Line recall | Character precision | Character recall | F1 |
|---|---:|---:|---:|---:|---:|
| Frozen component-scope baseline | 0.968017 | 0.917121 | 0.976257 | 0.935004 | 0.941882 |
| Post-review corrected | 0.967047 | **0.926362** | 0.973123 | **0.946300** | **0.946268** |

The corrected model gains 0.92 points line recall and 1.13 points character
recall, while losing 0.10 points line precision and 0.31 points character
precision. F0.5, F1, and F2 all improve. The evaluation is at
`postreview_opened_test_v1/` (job 2864431).

This is explicitly a post-review opened-test result, not an unbiased sealed
result: the reviewer derived the corrections from this cohort. A fresh sealed
cohort remains required for an unbiased final comparison.
