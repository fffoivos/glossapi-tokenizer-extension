# Native-Greek benchmark contamination decision

Date: 2026-08-12

## Decision

For the completed Apertus 8B CPT run, report two versions of every affected
benchmark:

1. the complete public benchmark score; and
2. a conservative contamination-filtered score that excludes exactly the
   scored example ids in `recommended_excluded_example_ids.jsonl`.

Do not delete or edit benchmark source data. The exclusion file is a reporting
filter for models that have already trained on the corpus. Future corpus builds
can instead use the published document matches to exclude selected training
documents before packing.

The exact exclusion authority is:

- dataset: `fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2`;
- audited dataset revision:
  `987b8955fcd395c6219e39df9e64715457f69065`;
- exclusion-file SHA-256:
  `7a8559461b15a308f599faf0ff25cd16c07be0a597078864f779af7f2f1fdd32`;
- match-table SHA-256:
  `1b23a9dc14a6175c18e0530210cc47795e24e3841bb1b3229c666877ac4b4b19`;
- query SHA-256:
  `ef9d601b8f91c6845818b9584c6634a13337c77b07e3f101f755a4884634c0eb`.

## Exclusions

| Benchmark family | Evaluation units audited | Units with strict evidence | Scored examples excluded | Scored examples retained |
| --- | ---: | ---: | ---: | ---: |
| ASEP MCQA | 1,200 | 20 | 20 | 1,180 |
| DemosQA | 600 | 1 | 1 | 599 |
| GPCR | 208 | 14 | 14 | 194 |
| Medical MCQA | 432 | 13 | 13 | 419 |
| OYXOY metaphor | 3,015 | 973 | 973 | 2,042 |
| OYXOY NLI | 1,762 source pairs | 14 | 42 binary decisions | 5,244 binary decisions |
| OYXOY WIC | 58,831 | 4,614 | 4,614 | 54,217 |
| OYXOY WSD | 14,398 | 4,399 | 4,399 | 9,999 |
| **Total** | **80,446 units** | **10,048** | **10,076 scored examples** | **73,894 scored examples** |

An evaluation unit is one MCQ or lexical item, except for OYXOY NLI, where one
premise/hypothesis source pair becomes three scored binary decisions. All
three decisions are excluded together when their shared source pair has strict
evidence.

## Why these examples are excluded

The reusable audit first requires a match to the question or first source
surface. It recommends exclusion only when the label-bearing second surface
also appears in the same training document and within the configured proximity
window:

- MCQ question plus the correct answer;
- NLI premise plus hypothesis;
- WIC usage one plus usage two;
- WSD usage plus the correct definition;
- metaphor usage plus its source definition.

Question-only hits are published as candidate evidence but are not excluded.
One- and two-token surfaces are unmeasurable and are not treated as evidence.

Manual review used a deterministic 401-row sample spanning every available
benchmark/match-category cell and a compact packet containing every one of the
10,048 strict units. The standard MCQ cases contain the question and answer in
the source text. The OYXOY lexical matches are dominated by direct source
material from `glossAPI/modern-greek-dictionary`: 930 of 973 metaphor units,
4,294 of 4,614 WIC units, and 4,247 of 4,399 WSD units have at least one strict
match there. These are source collisions, not generated prompt scaffolding.

The 14 OYXOY NLI source pairs include some short or generic propositions. Their
two required surfaces nevertheless co-occur within the predeclared proximity
window. Because only 42 scored rows are affected, the conservative filtered
score excludes all 14 groups while the complete public score remains available
for comparison.

## Evidence and reproducibility

The CSCS audit scanned all 431 published Parquet shards and all 51,839,746
rows. It emitted 18,166,197 trace rows with stable dataset shard, zero-based
row, source dataset/document id, 1-based source-line range, match class and a
bounded snippet.

CSCS authorities:

- audit run:
  `/capstor/scratch/cscs/fffoivos/benchmark_contamination_audits/runs/20260812T171530Z-native-greek-v1`;
- full match table:
  `publish_payload/qa_document_line_matches.parquet`;
- exact exclusion list:
  `publish_payload/recommended_excluded_example_ids.jsonl`;
- deterministic review sample:
  `review_sample_v1.jsonl`, SHA-256
  `e3c832319ef742bd48f15996686f1dafd38b6c69414327ff19ec01530ffd15c4`;
- all-strict-unit adjudication packet:
  `adjudication_packet_v1.jsonl`, SHA-256
  `28db01d125e9c8e12a646019a89c0ba39254695bb3e83010c034b2fc38f805c1`;
- filtered evaluation receipt:
  `filtered_scores_v1/receipt.json`.

The three scored checkpoints are initialization (`iter_0000000`), approximately
39.997B tokens (`iter_0009536`), and the final checkpoint at approximately
76.689B tokens (`iter_0018284`).

## Protipa boundary

Greek Protipa Exams is not silently omitted from the intended broader suite.
Its repository metadata is visible at revision
`8ad3757de70b13f9e2dd8cdc74eff35ad3268895`, but an authenticated file download
on CSCS returned HTTP 403 on 2026-08-12 because access is still awaiting manual
approval. No Protipa score or contamination claim is made until its actual
Parquet rows can be frozen and audited.
