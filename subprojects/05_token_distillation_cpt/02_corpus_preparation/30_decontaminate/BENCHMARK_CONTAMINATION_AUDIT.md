# Reusable benchmark-contamination audit

## Decision

The historical `decontaminate.py` remains the calibrated GreekMMLU corpus
filter. Its question-k-gram plus nearby option rule is suitable for conventional
MCQs. It is not, by itself, a safe reusable audit for the broader native-Greek
suite:

- question stems shorter than `K` were invisible;
- OYXOY evaluation prompts contain evaluator-authored scaffolding that is not
  part of the upstream source;
- OYXOY NLI represents one source pair as three scored binary decisions;
- OYXOY WIC, WSD and metaphor labels are encoded by source pairs or
  usage/definition pairs, not by the generated `Ναι`/`Όχι` choice text;
- the JSONL scanner retained document ids but not published shard, row and
  source-line evidence;
- its 92.4% precision estimate is specific to the reviewed GreekMMLU rule and
  must not be asserted for a new benchmark family without review.

The reusable path therefore uses:

1. `scripts/build_decontamination_queries.py --frozen-examples-jsonl ...` to
   bind the audit to the exact examples scored by the evaluator and map OYXOY
   back to its human-authored source surfaces;
2. `scripts/audit_benchmark_contamination_parquet.py` to scan the exact
   published Parquet revision without changing it;
3. `scripts/finalize_benchmark_contamination_audit.py` to emit complete,
   receipt-bound Hugging Face tables;
4. `subprojects/09_full_8b_cpt_results_analysis/evaluation/rescore_contamination_filtered.py`
   to report both full and strict-filtered scores for already-trained models.

## Match and discount policy

Text is normalized with NFKC, combining-mark removal, Unicode case-folding and
Unicode word tokens. Questions with at least eight tokens use 8-token k-grams.
A 3–7-token question uses its complete normalized stem, fixing the old blind
spot without treating a one- or two-token phrase as evidence.

Every question/source-anchor hit is published. It is recommended for score
exclusion only when a second, label-bearing surface also appears in the
proximity window:

| Evaluation family | First surface | Required second surface |
| --- | --- | --- |
| DemosQA, Medical MCQA, ASEP, GPCR | question | correct answer |
| OYXOY NLI | premise | hypothesis |
| OYXOY WSD | usage example | correct definition |
| OYXOY WIC | first usage | second usage |
| OYXOY metaphor | usage example | source definition |

Question-only matches remain `candidate` evidence. They are never
automatically discounted. OYXOY NLI exclusions apply to all three scored
decisions belonging to the matched premise/hypothesis group.

This is deliberately a post-hoc evaluation correction for a completed run. It
does not claim to undo training exposure and it does not mutate the published
dataset. Future training can select any benchmark's strict document list from
the table before packing.

## Published table contract

The auxiliary payload is uploaded below the audited dataset as
`benchmark_contamination/native_greek_suite_v1/`:

- `qa_document_line_matches.parquet`: one benchmark-unit/document match with
  source dataset, source document id, immutable dataset shard, zero-based row,
  1-based text lines, match class, hashes and a bounded evidence snippet;
- `benchmark_question_summary.parquet`: every audited evaluation unit,
  including unmatched units;
- `recommended_excluded_example_ids.jsonl`: the strict score filter;
- `audit_receipt.json`: dataset revision, query and shard bindings and counts;
- `publish_manifest.json`: hashes of the exact HF payload.

The current panel excludes text-only Protipa because its Hugging Face access
gate has not been approved. When access is granted, freeze its exact examples,
append a source-surface mapping if its evaluator prompt is synthetic, and rerun
the same scan. Existing table versions remain immutable evidence for the older
panel.

## CSCS execution

The full 131 GiB release already exists at its publication root on Clariden.
Run the resumable scanner on one CPU-only `debug` allocation; it uses four
lanes over the 431 local Parquet shards. The job script requests no production
training allocation and finalizes only after all 431 shard receipts validate.

```bash
sbatch --partition=debug \
  --export=ALL,AUDIT_CODE_ROOT=...,AUDIT_CODE_RECEIPT=...,AUDIT_RUN_ROOT=...,AUDIT_DATASET_RUN_ROOT=...,AUDIT_FROZEN_EXAMPLES=...,AUDIT_PYTHON=... \
  scripts/run_benchmark_contamination_audit.sbatch
```

Publishing is a separate operation after the payload manifest and a sampled
manual review of strong and candidate matches pass. Uploading the auxiliary
folder must not overwrite `data/*.parquet` or the root dataset card.
