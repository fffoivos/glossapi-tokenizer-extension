# Bibliography sealed test v2 — 2026-07-23

## Purpose

Create a fresh, source-balanced 150-document bibliography test cohort and have
every line labeled independently by two `gpt-5.6-terra` reviewers at high
reasoning. The cohort is sealed from model-development predictions and excludes
all documents used by the earlier source-matched holdout and sealed test.

## Frozen code and storage

- Code commit: `d4f2ac7aecf09cd45281ec9a9fee77c8d33c1563`
- Clariden checkout:
  `/capstor/scratch/cscs/fffoivos/code/train-apertus-d4f2ac7a`
- Clariden test root:
  `/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/sealed_tests/bibliography_150_20260723_v2`
- Annotation prompt:
  `SEALED_BIBLIOGRAPHY_ROLE_PROMPT_V2.md`

## Selection

- Prior exclusions: 650 documents: the complete 500-document source-matched
  holdout plus the complete earlier 150-document sealed cohort.
- Fresh candidate pool: 600 documents, 200 from each of Greek PhD, Kallipos,
  and OpenArchives.
- Candidate selection was prediction-blind and applied global identity,
  exact-text, and near-duplicate exclusions.
- The extraction-quality gate sent 36 flagged candidates to two independent
  `gpt-5.6-terra`/high reviewers. They agreed on every document: 28 `KEEP` and
  8 `UNUSABLE`.
- Final cohort: 150 unique documents, exactly 50 per source. One ranked Greek
  PhD candidate was excluded by the quality gate and replaced.
- Final sealed corpus: 210,704 lines.

Key receipts:

- `00_exclusions/receipt.final.json`
- `00_candidate_selection/selection.receipt.json`
- `05_quality/quality.consensus.json`
- `10_sealed_inputs/selection.final.receipt.json`
- `10_sealed_inputs/annotation-packets.receipt.json`
- `10_sealed_inputs/run.receipt.json`

The finalized document payload SHA-256 is
`ad652cfded35cc6766d4cfd325f814b660e462363381cd4ecc744875714f7947`.
The packet receipt explicitly records `packet_contains_predictions: false`.

## Independent Terra annotation

The two passes use different opaque aliases, chunk boundaries, and presentation
order. Neither pass receives predictions, prior labels, or the other pass's
answers.

### Pass A

- Reviewer: `sealed-role-terra-high-a-v2`
- Model/reasoning: `gpt-5.6-terra`, `high`
- Packet: `10_sealed_inputs/pass-a.packet.private.jsonl`
- Packet SHA-256:
  `7e539d39f1ca1abaf6ac69a80eb6ec3487e9627f63234e197112bfce5b88259f`
- Chunks: 752
- Immutable batch records: `20_role_terra_high_a/run`
- Final output: `20_role_terra_high_a/pass-a.json`

### Pass B

- Reviewer: `sealed-role-terra-high-b-v2`
- Model/reasoning: `gpt-5.6-terra`, `high`
- Packet: `10_sealed_inputs/pass-b.packet.private.jsonl`
- Packet SHA-256:
  `414d42e7ad2581fb232cb4aa75db42fbb2ca0e88a246e89f6c3405e1a842921e`
- Chunks: 814
- Immutable batch records: `21_role_terra_high_b/run`
- Final output: `21_role_terra_high_b/pass-b.json`

Both one-batch preflights passed on the first attempt. The exhaustive runs then
resumed those same contracts with `--pending-only`; completed batches are never
rerun.

## Label invariants corrected for this cohort

- `CONTINUATION` and `FILLER` exist only inside a contiguous bibliography-role
  component containing an `ENTRY` anchor. Similar-looking text elsewhere is
  `OTHER`.
- Heading roles apply only to ATX Markdown headings.
- Ordinary footnotes/endnotes, inline citations, and tables of contents are
  `OTHER`.
- Bibliographic source lists, webographies, and CV publication lists may be
  bibliography components when they actually list sources or publications.

## Completion gate

Do not call the test set labeled until both final pass receipts exist and cover
all 210,704 keyed lines. Preserve raw pass outputs. Any contextual repair must
be emitted as a separate derived artifact and reported; it must never overwrite
either reviewer record.
