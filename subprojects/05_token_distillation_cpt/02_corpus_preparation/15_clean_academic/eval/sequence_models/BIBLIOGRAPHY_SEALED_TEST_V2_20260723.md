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
resumed those same contracts with `--pending-only`; completed batches were
never rerun. Pass A completed 376 batches and pass B completed 407 batches.
Both final outputs cover all 210,704 lines.

The final pass SHA-256 values are:

- pass A: `0ae6c019f865fb71488233ad1a804f586c8603508aa12ff61fbcc7c14149bc1e`
- pass B: `0885b4777b80652644a61c3ef1be78b95358a7dc40d8d36041c854a251bf9b03`

## Label invariants corrected for this cohort

- `CONTINUATION` and `FILLER` exist only inside a contiguous bibliography-role
  component containing an `ENTRY` anchor. Similar-looking text elsewhere is
  `OTHER`.
- Heading roles apply only to ATX Markdown headings.
- Ordinary footnotes/endnotes, inline citations, and tables of contents are
  `OTHER`.
- Bibliographic source lists, webographies, and CV publication lists may be
  bibliography components when they actually list sources or publications.

## Completed annotation audit

Clariden job `2875593` audited both complete raw passes without mutating them.

- Contextual-role violations: zero in both passes. No `CONTINUATION` or
  `FILLER` label required repair.
- Non-Markdown heading-role violations: 20 in pass A and 34 in pass B. These
  were changed only in separately derived audit copies under
  `22_annotation_audit`; the raw evidence remains unchanged.
- The side-by-side review reader and its agreement manifest are under
  `23_ab_review_site`. Clariden job `2875604` completed the build.

Agreement after the narrow Markdown-header repair:

- BIB/non-BIB: 99.7328% overall, Cohen's kappa 0.9868.
- BIB/non-BIB by source: 99.6470% Greek PhD, 99.5628% Kallipos, and 99.9966%
  OpenArchives.
- Heading detected by both versus either: 85.1247%.
- Heading subtype when both detected a heading: 99.6815%.
- Filler/continuation detected by both versus either: 86.2475%.
- Filler versus continuation when both detected one: 98.4966%.

The cohort passes the predeclared 98% overall and 95% per-source primary
BIB/non-BIB agreement gates. The 563 primary-task disagreements remain
unresolved; they must be masked rather than silently adjudicated. The two raw
passes and the derived audit copies must all be preserved.
