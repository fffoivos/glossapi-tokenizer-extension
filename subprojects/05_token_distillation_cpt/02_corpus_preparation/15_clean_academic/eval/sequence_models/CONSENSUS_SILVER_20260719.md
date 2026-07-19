# Bibliography consensus-silver materialization — 2026-07-19

## Decision

The final source-matched bibliography evaluation labels use the two repaired
A/B annotation passes and no third-pass adjudication. Seven documents with
systematic footnote-versus-bibliography disagreement were removed at Foivos's
direction. On every remaining line, a label is trusted only when A and B agree
after recoding for the particular downstream task.

This is **dual-Codex consensus silver**, not human gold. Disagreements remain in
the row inventory as masked targets; they are not resolved by a model or by an
implicit majority rule.

## Task-specific masks

The same A/B pair can provide supervision for one component while remaining
unresolved for another:

- `bibliography_membership`: `BIB` versus `NON_BIB`;
- `entry_seed`: `ENTRY` versus `NOT_ENTRY`;
- `heading_type`: `BIB_HEADER`, `BIB_SUBHEADER`, `NON_BIB_HEADER`, or
  `NOT_HEADER`;
- `context_role`: `CONTINUATION`, `FILLER`, or `OTHER`; and
- `fine_role`: exact agreement on the seven-role annotation.

For example, `CONTINUATION` versus `FILLER` is trusted as `BIB` and
`NOT_ENTRY`, but is masked for `context_role` and `fine_role`. If either pass
uses `UNKNOWN`, every task is masked for that line.

## Executed result

Clariden CPU job `2798789` materialized the artifact from code commit
`4256753`. Independent CPU job `2798796`, using audit code commit `a612390`,
then re-read the original documents, line key, both repaired passes, every
derived row, and every hash.

| quantity | result |
|---|---:|
| included documents | 143 |
| excluded documents | 7 |
| emitted lines | 173,609 |
| comparable A/B lines | 173,055 |
| trusted BIB/non-BIB labels | 172,905 |
| masked BIB/non-BIB disagreements | 150 |
| BIB/non-BIB agreement on comparable lines | 99.9133% |
| exact seven-role agreements | 171,483 |
| lines with at least one `UNKNOWN` vote | 554 |

The independent receipt confirms that included and excluded documents exactly
partition the 150-document source, excluded documents occur in no derived
training/evaluation output, line identity/order/coverage are exact, all task
labels reproduce from A/B, and every source/output hash matches.

## Clariden artifacts

Root:

`/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/sealed_tests/bibliography_150_20260718/48_consensus_silver`

Materialized artifact:

`run-4256753/`

- `documents.consensus-silver.jsonl` — the 143 retained full documents;
- `line-key.consensus-silver.jsonl` — retained private line identities;
- `labels.task-consensus.jsonl` — A/B votes plus all five task labels/masks;
- `fine-role.overlay-v3.jsonl` — compatibility overlay that trusts exact
  seven-role agreements only;
- `exclusions.json` — the seven removed document IDs and source metadata; and
- `receipt.json` — source/output paths, hashes, counts, and invariants.

Independent audit:

`audit-a612390.receipt.json`

The original inputs and repaired passes remain unchanged under
`10_sealed_inputs/` and `47_markdown_header_repair/`.

## Use in the remaining plan

This artifact is reserved for the final source-matched comparison. It must not
be used to choose features, thresholds, decoder costs, or candidate ordering.
Development resumes on grouped train-only OOF evidence. When a small Pareto set
of candidates is frozen, the evaluation adapter will score each candidate on
the trusted task masks and report the 150 binary disagreements separately as
unresolved coverage rather than errors.
