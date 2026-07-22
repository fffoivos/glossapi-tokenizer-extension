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

## Terminal seal

Clariden CPU job `2799787` wrote and locked the canonical v2 seal:

`run-4256753/FROZEN.consensus-silver-v2.receipt.json`

The seal was produced by commit `3bfee86`. It rechecked the independent audit,
every artifact hash, exact retained-line identity and coverage, the 143 + 7 =
150 document partition, and the unchanged numerical terminal thresholds for
the primary `BIB`/`NON_BIB` membership target. All cohort files, the
materialization receipt, independent audit, and terminal seal are mode `0440`.

The earlier v1 seal from job `2799088` incorrectly called trusted-label
coverage “agreement.” It is preserved read-only for audit but superseded. V2
uses comparable repaired A/B votes for agreement and all retained lines for
coverage/unresolved rates.

| terminal membership gate | corrected result | passed |
|---|---:|---:|
| overall A/B agreement | 172,905 / 173,055 comparable = 99.9133% | yes (`>= 98%`) |
| Greek PhD | 92,978 / 93,047 = 99.9258% | yes (`>= 95%`) |
| Kallipos | 29,683 / 29,693 = 99.9663% | yes (`>= 95%`) |
| OpenArchives agreement | 50,244 / 50,315 comparable = 99.8589% | yes (`>= 95%`) |
| trusted-label coverage | 172,905 / 173,609 = 99.5945% | reported separately |
| unresolved coverage | 704 / 173,609 = 0.4055% | yes (`<= 0.5%`) |

### Effect of the repairs and document exclusion

These are three genuinely different states, not the same old calculation:

| metric | original 150 | both repairs, 150 | both repairs, seven dropped |
|---|---:|---:|---:|
| BIB/non-BIB agreement | 98.04% | 98.93% | **99.9133%** |
| header detection, both / union | 85.91% | 88.22% | **87.62%** |
| header subtype, conditional on both | 99.85% | 99.85% | **99.87%** |
| continuation/filler detection, both / union | 51.02% | 74.82% | **83.97%** |
| continuation/filler subtype, conditional on both | 99.26% | 99.00% | **98.86%** |

Dropping the seven footnote-confusion documents raises binary membership by
almost a full percentage point beyond the repairs and raises contextual-line
detection by another 9.15 points. Header detection falls slightly because the
dropped documents were not the main header-disagreement contributors; header
subtype agreement remains almost perfect once both annotators detect a header.

The exact post-drop analysis is locked at:

`run-4256753/agreement-analysis-v1.receipt.json`

Auxiliary masks remain intentionally incomplete and are recorded rather than
hidden. On comparable lines, exact task agreement is 99.8353% for entry seed,
99.8151% for context role, 99.3100% for heading type, and 99.0916% for exact
fine role. Their all-line trusted coverage is lower because 554 OpenArchives
lines contain an `UNKNOWN` vote. These are evaluation masks, not claims that
the original exact-role adjudication protocol passed.

The terminal status is
`frozen_posthoc_consensus_silver_evaluation_set`. It does **not** rewrite the
failed original 150-document prediction-blind attempt. The original
`40_frozen/consensus.receipt.json` remains blocked at 97.7583% raw A/B binary
agreement, and the new seal binds its hash and failed-gate provenance.

Small receipts and the exclusion manifest are also archived in the repository
under `results/sealed_bibliography/20260719/`. Sealed text, the private line
key, and line-level labels remain only on Clariden.

## Use in the remaining plan

This frozen artifact is reserved for the final source-matched comparison. It must not
be used to choose features, thresholds, decoder costs, or candidate ordering.
Development resumes on grouped train-only OOF evidence. When a small Pareto set
of candidates is frozen, the evaluation adapter will score each candidate on
the trusted task masks and report the 150 binary disagreements separately as
unresolved coverage rather than errors.
