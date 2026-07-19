# Sealed bibliography annotation completion audit — 2026-07-19

## Bottom line

The exhaustive annotation workflow completed through A, B, label-blind C, and
merge. The original prediction-blind 150-document attempt correctly failed its
predeclared raw A/B agreement gate and is permanently **not frozen**.

After annotation-QA repairs, Foivos directed a different final policy: exclude
seven systematically problematic documents and keep only repaired A/B
agreements, separately by task. That 143-document successor is now materially
complete, independently audited, terminally sealed, and read-only. Its exact
status is `frozen_posthoc_consensus_silver_evaluation_set`; it must never be
described as the original 150-document prediction-blind freeze.

## Requirement-by-requirement evidence

| requirement | authoritative evidence | result |
|---|---|---|
| Select 50 documents/source and build two complete prediction-blind packets | `10_sealed_inputs/run.receipt.json`; `annotation-packets.receipt.json`; job `2789808` | Passed: 150 documents, 194,273 lines, 50/50/50, packets contain no predictions |
| Preserve completed Sol/high work and finish lane A with Terra/high | `26_role_sol_terra_high_a/pass.json`; import receipt | Passed: 174 imported Sol batches + 184 Terra batches, 716 chunks, 194,273 lines |
| Preserve completed Sol/high work and finish lane B independently with Terra/high | `27_role_sol_terra_high_b/pass.json`; import receipt | Passed: 176 imported Sol batches + 212 Terra batches, 775 chunks, 194,273 lines |
| Finalize both aggregates | A SHA `9d254ed0...`; B SHA `890366c3...` | Passed; both current hashes reproduce the merge and adjudication receipts |
| Build a label-blind A/B disagreement/UNKNOWN packet | `30_adjudication/packet.receipt.json` | Passed: 333 context chunks, 6,286 targets, no A/B labels; packet SHA `ec69bb52...` |
| Run an independent Terra/high C pass | `30_adjudication/pass.json` | Passed: reviewer `sealed-role-terra-high-c-v1`, 167 Terra/high batches, 6,286 target lines; SHA `3852ae7f...` |
| Merge and enforce the frozen gates without relaxing them | `40_frozen/consensus.receipt.json`; job `2793742` | Enforced fail-closed: 97.7583% raw A/B agreement failed the fixed 98% gate; the other three gates passed |
| Do not misrepresent the failed original attempt | absence of `40_frozen/FROZEN.receipt.json`; blocked receipt SHA `9665af25...` | Passed: no original terminal seal exists and the failure is preserved |
| Apply the later agreement-only decision | `48_consensus_silver/run-4256753/receipt.json`; job `2798789` | Passed: 143 retained + 7 excluded, 173,609 lines, task-specific masks, no C or implicit adjudicator |
| Independently reconstruct and audit the successor | `48_consensus_silver/audit-a612390.receipt.json`; job `2798796` | Passed: source/output hashes, partition, identities, order, coverage and every task decision reproduced |
| Freeze and lock the successor transparently | `FROZEN.consensus-silver-v2.receipt.json`; job `2799787`; commit `3bfee86` | Passed: agreement and coverage use distinct denominators, all primary membership gates true, and every bound cohort file and receipt mode `0440` |
| Keep the evaluation set unopened during continued model development | continuation experiment receipts and `BIB_CONTINUATION_HEAD_RUN_20260719.md` | Passed for the recorded experiment: grouped train-only evidence was used; validation and consensus labels were unopened |
| Leave no annotation/coordinator jobs running | Mac process audit and Clariden queue audit after the seal jobs | Passed: no matching coordinator process or sealed annotation job remained |

## Canonical terminal metrics

The successor's primary target is `bibliography_membership`:

| scope | agreements | comparable | agreement | trusted coverage over all lines |
|---|---:|---:|---:|---:|
| all | 172,905 | 173,055 | 99.9133% | 99.5945% |
| Greek PhD | 92,978 | 93,047 | 99.9258% | 99.9258% |
| Kallipos | 29,683 | 29,693 | 99.9663% | 99.9663% |
| OpenArchives | 50,244 | 50,315 | 99.8589% | 98.7714% |

Unresolved primary labels are 704 / 173,609 = 0.4055%. Thus the unchanged
98% overall, 95% per-source, and 0.5% unresolved numerical thresholds all pass
for the successor's primary target.

Auxiliary agreement on comparable lines is 99.8353% for entry seed, 99.8151%
for context role, 99.3100% for heading type, and 99.0916% for exact fine role.
Trusted coverage is reported separately. Category-sensitive metrics after both
repairs and the drop are 87.62% header detection with 99.87% conditional header
subtype agreement, and 83.97% continuation/filler detection with 98.86%
conditional subtype agreement. These are masked evaluation channels, not fully
adjudicated targets.

## Paths

Clariden root:

`/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/sealed_tests/bibliography_150_20260718`

Successor artifact and terminal seal:

`48_consensus_silver/run-4256753/`

Repository-local receipt archive, containing no sealed text or line labels:

`results/sealed_bibliography/20260719/`

## Verification performed for this audit

- Recomputed current SHA-256 values for canonical A, B, the label-blind packet,
  C, the original blocked receipt, and the successor terminal seal; every hash
  matches its consuming receipt.
- Inspected current A/B/C aggregate metadata, reviewer identities, model,
  reasoning level, imported/direct batch counts, line counts, and packet hashes.
- Verified every successor data/metadata file and the independent audit is
  mode `0440`.
- Verified corrected seal job `2799787` and agreement-analysis job `2799790`
  completed with exit code 0.
- Ran the combined sealed workflow and consensus test set after the correction:
  55 tests passed.
