# Sealed annotation model switch — 2026-07-18

## Corrected decision

The canonical exhaustive A/B role annotation preserves every accepted
`gpt-5.6-sol`/high batch and uses `gpt-5.6-terra`/high only to complete the
remaining batches. The accepted Sol records are the output of the stronger
model and were not re-annotated. They form sparse batch sets because the
original two-worker runs completed out of order. Both hybrid passes are now
complete and finalized.

The completed quality gate remains historical `gpt-5.6-sol`/high evidence. It
is not rerun by this switch.

## Stopped Sol role runs

Both local Sol coordinator process groups were terminated before starting
Terra. Their last persisted remote state was:

| Lane | Run directory | Accepted batches | Rejected batches | Aggregate |
|---|---|---:|---:|---|
| Sol A | `20_role_a/run` | 174 | 0 | absent |
| Sol B | `21_role_b/run` | 176 | 0 | absent |

The sealed root is
`/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/sealed_tests/bibliography_150_20260718`.

These responses are immutable source evidence for the continuation. They are
verified against their original contracts and packets, rebound into the
canonical continuation contracts, and retain hashes identifying the original
contract, response record, model, effort, reviewer, and batch ID.

## Canonical continuation runs

| Lane | Packet | Run directory | Aggregate | Reviewer |
|---|---|---|---|---|
| Hybrid A | `10_sealed_inputs/pass-a.packet.private.jsonl` | `26_role_sol_terra_high_a/run` | `26_role_sol_terra_high_a/pass.json` | `sealed-role-sol-terra-high-a-v1` |
| Hybrid B | `10_sealed_inputs/pass-b.packet.private.jsonl` | `27_role_sol_terra_high_b/run` | `27_role_sol_terra_high_b/pass.json` | `sealed-role-sol-terra-high-b-v1` |

Every continuation contract records model `gpt-5.6-terra`, reasoning effort
`high`, sandbox `read-only`, and `ephemeral: true` for newly produced batches.
The import receipt binds the actual 174 Sol/high batch indices for A and 176
Sol/high batch indices for B. Terra receives only indices reported missing, so
accepted Sol batches were not sent to a model again.

All downstream commands must consume only the Terra aggregate paths above.
Where A/B disagree or return `UNKNOWN`, adjudication uses a fresh
`gpt-5.6-terra`/high reviewer
`sealed-role-terra-high-c-v1` that receives only the label-blind adjudication
packet.

## Completed canonical receipts

- Hybrid A finalized all 358 batches and 194,273 lines. Its
  `overlap_exact_role_agreement` is `0.9734982332155477`; aggregate SHA-256 is
  `9d254ed0806fcb9c83059504479bf08fe3cc19ad24a9efb04123c4040c9cc067`.
- Hybrid B finalized all 388 batches and 194,273 lines. Its
  `overlap_exact_role_agreement` is `0.9771733333333333`; aggregate SHA-256 is
  `890366c3b952f599f9cb1d35adb4f5606f8581c41d306d7e6f442607bc63bb0c`.
- The label-blind C packet contains 333 context chunks and 6,286 target lines.
  Terra/high completed all 167 batches with reviewer
  `sealed-role-terra-high-c-v1`. The aggregate SHA-256 is
  `3852ae7f784b55d6098c5b79ca0aa750a888273670eabd0b352de9a90610fa30`.

## Frozen-gate result

The merge ran as Clariden CPU job `2793742` and failed closed as designed. The
preserved receipt is `40_frozen/consensus.receipt.json`; its status is
`blocked`. Exact results are:

| Gate | Result | Passed |
|---|---:|---|
| Complete 194,273-line coverage | complete | yes |
| Overall A/B binary agreement | `0.9775830918346863` (required `>= 0.98`) | **no** |
| Greek PhD agreement | `0.983506626638354` (required `>= 0.95`) | yes |
| Kallipos agreement | `0.9966995588185767` (required `>= 0.95`) | yes |
| OpenArchives agreement | `0.955625045188345` (required `>= 0.95`) | yes |
| Unresolved after C | 690 / 194,273 = `0.003551703015859126` (required `<= 0.005`) | yes |

The overall raw A/B agreement gate is independent of C adjudication, so the C
pass cannot repair it. The threshold must not be relaxed or tuned after seeing
sealed annotations. `40_frozen/FROZEN.receipt.json` therefore does not exist,
and the generated merged labels are not a frozen test set.

## A/B visual comparison

The annotation-QA reader is under
`45_ab_comparison_site/site-76cd045/`. It shows all 150 documents and all
194,273 lines with pass A and pass B side by side, uses a single aligned scroll,
colours each detailed role, highlights binary disagreements, and sorts the
document menu from worst to best agreement. Previous/next controls jump between
binary disagreements. The reader contains no model predictions and is not an
input to development selection.

- Generator commit: `76cd045ff3c411accac93fe287fb5d925a53c4f2`.
- Focused Clariden test job `2794043`: 3 passed.
- Clariden build job `2794044`: `COMPLETED 0:0`; empty stderr.
- Receipt SHA-256:
  `5cb94f3afc801e6dc6b4849c35a5d0a74aefd72754292344c690c6753255a396`.
- Manifest SHA-256:
  `7254a75c1aaa021ef65ba006f720f2e9a4851dbaf006278fe186bf6cd96b13ac`.
- Visual QA was performed at 1800 by 1200 pixels after correcting the sticky
  column-heading alignment.

### Actual annotator provenance and task-specific agreement

The comparison reader was extended to resolve every owned packet line through
its immutable response record. It now identifies the actual annotation model
(`Sol`, `Terra`, or `Mixed`) for each document and shows the actual model beside
every displayed line. This is distinct from the aggregate reviewer ID, which
describes the combined pass rather than the model that produced a particular
response.

The same build computes symmetric A/B agreement after recoding the seven roles
for the three downstream learning tasks. Neither pass is treated as ground
truth. Per-class agreement is therefore reported as symmetric F1, alongside
the A-to-B confusion matrix, observed agreement, and Cohen's kappa.

- Bibliography membership maps `ENTRY`, `CONTINUATION`, `FILLER`,
  `BIB_HEADER`, and `BIB_SUBHEADER` to `BIB`; `NON_BIB_HEADER` and `OTHER` map
  to `NON_BIB`. Lines with `UNKNOWN` in either pass are excluded. On 193,718
  comparable lines, agreement is `0.9803838569466957`, kappa is
  `0.9149675418331754`, BIB symmetric F1 is `0.9262708575863408`, and NON-BIB
  symmetric F1 is `0.9886869745397385`.
- Heading review uses the three heading types plus a diagnostic `NON_HEADER`
  bucket. Across the 11,738-line union where either pass saw a heading, exact
  agreement is `0.8578122337706594`. On the 10,084 lines where both passes saw
  a heading, exact heading-type agreement rises to `0.9985124950416502`:
  symmetric F1 is `0.9735099337748344` for `BIB_HEADER`,
  `0.973404255319149` for `BIB_SUBHEADER`, and `0.9993842996408414` for
  `NON_BIB_HEADER`.
- Gap-line review maps all roles other than `FILLER` and `CONTINUATION` to a
  diagnostic `OTHER` bucket. Across the 4,784-line union where either pass saw
  a gap line, exact agreement is `0.5064799331103679`. On the 2,441 lines where
  both passes saw a gap line, filler/continuation type agreement is
  `0.9926259729619009`: symmetric F1 is `0.9929022082018928` for `FILLER` and
  `0.9923273657289002` for `CONTINUATION`.

The interpretation is clear: once both annotators detect a heading or a gap
line, they almost never disagree about its subtype. The material annotation
uncertainty is the detection boundary--whether a special heading or gap line is
present at all--rather than the subtype distinction. That boundary should be
the focus of annotation review and model evaluation.

The source-specific results make the difference clearer:

| Source | Comparable lines | BIB/non-BIB agreement | Heading detected by both / union | Heading-type agreement | Gap line detected by both / union | Filler/continuation agreement |
|---|---:|---:|---:|---:|---:|---:|
| Greek PhD | 109,255 | 98.35% | 6,926 / 7,782 = 89.00% | 99.84% | 917 / 1,512 = 60.65% | 99.02% |
| Kallipos | 29,693 | 99.67% | 12 / 13 = 92.31% | 100.00% | 100 / 213 = 46.95% | 100.00% |
| OpenArchives | 54,770 | 96.53% | 3,146 / 3,943 = 79.79% | 99.87% | 1,424 / 3,059 = 46.55% | 99.37% |

The conditional symmetric F1 values by subtype are:

| Source | BIB | NON-BIB | BIB header | BIB subheader | NON-BIB header | Filler | Continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Greek PhD | 94.08% | 99.04% | 96.71% | 97.05% | 99.94% | 97.83% | 99.37% |
| Kallipos | 98.37% | 99.82% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| OpenArchives | 87.42% | 97.99% | 98.85% | 97.81% | 99.93% | 99.56% | 98.87% |

Heading-subtype F1 is conditional on both passes identifying a heading;
filler/continuation F1 is conditional on both identifying a gap line. Detection
agreement is reported separately so a dominant `OTHER` or `NON_HEADER` class
cannot hide missed special lines.

Actual model pairing is not balanced across sources. Kallipos and OpenArchives
are entirely `Terra -> Terra`. Greek PhD contains 93,264 `Sol -> Sol`, 6,038
`Sol -> Terra`, 1,459 `Terra -> Sol`, and 8,495 `Terra -> Terra` owned lines.
Source differences therefore must not be interpreted as clean Sol-versus-Terra
model differences.

### Post-hoc contextual-role repair

The later review found `FILLER` and a few `CONTINUATION` labels in components
with no bibliography entry. A derived, fully audited repair changes only those
logically impossible contextual labels to `OTHER`; the canonical passes remain
unchanged. It raises filler/continuation detection agreement from 51.02% to
74.82% and bibliography-membership agreement from 98.04% to 98.93%.

The full rule, hashes, counts, before/after figures, and analysis of the
remaining heading disagreement are recorded in
`CONTEXTUAL_ROLE_REPAIR_20260719.md`. This is a post-hoc annotation audit and
does not retroactively change the frozen-gate result.

The current reader is `45_ab_comparison_site/site-cab8d7f/`, built by Clariden
CPU job `2794761` after six focused tests passed in job `2794755`. Its manifest
SHA-256 is
`a81a1155302689dac2b09494e4a9dba0e2ca4ac7b312d0cee19e690d6d8515c3`.
The reader shows the five task-specific agreement values for every source and
for the currently selected document, and identifies the actual model beside
each pass and each line.

## Final user-directed consensus successor

The original 150-document attempt remains blocked exactly as reported above.
It was not rescued by changing the 0.98 threshold or rerunning the annotators.

After the two deterministic annotation repairs and review of the documents
dominating residual disagreement, Foivos directed that seven problematic
documents be excluded and that only repaired A/B task agreements be retained.
The resulting 143-document consensus-silver cohort was materialized by job
`2798789`, independently audited by job `2798796`, and terminally sealed with
corrected agreement denominators by job `2799787` using commit `3bfee86`.

The terminal file is:

`48_consensus_silver/run-4256753/FROZEN.consensus-silver-v2.receipt.json`

It reports status `frozen_posthoc_consensus_silver_evaluation_set`, locks all
bound cohort files mode `0440`, preserves the hash and failure of the original
150-document receipt, and passes the unchanged numerical membership gates:
99.9133% agreement on comparable repaired votes, at least 99.8589% in every
source, and 0.4055% unresolved over all retained lines. Trusted-label coverage
is separately 99.5945%. See `CONSENSUS_SILVER_20260719.md` for task-specific
agreement, coverage, detection metrics, and exact artifact paths.

The previous v1 seal from job `2799088` is preserved but superseded: it used
the strict trusted-coverage fraction as if it were inter-annotator agreement.

## Verification

- Focused sealed-suite test: 33 passed.
- The mistakenly started medium Terra runs stopped after 10 batches in each
  lane. They overlap the retained Sol records and are not canonical inputs.
- An initial continuation-import attempt under `24_role_sol_terra_high_a` and
  `25_role_sol_terra_high_b` assumed a contiguous prefix and failed closed. The
  corrected sparse-record contracts are `26_role_sol_terra_high_a` and
  `27_role_sol_terra_high_b`.
- Both corrected imports passed, preserving exactly 174 A and 176 B Sol/high
  records. Terra/high completed every missing batch and both canonical
  aggregates finalized.
- The coordinator now requires an explicit supported model and binds the exact
  model/reasoning effort into the remote immutable contract.
- Finalized pass and quality receipts report the model/reasoning values from
  their run contract rather than global constants.
- Finalized role passes report batch counts for every actual annotation runtime,
  distinguishing imported Sol/high batches from direct Terra/high batches.
- Unknown review models and reasoning levels are rejected.
