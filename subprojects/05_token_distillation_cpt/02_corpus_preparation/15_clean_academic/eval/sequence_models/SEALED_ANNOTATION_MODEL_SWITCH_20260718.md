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
