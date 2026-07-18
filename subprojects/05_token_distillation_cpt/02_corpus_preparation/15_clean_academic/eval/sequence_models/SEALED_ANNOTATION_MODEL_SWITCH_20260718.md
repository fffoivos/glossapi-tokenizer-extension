# Sealed annotation model switch — 2026-07-18

## Corrected decision

The canonical exhaustive A/B role annotation preserves every accepted
`gpt-5.6-sol`/high batch and uses `gpt-5.6-terra`/high only to complete the
remaining batches, one worker per pass. The Sol prefixes are the output of the
stronger model and are not re-annotated.

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
| Hybrid A | `10_sealed_inputs/pass-a.packet.private.jsonl` | `24_role_sol_terra_high_a/run` | `24_role_sol_terra_high_a/pass.json` | `sealed-role-sol-terra-high-a-v1` |
| Hybrid B | `10_sealed_inputs/pass-b.packet.private.jsonl` | `25_role_sol_terra_high_b/run` | `25_role_sol_terra_high_b/pass.json` | `sealed-role-sol-terra-high-b-v1` |

Every continuation contract records model `gpt-5.6-terra`, reasoning effort
`high`, sandbox `read-only`, and `ephemeral: true` for newly produced batches.
The import receipt binds 174 Sol/high batches for A and 176 Sol/high batches for
B. Terra starts at the next batch index, so the prefixes are not sent to a
model again. Both continuations run with one worker.

All downstream commands must consume only the Terra aggregate paths above.
If A/B disagree, adjudication uses a fresh `gpt-5.6-terra`/high reviewer
`sealed-role-terra-high-c-v1` that receives only the label-blind adjudication
packet.

## Verification

- Focused sealed-suite test: 33 passed.
- The mistakenly started medium Terra runs stopped after 10 batches in each
  lane. They overlap the retained Sol prefixes and are not canonical inputs.
- The coordinator now requires an explicit supported model and binds the exact
  model/reasoning effort into the remote immutable contract.
- Finalized pass and quality receipts report the model/reasoning values from
  their run contract rather than global constants.
- Finalized role passes report batch counts for every actual annotation runtime,
  distinguishing imported Sol/high batches from direct Terra/high batches.
- Unknown review models and reasoning levels are rejected.
