# Sealed annotation model switch — 2026-07-18

## Decision

The canonical exhaustive A/B role annotation uses two independent
`gpt-5.6-terra` passes at `medium` reasoning, one worker per pass. The switch
reduces annotation usage while preserving packet blindness, immutable run
contracts, schema validation, and independent A/B review.

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

These partial responses are immutable aborted-run evidence. They must not be
resumed and must never be used in pass aggregation, adjudication, label merge,
freeze, model selection, or test reporting.

## Canonical Terra runs

| Lane | Packet | Run directory | Aggregate | Reviewer |
|---|---|---|---|---|
| Terra A | `10_sealed_inputs/pass-a.packet.private.jsonl` | `22_role_terra_a/run` | `22_role_terra_a/pass.json` | `sealed-role-terra-a-v1` |
| Terra B | `10_sealed_inputs/pass-b.packet.private.jsonl` | `23_role_terra_b/run` | `23_role_terra_b/pass.json` | `sealed-role-terra-b-v1` |

Every Terra contract must record model `gpt-5.6-terra`, reasoning effort
`medium`, sandbox `read-only`, and `ephemeral: true`. Lane A starts with one
bounded preflight batch. Once accepted, A resumes from that same contract and B
starts from its own fresh contract. Both run with one worker.

All downstream commands must consume only the Terra aggregate paths above.
If A/B disagree, adjudication uses a fresh `gpt-5.6-terra`/medium reviewer
`sealed-role-terra-c-v1` that receives only the label-blind adjudication packet.

## Verification

- Focused sealed-suite test: 32 passed.
- The coordinator now requires an explicit supported model and binds the exact
  model/reasoning effort into the remote immutable contract.
- Finalized pass and quality receipts report the model/reasoning values from
  their run contract rather than global constants.
- Unknown review models and reasoning levels are rejected.
