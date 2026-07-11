# Production / probe peak-LR decision - 2026-06-13

**Status:** decided. Use **peak LR `5.5e-5`** as the default for the next
full-corpus probe / production-design runs.

**Scope:** this settles the peak-LR choice after the `curriculum_sweeps_v2` LR
sweep at the selected replay split: 79% new Greek, 20% foreign replay, 1%
old-Greek replay. Alpha, beta3 and beta2 were settled later in
`PRODUCTION_HYPERPARAMETERS_DECISION_20260711.md`; corpus and curriculum policy
remain separate.

## Decision

Use `LR_PEAK=5.5e-5`.

Rationale:

- It is the best overall loss-based compromise.
- It is almost as strong as the high-LR arms on GreekMMLU, but without their
  old-data loss cost.
- It keeps all foreign held-out loss deltas negative at the final checkpoint.
- The apparent GreekMMLU win at `1.1e-4` is not enough to override the LM-loss
  evidence.

## Evidence

From `03_training_experiments/curriculum_sweeps_v2/results/peak_lr_decision_table_20260613.md`:

| peak LR | final GreekMMLU | best GreekMMLU | foreign mean delta loss | foreign max delta loss | old Greek delta loss | code proxy delta loss |
|---:|---:|---:|---:|---:|---:|---:|
| `2.75e-5` | 0.5721 | 0.5738 | -0.0900 | -0.0675 | -1.9487 | -0.1111 |
| `5.5e-5` | 0.5850 | 0.5868 | -0.0579 | -0.0435 | -1.9296 | -0.0824 |
| `8.25e-5` | 0.5874 | 0.5885 | -0.0279 | -0.0175 | -1.8376 | -0.0564 |
| `1.1e-4` | 0.5921 | 0.5921 | +0.0011 | +0.0174 | -1.7774 | -0.0300 |

New-Greek held-out LM-loss adaptation was slightly strongest at `5.5e-5` by
delta, while `2.75e-5` was strongest for old-data retention. The selected LR is
therefore the loss-based middle: materially better GreekMMLU than `2.75e-5`,
cleaner old-data retention than `8.25e-5` / `1.1e-4`, and no positive foreign
loss deltas.

## Interpretation

GreekMMLU alone points to `1.1e-4`, but the run is not being selected as an
MMLU-only model-selection problem. The safer read is:

- `1.1e-4` may improve multiple-choice behavior while being a less conservative
  continued-pretraining language-model update.
- `2.75e-5` is the retention winner, but gives up too much GreekMMLU.
- `5.5e-5` is the best overall balance under a loss-first decision rule.

## Implementation

The v2 launch env already defaults to this value:

- `03_training_experiments/curriculum_sweeps_v2/train/curriculum_common.env`
- `03_training_experiments/curriculum_sweeps_v2/train/submit_curriculum_two_phase.sh`

Future launch scripts should leave `LR_PEAK` unset, or explicitly set
`LR_PEAK=5.5e-5`.
