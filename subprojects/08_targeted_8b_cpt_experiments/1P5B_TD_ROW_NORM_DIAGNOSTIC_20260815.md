# 1.5B Token-Distillation row-norm diagnostic

Date: 2026-08-15

## Decision

The 1.5B arm remains blocked at its predeclared Token-Distillation row-norm
gate.  The gate was not relaxed after observing the result, the TD output was
not rescaled, and no alternate source layer was tried.

This block does not apply to the independently verified 8B arm.

## Immutable evidence

- Failed TD verification job: `3086780`
- Diagnostic job: `3087726` (`COMPLETED`, 17 seconds, one `debug` node)
- Diagnostic receipt:
  `/capstor/scratch/cscs/fffoivos/cpt_runs/hard_h2g_matched/20260814T201715Z-r2-v14/receipts/1p5b_td_row_norm_failure_diagnostic_3087726.json`
- Diagnostic bundle:
  `/iopsstor/scratch/cscs/fffoivos/orchestration/targeted-8b-cpt/20260815T123000Z-hard-h2g-r2-v59`
- Bundle tree SHA-256:
  `9cbb4e555e127f291f141c2bf9b8f103f1e5d6df0dcf43039f01aa3217a110eb`

The diagnostic measured all 17,408 added rows: 17,345 TD-trained rows and 47
rows skipped by the TD trainer.

## Result

| Matrix | Frozen 8B-derived band | 1.5B reference median ratio | 1.5B TD median ratio | Rows inside band | Median gate |
|---|---:|---:|---:|---:|---:|
| `model.embed_tokens.weight` | 1.023066-1.030353 | 0.954046 | 0.954301 | 0.0% | pass |
| `lm_head.weight` | 0.997117-1.020100 | 0.908103 | 0.910811 | 0.0% | pass |

The TD deltas relative to the 1.5B reference are small:

- input-embedding median ratio delta: approximately `+0.000253`;
- output-head median ratio delta: approximately `+0.002702`.

The failed fraction-in-band check is therefore not evidence of an exploding or
collapsed TD optimization.  The 1.5B reference initialization itself is below
the frozen 8B-derived bands, by approximately 4.6% for input embeddings and
9.2% for the output head.  TD remains close to that 1.5B reference.

## Interpretation boundary

The diagnostic supports this narrow statement: the rejected arm is
architecturally scale-mismatched to the frozen cross-scale row-norm band.
It does not establish that the TD initialization is safe to train.  Accepting
it would require an explicit prospective revision of the scientific contract,
with a scale-aware gate justified and approved before any new 1.5B trial.

