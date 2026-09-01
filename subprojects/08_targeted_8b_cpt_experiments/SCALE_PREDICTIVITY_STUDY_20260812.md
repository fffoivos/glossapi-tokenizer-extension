# Apertus CPT replication and scale-predictivity study

Date: 2026-08-12  
Status: design only; not launch-authorized

## Scope decision

The existing targeted Experiment A remains active and unchanged. The former
Experiment B, which continued update 9,536 over unseen non-HPLT sequences, was
retired by the owner on 2026-08-12. Its code and receipts remain immutable for
possible future reuse, but it is not part of the next training program.

The next program asks two different questions:

1. Can the historical 13.5B-token Apertus 8B HPLT-to-GlossAPI result be
   independently reproduced?
2. Do matched 0.5B and 1.5B runs reproduce the *shape and treatment effect* of
   the 8B trajectory well enough to screen future experiments cheaply?

These are deliberately separated. A single small-model trajectory can test
trajectory similarity, but it cannot prove that a small model will rank two
competing training designs in the same order as 8B.

## Historical target

The target is the selected beta2 arm from the completed curriculum sweep:

- run tag: `curr_td_b20p999_b3p999_13b_20260616T093527Z`;
- model: `swiss-ai/Apertus-8B-2509`;
- 3,218 optimizer updates and 4,194,304 token slots per update;
- 13,497,270,272 total token slots;
- hard HPLT-to-GlossAPI transition at update 2,261, after
  9,483,321,344 token slots;
- 79% new Greek, 20% foreign replay and 1% Old-Greek replay in both phases;
- peak/final LR `5.5e-5` / `5.5e-6`, fixed 400-update warmup, final 20%
  one-minus-square-root WSD cooldown;
- AdEMAMix beta1/beta2/beta3 `0.9/0.999/0.999`, alpha `4`, with alpha and
  beta3 ramped over the full run;
- sequence length 4,096, RoPE base 500,000, Goldfish `k=h=50`;
- final GreekMMLU accuracy `0.5993867244`; best observed accuracy
  `0.5996272246`.

Evidence:

- `../05_token_distillation_cpt/PRODUCTION_HYPERPARAMETERS_DECISION_20260711.md`
- `../05_token_distillation_cpt/03_training_experiments/curriculum_sweeps_v2/results/beta2_decision_table_20260711.csv`
- `../05_token_distillation_cpt/03_training_experiments/curriculum_sweeps_v2/results/sweep_config_audit_20260711.json`

The original packed curriculum binaries were deleted. Therefore the strongest
possible rerun is an independent, receipt-bound reconstruction from the same
source recipe, source identities where recoverable, seed and schedule. It is
not a byte-for-byte replay of the old sequence stream. The report must use the
term `recipe-level independent replication` unless exact packed payload hashes
are recovered.

The historical arm used the 148,480-token Modern-Greek extension. The current
production tokenizer has 148,992 tokens and adds the later 512-token polytonic
stage. Changing tokenizer/initialization makes the run a production-stack
near-replication rather than a strict historical replication. This choice must
be made and named before building the scale matrix; it must be identical at all
three scales.

## Stage 1: matched trajectory study

Run one hard HPLT-to-GlossAPI trajectory at each scale:

| Cell | Base model | Purpose |
|---|---|---|
| `R-HG-0p5B` | pinned `swiss-ai/Apertus-v1.1-0.5B` base revision | cheapest proxy |
| `R-HG-1p5B` | pinned `swiss-ai/Apertus-v1.1-1.5B` base revision | intermediate proxy |
| `R-HG-8B` | pinned `swiss-ai/Apertus-8B-2509` base revision | replication target |

All three cells must consume the same post-exclusion document identities in
the same HPLT and GlossAPI order, with the same replay identities at matched
token positions. Hold fixed:

- 13.497B token-slot horizon;
- HPLT-to-GlossAPI switch after 70.260% of token progress;
- 79/20/1 Modern-Greek/foreign/Old-Greek shares in small quota windows;
- sequence length 4,096 and cross-document attention, position and EOD-loss
  masks;
- tokenizer vocabulary and merge rules selected for the study;
- Goldfish loss and AdEMAMix settings;
- warmup ending after the same number of active token slots;
- WSD cooldown beginning at 80% token progress and ending at 10% of peak;
- checkpoint/evaluation positions expressed in active tokens, not wall time or
  scale-specific optimizer-step labels.

The native architecture stays unchanged at each scale. In particular, the
0.5B model has tied embeddings, while the released 1.5B and 8B models have
untied embeddings. Token Distillation must use the already validated tied
implementation for 0.5B and the validated untied implementation for 1.5B/8B.
Base vocabulary rows must remain byte-identical, and the extended rows must
pass the existing row-norm, coverage, finite-value and round-trip gates.

### Evaluation cadence

Evaluate the same fixed panels at update zero, after warmup, immediately before
and after the hard transition, at cooldown start, final, and approximately
every 1B active tokens. Every checkpoint is scored on:

- decontaminated GreekMMLU accuracy;
- GreekMMLU choice NLL, which is the primary benchmark trajectory;
- GreekMMLU correct-answer BPB;
- every source-conditioned heldout panel separately;
- balanced HPLT/GlossAPI BPB and replay-retention BPB;
- newly added-token and base-token target strata.

The public full GreekMMLU score is retained only to compare with the historical
59.94% number. Scientific conclusions use the fixed clean subset and the
source-conditioned validation panel.

### What counts as a similar shape

Do not compare absolute GreekMMLU accuracy across model sizes. Instead compare
changes from each model's own update-zero anchor and normalized token progress.
Predeclare the following diagnostics:

1. sign and magnitude of the pre-switch and post-switch slope for clean
   GreekMMLU choice NLL;
2. sign and magnitude of HPLT, GlossAPI and replay BPB slopes;
3. location of the best clean GreekMMLU NLL checkpoint as a fraction of total
   tokens;
4. the immediate one-checkpoint response to the HPLT-to-GlossAPI switch;
5. rank correlation of matched checkpoint deltas after allowing one affine
   rescaling of the smaller model's metric magnitude;
6. whether the same qualitative forgetting conclusion is reached.

A small scale is a viable *trajectory proxy* only if it agrees with 8B on the
direction of every primary slope, places the best checkpoint within 10
percentage points of normalized progress, and achieves at least 0.8 Spearman
correlation on both clean GreekMMLU NLL deltas and balanced-Greek BPB deltas.
Confidence intervals use paired-question bootstrap for GreekMMLU and
document-cluster bootstrap for heldout BPB.

## Learning-rate policy

Using `5.5e-5` blindly at all scales would test an absolute LR transfer, not
whether the smaller models can serve as well-tuned proxies. Conversely, tuning
the smaller LRs against the 8B GreekMMLU curve would be circular.

The policy is:

1. Keep the proven 8B recipe fixed at peak LR `5.5e-5`.
2. Treat the previously stable 0.5B peak LR `1.5e-4` as the preregistered first
   candidate, but revalidate it under the matched 13.5B batch geometry.
3. For 1.5B, run a short three-point LR calibration on one frozen training
   prefix. Center the candidates on a scale-derived prior from the model's
   original training LR; freeze exact numerical candidates before any result
   exists.
4. Select the 0.5B/1.5B LR using only source-conditioned Greek adaptation,
   replay retention, finite-update and gradient-stability gates. Do not inspect
   GreekMMLU during LR selection.
5. Reset to the exact Token-Distillation initialization and run the complete
   trajectory with the selected LR.

The LR *shape*, warmup-token mass, cooldown fraction and final-to-peak ratio
remain common. Only the peak magnitude may differ by model scale. LR pilots
are calibration jobs, not extra scientific arms.

## Stage 2: can a small model select the same arm?

Stage 1 is insufficient to authorize broad small-model screening. If at least
one smaller scale passes the trajectory-proxy gate, add one matched stationary
mixed control at every retained scale:

| Cell | Schedule |
|---|---|
| `C-MIX-0p5B` | stationary HPLT/GlossAPI mixture over the same 13.497B tokens |
| `C-MIX-1p5B` | identical schedule and token identities at 1.5B |
| `C-MIX-8B` | identical schedule and token identities at 8B |

This second arm answers the decision question. For each checkpoint and final
selector compute the treatment effect

`effect(scale) = score(H->G, scale) - score(Mixed, scale)`.

The smaller model is an *arm-selection proxy* only if the effect has the same
sign as 8B on clean GreekMMLU NLL, balanced-Greek BPB and the retention
constraint, and the smaller scale selects the same winner. The already
completed 80.7B-token 0.5B five-arm experiment is supporting prior evidence,
not a substitute: its hard transition occurred around 55.8B tokens rather
than 9.48B, so its first 13.5B tokens never saw the corresponding GlossAPI
phase.

## Execution order and stop rules

1. Finish Experiment A independently; do not reuse its different mixture as a
   control for this study.
2. Audit whether the historical 148,480-token data/init path is reconstructable
   and decide strict replication versus the common current 148,992-token
   production stack.
3. Freeze one cross-scale document, replay and checkpoint manifest.
4. Build and verify 0.5B tied and 1.5B/8B untied Token-Distillation initial
   checkpoints.
5. Run the non-GreekMMLU LR calibrations for 0.5B and 1.5B.
6. Launch the three Stage-1 HPLT-to-GlossAPI trajectories. Small cells may run
   concurrently on disjoint GPUs after per-scale restart parity passes.
7. Complete every checkpoint evaluation and apply the trajectory-proxy gate.
8. Only if a small scale passes, launch the matched Mixed controls required to
   test arm-selection predictivity.

No Stage-2 run is launched merely because a smaller model is faster. The
proxy gate is the scientific justification for using it to screen later data
mixtures.

## CSCS resource policy

All metadata inspection, model/config freezing, decontamination control,
receipts, checkpoint conversion and evaluation orchestration use one-node
`debug` jobs when they fit the 1.5-hour limit. Multi-node `normal` allocations
are reserved for training and the bounded distributed restart-parity smoke.

Each scale receives its own measured execution profile. Runtime optimizations
may differ only if they preserve that cell's scientific trajectory and pass
fixed-batch loss/gradient plus exact checkpoint/data-cursor restart gates.
Allocations are requested using measured end-to-end segment wall time, and
successors follow the audited scarce-allocation handoff rule. No allocation or
launch is authorized by this design document alone.
