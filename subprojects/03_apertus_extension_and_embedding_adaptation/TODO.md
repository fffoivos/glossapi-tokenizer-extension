# TODO

## Current active items (2026-05-23)

- Treat [`PRODUCTION_DECISION_STATE.md`](PRODUCTION_DECISION_STATE.md) as the
  current production-path state. The 2B Vanilla / ReTok / Centroid bakeoff has
  completed; Vanilla is the safe default for the 15-20B CPT path, Centroid is
  eliminated, and ReTok is not selected as-is.
- Token Distillation is now an active bounded challenger, not just a
  parallel-ready plan. Coverage prepass, smoke, layer pilot, and pilot intrinsic
  eval have run; full-token TD job `2353960` is in progress. Track live state in
  [`RUN_LOG_20260523.md`](03_4_implementation_experiments/init_bakeoff/token_distillation/RUN_LOG_20260523.md).
- Keep CPU-only dataset/snippet/preservation work on `xfer`. The queued
  preservation checks are jobs `2355706` and `2355707`, dependent on full TD
  completion.
- After preservation passes, run/read packed full-token intrinsic eval job
  `2355714`; only then decide whether `retok_td` is worth the R17 conversion
  gate.

## Historical TODO

- tokenizer extension candidates are now frozen for the current bakeoff;
- embedding resize, initialization, warmup, and full CPT are planned in
  `cpt_plan.md`, `03_4_implementation_experiments/`, and the TD companion plan.

## Inputs available now (gathered ahead of execution)

- Apertus pretraining data inventory and Greek-share estimation plan:
  [`docs/APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md`](/home/foivos/Projects/glossapi-tokenizer-extension/docs/APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md).
  Replay-ratio decisions during embedding adaptation depend on the
  Greek-share number — execute §2.4 of that doc before fixing the
  multilingual replay ratio. Final Path-A result: Greek = **0.023 %**
  of Apertus-8B-2509 realised pretraining (3.11 B of 13.55 T).

- Apertus architecture choices that force cross-language embedding-norm
  convergence:
  [`docs/APERTUS_ARCHITECTURE_FOR_EMBEDDING_NORM_ANALYSIS.md`](/home/foivos/Projects/glossapi-tokenizer-extension/docs/APERTUS_ARCHITECTURE_FOR_EMBEDDING_NORM_ANALYSIS.md).
  Read before designing the CPT replay schedule and the
  forgetting-prophylaxis check — the architecture preserves multilingual
  breadth even under skewed data shares, but norm parity alone is not
  a quality signal (only a "training reached saturation plateau"
  signal). The CPT recipe should keep the same training-dynamics
  mechanisms (gradient clipping value, Pre-Norm + RMSNorm + QK-Norm,
  AdEMAMix) so the same convergence property holds for the new added
  Greek vocab.
