# TODO

## Current active items (2026-05-23)

- Treat [`PRODUCTION_DECISION_STATE.md`](PRODUCTION_DECISION_STATE.md) as the
  current production-path state. The 2B Vanilla / ReTok / Centroid bakeoff has
  completed; Vanilla is the safe default for the 15-20B CPT path, Centroid is
  eliminated, and ReTok is not selected as-is.
- Prepare Token Distillation as a parallel-ready follow-up using
  [`TOKEN_DISTILLATION_PLAN.md`](TOKEN_DISTILLATION_PLAN.md). The intended
  candidate is `retok_td`: ReTok initialization plus embedding-only TD, with
  explicit handling for Apertus's untied input/output embeddings.
- Before any GPU TD compute run, execute the CPU-only firing/coverage prepass
  on `xfer`:
  [`03_4_implementation_experiments/init_bakeoff/token_distillation/`](03_4_implementation_experiments/init_bakeoff/token_distillation/).
  Do not use GPU nodes for dataset/snippet work.
- Before any TD model update, implement the exact-tokenizer adapter so the
  production extended BPE tokenizer and token IDs are preserved. Do not use a
  base-tokenizer `add_tokens(...)` path for production artifacts.

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
