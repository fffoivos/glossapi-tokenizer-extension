# TODO

- do not execute yet
- wait until tokenizer extension candidates are frozen
- then plan embedding resize, initialization, warmup, and full CPT in detail

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

