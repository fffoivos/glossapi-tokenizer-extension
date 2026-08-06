# Sanitized full-8B CPT restart contract

Status: implementation authority for the replacement D0 run. The stopped
pre-sanitization trajectory remains exploratory evidence only.

## Data contract

- Start from the frozen full v2 task inventory and preserve libduth.
- Exclude every `source_dataset == "openarchives.gr"` row whose
  `needs_ocr == true`. The independent audit requires exactly 6,648 excluded
  rows, zero retained matches, and the expected PII-token IDs.
- Apply the Apertus-parity email/IP masker and validated country-length IBAN
  masker to Modern Greek, foreign replay, and Old-Greek replay before
  tokenization.
- Globally exact-deduplicate the masked text. Training is also checked against
  the union of raw and masked representations of every frozen validation
  document.
- A completed v5 masked inventory may be reused only when the canonical task
  contract and masker SHA-256 are identical. Every Modern-Greek task is rebuilt
  because its eligibility policy changed. Binary training shards are all built
  anew.
- Recompute the complete D0 79/20/1 schedule from the exact retained Modern
  Greek token mass. Consume the eligible sanitized Modern-Greek pool once and
  derive replay quotas and the optimizer horizon from the receipt; do not carry
  forward the old 19,248-update constant.

## Scientific contract

The tokenizer, model geometry, Token-Distillation weights, optimizer, global
batch, sequence length, Goldfish mask, WSD-10 LR shape, and D0 document order
policy are unchanged. The existing initialization is disclosed accurately as
dropout-active Token Distillation (`model.train()`, attention and hidden dropout
0.1 during teacher-state extraction); it is preserved rather than rebuilt.

The derived recipe sets source-conditioned validation to every 238 updates
(about 1B training-token slots) plus exact saved-checkpoint milestones. It pins
GreekMMLU to revision `6a03aa06b68beb932fb75edff3a34e50b3674649`; every
receipt must also contain the resolved dataset fingerprint. Per-document
validation runs at initialization, cooldown start, and the terminal checkpoint.
Checkpoint averaging remains disabled.

## Operational contract

- Benchmark 288 updates with 32-update burn-in on the sanitized D0 prefix.
- DP64 is promoted only by the existing trajectory and performance gates.
- DP32 restart acceptance requires two independent restart allocations plus the
  real graceful-stop/resume smoke. The predeclared gradient tolerance and its
  post-hoc history remain disclosed in the launch receipt.
- Production uses five DP32 segments when the proven rate keeps a 4,000-update
  segment below the 12-hour allocation limit. All happy-path training segments,
  receipt supervisors, GreekMMLU queues, and the terminal evidence waiter are
  submitted up front. A failed segment cancels only its rejected queued suffix
  and switches to the bounded receipt-gated recovery chain.
- The finalizer reads initial evidence exclusively through the passed launch
  gate and derives all expected counts and terminal updates from the sanitized
  recipe.
