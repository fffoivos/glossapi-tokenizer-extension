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
- Materialize each dedup decision by `(doc_id, masked SHA-256)` and preserve its
  row multiplicity. `doc_id` alone is not a physical-row identity: v7 found 27
  repeated IDs spanning 32 additional records in one shard. A set of document
  IDs could therefore drop the wrong text or every copy. The v2 receipt keeps
  one selected row per masked-text group and the shard gate consumes exactly
  the recorded number of matching occurrences.
- Within a safe duplicate group, prefer the quota-limited Old-Greek replay row
  and otherwise preserve the legacy lowest-task-index/document-ID ordering.
  This transfers ownership only: every exact text still survives once. The
  original task-index-only rule left 11,529,074 Old-Greek tokens and failed the
  1% capacity gate; the source has 2,666,110,500 pre-sanitization tokens.
- A completed prior masked inventory may be reused only when the canonical task
  contract and masker SHA-256 are identical. Every Modern-Greek task is rebuilt
  when its eligibility policy changes. A dedup-policy-only retry may promote
  all inventories only when the receipt closes to all 1,457 tasks; binary
  training shards are built anew against the new dedup receipt.
- Recompute the complete D0 79/20/1 schedule from the exact retained Modern
  Greek token mass. Consume the eligible sanitized Modern-Greek pool once and
  derive replay quotas and the optimizer horizon from the receipt; do not carry
  forward the old 19,248-update constant.
- The sanitized bridge itself must prove sufficient foreign and Old-Greek
  capacity under exact integer-nearest 79/20/1 arithmetic. The launch gate
  rejects a bridge without that passed capacity receipt.

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

Before this restart, forgetting alerts are pre-registered on the fixed English,
German, Russian, Chinese, code, math and Greek-replay panels. Relative to each
panel's post-warmup running minimum, two consecutive observations at +0.05 nats
raise a warning; +0.08 on any panel or +0.05 on the macro mean raises a critical
endpoint-selection flag. These are reporting constraints, not automatic stop
rules: the earlier trajectory established drift but no validated mid-run
intervention, and cooldown recovery must remain measurable.

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
- When an all-compatible promotion receipt accounts for every inventory task,
  the redundant resume-only inventory array may be bypassed; post-mask dedup
  independently revalidates every promoted manifest and payload. Do not use
  this shortcut when any task is marked changed.
- Reserve at least 45 minutes for the Old-Greek-priority post-mask dedup pass.
  The distributed per-task drop-ledger tail is materially slower than v6's two
  concentrated Old-Greek ledgers; the checked-in default is one hour. Forecast
  this step from sorted-catalog read progress, not drop-ledger byte growth.
- Treat a shard accounting failure as an artifact-contract failure, not as a
  retryable scheduler event. Preserve completed outputs, diagnose the exact
  identity mismatch, and start a new stage when the dedup receipt schema or
  contents change; never weaken a row-closure assertion to make the shard pass.
