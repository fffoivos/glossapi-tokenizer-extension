# Five-arm data-order experiment

**Status:** designed; data materialization and GPU launch are not authorized.

This is the experiment authority for the five data-order schedules under one
fixed WSD-10 learning-rate trajectory. The machine-readable contract is
[`configs/experiment_matrix.json`](configs/experiment_matrix.json). Token
counts in this document are planning values under the production tokenizer,
before the final heldout/decontamination exclusions. All schedule boundaries
must be regenerated after the Mini-overlay recount.

## 1. Number of experiments

The primary round has five data-order levels and one fixed LR schedule:

| Data order | Fixed LR |
|---|---|
| `D0_mixed` | `L0_wsd10` |
| `D1_hard_h_to_g` | `L0_wsd10` |
| `D2_hard_g_to_h` | `L0_wsd10` |
| `D3_gradual_h_to_g` | `L0_wsd10` |
| `D4_gradual_g_to_h` | `L0_wsd10` |

This means:

- **5 optimization trajectories**;
- **1 raw final endpoint per trajectory**;
- **5 evaluated endpoint artifacts**;
- **5.0 full-run equivalents**;
- exactly **403,649,695,335 aggregate active tokens** under the frozen
  post-exclusion pool receipt.

There is no LR branching and no checkpoint averaging. A possible WSD-10/20/30
floor experiment is a later, separate study and must not be folded into this
primary round.

Token Distillation pilots, the common LR smoke, evaluation jobs and possible
confirmation seeds are prerequisite or follow-up work and are not included in
these counts.

## 2. Exact data invariant

Every data-order arm must consume the same post-exclusion loss-active token and
document-identity multiset exactly once. The following are frozen across all
five arms:

- eligible exact-record identity set `(source_document_cluster_id,
  text_sha256)` with `source_dataset` retained as provenance;
- HPLT, aggregate non-HPLT and replay packed-sequence manifests;
- tokenizer, packing, cross-document mask, EOD mask and position resets;
- one deterministic identity-hash permutation for each top-level Greek pool;
- replay sequence IDs and their global token positions;
- total HPLT, non-HPLT, foreign-replay and Old-Greek tokens.

Source-local document ordering is an I/O optimization, not a scientific
schedule. The exact seeded-prefix document set is selected first, then those
selected rows are reordered by source task and document index solely for
packing reads. Once immutable sequence payloads exist, a frozen seeded
SplitMix64 permutation of stable sequence IDs defines each pool's randomized
scientific order. All five arms consume those same payloads and may differ only
in how the four frozen sequence catalogs are interleaved.

GlossAPI/non-HPLT is one aggregate randomized pool. `source_dataset` must not
be used as an ordering key inside it. There is no internal academic, legal,
book or other source curriculum in this experiment.

Goldfish makes the packing invariant stronger than a document-count invariant.
The pinned implementation hashes each target's preceding label context inside
its 4096-token sample; repacking the same documents after reordering could
therefore change which token targets contribute loss. Pack HPLT, aggregate
GlossAPI and each replay pool independently into immutable sequence payloads
before scheduling. Give every payload a stable sequence ID and Goldfish-mask
hash. D0–D4 may only interleave those IDs; they may not repack or change a
sequence's token context. The receipt must prove the same per-sequence mask
bitmap and hence the same loss-active token multiset in every arm. Keep the
frozen `k=h=50`, table size 1,000,003 and seed 2,971,215,073, and rerun the
added-token Goldfish-uniformity gate for the Mini extension.

The frozen post-exclusion pool counts are:

| Pool | Tokens | Fraction of Modern Greek | Fraction of total run |
|---|---:|---:|---:|
| HPLT | 44,042,201,419 | 69.0569356% | 54.5549791% |
| GlossAPI/non-HPLT | 19,734,450,444 | 30.9430644% | 24.4450209% |
| Foreign replay | 16,145,987,813 | — | 20.0000000% (integer-rounded) |
| Old-Greek replay | 807,299,391 | — | 1.0000000% (integer-rounded) |

The operative total is 80,729,939,067 active tokens. The source is the frozen
post-exclusion pool receipt with SHA-256
`76658cc8495b58a3a3dadc8aca6d16c7fed627ef8bced89d17a877f9f9125014`.
Replay is enforced by deterministic
small-window quotas, not unconstrained weighted sampling. The selected replay
manifest is consumed without replacement and must be identical in all arms.

## 3. Five data-order schedules

Let `u` be normalized total-token progress, `q_G = 0.3094306437784159...`,
`q_H = 1-q_G`, and
`a = 1/q_G-1 = 2.2317419754848279...`. Replay occupies 21% of every scheduling
window; `g(u)` controls the GlossAPI fraction only inside the remaining 79%.

### D0: stationary mixed

Every window targets 54.5549791% HPLT, 24.4450209% GlossAPI, 20% foreign
replay and 1% Old-Greek replay, subject only to deterministic integer-sequence
rounding. Window quotas prevent incidental long-range mixture drift.

### D1 and D2: hard mirror schedules

D1 consumes HPLT with replay, then GlossAPI with replay. Its switch is at
69.0569356% progress, approximately 55.749622B total tokens. D2 consumes the
exact reverse and switches at 30.9430644%, approximately 24.980317B total
tokens. Optimizer,
scheduler and global iteration never reset at either transition.

Save checkpoints immediately before the switch and after the first complete
optimizer update in the new phase. Save matched-token checkpoints in every
other arm so comparisons do not depend on unequal training progress.

### D3 and D4: gradual mirror schedules

D3 uses `g(u)=u^a`; D4 uses `g(u)=(1-u)^a`. Both integrate to `q_G`, so each
consumes the exact HPLT and GlossAPI totals. Implement them with 128 equal
loss-active-token windows.

For D3 window `j` of `W=128`, use the integrated rather than midpoint curve:

```text
mean_g_j = W * [((j+1)/W)^(a+1) - (j/W)^(a+1)] / (a+1)
G_j      = modern_tokens_j * mean_g_j
H_j      = modern_tokens_j - G_j
```

D4 uses the D3 window quotas in reverse order. Convert fractional quotas to
integer token/sequence budgets with a deterministic largest-remainder rule,
carry sequence-granularity residuals forward, and correct late windows so both
fixed pool manifests are exhausted exactly once. The receipt must show per
window requested versus realized tokens, cumulative residuals, zero missing or
duplicate sequence IDs, and exact terminal pool totals.

## 4. Fixed learning-rate schedule

The common stability smoke invoked the predeclared fallback: `3e-4` passed the
finite-loss, gradient, checkpoint and catastrophic-regression checks but failed
the retention-panel non-inferiority gate. Repeating the same smoke at `1.5e-4`
passed every check, so **`1.5e-4` is the frozen common peak** and `1.5e-5` the
WSD-10 floor for all five arms. Every arm has the same initial LR, 800-step
warmup, peak plateau and final 20% `1-sqrt` cooldown.

`L0_wsd10` ends at `0.10 * peak`. No arm branches at 80%, and neither optimizer
nor scheduler state resets at a data boundary.

## 5. Endpoint policy

Evaluate only the raw final checkpoint from each optimization trajectory.
Checkpoint averaging, SMA, EMA and model-weight interpolation are outside this
experiment. Intermediate checkpoints remain recovery and trajectory-evaluation
artifacts; they do not create additional endpoint treatments.

## 6. Evaluation design

Heldouts are split before packing at globally deduplicated document-cluster
level. Pages of one work, mirrors, HTML/PDF copies, cross-pool duplicates and
substantially overlapping editions must remain in one partition.

Freeze these source-conditioned panels before training:

1. HPLT heldout, stratified by host/quality/length/time/category where possible.
2. Aggregate GlossAPI/non-HPLT heldout.
3. Per-source or predeclared GlossAPI-family heldouts.
4. Neutral external Modern Greek absent from both training pools.
5. Per-language foreign-replay heldouts.
6. Old-Greek heldout.
7. Added-token-stratified heldout.

Start with 1–2M loss-active tokens for each fast top-level panel, 10–20M for
each full top-level panel, and 1–5M for each viable GlossAPI family. Increase
sizes until document-cluster bootstrap confidence intervals are smaller than
the pairwise BPB effect the screen must resolve.

Evaluate the fast panel at the initial checkpoint, after warmup, every 512
steps (about 1.074B tokens), around both hard switches and their matched-token
controls and at cooldown start. Evaluate the full panel on each raw final
endpoint.

For every source report token NLL and bits per UTF-8 byte, separately for base-
token and new-token targets. Also report:

```text
natural Greek loss  = q_H * L_H + q_G * L_G
balanced Greek loss = 0.5 * L_H + 0.5 * macro_mean(L_GlossAPI_families)
relative gain_s     = (L_s(base) - L_s(model)) / L_s(base)
forgetting_s        = L_s(final) - min_t L_s(t)
```

Never collapse the source panels into only one selector number.

## 7. Benchmarks

GreekMMLU is natively authored in Greek and is a mandatory trajectory
evaluation, although it is confirmatory rather than the primary 0.5B selector.
Run the frozen zero-shot evaluator at **every required evaluation checkpoint in
every arm**, not only at final endpoints. Freeze its harness commit, prompts,
examples, answer order, normalization, tokenizer, context and scoring code.
Report zero-shot accuracy, multiple-choice
cross-entropy from normalized answer scores, and correct-answer continuation
BPB. Run five-shot only on finalists. Report both the full public and frozen
decontaminated subsets.

For the frozen 38,496-update schedule, the current plan contains 83 checkpoints
per arm, hence 415 native-GreekMMLU evaluations. B2 froze the exact restart
boundary at update 19,456. It already coincides with a regular 512-step point,
so the count remains 83; the plan records both the cadence and segment-boundary
reasons on that checkpoint.

Materialize non-periodic checkpoints with the receipt-bound scheduled
entrypoint's exact-iteration trigger hook. It requests Megatron's native async
save at the declared iteration; it must not approximate a transition checkpoint
by polling logs and saving one update late.

The compact Greek suite is GreekMMLU, Greek Belebele and DemosQA (secondary).
The endpoint retention command preserves the prior replay-experiment suite:
MMLU, ARC-Easy, ARC-Challenge, HellaSwag, WinoGrande, PIQA, Global-MMLU-Lite,
XNLI and XCOPA. The official lm-eval 0.4.11 release has 15 language-level
Global-MMLU-Lite groups but no top-level `global_mmlu` alias, so the frozen
runtime defines that alias explicitly as their aggregate. Per-language replay
BPB remains a separate continuous retention signal. Continuous NLL or BPB
accompanies accuracy for all near-floor tasks.

## 8. Predeclared selection

Before observing endpoints, derive foreign, Old-Greek and general-capability
non-inferiority margins from checkpoint/seed variability. Reject configurations
that violate them. Among passing configurations, select by:

1. lowest neutral-external-Greek BPB;
2. best balanced HPLT/GlossAPI relative gain;
3. best macro GlossAPI-family result;
4. confirmation by GreekMMLU continuous metrics, accuracy, Belebele and DemosQA.

Use document-cluster bootstrap intervals for BPB, paired-question bootstrap for
benchmark NLL, and paired bootstrap or McNemar intervals for accuracy. One seed
screens all five arms. If differences are not clearly larger than their
intervals, repeat D0 and the best one or two curricula at the selected LR with
two additional source-order seeds. That follow-up adds
four to six trajectories, not another full screen.

## 9. Evidence map

- [Curriculum/LR paper](https://arxiv.org/html/2511.18903): motivation for
  testing how the surviving learning rate interacts with curriculum order.
- [Signal and Noise](https://arxiv.org/html/2508.13144v1): continuous
  loss/BPB-style metrics for higher-signal small-model comparisons.
- [GreekMMLU](https://arxiv.org/pdf/2602.05150): native Greek benchmark and
  official multiple-choice evaluation context.
- [OLMES](https://arxiv.org/html/2406.08446v2): frozen prompt, normalization
  and multiple-choice evaluator controls.
- [Belebele](https://arxiv.org/html/2308.16884v2): multilingual
  passage-grounded comprehension.
- [DemosQA](https://arxiv.org/pdf/2602.16811): informal/community Greek QA.
- [Pinned SwissAI Megatron Goldfish implementation](https://github.com/swiss-ai/Megatron-LM/blob/c92402e39ef3c8e69ea378a59e79059dc14541f4/megatron/core/datasets/gpt_dataset.py):
  sequence-context hash mask, table size and seed used by the project recipe.

## 10. Infrastructure boundary and launch gates

The runtime uses ordinary data parallelism, one model and process per GPU and
five disjoint equal-size groups. The accepted B2 geometry is DP=16 per arm,
four nodes per arm and 20 training nodes total, with microbatch 4 and gradient
accumulation 8. The geometry is identical in all arms.
CUDA MPS, multi-model GPU colocation, `vmap`, FP8 and custom CUDA/Triton kernels
remain forbidden. See
[`RUNTIME_SCALING_36H_PLAN.md`](RUNTIME_SCALING_36H_PLAN.md).

No production arm may launch until the Mini tokenizer/TD checkpoint, final
post-exclusion pool and replay manifests, source-conditioned validation splits,
LR smoke, exact schedule and per-sequence Goldfish-mask receipts, added-token
Goldfish-uniformity gate and checkpoint-resume smoke pass. The runtime B2 gate
is closed: its selected receipt forecasts 19.9979 hours end to end for training,
below both the 24-hour training target and the 36-hour complete-round ceiling.
