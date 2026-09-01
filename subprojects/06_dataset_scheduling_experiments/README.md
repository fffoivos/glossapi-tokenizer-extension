# 06 — Dataset scheduling experiments

**Status:** prelaunch closure passed in dry-run mode. The five-arm data,
schedules, initialization, validation, LR, DP=16 systems geometry, evaluator
runtime and complete real-data five-arm load/resume path are frozen behind 16
passing gates. The campaign manifest is ready, but no production GPU job or run
root has been created; live launch still requires separate explicit
confirmation.

This subproject uses the base (not instruct)
`swiss-ai/Apertus-v1.1-0.5B` checkpoint to measure the effect of the temporal
order of HPLT versus aggregate GlossAPI/non-HPLT under one fixed WSD-10
learning-rate schedule. The complete scientific authority is
[`FACTORIAL_EXPERIMENT_DESIGN.md`](FACTORIAL_EXPERIMENT_DESIGN.md).

The model is pinned to Hugging Face revision
`1b7276176e564fc0cc7d7c3b991a8d653c8b8792`. The official
[model card](https://huggingface.co/swiss-ai/Apertus-v1.1-0.5B) identifies a
20-layer, 1024-dimensional, tied-embedding base model trained on about 1.7T
tokens. Its [release paper](https://arxiv.org/abs/2605.29128) reports
AdEMAMix/WSD and, for the 0.5B model, LR `6e-4`, GBS `512` sequences and
800,000 pretraining iterations. Those are source facts, not yet an automatic
choice of CPT peak LR.

## Frozen primary question

Every data-order arm consumes every eligible document from
`fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2` exactly once after
heldout and GreekMMLU decontamination exclusions. Identities, loss-active
tokens, fixed top-level pool permutations and replay positions are identical;
only the HPLT/GlossAPI temporal order changes.

| ID | Greek schedule |
|---|---|
| `D0_mixed` | stationary randomized natural mixture |
| `D1_hard_h_to_g` | all HPLT, then all aggregate GlossAPI/non-HPLT |
| `D2_hard_g_to_h` | exact hard reverse |
| `D3_gradual_h_to_g` | 128-window quota curve `g(u)=u^2.231741975...` |
| `D4_gradual_g_to_h` | exact mirrored window quotas |

Every schedule uses the same `L0_wsd10` trajectory: the cooldown is the same
`1-sqrt` shape and ends at 10% of peak LR. There is no LR branching and no
checkpoint averaging. The primary round therefore has exactly **five
optimization trajectories**, **five raw final endpoints**, **five full-run
equivalents**, and exactly **403,649,695,335 aggregate active tokens** under
the frozen post-exclusion receipt. A
future 10/20/30% LR-floor study is separate and out of scope here.

For this experiment, **GlossAPI means the complete eligible non-HPLT portion**
of the published dataset, selected by `source_dataset !~ ^HPLT/`. This is
deliberately broader than selecting only repositories whose names begin with
`glossAPI/`; otherwise OpenArchives, academic, historical and other integrated
sources could be silently dropped.

Replay remains stationary in every arm: 79% changing Modern Greek, 20% foreign
replay and 1% Old-Greek replay in deterministic small windows. The exact same
replay sequence IDs occupy the same cumulative-token positions in all data
arms.

## What “full run” means

The earlier published-corpus planning receipt reported:

- 63,822,761,532 published Greek training tokens before final exclusions;
- 44,054,228,362 HPLT tokens;
- 19,768,533,170 non-HPLT tokens.

The now-frozen post-exclusion pool receipt
`76658cc8495b58a3a3dadc8aca6d16c7fed627ef8bced89d17a877f9f9125014`
is the operative authority. It contains 44,042,201,419 HPLT tokens,
19,734,450,444 aggregate non-HPLT tokens, 16,145,987,813 foreign-replay
tokens and 807,299,391 Old-Greek tokens: 80,729,939,067 active tokens per arm.
The final packing receipt contains 19,709,692 real sequences; 260 loss-inactive
filler sequences align the global batch, freezing the horizon at 19,709,952
scheduled sequences and **38,496 optimizer updates per arm**. The resulting
2,024,325 token-slot overhead is only 0.0025% and contributes no loss.

Full coverage is an identity-level contract, not merely a capacity check. A
run is invalid if weighted sampling happens to omit an eligible Greek document,
repeats one in place of another, or allows heldout/decontaminated identities
back into training.

Packing uses source-local document order only to avoid pathological scattered
storage reads. The selected document set is still the exact seeded-prefix set.
After immutable 4096-token payloads are built, each pool receives one frozen
seeded SplitMix64 permutation of stable sequence IDs; that randomized catalog,
not the storage-access order, is the scientific pool order used by D0–D4. No
arm may repack a payload or alter its Goldfish mask.

## Tokenizer and initialization decision

The experiment will use our 148,992-token modern+polytonic Greek extension,
adapted safely to the Mini checkpoint. A compatibility audit found that Mini
and the production extension have the same 269,443-merge base prefix and that
the 17,920 appended merges are dependency-safe, but 14 reserved/special-token
IDs differ in the base vocabulary. Attaching the 8B-oriented tokenizer files
directly to Mini would therefore give some pretrained rows the wrong meaning.

[`build_mini_tokenizer_overlay.py`](initialization/build_mini_tokenizer_overlay.py)
preserves Mini IDs `0..131071` and its `tokenizer.json` front end, then
appends the exact production Greek tokens and merges at IDs `131072..148991`.
The resulting vocabulary is divisible by 256 with no alignment-only padding
rows. The exact audit
is in
[`tokenizer_compatibility_audit_20260801.json`](evidence/tokenizer_compatibility_audit_20260801.json).
The post-exclusion pool and packed-corpus receipts contain the completed fresh
counts under this exact overlay.

The audit also found a pinned Mini metadata inconsistency: model
`pad_token_id=3` points to `[INST]`, while the vocabulary already contains
`<pad>` at ID 10 and the tokenizer sidecars declare no pad token. The overlay
does not change any ID or merge; it declares the existing ID 10 token as pad,
and the resized model config is reconciled to ID 10. Fixed-length packed CPT is
not padding-dependent, but TD batching and evaluation must not treat `[INST]`
as padding.

Mini uses one tied 1024-dimensional input/output embedding table, so no 8B
embedding rows may be reused. The initialization path is:

1. canonical FVT/subtoken-mean pre-initialization in
   [`build_tied_retok_init.py`](initialization/build_tied_retok_init.py);
2. a four-cell, shared 1,024-token Token Distillation pilot: target layer 7
   versus the last layer, crossed with MSE-on-hiddens alone versus MSE plus the
   upstream `CE-auto-weighted` next-token term;
3. evaluate the untouched tied-FVT initialization on the same three heldout
   slices, then select one layer/loss recipe only if its macro BPB does not
   regress against that baseline;
4. run the selected recipe once over the complete requested added
   range, and retain deterministic FVT rows for any token lacking 25 valid
   contexts;
5. select the full artifact using heldout BPB, row preservation, tied-storage,
   finite/norm and generation-collapse gates; use that same artifact in every
   data-order arm.

[`verify_tied_initialization.py`](initialization/verify_tied_initialization.py)
implements the independent structural, norm, teacher-forced argmax and greedy
generation-collapse checks.

The NTP variant is included because the Token Distillation paper reports a
large-norm generation-collapse failure on a tied-embedding model and presents
the weighted NTP term as the relevant safeguard. It backpropagates through the
same shared table; the separate output-only CE update remains disabled.

## Shared training and evaluation controls

All five trajectories share every non-factor control:

- the exact base-model weight hash and conversion receipt;
- sequence length 4096 and the checkpoint's native RoPE geometry
  (`theta=500000`, default/no scaling);
- a common global batch, optimizer, complete WSD-10 LR trajectory, precision,
  loss objective, seed family and checkpoint cadence;
- the exact same eligible document manifest and replay identity manifest;
- fast evaluation every 512 steps (approximately 1.074B tokens), immediately
  around hard transitions and at matched-token controls, at cooldown start and
  at every raw final endpoint;
- HPLT, non-HPLT/GlossAPI, historical/polytonic, foreign-language, code and
  math evaluations, plus the frozen natively authored GreekMMLU zero-shot
  evaluator at every required evaluation checkpoint in every arm (accuracy,
  choice NLL and correct-answer BPB; not endpoint-only).

The 512-step evaluation cadence uses asynchronous full-state checkpoints so
GreekMMLU always scores an exact, receipt-bound model state. A non-boundary
payload may be pruned only after its GreekMMLU and heldout-evaluation receipts
are frozen; transition, resume and final checkpoints remain retained.

The frozen 38,496-update schedule currently yields **83 required checkpoints
per arm and 415 native-GreekMMLU checkpoint evaluations** across D0–D4. This
includes the initial state, warmup, every 512 steps, matched hard-transition
controls, cooldown start and raw final endpoint. A B2-selected segment boundary
is frozen at update 19,456; because it is already a regular 512-step point, it
does not increase the 83-per-arm count. That exact restart checkpoint is also a
mandatory native-GreekMMLU evaluation point.

Regular 512-step saves use Megatron's native `--save-interval`. The scheduled
entrypoint installs a narrow trigger hook for non-periodic points (warmup,
hard-transition controls, cooldown and any segment boundary) from the frozen
`MINI_SCHEDULE_SAVE_ITERATIONS` list. The hook invokes Megatron's own
asynchronous `torch_dist` save path at the exact optimizer iteration; it does
not implement a second checkpoint format.

Training checkpoints are SwissAI Megatron `torch_dist`, while the frozen
GreekMMLU evaluator consumes Hugging Face Apertus checkpoints. The required
path is therefore the canonical SwissAI conversion chain:
`scripts/conversion/torchdist_2_torch.py`, then
`tools/checkpoint/convert.py --loader core --saver swissai_hf` with the frozen
extended tokenizer. Every export receipt must bind the source checkpoint hash
and iteration to the HF tree hash. Before the campaign, one converted
checkpoint must pass the converter's logit-equivalence check and the complete
native-GreekMMLU evaluator. Conversion and evaluation run on separate nodes so
they do not pause training.

The complete bridge was smoke-tested on 2026-08-02 using exact iteration 48
from a runtime-only mock-data checkpoint. Canonical conversion achieved 99.99%
prediction agreement and 99.88% close logits, then all 16,632 native
GreekMMLU questions were scored and bound to the source-checkpoint and HF-tree
hashes in a final receipt. Those benchmark values are not scientific model
results—the checkpoint used FVT initialization and mock training data—but the
conversion and evaluator path is proven. See
`evidence/exact_checkpoint_native_greekmmlu_smoke_20260802.json`.
The production dataset/schedule/checkpoint-plan receipt is
`evidence/dataset_schedule_and_native_greekmmlu_plan_20260802.json`.
That receipt records the previously frozen plan and its then-current matrix
hash. Subsequent fail-closed launch-policy additions changed the matrix hash
without changing the scientific cadence. The historical plan is therefore
evidence of the 83/415 cadence, but is no longer launch-authorizing. Before the
campaign closes, the small checkpoint-plan JSON must be regenerated from the
same frozen schedule and the current unauthorized matrix. The launch gate now
checks those exact input paths, byte sizes and hashes, as well as every
checkpoint-row semantic; it no longer accepts a stale plan merely because its
old hash was compiled into the code.

The corpus verifier distinguishes document-cluster IDs from exact record
identities. Some source IDs legitimately name multiple distinct text records;
coverage therefore uses `(cluster_id, text_sha256)`. HPLT and
GlossAPI/non-HPLT must be globally exact-content unique. Replay preserves the
original training-source records—including measured repeated content—because
silently deduplicating replay would change its distribution. Its duplicate
rate is receipted, and the same replay sequence IDs remain fixed at the same
positions in every arm.

The execution policy is deliberately conventional and optimized for wall-clock
speed. B2 selected ordinary DP=16 per arm: four four-GPU nodes per arm and 20
training nodes for all five concurrent arms, with microbatch 4 and eight-way
gradient accumulation preserving the exact 512-sequence global batch. The
five-arm contention receipt projects 19.9979 hours end to end for the
controlling arm, including checkpoint and two-segment allowances. Existing
Apertus fused kernels, bf16,
distributed-optimizer overlap, fixed-length packing, input prefetch,
asynchronous distributed checkpoints and disabling activation recomputation
when memory-safe are in scope. Multi-model GPU colocation, CUDA MPS, `vmap`,
custom grouped GEMMs, FP8 and new CUDA/Triton kernels remain explicit non-goals.
The measured plan satisfies the at-most-24-hour training and less-than-36-hour
complete-round targets; see
[`RUNTIME_SCALING_36H_PLAN.md`](RUNTIME_SCALING_36H_PLAN.md).

The frozen CPT peak LR is `1.5e-4`, with WSD-10 floor `1.5e-5`. The first
candidate, `3e-4`, passed finite-loss, gradient, checkpoint, added-token and
catastrophic-regression checks but failed the predeclared retention-panel
non-inferiority gate. The identical 1,024-step smoke passed every gate at the
predeclared `1.5e-4` fallback. The warmup is 800 Mini steps, preserving the 8B
recipe's approximately 1.678B warmup-token mass at Mini's 2,097,152-token
global batch. The complete WSD-10 trajectory is identical in all five arms.

RoPE is not ratio-scaled: the checkpoint-native geometry is frozen at sequence
length/max position 4096, `rope_theta=500000`, default RoPE and no scaling.
Likewise the native global batch is 512 sequences. Dimensionless tested
controls transfer unchanged: AdEMAMix `(beta1,beta2,beta3,alpha) =
(0.9,0.999,0.999,4)`, weight decay 0.1, gradient clipping 0.1 and Goldfish
`k=h=50`. See
[`INITIALIZATION_AND_TRAINING_DECISIONS.md`](INITIALIZATION_AND_TRAINING_DECISIONS.md)
for the rationale and exact gate table.

The primary decision considers source-conditioned Greek quality and forgetting
together. A schedule is not a winner merely because it ends with low loss on
the last-seen pool. Apply predeclared retention constraints, then prioritize
neutral-external-Greek BPB, balanced HPLT/GlossAPI relative gain and the macro
GlossAPI-family result. GreekMMLU, Belebele and DemosQA confirm rather than
select the 0.5B winner by themselves.

The prior target-directory installation used by the general retention suite
was no longer structurally usable: its `lm_eval` source and dependency files
had been stripped. The replacement is therefore an explicit reconstruction,
not a false claim of byte identity: official `lm-eval==0.4.11`, the last-known
exact dependency-version inventory, and the shared pinned PyTorch 2.9.1 uenv
are all file- and version-receipted. Because official 0.4.11 exposes 15
language-level Global-MMLU-Lite groups but not the old command's top-level
`global_mmlu` label, the runtime freezes a transparent alias aggregating those
15 groups. All five endpoints use this one runtime, so comparisons within the
new round remain exact; comparisons to any old reported retention values must
be labelled evaluator-reconstructed rather than byte-identical.

## Files

- [`configs/experiment_matrix.json`](configs/experiment_matrix.json) —
  machine-readable five-arm controls, schedules, planning arithmetic and
  launch gates.
- [`FACTORIAL_EXPERIMENT_DESIGN.md`](FACTORIAL_EXPERIMENT_DESIGN.md) —
  five-order design, exact quotas, evaluation and selection.
- [`RUNTIME_SCALING_36H_PLAN.md`](RUNTIME_SCALING_36H_PLAN.md) —
  accepted DP=16 B1/B2 evidence, two-segment launch shape and the sub-36-hour
  production policy.
- [`evidence/clariden_capacity_probe_20260802.md`](evidence/clariden_capacity_probe_20260802.md)
  — non-submitting live scheduler-capacity receipt.
- [`evidence/apertus_v1_1_0_5b_snapshot.json`](evidence/apertus_v1_1_0_5b_snapshot.json)
  — pinned model/release-paper facts used by the design.
- [`evidence/token_distillation_upstream_snapshot_20260801.json`](evidence/token_distillation_upstream_snapshot_20260801.json)
  — pinned upstream commit and byte-exact vendored-code hashes.
- [`scripts/validate_experiment_matrix.py`](scripts/validate_experiment_matrix.py)
  — dependency-free invariant checker.
- [`INITIALIZATION_AND_TRAINING_DECISIONS.md`](INITIALIZATION_AND_TRAINING_DECISIONS.md)
  — tokenizer, tied-TD, LR, RoPE and batch-transfer decisions with evidence.
- [`initialization/`](initialization/) — exact tokenizer overlay, merge-DAG
  decomposition, tied FVT/TD construction, coverage normalization and
  independent collapse verification scripts.
- [`tests/test_experiment_matrix.py`](tests/test_experiment_matrix.py) — tests
  that reject schedule leakage, coverage drift and planning-math drift.
- [`../05_token_distillation_cpt/CPT_LAUNCH_RESOURCE_SPEC_20260801.md`](../05_token_distillation_cpt/CPT_LAUNCH_RESOURCE_SPEC_20260801.md)
  — production data/replay/tokenizer evidence map.

## Validate locally

These checks are lightweight and safe on the MacBook:

```bash
python3 scripts/validate_experiment_matrix.py
python3 -m unittest discover -s tests -v
```

## Launch status

All 16 declared semantic gates have passed. The authoritative frozen manifest
is
`/iopsstor/scratch/cscs/fffoivos/orchestration/dataset-scheduling-0p5b/20260803T074000Z-final-prelaunch-closure-v2/campaign_manifest.json`
(SHA-256 `85311d99f0f8a0b901d6c0895cdd00f7696ab5af8cf10e3fda2dd5b657f8cc02`).
The prelaunch closure receipt has status `passed_dry_run_only` and records zero
GPU jobs at closure time. See
[`final_prelaunch_closure_20260803.json`](evidence/final_prelaunch_closure_20260803.json).

The user authorized the live five-arm launch on 2026-08-03. Campaign
`mini_cpt5_20260803T074854Z` was submitted at
`2026-08-03T07:48:55.470786+00:00` with these initial Slurm jobs:

- initial validation: `2989297`;
- segment-0 five-arm training: `2989298` (`afterok:2989297`);
- checkpoint-evaluation watcher: `2989299` (`after:2989298`);
- segment supervisor: `2989300` (`after:2989298`).

The immutable run root is
`/capstor/scratch/cscs/fffoivos/runs/06_dataset_scheduling_experiments/mini_cpt5_20260803T074854Z`.
The campaign is self-advancing and receipt-gated: later segment attempts are
submitted only by the supervisor from a verified common five-arm checkpoint.
Initial validation job `2989297` completed successfully at
`2026-08-03T08:08:15Z` with exit code `0:0`; its receipt records all 13 panels
and is shared by all five arms. The dependency on training job `2989298` then
cleared, leaving it eligible and pending for 20 nodes/80 GPUs.
See
[`live_campaign_submission_20260803.json`](evidence/live_campaign_submission_20260803.json).

For a read-only live snapshot from the MacBook, stream the status collector to
Clariden without modifying the frozen scientific bundle or campaign run root:

```bash
ssh clariden \
  'RUN_ROOT=/capstor/scratch/cscs/fffoivos/runs/06_dataset_scheduling_experiments/mini_cpt5_20260803T074854Z bash -s' \
  < clariden/status_production_campaign.sh
```

The snapshot reports the submission graph, Slurm states, immutable receipts,
latest per-arm training/checkpoint iterations and matches for fatal or
non-finite diagnostics.
