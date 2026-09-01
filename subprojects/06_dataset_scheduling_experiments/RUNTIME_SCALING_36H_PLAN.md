# Five-arm runtime scaling plan

Status: DP=16 real-scheduled-data, checkpoint/restart and five-arm contention
gates passed. DP=16 is selected for the first round.

## Objective and fixed scientific contract

The primary round contains exactly five optimization trajectories: `D0` through
`D4`. All five use the same fixed `L0_wsd10` learning-rate schedule. There is no
LR branching and no checkpoint averaging in this round.

The frozen post-exclusion horizon is 80,729,939,067 active tokens per arm, or
403,649,695,335 aggregate active tokens across the five arms. The final
packing/schedule receipt freezes 19,709,952 scheduled sequences and 38,496
optimizer updates per arm, including 260 loss-inactive global-batch filler
sequences. The global batch remains 512 sequences at length 4096 (2,097,152
scheduled token slots/update). Only execution
parallelism and conventional runtime options may be tuned. The selected runtime
geometry must then be frozen identically for all five arms.

The wall-clock objective is:

- training target: preferably at most 24 hours;
- complete first-round target: less than 36 hours;
- selected DP=16 path: finish training in two restart-safe `normal`-partition
  segments split at update 19,456, with an end-to-end training forecast of
  19.9979 hours for the controlling arm.

## Required measured throughput

For one exact 38,496-update arm:

| Training budget | Required arm throughput | Required step time |
|---|---:|---:|
| 36 hours | 622,916 active tokens/s | 3.367 s |
| 24 hours | 934,374 active tokens/s | 2.244 s |
| 12 hours | 1,868,749 active tokens/s | 1.122 s |

Every arm runs concurrently, so the per-arm requirement is also the campaign
critical path. The five-arm aggregate throughput must be five times the values
above.

## Strong-scaling ladder

Use TP=1 and PP=1 throughout. Increase ordinary data parallelism while keeping
the global batch at exactly 512 sequences.

| DP per arm | Nodes per arm | Sequences/GPU/update | Initial microbatch | Total nodes for five arms | Per-GPU throughput needed for 12 hours |
|---:|---:|---:|---:|---:|---:|
| 16 | 4 | 32 | 32, then 16/8 if needed | 20 | 116,881 tokens/s |
| 32 | 8 | 16 | 16, then 8 if needed | 40 | 58,441 tokens/s |
| 64 | 16 | 8 | 8 | 80 | 29,220 tokens/s |
| 128 | 32 | 4 | 4 | 160 | 14,610 tokens/s |

DP=128 is the deliberate ceiling. DP=256 would leave only two sequences per
GPU/update, consume 320 nodes for the five arms and is likely to be dominated by
gradient synchronization. It is not required to test the 36-hour objective.

The completed B2 gate selected DP=16 because it meets the wall-clock objective
with the smallest tested production allocation. A larger geometry is not part
of the first round unless a pre-launch capacity or performance recheck
invalidates this receipt.

## Benchmark procedure and accepted stopping rule

### B0: scheduler-capacity probe

Run non-submitting `sbatch --test-only` probes for 20, 40, 80 and 160 nodes at
the intended 12-hour limit. Immediately before the real launch, repeat the
probe because an estimate is not a reservation.

### B1: single-arm strong scaling

The original candidate ladder allowed DP=16, 32, 64 and 128. Start with the
smallest geometry, using one frozen representative training manifest and the
exact production model, tokenizer, initialization, optimizer and loss path.
Stop the ladder once a real-data B1 and five-arm B2 at the same geometry meet
the 24/36-hour objectives; additional nodes then add queue risk without being
needed to satisfy the stated speed constraint. For each geometry actually run:

1. run 256 optimizer steps;
2. discard the first 32 steps from throughput statistics;
3. exercise one fast validation interval and one asynchronous distributed
   checkpoint;
4. record median and p10 tokens/s, step-time distribution, peak memory,
   dataloader wait, NCCL time, checkpoint pause and any skipped iterations;
5. project end-to-end arm time from the measured interval, including the
   observed validation/checkpoint overhead.

The benchmark must use the same global batch. It is invalid if it changes the
number of tokens/update, sequence length, packing, Goldfish masks or optimizer
trajectory.

### B2: five-arm concurrency stress

At the fastest viable DP from B1, request the complete node count and run all
five arms simultaneously for at least 256 steps in disjoint node groups. Include
one simultaneous fast validation and one simultaneous asynchronous checkpoint.
The B1 result is not sufficient: the production decision uses the concurrent
receipt because filesystem, checkpoint and dataloader interference occur only
there.

Accept the geometry only when:

- every arm is numerically healthy and has zero skipped iterations;
- the slowest arm's projected training time fits the selected budget;
- the projected complete first round is below 36 hours;
- no arm has persistent dataloader starvation;
- asynchronous saves complete and reload successfully;
- all arms use exactly the same parallel and microbatch geometry.

## Production launch shape

Prefer one aggregate Slurm allocation and create five explicit, disjoint node
lists inside it. Launch one independent `srun --exclusive` step per data-order
arm. This guarantees that the five arms start together and prevents one arm from
waiting in the queue while the others advance.

Each arm has its own rendezvous endpoint, checkpoint directory, logs and failure
status. A failure in one arm must not silently terminate or invalidate the
others. Production uses the accepted B2 geometry:

- DP=16 per arm;
- four nodes / 16 GPUs per arm;
- 20 training nodes / 80 GPUs for all five arms;
- microbatch 4 and gradient accumulation 8 in every arm.

The exact-shape B0 memory screen found that TP=1 microbatches 32 and 16 OOM
while materializing the 148,992-way FP32 cross-entropy logits, and microbatch 8
exhausts memory during backward/gradient synchronization. Microbatch 4 is
therefore the no-recomputation baseline. Larger microbatches remain eligible
only with an established selective-recomputation setting and must beat the
microbatch-4 result after including their recomputation cost.

The `normal` partition currently limits jobs to 12 hours. Submit two
receipt-bound segments from complete distributed checkpoints, with the common
boundary at update 19,456. That update is already a regular 512-step checkpoint
and native-GreekMMLU evaluation point. The projected segments are approximately
10.00 and 9.79 compute hours before the small recorded segment allowance.

## Conventional speedups that remain in scope

- bf16 and the existing Apertus Transformer Engine/fused-kernel path;
- fused cross entropy, xIELU/RMSNorm/QK normalization and existing fused linear
  kernels;
- distributed optimizer with gradient-reduce and parameter-gather overlap;
- no activation recomputation when the selected microbatch fits safely, or
  selective recomputation only when required;
- immutable fixed-length packed samples, pinned memory and asynchronous
  prefetch;
- `torch_dist` asynchronous checkpoint save and fully parallel load;
- in-process fast validation without requiring a full checkpoint;
- asynchronous full-state checkpoints every 512 steps for native GreekMMLU,
  plus warmup, segment, hard-switch, matched-control, cooldown-start and final
  checkpoints.

Every checkpoint in the native-GreekMMLU trajectory must correspond to the
exact evaluated model state. Non-boundary 512-step payloads may be pruned only
after GreekMMLU and the validation-panel outputs have immutable receipts. The
B2 concurrency benchmark therefore includes simultaneous asynchronous saves
and evaluator reads; checkpoint pressure is part of the measured end-to-end
target rather than omitted from the forecast.

The evaluator does not silently treat a Megatron directory as an HF model.
Each `torch_dist` checkpoint is converted through the canonical SwissAI path
(`torchdist_2_torch.py`, then the `core` loader with `swissai_hf` saver), with
source iteration/hash and output-tree hash in the conversion receipt. A
pre-campaign conversion must pass logit equivalence.

The 2026-08-02 exact-checkpoint smoke measured the complete path. DP=16 with
microbatch 4 and gradient accumulation 8 sustained a 1.658 s median and 1.7391
s p90 after burn-in, or 17.73 compute-only hours per arm. An 8.223 GB async
save introduced a 23.344 s observable pause and completed successfully while
training continued; approximately 80 such pauses add about 31 minutes, giving
an indicative 18.26-hour arm before real-data and five-arm contention effects.
Canonical conversion plus its initial receipt work took 98 seconds, export
receipt finalization took 12 seconds, full native GreekMMLU took 200 seconds,
and the checkpoint-bound result receipt took 10 seconds. Conversion achieved
99.99% prediction agreement and 99.88% close logits.

At 1.658 s/update, regular checkpoints arrive every 512 updates, or every
849 seconds. A five-arm wave therefore creates 1,600 GPU-seconds of measured
conversion/evaluation/receipt service every 849 seconds: two continuously
available GPU lanes are the theoretical minimum, while one four-GPU evaluation
node supplies adequate headroom. Keep that node separate from the 20 training
nodes. Process the five simultaneous checkpoints as four concurrent lanes plus
one queued task; this completes a wave before the next cadence point. Named
warmup, transition, segment, cooldown and final checkpoints may add short
bursts, but do not change the steady-state requirement. Prune only each
non-boundary checkpoint whose GreekMMLU and heldout receipts have completed.

The concrete implementation is
`clariden/run_checkpoint_native_greekmmlu_wave.sbatch`. It follows CSCS's
documented node-sharing pattern: one exclusive four-GPU allocation, four
resource-isolated one-GPU `srun --exclusive` steps with 64 CPU cores and 105 GB
host memory each, then the fifth arm on the first freed lane. Each lane runs
canonical conversion, the pinned natively authored GreekMMLU evaluator, and
receipt finalization sequentially. This replaces the earlier nested one-GPU
batch jobs, which Slurm accounted as complete four-GPU nodes on Clariden.

The frozen 38,496-update schedule currently has 83 required checkpoint points
per arm (415 native-GreekMMLU evaluations total). The B2-selected update-19,456
restart boundary coincides with a regular cadence point, so the count remains
83. Its reasons must contain both `regular_512_step_cadence` and
`normal_partition_segment_boundary`.

## Accepted B2 receipt

Clariden job `2983767` ran all five exact schedule prefixes concurrently on 20
training nodes (80 GPUs) for 288 updates per arm while job `2983768` exercised
the separate four-lane native-GreekMMLU service. All arms completed with zero
skipped and zero NaN iterations and produced reloadable checkpoints at updates
120, 160 and 288. The controlling D3 projection was 19.7912 compute hours;
checkpoint and two-segment allowances produce the frozen 19.9979-hour training
forecast. The machine-readable evidence is
`/Users/foivoskarounos-zamparloukos/Projects/apertus-cscs-efficiency/evidence/mini_b2_five_arm_contention_20260802.json`.

## Explicitly out of scope

Do not use FP8, change the global batch or sequence length, place multiple models
on one GPU, use CUDA MPS, stack models with `vmap`, introduce custom grouped GEMM
kernels, add custom CUDA/Triton work, or share forward/optimizer state across
arms. These would either change training semantics or turn the scheduling study
into a systems experiment.
