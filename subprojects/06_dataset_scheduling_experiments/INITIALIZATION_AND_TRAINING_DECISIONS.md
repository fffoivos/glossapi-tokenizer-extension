# Mini initialization and training decisions

**Status:** implementation contract and operational choices frozen; tokenizer,
tied-TD, fresh token count and common-LR gates have passed.

This document answers which settings should follow the prior Apertus-8B CPT
experiments, which should be rescaled for `Apertus-v1.1-0.5B`, and which must
remain native to the Mini checkpoint. The machine-readable authority is
[`configs/experiment_matrix.json`](configs/experiment_matrix.json).

## Decision summary

| Setting | Mini experiment decision | Transfer rule |
|---|---:|---|
| Tokenizer | Mini base IDs `0..131071` + our exact Greek IDs `131072..148991` | preserve model/token semantics, append our merge chain |
| Embeddings | one tied `[148992, 1024]` table | Mini-native architecture |
| TD pilot | layers `7` and `-1`; each with MSE and MSE+auto-weighted CE | smallest useful tied-model bakeoff |
| TD LR | `1e-4`, one epoch, 25 snippets/token, batch 16 | canonical Token Distillation defaults already used by this project |
| CPT peak LR | `1.5e-4` selected by the common smoke; WSD-10 floor `1.5e-5` | predeclared fallback after `3e-4` failed retention non-inferiority |
| CPT LR | start at `0.1 * peak`; 800-step warmup; stable to 80%; `1-sqrt` cooldown to `0.1 * peak` | fixed WSD-10 schedule in every data-order arm |
| Endpoint policy | raw final checkpoint only; no SMA or EMA | checkpoint averaging excluded from this experiment |
| Global batch | 512 sequences = 2,097,152 tokens | Mini's native pretraining batch |
| Context/RoPE | 4096; theta 500,000; default; no scaling | exact Mini checkpoint geometry |
| Optimizer | AdEMAMix, betas `(0.9,0.999,0.999)`, alpha `4`, WD `0.1`, clip `0.1` | locally tested dimensionless controls |
| Loss | Goldfish `k=50,h=50` | Apertus/local CPT control |
| Parallelism | TP=1, PP=1, DP=16 per arm; four nodes/arm, 20 training nodes total | B2-selected ordinary data parallelism; one process and one model per GPU |
| Utilization | microbatch 4, accumulation 8, fused kernels, optimizer overlap, packing/prefetch and async distributed checkpoints | B2 projects 19.9979 hours in two restart-safe segments, below the 24/36-hour targets |

## 1. Tokenizer compatibility

The selected extension is
[`fffoivos/apertus-tokenizer-extension`](https://huggingface.co/fffoivos/apertus-tokenizer-extension)
at revision `fcd33ec09fb7d86bc072b3a4b3e890efa6473b66`, subfolder
`greek-modern-polytonic-tokenizer`. Its `tokenizer.json` SHA-256 is
`bbb08e71929b519c5c2362338b0fc6a0e99955cb8fdbf0729ae1311117e6561b`.
It contains 17,920 added units and reaches vocabulary size 148,992, exactly
`256 * 582`; there are no alignment-only padding rows.

It cannot be copied verbatim onto Mini. The pinned
[`swiss-ai/Apertus-v1.1-0.5B`](https://huggingface.co/swiss-ai/Apertus-v1.1-0.5B)
tokenizer and the production extension share the complete 269,443-entry base
merge prefix, but 14 base token IDs have different meanings and the tokenizer
front-end metadata differs. The exact mismatches are receipted in
[`evidence/tokenizer_compatibility_audit_20260801.json`](evidence/tokenizer_compatibility_audit_20260801.json).

The only permitted construction is therefore:

- retain the pinned Mini `tokenizer.json` and sidecar semantics for IDs
  `0..131071`;
- append the production extension's exact tokens and dependency-ordered merges
  at IDs `131072..148991`;
- reject base-merge drift, orphan operands, ID gaps, alignment-only rows or a vocabulary
  not divisible by 256.

The pinned Mini model config declares `pad_token_id=3`, but Mini token ID 3 is
`[INST]`; the existing `<pad>` token is ID 10 and its tokenizer sidecars do not
declare a pad token. This is an upstream metadata inconsistency, not a reason to
move vocabulary rows. The overlay declares the existing ID 10 token as pad,
and the ReTok model/generation configs are set to ID 10. No token ID, merge or
embedding meaning changes. TD batching and evaluation must assert this receipt.

[`initialization/build_mini_tokenizer_overlay.py`](initialization/build_mini_tokenizer_overlay.py)
implements these checks and emits `overlay_manifest.json`. Its output must pass
a Transformers load/save/reload round trip and ordinary-text equivalence on the
base-tokenizer side before it can be used. Because the front-end is Mini's, the
existing 63.823B production-tokenizer count remains planning evidence only; a
fresh count over the final train manifests is mandatory.

## 2. Tied Token Distillation

Mini has `tie_word_embeddings=true`, hidden size 1024 and a single shared input
and output table. The canonical implementation in
[`konstantinjdobler/token-distillation`](https://github.com/konstantinjdobler/token-distillation)
is pinned at commit `35702b5809599ecd68b7845eca27a0d7b7cec0da` and is already vendored under
subproject 03. Its tied path resizes one table, calls `tie_weights()`, and does
not create or separately train an output table.

[`initialization/build_tied_retok_init.py`](initialization/build_tied_retok_init.py)
first initializes each new row from the mean of the token's decomposition under
the pinned Mini vocabulary. The decomposition is derived from the exact
dependency-ordered appended merge DAG, not from decoding one ByteLevel token in
isolation, which can be lossy for partial multibyte fragments. Existing 131,072
rows must remain bitwise exact. Coverage produced by the reusable subproject-03
prepass is converted to the same exact geometry by
[`initialization/normalize_td_coverage_geometry.py`](initialization/normalize_td_coverage_geometry.py)
before a TD job can consume it.

The [Token Distillation paper](https://openreview.net/pdf/a0db5e51fc87ea60d721e19b0ed55984b4b0670a.pdf)
reports a tied-embedding failure mode in which an unbounded new-row norm causes
one new token to dominate generation. It also evaluates an automatically
weighted next-token-prediction term as the relevant tied-model mitigation.
Consequently, we do not assume that the successful untied 8B layer-11 artifact
transfers. We run these four bounded cells on the exact same stratified set of
1,024 added tokens on Mini:

| Cell | Target hidden state | Losses |
|---|---:|---|
| `layer7_mse` | 7, approximately one-third depth | MSE on hidden states |
| `layer7_mse_ce_auto` | 7 | MSE + `CE-auto-weighted` |
| `last_mse` | `-1`, paper-default final state | MSE on hidden states |
| `last_mse_ce_auto` | `-1` | MSE + `CE-auto-weighted` |

All cells use AdamW, LR `1e-4`, no weight decay, one epoch, batch 16 and 25
accepted contexts for every pilot token. In the CE-auto cells, CE gradients
flow through the same tied matrix. `learn_output_with_ce` stays false because a
second output-only pass would double-handle the same parameter.

After choosing the layer/loss recipe, run it once over the complete requested
range `131072..148991`. Tokens with fewer than 25 valid contexts retain their
deterministic FVT row; they are not trained on duplicated evidence merely to
make the coverage percentage read 100%. The full job requires at least 90% of
the requested rows to be TD-trained and receipts every fallback ID. If that
gate fails, collect better contexts or stop; do not silently lower it.

Before choosing that recipe, evaluate the untouched tied-FVT initialization
on the exact same HPLT, non-HPLT and polytonic slices. The winning TD cell must
have the lowest unweighted three-slice macro BPB among structurally valid cells
and must not regress that macro against tied FVT. If every TD cell is worse
than FVT, fail closed instead of selecting the least harmful cell.

[`initialization/train_tied_token_distillation.py`](initialization/train_tied_token_distillation.py)
enforces complete coverage of IDs `131072..148991`, exact preservation of every
other row, shared input/output storage, finite values, and a project safety gate
of `max(new-row norm) / p99.9(base-row norm) <= 4`. This norm threshold is an
explicit rejection gate, not a clipping transformation and not a claim from
the paper.

Selection also requires heldout HPLT, non-HPLT and polytonic BPB/NLL plus an
argmax/generation-collapse probe. Retention on non-Greek text is a tie-breaker.
One winning initialization hash is then frozen and used by every data-order
arm. If all four pilot cells fail, or if all regress macro BPB against tied
FVT, the full CPT experiment does not launch.
The independent structural and collapse checks are implemented in
[`initialization/verify_tied_initialization.py`](initialization/verify_tied_initialization.py).

## 3. Learning rate and schedule

We should not copy the 8B absolute peak LR `5.5e-5`. The prior project recipe
selected that value as one half of the 8B pretraining peak `1.1e-4`; the local
evidence map is
[`../05_token_distillation_cpt/CPT_LAUNCH_RESOURCE_SPEC_20260801.md`](../05_token_distillation_cpt/CPT_LAUNCH_RESOURCE_SPEC_20260801.md).
The Mini [release paper](https://arxiv.org/abs/2605.29128) reports pretraining LR
`6e-4`. Preserving the experimentally chosen dimensionless fraction gives:

```text
Mini CPT peak = Mini pretraining peak * tested CPT fraction
              = 6e-4 * 0.5
              = 3e-4
```

Using `5.5e-5` would instead be only 0.0917 times Mini's pretraining peak. The
`3e-4` value was therefore a reasonable candidate, but it remained an
extrapolation rather than a completed 0.5B CPT sweep. The common smoke passed
its finite-loss, gradient, checkpoint, added-token and catastrophic-regression
checks at `3e-4`, but failed the predeclared retention-panel non-inferiority
gate. The identical smoke then passed every gate at the predeclared `1.5e-4`
fallback. Consequently every arm uses peak `1.5e-4` and WSD-10 floor `1.5e-5`;
the peak is not an experimental variable.

The completed smoke was 1,024 Mini optimizer steps (2,147,483,648 tokens) on one frozen
diagnostic manifest balanced between HPLT and non-HPLT Greek, with the same
stationary replay policy. It uses the real 800-step warmup, leaving 224 steps at
peak LR; a shorter test would never exercise the candidate. It is a disposable LR
selection run: all full schedule arms restart from the same frozen TD
initialization. The selection receipt is
`/capstor/scratch/cscs/fffoivos/cpt_runs/dataset-scheduling-0p5b/20260803T110500Z-common-lr-fallback-v1/lr_selection.json`
(SHA-256 `39c74abd9f4cec8aa4dcda1c2daafca77245540f482be79fedf6e9cb309fa19d`).
The smoke did not choose between schedule arms.

The 8B production recipe used 400 warmup steps at 4,194,304 tokens/step:

```text
400 * 4,194,304 = 1,677,721,600 warmup tokens
1,677,721,600 / 2,097,152 Mini tokens/step = 800 Mini steps
```

Thus the common Mini schedule starts at `0.1 * selected_peak`, warms to the peak
over 800 steps, holds it through 80% of the final exact run horizon and uses the
same `1-sqrt` cooldown to `0.1 * peak` in every arm. AdEMAMix beta3 and alpha
warm over the complete final run and are not reset. Regenerate all step
boundaries after the fresh token count.

A possible 10%–30% LR-floor study motivated by [How Learning Rate Decay Wastes
Your Best Data in Curriculum-Based LLM
Pretraining](https://arxiv.org/html/2511.18903) is explicitly deferred. It is
not crossed with the five data-order arms in this primary round.

## 4. RoPE, context and architecture

RoPE is model geometry, not a dimensionless optimization setting. The pinned
Mini config specifies max position 4096, `rope_theta=500000`, default RoPE and
no scaling. Keep those values exactly. Do not copy the 8B post-long-context
theta, Llama-3 scaling, or any earlier corrected 8B geometry into this model.

Also keep Mini-native:

- 20 layers, hidden size 1024, MLP size 6144;
- 16 attention heads and 4 KV heads;
- xIELU, QK normalization and RMSNorm epsilon `1e-5`;
- tied embeddings and initializer range `0.02` (all new rows are overwritten by
  the explicit initialization above);
- no architecture, dropout, bias or context-length changes.

The model card/release snapshot is frozen in
[`evidence/apertus_v1_1_0_5b_snapshot.json`](evidence/apertus_v1_1_0_5b_snapshot.json).

## 5. Batch, infrastructure and checkpoint cadence

Use Mini's reported pretraining global batch: 512 sequences at length 4096, or
2,097,152 tokens per optimizer step. TP=1 and PP=1 are fixed for this 0.5B
model. B2 selected ordinary DP=16, microbatch 4 and gradient accumulation 8 per
arm. Each arm consumes four four-GPU nodes; all five concurrent arms consume 20
training nodes. This geometry is frozen identically for every arm.

This is an infrastructure optimization, not an additional research axis. The
permitted utilization techniques are bf16, the existing Apertus fused-kernel
path, fused cross entropy, distributed-optimizer communication overlap,
fixed-length sequence packing, pinned-memory/asynchronous input prefetch,
standard NCCL data parallelism, asynchronous `torch_dist` checkpoints and
disabling activation recomputation when the selected microbatch fits safely.
Do not add a population axis,
stack model parameters with `vmap`, colocate multiple training models on one
GPU with CUDA MPS, introduce grouped multi-model GEMMs, share forward or
optimizer state between models, or write new CUDA/Triton kernels for this
experiment.

The production decision is backed by a real-scheduled-data single-arm B1 and a
five-arm DP16 B2 concurrency benchmark. B2 checked stable throughput, NCCL
health, simultaneous checkpoint I/O and the four-lane GreekMMLU evaluation
service. Use one aggregate 20-node allocation with five explicit, disjoint
four-node groups. We do not recover throughput by sharing GPUs or changing
training semantics. See
[`RUNTIME_SCALING_36H_PLAN.md`](RUNTIME_SCALING_36H_PLAN.md).

Pack each top-level data pool into immutable 4096-token sample payloads before
constructing D0–D4. Goldfish hashes the label context inside each sample, so a
second repacking pass after schedule interleaving could change which targets
are loss-active even if the document multiset were unchanged. The schedules
therefore interleave stable sequence IDs only. Receipt each sequence payload
and its Goldfish mask hash, and require identical hashes in every arm.

The locally selected dimensionless controls remain AdEMAMix beta1 `0.9`, beta2
`0.999`, beta3 `0.999`, alpha `4`, weight decay `0.1`, gradient clipping `0.1`,
bf16 with fp32 master gradients, and Goldfish `k=h=50`. Keep cross-document
attention masking, EOD loss masking and position-ID reset at document
boundaries.

For recovery and evaluation:

- asynchronously checkpoint full state every 512 Mini steps =
  1,073,741,824 tokens so every native-GreekMMLU point binds to an exact model
  state, plus warmup, update-19,456 segment, transition, matched-control,
  cooldown-start and final boundaries;
- run the fast heldout panel every 512 steps = 1,073,741,824 tokens;
- save immediately before and after each hard boundary and matched cumulative-
  token controls in all other arms;
- evaluate and retain the raw final checkpoint for endpoint selection; do not
  construct SMA or EMA checkpoints;
- prune a non-boundary checkpoint payload only after all of its validation and
  native-GreekMMLU evaluation receipts are frozen;
- convert each exact Megatron `torch_dist` checkpoint through SwissAI's
  `torchdist_2_torch.py` and `convert.py --loader core --saver swissai_hf`,
  bind source iteration/hash to the HF tree hash, and require a pre-campaign
  logit-equivalence smoke before evaluating converted checkpoints;
- never reset model, optimizer, AdEMAMix ramps or schedule state at a data
  boundary.

## 6. Remaining launch sequence

1. Build and round-trip the Mini-compatible tokenizer overlay on a worker.
2. Generate complete per-token TD contexts and execute the four initialization
   cells on GPU.
3. Run the norm, preservation, collapse and heldout selection gates; freeze one
   checkpoint hash.
4. Re-tokenize the final post-exclusion Greek and replay identity manifests;
   freeze exact token counts and schedule boundaries.
5. Use the common smoke's frozen `1.5e-4` selection in every arm; do not branch
   or reselect LR by schedule.
6. Materialize and audit all five schedule manifests; retain one raw final
   endpoint per trajectory.
7. Preserve the accepted B2 DP=16 geometry and its 19.9979-hour forecast; rerun
   the live 20-node scheduler/capacity probe immediately before launch.
8. Submit one aggregate 20-node, five-group allocation only after the remaining
   scientific launch gates pass.

No undeclared setting may change in only one arm. The only optimization factor
is Greek document order. Checkpoint averaging and LR-floor branching are
excluded.
