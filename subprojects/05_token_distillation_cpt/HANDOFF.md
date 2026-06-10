# HANDOFF — two-arm 13.5B Greek CPT: dataset → checkpoints → experiments

**Entry point for the execution agent.** Everything decided, built, or specified
for the full pipeline: build the dataset, build the init checkpoints, train
**both arms in parallel** with **per-set held-out validation losses**, plus the
auto-benchmark sidecars. Detail docs: dataset build →
[`03_training_experiments/dataset_build/HANDOFF.md`](03_training_experiments/dataset_build/HANDOFF.md);
build runbook → [`03_training_experiments/BUILD_PLAN.md`](03_training_experiments/BUILD_PLAN.md);
launch ops → [`03_training_experiments/LAUNCH_RUNBOOK.md`](03_training_experiments/LAUNCH_RUNBOOK.md);
corpus-prep policy → [`02_corpus_preparation/PIPELINE.md`](02_corpus_preparation/PIPELINE.md);
hyperparameters → [`../CURRENT_HYPERPARAMETERS.md`](../CURRENT_HYPERPARAMETERS.md) (v1.0, finalized).
`$SC = /iopsstor/scratch/cscs/fffoivos`, account `a0140`. All data work must
run CPU-only on `xfer`; `normal` allocates GH200 nodes and should be reserved
for the training runs.

**Current 2026-06-10 state:** dataset build, Stage-C ordering, base/ext
tokenization, init checkpoints, extra-validation patch, and artifact gate are
complete. The two full chains were attempted but must not be relaunched yet:
multi-node Megatron fails before iteration 1 with
`NET/OFI ... NO_SPACE` in `DATA_PARALLEL_GROUP_WITH_CP`. See
[`reports/CLARIDEN_MEGATRON_NCCL_NO_SPACE_20260610.md`](reports/CLARIDEN_MEGATRON_NCCL_NO_SPACE_20260610.md)
and `RUN_LOG_20260609_CPT_2ARM.md`. A 1-node real-data smoke of the current
recipe completed 2 iterations cleanly; the blocker is inter-node full-Megatron
DDP communication on the AWS Libfabric/CXI path, not
data/checkpoints/hyperparameters. Socket over HSN has since passed 2-node
real-data, 4-node mock-data, and 16-node mock-data Megatron smokes. However,
the 16-node smoke ran at `30046.7 ms/iter`, implying ~`26.9h` raw training per
arm before validation/checkpoint overhead. That is functional but not the
desired ~12h target, so do not launch the full two-arm run as-is without
accepting that walltime or finding a faster transport/parallelism path.

## The two experiments

| | Arm 1 — vanilla | Arm 2 — modern-greek TD |
|---|---|---|
| Tokenizer / vocab | `$SC/models/apertus-8b-2509` (131,072) | `$SC/tokenizers/apertus_greek_modern_only_148480` (148,480) |
| Init checkpoint | Apertus-8B-2509 `main`, geometry-reverted, R17-patched | TD layer-11 (E + U), R17-patched |
| Data binary | `bulk_mix_ordered_replay_fixed_base_text_document` | `bulk_mix_ordered_replay_fixed_ext_text_document` |
| Everything else | identical: 13.5B full-run WSD, AdEMAMix, Goldfish | identical |

Configs already encode this: `03_training_experiments/configs/{common_cpt,arm1_vanilla,arm2_modern_greek}.env`
(peak 5.5e-5, warmup 2/(1−β₂), 20% 1-sqrt cooldown, β₂ 0.995/β₃ 0.999/α 4, rope
500k/4096 scaling ON, goldfish 50/50, vocab pad 256). Submitter:
`03_training_experiments/scripts/submit_two_arm_full_run.sh` (full-run WSD anchor, `--exit-interval` segments).

## Phase 1 — Dataset (CPU; ~1 day wall; detail = 03_training_experiments/dataset_build/HANDOFF.md)

Mixture: **10B unseen = 70% HPLT + 30% openarchives** + replay of-new
{24% multilingual, 4% code, 2% math, 5% Greek-replay} → 13.5B. Three held-out
val sets (0.5B each: hplt, openarchives, greek_phd) excluded from training.
Greek-replay parquet ALREADY BUILT: `$SC/cpt_corpus/greek_replay/greek_replay.parquet`.
Final stream order is slot-preserving: replay/code/math/Greek-replay rows stay
at their original interleaved positions, while the new-Greek slots are rewritten
so the filtered new-Greek subsequence is HPLT first, then openarchives.

```bash
DB=$SC/repo/glossapi-tokenizer-extension/subprojects/05_token_distillation_cpt/03_training_experiments/dataset_build
# venv deps first (dataset_build/HANDOFF §3): transformers(Apertus)+httpx+datatrove into cpt_build_py312
# bulk_13b.json is already committed in $DB — regenerate ONLY if changing the mixture:
#   python3 $DB/make_recipe_13b.py $SC/repo/03_apertus_*/.../init_bakeoff/corpus_build/recipes/bulk.json $DB/bulk_13b.json
V=$(sbatch  --parsable $DB/build_holdout_vals.sbatch)
J2=$(sbatch --parsable --array=0-7%2 --dependency=afterok:$V $DB/mix_13b.sbatch)
A=$(sbatch  --parsable --dependency=afterok:$J2 $DB/stageA_clean_decontam.sbatch)
B=$(sbatch  --parsable --dependency=afterok:$A  $DB/stageB_anon_preprocess.sbatch)
C=$(sbatch  --parsable --dependency=afterok:$B  $DB/stageC_order_replay_fixed_preprocess.sbatch)
TV=$(sbatch --parsable --array=0-2 --dependency=afterok:$V $DB/tokenize_vals.sbatch)
```
Order inside the chain: mix(70/30+replay, holdouts excluded) → E001 clean →
decontaminate `correct_only` (covers Greek-replay) → anonymize full mix →
Stage-C reorder of only new-Greek slots (replay positions preserved) → tokenize ×2.
**Gate:** both `bulk_mix_ordered_replay_fixed_{base,ext}_text_document.{bin,idx}`
and the 6 `val_*_{base,ext}_text_document` files exist, non-trivial size.

## Phase 2 — Init checkpoints (BUILD_PLAN §0–§1)

- **Arm 1:** HF `main` → revert geometry in config.json (`rope_theta` 12M→500000,
  `max_position_embeddings` 65536→4096, **keep** `rope_scaling`) → fork
  `tools/checkpoint/convert.py` (`--loader apertus_hf --saver core --bf16
  --loader-transformer-impl transformer_engine --target-tensor-parallel-size 2`)
  → `patch_apertus_extras.py` (R17). Template: `init_bakeoff/megatron_patches/td_layer11_r17_roundtrip.sbatch`.
- **Arm 2:** reuse `$SC/token_distillation/td_full25_layer11_r17_roundtrip_2357565/megatron_tp2_r17patched`
  **iff** its tokenizer == `apertus_greek_modern_only_148480` (byte-compare tokenizer.json); else re-run TD
  (`03_training_experiments/docs/TOKEN_DISTILLATION_E_AND_U.md`).
- **Gate (both):** `verify_hf_roundtrip.py --require-r17-match --logits` passes.
  Update the two `INIT_CKPT` lines in `03_training_experiments/configs/arm*.env` if paths differ.

## Phase 3 — Per-set validation patch (complete)

Applied `03_training_experiments/dataset_build/EXTRA_VALID_README.md` to
`$SC/code/training/Megatron-LM-Swiss-AI`: adds `--extra-valid-data-path NAME PATH…`
→ separate `[name] lm loss validation` curves per held-out set at
`--eval-interval` (default 1; raise if too slow). Wire `EXTRA_VALID_FLAGS` +
`--eval-interval/--eval-iters` into `bakeoff_train.sbatch` DATA_ARGS, with
`TOK=base|ext` per arm. **Gate:** artifact gate confirms the patch is present;
full training is still blocked by the multi-node Megatron/NCCL issue above.

## Phase 4 — Launch both arms in parallel + benchmarks

```bash
S=$SC/repo/glossapi-tokenizer-extension/subprojects/05_token_distillation_cpt/03_training_experiments
bash $S/scripts/launch_all.sh                          # dry-run, inspect
NCCL_NET=Socket NCCL_SOCKET_IFNAME=hsn bash $S/scripts/gate_cpt2arm_artifacts.sh
NCCL_NET=Socket NCCL_SOCKET_IFNAME=hsn DRY_RUN=0 CONFIRM_LAUNCH=1 bash $S/scripts/launch_all.sh
```
Per arm: walltime-bounded training chain (full-run WSD; Socket/16-node launches
default to 4 longer segments to avoid queue churn) + an eval watcher
(checkpoint every 119 iters, benchmarks every 238 ≈ 1B tokens: greekmmlu dascim
+ ilsp×2 + plutus, Greek-NLP, BPB, retention). Watchers default to `xfer` —
fine after 2026-06-11. Monitor:
`python3 $S/scripts/collect_metrics.py --arm vanilla:'…/cpt13b_vanilla_*/*.out' --arm td:'…/cpt13b_td_*/*.out'`.

Full-scale launch is `16` nodes/arm and must use `LAUNCH_MODE=torchrun` (one
Slurm task per node; torchrun fans out to 4 GPUs). Direct multi-task Slurm
launch has reproduced an inter-node NCCL/OFI `NO_SPACE` failure before
iteration 1. The trainer sets the CSCS Alps/uenv NCCL/libfabric runtime
variables; the artifact gate checks this before launch.

Do not run the live launch command again on the AWS Libfabric/CXI path. Pure
PyTorch NCCL controls pass, including Megatron-shaped 40M and exact-size 67M
bfloat16 all-reduce/reduce-scatter/all-gather; direct Slurm rank launch and
many CXI tuning variants still fail. The current practical path is
`NCCL_NET=Socket NCCL_SOCKET_IFNAME=hsn`: it has reached real iteration/loss
lines on 2-node real data and completed 4-node and 16-node mock-data first-step
smokes. The 16-node result is stable but slow: `30046.7 ms/iter`, ~`26.9h` raw
training per arm. Launching both arms truly in parallel would require 32 nodes /
128 GPUs and still inherit that raw walltime. If using this fallback anyway,
increase the segment exit interval to reduce queue churn; otherwise keep
working on CXI/support or a different parallelism shape before full launch.

## Environment gotchas (cost us hours — DO NOT rediscover)

1. All python in **`$SC/python_envs/cpt_build_py312`** inside `uenv start
   pytorch/v2.9.1:v2 --view=default --ignore-tty` only. Needs recent
   `transformers` + `httpx` (uenv transformers lacks `ApertusConfig`; the
   first mix run died here). `lm_eval`-venv pyarrow is broken.
2. `mix_builder.py` lives at the LEGACY mirror `$SC/repo/03_apertus_*/.../corpus_build/`.
3. nanochat join key is `(source_dataset, source_doc_id)` — no `doc_key`.
4. `normal` ≤ 12 h walltime; whole GH200 node billed even CPU-only.
5. Pin uenv `pytorch/v2.9.1:v2` for conversions (older transformers lacks Apertus).
