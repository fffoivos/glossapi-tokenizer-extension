# 03 — Training experiments

> **In one line:** where the training actually happened — the 13.5 B two-arm pilot that
> settled tokenizer-vs-vanilla, then five sweeps that settled replay, peak LR and three
> AdEMAMix parameters.
> **Period:** 2026-06-10 → 2026-07-12 (36 commits). **Status:** completed and frozen on
> 2026-07-11; **not relaunchable** — the Megatron `.bin`/`.idx` payloads were deleted.
> **Came from / led to:** the 5 B TD diagnostics in [`../scripts`](../scripts/README.md) →
> this → [`../06_25b_midtraining_probe`](../06_25b_midtraining_probe/README.md) and the full
> run in [`../../07_full_8b_cpt`](../../07_full_8b_cpt).

## Why this existed

Two questions, in order. First: at a real training scale, does the 148,480-token Greek
tokenizer with Token-Distillation initialization beat the unmodified vanilla model on the
same data and compute? Second, once that is answered: what mixture and optimizer settings
should the production run use? The pilot answered the first in one shot; the sweeps answered
the second one axis at a time, each on the same 13.5 B geometry so the arms stay comparable.

## History

### 2026-06-10 · getting 64 GPUs to talk

Everything was blocked by `NET/OFI ... NO_SPACE` in the first inter-node data-parallel
collective. Single-node worked; pure PyTorch/NCCL inter-node worked. Request-buffer sizing,
hybrid/software CXI matching, Socket fallback and smaller node counts were all tried and none
was the answer — the trainer was forcing `NCCL_NET_FORCE_FLUSH=1`, which the OFI plugin
rejected. `102ac8a6` set it to `0` and 2-, 4- and 16-node smokes passed. Four 16-node smokes
then fixed the budget (`bd0a78e8`): iteration 1 ≈ 15.45 s, iterations 2–10 ≈ 8.63 s, held-out
validation ≈ 11 s, checkpoint save ≈ 22 s → 8.3–8.5 h allocated per arm. `4924bb0b`
right-sized the segments to 6 h/6 h/6 h/3 h. A prelaunch review also caught two things before
launch: a missing `pretrain_gpt_te_guard.py` that would have crashed both arms, and per-set
held-out losses that were not separable in TensorBoard ([`../ARCHIVE.md`](../ARCHIVE.md)).

### 2026-06-10 → 06-11 · the pilot

Both arms launched as `STAMP=20260610T200344Z` on 16 nodes each, four segments, and both
reached **iteration 3218/3218 with 0 NaN and 0 skipped iterations**. The dataset was 10 B new
Greek (70 % HPLT / 30 % OpenArchives) + replay at 35 % of new Greek, with three 0.5 B held-out
sets excluded by document id (`dataset_build/bulk_13b.json`). Result and caveats:
[`../reports/`](../reports/README.md).

The one substantive failure was epistemic, not operational. Stage C had physically ordered
the new-Greek slots HPLT-before-OpenArchives, but Megatron randomizes sample consumption, so
**the ordering never executed** (`47601092`). `d06b1ac4` implemented the fix for future runs:
`CURRICULUM_ORDER_MODE=physical_order`, `MEGATRON_GPT_DATASET_NO_SHUFFLE=1`, a fail-closed
check that the GPTDataset patch is present, and `verify_megatron_curriculum_indices.py` to
prove the generated `document_index` is monotonic and the `shuffle_index` is the identity.

### 2026-06-11 → 06-17 · the sweeps

[`curriculum_sweeps_v2/`](curriculum_sweeps_v2/README.md) — replay, peak LR, α, β₃, β₂, in
that order, each reusing the previous decision. Full narrative and numbers there.

### 2026-06-19 → 07-12 · closing out

`c958a296` added `dataset_build/validate_selected_badness.py` to gate the selected pool's
badness scores. `305feeb0` (2026-07-11) froze the recipe and added
`scripts/gate_frozen_hyperparameters.sh`, an offline gate for the frozen values. `76a44479`
(2026-07-12) was the last touch here, as work moved to full-corpus preparation.

## Outcome

- TD ≫ vanilla at 13.5 B (58.7 % vs 55.3 % native Greek MCQ, base 48.3 %), at tied bits/byte
  and −31 % tokens. The vanilla arm was retired.
- `NCCL_NET_FORCE_FLUSH=0` + 16 nodes/64 GPUs per arm became the standing launch recipe.
- The production recipe frozen here — 79/20/1 replay, peak LR 5.5e-5, α 4, β₃ 0.999,
  β₂ 0.999, fixed 400-iteration warmup — is what
  [`../../07_full_8b_cpt`](../../07_full_8b_cpt) trains with.
- **Reproducibility limit:** all run logs, checkpoints, `run_metadata.json` and eval sidecars
  survive on Clariden, but the `curriculum_v2/megatron` payloads are gone (zero files), so
  payload hashes cannot be recomputed and every sweep launcher now fails its data preflight.

## Where things are

| Path | What it is |
|---|---|
| `configs/common_cpt.env` | The single source of live training values; `arm1_vanilla.env` / `arm2_modern_greek.env` differ only in tokenizer, vocab, init checkpoint and data prefix. |
| `dataset_build/bulk_13b.json` | The realized pilot mix (buckets: greek 0.7407, replay 0.1778, code 0.0296, math 0.0148, greek_replay 0.0370; seed 20260520), built by `make_recipe_13b.py`. |
| `dataset_build/stage{A,B,C}_*.sbatch` | The build order: clean + GreekMMLU `correct_only` decontaminate → anonymize → replay-fixed physical-order preprocess + base/ext tokenization. |
| `dataset_build/build_holdout_vals.py`, `build_greek_replay.py` | The three new-Greek held-outs and the `greek_replay` (nanochat ∩ Apertus-overlap) pool. |
| `scripts/launch_all.sh`, `submit_two_arm_full_run.sh` | The pilot launcher (`DRY_RUN=1` first; `CONFIRM_LAUNCH=1` to go live). |
| `scripts/gate_cpt2arm_artifacts.sh` | The prelaunch artifact gate: regime invariants, tokenizer 256-divisibility, init checkpoints, TE guard, CXI settings, per-set validation patch, held-out and training binaries. `VERIFY_CURRICULUM_CACHE=1` additionally proves the no-shuffle index. |
| `scripts/gate_frozen_hyperparameters.sh` | Offline check of a config against the 2026-07-11 frozen recipe. |
| `scripts/collect_metrics.py` | Parses arm `.out` files into a metrics CSV. |
| [`curriculum_sweeps_v2/`](curriculum_sweeps_v2/README.md) | The five-sweep harness, results and decision tables. |

## Working documents

Six docs that lived here — `BUILD_PLAN.md`, `HANDOFF.md`, `LAUNCH_RUNBOOK.md`, `README.md`,
`TOOLING_DECISIONS.md`, `dataset_build/{HANDOFF,EXTRA_VALID_README}.md` and
`docs/{SCHEDULER_MATH,TOKEN_DISTILLATION_E_AND_U}.md` — were deleted in the 2026-06-10
cleanup after their content was folded into [`../ARCHIVE.md`](../ARCHIVE.md),
[`../LOG.md`](../LOG.md) and [`../RUNBOOK.md`](../RUNBOOK.md). They remain in git history.
