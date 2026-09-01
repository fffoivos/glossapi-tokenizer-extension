# 06 — 25 B midtraining probe

> **In one line:** a fully receipt-gated two-phase 25 B CPT run — dataset, initialization,
> smokes and launch gates all built and frozen — that was never submitted for training; its
> prepared assets went straight into two other experiments instead.
> **Period:** 2026-07-31 → 2026-08-01 (13 commits, `8dbb6d25` … `86c1b8fe`).
> **Status:** prepared, never launched. `configs/recipe_25b_midtraining.json` still reads
> `status: frozen_pending_clariden_asset_receipts`.
> **Came from / led to:** [`../05_training_dataset_bridge`](../05_training_dataset_bridge/)
> (superseded by this) → this → [`../07_8b_lr_floor_reconstruction`](../07_8b_lr_floor_reconstruction/README.md)
> and [`../../07_full_8b_cpt`](../../07_full_8b_cpt), both of which consume its outputs.

## Why this existed

[`../ROADMAP_20260611.md`](../ROADMAP_20260611.md) §6 made a mid-scale probe the gate before
any large spend: does GreekMMLU break past the pilot's ~59 % plateau once the data is mostly
fresh, and does the frozen recipe hold at scale? The single-blend launch path in
`05_training_dataset_bridge` had gone stale, so this directory replaced it with a build that
pins every input by hash and refuses to advance a stage without the previous stage's receipt.

## What was frozen

From [`configs/recipe_25b_midtraining.json`](configs/recipe_25b_midtraining.json):

- **Model** `swiss-ai/Apertus-8B-2509`, layer-11 Token-Distillation initialization, TP = 2,
  untied embeddings, zero old-row drift required.
- **Tokenizer** `fffoivos/apertus-tokenizer-extension@fcd33ec…`, subfolder
  `greek-modern-polytonic-tokenizer` — **148,992** rows (the 148,480 modern set plus 512
  polytonic), no padding. This is the first appearance of the tokenizer the full run uses.
- **Dataset** `fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2@3f97cec4…`,
  51,839,746 documents / 431 Parquet files / 63,822,761,532 training tokens, GreekMMLU
  decontamination required before training.
- **Horizon** 5,960 iterations / 24,998,051,840 effective tokens at 4,194,304 tokens/update.
- **Two phases, one optimization run.** Phase 1 (0–3,570): 79 % HPLT / 20 % foreign replay /
  1 % old Greek. Phase 2 (3,570–5,960): HPLT and non-HPLT Greek weighted so the *whole run*
  retains the corpus's natural HPLT share, same replay. Seed `20260609`; document allocation
  disjoint across phases by a domain-separated SHA-256 60/40 split, non-HPLT Greek phase-2 only.
- The boundary changes **only the data blend** — optimizer, scheduler, RNG and global
  iteration resume normally, and the runtime wrapper
  (`train/runtime_patches/phase_relative_data_index.py`) converts the checkpoint's global
  consumed-sample count into a phase-relative data index on *every* segment, fixing the
  earlier first-segment-only reset.

## The gate chain that was built

1. Local: `scripts_validate_recipe.py` + `tests/`.
2. Clariden: materialize and hash the exact dataset revision, tokenizer tree, GreekMMLU
   decontamination inputs, replay inputs, clean Megatron commit and production init. The
   pinned GreekMMLU revision has 25 labelled empty-answer rows among 16,632; their questions
   stay bound for exact full-prompt matching while answer-dependent drop rules are disabled
   for those rows and the exception count is receipted.
3. Heldouts built *before* training binaries, their identities excluded, decontamination
   applied to new Greek and replay.
4. Phase manifests and binaries, with the finalizer proving no retained `doc_id` intersection
   across phases, every blend summing exactly to 1, and ≥ 1.005× exact-content-unique capacity
   plus one boundary sample for every pool, source and physical prefix.
5. Checkpoint-load and two-iteration GPU smokes, including a synthetic phase-boundary resume
   that must show `consumed_samples 8 -> 0`.
6. Asset freeze, then a 16-node dry run. The 64-GPU submission stayed a separate gate with
   its own confirmation string (`GREEK_CPT25B_64GPU`, distinct from the smoke's
   `GREEK_CPT25B_SMOKE`), and the preparation submitter could never launch training.

Segment mechanics were worked out too: iterations 1,785 and 3,570 are periodic-save
boundaries, and because the exact final iteration 5,960 falls ten iterations after the last
periodic save, the final segment must force and receipt a terminal checkpoint rather than
silently rounding to 5,950. After each segment a dependent CPU job freezes only that loadable
boundary into `checkpoint_receipts/iteration_<N>.json`, which becomes the next segment's
required `RESUME_CHECKPOINT_RECEIPT`. One deliberate prohibition is encoded in the recipe:
the CPT-trained `TokenDistil-3.5B` cutoff-probe checkpoint is **forbidden** as production
initialization; `prereqs` rebuilds the appended 512 rows from the uncpt `TokenDistil-Init`
checkpoint instead.

## Outcome

- **The preparation ran; the probe did not.** No training receipt, log, checkpoint or
  GreekMMLU result for a 25 B run exists in this repository, and the recipe status was never
  advanced past `frozen_pending_clariden_asset_receipts`. The scaling question the probe was
  built to answer was instead settled at 0.5 B scale by
  [`../../06_dataset_scheduling_experiments`](../../06_dataset_scheduling_experiments), whose
  five-arm study selected **D0 stationary mixing** — retiring the two-phase design here.
- **The assets were reused, twice.** The materialized stage
  `cpt25b_midtraining/20260731T124000Z-cpt25b-v1` (1,457 training tasks, 12 heldout sets) and
  the verified layer-11 TD initialization under
  `greek-cpt25b-init{,-roundtrip}/20260731T124000Z-cpt25b-v1` are pinned as `source_stage` /
  `initialization` in [`../07_8b_lr_floor_reconstruction/configs/recipe_13b_lr_floor.json`](../07_8b_lr_floor_reconstruction/configs/recipe_13b_lr_floor.json)
  and as `source_binary_root` / `initialization` in
  [`../../07_full_8b_cpt/configs/recipe_8b_full_mixed.json`](../../07_full_8b_cpt/configs/recipe_8b_full_mixed.json).
- **It became the reference implementation, not the run.**
  [`../CPT_LAUNCH_RESOURCE_SPEC_20260801.md`](../CPT_LAUNCH_RESOURCE_SPEC_20260801.md) (the
  next day) calls this phase exactly that: its receipt gates, two-phase resume and evaluation
  are the pattern to scale, while its 25 B horizon reaches only ~30.9 % of the published
  corpus and is therefore "a diagnostic, not the production run".

## Where things are

| Path | What it is |
|---|---|
| [`configs/recipe_25b_midtraining.json`](configs/recipe_25b_midtraining.json) | The frozen machine contract — the authority for every number above |
| `clariden/submit_data_pipeline.sh` | The preparation submitter (`prereqs` → `after-freeze` → `assets`, plus a read-only `status`); never launches training |
| `dataset/` | `freeze_inputs.py`, `freeze_decontamination_binding.py`, `phase_partition.py`, `finalize_phase_bridge.py` |
| `initialization/` | Production init build, `freeze_training_assets.py`, `verify_production_init.py` |
| `train/` | `submit_segment.sh`, `submit_smoke.sh`, `phase_config.env`, the phase-relative data-index wrapper, `freeze_resume_checkpoint.py` |
| `eval/` | Per-checkpoint GreekMMLU watcher over all 16,632 items, freezing an `evaluation_receipt.json` |
| `tests/test_phase_contracts.py`, `scripts_validate_recipe.py` | The local gates |
