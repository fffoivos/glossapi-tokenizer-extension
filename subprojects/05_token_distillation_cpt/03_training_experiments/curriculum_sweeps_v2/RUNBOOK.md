# curriculum_sweeps_v2 — RUNBOOK

Implements `EPISTEMIC_PLAN.md` steps (a)–(d): the two-phase HPLT→GlossAPI curriculum, the replay
sweep (TD), and the peak-LR sweep (TD), plus the one Vanilla control. Built on the deployed pilot
infra; every path is in `paths.env`. **Read §6 (decisions) and §7 (upstream edits) before running.**

## 0 · Key decisions (with the deployed evidence)

- **Two-phase curriculum = segmented-resume chain with a data-index reset, NOT an ordered binary
  (Megatron's global shuffle defeats doc order — the pilot bug, `gpt_dataset.py:624`) and NOT
  `--finetune` (it would drop AdEMAMix + WSD `num_steps`, `checkpointing.py:1264/1283/1312`).**
  Phase-1 trains `blend(hplt, replay)` to `PHASE1_EXIT_ITER` with the full-run `--train-samples` +
  WSD; phase-2 resumes (`RESUME_TRAINING=1` → optim/scheduler loaded, `num_steps` continuous) on
  `blend(glossapi, replay)` with `RESET_DATA_INDEX=1`, which monkeypatches the train dataloader to
  `consumed_samples=0` against the new binary while the scheduler/optimizer keep the phase-1 counts
  (`train/runtime_patches/reset_data_index_guard.py`; the fork left the reset hook commented in
  `data_samplers.py`).
- **Replay % = a Megatron `--data-path` BLEND WEIGHT** (`arguments.py:2020-2027`, `--data-path nargs='*'`,
  `"w1 p1 w2 p2"`). bakeoff_train.sbatch:390 emits `--data-path $ARM_DATA_PREFIX` **unquoted**, so the
  weighted string word-splits. → build the 3 binaries ONCE; the sweep only changes `R`.
- **Eval = GreekMMLU only** (decontam = GreekMMLU only); forgetting = the old-data held-out **loss**.

## 1 · Dataset build (order matters — both holdout parquets must precede the mix)

```bash
cd $V2   # = .../03_training_experiments/curriculum_sweeps_v2
source paths.env
# 1. recipes
$PY dataset/make_phase_recipes.py $THIS/dataset_build/bulk_13b.json $V2/dataset/recipes
# 2. old-data validation sets. Stage StarCoderData first so code is an Apertus-family source.
#    build_forgetting_vals uses the absolute 2B-char target only when it leaves
#    enough train data; finite pools fall back to MAX_VAL_FRACTION=0.25.
sbatch dataset/stage_starcoderdata_subset.sbatch
sbatch dataset/build_forgetting_vals.sbatch                # english + old_greek (+code +de/ru/zh if present)
# 3. new-Greek held-outs (REUSE the deployed builder on a CPU Slurm node, output into v2 STAGE):
sbatch dataset/build_newgreek_vals.sbatch
# 4. the 3 phase binaries (afterok on 2+3 — both holdout parquets must exist)
sbatch --array=0-2 --dependency=afterok:<forget_job>:<newgreek_job> dataset/mix_phase_binaries.sbatch
# 5. clean + GreekMMLU decontam
sbatch --array=0-2 --dependency=afterok:<mix_job> dataset/stageA_clean_decontam_binary.sbatch
# 6. anonymize + tokenize ×2  -> hplt_only/glossapi_only/replay_only _{base,ext}_text_document
sbatch --array=0-2 --dependency=afterok:<stageA_job> dataset/stageB_anon_preprocess_binary.sbatch
# 7. tokenize the held-outs ×2 into $MEGOUT (both use v2-local sbatch — the deployed tokenize_vals.sbatch
#    hardcodes STAGE=cpt_2arm_13b + partition=xfer and CANNOT be overridden, so it is NOT reused):
sbatch --array=0-2 dataset/tokenize_newgreek_vals.sbatch     # val_<name>_{base,ext}   (hplt/openarchives/greek_phd)
sbatch --array=0-5 dataset/tokenize_forgetting_vals.sbatch   # val_forget_<name>_{base,ext}  (the 6 old-data sets)
```

## 2 · Pin `PHASE1_EXIT_ITER` (after step 6)

The 70/30 split is an ABSOLUTE iteration, a multiple of `SAVE_INTERVAL=119`. Compute it from the
realized hplt_only vs glossapi_only token counts (`.bin size / 4`): `PHASE1_EXIT_ITER ≈ round_to_119(
TOTAL_ITER × hplt_tokens/(hplt_tokens+glossapi_tokens))`. The provisional default is `2261`
(`19×119`) for the resized 8.5B/3.7B targets; overwrite `train/curriculum_common.env`,
`train/submit_curriculum_two_phase.sh`, and `eval/cadence_curriculum.tsv` if the realized binaries
round differently.

## 3 · Smoke the phase boundary (before any sweep)

Run a tiny 2-segment chain crossing the boundary, using the launcher's smoke overrides, and confirm
in the phase-2 segment's log:

```bash
DRY_RUN=0 CONFIRM_LAUNCH=1 TOTAL_ITER=2 PHASE1_EXIT_ITER=1 SAVE_INTERVAL=1 SEG=1 NODES=1 \
  TIME_LIMIT=00:30:00 RUN_TAG=curr_smoke_boundary_$(date -u +%Y%m%dT%H%M%SZ) \
  bash train/submit_curriculum_two_phase.sh
```

- `[reset_data_index_guard] ... train dataloader consumed_samples N -> 0`,
- the iteration/`lm loss`/`learning rate` continue (no schedule reset, no NaN),
- the extra-valid `[english]/[old_greek]/...` lines appear.

## 4 · Sweeps (b)→(c)→(d)

```bash
DRY_RUN=0 CONFIRM_LAUNCH=1 bash train/submit_vanilla_control.sh  # (b) 1 vanilla chain, R=0.35
DRY_RUN=0 CONFIRM_LAUNCH=1 bash train/sweep_replay.sh            # (c) TD, R in {0.35,0.25,0.15}
#   ... choose R* from the forgetting↔adaptation tradeoff ...
DRY_RUN=0 CONFIRM_LAUNCH=1 R_STAR=0.25 bash train/sweep_peak_lr.sh # (d) TD, LR in {2.75e-5,5.5e-5,8.25e-5,1.1e-4}
# per run, fire the GreekMMLU-only watcher:
RUN_TAG=<tag> EVAL_ARM=td bash eval/curriculum_eval_watcher.sh
```

## 5 · Reading results

```bash
python analysis/collect_greekmmlu.py     $RUN_ROOT            # adaptation: GreekMMLU per checkpoint
python analysis/collect_forgetting_loss.py $RUN_ROOT/<run>... # forgetting: old-data held-out loss
```
Adaptation = GreekMMLU ↑ + new-Greek held-out loss ↓; forgetting = the 6 old-data held-out losses'
rise. Pick R* (c) and LR (d) at the best balance. **Per-token loss is not comparable across arms**
(ext vs base tokenizer) — compare within an arm/sweep.

## 6 · DECISIONS (resolve before building) — no hard blockers remain

All 6 old-data validation sets are buildable, but their provenance differs:
- **english / de / ru / zh** — Apertus pretraining source families per
  `docs/APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md`: FineWeb-Edu for English and
  FineWeb2-HQ high-resource multilingual for de/ru/zh. They are carved from the
  same staged replay pools that `replay_only` trains on, then their `id`s are
  excluded via `drop_doc_keys_parquet`. The builder writes
  `$STAGE/forgetting_val_manifest.json` with the exact held-out fraction.
- **old_greek** — strictest provenance: same file + `source_doc_id` drop as
  `greek_replay_apertus_original`, built from nanochat rows intersecting the
  Apertus-overlap drop overlay.
- **code** — StarCoderData staged subset (`bigcode/starcoderdata`), matching an
  Apertus pretraining source family. The staging pass rewrites shards with a
  stable `doc_id` so `val_forget_code` can be dropped from replay training.

Validation split policy:
- Try the absolute target (`FORGET_CHAR_BUDGET`, default 2B chars ≈ 0.5B tokens).
- If that would consume more than `MAX_VAL_FRACTION` of a finite eligible pool,
  fall back to a relative validation split (default `MAX_VAL_FRACTION=0.25`, so
  at least ~75% remains trainable).
- For future CPT runs that use all available data, this absolute-then-relative
  rule is mandatory: never let a held-out validation set swallow the entire
  train pool.

1. **PHASE1_EXIT_ITER** is provisional — pin from realized token counts (§2).
2. **Per-tokenizer non-comparability** of forgetting loss (vanilla base vs td ext) — within-arm only.

## 7 · Required upstream edits (see `train/UPSTREAM_EDITS.md`)

Three small env-gated edits (no-ops unless the new env is set): (1) bakeoff_train.sbatch builds
extra-valid from `EXTRA_VALID_SETS` (the 9-set list); (2) bakeoff_train.sbatch prepends
`$TRAINER_WRAPPER` (the reset guard); (3) the watcher keeps `NATIVE_BENCHMARKS`/`SUBMIT_*` across its
self-resubmit. Apply, dry-run the pilot config to confirm zero behavior change, then §3 smoke.

## Folder map

```
paths.env                         single source of truth
dataset/  make_phase_recipes.py   -> recipes/{hplt,glossapi,replay}_only.json
          mix_phase_binaries.sbatch   stageA_clean_decontam_binary.sbatch   stageB_anon_preprocess_binary.sbatch
          build_forgetting_vals.{py,sbatch}   stage_starcoderdata_subset.{py,sbatch}
          tokenize_newgreek_vals.sbatch   tokenize_forgetting_vals.sbatch
train/    runtime_patches/reset_data_index_guard.py   curriculum_common.env   phase1_hplt.env   phase2_glossapi.env
          submit_curriculum_two_phase.sh   sweep_replay.sh   sweep_peak_lr.sh   submit_vanilla_control.sh   UPSTREAM_EDITS.md
eval/     curriculum_eval_watcher.sh   cadence_curriculum.tsv
analysis/ collect_greekmmlu.py   collect_forgetting_loss.py
```
