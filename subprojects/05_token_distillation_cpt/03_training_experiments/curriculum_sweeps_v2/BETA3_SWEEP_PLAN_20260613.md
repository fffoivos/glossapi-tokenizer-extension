# EXP — AdEMAMix β3 sweep on a 2× HPLT (~27 B) run

**Date:** 2026-06-13 · **Status:** ready to build/run · **Owner:** execution agent
**One-liner:** sweep `ADEMA_BETA3 ∈ {0.99, 0.995, 0.999}` on a longer (2× HPLT, 70/30, ~27 B)
curriculum run at the settled HP set, **keeping the production (fraction-of-run) ramp**, to see
whether the β3 target matters at a more production-like horizon and whether the HPLT→OA shift
disturbs the slow momentum.

## 0 · Decision / scope (settled with the user)
- **Sweep:** `ADEMA_BETA3 ∈ {0.99, 0.995, 0.999}` (3 TD arms). Everything else fixed at the production
  set: **LR 5.5e-5 · α=4 · split replay 79/20/1** (FOREIGN_REPLAY_R=20/79, OLD_GREEK_REPLAY_R=1/79).
- **Scheduler: KEEP production.** β3/α warmup stays tied to `TRAIN_ITERS` (no decoupling). The user
  chose this deliberately — it matches what the 60 B run will do, so the result is directly actionable.
  Consequence to accept: each arm only reaches its β3 *max* at the final step; the arms still differ
  throughout because they ramp to *different* endpoints (0.99 vs 0.999), with the gap widening to the
  end. This is a "does the β3 target matter under the production ramp at ~27 B" test, not a "β3 at full
  strength" test.
- **Geometry:** 2× HPLT, **same 70/30 proportion** → `TOTAL_ITER=6436` (~27.0 B), `PHASE1_EXIT_ITER=4522`
  (=2×2261=38×119), `TRAIN_TOKENS=26994540544` (=6436×4,194,304). **No OpenArchives epoch-repeat needed**
  — phase-2 only *consumes* ~6.34 B, which fits the ~7.1 B OA pool (we size the OA binary to consumption,
  not to 2×; the earlier "~6% repeat" worry came from over-sizing it to 7.4 B). Dropped: the replay grid
  and the code-boost (code left as-is).

## 1 · Scripts (already prepared in this folder)
- **NEW** `train/sweep_beta3.sh` — the launcher (mirrors `sweep_alpha.sh`); sets the β3 grid + the 27 B
  geometry + an isolated `STAGE`.
- **EDIT** `train/submit_curriculum_two_phase.sh` — `ADEMA_BETA3` now read + added to the segment
  `--export` (it was missing; only `ADEMA_ALPHA` was exported, so any β3 would have been silently
  dropped). Default 0.999 → no change to prior sweeps.
- **EDIT** `paths.env` — `STAGE`/`MEGOUT` now overridable (`${STAGE:-…}`), so this build stages into its
  own dir without clobbering the 13.5 B binaries.
- **EDIT** `dataset/mix_phase_binaries.sbatch` — per-binary targets overridable (`HPLT_TGT`/`GLOSSAPI_TGT`/
  `REPLAY_TGT`).

## 2 · Token targets for the 2× build
| binary | target (`*_TGT`) | phase consumes | pool | note |
|---|---:|---:|---:|---|
| hplt_only | `16000000000` | ~14.98 B (phase-1) | ~41 B | ample |
| glossapi_only | `6900000000` | ~6.34 B (phase-2) | ~7.1 B | **fits, no repeat** (tight — verify) |
| replay_only | `6500000000` | ~5.67 B (blended throughout) | — | small de/ru/zh repeat as before |

## 2b · Dataset prep + held-out validation (NOT reused — re-run on the 2× draw)
The 2× HPLT / scaled-OA binaries contain **new docs** not in the 13.5 B build, so decontam + anon are
**re-run** (Stage A/B, §3), not reused:
- **Decontamination — Stage A:** HPLT clean (E001) + **GreekMMLU decontaminate.py** (`--benchmark
  greekmmlu --primary-rule correct_only`), dropping any GreekMMLU-contaminated doc from the larger
  HPLT/OA draw. Scope = **GreekMMLU-only**, identical to the LR/α sweeps (keeps β3 comparable to them).
  *Extended decontam across all benchmarks is a production-build item (ROADMAP §5) — out of scope here.*
- **Anonymization — Stage B:** datatrove PII pass (email / ip / iban) + Megatron tokenize ×2 (base+ext).

**Held-out validation sets** (the 9 in `EXTRA_VALID_SETS`, reused corpus-absolute via copy; the recipes
drop their ids so they stay unseen even in the larger draw) — they directly cover your three axes:
| axis | sets | what it answers |
|---|---|---|
| **Adaptation** (learn new Greek) | `hplt` (web), `openarchives` (academic) + **GreekMMLU** (downstream) | does it fit/learn the trained new-Greek distribution |
| **Generalization** (unseen Greek domain) | `greek_phd` (theses — **never trained**) | does Greek adaptation transfer to a held-out domain |
| **Forgetting** (retain Apertus) | `english`/`de`/`ru`/`zh`/`code` (foreign families) + `old_greek` (Apertus-original Greek) | does it keep what Apertus knew |
All 9 are measured **every 25 iters** in-training (extra-valid loss); GreekMMLU via the sidecar watcher.
Report on **absolute** held-out loss (β3 arms share the 148k tokenizer).

## 3 · Build sequence (isolated dir; ~run on Clariden)
```bash
export STAGE_BETA3=/iopsstor/scratch/cscs/fffoivos/cpt_corpus/curriculum_beta3
V2=.../curriculum_sweeps_v2     # source paths.env for $MEGOUT etc.

# (a) seed the isolated stage with the SAME holdout-id parquets (recipes' drop wiring reads them)
mkdir -p "$STAGE_BETA3/megatron"
cp /iopsstor/scratch/cscs/fffoivos/cpt_corpus/curriculum_v2/{val_holdout_ids,forget_holdout_ids}.parquet "$STAGE_BETA3/"

# (a2) regenerate recipes so drop_doc_keys_parquet points at curriculum_beta3, not curriculum_v2
export STAGE="$STAGE_BETA3"
source paths.env
$PY dataset/make_phase_recipes.py "$THIS/dataset_build/bulk_13b.json" "$V2/dataset/recipes_beta3" "$STAGE"
grep -R "$STAGE_BETA3" dataset/recipes_beta3/*.json

# (b) build the 3 phase binaries at the 2× targets  (array 0-2)
sbatch --export=ALL,STAGE=$STAGE_BETA3,RECIPES_DIR=$V2/dataset/recipes_beta3,HPLT_TGT=16000000000,GLOSSAPI_TGT=6900000000,REPLAY_TGT=6500000000 \
       --array=0-2 dataset/mix_phase_binaries.sbatch
# (c) Stage A (clean + GreekMMLU decontam) then Stage B (anon + tokenize ×2), STAGE threaded through
sbatch --export=ALL,STAGE=$STAGE_BETA3 --dependency=afterok:<mix> --array=0-2 dataset/stageA_clean_decontam_binary.sbatch
sbatch --export=ALL,STAGE=$STAGE_BETA3 --dependency=afterok:<A>   --array=0-2 dataset/stageB_anon_preprocess_binary.sbatch
# (d) split replay → foreign_replay_only + old_greek_replay_only (×2 toks)
sbatch --export=ALL,STAGE=$STAGE_BETA3 --dependency=afterok:<B> dataset/split_replay_final_and_tokenize.sbatch

# (e) REUSE the held-out val binaries (corpus-absolute, already built for 13.5B) — copy, don't rebuild
cp /iopsstor/scratch/cscs/fffoivos/cpt_corpus/curriculum_v2/megatron/val_*_text_document.{bin,idx} "$STAGE_BETA3/megatron/"
```
The new-Greek val docs stay held-out because the recipes (`hplt_only.json`/`glossapi_only.json`) still
drop `val_holdout_ids.parquet` from the (now larger) draw — same mechanism as the 13.5 B build.

## 4 · Re-pin the boundary (after Stage B, before launch)
Compute the realized ext-tokenizer token counts (`.bin size / 4`) and pin:
`PHASE1_EXIT_ITER = round_to_119( 6436 × hplt_ext / (hplt_ext + glossapi_ext) )` — expect **4522**.
Verify: (i) ratio ≈ 0.697–0.703; (ii) **no looping** — `hplt_ext ≥ 14.98 B` and `glossapi_ext ≥ 6.34 B`
(if glossapi fell below 6.34 B after decontam, either raise `GLOSSAPI_TGT` toward the 7.1 B pool or accept
a small OA repeat). Regenerate the eval cadence (`eval/cadence_curriculum.tsv`) for the 6436-iter run
(checkpoints every ~238 iters + the boundary 4522 + final 6436).

## 5 · Smoke + launch
```bash
# boundary smoke (tiny PHASE1_EXIT_ITER/TOTAL_ITER) — confirm the reset-guard fires and β3 is plumbed:
#   grep the s2 .err for "[reset_data_index_guard] ... consumed_samples N -> 0" and "--ademamix-beta3 0.99"
STAGE_BETA3=$STAGE_BETA3 DRY_RUN=1 bash train/sweep_beta3.sh        # inspect the 3 chains
STAGE_BETA3=$STAGE_BETA3 DRY_RUN=0 CONFIRM_LAUNCH=1 bash train/sweep_beta3.sh
```
Each arm = ~7 afterok segments (5 phase-1 + 2 phase-2) × ~2.7 h ≈ **~19 h/arm**; 3 arms ≈ 57 h serial
(or ~19 h if 3 chains run concurrently — needs 48 nodes free).

## 6 · Eval + analysis (reuse)
Per run: `RUN_TAG=<tag> EVAL_ARM=td bash eval/curriculum_eval_watcher.sh` (GreekMMLU-only). Then
`collect_greekmmlu.py $RUN_ROOT` + `collect_forgetting_loss.py <run dirs>` → the standard CSVs; report
via the **arxiv-report** skill (absolute held-out loss, β3 gradient, GreekMMLU as the cross-arm metric,
+ a boundary-disturbance readout: any loss bump at iter 4522 vs β3).

## 7 · Acceptance criteria (verify before/at launch)
- `ADEMA_BETA3` appears in the per-segment `--export` and reaches the trainer as `--ademamix-beta3 <b3>`
  (grep a launched `.out`/`scontrol`); the 3 arms show 0.99 / 0.995 / 0.999 respectively.
- β3/α warmup steps = `TRAIN_ITERS` ≈ 6436 (production ramp kept; **not** decoupled) — confirm in the
  config JSON the trainer dumps.
- Realized ratio ≈ 70/30, `PHASE1_EXIT_ITER` a multiple of 119, no binary looping (§4).
- Build went to `curriculum_beta3`, **not** `curriculum_v2` (13.5 B binaries untouched); the 9 held-out
  val binaries (×2 tok) are present in `curriculum_beta3/megatron` (copied).
- α=4, LR 5.5e-5, replay 79/20/1 identical across the 3 arms (only β3 differs).

## 8 · Notes / gotchas
- **Production scheduler kept on purpose** (§0). If a future run wants the *mechanistic* "β3 at full
  strength" test instead, decouple by pinning `ADEMA_BETA3_WARMUP_STEPS`/`ADEMA_ALPHA_WARMUP_STEPS` to a
  fixed step (≈ PHASE1_EXIT_ITER) in `configs/common_cpt.env` — explicitly **not** done here.
- The two `bakeoff_train.sbatch` UPSTREAM edits (EXTRA_VALID_SETS, TRAINER_WRAPPER) must already be applied
  (they are, from the prior sweeps) — `UPSTREAM_EDITS.md`.
- CSCS key expires periodically; re-sign (`cscs-key --headless sign`) before the build if SSH fails.
