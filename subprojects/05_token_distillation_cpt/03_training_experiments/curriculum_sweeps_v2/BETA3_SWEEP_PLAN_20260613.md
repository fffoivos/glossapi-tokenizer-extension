# EXP — AdEMAMix β3 sweep on the existing 13.5 B curriculum_v2 dataset

**Date:** 2026-06-13 · **Pivoted:** 2026-06-15 · **Status:** LIVE — active arms are β3 `0.99` + `0.995`; the launched `0.999` duplicate was canceled after config verification · **Owner:** execution agent
**One-liner:** sweep `ADEMA_BETA3 ∈ {0.99, 0.995}` on the **already-built 13.5 B `curriculum_v2`**
dataset at the settled HP set, reusing the completed α=4 run as the β3 `0.999` control point, **keeping
the production (fraction-of-run) ramp**, to see whether the β3 target matters under the production
schedule and whether the HPLT→OA shift disturbs the slow momentum.

## 0 · Decision / scope (settled with the user)
- **Pivot (2026-06-15):** node availability won't support the 2× HPLT (~27 B) build + run. We **reuse the
  existing 13.5 B `curriculum_v2` binaries** (already decontam'd / anon'd, with the 9 held-out vals) and
  run the β3 sweep at the **same geometry as the LR/α sweeps**. The 27 B plan is **shelved, not deleted**
  (recoverable in §7).
- **Sweep:** `ADEMA_BETA3 ∈ {0.99, 0.995}` as new TD arms; β3 `0.999` is represented by the completed
  α=4 control run. Everything else fixed at the production set: **LR 5.5e-5 · α=4 · split replay 79/20/1**
  (FOREIGN_REPLAY_R=20/79, OLD_GREEK_REPLAY_R=1/79).
- **Scheduler: KEEP production.** β3/α warmup stays tied to `TRAIN_ITERS≈3218` (no decoupling) — it
  matches what the 60 B run will do, so the result is directly actionable. Consequence to accept: each
  arm only reaches its β3 *max* at the final step; arms still differ throughout because they ramp to
  *different* endpoints (0.99 vs 0.999), with the gap widening to the end.
- **Geometry:** 13.5 B, identical to the prior sweeps → `TOTAL_ITER=3218`, `PHASE1_EXIT_ITER=2261`
  (70/30 boundary, multiple of 119), `TRAIN_TOKENS=13500000000`. No rebuild, no boundary re-pin.
- **Caveat (accept, eyes open):** the 13.5 B horizon is shorter than the shelved 27 B — the slow-EMA
  signal is **more compressed** (a shorter back-portion where the arms diverge; this compression was the
  original motivation for going to 2× HPLT). Read the result as a **conservative, production-matched**
  test: if β3 separates the arms even here, it's real; a *null* is partly a horizon/ramp artifact and is
  **not** proof β3 is inert — the longer 27 B confirm (§7) stays available if the 13.5 B result is
  ambiguous and nodes free up.
- **β3=0.999 arm = the settled α=4 run (confirmed duplicate).** The launched
  `curr_td_b30p999_13b_20260615T130321Z` matched the completed
  `curr_td_f20_g1_lr5.5e-5_a4_20260613T090652Z` on the relevant config fields: `data_seed=20260609`,
  `LR_PEAK=5.5e-5`, `ADEMA_ALPHA=4`, `ADEMA_BETA3=0.999`, warmups `3218`, split replay, 3218 steps, and
  the same HPLT→GlossAPI phase handoff at iter `2261`. It was canceled on 2026-06-15; use the α=4 curves
  as the β3 `0.999` point.

## 1 · Scripts (already prepared in this folder)
- `train/sweep_beta3.sh` — the launcher; β3 grid + 13.5 B geometry; `STAGE` **defaults to
  `curriculum_v2`** (reuse). Preflight asserts the 4 training binaries
  (`hplt_only`, `glossapi_only`, `foreign_replay_only`, `old_greek_replay_only`) + the 9 vals, each ×2
  tokenizers (base+ext) × {bin,idx}, already exist in `$STAGE/megatron`; fails fast if not.
- `train/submit_curriculum_two_phase.sh` — reads `ADEMA_BETA3` and threads it into the per-segment
  `--export` (it was missing; only `ADEMA_ALPHA` was exported, so any β3 would have been silently
  dropped). Default 0.999 → no change to prior sweeps.
- `paths.env` / `dataset/mix_phase_binaries.sbatch` — the STAGE/MEGOUT + per-binary `*_TGT` overrides
  (added for the 27 B build) are **unused on this path** but harmless; they default to curriculum_v2.

## 2 · Data — REUSED (no build)
The 13.5 B `curriculum_v2` dataset is already built and was used by the LR / α / replay sweeps:
- training binaries (×2 tok, base+ext): `hplt_only`, `glossapi_only`, `foreign_replay_only`,
  `old_greek_replay_only` — in `curriculum_v2/megatron`.
- decontam (**Stage A**, GreekMMLU `--primary-rule correct_only`) + anon (**Stage B**, datatrove PII)
  already applied. Scope = GreekMMLU-only, identical to the LR/α sweeps (keeps β3 comparable to them).
- the 9 held-out val binaries already built (their ids dropped from the training draw).

The β3 sweep reads these **read-only**; run outputs go to **new** `RUN_TAG`s under `runs/curriculum_v2/`,
so the prior runs' checkpoints and the dataset binaries are untouched.

**Held-out validation → your three axes** (unchanged; these are the reused vals):
| axis | sets | what it answers |
|---|---|---|
| **Adaptation** (learn new Greek) | `hplt` (web), `openarchives` (academic) + **GreekMMLU** (downstream) | does it fit/learn the trained new-Greek distribution |
| **Generalization** (unseen Greek domain) | `greek_phd` (theses — **never trained**) | does Greek adaptation transfer to a held-out domain |
| **Forgetting** (retain Apertus) | `english`/`de`/`ru`/`zh`/`code` (foreign families) + `old_greek` (Apertus-original Greek) | does it keep what Apertus knew |
All 9 measured **every 25 iters** in-training (extra-valid loss); GreekMMLU via the sidecar watcher.
Report on **absolute** held-out loss (β3 arms share the 148k tokenizer).

## 3 · Run sequence (no build; STAGE defaults to curriculum_v2)
```bash
V2=.../curriculum_sweeps_v2
# (preflight asserts the binaries are present)
DRY_RUN=1 bash train/sweep_beta3.sh                          # inspect the 3 chains
DRY_RUN=0 CONFIRM_LAUNCH=1 bash train/sweep_beta3.sh         # launches 0.99 + 0.995 by default
# To force a reproducibility rerun of the 0.999 control:
BETA3_GRID="0.999" DRY_RUN=0 CONFIRM_LAUNCH=1 bash train/sweep_beta3.sh
```
Each arm = ~5 phase-1 + 2 phase-2 afterok segments × ~2.7 h ≈ **~9.5 h/arm**; 3 arms ≈ 28.5 h serial
(or ~9.5 h if 3 chains run concurrently — needs 48 nodes). Current active run skips 0.999 → 2 arms / ~19 h
serial, or ~9.5 h while both chains run concurrently.

## 4 · Smoke (recommended before the real launch)
```bash
# tiny boundary smoke — confirm the reset-guard fires and β3 reaches the trainer:
#   grep the s2 .err for "[reset_data_index_guard] ... consumed_samples N -> 0" and "--ademamix-beta3 0.99"
TOTAL_ITER=4 PHASE1_EXIT_ITER=2 SAVE_INTERVAL=2 SEG=2 NODES=1 GPUS_PER_NODE=1 \
  BETA3_GRID=0.99 DRY_RUN=0 CONFIRM_LAUNCH=1 bash train/sweep_beta3.sh
```

## 5 · Eval + analysis (reuse)
Per run: `RUN_TAG=<tag> EVAL_ARM=td bash eval/curriculum_eval_watcher.sh` (GreekMMLU-only). Then
`collect_greekmmlu.py $RUN_ROOT` + `collect_forgetting_loss.py <run dirs>` → the standard CSVs; report
via the **arxiv-report** skill (absolute held-out loss, β3 gradient, GreekMMLU as the cross-arm metric,
and a boundary-disturbance readout: any loss bump at iter 2261 vs β3).

## 6 · Acceptance criteria (verify before/at launch)
- `ADEMA_BETA3` appears in the per-segment `--export` and reaches the trainer as `--ademamix-beta3 <b3>`
  (grep a launched `.out`/`scontrol`); the active arms show 0.99 / 0.995, with the completed α=4 run as
  the 0.999 point.
- β3/α warmup steps = `TRAIN_ITERS` ≈ 3218 (production ramp kept; **not** decoupled) — confirm in the
  config JSON the trainer dumps.
- Geometry: `TOTAL_ITER=3218`, `PHASE1_EXIT_ITER=2261` (multiple of 119); reads `curriculum_v2` binaries.
- α=4, LR 5.5e-5, replay 79/20/1 identical across the active arms and the reused 0.999 control.
- **No writes** into `curriculum_v2/megatron`; run outputs under new `RUN_TAG`s.

## 7 · Shelved 27 B plan (recoverable — the longer-horizon confirm)
If the 13.5 B result is ambiguous on β3 and nodes free up, run the 2× HPLT (~27 B) version for a longer
back-portion where the arms diverge:
- Build into an **isolated** `STAGE=$SC/cpt_corpus/curriculum_beta3` (don't clobber curriculum_v2):
  targets `HPLT_TGT=16e9 · GLOSSAPI_TGT=6.9e9 · REPLAY_TGT=6.5e9` (OA sized to consumption ~6.34 B < ~7.1 B
  pool → no epoch-repeat); regenerate recipes so `drop_doc_keys_parquet` points at `curriculum_beta3`;
  re-run Stage A/B; copy the 9 vals in.
- Geometry: `TOTAL_ITER=6436`, `TRAIN_TOKENS=26994540544` (=6436×4,194,304); **re-pin** `PHASE1_EXIT_ITER`
  (≈4522) from the realized ext-tokenizer Stage-B `.bin` sizes; verify ratio ≈ 0.70 and no looping
  (`hplt_ext ≥ 14.98 B`, `glossapi_ext ≥ 6.34 B`); regenerate `eval/cadence_curriculum.tsv` for 6436.
- Launch via `STAGE_BETA3=.../curriculum_beta3 ... bash train/sweep_beta3.sh` with the 27 B geometry
  passed explicitly (`TRAIN_TOKENS/TOTAL_ITER/PHASE1_EXIT_ITER`).

## 8 · Notes / gotchas
- **Production scheduler kept on purpose** (§0). The mechanistic "β3 at full strength" test (decouple the
  warmup by pinning `ADEMA_BETA3_WARMUP_STEPS`/`ADEMA_ALPHA_WARMUP_STEPS` to a fixed step in
  `configs/common_cpt.env`) remains a separate, future run — explicitly **not** done here.
- The two `bakeoff_train.sbatch` UPSTREAM edits (EXTRA_VALID_SETS, TRAINER_WRAPPER) must already be applied
  (they are, from the prior sweeps) — `UPSTREAM_EDITS.md`.
- CSCS key expires periodically; re-sign (`cscs-key --headless sign`) before launch if SSH fails.
