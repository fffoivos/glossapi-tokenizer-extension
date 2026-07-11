# EXP — AdEMAMix β2 sweep on the existing 13.5 B curriculum_v2 dataset

**Date:** 2026-06-16 · **Status:** COMPLETE — selected beta2 `0.999` at the fixed 400-iteration LR warmup; see `results/beta2_decision_table_20260711.csv` and `../../PRODUCTION_HYPERPARAMETERS_DECISION_20260711.md` · **Owner:** Foivos / execution agent
**One-liner:** sweep `ADEMA_BETA2 ∈ {0.99, 0.995, 0.999}` at β₃=0.999 on the already-built 13.5 B
`curriculum_v2` dataset, settled HP set, **LR warmup pinned to 400 it to isolate β₂**.

## 0 · Decision / scope (settled with the user)
- **Sweep:** β₂ ∈ {0.99, **0.995**, 0.999} at **β₃=0.999**. At launch the baseline
  default was 0.995, so the completed alpha=4 run
  (`curr_td_f20_g1_lr5.5e-5_a4`) **is the β₂=0.995 middle point** — integrate it, don't re-run.
  The completed comparison selected **0.999**; current production defaults now reflect that result.
  → **2 new arms:** β₂ ∈ {0.99, 0.999}.
- **Dropped this round:** the β₃=0.9999 probe and the 27 B long-dataset build (shelved; both recoverable).
- **Held fixed:** LR 5.5e-5 · β₃=0.999 · α=4 · split replay 79/20/1 (FOREIGN=20/79, OLD_GREEK=1/79).
- **WARMUP PINNED to 400 it (all arms)** — NOT the config's coupled `2/(1-β₂)`. On a 3218-it run, coupled
  would give β₂=0.999 a **2000-it warmup (62 % of the run)** — pathological and confounding. 400 it
  (= 2/(1-0.995), the a4 anchor's warmup) makes the three β₂ points share one warmup → a clean β₂-only
  contrast, and a4 slots in exactly. `COUPLE_WARMUP=1` reverts to the coupled policy.
- **Geometry:** 13.5 B, identical to the α/LR/β₃ sweeps → TOTAL_ITER=3218, PHASE1_EXIT_ITER=2261.
- **Data:** REUSES curriculum_v2 binaries + 9 held-out vals (no build); read-only, outputs to new RUN_TAGs.

## 1 · Scripts
- **NEW** `train/sweep_beta2.sh` — launcher; β₂ grid + pinned warmup + 13.5 B geometry; STAGE=curriculum_v2.
- **EDIT** `train/submit_curriculum_two_phase.sh` — reads `ADEMA_BETA2` + `LR_WARMUP_ITERS` and threads
  both into the per-segment `--export`. At sweep time its defaults were beta2=0.995 and coupled warmup;
  after the decision they are beta2=0.999 and fixed 400 iterations.

## 2 · β₂ → warmup (pinned vs coupled)
| β₂ | variance half-life 1/(1-β₂) | coupled warmup 2/(1-β₂) | % of 3218 it | **PINNED (used)** |
|---|---:|---:|---:|---:|
| 0.99  | 100  | 200  | 6 %  | **400** |
| 0.995 | 200  | 400  | 12 % | **400** (= a4) |
| 0.999 | 1000 | 2000 | 62 % | **400** |

## 3 · Run
```bash
DRY_RUN=1 bash train/sweep_beta2.sh                  # inspect the 2 chains
DRY_RUN=0 CONFIRM_LAUNCH=1 bash train/sweep_beta2.sh
```
2 arms × (~5 phase-1 + 2 phase-2 afterok segments) × ~2.7 h ≈ **~9.5 h/arm**. Cluster is busy → the arms
will **queue** (PENDING) until 16-node blocks free; the afterok chains advance as nodes free.

## 4 · Eval + integration
Same path as β₃: `RUN_TAG=<tag> EVAL_ARM=td bash eval/curriculum_eval_watcher.sh` (GreekMMLU) +
`analysis/collect_greekmmlu.py` + `analysis/collect_forgetting_loss.py`. **Integrate the a4 run as the
β₂=0.995 point** — same dataset, same geometry, same warmup (400) → a clean 3-point β₂ curve. Report via
the **arxiv-report** skill: absolute held-out loss + GreekMMLU (cross-arm), β₂ gradient palette.

## 5 · Acceptance / caveats
- `ADEMA_BETA2` reaches the trainer (`--adam-beta2 <β2>`) and `lr_warmup_samples` = 400×1024 = 409,600 for
  **all** arms (confirm in the dumped config JSON) — i.e. the warmup pin took and only β₂ differs.
- a4 = (β₂=0.995, β₃=0.999, warmup 400, 13.5 B) — same dataset/geometry → a **valid** anchor (no
  cross-horizon confound, unlike the shelved 27 B idea).
- All arms share the 148,480 ext tokenizer → absolute held-out loss is directly comparable.
- GreekMMLU is the cross-arm comparator; the β₂ effect on a 13.5 B horizon may be modest.
- The β₃=0.9999 probe + the 27 B grid remain available if β₂ shows signal worth a longer-horizon look.
