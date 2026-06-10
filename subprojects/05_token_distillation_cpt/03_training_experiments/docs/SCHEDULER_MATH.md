# Scheduler math — 13.5B full run

All values for both arms. Recompute if `TRAIN_TOKENS` or the batch changes;
`common_cpt.env` derives them in bash from `TRAIN_TOKENS`.

## Budget

| | tokens |
|---|---|
| New Greek (70% HPLT + 30% openarchives, unseen), primary | 10.00 B |
| + multilingual replay (24% of new) | 2.40 B |
| + code (4% of new) | 0.40 B |
| + math (2% of new) | 0.20 B |
| + Greek replay (5% of new) | 0.50 B |
| **Total (`TRAIN_TOKENS`)** | **13.50 B** |

> Replay percentages are **of the new-Greek budget** ("add 35% on top of the
> 10B") — decided 2026-06-09; this is what gives ≈13.5B. The per-step interleave
> ratio that follows is ≈ 74.1% new / 17.8% ML / 3.0% code / 1.5% math / 3.7%
> Greek-replay. `CURRENT_HYPERPARAMETERS §5` is reconciled to this.

## Steps

- Global batch = 1024 seq × 4096 = **4,194,304 tokens/iter**.
- `TRAIN_SAMPLES` = 13.5e9 / 4096 = **3,295,898**.
- `TRAIN_ITERS` = 3,295,898 / 1024 = **≈ 3,218 iters**.

## WSD schedule (full run, all schedulers span the whole 3,218 iters)

| Phase | Iters | Tokens | LR | Flag |
|---|---|---|---|---|
| Warmup | **400** (12.4%) | 1.68 B | 5.5e-6 → 5.5e-5, linear | `--lr-warmup-samples 409600` |
| Stable | ~2,174 (67.6%) | ~9.1 B | 5.5e-5 | — |
| Cooldown | **644** (20%) | ~2.7 B | 5.5e-5 → 5.5e-6, 1-sqrt | `--lr-wsd-decay-samples 659179` |

- **Peak 5.5e-5** = 0.5× Apertus; **min 5.5e-6** = 0.1× peak; warmup starts at 0.1× peak.
- **Warmup = 400 iters, not 1% (32).** At 13.5B, 1% is far below the
  `2/(1-β2)` second-moment-reliability floor (β2=0.995 → 400 iters), so we
  launch at the floor (`CURRENT_HYPERPARAMETERS §2` fallback). If β2 sweeps to
  0.999 the floor jumps to 2,000 iters (62% of the run) — then 1% would need to
  win on test evidence. **`LR_WARMUP_ITERS` scales with β2; recompute on sweep.**
- **Cooldown = 20% = 644 iters**, 1-sqrt, consuming the new
  `LR_WSD_DECAY_SAMPLES` knob added to `bakeoff_train.sbatch` (the legacy
  hardcode `--lr-wsd-decay-samples $TRAIN_SAMPLES` = 100% decay / 0% stable was
  wrong for a trapezoid).

## AdEMAMix α/β3 warmup

- **Over the whole run**: `ADEMA_BETA3_WARMUP_STEPS = ADEMA_ALPHA_WARMUP_STEPS = TRAIN_ITERS ≈ 3,218`, warmed from `β_start = β1 = 0.9` (AdEMAMix `T_{α,β3}=T`).
- This is **NOT** the submitter's old hardcode of 287 — the new submitter does
  not export the warmup steps, so the config's whole-run value wins.
- ⚠ Open: on a 3.2k-step run, warming β3→0.999 over the *entire* run means the
  slow EMA barely engages until late. Verify it's worth keeping vs. a shorter
  α/β3 warmup (`CURRENT_HYPERPARAMETERS §6`).

## Segmentation (Clariden 12 h cap)

- Measured 16-node CXI real-data steady state (2026-06-10, job `2515841`):
  iterations 2-10 median `8.5598 s`, mean `8.6299 s`.
- Measured overheads:
  - three-set held-out validation event: about `11 s` (job `2515891`);
  - checkpoint save: about `22.17 s` (job `2515966`);
  - segment startup/load allowance: about `95 s`.
- Estimated allocated runtime per arm:
  - training: `3218 * 8.63 s = 7.71 h`;
  - validation: `128` events at about `11 s` = `0.39 h`;
  - saves: `27` interval saves at about `22.17 s` = `0.17 h`;
  - four segment startups: about `0.11 h`;
  - total: about `8.3-8.5 h`, excluding Slurm queue wait and sidecar
    benchmark jobs.
- Each segment passes the **same full-run `TRAIN_TOKENS`** (anchors WSD) and is
  capped by `--exit-interval=952` (= 8×`SAVE_INTERVAL`; rank-deterministic
  so all ranks exit at the same absolute iteration — no SIGUSR2 signal race). A
  checkpoint lands at the boundary; the next segment resumes from it.
  `#SBATCH --signal=SIGUSR2@600` + `--exit-signal-handler` stays as a walltime
  backstop. The submitter launches `N_SEGMENTS=4` (extras no-op once train completes).
- `--override-opt_param-scheduler` makes every segment use the command-line WSD
  args (not the checkpoint's), so the schedule is continuous across resumes.
