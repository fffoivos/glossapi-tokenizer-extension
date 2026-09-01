# 05 — Token-Distillation CPT (Task 2)

> **In one line:** the Greek continued-pretraining track — a 13.5 B two-arm pilot that
> proved the Greek-extended tokenizer + Token-Distillation recipe, five 13.5 B
> hyperparameter sweeps that froze the production recipe, the corpus/dataset machinery
> that fed the full 8 B run, and a final three-way cooldown-floor experiment; the 25 B
> probe it was all building toward was prepared but never trained.
> **Period:** 2026-06-10 → 2026-08-12 (commit dates on this branch; the earliest artifacts
> here were produced 2026-06-02 and committed later, and two units were recovered from an
> uncommitted working tree on 2026-09-01).
> **Status:** completed / superseded — the recipe froze on
> [2026-07-11](PRODUCTION_HYPERPARAMETERS_DECISION_20260711.md) and execution moved to
> [`../07_full_8b_cpt`](../07_full_8b_cpt).
> **Came from / led to:** [`../04_cpt_training_regime_on_vanilla`](../04_cpt_training_regime_on_vanilla)
> → **this** → [`../06_dataset_scheduling_experiments`](../06_dataset_scheduling_experiments)
> + [`../07_full_8b_cpt`](../07_full_8b_cpt).

## Why this existed

Subproject 04 had shown that a corrected CPT regime (AdEMAMix, Goldfish, WSD, explicit
replay) recovers Greek on the *vanilla* tokenizer. Task 2 asked the next question: does
extending Apertus-8B's vocabulary with 17,408 modern-Greek tokens and initializing the new
rows by Token Distillation actually beat the vanilla control at a real training scale — and
if so, what mix, learning rate and optimizer settings should the production run use? This
directory holds the pilot that answered the first question, the sweeps that answered the
second, and the corpus/dataset builds that turned the answers into a launchable dataset.

## History

### 2026-06-02 → 06-09 · sizing and dataset build

The decontamination scanner was first measured on a 5 B-token slice (Slurm `2453791`,
1 h 13 m, 872 rows/s) and scaled to the full HPLT + GlossAPI pool — ~30 h reserved for
~61.6 B tokens ([`01_decontamination_runtime_scaling/`](01_decontamination_runtime_scaling/README.md)).
Earlier 5 B TD runs from this period were diagnostics, not the recipe; their launcher
survives in [`scripts/`](scripts/README.md) (`submit_td_17k_5b_chain.sh`, Path-A geometry).
On 2026-06-09 the two-arm 13.5 B dataset was built and validated: 10 B new Greek
(70 % HPLT / 30 % OpenArchives) plus replay measured against new Greek — 24 % multilingual,
4 % code, 2 % math, 5 % Greek replay ([`LOG.md`](LOG.md), `dataset_build/bulk_13b.json`).

### 2026-06-10 · the CXI blocker, then launch

Multi-node Megatron died before iteration 1 with `NET/OFI ... NO_SPACE`. Buffer sizing,
Socket fallback and smaller node counts were all tried; the actual cause was the trainer
forcing `NCCL_NET_FORCE_FLUSH=1`. With it disabled, 2-, 4- and 16-node CXI smokes passed
(`102ac8a6`), and four 16-node smokes fixed the timing budget — 8.63 s/iter steady,
~22 s per checkpoint save, 8.3–8.5 h/arm (jobs `2515665`/`2515841`/`2515891`/`2515966`,
`bd0a78e8`, [`ARCHIVE.md`](ARCHIVE.md)). The doc tree was collapsed the same day into
`RUNBOOK` + `LOG` + `ARCHIVE` (`3bbe36ba`), and both arms launched as
`STAMP=20260610T200344Z` in four dependency-chained segments each.

### 2026-06-11 · pilot completes, and the curriculum claim is retracted

Both arms reached iteration 3218/3218 with **0 NaN and 0 skipped iterations**
([`LOG.md`](LOG.md)). The report — [`reports/`](reports/README.md) — gives the headline:
native Greek MCQ micro-accuracy **48.3 % base → 55.3 % vanilla → 58.7 % TD** over 18,489
questions, at tied bits/byte and **−31 % tokens** to encode Greek, with English MMLU up
(56.2 → 59.5). Immediately afterwards a review found that the Stage-C physical
HPLT→OpenArchives ordering **never executed**: Megatron randomized sample consumption, so
the run is the intended *mixture* with shuffled order, not a curriculum (`47601092`;
the fix, `CURRICULUM_ORDER_MODE=physical_order` + a no-shuffle patch, landed in `d06b1ac4`).
[`ROADMAP_20260611.md`](ROADMAP_20260611.md) then declared the pilot's job done, showed
~55 B unique Greek exists against the ~10 B the pilot used, and proposed a 60 B run gated
on a 25 B probe. [`EPISTEMIC_PLAN.md`](EPISTEMIC_PLAN.md) narrowed evaluation to GreekMMLU
plus held-out LM loss and deliberately **deferred** the distribution-shift study as
wrong-scale at 13.5 B.

### 2026-06-11 → 06-17 · five sweeps at 13.5 B

All in [`03_training_experiments/curriculum_sweeps_v2/`](03_training_experiments/curriculum_sweeps_v2/README.md),
on a rebuilt two-phase dataset, run through a nine-hour Clariden SSH/maintenance outage:

| Sweep | Grid | Chosen | Evidence |
|---|---|---|---|
| Replay (2026-06-12) | R ∈ {0.35, 0.25, 0.15} + vanilla control | **79/20/1** (R=0.25, old-Greek 5 %→1 %) | [`PRODUCTION_MIX_DECISION_20260612.md`](PRODUCTION_MIX_DECISION_20260612.md) |
| Peak LR (2026-06-13) | {2.75e-5, 5.5e-5, 8.25e-5, 1.1e-4} | **5.5e-5** | [`PRODUCTION_LR_DECISION_20260613.md`](PRODUCTION_LR_DECISION_20260613.md) |
| AdEMAMix α (2026-06-13) | {0, 4, 8} | **4** (59.48 % vs 56.63 / 57.82) | `curriculum_sweeps_v2/results/alpha_decision_table_20260613.md` |
| AdEMAMix β₃ (2026-06-15) | {0.99, 0.995} + α=4 run as 0.999 | **0.999** (59.48 % vs 57.91 / 57.20) | `curriculum_sweeps_v2/results/beta3_decision_table_20260711.csv` |
| AdEMAMix β₂ (2026-06-16) | {0.99, 0.999} + α=4 run as 0.995 | **0.999** (59.94 %), warmup pinned at 400 it | `curriculum_sweeps_v2/results/beta2_decision_table_20260711.csv` |

Two reversals are worth naming. The ROADMAP's pre-sweep guess of 10–15 % replay (DA3) was
**superseded**: R=0.15 (~13 %) clearly hurt adaptation while R=0.25 and R=0.35 tied, so 20 %
foreign replay was taken as the cheap end of the tie. And the β₃ sweep's original plan — a
2× HPLT (~27 B) dataset built for a longer horizon — was abandoned on 2026-06-15 after
`xfer` went into maintenance; ~70 G of staged scratch was deleted and the sweep re-ran on
the existing 13.5 B binaries (`c98aa5d5`). A duplicate β₃=0.999 arm was launched and then
cancelled once it was shown field-for-field identical to the completed α=4 run.

### 2026-07-11 · freeze and audit

[`PRODUCTION_HYPERPARAMETERS_DECISION_20260711.md`](PRODUCTION_HYPERPARAMETERS_DECISION_20260711.md)
(`305feeb0`) froze the whole recipe and backed it with a mechanical comparability audit:
all 14 informative TD runs reached checkpoint 3218, and within each sweep every run's
normalized `run_metadata.json` fingerprint was identical once the swept axis and run-local
paths were removed (β₂ arms: `72992288d2117774…`). The same commit re-labelled ROADMAP,
EPISTEMIC_PLAN, LOG and RUNBOOK as historical and recorded the evidence limit: Clariden
still holds the runs, checkpoints and eval sidecars, but the `curriculum_v2` Megatron
`.bin`/`.idx` payloads were **deleted**, so none of these runs can be byte-reproduced and
the sweep launchers now fail their data preflight by design.

### 2026-07-11 → 08-01 · corpus, bridge, probe, launch spec

Corpus work ran in parallel throughout and outlived the sweeps:
[`02_corpus_preparation/`](02_corpus_preparation/) (the largest directory in the repo) and
then the tracked production build in [`04_full_corpus_preparation/`](04_full_corpus_preparation/).
[`05_training_dataset_bridge/`](05_training_dataset_bridge/) (2026-07-12) turned the
validated release into Megatron binaries for a single-blend 25 B probe; it was superseded
on 2026-07-31 by [`06_25b_midtraining_probe/`](06_25b_midtraining_probe/README.md), a
receipt-gated two-phase rebuild pinned to the bibliography-cleaned HF v2 corpus and a new
**148,992**-token modern+polytonic tokenizer. On 2026-08-01
[`CPT_LAUNCH_RESOURCE_SPEC_20260801.md`](CPT_LAUNCH_RESOURCE_SPEC_20260801.md) mapped the
whole launch — data, tokenizer, init, recipe, Clariden orchestration — separating upstream
facts, project decisions, published artifacts and as-run receipts, and closed with 13 open
gates and a launch checklist. Its verdict was explicit: the 25 B recipe reaches only ~30.9 %
of the published corpus and is a *diagnostic, not the production run*; a full pass implies
~80.788 B tokens / ~19,262 iterations at 79/20/1.

### 2026-08-01 → 08-02 · the cooldown-floor experiment

[`07_8b_lr_floor_reconstruction/`](07_8b_lr_floor_reconstruction/README.md) rebuilt the
13.5 B HPLT→GlossAPI experiment as a new immutable run on the v2 corpus and the 148,992
tokenizer, then branched at update 2,574 into WSD floors of 10 %, 20 % and 30 % of the
`5.5e-5` peak. It survived two recoveries — an `xfer`/Python-3.6 freezer failure at the
phase boundary, and a resume-only bug where restoring the shared optimizer state put T10's
`min_lr` back into every parameter group so all three arms logged identical LRs. All three
tails finished (644 updates each, 0 skipped, 0 NaN) with frozen terminal receipts.

### The end state: the probe never ran

`06_25b_midtraining_probe/configs/recipe_25b_midtraining.json` is still
`status: frozen_pending_clariden_asset_receipts`, and no training receipt, log or result
for a 25 B run exists anywhere in the repo. Instead the scaling/ordering question was
answered more cheaply by the five-arm 0.5 B study in
[`../06_dataset_scheduling_experiments`](../06_dataset_scheduling_experiments), which picked
**D0 stationary mixing** — retiring the two-phase HPLT→GlossAPI curriculum this subproject
had built. The probe's *preparation* was not wasted: its dataset binaries
(`cpt25b_midtraining/20260731T124000Z-cpt25b-v1`) and its production TD-init checkpoint are
pinned as `source_binary_root` and `initialization` in
[`../07_full_8b_cpt/configs/recipe_8b_full_mixed.json`](../07_full_8b_cpt/configs/recipe_8b_full_mixed.json),
and the same stage fed the LR-floor reconstruction.

## Outcome

- **The tokenizer question is settled.** TD + the Greek-extended tokenizer beats the vanilla
  control at 13.5 B: 58.7 % vs 55.3 % native Greek MCQ (base 48.3 %), tied bits/byte,
  −31 % tokens per unit of Greek text (`reports/cpt_2arm_performance.html`). The vanilla arm
  was dropped from every later run.
- **The infrastructure question is settled.** `NCCL_NET_FORCE_FLUSH=0` unblocked 64-GPU CXI
  training (~394 TFLOP/s/GPU, ~96 % DP efficiency, 8.3 h/arm) and remained the launch recipe.
- **The recipe carried into [`../07_full_8b_cpt`](../07_full_8b_cpt) verbatim:** 79/20/1
  mix, peak LR 5.5e-5 → 5.5e-6 with a fixed 400-update warmup and a final-20 % `1-sqrt`
  cooldown, AdEMAMix (0.9, 0.999, 0.999, α=4), weight decay 0.1, clip 0.1, Goldfish k=h=50,
  4096 seq / RoPE 500 k, global batch 1024 sequences. Compare
  `../07_full_8b_cpt/configs/recipe_8b_full_mixed.json`.
- **What did *not* carry:** the 60 B target (the full run is ~80.7 B active tokens over
  19,248 updates), the two-phase curriculum (replaced by D0 stationary mixing), the
  148,480 modern-only tokenizer (replaced by 148,992 modern+polytonic), and the 25 B probe.
- **Left open:** the deferred distribution-shift / boundary study (EPISTEMIC_PLAN §e) was
  never run at scale; the sweep decisions rest on a single 13.5 B seed per arm with no
  repeats; the replay decision was taken on mid-run checkpoints (see below); the three
  LR-floor tails were never compared in-repo; and the sweeps cannot be reproduced because
  their data binaries were deleted.

## Sub-subprojects

| Dir | Role | Period | Status | Result |
|---|---|---|---|---|
| [`01_decontamination_runtime_scaling`](01_decontamination_runtime_scaling/README.md) | Size the GreekMMLU decontamination scan from a measured 5 B pilot | 2026-06-02 (committed 06-11) | completed | ~30 h reservation for the 61.6 B HPLT+GlossAPI pool; CPU-only |
| [`02_corpus_preparation`](02_corpus_preparation/) | Reusable per-source cleaning, dedup, decontamination and anonymization components | 2026-06-11 → 2026-08-12 | completed | The cleaning library every later build calls |
| [`03_training_experiments`](03_training_experiments/README.md) | The 13.5 B two-arm pilot and the five hyperparameter sweeps | 2026-06-10 → 2026-07-12 | completed | TD ≫ vanilla; the frozen production recipe |
| [`04_full_corpus_preparation`](04_full_corpus_preparation/) | The tracked, audit-first production corpus build (Phase 04) | 2026-07-11 → 2026-07-31 | completed | Validated private training release + license/token accounting |
| [`05_training_dataset_bridge`](05_training_dataset_bridge/) | Phase-04 release → fresh Megatron binaries for a single-blend 25 B probe | 2026-07-12 → 2026-08-07 | superseded by `06_25b_midtraining_probe` | Bridge contracts and tests; its launch path went stale |
| [`06_25b_midtraining_probe`](06_25b_midtraining_probe/README.md) | Receipt-gated two-phase 25 B probe: data, init, smokes, launch gates | 2026-07-31 → 2026-08-01 | prepared, never launched | Its binaries + TD init became the full 8 B run's inputs |
| [`07_8b_lr_floor_reconstruction`](07_8b_lr_floor_reconstruction/README.md) | Rebuild the 13.5 B HPLT→GlossAPI run on current assets and branch it into 3 cooldown floors | 2026-08-01 → 2026-08-02 (recovered 2026-09-01) | completed, unanalysed | Three valid T10/T20/T30 tails with frozen terminal receipts |

## Where things are

| Path | What it is |
|---|---|
| [`PRODUCTION_HYPERPARAMETERS_DECISION_20260711.md`](PRODUCTION_HYPERPARAMETERS_DECISION_20260711.md) | The frozen recipe + the mechanical comparability audit. The authority. |
| [`PRODUCTION_MIX_DECISION_20260612.md`](PRODUCTION_MIX_DECISION_20260612.md) · [`PRODUCTION_LR_DECISION_20260613.md`](PRODUCTION_LR_DECISION_20260613.md) | The two decisions taken by the owner from sweep tables |
| [`CPT_LAUNCH_RESOURCE_SPEC_20260801.md`](CPT_LAUNCH_RESOURCE_SPEC_20260801.md) | The 2026-08-01 evidence map for the 8 B launch: every dataset/tokenizer/init/recipe/orchestration dependency, its provenance class, 13 open gates and the launch checklist. Recovered from an uncommitted tree (`2aec4a66`). Cited by subprojects 06 and 07. |
| [`reports/cpt_2arm_performance.html`](reports/cpt_2arm_performance.html) | The pilot result, self-contained (built by `reports/build_report.py`) |
| `03_training_experiments/configs/common_cpt.env` | The as-run pilot env; later re-pointed at the frozen recipe |
| `03_training_experiments/dataset_build/bulk_13b.json` | The realized 13.5 B pilot mix recipe |
| `03_training_experiments/curriculum_sweeps_v2/results/` | Every sweep decision table + `sweep_config_audit_20260711.json` |
| `06_25b_midtraining_probe/configs/recipe_25b_midtraining.json` · `07_8b_lr_floor_reconstruction/configs/recipe_13b_lr_floor.json` | The two frozen machine recipes: the probe that never launched, and the LR-floor run that did |
| [`../CURRENT_HYPERPARAMETERS.md`](../CURRENT_HYPERPARAMETERS.md) | Program-level training config v1.0, with each value's provenance |

## Working documents

Historical; kept for provenance, none of it is current instruction.

- **Plans / agendas:** [`ROADMAP_20260611.md`](ROADMAP_20260611.md) (the 60 B plan, partly
  superseded — its DA3 replay guess was overturned by the sweep, its 60 B target by the
  ~80.7 B full run, its probe by the 0.5 B scheduling study);
  [`EPISTEMIC_PLAN.md`](EPISTEMIC_PLAN.md) (evaluation policy + the deferred shift study);
  `03_training_experiments/curriculum_sweeps_v2/BETA{2,3}_SWEEP_PLAN_*.md`.
- **Operator runbooks:** [`RUNBOOK.md`](RUNBOOK.md) and
  `03_training_experiments/curriculum_sweeps_v2/RUNBOOK.md` — both carry explicit
  "do not relaunch" notices; the data binaries they need are gone.
- **Logs:** [`LOG.md`](LOG.md) (decision log plus a long poll-by-poll launch narrative) and
  [`EXECUTION_LOG_CURRICULUM_SWEEPS_V2.md`](EXECUTION_LOG_CURRICULUM_SWEEPS_V2.md)
  (2,408 lines, 2026-06-11 → 06-15; every Slurm job id, failure and retry of the sweeps —
  the only record of several intermediate GreekMMLU readings).
- **Archive:** [`ARCHIVE.md`](ARCHIVE.md) — summarizes the ~20 launch/handoff/plan docs
  deleted in the 2026-06-10 cleanup, recoverable from git history.

## Known gaps in the record

- [`PRODUCTION_MIX_DECISION_20260612.md`](PRODUCTION_MIX_DECISION_20260612.md) cites
  `reports/cpt_curriculum_forgetting_learning.html`, which is not in the repo, and its
  "GreekMMLU peak" figures (55.1 / 54.9 / 52.7) match the *iteration-1190* (`curr-5.0B`)
  readings in the execution log, not final checkpoints. The replay sweep's final GreekMMLU
  is not recorded anywhere.
- [`ROADMAP_20260611.md`](ROADMAP_20260611.md) §1 quotes GreekMMLU 48.8 → 55.6 → 59.3; the
  report gives 48.3 → 55.3 → 58.7 for a different (18,489-question, three-benchmark)
  metric. The ROADMAP's GreekMMLU-only figures are not reproduced by any file here.
- The LR-floor experiment and the launch resource spec ran unversioned: both were recovered
  from a local working tree on 2026-09-01 (`2aec4a66`), months after the work, so there is
  no commit-by-commit trail for either.
