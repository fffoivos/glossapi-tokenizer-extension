# 04 / goal — the locked spec the run was held to

> **In one line:** three files that fixed the experiment before launch — scope, machine-readable hyperparameters, and (added mid-run) the canonical eval-task list — and that every report in this subproject cites as its source of truth.
> **Period:** 2026-05-28 (pre-launch) → 2026-06-11 (last link edit, `a19c136f`). **Status:** frozen; superseded as *forward* guidance by [`../_archive/superseded_drafts/task1_20260601/TASK2_HANDOFF.md`](../_archive/superseded_drafts/task1_20260601/TASK2_HANDOFF.md).
> **Parent:** [`../README.md`](../README.md)

## Why this existed

The rule for the run was that sbatch generation reads `hyperparameters.json`, never prose — so a spec drift shows up as a diff rather than as an unexplained number. [`goal.md`](goal.md) states the fixed scope (base 131,072-token tokenizer, HPLT-only Greek `ell_Grek_ge8_no_mt_clean60`, B1 replay at 70/24/4/2, no GlossAPI Greek, no extension, no staged curriculum), the locked settings, the six execution steps, the required artifacts, and the stop conditions that would have killed the run.

## History

- **2026-05-28, pre-launch.** `goal.md` + `hyperparameters.json` locked; every sbatch derived from them. The RUN_LOG carries repeated "Goal Hyperparameter Reference" checks confirming the live training command matched the file.
- **2026-05-29.** `hyperparameters.json` was **corrected**: `base_model.architecture` had been describing a mix of Path A and Path B that matched neither the released base nor the training command. It now carries Path A (`rope_theta=12000000`, `max_position=65536`, llama3 scaling) in `base_model.architecture`, a new top-level `training_geometry` block holding the Path-B override actually used plus `cost_of_path_b` and the `matched_config_diagnostic` status note, and `task2_geometry_recommendation` = Path A (RUN_LOG §"Matched-Config Base Eval Complete + Geometry Docs Fixed"). `../reports/config_geometry_audit_iter_0000119.md` preserves the pre-fix state.
- **2026-05-30, 16:00 UTC.** [`canonical_eval_tasks.json`](canonical_eval_tasks.json) (`canonical-eval-tasks-v1`) added, because the retention sidecar was emitting 201 upstream task entries of which only 12 were in scope. It pins the 3-task Greek MCQ headline, Plutus as diagnostic-separate, the four MT-derived Greek tasks as excluded, the en/fr/de/ru retention set (and flags `xstorycloze_en` / `xstorycloze_ru` as absent on disk, leaving Russian covered by `xnli_ru` alone), and the three heldout BPB sets. The status renderer and the reports filter through it.
- **2026-06-11 (`a19c136f`).** `goal.md`'s `cpt-plan.md` link was repointed into `_archive/`, and its closing note was reworded to say later runs keep this file as the fixed regime baseline while running their own goals elsewhere. A frozen copy of all three files sits at [`../_archive/superseded_drafts/task1_20260601/goal/`](../_archive/superseded_drafts/task1_20260601/goal); it differs from the live copy only in those two edits.

## Outcome

- The locked regime — Goldfish `k=h=50` (seed 2971215073), LR `1.1e-5` warmed over 1.2 B tokens (287 steps) then constant, AdEMAMix `β1/β2/β3/α = 0.9/0.999/0.99/8`, 4096 seqlen, 4.194 M-token global batch, TP=2 on one GH200 node, bf16 with fp32 master grads — is the artifact Task 2 was told to inherit.
- `production_blockers_status` records four gates and their disposition: V1 decontamination (`not_required_for_diagnostic`, and it stayed undone), V4 bootstrap CIs (done — `../reports/v4_bootstrap_cis_native_mcq.json`), V8 Goldfish hash uniformity on an extended vocab (untested here; production-blocking for Task 2), R17 Apertus-extras patch (applied pre-experiment; the init checkpoint is the R17-patched TP=2 Vanilla).
- `extension_specific_settings` is null throughout — this arm has no extension, and filling it was explicitly left to Task 2.
- What the goal asked for and did **not** get: `codex exec` reviews (codex was offline; Claude Code subagents substituted), and the 10 B continuation, which `goal.md` gates on the 5 B report and which was never launched.

## Where things are

| File | What it is |
|---|---|
| [`goal.md`](goal.md) | Scope, locked settings, execution steps, required artifacts, stop conditions. |
| `hyperparameters.json` | The authoritative machine-readable spec; 20 top-level blocks including `training_geometry`, `eval`, `compute`, `production_blockers_status`. |
| `canonical_eval_tasks.json` | The eval-task lockdown; the answer to "which of the 201 task entries count". |
