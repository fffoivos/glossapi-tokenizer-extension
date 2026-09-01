# 04 / _archive — superseded docs and bootstrap scratch

> **In one line:** the plan, handoff and probe-plan documents that were moved out of the subproject root when it closed, the draft the final 5 B report replaced, and the five throwaway workspaces where the bootstrap CIs were actually computed.
> **Period:** archived in two moves — 2026-05-30 (drafts + workspaces) and 2026-06-11 (`a19c136f`, the root docs). **Status:** historical; kept for reproducibility and provenance.
> **Parent:** [`../README.md`](../README.md)

## Why this existed

Two different kinds of clutter needed to survive without staying in the way. First, the in-flight agent workspaces: each bootstrap CI in the reports was computed by a short-lived agent in its own scratch directory, and deleting them would have left the numbers unreproducible. Second, once Task 2 moved to [`../../05_token_distillation_cpt`](../../05_token_distillation_cpt), the planning-level documents at the root of this subproject stopped being live guidance and became history — but they are still the best narrative account of the whole program at that point, so they were moved rather than deleted.

## History

- **2026-05-30.** As part of the post-endpoint cleanup, `5B_REPORT_DRAFT.md` was superseded by `../reports/5B_REPORT.md` and moved to `superseded_drafts/`, and five bootstrap workspaces were moved to `v4_workspaces/` "preserved for reproducibility" (RUN_LOG §"04-folder Organization + Retention Baseline Correction + Task-2 Handoff").
- **2026-06-11 (`a19c136f`).** `cpt-plan.md`, `TASK2_HANDOFF.md`, `PATH_A_GEOMETRY_PROBE_PLAN.md` and a frozen copy of `goal/` were moved from the subproject root into `superseded_drafts/task1_20260601/`, and the parent README was repointed at the new paths. The archived `cpt-plan.md` is its final state, including the §3.2.1 layer-11 evidence audit added on 2026-06-01; the archived `goal/goal.md` differs from the live [`../goal/goal.md`](../goal/goal.md) only in a link target and a reworded closing note.

## Outcome

- `cpt-plan.md` v1.0 remains the plan of record for the whole program at that date: §1 bakeoff background, §2 Task 1 (this run), §3 Task 2 (extension, with §3.4 Q3.4.10 = the Path-A recommendation and §3.2.1 = the layer-11 audit), §4 Task 3 data mix, §5 commitments, §6 non-commitments — the section that explicitly refused to pre-commit thresholds.
- `TASK2_HANDOFF.md` is the most useful single document in the archive: what Task 1 established, 17 numbered errors with how each was caught and recovered, 11 recommendations, and 6 questions Task 1 could not answer. §3.1 carries the status flip to "CONFIRMED, LOCKED" for Path A on 2026-05-31.
- `PATH_A_GEOMETRY_PROBE_PLAN.md` is a pre-registered plan with its decision rule written before the run — §7's two prongs are what the probe was later judged against.
- The workspaces are the computation of record for numbers quoted throughout `../reports/`: `v4_workspace_v2_genesis` (the V4 v2 emit), `v4_workspace_iter834` (the paired iter-477-vs-iter-834 plateau CI and iter-834-vs-bakeoff), `v4_workspace_iter1192` (the five endpoint CIs), `v4_workspace_plutus` (marginal + paired Plutus CIs, item-level diff), `v4_workspace_v3` (the V4 v3 re-emit). Prediction JSONLs pulled from Clariden were deleted after computation by design; the scripts and result JSONs remain.

## Where things are

| Path | What it is |
|---|---|
| [`superseded_drafts/task1_20260601/cpt-plan.md`](superseded_drafts/task1_20260601/cpt-plan.md) | The experimental plan v1.0, in its final state (62 KB). |
| [`superseded_drafts/task1_20260601/TASK2_HANDOFF.md`](superseded_drafts/task1_20260601/TASK2_HANDOFF.md) | Task 1 → Task 2 handoff; errors, recommendations, open questions. |
| [`superseded_drafts/task1_20260601/PATH_A_GEOMETRY_PROBE_PLAN.md`](superseded_drafts/task1_20260601/PATH_A_GEOMETRY_PROBE_PLAN.md) | The probe's pre-registered plan and decision rule. |
| `superseded_drafts/task1_20260601/goal/` | Frozen copy of the goal spec as of the archive move. |
| [`superseded_drafts/5B_REPORT_DRAFT.md`](superseded_drafts/5B_REPORT_DRAFT.md) | The live draft, with its completion gates still visible; superseded by `../reports/5B_REPORT.md`. |
| `v4_workspaces/*/run_*.py` + `*.json` | Bootstrap drivers and their outputs, one directory per computation. |
