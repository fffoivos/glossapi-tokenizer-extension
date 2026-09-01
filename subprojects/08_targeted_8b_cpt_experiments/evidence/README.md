# evidence — Experiment-B allocation-transition receipts

> **In one line:** the only surviving in-repo trace of the retired Experiment-B restart-parity smoke — two receipts recording how a pending 16-node allocation was swapped for a better-placed one.
> **Period:** 2026-08-12 (file dates; committed 2026-08-16 in `de6d9b79`). **Status:** superseded — Experiment B was retired by the owner on 2026-08-12 and never trained.
> **Came from / led to:** [`../CPT_EXPERIMENT_AND_RESOURCE_PLAN_20260811.md`](../CPT_EXPERIMENT_AND_RESOURCE_PLAN_20260811.md) → this → nothing; the hard-H2G study that followed keeps its receipts on CSCS scratch, not here.

## Why this existed

The 08-11 plan required one bounded 16-node `normal` allocation for a three-update restart-parity smoke before any production submission. Capacity on Clariden is leaf-switch-constrained, so a pending request was twice replaced by a better-placed one. Each swap had to be receipted so that "we cancelled a queued job" could never be confused with "we lost or reshaped a scientific asset".

## History

| Date | What happened | Result | Evidence |
| --- | --- | --- | --- |
| 2026-08-12 01:33 | Job `3060203` (16 nodes, 1 h, `--switches=1`, leaf `group29`) was cancelled and replaced by `3060221` on leaf `group36`, whose test-only estimated start was ~1 h 14 m earlier | replacement submitted while the old request was still pending; scientific bundle v33 (`29d05937…`) unchanged | [`restart_smoke_allocation_transition_20260812.json`](restart_smoke_allocation_transition_20260812.json) |
| 2026-08-12 01:41 | `3060221` was in turn cancelled for `3060256`, which lets Slurm auto-select a single leaf instead of pinning one; bundle v34 (`cf3f1119…`) passed full CSCS validation and the nested-submit proof (`3060243`/`3060247`) | estimated start moved earlier again; no leaf hard-pin retained | [`restart_smoke_auto_leaf_transition_20260812.json`](restart_smoke_auto_leaf_transition_20260812.json) |
| 2026-08-12 (same day) | The owner retired Experiment B; pending jobs `3061757`/`3061758` were cancelled at `00:00:00` | the smoke never ran to completion for B; no optimizer update was consumed | [`../CPT_EXPERIMENT_AND_RESOURCE_PLAN_20260811.md`](../CPT_EXPERIMENT_AND_RESOURCE_PLAN_20260811.md), "Implementation status" |

## Outcome

- Both receipts record `status: completed` for the *transition*, not for the smoke — the smoke itself was overtaken by B's retirement.
- The reusable lesson (a pending allocation is not evidence of capacity, and `--switches=1` is a preference Slurm may relax after its wait threshold) was carried into the 08-14 plan's CSCS resource policy.
