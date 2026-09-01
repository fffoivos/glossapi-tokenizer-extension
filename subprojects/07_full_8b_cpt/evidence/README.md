# 07 · evidence — one disclosure

> **In one line:** a single document that withdraws an earlier claim about DP32 restart parity and downgrades it to "numerically continuous".
> **Period:** 2026-08-06 (commit `14a61ca0`). **Status:** frozen; still governs how the run's recovery evidence may be described.
> **Came from / led to:** the restart tolerances added in `71d1bba2` → this disclosure → the synchronous-save parity gate of 2026-08-07 (`3d1c015b`).

## Why this existed

Commit `71d1bba2` made the restart gate topology-aware and, in doing so, added `restart_gradient_norm_atol 0.001` and `rtol 0.02` to [`../configs/execution_profiles.json`](../configs/execution_profiles.json). Those tolerances were written **after** the DP32 restart result already existed and before its promotion receipt was frozen. Rather than leave that ordering implicit, it was written down.

## What [`DP32_RESTART_ACCEPTANCE_DISCLOSURE_20260806.md`](DP32_RESTART_ACCEPTANCE_DISCLOSURE_20260806.md) records

- The threshold "must therefore never be cited as an independently predeclared acceptance criterion or as proof of bitwise equality."
- The frozen receipt at update 161: uninterrupted and restarted loss both `2.204937`; parameter norm both `7142.029`; gradient norm `0.873` vs `0.881` — an absolute delta of `0.008`, about `0.916%`. DP64 matched exactly on all three logged fields.
- The earlier explanation attributing the DP32 difference to collective reduction order is **withdrawn as unsupported**; the cause of the logged difference is not established.
- The parallelism choice is unaffected: DP64 was rejected independently on trajectory drift, and production used the DP32 control profile.
- The run's recovery evidence must be described as **numerically continuous, not bitwise-exact**, and every production segment boundary must still pass its checkpoint and loss-continuity audit.

## Outcome

The next day this disclosure got a sequel: the v35 sanitized benchmark showed two independent DP32 restarts matching *each other* exactly while both differed from the uninterrupted update-161 gradient norm (3.202 vs 2.210). That benchmark was quarantined, asynchronous save was forbidden at resumable boundaries, and a fresh synchronous-save parity smoke became a hard gate ([`../README.md`](../README.md), 2026-08-07).
