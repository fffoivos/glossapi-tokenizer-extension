# 04/slurm — the Agent 1 v5 execution and dedup-acceleration layer

> **In one line:** the twenty shell scripts that actually drove the corpus build on Clariden `debug` nodes, and the receipt-bound package built to make its deduplication 4–5× faster — which was implemented and validated but whose automatic cutover was blocked by a scheduler-QoS mismatch.
> **Period:** 2026-07-15 → 2026-07-20. **Status:** completed as code; the *automatic* held-fence cutover never fired, and the accelerated arrays that followed were submitted, refilled and recovered through explicit operator steps (`9ad9fe30`, `2fb9d841`, `8c5969e6`).
> **Came from / led to:** [`../docs/agent1_v5_eiger_pipeline.md`](../docs/agent1_v5_eiger_pipeline.md) and [`../scripts`](../scripts) → these wrappers → the published corpus.

## Why this existed

The v5 run was approved for Clariden `debug`, where `debug-qos` allows one running and two submitted jobs per user and an 85-minute wall. Everything here follows from that: arrays throttled to four concurrent nodes, per-task immutable receipts, self-chaining submitters, and canaries before each expensive stage. Directory: `agent1_v5_eiger/` (named for the originally intended cluster; the run moved to Clariden, `33a3be81`).

## History

| Date | Script(s) | What it did | Evidence |
|---|---|---|---|
| 2026-07-15 | `stage.sh`, `bundle.sh`, `stage_acquisition_xfer.sh`, `clariden_debug_stage.sh`, `clariden_debug_bundle.sh` | The base execution layer: one stage or a bundle of tasks per job, plus the `xfer`-partition staging that copied 18 sources + the NanoChat base to Capstor and re-hashed every file against its pinned LFS SHA-256. | `c144116c`, `33a3be81`, `43451b41`, `b0845e0b`, `0b164d15`, `30c72e99` |
| 2026-07-18 | `audit_release_quality.sh`, `audit_signature_inputs.sh` | The receipt-bound candidate quality audit, and the one-time full input audit that replaced re-validating all 149.7 GB before every signature rank (job `2790548`). | `cad947b4`, `730b6acd` |
| 2026-07-18 → 07-19 | `normal_signature_benchmark.sh`, `submit_signature_benchmark.sh`, `capture_acceleration_scheduler_snapshot.sh`, `capture_signature_array_evidence.sh` | The immutable 1→2→4→5 local-worker benchmark on one `normal` node, submitted as a nonce-bound held job and released only after identity proof; plus live scheduler and `sacct` evidence capture. | `730b6acd`, `2e9150a9`, `d2fe7703`, `f5cd0b6c`, `3c573a53` |
| 2026-07-19 | `arm_signature_sentinel_takeover.sh`, `cutover_to_accelerated_signatures.sh`, `finalize_signature_sentinel_handoff.sh` | The held-fence migration: arm a checksum-bound sentinel, fence the legacy chain **only** if its effective QoS provides the documented two-submission boundary, then close the handoff once the serial queue is empty. It never cancels a running job. | `ee1e2743`, `06fb8baf`, `03ca511d`, `59f5840e`, `9a103e80`, `edafc327` |
| 2026-07-19 → 07-20 | `submit_accelerated_signature_array.sh`, `normal_signature_runner.sh`, `normal_bounded_stage_runner.sh`, `bounded_stage_runner.sh`, `finalize_accelerated_signatures_and_merge.sh` | Held normal-partition arrays with nonce-bound release, bounded local concurrency, worker refill as ranks finish, safe recovery of failed arrays, and closure of an exact array before its manifest merge. | `1c5db108`, `8c5969e6`, `2bf94d92`, `82c5e40a`, `3d41fd15`, `6d779c00`, `9ad9fe30`, `2fb9d841`, `a1e7cde3` |
| 2026-07-20 | `pair_merge_capacity_canary.sh` | Measured node-local capacity for the LSH pair merge, then promoted the pair database off node-local storage. | `4eeda9dc`, `1769da6d` |

## Outcome

- **Measured problem:** at the rank-44 boundary the serial design implied ~9.82 days of remaining signature work and 57.8 TB of redundant validation reads; the acceleration package projected 42–49 hours instead, with dedup semantics byte-identical ([`../../../../docs/AGENT1_V5_CSCS_DEDUP_ACCELERATION_PLAN_2026-07-18.md`](../../../../docs/AGENT1_V5_CSCS_DEDUP_ACCELERATION_PLAN_2026-07-18.md)).
- **Blocker recorded rather than worked around:** the held-fence design needed the *running* legacy job to sit under a QoS enforcing `MaxJobsPU=1` / `MaxSubmitJobsPU=2`, but the live chain job `2790404` ran with `normal` QoS and unset limits. `cutover_to_accelerated_signatures.sh` hard-fails on that mismatch — it does not submit a fence, cancel a job or edit the legacy helper ([`../../../../docs/AGENT1_V5_DEDUP_ACCELERATION_IMPLEMENTATION_STATUS_2026-07-18.md`](../../../../docs/AGENT1_V5_DEDUP_ACCELERATION_IMPLEMENTATION_STATUS_2026-07-18.md)).
- The standing rule enforced by every script here — never cancel a running validated rank, migrate only at a receipt boundary — is the reason no completed signature work was lost across the whole run.
