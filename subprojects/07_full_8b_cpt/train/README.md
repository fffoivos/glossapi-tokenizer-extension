# 07 · train — checkpoint freezing and training-attempt auditing

> **In one line:** three small fail-closed scripts that decide whether a finished 8B training segment counts, and which checkpoint a failed one may resume from.
> **Period:** 2026-08-05 → 2026-08-07. **Status:** completed; five audited training attempts, all clean.
> **Came from / led to:** [`../clariden/train_segment.sbatch`](../clariden/train_segment.sbatch) → these auditors → the supervisors in [`../scripts/`](../scripts/) and the campaign completion receipt.

## Why this existed

The run was six (later five) segments long on a 12-hour partition, advanced by an `afterany` supervisor. That design is only safe if "the segment finished" is a *proved* statement rather than a Slurm exit code, and if recovery can only ever restart from a checkpoint that has itself been verified.

## The three scripts

- [`audit_training_attempt.py`](audit_training_attempt.py) — "fail closed on incomplete, skipped, nonfinite, or under-evaluated training". It is the gate that turns a completed job into an accepted segment; the campaign's completion receipt binds **5 training-attempt audits**.
- [`freeze_checkpoint.py`](freeze_checkpoint.py) — hashes and receipts one exact segment checkpoint. Segment 0's receipt, for example, contains 131 files and 147,638,448,520 bytes with tree SHA-256 beginning `797e3f` ([`../FULL8B_RERUN_LAUNCH_HANDOFF_20260808.md`](../FULL8B_RERUN_LAUNCH_HANDOFF_20260808.md)).
- [`freeze_recovery_checkpoint.py`](freeze_recovery_checkpoint.py) — freezes the latest clean checkpoint *inside* a failed segment, added by `c3c84fdb` ("Recover failed segments only from verified clean checkpoints", 2026-08-06).

## The policy they enforce

From the launch handoff: `TIMEOUT`, `NODE_FAIL`, `BOOT_FAIL`, `PREEMPTED` and `REVOKED` recover only from a verified **synchronous** checkpoint; a `FAILED` job is retried only when a valid recovery checkpoint is proven; replacement training stays 16-node `normal` while supervisors and evaluation stay on `debug`; and DP, micro-batch, accumulation, precision, data order, optimizer and LR schedule may never be changed to rescue an allocation. The synchronous-save requirement is not decoration — asynchronous save was forbidden at resumable boundaries on 2026-08-07 after the v35 benchmark showed a restart gradient-norm mismatch (see [`../evidence/README.md`](../evidence/README.md)).

## Outcome

Five segments, all audited clean, zero skipped and zero non-finite updates across the whole 18,284-update run. The recovery path existed but the happy path held: the operational difficulty in this run was allocation scheduling, not numerical failure.
