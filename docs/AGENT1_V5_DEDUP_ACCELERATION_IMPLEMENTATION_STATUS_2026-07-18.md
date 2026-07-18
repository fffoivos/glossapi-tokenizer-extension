# Agent 1 v5 dedup acceleration implementation status

**Status:** implementation and remote validation complete; full input audit running; live cutover safely gated.

## Immutable implementation artifacts

- Local commits:
  - `730b6ac` — receipt-bound full audit, accelerated signature worker,
    benchmark/selection logic, held-array release authorization, and cutover
    controller.
  - `2e9150a` — correct five-minute bandwidth measurement from sampled byte
    deltas rather than treating Slurm byte counters as rates.
- CSCS immutable deployments:
  - `/capstor/scratch/cscs/fffoivos/agent1-v5-code/730b6ac`
  - `/capstor/scratch/cscs/fffoivos/agent1-v5-code/2e9150a`
- Active implementation root:
  `/capstor/scratch/cscs/fffoivos/agent1-v5-code/2e9150a/subprojects/05_token_distillation_cpt/04_full_corpus_preparation`

The deployed tests directly passed under the pinned CSCS runtime (the runtime
does not package `pytest`): receipt/output closure, chunk-plan approval, and
the deterministic five-worker benchmark selection path. All new Bash wrappers
also passed `bash -n`.

## What is executing now

The independent one-time full input audit is job `2790548`:

```text
partition: normal
account: a0140
job name: a1v5-dedup-full-audit
output: <run>/dedup_full_input_audit.json
```

It SHA-256 validates each release input exactly once. It does not write a
signature receipt, alter a Parquet file, or interact with the live serial
chain.

## Cutover safety gate

The planned held-fence migration requires the *effective QoS of the running
legacy job* to impose `MaxJobsPU=1` and `MaxSubmitJobsPU=2`. The scheduler
snapshot at implementation time instead showed:

```text
debug partition configured QoS: debug-qos (1 running / 2 submitted)
active chain job: 2790404, a1v5-signature-chain-r52
active chain job effective QoS: normal
normal QoS per-user submission limits: unset
```

Thus a held `debug-qos` job cannot reliably make the legacy helper's successor
submission fail: the helper itself runs with `normal` QoS. The new preflight
and cutover scripts hard-fail on this mismatch and do not submit a fence,
cancel a job, or modify the legacy helper.

This is a scheduler-policy blocker for *cutover only*, not for construction,
the full input audit, or the continuing serial chain.

## Required activation decision

To activate the automatic cutover as designed, CSCS must provide an effective
QoS for the legacy chain which both permits the debug partition and enforces
the documented one-running/two-submitted user limit. The alternative requires
explicit approval to change the immutable legacy chain helper to honor a
receipt-bound stop sentinel before submitting its successor; this changes the
previous no-overwrite constraint and must not be inferred.

Until one of these is resolved, the correct operation is to keep the serial
chain running and retain the completed audit/code artifacts for the safe
cutover.
