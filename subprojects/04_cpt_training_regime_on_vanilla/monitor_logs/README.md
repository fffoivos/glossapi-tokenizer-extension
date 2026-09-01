# 04 / monitor_logs — checkpoint handoff timestamps

> **In one line:** the local watcher's state directory, holding one line per checkpoint recording the moment its eval sidecars were all verified complete.
> **Period:** 2026-05-28 → 2026-05-30. **Status:** historical machine state.
> **Parent:** [`../README.md`](../README.md)

`../scripts/watch_checkpoint_sidecar_verification.sh` ran read-only on the workstation, calling `../scripts/verify_checkpoint_sidecars.py` for each planned checkpoint and writing an `iter_<N>.handoff_done` marker once the hardened gate passed — checkpoint metadata present, all expected sidecar outputs non-empty, Slurm jobs completed, checksum manifest written. A checkpoint's numbers were not allowed into a report or an adversarial review before its marker existed.

The five markers under `04_vanilla_goldfish_5b_20260528T112539Z_sidecar_verify_state/` are the run's evaluation cadence:

| Checkpoint | Tokens | `handoff_done` (UTC) |
|---|---|---|
| iter 119 | 0.5 B | 2026-05-28T21:28:38Z |
| iter 238 | 1.0 B | 2026-05-29T00:47:26Z |
| iter 477 | 2.0 B | 2026-05-29T11:14:32Z |
| iter 834 | 3.5 B | 2026-05-29T23:42:43Z |
| iter 1192 | 5.0 B | 2026-05-30T15:33:56Z |

The corresponding verifier payloads live in [`../reports/`](../reports) as `iter_<N>_checkpoint_sidecar_handoff_pass.json` and `iter_<N>_checkpoint_sidecar_verify_latest.json`; the review-side watcher kept its own state under [`../adversarial_reviews/`](../adversarial_reviews). Nothing here is a result.
