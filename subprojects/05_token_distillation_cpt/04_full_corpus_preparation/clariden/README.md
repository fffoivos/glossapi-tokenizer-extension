# 04/clariden — Slurm launchers for the CSCS CPU work

> **In one line:** three generations of dry-run-first Clariden submitters — the numbered v2 production DAG (`00`–`99`), the v3 ordered lane, and the v4 raw-review lane — of which only the v4 lane and the separate v5 launcher under [`../slurm`](../slurm) ever ran production work.
> **Period:** 2026-07-11 → 2026-07-15. **Status:** completed as code; the v2 and v3 launchers were never executed end to end.
> **Came from / led to:** [`../RUNBOOK.md`](../RUNBOOK.md) describes how these were meant to be driven → [`../slurm`](../slurm) is where the lane that actually ran lives.

## Why this existed

Clariden has no CPU-only production partition for project `a0140`: `normal` and `debug` allocate exclusive 288-core GH200 nodes and `xfer` is transfer-only. Every corpus job therefore had to request CPU-side capacity on a GPU node without GRES, keep ~256 cores busy, fit a 12-hour allocation (85 minutes on `debug`), and be resumable from receipts. These wrappers encode that, plus a hard rule that preparing a script never authorises a download, a submission or a policy change.

## History

| Date | What happened | Evidence |
|---|---|---|
| 2026-07-11 | First launchers: `prepare.sh` (local validation only), `bootstrap_runtime.sh`, `build_detector.sh`, `00_acquire_sources.sbatch`, `10_quality_audit.sbatch`, `20_structural_detect.sbatch`, `30_structural_token_loss.sbatch`. | `26162a1c`, `1d3b71f4` |
| 2026-07-12 | The full receipt-bound DAG landed: normalize `40` → dataset quality `41` → lineage `42` → review packet `44` → aggregate `46` → clean `60` → post-clean packet/aggregate `62`/`64` → finalize `66` → GreekMMLU freeze `68` → decontaminate `70` → dedup `80` → materialize/validate `90` → publish `99`, plus MDC acquisition `02`/`08`/`09` and receipt merge `03`. `submit.sh` gained `chain-to-review`, `chain-after-admission`, `chain-after-post-clean`, `chain-structural-to-audit`, `chain-finalize-noop` and `chain-finalize-promoted`. | `14d803fb`, `01cba0ee`, `8a9efebd`, `3d063bfd`, `309d37f3` |
| 2026-07-12 | Structural stages `52`/`53`/`54` and the joint classifier-selection bridge were wired in behind the frozen `audit_only` policy. | `9014a705`, `074aa621` |
| 2026-07-13 | `agent1_v3_*` — a parallel lane with its own `paths.env`, run-ID grammar (`agent1-full-corpus-v3-<UTC>-<sha>`) and a `stage.sbatch` that refuses to reuse the v2 pipeline identity. Real submission additionally requires `CONFIRM_CLARIDEN_CPU_EXCEPTION=REQTRES_NO_GPU`, because `normal` nodes report physical GPUs in `AllocTRES` even for a GPU-free `ReqTRES`. | `3a887c36`, `17c36de6`, `f3bd8b20`, `528497f3` |
| 2026-07-13 → 07-14 | `agent1_v4_*` — the raw-review submitter (`bootstrap-runtime`, `freeze`, `sample`, `validate-responses`, `build-site`, `validate-human-gate`, `profile-fields`, `materialize-envelope`). `sample` builds the 18 × 20 packet and never invokes a model; reviews ran on the authenticated Mac and only a validated response JSONL comes back. | `bf81861a`, `372a837d`, `894d7660` |
| 2026-07-15 | `07_rehydrate_span_silver.sbatch` added for the recovered `STRUCT_2K` LLM-silver handoff. | `9014a705` lineage; see [`../STRUCTURAL_SPAN_PRODUCTION.md`](../STRUCTURAL_SPAN_PRODUCTION.md) |

## Outcome

- **Gating that held:** every submission is a dry run unless `CONFIRM_LAUNCH=1`; a payload download also needs `CONFIRM_ACQUIRE=1`; a structural no-op needs `CONFIRM_STRUCTURAL_NOOP=1`; a promoted apply needs the exact Stage54 model-receipt SHA-256 and never falls back to a no-op; publication needs `CONFIRM_PUBLISH` equal to the target repository ID and is never part of a chain.
- **A completed stage is a `stage_receipt.json` plus a `COMPLETED` marker.** An incomplete directory can only be re-entered through `submit.sh resume <stage>`, which revalidates every reused byte. `stage_contract.py` is the shared, stdlib-only implementation of that contract.
- **Unexecuted:** no run of the numbered v2 DAG or the v3 lane is recorded anywhere in this tree. Read `submit.sh --help` as a specification of an intended operation, not as history.
- The `debug`-partition constraints that these wrappers had to fight (one running job per user, two submitted, 85-minute wall) are what made the v5 lane's self-chaining design in [`../slurm`](../slurm) necessary.

## Where things are

| File | Role |
|---|---|
| `prepare.sh` | Local-only gate: validates configs, compiles Python, checks shell syntax, runs the Rust and pytest suites. No SSH, download or submission. |
| `submit.sh` | The v2 dispatcher — stage names, chains, `status`, `resume`. |
| `paths.env`, `common.sh` | Shared roots and helpers for the v2 lane; `agent1_v3_paths.env` / `agent1_v4_paths.env` and `agent1_v3_common.sh` / `agent1_v4_common.sh` are the isolated equivalents. |
| `stage_contract.py` | Stage receipt/marker contract shared by every sbatch. |
| `00`–`99` `*.sbatch` | The v2 production DAG, in execution order. |
| `agent1_v3_submit.sh`, `agent1_v3_stage.sbatch`, `agent1_v3_*.sh` | The v3 ordered lane. |
| `agent1_v4_submit.sh`, `agent1_v4_stage.sbatch` | The v4 raw-review lane. |
