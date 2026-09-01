# 10 — Early-cooldown causal experiment

> **In one line:** a single-variable causal branch that would move the WSD-10 cooldown to the update-9,536 GreekMMLU peak, to find out whether cooldown timing *causes* the peak — it never trained: five launch attempts were rejected in two days, all of them by the experiment's own restart-parity gate or by Slurm placement, and the work was abandoned.
> **Period:** 2026-08-12 → 2026-08-14 (25 commits, `7a2d7993` … `19015a8b`, branch `agent/early-cooldown-causal`). **Status:** abandoned before any scientific training; no branch checkpoint, no milestone evaluation and no result document was ever produced.
> **Came from / led to:** [`09_full_8b_cpt_results_analysis`](../09_full_8b_cpt_results_analysis/) (which found the peak) → this → nothing; repository activity moved to [`08_targeted_8b_cpt_experiments`](../08_targeted_8b_cpt_experiments/) on 2026-08-16 (`de6d9b79`).

## Why this existed

Subproject 09 established that GreekMMLU peaks at update 9,536 (56.81 %) and is 1.96 points lower at the terminal checkpoint, while Greek language-model loss keeps improving. One candidate explanation is the learning-rate schedule: the parent run began its 3,657-update `1-sqrt` cooldown at update 14,627, and the peak sits far before it. The experiment asked one question, stated verbatim in [`configs/experiment_contract.json`](configs/experiment_contract.json):

> Does immediate LR cooldown preserve GreekMMLU when updates 9,537 through 13,193 consume the exact same D0 sequence stream as the parent run?

The design was aggressively lean and single-variable. Load the parent's exact update-9,536 checkpoint with full optimizer, scheduler and RNG state; read the immutable parent D0 schedule in *prefix* mode (no copying, repacking, reshuffling, re-deduplication, re-anonymization, decontamination or retokenization); change only the LR trajectory. 3,657 updates, 15.338569728 B token slots, ending at 55.335452672 B. One 16-node `normal` allocation; everything else on `debug`. Checkpoint averaging explicitly disabled.

## History

### 2026-08-12 → 08-13 — two restarts rejected, and a real finding about restart invariants

Both attempts are documented in [`PARITY_FAILURE_DIAGNOSIS_20260813.md`](PARITY_FAILURE_DIAGNOSIS_20260813.md).

- **v1, direct reload** (run root `…/20260812T231500Z-causal-wsd10-v1`). At update 9,537 the restart matched the scheduled target counts and bytes exactly, losses to within 7.5e-5 and parameter norm to logged precision — but the gradient norm was `2.016` against the historical `0.669`. Failed closed.
- **v2, replay from update 8,000** (job `3072568`). It reproduced update 8,001 *exactly*, including gradient norm, then diverged from 8,002 onward, after the first BF16 distributed optimizer step. By update 9,536 LM loss differed from the historical log by 0.002222 and parameter norm by −0.038; at 9,537 the gradient norm was `0.127` versus `0.669`. Target counts and UTF-8 bytes stayed exact throughout, with no skipped or non-finite updates. The gate failed and the branch holder was put in a user hold before it could allocate nodes.

The conclusion — the durable finding of this subproject — is that **a historical gradient norm measured on a different GH200 allocation is not a restart invariant**, and that replaying 1,536 peak-LR updates to reconstruct a checkpoint that already has a complete per-file SHA-256 receipt is wasted allocation. Checkpoint completeness was verified independently: `scripts/inspect_dcp_metadata.py` and `scripts/inspect_checkpoint_common.py` confirmed 230 logical state entries, 18,798 storage entries, 52 optimizer entries, two RNG and two rerun-state entries, the same 131 filenames and identical tensor metadata across the original and replayed checkpoints, all recording iteration 9,536 / 39,996,882,944 tokens / 9,764,864 consumed samples. Notably, the project **did not loosen its numerical tolerance** after either failure.

### 2026-08-13 — the paired gate also fails, by one display quantum

Contract v3 replaced the historical comparison with a same-allocation paired probe: run update 9,537 twice from the same checkpoint on the same nodes, once at parent LR and once with the cooldown. Job `3075352` ran 5 minutes 42 seconds on 16 four-GPU nodes (6.08 `normal` GPU-hours) and **rejected the intervention**: every declared field matched exactly — consumed samples and tokens, global batch, LM/base-token/added-token losses, target counts and bytes, parameter norm, loss scale, NaN and skipped counts — except the displayed gradient norm, `2.011` in the control against `2.010` in the cooldown probe, one quantum of the logger's three-decimal precision. Supervisor job `3075353` classified it as a permanent blocker, submitted no recovery training and no evaluations, and exited nonzero by design. The saved cooldown checkpoint was rejected and never resumed.

[`PROCESS_REVIEW_20260813.md`](PROCESS_REVIEW_20260813.md) is the review of that failure and is the most useful document here. It found that the v3 contract had omitted explicit equality checks for consumed samples, consumed tokens, global batch size and loss scale — those happened to match, but relying on that without a declared check was a gap — and that an exact two-run gradient comparison is not a robust discriminator at three decimals. Rather than widen the threshold on the failed result (explicitly refused: "this avoids selecting a tolerance from the failed intervention itself"), contract v4 predeclared a **peak/cooldown/peak sandwich**: two no-save parent-LR controls around the saved cooldown probe, all three from the same fully hashed checkpoint on one allocation; every non-gradient field exact across all three; the two control gradients allowed to differ by at most one logger quantum; the intervention gradient required to fall inside that independently measured envelope, with exact equality required if the controls agree.

### 2026-08-13 — three days of Slurm placement, compressed into one evening

The sandwich requires all three probes on one allocation, which turned placement into a scientific constraint. The first launch had pinned the request to leaf `group29` by excluding every other leaf — operationally unnecessary and capacity-reducing. The replacement let Slurm choose any eligible leaf under `--switches=1`, mapping the result and failing closed unless exactly one leaf was observed; live job `3076070` proved that **Clariden normalizes `--switches=1` to `Switches=1@00:05:00`**, so after five minutes the preference is relaxed. It started across four leaves and exited before parity or training in 41 seconds. An explicit seven-day switch wait was normalized back to five minutes the same way, and the post-submit audit cancelled jobs `3076846` and `3076847` at zero runtime. The final path hard-excludes every level-0 leaf except `group36` — the leaf that test-only probes selected as earliest-eligible and that the proven DP32 profile already used — so the five-minute relaxation cannot produce another multi-leaf allocation (`85b2a1fd`, `66203d88`).

Verification before the last deployment was thorough: 14 causal contract tests, 205 inherited scheduling and full-8B tests plus 61 subtests, `bash -n` on every shell entry point, every Python source compiled, the static owner-authorized v4 contract passed. Debug job `3076023` froze an 840-file immutable bundle in 54 seconds (tree SHA-256 `f5eb8be3…`), later superseded by v10 with 843 files (`ffdf1b81…`) for the hard-placement path.

A parallel replay-portability chain ran off the training critical path and did complete: packing job `3075053` (2 m 19 s) and finalization `3075072` (40 s) produced 7,604,182 packed replay sequences on an exact 79/20/1 active-token profile; the first upload `3075091` was scheduler-cancelled at a maintenance boundary; a replacement failed in 10 seconds because `uenv` is not installed on Xfer nodes; the first pinned runtime was built on a GH200 debug node and rejected by the Xfer smoke because its aarch64 NumPy wheel could not load on the x86_64 transfer node. The corrected x86_64 runtime was built and twice verified in 2 m 31 s (job `3076545`), upload `3076559` froze private HF revision `9a520bd8…` in 5 m 03 s, hydration `3076562` full-SHA verified 1,039 payload files totalling 124,965,755,323 bytes, and after one over-escaped `bash -lc` fix the production-reader smoke `3076649` passed all checks in 46.68 s.

### 2026-08-13 → 08-14 — the watcher, and the end

Because a 12-hour `normal` allocation cannot be babysat interactively, an allocation-free observer was built and put under launchd: it only reads Slurm and receipts, holds a lock directory, has no scheduler-mutation path, and notifies on milestones and classified stops (`da6db75c` → `019fc877`, then `38a52826`, `1ad5b6c4`, `19015a8b`). [`clariden/com.fffoivos.apertus-early-cooldown-watch.plist`](clariden/com.fffoivos.apertus-early-cooldown-watch.plist) pins the final attempt: run root `…/20260813T205520Z-causal-wsd10-direct-v6`, training job `3076888`, replay-upload job `3076559`, a 30-hour bound.

**That run did not produce the branch.** The watcher's own state file (on the operator's Mac, not in this repository) records the training job starting at `2026-08-13T23:36:01Z` on 16 nodes and, nine minutes later at `2026-08-13T23:45:22Z`, the terminal line:

```
training=FAILED;replay_upload=COMPLETED;parity=failed;terminal=sandwich_restart_control_failed;evaluations=0;native_endpoint=absent;
```

The sandwich gate rejected the restart controls, exactly as its two predecessors had. The last two commits (2026-08-14) only refine the watcher's alerting. No training receipt, no milestone evaluation, no endpoint result and no result document was ever committed here, and the branch working tree is clean at `19015a8b`.

## Outcome

- **The experiment never ran.** Five attempts — v1 direct reload, v2 replay, v3 paired gate, v4 (multi-leaf placement exit), v6 sandwich — produced zero scientific updates of the cooldown branch. Its four planned milestone checkpoints (10,728 / 11,920 / 13,112 / 13,193) and their GreekMMLU + 13-panel evaluations do not exist.
- **Total measured cost** of the one attempt that reached the gate is 6.08 `normal` GPU-hours (job `3075352`); the remaining attempts exited in 41 seconds or less, or never allocated.
- **What it did establish**, and what is genuinely reusable:
  - a historical gradient norm from a different allocation is not a restart invariant, and BF16 distributed-optimizer trajectory drift starts at the *second* replayed update — so replaying to reconstruct a receipted checkpoint is never worth the allocation;
  - the parent update-9,536 checkpoint is complete and receipted at file level (131 files, `.metadata` SHA-256 `0b082069…`), which is why the reconstruction was unnecessary in the first place;
  - Clariden clamps any `--switches=1` preference to five minutes, so single-leaf placement must be enforced by hard node exclusion, verified at runtime, not requested;
  - a three-decimal displayed gradient norm is too coarse to be a fail-closed equality discriminator between two otherwise identical probes — the honest reading of the whole subproject is that its acceptance criterion, not the cluster, is what stopped it.
- **Left open:** the causal question is unanswered. Whether the early-cooldown intervention preserves GreekMMLU is still unknown, and any future attempt needs a reproducibility criterion that is not the logger's display precision.

## Where things are

| What | Path |
| --- | --- |
| The normative numerical and artifact bindings, including the three recorded rejected attempts | [`configs/experiment_contract.json`](configs/experiment_contract.json) (`schema_version: apertus_full8_early_cooldown_contract_v4`) |
| Why the restarts failed and how the gate was redesigned | [`PARITY_FAILURE_DIAGNOSIS_20260813.md`](PARITY_FAILURE_DIAGNOSIS_20260813.md) |
| Launch-process review: the v3 rejection, the contract gaps, the placement corrections, the replay chain | [`PROCESS_REVIEW_20260813.md`](PROCESS_REVIEW_20260813.md) |
| The sandwich gate itself | [`scripts/finalize_branch_restart_control.py`](scripts/finalize_branch_restart_control.py) |
| Checkpoint inspection without loading tensors (the tools that proved completeness) | [`scripts/inspect_dcp_metadata.py`](scripts/inspect_dcp_metadata.py), [`scripts/inspect_checkpoint_common.py`](scripts/inspect_checkpoint_common.py) |
| Launch preparation, freezing and gating | [`scripts/prepare_launch.py`](scripts/prepare_launch.py), [`scripts/freeze_launch_graph.py`](scripts/freeze_launch_graph.py), [`scripts/freeze_test_only.py`](scripts/freeze_test_only.py), [`scripts/verify_launch_gate.py`](scripts/verify_launch_gate.py), [`scripts/finalize_operational_gate.py`](scripts/finalize_operational_gate.py) |
| Slurm audit and scheduler snapshots | [`scripts/audit_submitted_job.py`](scripts/audit_submitted_job.py), [`scripts/capture_scheduler_snapshot.py`](scripts/capture_scheduler_snapshot.py) |
| Training + gate job, supervisor, bundle freeze, prelaunch, evaluation jobs | [`clariden/train_and_gate.sbatch`](clariden/train_and_gate.sbatch), [`clariden/supervise_after_training_debug.sbatch`](clariden/supervise_after_training_debug.sbatch), [`clariden/freeze_bundle_debug.sbatch`](clariden/freeze_bundle_debug.sbatch), [`clariden/prepare_launch_debug.sbatch`](clariden/prepare_launch_debug.sbatch), [`clariden/run_checkpoint_evaluation_debug.sbatch`](clariden/run_checkpoint_evaluation_debug.sbatch), [`clariden/run_native_endpoint_debug.sbatch`](clariden/run_native_endpoint_debug.sbatch) |
| Submission and deployment entry points | [`clariden/submit_experiment.sh`](clariden/submit_experiment.sh), [`clariden/deploy_bundle.sh`](clariden/deploy_bundle.sh) |
| The read-only watcher and its launchd job | [`clariden/watch_early_cooldown_readonly.sh`](clariden/watch_early_cooldown_readonly.sh), [`clariden/watch_early_cooldown_macos.sh`](clariden/watch_early_cooldown_macos.sh), [`clariden/com.fffoivos.apertus-early-cooldown-watch.plist`](clariden/com.fffoivos.apertus-early-cooldown-watch.plist) |
| Contract tests | [`tests/test_contracts.py`](tests/test_contracts.py) |
| The parent run, its checkpoints and receipts | subproject [`07_full_8b_cpt`](../07_full_8b_cpt/) |

Note: the plist embeds an absolute path to a different local worktree and to the operator's log directory; it is a historical artifact of the 2026-08-13 launch and is not portable.

## Working documents

Both prose documents in this directory are dated 2026-08-13 status/diagnosis snapshots and are historical:

- `PARITY_FAILURE_DIAGNOSIS_20260813.md` — the restart-invariant diagnosis and the corrected gate design. Still the most transferable content here.
- `PROCESS_REVIEW_20260813.md` — the launch-process review. Its closing sections describe the v6 relaunch as *queued*; per the watcher record above, that run failed its sandwich gate nine minutes after starting, so the document's forward-looking passages were never realized.

The 2026-08-13 `README.md` this file replaces described the design in the present tense as if it were about to run; nothing in that description was ever executed beyond the gate.
