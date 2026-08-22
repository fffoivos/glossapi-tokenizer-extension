# Hard H2G full-panel + no-decay execution notes for review

Date: 2026-08-22

Binding plan:
[HARD_H2G_FULL_PANEL_SCORING_AND_NO_DECAY_BRANCH_PLAN_20260822.md](HARD_H2G_FULL_PANEL_SCORING_AND_NO_DECAY_BRANCH_PLAN_20260822.md)

Status: **execution in progress**

This is the append-only reviewer-facing operational record. It separates
scientific decisions, allocation-free checks, scheduler events, execution
receipts, failures, and any deviation from the plan. Commands are recorded in
sanitized form; credentials and private infrastructure tokens are excluded.

## Frozen owner decisions

- Apply Track A full-public 8B scoring and Track B no-decay 8B branch.
- Full 16,632-question GreekMMLU is the corrected primary evaluation.
- Existing 16,159-question results remain a secondary sensitivity analysis.
- The formal historical target is the selected beta2=0.999 arm
  `curr_td_b20p999_b3p999_13b_20260616T093527Z`, not the earlier surviving
  two-arm visualization curve.
- Replication band ratified before A2 results: best legacy-BF16 accuracy must
  be within plus or minus 1.0 percentage point of `9973/16632 = 59.9627%`.
- 1.5B peak-LR search remains deferred.
- B3 is conditional on the measured no-decay trajectory through update 3,218.
- No dataset rebuild, re-deduplication, or anonymization audit is in scope.

## Event log

### 2026-08-22 19:14–19:20 Europe/Athens — intake

- Read the binding plan in full.
- Read the installed `prepare-apertus-experiment` and
  `optimize-apertus-cscs` skills and their launch, benchmark, runner, and live
  cluster references.
- Verified local CSCS certificate validity through 2026-08-23 12:55:26.
- Verified read-only SSH access to `clariden`; no jobs were present for user
  `fffoivos` at 19:16:44.
- Fetched canonical `fffoivos/apertus-cscs-efficiency`; current
  `origin/main` is `ef69de358f8fe2c84d4842be074f117a9199f4df`.
- Read canonical `docs/REPOSITORY_SCOPE.md` and
  `docs/CANONICAL_CAMPAIGN_RUNNER.md` before adding lifecycle code.
- No Slurm job or allocation had been requested at this point.

### 2026-08-22 19:28 Europe/Athens — first allocation

- Allocation-free remote inventory confirmed all 17 8B checkpoint sources,
  the full-state update-2,499 branch checkpoint, and all reusable HF exports.
- Requested a direct foreground `salloc --no-shell` for Track A rather than a
  fire-and-forget batch job.
- Slurm job `3151839`: `normal`, four nodes / 16 GH200 GPUs, four-hour limit,
  job name `h2g-full-public-score`.
- The allocation was granted immediately. Its controlling SSH/salloc session
  remains attached while the immutable scientific bundle and inputs are
  finalized.

### 2026-08-22 20:01–20:12 Europe/Athens — corrected scorer and B2 adapter

- Frozen scientific bundle v4 after correcting nested Slurm step mode; its
  independent verifier reports tree SHA-256 `9940b5e7b314a8cc10e21996557e85b9bec3576640b38ed9d4a778f83e138ecb`.
- Reattached to live allocation `3151839`. All 16 evaluator ranks loaded the
  first checkpoint; the full-public trajectory is now computing.
- Added one explicit LR-policy axis to the existing canonical training
  adapter. `matched_wsd` is the default and preserves the old path;
  `stable_peak` is fail-closed to 8B, Phase 2, update 2,499 to 3,218, and
  5.5e-5. It reuses the already proven scheduler-restore wrapper, which also
  chains into the existing phase-local data-index guard.
- Added an intermediate-save branch materializer. It creates an independent
  load root with a 2,499 tracker and same-filesystem hard links to the exact
  DCP files, not a second checkpoint copy. Its receipt records common/metadata
  bindings, inode identity, file count, bytes, and the training-log prefix
  through the successful 2,499 save. The existing checkpoint audit and permit
  then validate this branch like any other resume source.
- Focused adapter/evaluator tests after these changes: `31 passed`.

### 2026-08-22 20:25 Europe/Athens — intermediate-save log ordering

- The first update-2,499 branch-materialization attempt failed closed before
  audit/permit creation. Megatron's successful-save message for 2,499 is
  asynchronous and occurs after optimizer rows beyond 2,499, so a literal log
  prefix did not end on the selected update. The no-copy checkpoint view and
  failed log were retained under a `_failed_*` source path.
- Correction: preserve every non-optimizer line through the exact 2,499 save
  confirmation, while excluding only parsed optimizer rows whose update is
  greater than 2,499. The standard audit still requires the last retained
  optimizer row to be 2,499, finite, zero-skipped/zero-NaN, and accompanied by
  the exact successful-save message.
- The corrected materializer then completed and froze the no-copy view. Its
  first audit invocation used `/usr/bin/python3.11` inside the PyTorch uenv;
  that interpreter does not expose the uenv's `torch` package and failed at
  import before producing an audit. The audit is retried with the uenv's
  `python3`; no checkpoint or scientific input changed.
- The uenv audit then correctly rejected the old Phase-2 cache because the new
  executing bundle had no producer-compatibility authority yet. Separately,
  attempting to extend that authority from the full reporting/evaluation
  bundle produced a very large unaudited-path list dominated by presentation
  evidence. Rather than broadening producer trust to irrelevant files, Track B
  now uses a minimal bundle cloned from the already accepted v103 training
  bundle and overlays only the LR-policy adapter, intermediate-branch tools,
  and their direct validators. This keeps the compatibility delta narrow and
  reviewable.
- The first compatibility-backed audit selected the post-processing recovery
  preflight, which binds the later v103 recovery-cache receipt, while the
  source checkpoint's original allocated preflight binds the v89r1 Phase-2
  cache used during training. The audit rejected that mixed pair. Retry uses
  the original `8b_p2_2261_3218_3102006.json` together with its exact v89r1
  cache receipt; neither artifact is modified.
- The next audit reached `common.pt` and exposed a missing runtime import path:
  deserializing the scheduler state requires the pinned Megatron package on
  `PYTHONPATH`. The PyTorch uenv alone is insufficient. Retry adds the already
  receipt-bound Megatron root; this is environment completion, not a model or
  checkpoint change.

## Deviations

None yet.

## Failures and reusable findings

- The historical `deploy_targeted_bundle.sh` could not build from this modern
  experiment worktree because it requires a training patch that is no longer
  present in the repository revision. No remote bundle had been created when
  it refused. Workaround: copy the already proven evaluator bundle server-side,
  overlay only this committed experiment subproject, re-run shell/Python parse
  gates, freeze a new complete receipt, and verify the resulting tree. This is
  an experiment-local deployment workaround; canonical follow-up will be
  checked against the existing issue tracker before handoff.
- The first public-panel freeze invocation omitted the required
  `H2G_CODE_ROOT`/`H2G_CODE_RECEIPT` environment. It failed before writing the
  immutable output; retry with the verified bundle environment succeeded.
- The broad historical test module currently has five unrelated failures
  (one missing optional `tokenizers` dependency and four stale assertions in
  older mix-builder/profile/uenv tests). The focused evaluator/adapter gate is
  `29 passed`; no unrelated production code was changed to force the broad
  legacy suite green.
- Direct `salloc --no-shell` adoption adds an outer `srun` control step. The
  existing four-node scorer was written as a top-level batch payload and its
  inner shard step did not declare overlap. The scientific wrapper now adds
  `--overlap` only when `SLURM_STEP_ID` proves it is nested; top-level batch
  behavior is unchanged. This held-allocation compatibility should be checked
  against the canonical issue tracker before handoff.
- The first held-allocation retry exposed a second incompatibility in that
  compatibility change: it supplied both `--overlap` and the pre-existing
  `--exclusive`, which Slurm rejects as mutually exclusive. No evaluator rank
  started and no score was produced. The failed immutable output was retained
  under its `_failed_*` path. The correction selects exactly one step mode:
  `--overlap` when nested, otherwise `--exclusive`.
- Both SSH processes that originally owned `salloc --no-shell` disconnected,
  but Slurm correctly retained allocations `3151839` and `3152052`. Work was
  reattached with audited `srun --jobid ... --overlap` steps; neither allocation
  nor any completed score was lost. This is a useful distinction for the
  canonical runner: controller transport loss is not allocation loss.
- The first source view of update 2,499 was retained under
  `_failed_source_async_log_prefix_20260822T202500Z`; the corrected no-copy
  branch passed the standard checkpoint audit and permit as
  `checkpoint_audit_v2.json`, `checkpoint_permit_v2.json`, and
  `checkpoint_reference_v2.json`.
- Producer compatibility initially rejected the full reporting bundle because
  it included many unrelated presentation/evaluation paths. The accepted
  training bundle is instead a minimal overlay of the already accepted v103
  trainer. This narrowed rather than widened the trusted producer surface.
- The current training-run permit had to be rebuilt after that bundle changed;
  the stable-peak authorization gate correctly rejected the stale permit and
  passed only after binding `training_run_permit_8b_stable_peak_v1.json`.
- Static B2 preflight passed exactly for 8B, Phase 2, update 2,499 to 3,218,
  DP32 on 16 nodes, and Phase-2 cache tree
  `486b7d724d6a9d7ebc163293d971726b5e5a79a2f9919abc434308a3a7048068`.
- Freezing the latest canonical runner with `/usr/bin/python3.11` as a vendor
  installer failed because that interpreter has no `pip`. Reusing the previous
  proven dependency directory produced a bundle that imports under the uenv's
  Python 3.12 but not under host Python 3.11: its `rpds` extension is
  `cpython-312` only. The training controller will therefore execute in the
  uenv; this packaging gap will be filed against the efficiency repository.
- The first campaign status invocation correctly rejected a source campaign
  JSON where a compiled manifest was required. After copying the exact prior
  runtime/evaluation source contracts, the latest compiler exposed an older
  campaign-format gap: `/usr/bin/env`, the PYTHONPATH value, and mutable
  directory arguments were not covered by the new argv-closure gate. System
  executables were file-bound. The four directory arguments are being moved to
  declared environment inputs in the existing adapter, preserving their exact
  values while avoiding false file bindings for directories. No GPU training
  was started under the incomplete contract.
- Adapter regression coverage after moving the four directory arguments to
  declared environment inputs is `54 passed` (`test_canonical_train_adapter`
  plus `test_r2_orchestration`). The flags remain accepted for every historical
  caller; only the stable branch uses the environment form.
- Scientific bundle v9 was materialized as a one-file sparse overlay on v8,
  then fully re-inventoried and verified. Producer compatibility v3, checkpoint
  audit/permit/reference v3, training-run permit v2, branch gate v2, DCP runtime
  compatibility v2, and static preflight v2 all passed under that exact bundle.
- The first v9 branch-gate invocation rejected the v8 checkpoint permit. This
  is the intended code-binding behavior; the checkpoint was re-audited under
  v9 plus the narrow compatibility authority, rather than copying or editing
  the old permit.
- The first current-runner compile found that the portable training-data
  manifest had been copied without its relative prepared-dataset receipts.
  The original frozen v108 sibling receipts were copied byte-for-byte into the
  campaign-contract directory; no dataset payload or training order changed.
- The next compile rejected evaluation milestones outside the 2,499--3,218
  branch. The branch evaluation contract now contains only paired checkpoints
  2,618, 2,856, 3,094 and 3,218 (with the native suite at 2,618 and 3,218), as
  specified by the plan.
- Reusing the original run's readiness approval correctly failed because it
  authorized a different arm and horizon. A new intake-v2 bound to the user's
  2026-08-22 plan and exact branch horizon was frozen; its generated readiness
  plan reports `ready`, zero pending tasks, and production authorization. The
  candidate campaign then compiled successfully with contract digest
  `fcc708ca56792ef3a972944070e26a88a8efb160c5f3e4cf6a5754fbd4c47f74`.
- At the final proven-runtime rebind, new Clariden SSH connections began
  rejecting a freshly valid CSCS certificate. The certificate was refreshed,
  its UTC validity and key fingerprint were checked, and the current key plus
  certificate were reloaded into the agent. Existing Slurm work continued;
  this is an access-path interruption, not a training/evaluation failure.

## Receipt index

- Scientific bundle v1:
  `/iopsstor/scratch/cscs/fffoivos/orchestration/targeted-8b-cpt/20260822T192900Z-hard-h2g-full-public-stablelr-825e60be-v1`
  with tree SHA-256
  `8364c16f3711f25fd2450fdbd3361537d91acb4663754c585a5d2eb8ab3cc619`.
- Full-public examples:
  `/capstor/scratch/cscs/fffoivos/cpt_runs/hard_h2g_matched/20260814T201715Z-r2-v14/evaluation/greekmmlu_full_public/20260822T193500Z-v1/public_examples.json`.
- Scientific bundle v4 (active scorer):
  `/iopsstor/scratch/cscs/fffoivos/orchestration/targeted-8b-cpt/20260822T200100Z-hard-h2g-full-public-stablelr-b1951246-v4`
  with tree SHA-256
  `9940b5e7b314a8cc10e21996557e85b9bec3576640b38ed9d4a778f83e138ecb`.
- Stable-peak source branch:
  `/capstor/scratch/cscs/fffoivos/cpt_runs/hard_h2g_matched/20260814T201715Z-r2-v14/stable_peak_branch/20260822T202000Z-v1`.
- Minimal stable-peak scientific bundle v8:
  `/iopsstor/scratch/cscs/fffoivos/orchestration/targeted-8b-cpt/20260822T211000Z-hard-h2g-stablelr-minimal-59231dc7-v8`, tree SHA-256
  `12ed7a6d6de52dc5263ae463f69e129c1ea224a7b4b4c135d6136ad977038a21`.
- Latest canonical runner bundle v1:
  `/iopsstor/scratch/cscs/fffoivos/orchestration/apertus-cscs-efficiency/20260822T212000Z-9006ae31-stable-peak-v1`, tree SHA-256
  `54e39f4b59155bd88233e6056185648cb466665b06f7b58180845a27930472ae`.
- Minimal stable-peak scientific bundle v9:
  `/iopsstor/scratch/cscs/fffoivos/orchestration/targeted-8b-cpt/20260822T213000Z-hard-h2g-stablelr-minimal-6e945d38-v9`, tree SHA-256
  `fb575d1f70634c317b2251c2df109b796e327bcaed5038b029141a34ccae227b`.
- Current branch campaign contracts and readiness evidence:
  `/capstor/scratch/cscs/fffoivos/cpt_runs/hard_h2g_matched/20260814T201715Z-r2-v14/stable_peak_branch/20260822T202000Z-v1/campaign_v1`.
