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
- A forced fresh device authorization then timed out at CSCS's token endpoint;
  the following retry could not fetch the OIDC discovery document. No code or
  cluster state was changed by those failed authentication requests. The
  older full-panel controller connection remains live and continues to report
  completed checkpoints.
- The newly issued certificate is structurally valid for principal `fffoivos`,
  has the expected key fingerprint, and covers the current wall clock. A
  verbose SSH probe shows the gateway recognizes the certificate/public-key
  identity but authentication still does not complete; at the same time direct
  HTTPS requests to `auth.cscs.ch` time out. This narrows the interruption to
  the CSCS authentication path rather than a local key, certificate-expiry, or
  Clariden job-state problem. A bounded Mac-side reconnect monitor now retries
  without touching the live Slurm allocations.
- The full-public 8B trajectory has completed through update 2,142 while new
  SSH sessions are unavailable. Each completed scorer receipt passes the
  immutable scientific-bundle check. The repeated Slurm warning that the
  nested step asks for more CPUs per task than the outer control step has not
  prevented any scorer result, but it is retained as operational evidence for
  the post-run runner review.
- Reusable follow-up is recorded in efficiency issue
  [#137](https://github.com/fffoivos/apertus-cscs-efficiency/issues/137)
  (document/scaffold directory-valued scientific inputs as `required_env`).
  Live nested-step evidence, including the `--overlap`/`--exclusive` mutual
  exclusion, was added to existing issue
  [#61](https://github.com/fffoivos/apertus-cscs-efficiency/issues/61#issuecomment-5381931332).

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

## 2026-08-22 21:18--21:35 UTC update

- Track A1 completed all 17 pre-existing 8B checkpoints on the public
  16,632-question panel. The immutable aggregate receipts run from update 0
  through update 3,694 under
  `evaluation/greekmmlu_full_public/20260822T193500Z-v1/trajectory`; every
  scorer-side bundle verification reported tree SHA-256
  `9940b5e7b314a8cc10e21996557e85b9bec3576640b38ed9d4a778f83e138ecb`.
- Stable-branch normal allocation `3152592` was granted immediately with 16
  nodes / 64 GH200 and a three-hour limit. It remains held; failed controller
  attempts did not release or replace the allocation and produced zero model
  updates.
- The first current-runner attempt rejected a stale entrypoint byte/hash
  binding before training. Campaign v2 corrected only that file binding to
  the already frozen v9 entrypoint (`25132` bytes, SHA-256
  `fb0ffc8584fb272e7f790b03c357fa404768fb26fed9e328873efd1d5f335fe5`).
- Direct `run-in-allocation` initially lacked the two canonical environment
  bindings `APERTUS_CAMPAIGN_MANIFEST` and `APERTUS_CAMPAIGN_CODE_ROOT`.
  Supplying the exact manifest and runner roots moved execution to the next
  fail-closed check; no scientific input changed.
- A one-node controller step exposed `SLURM_NNODES=1` to the scientific child,
  so the 16-node training-run permit correctly rejected it as profile drift.
  The permit never bound the allocation time limit; the earlier diagnosis of
  a three-hour-versus-twelve-hour permit mismatch was therefore withdrawn.
- Retrying inside the same allocation exposed a retry-key bug: DCP compatibility
  output is keyed only by Slurm job ID. The earlier valid 791-byte receipt was
  retained under `receipts/train_preflight/quarantine_retry_3152592_20260822T222900Z`
  before retry; no evidence was overwritten or deleted.
- A one-task adopted-allocation controller cannot launch the historical
  training payload unchanged because the frozen payload's nested `srun`
  inherits one task. The corrected operational wrapper starts one lightweight
  controller task per allocated node, executes the canonical controller only
  on rank 0, and holds the peers until rank 0 completes. The nested trainer
  then observes exactly 16 nodes / 16 launcher tasks and starts 64 model ranks.
- The active run is
  `stable_peak_b2_v6_fullgeometry_wrapper`. Before the first optimizer update,
  its log verified: Phase 2; resume update 2,499; TP2/DP32; microbatch 2;
  global batch 1,024 sequences / 4,194,304 tokens; RoPE base 500,000 with
  factor 8; vocabulary divisor 256; the frozen Phase-2 blend; and the
  constant-LR guard at `5.5e-5` on every launcher rank.
- Reusable runner follow-up from this sequence: adopted controllers must
  propagate scheduler-complete geometry to legacy scientific payloads; direct
  execution should inject its canonical manifest/code-root environment; and
  retry-scoped receipts must include an attempt nonce rather than only the
  Slurm job ID.
- Those three live failure modes are now filed together as efficiency issue
  [#146](https://github.com/fffoivos/apertus-cscs-efficiency/issues/146), with
  the exact 16-node reproduction and acceptance tests.
- The first real optimizer row was update 2,500 at 21:35:57 UTC. It loaded the
  exact update-2,499 checkpoint, reported Phase-2 local cursor 243,712 on all
  64 ranks, used LR `5.500000E-05`, and had zero skipped and zero non-finite
  updates. The append-only launch timeline now records this as
  `production_started` with event ID
  `798a5b88790fef37350f6e495c65e0e7b75414129242786dde9a20a3e2d09d5c`.

### Track A numerical results

- A1 full-public FP32 peak: update 2,618, `9,676 / 16,632 = 58.1770%`.
  The curve then declines to `9,495 / 16,632 = 57.0887%` at update 3,218
  and `9,478 / 16,632 = 56.9865%` at update 3,694. The corresponding choice
  NLL values are 1.05855, 1.08282 and 1.07896.
- A2 was run from the exact historical evaluator files at git revision
  `cfdd0e7b`, SHA-256
  `b9f75809b6e617cfd419dc5420e480dee72bb3f1df7fa8f82e04793b4dfd19c4`
  for the scorer and
  `fcf732c142efdd204fe8a64ac4fb1159f47e7b2bac0e947207c2a971329bf508`
  for its registry. Frozen settings were BF16, max input 3,072, candidate
  batch 16, example batch 16, and all 16,632 questions.
- The historical venv named `native_greek_eval_py312_cpt20260610` exists but
  lacks `datasets` under the current uenv. The failed output was retained.
  A2 therefore reused the already qualified full-panel runtime v2 while
  keeping the pinned evaluator code and all scoring parameters unchanged.
- The first parallel A2 retry shared one Hugging Face datasets cache across
  ranks. Concurrent materialization produced a missing Arrow-file failure;
  all partial outputs were retained under `_failed_shared_dataset_cache`.
  Isolating the cache per rank resolved the race without changing examples.
- A2 legacy-BF16 results: update 2,618 = `9,630 / 16,632 = 57.9004%`;
  update 3,218 = `9,447 / 16,632 = 56.8001%`; update 3,694 =
  `9,441 / 16,632 = 56.7641%`.
- Pre-registered decision: **replication miss**. The best legacy score is
  2.0623 percentage points below the selected historical beta2=0.999 target
  of 59.9627%, outside the ratified plus/minus 1.0-point band. The band was
  not revised after observing the result.
- The A2 output is frozen in
  `evaluation/greekmmlu_legacy_bf16/20260822T224500Z-v1/matrix_receipt.json`;
  the receipt hashes every result file and records the failed replication
  decision. Parallel cache materialization follow-up is efficiency issue
  [#147](https://github.com/fffoivos/apertus-cscs-efficiency/issues/147).

### First paired stable-LR checkpoint

- Update 2,618 was saved successfully after 33.1 seconds of checkpoint time;
  training continued to update 2,619 without a restart. The saved row has LR
  `5.5e-5`, zero skipped updates and zero non-finite updates.
- Its HF export completed with the same exact-weight and probability-space
  parity pipeline used by Track A. A one-node controller retry could export
  but could not launch the four-node nested scorer (`Only allocated 1 nodes
  asked for 4`). The failed score attempts remained in `progress.tsv`.
- Reattaching the scorer with a four-node lightweight outer step and a single
  rank-0 controller resolved the topology without changing its evaluator or
  model export. This is the evaluation-side instance of issue #146.
- Stable-LR update 2,618 full-public score: `9,686 / 16,632 = 58.2371%`,
  choice NLL `1.053410`, correct-answer BPB `0.164597`.
- Paired decayed-arm update 2,618: `9,676 / 16,632 = 58.1770%`, choice NLL
  `1.058548`, BPB `0.165548`. The initial accuracy difference is only
  `+0.0601` percentage points for stable LR, far below the approximately
  0.4-point checkpoint noise scale; no trajectory conclusion is drawn from
  this single pair.

## 2026-08-22 22:13 UTC monitoring update

- Stable-LR training reached update 2,716 / 3,218 on allocation `3152592`.
  The latest rows continue to show LR `5.500000E-05`, zero skipped updates,
  zero non-finite updates, and approximately 8.8--9.1 seconds per update.
- Allocation `3152592` had 2:04 remaining at this observation, while its
  model-reported training ETA was approximately 1:15. Allocation `3151839`
  remained live with 1:08 remaining for the update-2,856 full-panel score.
- The reusable full-geometry rank-zero controller plus peer-holder wrapper
  was committed and pushed to `apertus-cscs-efficiency` as commit `1a0d672`.
  Pull request [#148](https://github.com/fffoivos/apertus-cscs-efficiency/pull/148)
  connects the tested implementation to issue #146.
- The optimizer-row ETA excludes the nine-panel validation pass every 25
  updates. Measured end-to-end cadence was approximately 4.3 minutes per
  25-update block (about 10.3 seconds/update amortized), rather than the
  approximately 8.9-second optimizer-only row. The allocation plan was
  recomputed from this measured cadence and still fits, with less reserve.
  Reusable ETA handling is tracked in efficiency issue
  [#149](https://github.com/fffoivos/apertus-cscs-efficiency/issues/149).
- At update 2,800, stable LR had slightly lower OpenArchives validation loss
  than the decayed arm (`1.476476` versus `1.478760`) but higher loss on HPLT
  (`2.028437` versus `2.017839`) and each retention panel. This is an interim
  loss observation, not the pre-registered GreekMMLU branch decision, and the
  plan already warns against treating the stable-versus-decayed level gap as
  a pure curriculum result.

## Stable-LR update 2,856 result

- Update 2,856 saved successfully and training continued at update 2,857.
  The saved row used LR `5.5e-5` and retained zero skipped and zero non-finite
  updates.
- The exact-weight mapping passed for its HF export. The probability-space
  parity diagnostic remained a warning, as it did for the prior trajectory
  exports, so this result remains inside the explicitly documented
  trajectory-comparison scope.
- Full-public FP32 result: `9,612 / 16,632 = 57.7922%`, choice NLL
  `1.061940`, correct-answer BPB `0.167040`.
- Paired decayed result: `9,560 / 16,632 = 57.4796%`, choice NLL `1.069602`,
  correct-answer BPB `0.170266`. Stable LR is ahead by 0.3126 percentage
  points at this checkpoint, but its own accuracy fell 0.4449 points from
  stable update 2,618. The no-decay arm is therefore not rising over the first
  interval; the pre-registered decision remains open until 3,094 and 3,218.
- After the immutable aggregate receipt was checked, scoring allocation
  `3151839` was cancelled with 31:21 unused rather than being left idle.

## Endpoint allocation packing

- The initial post-training handoff would have scored updates 3,094 and 3,218
  serially on four nodes while the 16-node training allocation remained held.
  Before it triggered, that login-side waiter was terminated; no Slurm step or
  evaluation had begun.
- The replacement waits for the successful training-controller completion
  marker, final checkpoint metadata and tracker, then launches the two frozen
  four-node evaluators concurrently. Their checkpoint exports, result roots,
  TSV manifests and controller-done files are disjoint; scorer code, panel and
  model inputs are unchanged.
- This is operational bin-packing after training, not concurrent training or a
  scientific change. The exact manifests and launcher are committed under
  `operational_workarounds/`; reusable support is requested in efficiency
  issue [#150](https://github.com/fffoivos/apertus-cscs-efficiency/issues/150).

## Stable-branch post-save finalization repair

- Training itself completed all optimizer updates through 3,218, with zero
  skipped/non-finite updates, and durably saved `iter_0003218`. The frozen
  post-save launcher then failed because it invoked `/usr/bin/python3.11`
  outside the PyTorch uenv; checkpoint audit imports consequently raised
  `ModuleNotFoundError: No module named 'torch'`. This did not alter model,
  optimizer, scheduler, RNG or data-cursor state.
- Three subsequent audit attempts exposed required runtime bindings one at a
  time: the scientific bundle environment, producer-bundle compatibility and
  the frozen Megatron path. The retained fourth audit ran under
  `pytorch/v2.9.1:v2`, with all three bindings, and passed all checkpoint-state
  checks over 128 storage files.
- The canonical frozen permit builder then issued a passing update-3,218
  checkpoint permit using
  `producer_bundle_compatibility_stable_peak_v3.json`. A small, committed
  repair helper reproduced the reference schema from the canonical segment
  launcher without rerunning training. The campaign completion adapter
  accepted that reference and wrote a `status: completed` receipt for update
  3,218 at `2026-08-22T21:52:16Z`.
- The two independent frozen update-3,094 and update-3,218 full-public scorers
  were then launched concurrently as four-node steps inside the still-held
  16-node allocation `3152592`. This preserves the evaluator and scientific
  inputs while avoiding serial idle-node time.
- The implicit post-save runtime contract and repairability gap are tracked as
  efficiency issue
  [#151](https://github.com/fffoivos/apertus-cscs-efficiency/issues/151).

### Endpoint conversion concurrency correction

- The first concurrent endpoint attempt showed that independent result roots
  are not sufficient to make checkpoint conversion concurrent. Both converters
  initialize shared Megatron fused-kernel/NCCL resources; their first attempts
  failed with `ncclInvalidUsage` / `Failed to initialize any NET plugin` while
  entering the common build path. No aggregate score was produced or accepted.
- Conversion was therefore serialized while preserving the same frozen
  checkpoint, conversion overlay, tokenizer, examples and scorer. Scoring can
  still be packed concurrently after exports exist. Partial export directories
  and logs were retained under timestamped `failed-a*` roots.
- With less than one endpoint-runtime envelope left in allocation `3152592`, a
  four-node, one-hour `salloc` successor (`3153569`) was queued while the
  current allocation remained active. It is a bounded recovery request and is
  to be cancelled if both immutable aggregate receipts complete on the held
  allocation.
- The first successor request specified nodes but not the scorer's CPU/GPU
  geometry. It was relinquished immediately when granted because the scorer
  requires 16 tasks, four per node, 54 CPUs and one GPU per task; the held
  training allocation likewise could export on one node but could not create
  that 864-CPU scoring step. Correctly shaped successor `3153706` was then
  queued. This is an allocation-planning error and the unused first allocation
  is included in the audit trail rather than hidden.
- Full consumer-geometry compilation is tracked as efficiency issue
  [#152](https://github.com/fffoivos/apertus-cscs-efficiency/issues/152).

### Stable-LR update 3,094 result

- Both update-3,094 and update-3,218 checkpoint exports completed with exact
  weight mapping and the same trajectory-scoped probability-parity policy as
  the earlier branch points. The old 16-node training allocation was released
  immediately after the second immutable export receipt passed.
- Stable-LR update 3,094 full-public score: `9,432 / 16,632 = 56.7100%`,
  choice NLL `1.091757`, correct-answer BPB `0.175727`.
- Paired decayed update 3,094: `9,501 / 16,632 = 57.1248%`. Stable LR is now
  0.4148 percentage points behind the decayed arm and has fallen 1.5272 points
  from its own update-2,618 value. This is already inconsistent with a rising
  stable-LR trajectory; the pre-registered B3 decision is finalized only after
  the update-3,218 aggregate receipt.

### Stable-LR update 3,218 result and B3 decision

- Stable-LR update 3,218 full-public score: `9,270 / 16,632 = 55.7359%`,
  choice NLL `1.111750`, correct-answer BPB `0.183188`.
- Paired decayed update 3,218: `9,495 / 16,632 = 57.0887%`. Stable LR is
  1.3528 percentage points behind the decayed arm at the endpoint and has
  fallen 2.5012 points from its own update-2,618 score.
- **Pre-registered B3 outcome: stop.** The constant-LR trajectory is falling,
  not rising, across every observed interval after 2,618. The unauthorized
  3,219→3,694 extension was not submitted. This answers the exploratory
  question at the registered gate; it does not establish that no alternative
  cooldown could improve the endpoint.
- Both endpoint aggregate receipts report `status: completed`, FP32 scoring,
  all 16,632 frozen public examples and scorer tree
  `9940b5e7b314a8cc10e21996557e85b9bec3576640b38ed9d4a778f83e138ecb`.
  The correctly shaped scoring allocation was released immediately after the
  second receipt passed.

### Allocation accounting

- `3151839`: 4 nodes / 16 GPUs for 3:28:39 = **55.64 allocated GPU-h**.
  This exceeded the plan's approximately 35 GPU-h Track-A/branch-scoring
  envelope because the allocation remained held across scoring waits and
  operational retries; 31:21 was released at the end.
- `3152592`: 16 nodes / 64 GPUs for 2:52:29 = **183.98 allocated GPU-h**.
  The B2 estimate was approximately 112 GPU-h. Training itself completed in
  about 2:06, while recurrent validation, checkpointing, post-save repair and
  serialized endpoint export consumed the remainder. The last 7:31 was
  released once both export receipts passed.
- `3153569`: mis-shaped recovery allocation held for 0:52 = **0.23 allocated
  GPU-h**, then relinquished.
- `3153706`: correctly shaped 4-node scorer allocation held for 0:14:54 =
  **3.97 allocated GPU-h**, then relinquished after both endpoint receipts.
  Interactive `salloc` relinquishment records these holder jobs as `FAILED` in
  Slurm even though their scientific child steps and immutable result receipts
  completed; this distinction is preserved here.

## Final report and QA

- Final single-page report:
  `presentations/hard_h2g_full_panel_stable_lr_20260822/HARD_H2G_FULL_PANEL_AND_STABLE_LR_20260822.html`
  (`135,361` bytes; SHA-256
  `6e4ae4a66ac406e68c1077e385e0cc269c051431aa57d54a51845af56c18dedf`).
- Evidence analysis: `evidence/analysis.json` (`661,953` bytes; SHA-256
  `67a2989d0aeac7dc41ad212cdc50471f74ac93b8da723ff9e2ecd1088f37c293`).
  Its builder rejects missing aggregate receipts, scorer-tree drift, example
  identity drift, non-FP32 results, incomplete checkpoint grids, skipped or
  non-finite stable updates and non-finite values anywhere in the output.
- Complete layouts were rendered and visually inspected at 1,440×1,000 and
  430×932 viewports. The QA receipt is `qa/qa_receipt.json`, status `passed`,
  SHA-256
  `99405c20e8af9dd446bbc5ea987049a32bc29105535f29df4eb521d999f9e24a`.
- The report keeps the full 17-checkpoint public-panel curve primary, shows
  choice NLL and correct-answer BPB, retains every validation panel over the
  complete horizon, labels the legacy replication miss, records the B3 stop
  decision, and includes the allocated-compute ledger and parity caveat.
