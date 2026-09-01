# Apertus 8B CPT rerun: allocation-safe launch and operations handoff

Date: 2026-08-08 (Europe/Athens)

Status at original handoff creation: **production training had not started**.
The dated operational addendum below supersedes that historical launch-state
paragraph; the scientific authority sections remain current.

## Operational addendum — 2026-08-09

The rerun is active. Segment 0, job `3034915`, completed updates `0–4,000`
cleanly in `10:02:48`; its checkpoint receipt contains 131 files and
`147,638,448,520` bytes with tree SHA-256 beginning `797e3f`. Its signed
segment-1 permit exists. The submitted segment-1 holder is job `3037145`, a
16-node, 12-hour `normal` request. Scheduler start predictions are not
reservations and must be refreshed before reporting an ETA.

The allocation-overlap policy is now receipt-bound:

- derive each request point from the conservative wall time of the **target**
  segment: `maximum hold = 43,200 - target runtime - 1,200 seconds`;
- derive the delayed `after:` offset from the **source** segment's frozen
  conservative wall-time budget, including startup and final checkpointing:
  `source trigger = conservative source runtime - maximum hold`. Do not
  substitute the raw seconds-per-update product, which can make a holder expire
  before a slow-but-valid source produces its signed permit;
- while final evaluation for segment `k` completes and segment `k+1` is known,
  submit exactly one delayed `normal` holder for segment `k+2` with
  `after:<segment-k+1-job>+<offset-minutes>`;
- do not use a pending debug timer. Live `debug-qos` exposed
  `MaxJobsPU=1` and `MaxSubmitJobsPU=2`, so both submitted slots must remain
  available to the running serial evaluation and its continuation/supervisor;
- the holder must verify both immutable code receipts, its sole manifest row,
  exact update boundaries, remaining wall time, and the signed checkpoint
  permit before executing canonical `train_segment.sbatch`;
- audit the submitted holder before appending its manifest row; on audit
  failure cancel it and leave no adoptable row;
- on a repeated request, adopt only a recorded holder that remains `PENDING`
  or `RUNNING`; terminal or unknown jobs are rejected;
- bounded `sbatch` retries cover transient controller failures only. Persistent
  QOS saturation is handled by reducing or deferring debug submissions;
- immutable training-audit receipts are reproduced to a temporary path and
  compared on supervisor reruns, never overwritten.

The delayed holder can wait if it starts early, but it may train only while its
frozen minimum training time plus the 20-minute reserve remains. If its permit
arrives too late, it exits without training and the verified recovery path
submits a fresh successor. This supersedes the older blanket statement below
that no happy-path successor may be prequeued; at most one audited, delayed,
permit-gated successor is allowed.

The segment-2 holder was installed from tested operational bundle
`20260808T230500Z-prequeue-v20` (tree SHA-256 `555fb8c7…e30d`). The current
successor bundle for later control jobs is
`/iopsstor/scratch/cscs/fffoivos/orchestration/full8b-cpt-ops/20260809T003000Z-prequeue-v29`,
tree SHA-256
`0d2cb13e7c98bd73a6e63e5bcefc532c4ca279fe931aeeade2f6bd1afcee4d36`.
It additionally fails on delay/reserve drift and makes final evaluation and
supervisor submission rerun-safe. It also contains the receipt-producing
retention analyzer. Its 37 focused remote tests pass, and the
exact segment-3 request passed `sbatch --test-only` without changing the
manifest. v27 also charges bundle/manifest setup against the hold timer and
requires `minimum_train_seconds + 1,200` seconds to remain before training;
the 20-minute reserve can no longer be consumed by holder setup or waiting.
v29 contains the receipt-backed pending-supervisor transition helper and its
executable audit-before-cancel test: it accepts only
the old pending v19 supervisor, verifies both frozen bundles and every segment
binding, audits the v29 replacement, then and only then cancels the old job.
The lightweight Mac-side coordinator
`clariden/watch_pending_supervisor_transition_v29.sh` polls only state and
receipts (no GPU, data or compute work) and invokes that v29 helper only when
the legacy supervisor is the sole pending debug job. It has a finite six-hour
default watch horizon and records its sole state change under
`orchestration/supervisor_transitions/` in the live run root.

The historical v19 final-evaluation continuation submitted and resource-audited
supervisor `3037861` but omitted the later
`supervisor_submission_receipts/segment_1.json` schema.  Before the watcher can
transition it, the frozen v30 bridge reconstructs that one missing receipt only
after checking the completed evaluation queue and exact GreekMMLU receipt, the
campaign event, the prequeued segment-1 manifest row, the v19 allocation audit,
and the live pending Slurm job's command, dependency, partition, wall time and
node count.  The bridge has no `sbatch`, `scancel`, or training action.  It
then permits the existing audit-before-cancel transition helper to operate and
records the completed swap as `supervisor_transitions/segment_1_v30.json`.
The bridge was frozen as
`/iopsstor/scratch/cscs/fffoivos/orchestration/full8b-cpt-ops/20260809T030000Z-prequeue-v30-legacy-receipt-bridge`
with tree SHA-256
`aecab9756e8667e37ef8afb4499ecd95cbbbe61e73318e22ac42c564b68c9566`.
Its test-only bridge and replacement checks passed. The completed transition
receipt proves that job `3037873` was resource-audited while pending and only
then replaced cancelled legacy job `3037861`; its dependency remains
`afterany:3037145`. The finite Mac watcher was removed immediately afterward.

The segment-0 evaluation chain is complete. Segment-1's 16-node holder has its
allocation and is running; its v30 replacement supervisor was initially queued
behind that exact holder. On 2026-08-09 it was safely replaced while still
pending by v31 job `3038080`, after an exact Slurm test-only request and a
resource audit. v31's frozen operational bundle is
`/iopsstor/scratch/cscs/fffoivos/orchestration/full8b-cpt-ops/20260809T054740Z-prequeue-v31-conservative-source`
with tree SHA-256
`73d3829df676afd4e04e89ee4e4f350026f060a9ca55bccc75297bfb65971f05`.
The transition receipt is
`orchestration/supervisor_transitions/segment_1_v31.json`; its audit is
`orchestration/allocation_receipts/supervisor_transition_s1_3037873_3038080.json`.
It preserves job `3037145` and its segment-2 holder `3037241`, and ensures
later holder submissions use the corrected conservative-source trigger policy.

The already-queued segment-1 holder `3037145` names the earlier frozen v14
holder wrapper. It is retained deliberately: it verifies its v14 and v45 bundle
receipts, the exact segment-1 manifest row and the signed iteration-4,000
checkpoint permit before it `exec`s the canonical v45
`clariden/train_segment.sbatch`. A live read-only reproduction of those exact
preflight checks passed. Its older wrapper lacks the later explicit
training-plus-reserve comparison, but this holder has a pre-existing permit and
therefore performs no permit wait; it still requires the conservative 37,800
seconds before training. Requeuing a correctly bound scarce 16-node request to
change only that outer wrapper would create queue risk without changing model
execution. Segment 2 and later holders are submitted by the v30 controller and
use its tightened reserve accounting.

While the campaign waits for capacity or runs, the local
`clariden/watch_full8b_campaign.sh` LaunchAgent is a read-only status recorder.
It writes the current full8B Slurm graph, newest training-log marker, latest
loss/health line and last stderr line every two minutes, exits only when the
terminal training receipt appears, and has no `sbatch`, `scancel`, remote
write, GPU, or data-processing action.

### Segment-1 restart evidence

Holder `3037145` received its 16-node normal allocation at 2026-08-09
06:20 (+02) and passed the frozen v45 scientific bundle check, v14 holder
bundle check, recipe validation, DP32 profile validation and every rank/GPU/CPU
affinity assertion before entering training. It is running the exact
`4,000–8,000` update range at global batch 1,024. The first warmup update took
14.91 seconds; the first two warmed updates were 8.705 and 8.630 seconds
(7,529 and 7,594 tokens/s/GPU, respectively). Both reported peak LR
`5.5e-5`, loss scale 1.0, and zero skipped or NaN updates. The startup log
contains only known PyTorch checkpoint-planner and barrier warnings; the Slurm
stderr is empty. Treat the later stabilized-window median—not the one-step
startup estimate—as the segment ETA input.

The stabilized interval 4,002–4,034 contains 33 loss-active updates with
median step time 8.614 seconds (p90 8.624 seconds) and median 7,608
tokens/s/GPU. LR remained exactly `5.5e-5`; loss ranged 1.732–1.776, and all
33 updates reported zero skipped and zero NaN iterations. This clears the
segment-1 start/resume/geometry/throughput gate without changing the approved
scientific recipe.

The real audited segment-2 holder is job `3037241`, dependency
`after:3037145+540`, updates `8,000–12,000`, minimum train time 37,800 seconds,
maximum hold 4,200 seconds. Its allocation-routing receipt is
`orchestration/allocation_receipts/prequeued_s2_3037241.json` under the run
root. The manifest has exactly segment rows 1 and 2; no later holder is
submitted yet.

The corrected request schedule for future holder submissions implements the
unused-allocation rule as follows. `Minimum train` includes conservative
startup/checkpoint time; the additional 20 minutes is never spendable. The
source trigger is derived from the preceding segment's same conservative
budget, rather than a raw 9.0-second/update product. `Maximum early idle` is
therefore the holder's entire permitted hold window, and the live holder still
checks Slurm time remaining before training.

The already-submitted segment-2 holder `3037241` is deliberately not replaced:
it carries the earlier 540-minute trigger and retains its immutable receipt.
Replacing that scarce 16-node request would add queue risk without changing its
training payload. Its holder remains fail-closed if the permit arrives after
its existing hold budget, in which case the supervisor recovery path submits a
fresh audited leaf. Freeze the corrected operational bundle and replace only
the still-pending debug supervisor before it submits segment 3; do not cancel
the segment-2 holder to retrofit this bookkeeping correction.

| Target segment | Eligible after source starts | Minimum train | Maximum hold | Nominal early idle |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 9 h 20 m | 10 h 30 m | 1 h 10 m | 1 h 10 m |
| 2 | 9 h 20 m | 10 h 30 m | 1 h 10 m | 1 h 10 m |
| 3 | 6 h 00 m | 7 h 10 m | 4 h 30 m | 4 h 30 m |
| 4 | 5 h 10 m | 9 h 40 m | 2 h 00 m | 2 h 00 m |

This is the general rule: if a target needs 9 hours of a 12-hour allocation,
expose the request roughly 3 hours before the source's *conservative* forecast
to stop, minus the frozen safety reserve. The trigger is constrained to consume
exactly that allowance, so an in-budget source cannot make its holder expire
before the signed permit. Never request a full suffix of future allocations;
keep at most one audited successor beyond the next gated segment.

The update-400 HF export emits a Transformers warning because the frozen
Llama-3 scaling metadata has `original_max_position_embeddings=8192` while the
training/evaluation window is 4,096. This is not an evaluator-geometry drift:
the initial and update-400 HF configurations have identical RoPE fields, and
an exact runtime comparison found all 64 inverse-frequency elements bit-equal
between the pinned Megatron implementation and Transformers (`max_abs=0.0`,
attention factor `1.0`). The passing receipt is
`orchestration/geometry_evidence/evaluator_rope_parity_v1.json` under the live
run root. Preserve the warning as evidence; do not alter the model config
during the campaign.

Segment 0 has 273 finite source-conditioned measurements: 21 complete points
times all 13 frozen panels. Every Greek learning panel ends at its running-best
loss. The retention macro ends `+0.0194338` nats above its post-warmup running
minimum, below the predeclared `+0.05` critical threshold; no individual panel
has two consecutive `+0.05` warning observations. The resulting status is
`no_alert`, recorded at
`orchestration/retention_snapshots/segment_0_v26.json`. That receipt exactly
reproduces v25 after excluding the creation timestamp: 273 rows, 21 complete
points, zero ignored duplicates and identical metrics. Continue to
report the small upward foreign/code/math drift rather than calling it zero
forgetting.

The first post-training native GreekMMLU checkpoint, update 400, completed on
the same frozen full-test evaluator contract as the corrected initialization
anchor. Its authoritative receipt is
`checkpoint_evaluations/iter_0000400/attempt_0/exact_checkpoint_native_greekmmlu_receipt.json`
under the live run root. Full-test results are `0.53409` accuracy,
`1.19425` choice NLL and `0.20976` correct-answer BPB; the decontaminated
16,159-example subset is `0.53568`, `1.19641` and `0.21018`, respectively.
The compatible initialization values were `0.35901`, `1.45343` and `0.66254`.
This is an early trajectory measurement, not a model-selection result. The
serial chain accepted the receipt and submitted update-1,192 evaluation job
`3037285` with exactly one dependent continuation, `3037286`.

Update 1,192 also completed on that same contract and has an authoritative
receipt at `checkpoint_evaluations/iter_0001192/attempt_0/` under the live run
root. Full-test GreekMMLU is `0.53463` accuracy, `1.16920` choice NLL and
`0.19814` correct-answer BPB; the decontaminated subset is `0.53568`,
`1.17122` and `0.19830`. The continuous metrics improved from update 400
while accuracy is effectively unchanged at this early interval. The serial
chain accepted it and submitted update-2,384 job `3037372` with continuation
`3037374`.

## Objective

Restart the Apertus 8B CPT run from the verified anonymized and decontaminated
full dataset, preserve the approved scientific recipe, use only parity-gated
operational speedups, and collect source-conditioned loss, per-document
validation, and native GreekMMLU evidence through completion.

The resource rule is strict:

- `debug`: all short one-node metadata, receipt, launch-gate, control,
  conversion, GreekMMLU, and per-document jobs that fit the 90-minute limit;
- `normal`: only the five 16-node CPT training segments, or a workload that is
  proven not to fit `debug`;
- no 16-node request is submitted until every allocation-free and debug gate is
  green;
- submit only segment 0 plus its dependent debug supervisor initially; each
  verified segment supervisor submits exactly one next training leaf before
  starting the serial debug evaluation chain;
- `sbatch --test-only` is a prediction, never a reservation or priority
  guarantee.

The root operator rule is also recorded in
`/Users/foivoskarounos-zamparloukos/AGENTS.md` under “Clariden
Scarce-Allocation Handoff Rule.”

### Pending-supervisor replacement safeguard

The existing segment-1 holder remains preserved. Once the final segment-0
evaluation continuation has written the legacy v19 segment-1 supervisor
receipt, the receipt-bound v29 transition helper may replace that *pending*
supervisor only after it has audited the replacement job's exact dependency,
resources and bundle receipt. It cancels the legacy job only after that audit.
`clariden/watch_pending_supervisor_transition_v29.sh` is a finite Mac-side
coordinator for this one transition; it performs no GPU or data work, requires
the legacy supervisor to be the sole pending debug job, and is explicitly
compatible with macOS `/bin/bash` 3.2. It is not a training allocation and
must be removed after `DONE` or `TRANSITIONED`.

## Frozen scientific authority

These inputs do not change during the allocation repair:

| Item | Authority |
| --- | --- |
| Scientific code bundle | `/iopsstor/scratch/cscs/fffoivos/orchestration/full8b-cpt/20260808T023300Z-sanitized-v45` |
| Bundle receipt | `/iopsstor/scratch/cscs/fffoivos/orchestration/full8b-cpt/20260808T023300Z-sanitized-v45.receipt.json` |
| Bundle tree SHA-256 | `fe6993bc8776096bf4cf11ae4a081a9b8d9a1ec474ef1d4d9ec510aed93d2316` |
| Scientific digest | `41998a042d1c9d7ee88700b8692b488b2b6b1f936512a9f7bd07aff79542b666` |
| Sanitized source | `/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/full8_anonymized/20260807T035000Z-v8` |
| Successor evidence stage | `/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/full8_mixed_sanitized/20260808T064500Z-d0-v4-v45bridge` |
| Parent packed-data stage | `/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/full8_mixed_sanitized/20260807T063000Z-d0-v3` |
| Benchmark root | `/iopsstor/scratch/cscs/fffoivos/runs/07_full_8b_cpt/_benchmark/20260808T023300Z-dp32-fallback-v45` |
| Reusable preflight root | `/iopsstor/scratch/cscs/fffoivos/runs/07_full_8b_cpt/_preflight/20260808T023300Z-sanitized-v45` |
| Tokenizer | `/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_polytonic_148992` |
| Tokenizer SHA-256 | `bbb08e71929b519c5c2362338b0fc6a0e99955cb8fdbf0729ae1311117e6561b` |

The selected execution profile is `dp32_16node`: 16 nodes, 64 GPUs, TP=2,
PP=1, CP=1, DP=32, micro-batch 2, gradient accumulation 16, global batch
1,024 sequences, and sequence length 4,096. DP64 is rejected and must not be
used: it failed the frozen trajectory-drift thresholds despite being faster.

The sanitized active-token total is `76,685,490,476`; this produces 18,284
optimizer updates. The earlier 19,248-update/80.7B figure belongs to a
superseded corpus geometry and must not be used for this rerun.

## What has already passed and must be reused

Do not rerun these merely to regenerate job IDs:

| Evidence | Result |
| --- | --- |
| DP32 control and restart parity | Passed; selected compute estimate 44.1315 h, median 8.6892 s/update, p90 8.6980 s/update |
| DP64 promotion | Rejected on trajectory drift |
| Code-bundle verification | Passed for v45 |
| Initial checkpoint freeze and corrected RoPE geometry | Passed |
| Initial 13-panel source validation | Passed |
| Initial per-document validation | 13 completed, training-disjoint panels |
| Initial native GreekMMLU | Passed |
| Checkpoint-to-HF conversion smoke | Passed |
| Nested `sbatch` under uenv | Passed |
| Graceful stop, synchronous save, and exact resume | Passed; stop at update 11, resume to 12 |
| Packed-payload hashes | Passed for the parent payload |

Important reusable job evidence includes DP32 control `3030704`, initial
source validation `3033656`, graceful-stop run `3033664`, restart run
`3034250`, and final graceful-stop receipt job `3034508`.

## Why the successor stage exists

The data bytes, document identities, packed payload, tokenizer, and training
schedule did not change. The anonymization bridge was re-finalized by v45, so
the downstream inventory and recipe must bind the new receipt hash. Job
`3034541` regenerated only deterministic catalogs and the packing plan. It
completed with:

- active tokens: `76,685,490,476`;
- source-selection tasks: `845`;
- packing tasks: `512`.

The 330 GB packed payload is not rebuilt. It is reused only after the successor
plan proves exact task/document identity with the parent stage.

## Allocation policy and current Clariden limits

Live policy observed on 2026-08-08:

- `debug`: maximum 4 nodes, maximum 01:30:00, `debug-qos`;
- `debug-qos`: one running job and at most two submitted jobs per user;
- `normal`: 12-hour maximum for the production profile;
- every Clariden node has four GH200 GPUs, including `debug` nodes.

Consequences:

1. A debug controller must not remain alive while waiting for debug children;
   it would consume the only running slot and deadlock them.
2. Debug work is submitted as a short job that completes, or as a serial chain
   in which each job submits at most one successor before exiting.
3. At most two debug jobs may be present at once. Do not prequeue the complete
   evaluation graph on `debug`.
4. All partition choices are passed explicitly on the `sbatch` command line.
   Script-header defaults are not trusted.

## Required allocation-routing patch before launch

The v45 scientific bundle hardcodes `normal` in many one-node SBATCH headers.
The scientific bundle remains immutable. Add and freeze a small,
receipt-bound **operational allocation layer** which calls the v45 scientific
scripts but makes resource routing explicit. It must contain:

1. `clariden/submit_production_resource_aware.sh`
   - train: `--partition=normal --nodes=16 --time=12:00:00 --switches=1`;
   - supervisor: `--partition=debug --nodes=1 --time=00:20:00`;
   - checkpoint-evaluation chain: `--partition=debug`, at most one running and
     two submitted;
   - evidence finalizer: `--partition=debug --nodes=1 --time=00:10:00`;
   - submit only segment 0 and its dependent supervisor initially; each
     successful supervisor submits one next normal training leaf.
2. `clariden/run_checkpoint_evaluation_debug.sbatch`
   - one debug allocation per checkpoint;
   - perform exact Megatron-to-HF conversion, native GreekMMLU, and receipt
     finalization in the same allocation;
   - use the existing v45 evaluator and conversion scripts without changing
     prompts, precision, model, tokenizer, or scoring;
   - submit one short `afterany` continuation before doing evaluation work;
     that continuation alone advances or retries the serial chain;
   - at per-document milestones, use four debug nodes: groups 0--2 use four
     GPUs on separate nodes, while group 3 and GreekMMLU use distinct GPUs on
     the fourth node concurrently after the same exact checkpoint conversion.
   - use `run_per_document_group_resource_aware.sh` to preserve the exact GPU
     IDs Slurm grants to each step while calling the unchanged v45
     `score_documents_hf.py` with the same model, tokenizer, bfloat16 dtype,
     inputs, and receipt format.
3. `scripts/audit_submitted_job_resources.py`
   - read the launch graph and `scontrol show job`;
   - fail unless every `full8b_s*` train job is 16-node `normal` with one leaf
     switch;
   - fail unless every supervisor, evaluation, controller, conversion,
     GreekMMLU, per-document, and finalizer job is `debug` and within 90
     minutes;
   - write an immutable allocation-routing receipt.
4. `scripts/rebind_selected_execution_profile.py`
   - read the passing v45 DP32 selection and promotion evidence;
   - validate the successor recipe and profiles;
   - require the same scientific digest, profile ID, nodes, world size, DP,
     gradient accumulation, and segment boundaries;
   - require the successor-versus-parent exact task-identity receipt;
   - write a new selected-profile receipt whose recipe/profile bindings point
     to the successor stage;
   - do not alter or reinterpret the parity thresholds.
5. `tests/test_resource_routing.py`
   - assert explicit partition/time flags;
   - assert the debug queue never exceeds one running/two submitted jobs;
   - assert the evaluation chain does not wait for a child while holding a
     debug allocation;
   - assert training args and the scientific digest are unchanged.
6. `evaluation/continue_checkpoint_evaluation.py`
   - inspect the terminal state and authoritative receipt of one evaluation;
   - retry `BOOT_FAIL`, `FAILED`, `NODE_FAIL`, `PREEMPTED`, `REVOKED`, and
     `TIMEOUT` up to two times;
   - treat transient `UNKNOWN` as a delayed recheck, not a dead campaign;
   - advance one evaluation or one next-segment supervisor per debug job.

This operational layer must have its own SHA-256 receipt. It does not replace
or mutate the v45 scientific bundle, and `FULL8_CODE_ROOT` for training remains
v45.

## Tests in strict order

No later stage runs until the preceding stage has a written passing receipt.

### T0 — allocation-free static and unit tests

Allocation: none; run on the Mac only for lightweight source checks and on the
Clariden login node only for ordinary metadata inspection.

Scripts/checks:

- `subprojects/07_full_8b_cpt/tests/test_full8b_orchestration.py`;
- `subprojects/07_full_8b_cpt/tests/test_anonymization_pipeline.py`;
- new `subprojects/07_full_8b_cpt/tests/test_resource_routing.py`;
- new `subprojects/07_full_8b_cpt/scripts/rebind_selected_execution_profile.py`;
- `subprojects/07_full_8b_cpt/scripts/validate_recipe.py`;
- `subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py`;
- normalized successor-versus-parent `packing_plan.json` comparison;
- pool arithmetic, catalog hashes, schedule hash, validation-manifest hash,
  tokenizer hash, initial-checkpoint hash, and scientific digest comparison;
- dry-run production graph generation;
- static audit that no operational submission can route a one-node job to
  `normal` and no training job can route to `debug`.

Pass criteria:

- all tests pass;
- successor task IDs, selected document IDs, order, token counts, and task
  boundaries exactly match the parent packed payload;
- only receipt paths/hashes differ;
- derived recipe remains 18,284 updates and scientific digest
  `41998a...b666`;
- rebound selected profile binds the successor recipe/profiles while retaining
  exactly `[0, 4000, 8000, 12000, 14627, 18284]` and the proven DP32 geometry;
- dry-run graph has exactly segment 0 plus one dependent debug supervisor; the
  frozen recipe carries all five segment boundaries and evaluation milestones.

### T1 — debug routing and nested-submission smoke

Allocations: first, one `debug` node for at most 00:05:00 which submits one
dependent debug child; second, one four-node `debug` job for at most 00:05:00
which proves the exact concurrent `srun --exact` GPU placement used at the two
per-document milestones. These run serially, so no more than two debug jobs
exist and the `MaxJobsPU=1` limit is respected.

Scripts:

- existing `clariden/prove_nested_sbatch.sbatch` behavior;
- new resource-aware submission layer in dry-run/probe mode;
- `scripts/audit_submitted_job_resources.py`.
- `clariden/prove_evaluation_overlap.sbatch`.

Pass criteria:

- parent exits before the child needs the only running debug slot;
- nested submission works under uenv with `--uenv-passthrough=ignore`;
- the routing receipt observes `debug/debug-qos` for the child;
- no `normal` job is created.
- group 3 document validation and GreekMMLU receive distinct GPUs on the same
  fourth node, while groups 0--2 receive their own four-GPU nodes.

This is the only new scheduler-behavior smoke. Do not rerun DP32, DP64, full
conversion, or graceful-stop training.

### T2 — successor-stage finalization and launch gate

Allocation: one `debug` node, maximum 00:30:00.

Scripts:

- v45 `scripts/derive_sanitized_contracts.py`;
- v45 `scripts/validate_recipe.py`;
- v45 `scripts/build_launch_gate.py`;
- v45 `scripts/capture_launch_environment.py`;
- resource-aware equivalent of
  `clariden/finalize_and_submit_production.sbatch`.

Actions:

1. Copy the parent `packed_corpus_receipt.json` only after exact normalized
   packing-plan identity is proven.
2. Bind the successor pool receipt, unchanged packed payload, unchanged
   schedule, and unchanged validation panel.
3. Derive the successor recipe/profiles.
4. Rebind the selected DP32 profile to the successor contracts through the
   exact task/scientific-identity receipt; do not rerun the 16-node benchmark.
5. Build a new prelaunch root; never overwrite the failed v45 prelaunch root.
6. Run the launch gate.
7. Generate and audit the dry-run job graph.
8. Only if every check passes, submit the real production graph.

Pass criteria:

- `launch_gate.json` has `status: passed`;
- allocation-routing receipt passes;
- production run root does not exist before the authorized submit;
- segment 0 is a pending 16-node `normal` job and its sole dependent
  supervisor is a one-node `debug` job;
- `operational_launch_gate.json` binds the frozen operations bundle and both
  debug routing smokes;
- every later dynamic submission is audited immediately and respects QoS
  cardinality.

## Production training allocations

Training uses v45 `clariden/train_segment.sbatch` with the exact derived recipe
and selected DP32 profile. `submit_production_resource_aware.sh` supplies the
partition and allocation flags explicitly.

| Segment | Updates | Updates in segment | Allocation | Median compute-only estimate |
| ---: | ---: | ---: | --- | ---: |
| 0 | 0–4,000 | 4,000 | 16 `normal` nodes, 64 GPUs, one leaf switch, 12 h | 9.65 h |
| 1 | 4,000–8,000 | 4,000 | same | 9.65 h |
| 2 | 8,000–12,000 | 4,000 | same | 9.65 h |
| 3 | 12,000–14,627 | 2,627 | same | 6.34 h |
| 4 | 14,627–18,284 | 3,657 | same | 8.83 h |

Total measured compute-only estimate: **44.13 hours**. This excludes queueing,
checkpoint pauses, recovery, and evidence completion. It is not a wall-clock
promise.

At initial launch, only segment 0 and its dependent supervisor are submitted.
After a segment completes, the supervisor freezes and validates the exact
checkpoint, signs the successor permit, adopts the one audited delayed holder
when present (or submits one normal training leaf as fallback), and starts the
first debug evaluation. The next normal leaf can queue or run while evaluations
proceed.
Each evaluation submits one short `afterany` continuation before it starts;
that continuation retries failures or advances one link. The last evaluation
for a segment submits the next supervisor with an `afterany` dependency on the
already-submitted training leaf. This is the earliest safe point at which each
normal request can be exposed without bypassing the checkpoint gate. The final
continuation may also expose the following delayed normal holder using the
target-runtime budget described in the addendum. There is no reservation or
guaranteed priority.

Training script chain:

1. resource-aware `clariden/submit_production_resource_aware.sh`;
2. v45 `clariden/train_segment.sbatch`;
3. v45
   `subprojects/06_dataset_scheduling_experiments/training/pretrain_scheduled_gpt.py`
   through the pinned Megatron root;
4. resource-aware `scripts/supervise_campaign_resource_aware.py` on `debug`,
   calling unchanged v45 checkpoint/audit scripts;
5. v45 `train/freeze_checkpoint.py` and training-attempt audit;
6. resource-aware debug checkpoint-evaluation chain;
7. resource-aware `evaluation/continue_checkpoint_evaluation.py` for
   after-any retry/advance decisions;
8. v45 `scripts/finalize_training.py` and `scripts/finalize_campaign.py` on
   `debug`.

Recovery policy:

- `TIMEOUT`, `NODE_FAIL`, `BOOT_FAIL`, `PREEMPTED`, and `REVOKED` recover only
  from a verified synchronous checkpoint;
- a `FAILED` job is retried only when a valid recovery checkpoint is proven;
- at most one audited, delayed, permit-gated happy-path successor is prequeued;
  no multi-job suffix is allowed, and recovery submits only one replacement
  normal leaf and one dependent debug supervisor;
- replacement training remains 16-node `normal`; replacement supervisors and
  evaluation work remain `debug`;
- never silently change DP, micro-batch, accumulation, precision, data order,
  optimizer, or LR schedule to rescue an allocation.

## Validation and GreekMMLU schedule

Source-conditioned validation is embedded in `train_segment.sbatch` and runs
on the same 16-node training allocation every 238 updates. It evaluates all 13
frozen panels with `eval-iters=1`; it needs no separate node request.

Native GreekMMLU checkpoints are:

`0, 400, 1192, 2384, 3576, 4768, 5960, 7152, 8344, 9536, 10728, 11920, 13112, 14304, 14627, 15496, 16688, 17880, 18284`.

Update 0 is already complete. The remaining exact checkpoints run through the
serial debug evaluation chain. The authoritative output includes public and
decontaminated GreekMMLU accuracy, NLL, and BPB using the frozen native Greek
evaluator. Conversion is measured at roughly five minutes and native
GreekMMLU at roughly 54--56 minutes; ordinary milestones therefore fit the
85-minute debug limit. At per-document milestones, GreekMMLU and document
scoring run concurrently after conversion, keeping the measured path near an
hour instead of the unsafe 88--90 minute sequential tail.

Per-document validation milestones are `0`, `14,627`, and `18,284`. Update 0
is already complete. The two remaining milestones score every document in all
13 panels and preserve document-level outputs for learning/forgetting plots and
bootstrap confidence intervals.

## Monitoring and handoff protocol

Use Europe/Athens explicitly for every operator-facing scheduler timestamp:

```bash
TZ=Europe/Athens squeue -u fffoivos
TZ=Europe/Athens sacct -X -j JOB_ID --format=JobID,JobName,State,Start,End,Elapsed,Partition,QOS,AllocTRES
```

After every transition report:

- segment index and exact update/token range;
- job ID, partition, nodes, GPUs, leaf-switch placement, start time, and state;
- latest completed update and observed median/p90 seconds per update;
- checkpoint receipt count;
- required/completed GreekMMLU and per-document receipts;
- next dependency and whether it is submitted, eligible, or merely predicted;
- compute-only, training-complete, and evidence-complete ETA separately.

Immediate fail conditions:

- any short one-node operational job appears in `normal`;
- any training segment appears in `debug`, on fewer/more than 16 nodes, or
  across more than one leaf switch;
- job graph differs from five frozen segment boundaries;
- scientific digest, recipe, tokenizer, schedule, validation manifest, packed
  receipt, or initial checkpoint drifts;
- source-validation logs omit any of the 13 panels;
- a resume does not load the exact last verified checkpoint/data cursor;
- GreekMMLU uses a different model geometry, tokenizer, prompt, dtype, or clean
  subset.

## Operator resume point

1. Confirm `squeue -u fffoivos` is empty or identify every live job.
2. Complete T0 against successor stage
   `20260808T064500Z-d0-v4-v45bridge`.
3. Freeze and verify the operational allocation layer.
4. Run T1 on `debug`.
5. Run T2 on `debug`.
6. Audit the real submitted graph immediately.
7. Confirm segment 0 is the first new 16-node allocation.
8. Monitor through five training segments and all evaluation receipts.

No operator should rerun the earlier 16-node parity or graceful-stop tests
unless their bound scientific inputs change. Allocation-routing changes alone
must be tested with static tests and `debug`, not another 16-node smoke.
