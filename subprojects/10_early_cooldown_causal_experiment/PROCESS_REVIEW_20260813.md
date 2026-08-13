# Early-cooldown launch process review

## Outcome of the first direct same-allocation launch

The first direct causal allocation did **not** begin the 3,657-update branch.
Normal-partition job `3075352` ran for 5 minutes 42 seconds on 16 four-GPU
nodes and stopped at the predeclared update-9,537 gate. This consumed 6.08
normal-partition GPU-hours. No checkpoint from this attempt is accepted as
scientific branch state.

The gate established all of the following on one allocation:

- the fully hashed update-9,536 checkpoint loads with optimizer, scheduler,
  RNG and cursor state;
- the parent and cooldown probes consume the same update-9,537 samples;
- consumed samples/tokens, global batch, target counts and UTF-8 bytes match;
- LM, base-token and added-token losses match exactly at logged precision;
- parameter norm, loss scale, NaN count and skipped-update count match;
- the intervention LR is below the parent peak LR;
- the intervention update-9,537 distributed checkpoint is complete.

The gate rejected the run because the displayed gradient norm was `2.011` for
the parent control and `2.010` for the cooldown probe. The training receipt is:

`/capstor/scratch/cscs/fffoivos/runs/10_early_cooldown/20260813T145000Z-causal-wsd10-direct-v3/training_receipts/branch_job_3075352.json`

The supervisor job `3075353` correctly classified this as a permanent blocker,
submitted no recovery training and no evaluations, and exited nonzero. Its
nonzero Slurm state is expected fail-closed behavior, not a second training
failure.

## Contract issues found by the review

The v3 parity contract omitted explicit equality checks for consumed samples,
consumed tokens, global batch size and loss scale. Those fields happened to
match, but relying on that without a declared check was a gap. They are exact
fields in v4.

An exact two-run gradient comparison is not a robust discriminator at the
logger's three-decimal precision: the failed result differed by one displayed
quantum even though every forward/cursor field matched. The failed receipt is
not reinterpreted and its threshold is not widened.

Version 4 instead predeclares a fresh peak/cooldown/peak sandwich:

1. load update 9,536 and run a no-save parent-LR control;
2. reload update 9,536, run the cooldown intervention and save update 9,537;
3. reload update 9,536 and run a second no-save parent-LR control;
4. require every non-gradient field to match exactly across all three;
5. require the two parent gradients to differ by at most one logger quantum;
6. require the intervention gradient to fall inside that parent-control
   envelope; if the controls agree, exact equality remains mandatory;
7. verify the intervention checkpoint metadata did not change during the final
   control, restore `latest_checkpointed_iteration.txt` to 9,537 and only then
   continue the branch.

This makes gradient reproducibility a concurrent control measured around the
intervention. It does not accept or reuse the failed v3 checkpoint.

## Replay-reservoir operational chain

The replay reservoir itself completed successfully:

- gate `3075007`: completed;
- packing `3075053`: completed in 2 minutes 19 seconds;
- finalization `3075072`: completed in 40 seconds;
- standard profile: exact 79/20/1 active-token accounting;
- payload: 7,604,182 packed replay sequences.

The original xfer upload `3075091` started at the maintenance boundary and was
scheduler-cancelled after one second with `ReqNodeNotAvail`. It produced no
upload receipt. Its impossible `afterok` descendants were replaced only after
the new jobs were submitted and resource/dependency-audited:

- private HF upload `3075934` on `xfer`;
- immutable full-SHA hydration `3075937` after upload;
- production-reader smoke `3075941` on `debug` after hydration.

The reader smoke opens the first and quota-capped final sequence from both
standard replay pools through `ScheduledPackedGPTDataset` and verifies every
scheduled tail target is loss-inactive. The large transfer and full hash remain
off `normal` allocations.

## Allocation-selection correction

The first launch path pinned the request to `group29` by excluding every other
leaf. That constraint was operationally unnecessary: causal parity requires the
three probes to share one allocation, not one permanently named leaf. It also
reduced the eligible 16-node capacity while `group29` was fully allocated or
drained.

The first replacement allowed the scheduler to choose any eligible leaf with
`--switches=1`, while mapping the result and failing closed unless exactly one
leaf was observed. Live job `3076070` proved that Clariden normalizes this to
`Switches=1@00:05:00`: after five minutes the preference is relaxed. It started
across four leaves and exited before parity or training in 41 seconds. An
explicit seven-day wait was also normalized back to five minutes; the exact
post-submit audit canceled jobs `3076846` and `3076847` at zero runtime.

The final placement path uses the already validated leaf-exclusion helper.
Read-only test-only probes of every compute leaf selected `group36` as the
earliest eligible leaf, and it is also the leaf used by the proven DP32
profile. Every node outside group36 is hard-excluded from initial and recovery
allocations. Runtime topology mapping still fails closed and emits a classified
receipt containing every observed leaf if Slurm ever violates the binding.

## Allocation status at review time

At 2026-08-13 18:18 CEST there was no live training allocation. The replacement
replay jobs were pending behind xfer maintenance. A replacement 16-node branch
must use a new run root, new immutable v4 bundle, fresh debug prelaunch and a
fresh audited Slurm job; it must not resume the rejected v3 root.

## Replacement launch and final verification

The scheduler-selection correction was committed and pushed directly as
`7050d596` (no pull request). Verification completed before deployment:

- causal contract tests: 14 passed;
- inherited scheduling and full-8B suites: 205 passed plus 61 subtests;
- all causal shell entry points passed `bash -n`;
- all causal Python sources compiled;
- the static owner-authorized v4 contract passed;
- the replay-reader branch passed 208 tests plus 61 subtests and its changed
  reader/test files passed Ruff;
- the portable replay/HF tooling passed all 25 unit tests and Ruff.

Debug job `3076023` copied, validated, fully hashed and froze the replacement
scientific bundle in 54 seconds:

`/iopsstor/scratch/cscs/fffoivos/orchestration/early-cooldown-8b/20260813T164000Z-7050d596-v8`

Its 840-file tree SHA-256 is
`f5eb8be3ded21e13e6c4758210a83354156bcaa0427930715daf83659c54dcc9`.
Debug prelaunch job `3076055` then regenerated the recipe, scientific launch
gate, exact Slurm test-only receipt and operational gate from that immutable
bundle.

Production training job `3076070` and its after-any debug supervisor `3076071`
were submitted from the exact tested command. The audited training request is
16 `normal` nodes for 12 hours with `Switches=1@00:05:00` and
`ExcNodeList=(null)`. It was pending on priority at 18:39 CEST. The run root is:

`/capstor/scratch/cscs/fffoivos/runs/10_early_cooldown/20260813T164000Z-causal-wsd10-direct-v4`

The independent replay portability chain remains off the training critical
path. Upload `3075934` was later allocated to `nid001154` during maintenance
and scheduler-cancelled after one second, still before producing an upload
receipt. Replacement `3076327` began after maintenance but failed closed in
10 seconds because `uenv` is not installed on xfer nodes; it moved no data.
The first pinned standalone runtime was then built on GH200 `debug`, and an
actual xfer smoke rejected it because its aarch64 NumPy wheel could not load on
the x86_64 transfer node. This exposed an architecture error before upload.

The corrected transfer tooling was tested and pushed directly as `a32bbac`.
Immutable efficiency bundle v23 has tree SHA-256
`f626c9b9b67d1a1706e86a7a2d46ffb7a601ef77e40677ec4331cf7ccaff03fc`.
Xfer job `3076545` built and twice verified the x86_64 runtime in 2 minutes 31
seconds; its tree SHA-256 is
`7612964168d9a116c04dad34cc515ec7225ba40443d502576a8a86f63e7685e1`.
Private upload `3076559` was submitted on xfer from v23. Full-SHA hydration
`3076562` used `afterok:3076559`, and production-reader smoke `3076564` used
`afterok:3076562`. The obsolete descendants `3075937` and
`3075941` were canceled with zero runtime only after the corrected replacement
path was ready. No `normal` allocation was created or changed by this repair.

The v23 transfer chain subsequently completed: upload `3076559` took 5 minutes
3 seconds and froze private HF revision
`9a520bd83d4d70061bb4e477e69ee37800fd2b85`; hydration `3076562` took 8
minutes 53 seconds and full-SHA verified 1,039 payload files totaling
124,965,755,323 bytes. Its first reader smoke `3076564` then failed closed
before opening data because an over-escaped `bash -lc` passed the Python path
literally. Commit `ea6613c` replaced that shell layer with direct
`uenv ... env PYTHONPATH=... python3 ...` execution and added a regression
assertion. Replacement debug smoke `3076649` completed in 5 minutes 40 seconds;
the production-reader portion took 46.68 seconds and passed all checks: first
and quota-capped final sequences of both replay pools opened, every payload had
4,097 tokens and every sample 4,096 targets, and 3,142 foreign plus 885 Greek
terminal targets beyond the exact quotas were loss-inactive. The portable cache
receipt SHA-256 is
`7d9baf0d79aa00a45ec38b36767ba41643c77c30202d9c868c8c5e7808c33fd4`.

## Hard-placement relaunch

The hard-placement correction was committed and pushed directly as `66203d88`.
The full causal suite remained at 17 passing checks; the inherited scheduling
suite passed 147 tests and the full-8B orchestration suite passed 43. Debug job
`3076870` froze immutable v10 with 843 files and tree SHA-256
`ffdf1b81cc06ca380dc77b0b8b32a3412a2bb3fa29abb2c791a71de6a10cf691`.
Debug prelaunch `3076878` then regenerated the unchanged scientific recipe and
passed the exact group36 test-only command. Fresh run v6 is queued as training
job `3076888` with after-any supervisor `3076889`:

`/capstor/scratch/cscs/fffoivos/runs/10_early_cooldown/20260813T205520Z-causal-wsd10-direct-v6`

The audited request has 16 normal nodes, no required-node list, and a hard
exclusion containing every level-0 leaf except group36. Thus the cluster's
five-minute switch relaxation cannot produce another multi-leaf allocation.
