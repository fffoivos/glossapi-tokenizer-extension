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

## Allocation status at review time

At 2026-08-13 18:18 CEST there was no live training allocation. The replacement
replay jobs were pending behind xfer maintenance. A replacement 16-node branch
must use a new run root, new immutable v4 bundle, fresh debug prelaunch and a
fresh audited Slurm job; it must not resume the rejected v3 root.
