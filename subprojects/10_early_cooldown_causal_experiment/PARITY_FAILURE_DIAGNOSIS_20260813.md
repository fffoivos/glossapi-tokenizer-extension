# Restart-parity diagnosis and corrected causal gate

## Decision

Do not compare an update's gradient norm to a historical value measured on a
different GH200 allocation. Do not replay 1,536 peak-LR updates merely to
reconstruct a checkpoint that already has a complete per-file SHA-256 receipt.

The first production gate compared two update-9,537 probes from the same exact
checkpoint on the same nodes and allocation. Job `3075352` correctly rejected
the intervention because the displayed gradient norm was `2.011` in the parent
control and `2.010` in the cooldown probe; every other declared field matched
exactly. Its saved cooldown checkpoint is rejected.

The replacement is predeclared as a peak/cooldown/peak sandwich. The two parent
controls establish the reproducibility envelope concurrently on the exact same
nodes. All non-gradient scientific and cursor fields must match exactly. The
cooldown gradient must be inside the parent-control envelope and the controls
may differ by at most one quantum of the logger's three-decimal precision. If
the parent controls agree, the intervention must agree exactly. This avoids
selecting a tolerance from the failed intervention itself.

## Evidence from the rejected direct restart

Run:

`/capstor/scratch/cscs/fffoivos/runs/10_early_cooldown/20260812T231500Z-causal-wsd10-v1`

At update 9,537, the direct restart matched the scheduled target counts and
bytes exactly. Losses were within `7.5e-5` and parameter norm matched to the
logged precision, but gradient norm was `2.016` rather than the historical
`0.669`. The run failed closed.

## Evidence from the rejected parent replay

Run:

`/capstor/scratch/cscs/fffoivos/runs/10_early_cooldown/20260813T082200Z-causal-wsd10-replay-v2`

Job `3072568` loaded update 8,000 and reproduced update 8,001 exactly,
including gradient norm. Divergence began at update 8,002 after the first BF16
distributed optimizer step. At update 9,536 the observed LM loss differed from
the historical log by `0.002222` and parameter norm by `-0.038`; at update
9,537 LM loss differed by `0.001577` while the gradient norm was `0.127` versus
`0.669`. Target counts and UTF-8 bytes remained exact and there were no skipped
or non-finite updates. The replay gate failed and the branch holder was placed
in a user hold before it could allocate nodes.

The failed receipt is:

`replay/parent_replay_receipt.json`

## Checkpoint completeness

The original and replayed update-9,536 checkpoints were inspected with
`scripts/inspect_dcp_metadata.py` on a Clariden debug node. Both have:

- 230 logical state-dict entries;
- 18,798 storage entries;
- 52 optimizer entries;
- two RNG entries;
- two rerun-state-machine entries;
- the same 131 filenames;
- identical logical tensor metadata and file sizes, except for `.metadata` and
  `common.pt` serialization sizes.

`scripts/inspect_checkpoint_common.py` confirmed that both checkpoints record
iteration 9,536, `39,996,882,944` tokens, `9,764,864` consumed samples, the same
optimizer-scheduler step, the original 18,284-update AdEMAMix ramps and the same
training geometry.

Most importantly, the parent's existing source-checkpoint receipt hashes all
131 files, including every `.distcp` payload, `.metadata` and `common.pt`:

`/capstor/scratch/cscs/fffoivos/runs/07_full_8b_cpt/20260808T121000Z-d0-wsd10-sanitized-successor-v12/checkpoint_evaluations/iter_0009536/attempt_0/export/source_checkpoint_receipt.json`

Its `.metadata` SHA-256 is
`0b0820699a1fd1bf34bcade155a801a93a7aa6ac74be439301addef321726489`.

## Why the sandwich does not accept the failed result post hoc

The failed A/B receipt remains failed and cannot satisfy the new gate. A fresh
allocation must generate two new same-LR controls around a new intervention.
The gradient rule is tied to the logger's existing display precision, not to an
unbounded relative tolerance: the control spread can be at most one displayed
unit, and the intervention must be inside that independently measured spread.
Every other logged scientific and cursor field remains exact.
