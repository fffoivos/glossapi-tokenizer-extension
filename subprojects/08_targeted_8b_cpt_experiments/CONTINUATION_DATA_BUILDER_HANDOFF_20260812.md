# Continuation data-builder handoff

## What is preserved

The exact data builder used for Experiment B is preserved as two independent
stages:

1. `scripts/build_continuation_b_schedule.py` selects previously unseen
   sequences from a receipt-bound parent schedule suffix, preserves the exact
   parent prefix, creates the continuation ordering and writes the complete
   hybrid schedule.
2. `scripts/build_continuation_b_pool_view.py` binds that schedule to the
   already-packed parent corpus and its payload-integrity receipt without
   rewriting or deduplicating the payload.

The machine-readable preservation contract is
`configs/continuation_data_builder_v1.json`. It pins both implementation
hashes, the exact Experiment B invocation, its immutable scientific bundle and
the four resulting schedule/pool receipts. The v1 builder files must not be
edited in place for a different data mix.

## What can already vary

The v1 schedule builder accepts a different receipt-bound parent schedule,
checkpoint tree and receipt, parent arm, checkpoint iteration, global batch,
expected modern-token anchor, output window count and output directory. These
are execution inputs, not permission to change the mix semantics.

## What is fixed in v1

The current policy consumes every unseen non-HPLT (`G`) sequence after the
checkpoint, selects unseen foreign (`F`) and Old-Greek (`O`) replay by the
79/20/1 active-token geometry, and deterministically smooths those three
streams while preserving relative order inside each stream. It requires zero
overlap with the already-consumed prefix and adds only loss-inactive filler to
complete the final global batch.

The current builder does not expose HPLT selection, arbitrary pool weights,
source-family quotas or curricula as command-line options. That is deliberate:
adding one of those is a new scientific data-mix policy, not a harmless runtime
parameter.

## How to build another mix safely

Keep `continuation_data_builder_v1` and the Experiment B artifacts immutable.
Define a new policy ID and either create a new builder version or add an
explicit policy-schema layer around the shared primitives. A new mix must then
prove, at minimum:

- exact parent checkpoint and schedule bindings;
- byte-exact preservation of the consumed prefix;
- zero selected-sequence overlap with that prefix;
- exact source-sequence identities, quotas and deterministic ordering;
- active-token totals and residuals by pool;
- absence of any extra global deduplication;
- loss-inactive-only tail padding;
- restart/sample-cursor parity and a new launch-gate receipt.

This makes future HPLT/non-HPLT/replay comparisons possible without changing
or losing the builder that produced the current continuation experiment.
