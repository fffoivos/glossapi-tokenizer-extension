# Receipt-bound bibliography cleaning

This directory replaces the contaminated plan-relative receipt workflow. It is
specific to the 13 academic ranks in the Agent-1 v5 deduplicated release.

The sequence is deliberately fail-closed:

1. Stage exact Git archives for the train and GlossAPI commits.
2. Run `preflight.sbatch`; all 431 local hashes and the pinned Hub commit must agree.
3. Run `parity.sbatch`; the exact staged GlossAPI commit must reproduce
   210,704/210,704 line masks and 19,117 positives. The same job builds the wheel.
4. Create `contract.json`, then `work_plan.json`. The run directory is
   `<UTC>-<contract-payload-sha12>` and is never shared with another plan.
5. Run `dryrun.sbatch`. Each stable unit ID is
   `<rank>-rg<start>-<end>` and its completion receipt binds the contract, plan,
   source shard, commits, wheel and model hashes.
6. Run `aggregate.py`, `build_qa.py`, review every packet item, then run
   `check_qa.py`.

The dry-run writes only receipts and one Parquet ledger row per analyzed document.
It does not write cleaned corpus fragments. An apply contract is a separate run and
is not authorized by the dry-run contract.

Frozen policy:

- analyze all 202,792 academic documents from all eight sources;
- future apply scope remains 175,242 documents in Greek PhD, OpenArchives, elocus
  and libduth;
- Kallipos is not automatically promoted into the apply scope;
- the libduth exception applies only to this private variant and never authorizes
  public redistribution;
- publication is not authorized;
- size columns are recomputed only where the source value was non-null.

QA passes only when all packet decisions are complete, none is body-only,
catastrophic or uncertain, at least 27/30 median Kallipos cuts are primarily
bibliography, and every OpenArchives removal over 50% is acceptable bibliography.
