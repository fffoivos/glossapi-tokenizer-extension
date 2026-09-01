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
   `check_qa.py`. Packet construction rechecks the exact ledger set and every
   post-aggregation ledger hash; the gate accepts only an explicitly complete
   review with non-empty rationales.
7. For the separate apply contract, run all frozen apply units and aggregate
   them, then run `materialize_release.py`. It independently rederives cleaned
   text from every ledger span, verifies all nonmutable columns, and creates a
   non-publishable 431-shard candidate.
8. Use `count_tokens.py` to count the original 431 shards plus only the nine
   transformed candidate shards with the pinned tokenizer. The cleaned total
   reuses receipts only for checksum-identical shards and reports raw text
   tokens plus training tokens with one EOS per document.
9. Run `finalize_public_release.py`; it requires the candidate, token summary,
   apply summary, public policy, and all 431 hashes to agree. Only that final
   root has `publication_ready: true`.
10. Dry-run and then execute `publish_private_agent1_v5.py --visibility public`.
    The publisher rejects visibility drift, unexpected files, and any remote
    revision whose bytes cannot be verified against the immutable local root.

The dry-run writes only receipts and one Parquet ledger row per analyzed document.
It does not write cleaned corpus fragments. An apply contract is a separate run,
filters the shared work plan to units explicitly marked `apply`, and writes an
atomic schema-preserving fragment plus ledger and receipt for each selected unit.

Frozen policy:

- analyze all 202,792 academic documents from all eight sources;
- apply scope is 175,242 documents in Greek PhD, OpenArchives, elocus and
  libduth;
- Kallipos is not automatically promoted into the apply scope;
- the owner explicitly includes libduth and authorizes the cleaned v2 target to
  remain public; this directive is recorded separately from the existing
  source-rights warning and is not represented as rightsholder permission;
- the target is the already-public, manually gated
  `fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2`;
- the cleaning contract itself cannot publish; publication requires a later
  independently verified receipt-bound action;
- size columns are recomputed only where the source value was non-null.

QA passes only when all packet decisions are complete, none is body-only,
catastrophic or uncertain, at least 27/30 median Kallipos cuts are primarily
bibliography, and every OpenArchives removal over 50% is acceptable bibliography.
