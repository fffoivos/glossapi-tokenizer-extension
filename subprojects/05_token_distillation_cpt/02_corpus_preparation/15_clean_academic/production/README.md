# production — receipt-bound bibliography cleaning

> **In one line:** the second attempt at running the cleaner over the whole corpus — rebuilt fail-closed after the first attempt contaminated its own results — which analysed all 202,792 academic documents, removed 2.93 B characters in a dry run, and passed a 209-item human QA gate before stopping at the apply boundary.
> **Period:** 2026-07-27 (`74bd44dd`, `4adfdfd2`, `0c6117f9`, `ad2c43e4`) → 2026-07-28 (`b8cc3ea2`, `698ca31c`, `e8fbec2c`, `d262c4fd`, `6f12e9d5`). **Status:** dry-run + QA complete; the apply contract was authorized on 2026-07-28 but **no apply, materialization, token-count or publication receipt exists in this tree**.
> **Came from / led to:** [`../bib_line_model`](../bib_line_model/README.md) → this → the public v2 dataset (`fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2`)

## Why this exists in this form

It replaces a contaminated plan-relative receipt workflow. The first attempt, documented in
[`../BIB_CLEANING_HANDOVER_20260727.md`](../BIB_CLEANING_HANDOVER_20260727.md) §6, ran two dry
runs (jobs 2908121 and 2910962) that both stopped early and both wrote receipts named
`{rank}-u{unit}` into the same directory — with unit numbering **relative to each plan**. The
169 receipts could not be told apart, and the contamination is provable by arithmetic:
`glossAPI/libiep` reported 12,010 documents against a source of 6,005, exactly 2×. That
directory was kept as forensic evidence and never reused; every absolute number from it is
unusable.

The fix is structural: **immutable v2 run contracts and stable row-group unit IDs**
(`<rank>-rg<start>-<end>`), a run directory named `<UTC>-<contract-payload-sha12>` that is never
shared with another plan, and explicit separate `dry-run` and `apply` modes.

## The sequence

1. Stage exact Git archives for the train and GlossAPI commits.
2. `preflight.sbatch` — all 431 local hashes and the pinned Hub commit must agree.
3. `parity.sbatch` — the exact staged GlossAPI commit must reproduce **210,704/210,704 line
   masks and 19,117 positives**. The same job builds the wheel.
4. `create_contract.py` → `build_plan.py`.
5. `dryrun.sbatch` — each unit's completion receipt binds contract, plan, source shard, commits,
   wheel and model hashes.
6. `aggregate.py` → `build_qa.py` → review every packet item → `check_qa.py`. Packet
   construction rechecks the exact ledger set and every post-aggregation ledger hash; the gate
   accepts only an explicitly complete review with non-empty rationales.
7. Apply is a **separate contract**: run the frozen apply units, aggregate, then
   `materialize_release.py`, which independently rederives cleaned text from every ledger span,
   verifies all nonmutable columns, and creates a non-publishable 431-shard candidate.
8. `count_tokens.py` — the original 431 shards plus only the nine transformed candidate shards,
   with the pinned tokenizer; reuses receipts only for checksum-identical shards.
9. `finalize_public_release.py` — requires candidate, token summary, apply summary, public policy
   and all 431 hashes to agree. Only that root gets `publication_ready: true`.
10. `publish_private_agent1_v5.py --visibility public`, dry run first. The publisher rejects
    visibility drift, unexpected files, and any remote revision whose bytes cannot be verified
    against the immutable local root.

The dry run writes only receipts and one Parquet ledger row per analysed document — **no cleaned
corpus fragments**. The cleaning contract itself cannot publish.

## Frozen policy

[`policy.json`](policy.json), schema `bibliography-cleaning-policy-v2`:

- **Analyse** all 202,792 documents across 13 academic ranks; **apply** to 175,242 in greek_phd,
  openarchives.gr, elocus and libduth.
- **Kallipos is not automatically promoted** into apply scope (`kallipos_apply_authorized: false`),
  and neither are Pergamos, libiep or eellak-articles.
- Model `heading_lexgate`, line threshold 0.9, with
  `character_damage_measure_approved: true` — the owner-approved replacement of line precision by
  a character-denominated body-damage budget, accepted 2026-07-27.
- The libduth licence override is recorded as a dataset-owner directive that *"does not represent
  rightsholder permission or supersede the recorded source-license warning"*, approved 2026-07-28.
- QA gate: zero catastrophic / body-only / uncertain decisions, ≥27/30 Kallipos cuts primarily
  bibliography, and **every** OpenArchives removal over 50% acceptable.

## What ran

Full evidence in [`../BIB_CLEANING_IMPLEMENTATION_20260727.md`](../BIB_CLEANING_IMPLEMENTATION_20260727.md).

- **Preflight job 2912077:** 431 files, 51,839,746 rows, 141,797,094,485 bytes, zero drift,
  Hub commit `c368d37c…`.
- **Sealed parity job 2912714:** 210,704/210,704 masks, 19,117/19,117 positives.
- **Dry run `20260727T193808Z-ddf94a84b8b7`**, Slurm array 2912781, 157 units, all exit 0:
  157/157 receipts, 157/157 ledgers, 202,792/202,792 documents, zero partial files, zero
  `would_empty`, zero worker stderr.
  - **159,142 / 202,792 documents cut (78.475482%)**
  - **2,933,770,472 / 48,373,473,465 characters removed (6.064833%)**; 28,190,335 / 401,650,438 lines; 1,980,170 spans
  - p50 0.052067, p95 0.210334; 1,540 documents over 30%, 192 over 50%
- **QA:** packet = 30 deterministic median-sized Kallipos cuts + every OpenArchives >50% removal
  + every would-be-empty document. Result: **209/209 decisions complete and acceptable, zero
  catastrophic / body-only / uncertain, 30/30 Kallipos primarily bibliography, 179/179
  OpenArchives >50% removals acceptable, zero empty-document items.**

The dry run also **failed closed** three times on previously unseen long lines, isolating
`_PLACE_PUBLISHER_SHAPE`, then `_VOLUME_MARKER` on a 2,228-byte near-match; each was fixed by a
semantics-preserving atomic rewrite. A broader proactive author-pattern rewrite changed **9 of
210,704 sealed decisions and was completely reverted**. That sequence is the argument for making
exact parity a contract gate rather than an informal test.

## Where things are

| Path | What |
|---|---|
| [`policy.json`](policy.json) | The frozen policy — ranks, scopes, model, QA gate, licence override, publication target. |
| [`contracts.py`](contracts.py) · [`create_contract.py`](create_contract.py) · [`build_plan.py`](build_plan.py) | Immutable contract and work-plan construction. |
| [`preflight.py`](preflight.py)/[`.sbatch`](preflight.sbatch) · [`parity.py`](parity.py)/[`.sbatch`](parity.sbatch) | The two gates that must pass before any unit runs. |
| [`run_unit.py`](run_unit.py) · [`run_units.sh`](run_units.sh) · [`dryrun.sbatch`](dryrun.sbatch) · [`apply.sbatch`](apply.sbatch) | Unit execution; `apply.sbatch` sets `MODE=apply NODE_COUNT=3 SLOTS=36 THREADS=8`. |
| [`aggregate.py`](aggregate.py) | Exact aggregation; rejects duplicates, missing units and post-aggregation ledger mutation. |
| [`build_qa.py`](build_qa.py) · [`check_qa.py`](check_qa.py) | Deterministic QA packet rehydrated from source rows with verified span hashes, and the frozen fail-closed gate. |
| [`materialize_release.py`](materialize_release.py) · [`count_tokens.py`](count_tokens.py)/[`.sbatch`](count_tokens.sbatch) · [`finalize_public_release.py`](finalize_public_release.py) | The post-apply chain. None of it has a receipt here. |
| [`tests/test_production.py`](tests/test_production.py) | Five synthetic production tests (all passing per the implementation record). |
