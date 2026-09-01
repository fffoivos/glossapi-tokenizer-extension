# 06 · dataset — pool freeze, immutable packing, and the five schedules

> **In one line:** turned the published Greek corpus into four immutable packed sequence catalogs and five interleavings of them, so that D0–D4 could differ in *order* and in nothing else.
> **Period:** 2026-08-02. **Status:** completed; schedule manifest `ffeaa694…` frozen and consumed by the campaign.
> **Came from / led to:** the vetted 8B corpus receipt from [`../../05_token_distillation_cpt/`](../../05_token_distillation_cpt/) → this → [`../training/`](../training/) (the reader) and [`../evaluation/`](../evaluation/) (the disjoint heldouts).

## Why this existed

The experiment's whole claim is that the only difference between arms is temporal order. That is much stronger than "same document count". Goldfish hashes each target's preceding label context *inside* its 4096-token sample, so repacking the same documents in a different order would change which tokens contribute loss. The design therefore requires: pack each pool once into immutable payloads, give every payload a stable sequence ID and a Goldfish-mask hash, and let the schedules interleave IDs only ([`../FACTORIAL_EXPERIMENT_DESIGN.md`](../FACTORIAL_EXPERIMENT_DESIGN.md) §2).

## History

1. **Inputs derived, not re-selected.** [`derive_mini_corpus_inputs.py`](derive_mini_corpus_inputs.py) takes the frozen, already-vetted 8B corpus receipt as the source of eligible documents; [`prepare_mini_overlay.py`](prepare_mini_overlay.py) materializes the pinned Mini tokenizer and the Greek overlay on CSCS.
2. **Identity, not IDs.** [`validate_partition_group.py`](validate_partition_group.py) validates each physical partition against its frozen source records, using `(cluster_id, text_sha256)` because some source IDs legitimately name several distinct text records. [`diagnose_modern_content_duplicates.py`](diagnose_modern_content_duplicates.py) located exact Modern-Greek content duplicates across validated groups so HPLT and GlossAPI/non-HPLT could be made globally exact-content unique. Replay deliberately keeps its original records including measured repeats — deduplicating replay would change its distribution — with the duplicate rate receipted.
3. **Pool capacities frozen.** [`finalize_pool_corpus.py`](finalize_pool_corpus.py) reduced the partition receipts into the post-exclusion pool receipt `76658cc8…`: HPLT 44,042,201,419; GlossAPI/non-HPLT 19,734,450,444; foreign replay 16,145,987,813; Old-Greek replay 807,299,391 — **80,729,939,067 active tokens per arm**.
4. **Source-local packing as an I/O optimization only.** [`build_packing_plan.py`](build_packing_plan.py) and [`pack_catalog_bucket.py`](pack_catalog_bucket.py) select the exact seeded-prefix document set first, then reorder those rows by source task and document index purely to avoid scattered storage reads, packing into immutable 4097-token rows. The array ran as 512 tasks in 489 s (`../evidence/dataset_schedule_and_native_greekmmlu_plan_20260802.json`). [`finalize_packed_corpus.py`](finalize_packed_corpus.py) then freezes the sequence inventory and the stable per-pool IDs, and applies one seeded SplitMix64 permutation per pool — that randomized catalog, not the storage order, is the scientific pool order.
5. **The five schedules.** [`build_five_schedules.py`](build_five_schedules.py) materializes D0 (stationary windowed quotas), D1/D2 (hard mirrors switching at 69.0569356% / 30.9430644% progress) and D3/D4 (mirrored 128-window quota curves `g(u)=u^2.2317419754…`), with a deterministic largest-remainder rule and carried residuals. The frozen result: 19,709,692 real sequences + 260 loss-inactive filler sequences = 19,709,952 scheduled sequences = **38,496 optimizer updates**, zero terminal quota residual in every pool and arm, maximum single-window residual 35,320 tokens.
6. **A superseded first version.** The first manifest (`schedules_source_local`) was scientifically exact-once but claimed per-window residual reporting without emitting requested quotas and residual fields; v2 replaced it and its sequence-ID and active-token arrays were **byte-identical** to v1 for all five arms.
7. **Independent proof.** [`audit_schedule_only_factor.py`](audit_schedule_only_factor.py) proves out-of-band that D0–D4 share one exact sequence and active-token inventory, that replay sequence IDs sit at identical positions, and that five distinct Modern-Greek order trajectories exist — the check that passed in `../evidence/static_prelaunch_evidence_20260803.json`.
8. **One extra schedule for the LR smoke.** [`build_stability_smoke_schedule.py`](build_stability_smoke_schedule.py) built the balanced 1,024-step prefix used to select the common peak LR; see [`../training/`](../training/).

## Outcome

- A packing overhead of 2,024,325 token slots — 0.0025% of the horizon — contributing no loss.
- The schedule-only-factor and Goldfish-uniformity audits both passed before launch, which is what makes "order is the only factor" a checked statement rather than an assertion.
- Only the D0 schedule was reused later: subproject 07 re-emitted all five order schedules over the 8B binaries and consumed `D0_mixed` alone.

## Working documents

Eleven scripts, no prose docs. Contract: [`../FACTORIAL_EXPERIMENT_DESIGN.md`](../FACTORIAL_EXPERIMENT_DESIGN.md) §§2–3. Tests: [`../tests/test_source_local_packing.py`](../tests/test_source_local_packing.py), [`../tests/test_schedule_token_residuals.py`](../tests/test_schedule_token_residuals.py), [`../tests/test_partition_record_identity.py`](../tests/test_partition_record_identity.py), [`../tests/test_stability_smoke.py`](../tests/test_stability_smoke.py).
