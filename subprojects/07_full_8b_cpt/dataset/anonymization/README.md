# 07 · dataset/anonymization — the sanitized corpus

> **In one line:** the pipeline that masked PII, excluded the OCR-flagged rows, globally deduplicated the masked text and published a training bridge — the reason the first 8B trajectory was thrown away and the second one exists.
> **Period:** 2026-08-07 (commit `5b6dd260` plus ten follow-ups the same day). **Status:** completed; produced the corpus of run `20260808T121000Z-d0-wsd10-sanitized-successor-v12`.
> **Came from / led to:** the stopped pre-sanitization trajectory ([`../../presentations/FULL8_EXPLORATORY_PREFIX_20260806.html`](../../presentations/FULL8_EXPLORATORY_PREFIX_20260806.html)) → this → the derived sanitized recipe and the completed run.

## Why this existed

The first production run reached update 7,152 on text that was GreekMMLU-decontaminated but had **not** received the required Apertus PII anonymization pass. It was stopped for that reason alone. The full data contract for the replacement is [`../../SANITIZED_RESTART_RUNBOOK_20260807.md`](../../SANITIZED_RESTART_RUNBOOK_20260807.md).

## The pipeline

| Stage | Script | What it enforces |
|---|---|---|
| Freeze the inputs | [`freeze_anonymization_overlay.py`](freeze_anonymization_overlay.py) | Parent corpus, raw validation, masker and executing code are all hash-bound before anything is masked. |
| Prove eligibility | [`audit_sanitized_eligibility.py`](audit_sanitized_eligibility.py) | Independently re-derives the row-eligibility policy and the PII token contract: exactly **6,648** `openarchives.gr` rows with `needs_ocr == true` excluded, zero retained matches. |
| Mask and inventory | [`build_anonymization_inventory.py`](build_anonymization_inventory.py), [`anonymization_common.py`](anonymization_common.py) | Apertus-parity email/IP masking and a validated country-length IBAN masker, applied to Modern Greek, foreign replay and Greek replay; post-mask content hashes inventoried per task. |
| Reuse safely | [`promote_compatible_inventory.py`](promote_compatible_inventory.py) | A prior stopped stage's masked inventory may be promoted **only** if byte-identical and only when the canonical task contract and masker SHA-256 match. |
| Deduplicate | [`finalize_postmask_dedup.py`](finalize_postmask_dedup.py) | Global exact dedup of masked text plus removal of raw-validation collisions. |
| Build shards | [`build_sanitized_binary_shard.py`](build_sanitized_binary_shard.py) | One decontaminated, anonymized, globally deduplicated binary shard at a time. |
| Publish | [`finalize_sanitized_bridge.py`](finalize_sanitized_bridge.py) | Validates every shard and emits the training-bridge receipt the launch gate consumes. |

## The bugs this pipeline was built around

- **Row identity is not document identity.** `doc_id` alone is not a physical row: v7 found 27 repeated IDs spanning 32 additional records in one shard, so a set of IDs could drop the wrong text or every copy. Dedup is keyed by `(doc_id, masked SHA-256)` with row multiplicity preserved, and the shard gate consumes exactly the recorded number of occurrences (`317636c2`, `e80ad793`).
- **Ownership order changes capacity.** The original lowest-task-index rule left only **11,529,074** Old-Greek replay tokens against a 2,666,110,500-token source and failed the 1% capacity gate. Within a safe duplicate group the quota-limited Old-Greek row is now preferred; every exact text still survives once (`2bc4cf40`).
- **Index sentinels are not documents.** Every shard index carries one terminal sentinel, including empty shards; summing index entries inflated the v1 bridge display by exactly **1,457** — one per task. Bridge v2 requires `index_entries == documents + tasks` (`20fb7294`, `303ccf67`).
- **Heldouts must be masked before comparison.** Replay heldouts are reconstructed through the same frozen masker before their hashes or token lengths are compared with sanitized shard ledgers; comparing raw Parquet text to masked training text fails on any document containing an email, IP or IBAN, and that must stay a hard error (`5f4383f3`).
- **A shard accounting failure is not a retryable scheduler event.** The runbook forbids weakening a row-closure assertion to make a shard pass.

## Outcome

- Masking changed **2,515,489** documents; the post-mask exact pass dropped **2,386,676** documents (2,378,595 exact duplicates + 8,081 validation collisions) out of 97,136,622 inputs.
- The recomputed D0 horizon: **76,685,490,476 active tokens / 18,284 updates**, replacing the 80.73B / 19,248 planning geometry.
- **Open, deliberately:** the second global deduplication was outside the requested anonymization scope, and the evidence does not separate duplicates newly created by masking from duplicates already in the frozen v2 multiset. The owner deferred the review on 2026-08-09 with no change to the running job ([`../../DEFERRED_POSTMASK_DEDUP_REVIEW_20260809.md`](../../DEFERRED_POSTMASK_DEDUP_REVIEW_20260809.md)). Subproject 09 records the consequence: the run is internally valid but is **not** a data-identical replication of the earlier run or of the 0.5B study.

## Working documents

Eight scripts, no prose docs; the contract is the runbook. Test: [`../../tests/test_anonymization_pipeline.py`](../../tests/test_anonymization_pipeline.py).
