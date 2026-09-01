# 04/tests — the local gate

> **In one line:** 53 pytest modules that stood in for a corpus run, since almost nothing here could be exercised on the development Mac.
> **Period:** 2026-07-11 → 2026-07-28. **Status:** maintained to the end of the phase.
> **Came from / led to:** run by [`../clariden/prepare.sh`](../clariden/prepare.sh), which does no SSH, download or submission.

## Why this existed

Phase 04 is dry-run-first by design: preparing a script must not download a dataset or submit a job. The consequence is that correctness had to be demonstrated on synthetic fixtures before any Clariden time was spent, and that receipt contracts — not outputs — were the thing worth testing.

## What is covered

| Area | Modules |
|---|---|
| Contracts and receipts | `test_receipt_contracts.py`, `test_stage_contract_inventory.py`, `test_clariden_dag_contracts.py`, `test_validate_configs.py`, `test_finalization_pipeline.py` |
| Acquisition and sources | `test_download_locked_sources.py`, `test_mdc_acquisition.py`, `test_normalize_sources.py`, `test_source_lineage_review.py`, `test_source_license_adjudication.py`, `test_source_quality_profile.py`, `test_merge_source_admissions.py`, `test_apertus_overlap_actions.py` |
| Cleaning, decontamination, dedup, release | `test_apply_cleaning_policy.py`, `test_structural_last_cleaning.py`, `test_decontaminate_full_corpus.py`, `test_dedup_contract.py`, `test_release_integrity.py` |
| Structural detector and parity | `test_detector_cli.py`, `test_detector_parity_edges.py`, `test_rust_parity_struct_modern.py`, `test_structural_span_production.py`, `test_structural_token_loss.py`, `test_structural_classifier_selection.py` |
| Quality/review site | `test_dataset_quality_site.py` (grew across `72026703`, `a933d984`, `afaaeff4`, `8795e6bf`, `33653540`, `84b6ab63`) |
| Agent 1 v3 lane | 18 `test_agent1_v3_*.py` modules — contract, pipeline, review, review packet, review evidence, admission, dedup, decontaminate, anonymize, postmask closure, structural child/gate, transformation waterfall, release, candidate roster, dispatcher, masked review-sample quality, pre-review |
| Agent 1 v4 lane | `test_agent1_v4_raw_review.py`, `test_agent1_v4_gfm_normalization.py`, `test_gfm_luna_validation.py`, `test_run_agent1_v3_codex_reviews.py`, `test_run_codex_source_reviews.py` |
| Agent 1 v5 lane | `test_agent1_v5_pipeline.py`, `test_agent1_v5_dedup_acceleration.py`, `test_audit_agent1_v5_acquisition_integrity.py`, `test_audit_agent1_v5_release_quality.py`, `test_publish_private_agent1_v5_metadata.py` |

## Outcome

- `test_detector_parity_edges.py` and `test_rust_parity_struct_modern.py` exist because Rust span offsets are Unicode scalar values and Python indexes code points: the fixtures deliberately include astral and polytonic characters so a byte-offset bug cannot pass silently ([`../STRUCTURAL_SPAN_PRODUCTION.md`](../STRUCTURAL_SPAN_PRODUCTION.md)).
- The v5 tests were run *on the cluster* under the pinned CSCS runtime, which does not package `pytest` — receipt/output closure, chunk-plan approval and the deterministic five-worker benchmark selection were executed directly, and the new Bash wrappers passed `bash -n` ([`../../../../docs/AGENT1_V5_DEDUP_ACCELERATION_IMPLEMENTATION_STATUS_2026-07-18.md`](../../../../docs/AGENT1_V5_DEDUP_ACCELERATION_IMPLEMENTATION_STATUS_2026-07-18.md)).
- `test_agent1_v5_pipeline.py` was the last test file to change in the phase, extended with the publisher's visibility gate on 2026-07-28 (`e8fbec2c`).
