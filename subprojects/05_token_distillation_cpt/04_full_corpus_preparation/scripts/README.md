# 04/scripts — the CPU tooling, in four generations

> **In one line:** 102 CPU-only tools written across three weeks in four overlapping generations — the canonical v2 pipeline, the v3 ordered lane, the v4 raw-review lane and the v5 DataTrove lane — of which the v4 and v5 families are the ones that produced artifacts.
> **Period:** 2026-07-11 → 2026-07-28. **Status:** completed; the v2 and v3 families were implemented, tested and never executed.
> **Came from / led to:** driven by [`../clariden`](../clariden) and [`../slurm`](../slurm); validated by [`../tests`](../tests); bound by [`../schemas`](../schemas).

## Why this existed

Every stage had to be resumable, fail-closed and auditable on a shared cluster where a job could be killed at 85 minutes. That pushed nearly all logic out of the sbatch files and into Python that emits an immutable receipt, revalidates any byte it reuses, and refuses to overwrite an existing output.

## The four generations

### v2 — the canonical full-corpus pipeline (2026-07-11 → 07-12, never run)

Acquisition and registry: `resolve_sources.py`, `download_locked_sources.py`, `acquire_mdc_sources.py`, `merge_acquisition_receipts.py`, `finalize_acquisition.py`, `validate_input_receipt.py`, `validate_configs.py`.
Normalisation and lineage: `normalize_sources.py`, `full_corpus_io.py`, `source_lineage.py`, `build_source_lineage.py`, `build_apertus_overlap_actions.py` (the 2.2M-row overlap drop overlay), `stream_parquet_jsonl.py`.
Review: `build_source_review_packet.py`, `run_codex_source_reviews.py`, `aggregate_source_reviews.py`, `merge_source_admissions.py`.
Cleaning: `apply_cleaning_policy.py`, `cleaning_runtime.py`, `greek_pii.py` (high-precision Greek direct-identifier masking), `profile_source_quality.py` (the Diavgeia profiler that writes span and document-action ledgers without changing text), `finalize_structural_cleaning.py`.
Decontamination, dedup, release: `freeze_greekmmlu_queries.py`, `decontaminate_full_corpus.py`, `run_full_corpus_dedup.py`, `full_corpus_dedup_recipe.py`, `invoke_text_dedup.py`, `materialize_release.py`, `validate_release.py`, `build_token_waterfall.py`, `publish_release.py`, `finalization_io.py`.
Structural: `structural_span_production.py`, `structural_token_loss.py`, `structural_classifier_selection.py`, `check_source_route.py`, `validate_detector_run.py`, `validate_parity_receipt.py`, `write_detector_build_receipt.py`, `validate_detector_build_receipt.py`.
Quality lane: `profile_dataset_quality_rust.py`, `export_dataset_review_samples.py`, `build_dataset_review_site.py`, `build_dataset_review_presentation.py`, `fetch_public_dataset_review_samples.py` (`72026703`, `33653540`, `84b6ab63`).

### v3 — the ordered lane (2026-07-13, never run)

`agent1_v3_pipeline.py` is a *contract orchestrator*, not a runner: it fixes the order `10-normalize → 20-lineage → 30-review-packet → 35-quality-review-evidence → 40-admission → 50-dedup → 55-greekmmlu-freeze → 60-decontamination → 65-anonymization-sanitization → 70-prestructural-freeze → 75-structural-detection-audit → 78-structural-apply → 80-final-validation`, guards against reuse of the v2 ordering, and delegates all writes to the stdlib-only `agent1_v3_contract.py`. Around it: `agent1_v3_review*.py`, `agent1_v3_admission.py`, `agent1_v3_dedup.py`, `agent1_v3_decontaminate.py`, `agent1_v3_anonymize.py`, `agent1_v3_anonymization_ledger_closure.py`, `agent1_v3_postmask_duplicate_report.py`, `agent1_v3_structural_gate.py`, `agent1_v3_structural_child.py`, `agent1_v3_transformation_waterfall.py`, `agent1_v3_release.py`, `validate_agent1_v3_candidate_roster.py` (`3a887c36`, `528497f3`, `97506ce1`).

### v4 — raw review before canonicalisation (2026-07-13 → 07-15, ran)

`agent1_v4_raw_review.py` samples one immutable file per selected logical document straight from acquired Parquet and HMAC-binds every request. `export_agent1_v4_raw_review_packet.py`, `run_agent1_v4_terra_reviews.py`, `validate_agent1_v4_terra_responses.py`, `validate_agent1_v4_human_decisions.py`, `freeze_agent1_v4_review.py`, `build_agent1_v4_review_site.py`, `serve_agent1_v4_srun_bridge.py` (the Slurm loopback that let the site be viewed without moving data), `profile_agent1_v4_fields.py`, `materialize_agent1_v4_nanochat_envelope.py`, `audit_agent1_v4_vlm_repetition.py`, `prototype_agent1_v4_gfm_normalization.py`, `build_gfm_luna_review_packet.py`, `run_gfm_luna_reviews.py`, `aggregate_gfm_luna_reviews.py` (`bf81861a` → `b4ac157c`).

### v5 — the lane that built the corpus (2026-07-15 → 07-28, ran)

`agent1_v5_pipeline.py` (stage-oriented Slurm-array runner, one immutable shard + receipt per element), `agent1_v5_datatrove.py` (MinHash LSH candidates, exact 5-shingle Jaccard verification, disk-backed union-find, NanoChat rows as immutable representatives), `stage_agent1_v5_acquisition.py`, `submit_agent1_v5_eiger.py`, `publish_private_agent1_v5.py`, `publish_private_agent1_v5_metadata.py`, `audit_agent1_v5_acquisition_integrity.py`, `audit_agent1_v5_release_quality.py`, `export_agent1_v5_quality_tail_samples.py` (`c144116c`, `cad947b4`).
Acceleration (2026-07-18 → 07-20): `agent1_v5_dedup_acceleration.py`, `agent1_v5_pair_capacity_canary.py`, `agent1_v5_signature_takeover.py`, `build_signature_benchmark_plan.py`, `finalize_signature_sentinel_cutover.py`, plus the self-chaining shell helpers `run_signature_task_chain.sh`, `run_signature_task_chain_guarded.sh`, `run_signature_row_group_chain.sh`, `roll_signature_row_groups.sh`, `roll_signature_row_group_batches.sh`, `advance_signature_stages.sh` — all written to keep exactly one job running and one queued under `debug-qos` (`730b6acd`, `2e9150a9`, `ee1e2743`, `44bcca9a`).

## Outcome

- The v5 family produced the shipped corpus; `publish_private_agent1_v5.py` was the last file in this directory to change (2026-07-28, `e8fbec2c` / `65a49d00`), gaining an explicit `--visibility` gate and receipt schema v2 so a public release was possible.
- The self-chaining signature scripts are the concrete record of what running production work on a `debug` partition costs: six separate helpers exist only to keep one validated successor queued without ever cancelling a running rank.
- The v2 and v3 families remain the most complete written specification of a receipt-bound corpus build in this repository, and the least exercised code in it.
