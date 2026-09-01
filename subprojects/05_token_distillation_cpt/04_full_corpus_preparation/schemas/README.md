# 04/schemas — the receipt contracts

> **In one line:** 42 JSON Schemas that made every handoff in the full-corpus build machine-checkable, so a stage could refuse a predecessor's output instead of trusting it.
> **Period:** 2026-07-11 → 2026-07-22. **Status:** complete; roughly half describe stages that never ran.
> **Came from / led to:** the contracts in [`../docs`](../docs) → these schemas → the validators in [`../scripts`](../scripts) and the tests in [`../tests`](../tests).

## Why this existed

The phase's central design rule was that a completed stage is a hash-bound receipt, not a directory that happens to contain files. That only works if the receipt shape is written down: `verify_staged_schemas.py` and `test_receipt_contracts.py` check the schemas themselves, and each stage validates its inputs against them before doing work.

## History and grouping

| Group | Schemas | Introduced | What it binds |
|---|---|---|---|
| Source review and admission | `source_review_response`, `source_novelty`, `source_license_adjudication`, `lineage_document_action` | 2026-07-11 → 07-12 (`43fcdde2`, `01cba0ee`) | Reviewer verdicts (`include` / `include_after_cleaning` / `quarantine` / `exclude` / `pending_adjudication`), novelty evidence, and the default-deny licence matrix |
| Dataset-quality lane | `dataset_quality_document`, `dataset_quality_summary`, `dataset_quality_site_handoff`, `glossapi_rust_quality_build_receipt` | 2026-07-12 (`ed552eba`, `72026703`) | Per-document metrics, the `review_sample` vs `full_scan` distinction with its denominators, the compact text-free site handoff, and the pinned Rust module build |
| Dataset-review site and sampling | `dataset_review_site_manifest`, `dataset_review_site_sample`, `dataset_review_presentation_handoff`, `dataset_review_sample_export_contract`, `dataset_review_sample_export_shard_checkpoint`, `dataset_review_complete_sample`, `dataset_review_complete_sample_packet_receipt`, `dataset_review_complete_sample_site_attestation`, `dataset_review_public_sample_packet`, `dataset_review_public_site_sample` | 2026-07-12, extended 2026-07-22 (`a933d984`, `afaaeff4`, `33653540`, `84b6ab63`) | Resumable per-shard export checkpoints, and the separation between *public source excerpts* and *identifier-masked pipeline review samples* |
| Structural (ToC / bibliography) | `structural_span`, `structural_spans_manifest`, `structural_raw_predictions`, `structural_token_loss`, `structural_application_decision`, `structural_finalization_request`, `structural_audit_validation`, `structural_false_deletion_review_case`, `structural_false_deletion_annotation`, `structural_manual_audit_receipt`, `academic_structural_classifier_selection`, `academic_structural_model_receipt`, `struct_rust_parity_receipt` | 2026-07-11 → 07-12 (`1d3b71f4`, `9014a705`, `074aa621`) | The full Stage 52 → 53 → 54 → 58 chain, including the 100-case manual false-deletion audit. **No run produced any of these receipts.** |
| Agent 1 v3 lane | `agent1_v3_review_response`, `agent1_v3_anonymization_semantic_false_positive_clearance` | 2026-07-13 (`3a887c36`, `528497f3`) | The ordered lane's own review and anonymisation-clearance contracts |
| Agent 1 v4 raw review | `agent1_v4_raw_review_request`, `agent1_v4_terra_review_response`, `agent1_v4_human_decision_bundle`, `agent1_v4_field_mapping`, `gfm_transformation_review_response` | 2026-07-13 → 07-15 (`bf81861a`, `051b5e63`, `b4ac157c`) | Request/response identity binding for the 348-document review, the human decision gate, source-field mapping, and the GFM validation verdicts |
| Release and publication | `full_cpt_release_manifest`, `full_cpt_release_validation`, `full_cpt_publication_receipt`, `greekmmlu_query_manifest` | 2026-07-12 (`8a9efebd`, `01cba0ee`) | The content-bound release chain and the frozen GreekMMLU query set |

## Outcome

- The receipt-first design is what made the long `debug`-partition dedup run restartable: a rank was reusable only if its receipt validated, never because its output file existed.
- Two schema families document work that was designed and tested but never executed — the whole structural group, and the release/publication group for the v2 DAG. The corpus that shipped was published through the v5 path instead (`agent1_v5_hf_publication_receipt_v2`, defined in [`../scripts/publish_private_agent1_v5.py`](../scripts/publish_private_agent1_v5.py) rather than as a file here).
- `greekmmlu_query_manifest.schema.json` outlived the rest: freezing the exact GreekMMLU query set by repository, config, revision, example ID and split became a standing requirement for every later CPT subproject.
