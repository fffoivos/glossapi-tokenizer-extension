# 07 · dataset — reuse the binaries, rebuild the validation, sanitize the corpus

> **In one line:** froze the full-8B training pools over the *existing* production-tokenizer binaries rather than re-tokenizing, then found and replaced every validation panel that leaked into training, and finally handed the corpus to the anonymization pipeline that produced the stream the completed run consumed.
> **Period:** 2026-08-05 → 2026-08-07. **Status:** completed.
> **Came from / led to:** the 8B-tokenizer retained ledgers from [`../../05_token_distillation_cpt/`](../../05_token_distillation_cpt/) → this → [`anonymization/`](anonymization/) → the packed D0 stream of run `20260808T121000Z-d0-wsd10-sanitized-successor-v12`.

## Why this existed

Re-tokenizing 63.8B Modern-Greek tokens would have cost days for no scientific gain: the binaries already existed under the exact production tokenizer. What did have to be rebuilt was (a) the pool catalogs and packing plan, (b) the neutral external Greek heldout, and — as it turned out — (c) most of the inherited validation panels, because several of them were not disjoint from training.

## History

1. **Inventory over existing binaries (08-05).** [`freeze_source_inventory.py`](freeze_source_inventory.py) reads the completed 8B-tokenizer retained ledgers in indexed-document order, validates every source manifest, freezes stable pool catalogs and excludes repeated Modern-Greek exact content. The shared packing code from subproject 06 was generalized to take the receipt's pad token (`3` here, `10` in the Mini overlay) and global batch (`1024` here, `512` there). It emits all five order schedules; this run consumes only `D0_mixed`. `7f7db910` fixed it to derive the replay content inventory in source order.
2. **Payload integrity.** [`verify_packed_payload_hashes.py`](verify_packed_payload_hashes.py) re-hashes every packed `.bin`/`.idx`/`.active` payload recorded by the 512 manifests — the check that later allowed the 330 GB packed payload to be *reused* by the successor stage instead of rebuilt.
3. **The leaking panels (08-06, `8deb1976`).** [`build_training_disjoint_validation_manifest.py`](build_training_disjoint_validation_manifest.py) audits every validation document against training and rebuilds any panel that leaks. The worst case was the inherited `old_greek` panel: **5,784 of its 5,833 documents were exact training-content matches**. [`build_clean_replay_validation.py`](build_clean_replay_validation.py) built its training-disjoint replacement from unconsumed replay documents with zero exact-content overlap against all selected training pools. [`freeze_selected_training_content.py`](freeze_selected_training_content.py) freezes the exact text hashes of every document the run selects, which is what makes that audit possible in the first place.
4. **The successor stage (08-07/08).** [`prepare_corrected_stage_overlay.py`](prepare_corrected_stage_overlay.py) and [`finalize_corrected_stage_overlay.py`](finalize_corrected_stage_overlay.py) create a small stage that reuses the training data and replaces only the validation side — the pattern that let the v45 rebind change receipt bindings without touching data bytes, document identities, the packed payload, the tokenizer or the schedule.
5. **Sanitization.** See [`anonymization/`](anonymization/) — PII masking, the `needs_ocr` exclusion, post-mask global deduplication and the training bridge.

## Outcome

- The completed run's executed data contract: 41,512,804,679 HPLT + 19,068,732,797 non-HPLT Modern-Greek tokens, 15,337,098,095 foreign replay and 766,854,905 Greek-replay tokens — **76,685,490,476 active tokens**, 18,284 updates ([`../../09_full_8b_cpt_results_analysis/DATA_AND_LIMITATIONS.md`](../../09_full_8b_cpt_results_analysis/DATA_AND_LIMITATIONS.md)).
- The successor plan proved exact task/document identity with the parent stage, so the packed payload was reused: 845 source-selection tasks, 512 packing tasks, regenerated deterministically by job `3034541`.
- All 13 validation panels are training-disjoint in the final package; the contaminated `old_greek` panel is not part of the canonical result.

## Working documents

Seven scripts here plus [`anonymization/`](anonymization/). Tests: [`../tests/test_anonymization_pipeline.py`](../tests/test_anonymization_pipeline.py) and [`../tests/test_full8b_orchestration.py`](../tests/test_full8b_orchestration.py).
