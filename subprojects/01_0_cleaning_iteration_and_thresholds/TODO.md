# TODO — 01_0 Cleaning Iteration

## Wave 1 — closing

- [ ] User review of `deletion_band_500x500` sample → lock
      `deletion_threshold.pct_chars_removed_non_empty` in THRESHOLDS.yaml
- [ ] User review of content-size distribution → lock
      `min_content_size.content_chars_kept`
- [ ] Open PR for eellak/glossAPI branch `codex/three-counter-pipeline-20260421`
- [ ] Merge upstream noise-crate `write_files` speedup
- [ ] Pin cleaner version tag in THRESHOLDS.yaml / a `pinned_versions.yaml`

## Wave 2 — proposed

- [ ] Implement Greek-to-Greek codepage mojibake detector (bigram
      plausibility over Greek character pairs, against a reference
      Greek corpus). Gap: current ratios miss "Διδαςκαλία" type
      corruption entirely; only upstream `greek_badness_score` catches it.
- [ ] Implement duplicated-letter font-mojibake detector
      (e.g. "ΚΚΛΛΗΗΡΡ…"). Run-length heuristic over consecutive
      identical letters + letter-to-letter char-bigram rarity.
- [ ] Run the full canonical corpus (49.4 M docs post-dedup)
      — requires upstream speedup merged + user go-ahead
- [ ] Validate threshold distributions at full-corpus scale — sub-
      populations by source (HPLT / finewiki / openarchives) may
      require per-dataset thresholds
- [ ] Port scripts from upstream `eellak/glossAPI/cleaning_scripts/`
      into `scripts/` here, reference them in the pipeline

## Blocked / parked

- Corpus-level dedup of the cleaner-output re-run → handled by
  subproject `01_1_corpus_dedup`; only required for final tokenizer
  validation
- Training-mix regeneration after cleaner pin → handled by
  `01_2_training_dataset_mix`
- Tokenizer retrain at 4-arm matrix → handled by
  `02_1_tokenizer_experiments`

## Notes

- All full-corpus runs and Gemini calls require explicit user
  confirmation (paid compute).
- Sample-review filenames prefix metric value, zero-padded, so `ls`
  orders ascending. Per `feedback_metric_prefix_in_sample_filenames.md`.
