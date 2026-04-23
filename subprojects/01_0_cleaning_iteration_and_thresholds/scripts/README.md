# scripts/

Operator scripts for the cleaner-iteration loop. The canonical source
of these lives in `eellak/glossAPI/cleaning_scripts/` on branch
`codex/three-counter-pipeline-20260421`. This folder is a local
reference; when a script is frozen and the cleaner version is pinned,
relevant scripts get copied here verbatim or replaced by a thin
project-specific wrapper.

## Active iteration-loop scripts

### Cleaner driver
- `clean_and_stats_rowsharded.py` — the row-sharded cleaning + stats
  emission driver. Consumes source parquets, writes per-doc stats.jsonl
  + cleaned text gz shards. Uses `glossapi_rs_cleaner.clean_text_with_stats`
  + `glossapi_rs_cleaner.analyze_charset` +
  `glossapi_rs_noise.match_token_category_debug_text(write_files=False)`.

### Borderline-review samplers
- `pull_deletion_band_samples.py` — uniform-random N pre- and
  post-20%-deletion; filename prefix = zero-padded deletion-%.
- `sample_charset_threshold_bands.py` — docs near each charset cutoff
  (for threshold validation).
- `sample_broken_text_candidates.py` — stratified over 4 metric axes,
  doc-floor per dataset.

### Gemini + sub-agent reviewers
- `gemini_broken_text_reviewer.py` — multi-axis schema, no master
  binary label.
- `analyze_band_verdicts.py` — cross-band yes-rate comparison.
- `organize_verdicts_for_inspection.py` — reshuffles verdicts into
  band × verdict-class browsable tree.
- `analyze_broken_text_verdicts.py` — per-metric zone yes-rate
  histograms.
- `analyze_quality_vs_deletions.py` — joins cleaner stats with upstream
  scores, bucketed medians + Spearman correlations.
- `analyze_charset_filter_impact.py` — drop-count breakdown per
  dataset × rule.
- `analyze_cleaning_distributions.py` — corpus-wide shape plots
  (needs matplotlib).

### Upstream-score joiner
(helper inline in `pull_deletion_band_samples.py` — reads an
`upstream_scores.jsonl` extracted from source parquets).

## Convention

- Filenames in `reports/` sample trees MUST prefix the metric value
  zero-padded so `ls` orders ascending. See
  `feedback_metric_prefix_in_sample_filenames.md`.
- Driver scripts should support `--workers` with sensible default (e.g.
  `min(cpu_count(), 56)`), and respect `write_files=False` on the
  noise matcher.
- All Gemini API usage should log total cost estimate before the run
  and prompt for explicit confirmation.
