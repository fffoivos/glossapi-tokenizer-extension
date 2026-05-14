# Three-counter page-noise pipeline — end-to-end spec

**Purpose**: build the `drop_low_salvage_pages` threshold calibration
loop for the full corpus, using three independent noise counters
tracked per page.

Upstream specs:
- `font_name_literal_spec.md` → Counter 1 (`page_font_marker_count`).
- (existing `glyph_font_like` category) → Counter 2
  (`page_glyph_marker_count`).
- `script_residue_restricted_spec.md` → Counter 3
  (`page_script_residue_count`).

Review flow: `feedback_stratified_sampling.md` +
`feedback_data_inspection_format.md` + two locked prompt schemas.

## Stage 1 — matcher extension (local laptop work, no compute spend)

1. Add `font_name_literal` category spec to the matcher
   (`rust/glossapi_rs_noise/` — literal set + PDF-subset regex).
2. Add `script_residue_restricted` category (Ext-A + Ext-B per-char
   range match + script-salad bigram regex).
3. Tests: inline Rust tests confirming positive + negative cases,
   including code-fence guard.
4. Python binding: ensure `PAGE_CATEGORY_MATCH_COUNTS` surfaces the
   two new category names via
   `match_token_category_debug_text_internal`.

Estimated time: 3-4 hours.

## Stage 2 — full-corpus matcher run (GCP compute)

**Target**: run the matcher across all 20 source parquets in
`/home/foivos/data/glossapi_work/unified_corpus/data/` (49.7M rows,
~8 GB total), emitting per-page metrics.

**Compute posture**:
- Instance: `apertus-greek-tokenizer-20260408t160000z` in
  `europe-west4-b`, `m3-megamem-64` (64 vCPUs, ~$7/hr).
- Parallelism: `n_threads = 64`, one source parquet per rayon scope.
- Expected wall-clock: 15-30 min (full-corpus matcher runs have
  finished in ~20 min at this parallelism in past waves).
- Expected cost: $2-4 instance runtime, stopped immediately after.

**Output**: per-source `page_metrics.jsonl` files with columns
including `page_font_marker_count`, `page_glyph_marker_count`,
`page_script_residue_count`, `page_char_count`, `page_match_count`,
`source_path`, `page_number`.

**Hygiene**:
- Kick off with `nohup` / `tmux` so SSH disconnect doesn't kill the
  job.
- Post-run: `gcloud compute instances stop ...` BEFORE returning the
  result to the user (cost discipline rule).

## Stage 3 — stratified sampling (local laptop)

Per the `feedback_stratified_sampling.md` rule:

For **each** of the three counters (done independently — they have
different distributions):

1. Compute `min` and `max` across the full corpus population.
2. Slice `[min, max]` into **10 zones** (log-spaced if the metric
   spans multiple orders of magnitude, else linear).
3. Count pages per zone.
4. Allocate ~50 cases per counter (so ~150 total across three),
   proportional to zone population, floor 5 / zone.
5. Draw with fixed seed.

**Sample folder structure**:
```
samples/page_calibration_2026XXXX/
  font_marker/
    font_marker_00005__<source>_p<page>.md
    font_marker_00042__<source>_p<page>.md
    ...
  glyph_marker/
    ...
  script_residue/
    ...
  aggregate.json       # zone boundaries, per-zone counts, seed
  README.md            # rationale, population stats
```

Filenames prefixed with the metric value (per
`feedback_metric_prefix_in_sample_filenames.md`).

## Stage 4 — Gemini review wave (Task type 2 — page-level noise)

For each case in stages 3's sample folders:

**Prompt body** (2 blocks only — metadata client-side):

```
[CONTEXT]
<±100 lines / ±10k chars of page text, no inline tags>

[QUESTIONS]
1. keep_or_drop      (keep / drop / uncertain)
2. noise_character   (clean / mojibake / glyph_corruption /
                      script_salad / garbled_text_other /
                      mixed / unclear)
3. dominant_signal   (font_names / glyph_tags / script_residue /
                      other_unknown / none / multiple)
4. short_reason      (≤ 40 words, free text)
```

**Cost estimate**: 150 cases × ~4000 tokens input × gemini-2.5-flash
price ≈ $0.50–1.00.

**Output join**: Gemini response is indexed by prompt order; our
per-case record (containing match_id, source_path, counter values,
zone) is joined by enumeration.

## Stage 5 — threshold calibration

For each counter, plot:
- x-axis: counter value bins (the 10 zones).
- y-axis: `drop_rate` = fraction of cases in the zone where Gemini
  answered `keep_or_drop = drop`.

Threshold = smallest counter value where `drop_rate >= 0.80` (or
whatever we settle on — current page-noise evidence was
`match_count ≥ 160 → 93%`).

If `garbled_text_other` or `other_unknown` light up in `>= 10%` of a
zone, that signals a missing counter — read the `short_reason`
free-text for patterns and propose a new counter for the next wave.

Write three thresholds into `drop_low_salvage_pages` config:
```rust
pub struct PageSalvageConfig {
    pub font_marker_threshold: u32,
    pub glyph_marker_threshold: u32,
    pub script_residue_threshold: u32,
    // Drop page if ANY counter >= its threshold.
}
```

## Stage 6 — Feat-A font-name anchor review (separate, Task type 1)

Lower priority but runnable in parallel if budget allows: sample
~30-50 anchor hits of **bare-form** font names (`Palatino`, `Linotype`,
`Courier`, etc. — the bare forms we flagged as review-gated in
`font_name_literal_spec.md`), using Task 1 anchor-review prompt:

```
[CONTEXT]
<±100 line passage with the bare font name wrapped inline as
 <match kind="font_name">Palatino</match>>

[QUESTIONS]
1. anchor_flags_noise  (yes / no / uncertain)
2. sufficient_alone    (yes / no)
3. short_reason        (≤ 40 words)
```

Verdicts feed back into `font_name_literal_spec.md`'s candidate list —
which bare-form literals to promote to the direct-strip set.

## Sequence

Stages 1 → 2 → 3 → 4 → 5 is the critical path.
Stage 6 can be bundled with Stage 4 to save a Gemini-batch round-trip.

Go/no-go gate before Stage 2 compute spend: **user greenlight**.
