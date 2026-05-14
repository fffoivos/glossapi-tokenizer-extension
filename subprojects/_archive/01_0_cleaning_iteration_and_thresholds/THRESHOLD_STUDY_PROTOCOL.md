# Threshold Study Protocol

A reusable protocol for turning a candidate signal into a cleaner rule (span-strip + line-drop) with calibrated thresholds. Distilled from waves 1-4 (Apr 22 - Apr 29 2026) so future studies don't re-derive it.

## When to run this protocol

Run when:

- A new noise pattern has been identified (sub-agent or manual review of corpus samples).
- An existing detector would benefit from per-line coverage analysis (e.g., Rule B's `count ≥ 10 AND coverage ≥ 0.09` was set this way).
- A doc-level signal (e.g. `greek_badness_score`) needs to be turned into a per-line rule.

Do NOT run for:

- Corpus-level rejection thresholds (those live in `THRESHOLDS.yaml`, set by user only after sample review).
- Single-pattern bug fixes that don't need stratification.

## Hard rules carried in

- **No threshold lands without explicit user review.** The protocol produces calibration material; thresholds are user-set after review (memory: `feedback_no_threshold_rules_unprompted.md`).
- **Stratified per-metric sampling.** 10 zones × proportional population + ≥5 floor, 100-200 docs total per metric (memory: `feedback_stratified_sampling.md`).
- **Filename prefix = metric value** (zero-padded) so `ls` order = metric order (memory: `feedback_metric_prefix_in_sample_filenames.md`).
- **POST-cleaner output by default.** Pre-cleaner only when explicitly needed for diagnostic context (memory: `feedback_review_samples_post_cleaner_default.md`).
- **Full doc body, no truncation** in review samples (memory: `feedback_review_samples_full_no_truncation.md`).
- **Multi-axis Gemini prompts.** Orthogonal axes (quantity / location / comprehension / per-symptom binaries), never master accept/reject (memory: `feedback_multi_axis_gemini_prompts.md`).
- **Continuous passage with inline `<match kind="…">…</match>` wrapping**, never split before/match/after blocks (memory: `feedback_data_inspection_format.md`).
- **All deterministic per-doc work in Rust.** Python is a thin driver only (memory: `feedback_rust_for_corpus_pipelines.md`).
- **No heavy compute on laptop.** Corpus-scale work goes to gcloud; sample-pulling and per-line audit are OK locally if `iter_batches` is used (memory: `feedback_no_heavy_compute_on_laptop.md`).
- **Gemini API calls and full-corpus runs require explicit user go-ahead.** Always.

## Protocol shape

Two granularities — pick one explicitly per study, do not mix.

### Granularity A: Doc-level study (current canonical shape)

Stratify on a per-doc score. Output one sample doc per filename. Used for: deletion-% cutoffs, doc-level rejection thresholds (e.g., `greek_badness_score > 60`).

### Granularity B: Line-level study (newer; for span-strip + line-drop rules)

Stratify on per-line `(match_count, coverage_over_nonws)`. Output one sample line + its surrounding context per filename. Used for: turning a candidate detector into a Rule B-shaped gate.

The user's framing for the upcoming `greek_badness` + `mojibake_badness` study is **Granularity B** — per-line `(count, coverage%)` zones.

## Steps (Granularity B in detail; A is a degenerate case)

### Step 0 — Identify the candidate detector

A line-level detector emits, per line, an integer `match_count` and a `coverage_ratio = match_chars / non_ws_chars`. Examples:

- Rule A literal-match count + coverage of PostScript-name literals.
- Rule B regex-match count + coverage of glyph/font residue regex.
- (New) `greek_internal_mojibake_match_count + coverage` — per-line characters that participate in known Greek-internal codepage swap patterns.
- (New) `mojibake_byte_failure_match_count + coverage` — per-line characters from Latin-1 Supp / IPA / Specials / Latin-Ext-B (the existing `moji_residue_ratio` denominator), shifted to per-line granularity.

If the detector doesn't yet exist as Rust per-line code, building it is **Step 0a** — a separate concern from this protocol.

### Step 1 — Run the detector on a representative input

Input: a recent post-cleaner parquet with the detector's per-line output emitted as a sidecar (or a re-cleaning run with the new detector wired in). Volume: 1-5k docs is enough for stratification.

For doc-level signals being made line-level for the first time, the input may need to be a fresh re-cleaning run on a small parquet (50-100 docs is enough to characterize the per-line pattern distribution).

### Step 2 — Build the (count, coverage) zone matrix

Per-line, every line gets two coordinates:

- **count zone**: 10 buckets of `match_count` over the population's range. Boundaries by approximate-decile quantiles (not equal-width); use rounded thresholds (e.g., 0, 1, 2, 5, 10, 20, 50, 100, 200, ≥500) so adjacent zones are easy to read.
- **coverage zone**: 10 buckets of `coverage_ratio` at fixed boundaries `[0%, 10%, 20%, …, 90%, 100%]`.

Result: a 10×10 matrix. Most cells will be sparse; the diagonal + boundary cells carry signal.

### Step 3 — Sample from the zone matrix

Per cell: pick `min(5, available)` lines for the floor, then proportionally up to ~200 lines total. Cells with zero lines stay zero.

For each sampled line, output:

- a small file (one per line)
- filename: `c{count_zone:02}_v{coverage_zone:02}_{count:05d}_{coverage_pct:03d}_{stem}.txt` so `ls` orders by zone, then count, then coverage.
- file body: continuous passage with the line-of-interest wrapped inline as `<match kind="$detector_name">…</match>`, ±N lines of context (default ±10).

### Step 4 — Review

Two paths, run in parallel:

**Path A — Subagent triage** (cheap, parallel, produces structured output):

- Spawn N subagents, each gets one zone-cell's samples.
- Each agent extracts: per-symptom binaries (e.g., `is_legit_greek_text`, `is_known_mojibake_class`, `is_pdf_extraction_artifact`), suggested handling per zone (`leave_alone` / `span_strip` / `line_drop`), brief rationale.
- Agents do NOT propose threshold values — they characterize zone behavior.

**Path B — User review** (slow, expensive, authoritative):

- User reads the same sample files, in `ls` order so the count/coverage progression is visible.
- User decides: which zone is the leave/strip/drop boundary? Per-rule, or unified gate?

Both paths feed into Step 5.

### Step 5 — Threshold proposal

Synthesize the zone reviews into a candidate `count ≥ X AND coverage ≥ Y` predicate (or a more complex shape if needed). Document the predicate in a markdown plan, NOT in `THRESHOLDS.yaml` directly — that file only changes on user sign-off.

### Step 6 — Validation

Run the candidate predicate against:

- A held-out portion of the original sample (the lines NOT used for calibration).
- A small `bench` corpus (50-100 docs) to confirm it doesn't false-fire on legitimate content.

Report: precision (positive predictions that match the intent) + recall (intended positives that the predicate catches). Both should be high; the cleanup wave's Rule B targeted P=96.3%, R=60.4% on the 2026-04-22 ground truth — that level is the precedent.

### Step 7 — Land or revise

- If P ≥ ~95% and R is acceptable → user signs off → predicate lands as a Rule in the cleaner crate. Tests added per the wave's TDD pattern. New per-rule counter added to `CleanStats`.
- If P or R is unacceptable → revise the detector or the predicate; loop back to Step 4.

## Tooling map (existing scripts on dev)

These are already in `cleaning_scripts/` after the cleaner integration:

| Script | What it does | Granularity |
|---|---|---|
| `pull_deletion_band_samples.py` | Stratified per-doc by `pct_chars_removed_non_empty` + bucket-based output | Doc-level |
| `sample_charset_threshold_bands.py` | Stratified per-doc by charset ratios | Doc-level |
| `sample_char_strip_bands.py` | Bands by `char_strip_ratio` | Doc-level |
| `calibrate_counter_thresholds.py` | Threshold calibration over the three counters | Doc-level |
| `analyze_band_verdicts.py` | Aggregates Gemini verdicts per band | Both |

**Gap**: no per-line `(count, coverage)` zone sampler exists. Building one as `pull_per_line_zone_samples.py` is the natural extension; signature mirrors `pull_deletion_band_samples.py` but reads a per-line stats sidecar instead of a per-doc one.

## Worked example (planned for the upcoming study)

Study: turn `greek_badness_score` and `mojibake_badness_score` (currently doc-level rejection thresholds at `>60` and `>0.1` respectively) into per-line span-strip + line-drop rules.

Approach:

1. (Step 0) Identify per-line patterns that contribute to high doc-level scores. The existing `reports/user_review_notes.md` Cases 1-13 already enumerated these — Case 1 in particular characterizes Greek-Latin confusable-letter mojibake (the dominant `greek_badness_score` driver) and LaTeX-escape-density mojibake.
2. (Step 0a) Implement two new per-line detectors in Rust:
   - `greek_internal_mojibake` — the confusable-Latin-in-Greek-context detector (Case 1, Detector 3) plus the LaTeX-escape-density detector (Case 1, Detector 1).
   - `mojibake_byte_failure` — per-line shift of the existing `moji_residue_ratio` family.
3. (Step 1) Run the detectors on a 100-doc parquet stratified by existing `greek_badness_score` + `mojibake_badness_score`.
4. (Steps 2-3) Build the 10×10 `(count, coverage)` zone matrix; sample 100-200 lines.
5. (Step 4) Subagent triage in parallel + user review.
6. (Steps 5-7) Threshold proposal + validation + landing.

**Step 0a is a real prerequisite.** Until the per-line detector exists in Rust, the protocol can't run as Granularity B. For an immediate first pass, run Granularity A first: stratify the existing 100-doc bench by doc-level `greek_badness_score` and `mojibake_badness_score`, and characterize what each zone's docs LOOK LIKE per-line. That informs detector design before the Rust work begins.

## Filing convention

Per study, create a folder under `subprojects/01_0_cleaning_iteration_and_thresholds/runs/`:

- name: `{study_slug}_{YYYYMMDDtHHMM}/`
- contents: `samples/` (the zone files), `prompts/` (any Gemini prompts used), `reviews/` (subagent outputs + Gemini outputs), `verdicts.md` (synthesis), `next_steps.md` (proposed predicate + open questions).

## See also

- `NEXT_SAMPLE_WAVE_PLAN.md` — the wave-2 example of this protocol applied (still useful even though wave 2 has shipped).
- `THRESHOLDS.yaml` — current canonical threshold policy. This protocol's outputs feed it; it is never edited without user sign-off.
- `reports/user_review_notes.md` — gold-standard manual review of Cases 1-13. Read before designing any new detector.
- `WAVE2_PIPELINE_RUN_2026-04-26.md`, `WAVE3_PRODUCTION_PROGRESS_2026-04-28.md`, `WAVE4_GLYPH_POSTSCRIPT_PLAN_AND_CHANGES_2026-04-29.md` — concrete wave records showing the protocol in action.
