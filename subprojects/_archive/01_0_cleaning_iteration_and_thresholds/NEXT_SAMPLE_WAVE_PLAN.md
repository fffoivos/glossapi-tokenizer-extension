# Next Sample Wave — plan (drafted 2026-04-24)

Drafted by Claude-Cleaner after the post-C16 cleaner landed and the
162k-doc Phase A audit surfaced a sharper picture of what actually
alters in the corpus. Replaces `charset_run/deletion_band_500x500`
from v6 (20% cleaning-deletion split) — per user, 20% is too high a
mark to target now and the sample was also only partially produced
(500 lt + 192 ge).

## Scope

- Target the five **residual noise classes** the current counters
  still miss (see §1).
- Use **stratified per-metric sampling** per `feedback_stratified_sampling.md`
  (10 zones × proportional population + ≥5 floor, 100–200 total).
- Focus on PDF-extracted sources where the interesting artifacts live;
  keep digital-born sources as a small control slice.
- Produce BEFORE/POST-cleaner pairs so the reviewer can evaluate what
  lands in the training corpus (per `feedback_review_samples_post_cleaner_default.md`).
- **No threshold decisions land from this sample.** This is
  calibration material. Thresholds land only on user review.

## §1 — Residual noise classes to target

Source: `THRESHOLDS.yaml` open_gaps + `memory/glossapi_v2_corpus_residual_noise_20260421.md`:

| ID  | Class                                   | Example                                | Current detector? |
|-----|------------------------------------------|----------------------------------------|-------------------|
| N1  | Greek-to-Greek codepage mojibake         | `Διδαςκαλία ξζνων γλωςςών`             | only upstream greek_badness_score |
| N2  | Duplicated-letter font mojibake          | `ΚΚΛΛΗΗΡΡΟΟΝΝΟΟΜΜΙΙΚΚΗΗ`               | none |
| N3  | µ/μ swap                                 | `µικρό` (MICRO SIGN) vs `μικρό` (GREEK SMALL LETTER MU) | none |
| N4  | Base64 PDF binary blobs in MD            | 500+ char runs of `[A-Za-z0-9/+=]`     | none |
| N5  | ASCII gibberish runs                     | `asdasdfasdjfklasjdfk`                 | none |

Each of these needs a detector (see §2) + a per-metric top-N sample
slice (see §3).

## §2 — Detectors to add (Rust, under md_module / cleaning_module)

Per `feedback_rust_for_corpus_pipelines.md` + `feedback_group_cleaner_features_by_text_type.md`:

- **N1 — bigram plausibility** over Greek character pairs. Build a
  reference bigram table from a clean Greek corpus slice (wikisource
  + greek_phd after current cleaning). Score each doc by average
  log-prob of Greek-letter bigrams; outliers = codepage-swap
  suspects. Emit `greek_bigram_implausibility`.
- **N2 — duplicated-letter run length**. Rust pass: scan for
  consecutive-identical-letter runs (length ≥3) in uppercase Greek
  and lowercase Greek separately. Emit `duplicate_letter_runs` +
  `duplicate_letter_chars`.
- **N3 — µ vs μ mismatch**. Two counters: count U+00B5 MICRO SIGN
  and count U+03BC GREEK SMALL LETTER MU. Emit both; the ratio
  catches docs that use the wrong codepoint in math / unit contexts.
- **N4 — base64 blob detector**. Scan for contiguous runs of chars
  in `[A-Za-z0-9/+=]` of length ≥ K (K=200 default?) with entropy
  above a floor. Emit `base64_blob_chars`.
- **N5 — ASCII gibberish run detector**. Letter-run-length over
  very-low-bigram-probability Latin pairs (random keystrokes score
  low). Emit `ascii_gibberish_chars`.

Each detector is a separate counter per `feedback_split_counters_per_signal_type.md`.

## §3 — Sample layout (replaces v6/charset_run/deletion_band_500x500)

Target total: **120–180 docs** per wave, split across the noise
classes. Each lens produces its own top-N with stratified bands.

Directory skeleton:

```
/home/foivos/data/glossapi_work_cleaned_v7/charset_run/
  next_wave_2026_04/
    pdf_sources_top_altered/          # (already produced today: 90 docs)
    bigram_implausibility/            # N1 — top by greek_bigram_implausibility
    duplicate_letter_runs/            # N2
    micro_mu_mismatch/                # N3
    base64_blob/                      # N4
    ascii_gibberish/                  # N5
    stratified_content_chars_kept/    # for min_content_size calibration
```

Per folder:

- **Stratified picks.** 10 zones of the metric's population range;
  N/zone proportional to population with floor 5; 15–30 docs per
  lens.
- **Filename prefix = metric value** (zero-padded) per
  `feedback_metric_prefix_in_sample_filenames.md`.
- **BEFORE / POST-cleaner pair per doc** (per `feedback_review_
  samples_post_cleaner_default.md`: POST-cleaner is default, BEFORE
  alongside as context).

## §4 — Dependencies / blockers

- **N1–N5 detectors** need to land in Rust before their top-N slices
  can be cut.
- **Cleaner re-run on the corpus** (post-C16 + LaTeX default ON +
  new detectors) required to produce the scores to sample from.
  Expected wall on the 64-vCPU m3-megamem-64 at 32-vCPU cap: TBD,
  will benchmark with the new worker-scaling script first. Likely
  1–3 h.
- **`version.json` stamp** in each corpus-output dir (see KB-2 in
  `AGENT_COORDINATION.md`) so future agents can tell which cleaner
  produced which sample.

## §5 — Explicitly out-of-scope for this plan

- Threshold values. Those are the user's call, on review of the
  samples.
- Merging into the canonical 49.4M-doc corpus run. That's a
  separate paid run, needs explicit go-ahead.
- Removing any existing v6 artifacts. Keep them for diff / history.
- The `openarchives_knee_500x500` and `openarchives_top_residue_punct`
  sample shapes — those are still useful and should be re-cut
  under the new cleaner as a separate step (add to §2-CLEANER
  stack).

## §6 — Order of operations

1. Push this plan + get user's eyeball review.
2. Land N1–N5 detectors in Rust (one commit per detector; RED tests
   first per the review-response pattern).
3. Rerun `clean_and_stats_rowsharded.py` on the 168k-doc corpus with
   the new cleaner + new counters → v7 stats.
4. Cut the seven sub-samples into `v7/charset_run/next_wave_2026_04/`.
5. Hand off to user for review.
6. THEN, only then, consider threshold decisions.

Step 2 requires the cleaner-side counter additions to land on the
shared `codex/three-counter-pipeline-20260421` branch. Coordinate via
`AGENT_COORDINATION.md` §3 Q&A if Claude-MD also needs to make
md-module-touching changes in the same window.
