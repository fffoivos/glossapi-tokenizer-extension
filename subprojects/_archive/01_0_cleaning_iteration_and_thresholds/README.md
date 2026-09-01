# 01.0 — Cleaning iteration and thresholds

> **In one line:** four waves of a closed loop — train a tokenizer, read the noise it learned, patch the Rust cleaner, re-clean 49 M documents, retrain — which produced the cleaner generation and the rejection thresholds that C3 was trained on.
> **Period:** 2026-04-22 (first wave, per [`CURRENT_STATUS.md`](CURRENT_STATUS.md)) → 2026-05-14 (`002bddc5`, archive move); directory created 2026-04-23 (`296a7022`). **Status:** completed for the C3 shipping path; the per-line-badness successor branch was left in flight and never merged into the shipping path.
> **Came from / led to:** noise surfaced by [`../../02_1_tokenizer_experiments`](../../02_1_tokenizer_experiments/README.md) → this → [`../01_1_corpus_dedup`](../01_1_corpus_dedup/README.md) and [`../01_2_training_dataset_mix`](../01_2_training_dataset_mix/README.md), which consumed the cleaned parquets and the thresholds.

## Why this existed

The corpus is largely PDF-extracted Greek: PostScript glyph names, `GLYPH<…>` markers, font-substitution mojibake, dot-leader and table-separator runs. A BPE tokenizer trained on it spends vocabulary slots on that junk, so the tokenizer is the *measuring instrument* for corpus cleanliness. This subproject ran the loop between the two — the cleaner lives upstream in `eellak/glossAPI`'s Rust crates, but the rejection thresholds were deliberately kept here in [`THRESHOLDS.yaml`](THRESHOLDS.yaml) so iteration didn't require an upstream release each time. Numbering is retro-fitted: `01_0` was created on 2026-04-23, two weeks after `01`, `01_1` and `01_2`.

## History

### Wave 1 — cleaner refactor + charset filter (2026-04-22 → 04-23)

Established the rules the rest of the project kept: Rule A span-strip over 40 PostScript-glyph literals (`LeftmostLongest`); the Rule B coverage predicate `count ≥ 10 AND coverage ≥ 0.09`; **normalize after cleaning, not before**; four-way per-document character accounting (`content_chars_kept` / dropped by line-drop / normalization / per-char filter) emitted as separate columns; and the hard rule that upstream quality scores (`greek_badness_score`, `mojibake_badness_score`, `needs_ocr`, `filter`, …) are never overwritten, only added alongside (`296a7022`).

[`CURRENT_STATUS.md`](CURRENT_STATUS.md) also records the four mistakes this wave was correcting — cleaning the wrong corpus slice (168 k docs of `unified_corpus`, pre-dedup, missing HPLT/finewiki/finepdfs), conflating scoring with rejection inside one code path, early drivers overwriting upstream badness scores, and a POC tokenizer trained on that wrong data.

A 1,000-document deletion-band sample (500 below / 500 above 20 % deletion) was cut for user review to fix the deletion-% cutoff. **That cutoff never landed.** The sample was only partially produced (500 + 192) and on 2026-04-24 was superseded by [`NEXT_SAMPLE_WAVE_PLAN.md`](NEXT_SAMPLE_WAVE_PLAN.md), which retargeted sampling at five residual noise classes: Greek-to-Greek codepage mojibake (N1), duplicated-letter font mojibake (N2), `µ`/`μ` swap (N3), base64 blobs (N4) and ASCII gibberish (N5).

The review logs from this period are the sharpest evidence in the directory. [`reports/user_review_notes.md`](reports/user_review_notes.md) Case 1 documents a dissertation where Greek capitals were replaced by visually identical Latin ones — and every signal passed it (deletion 0.58 %, `charset_greek_ratio` 0.82, `greek_badness_score` 15.8, `mojibake_badness_score` 0.0). [`reports/v6_review_notes.md`](reports/v6_review_notes.md) and [`reports/v7_review_notes.md`](reports/v7_review_notes.md) (2026-04-24, cleaner SHA `375a48d`) log surviving noise in cleaner output under an explicit "record only, do not implement" rule.

### Wave 2 — pipeline cleanup and the first full-corpus run (2026-04-25 → 04-27)

[`CLEANER_PIPELINE_CLEANUP_PLAN_2026-04-25.md`](CLEANER_PIPELINE_CLEANUP_PLAN_2026-04-25.md) (`2f6b6b89`) is a ten-point architectural cleanup governed by four principles — prefer line-level threshold rules over per-char strip; no character belongs to more than one operation; one noise concept gets one definition serving both cleaner and counter; diagnostic counters off by default in production. What landed: parser-backed "Pilot B" as the production Phase-A default; four glyph engines collapsed to two (Rule B + R1∪R2); per-character operations reduced to STRIP and FOLD groups; a European-content policy (keep Latin-1/Latin-Ext-A/Cyrillic entirely, strip Latin-Ext-B except Romanian, strip IPA/Coptic); per-rule counters in `CleanStats`; and a silent-bug fix where the directory pipeline stripped ASCII punctuation and digits when callers passed restricted scripts. Verification: 374 cleaner unit tests + 10 noise tests, and a 100-document end-to-end run at 18.7 % chars removed, 25 docs/sec, with gzipped shards byte-identical to the in-process result.

The production run followed on 2026-04-26 ([`WAVE2_PIPELINE_RUN_2026-04-26.md`](WAVE2_PIPELINE_RUN_2026-04-26.md), `9be5193d`/`b3e92fa1`). Because the instance had no raw sources, the run re-cleaned the canonical parquets in place via a purpose-written driver ([`scripts/reclean_canonical_to_parquet.py`](scripts/reclean_canonical_to_parquet.py)) rather than `corpus_cli build`, which needed absent reeval data: **49,332,970 rows, 4.48 % chars removed, 67 minutes on 64 vCPUs.** Downstream it ran to dedup (see `../01_1_corpus_dedup`), mixes and splits, but the tokenizer stage stalled — F1 and F2 finished, **C1 was stopped by the user inside a single-core `count_segments` phase, C2 never started, `all_done.json` was never emitted.**

Two course corrections are recorded verbatim: an integration test had been skipped before the corpus-scale sweep (recovery: kill the dedup at 82 %, write a resumable orchestrator, run a real smoke first), and the plan itself had not been written down with status. Budget was ~$100–135 of instance time at $8.40/hr.

### Wave 3 — tokenizer-guided patch (2026-04-28)

[`HANDOFF_2026-04-28.md`](HANDOFF_2026-04-28.md) is the onboarding document for this state and states the methodology plainly: *"Wave 1 missed the noise families that wave-2 cleaned up. Wave 2 in turn missed the families catalogued in A–M. The loop is the methodology."* [`WAVE3_CLEANER_PATCH_PLAN_2026-04-28.md`](WAVE3_CLEANER_PATCH_PLAN_2026-04-28.md) narrowed the scope using a scan of the full F1 training split (310,019 docs / 60.8 B chars): table-separator fragments in 82.14 % of docs, dot-leader runs 62.01 %, long dash runs 51.37 %, escaped Markdown runs 8.93 %, bare `GLYPH`/glyph-name residue 735–1,566 docs. Decisions: implement run quantization as a true floor (length 4 → 3, not 5); **keep** intentional HTML comment placeholders (77.42 % of docs) and pictographs; extend the existing glyph Rule A/B machinery rather than build a glyph subsystem; **defer** mojibake repair and Cyrillic/homoglyph folding to `eellak/glossAPI` issue #99.

The wave-3 production run was first **aborted** at `2026-04-28T09:20Z` because HPLT rows carried null `greek_badness_score` that the split exporter still admitted ([`PRE_RESTART_CODE_AND_ANALYSIS_2026-04-28.md`](PRE_RESTART_CODE_AND_ANALYSIS_2026-04-28.md)). After the strict-filter and reclean fixes, `production_strict_v2` completed at `2026-04-28T17:47:28Z` ([`WAVE3_PRODUCTION_PROGRESS_2026-04-28.md`](WAVE3_PRODUCTION_PROGRESS_2026-04-28.md)): 274 tasks / 0 errors / 49,332,970 rows, 48,728,774 HPLT rows scored with 0 missing, chars 256.30 B → 244.51 B (≈4.6 % removed). The tokenizer review ([`WAVE3_TOKENIZER_REVIEW_2026-04-28.md`](WAVE3_TOKENIZER_REVIEW_2026-04-28.md)) measured the payoff: F1 (GlossAPI, 50 k vocab) 34,442 Greek tokens with 12 glyph/PDF hits and 8 mojibake markers; F2 (HPLT) 44,691 Greek tokens with 0 glyph, 0 PostScript, 1 mojibake; C1's 25,600 added units 25,163 Greek with 9 glyph hits and 0 Cyrillic. Three follow-up issues were opened rather than rerunning (glossAPI #99, #100; tokenizer-extension #1).

### Wave 4 — glyph/PostScript endgame (2026-04-29)

[`WAVE4_GLYPH_POSTSCRIPT_PLAN_AND_CHANGES_2026-04-29.md`](WAVE4_GLYPH_POSTSCRIPT_PLAN_AND_CHANGES_2026-04-29.md) (`edb98d6b`) traced the last two residue families to specific causes found with a purpose-built context-blind Rust scanner ([`rust/glyphscan`](rust/glyphscan), analysed in [`GLYPH_POSTSCRIPT_MATCHER_ANALYSIS_2026-04-29.md`](GLYPH_POSTSCRIPT_MATCHER_ANALYSIS_2026-04-29.md)): `GLYPHGLYPH…` came from dense failed display-math lines in `greek_phd` that the cleaner deliberately skipped inside `$$…$$`, and `/hyphenminus` survived because the URL guard protected dotted slash tokens like `4.600/hyphenminus5.600`. Fixes: replacement-aware glyph span rewriting (glued `/hyphenminus` → `-`), narrowed URL protection, and a display-math residue check that drops contaminated math lines while preserving clean ones. Full suite: 385 passed, 0 failed, 3 ignored.

The rerun also had to absorb the exporter cross-join correction (see `../01_2_training_dataset_mix`). Final reclean: 272 tasks, 270 outputs, 49,332,081 rows, chars 256.23 B → 244.41 B, with **291,107 rows still missing Greek badness**. Residue in C1's added-token slice: 0 uppercase `GLYPH`, 0 `/hyphenminus`, 0 structured glyph markers, 31 broad PostScript-name false positives (`/CP`, `/CX`, `/GE`, `/pi`), 5 mojibake markers. The explicit decision was to stop: *"This should become a future calibration issue rather than another production restart."* Two more issues were filed (tokenizer-extension #2, #3).

### After wave 4 (2026-05-04 → 05-14)

[`PER_LINE_CLEANER_BRANCH_PLAN_2026-05-04.md`](PER_LINE_CLEANER_BRANCH_PLAN_2026-05-04.md) opened `cleaner/per-line-badness-20260504` — math-symbol and Mathematical-Alphanumeric Greek folds plus a Rust port of a per-line Greek-badness scorer — explicitly decoupled from C3 and never merged into the shipping path. [`THRESHOLD_STUDY_PROTOCOL.md`](THRESHOLD_STUDY_PROTOCOL.md) distilled waves 1–4 (Apr 22 – Apr 29) into a reusable doc-level/line-level calibration protocol. C3 converged on 2026-05-11 and the directory was archived on 2026-05-14 (`002bddc5`), which also added the "Historical reference" banner to every doc here.

## Outcome

- **Shipped**: the cleaner generation and quality columns that C3's training input was produced with, plus [`THRESHOLDS.yaml`](THRESHOLDS.yaml) `version: wave_2_20260426` whose `standard_exclusions` — `greek_badness_score > 60`, `mojibake_badness_score > 0.1`, `charset_greek_ratio < 0.5`, empty-after-clean — became the exporter's defaults and are exactly the gates listed for C3 in [`../../../docs/C3_TRAINING_DATASETS.md`](../../../docs/C3_TRAINING_DATASETS.md).
- **Measured result of four waves**: glyph/PostScript and `GLYPH` residue went from a visible token family in F1 to zero in C1's added-token slice; HPLT-side residue was already near zero by wave 3.
- **Explicitly deferred, never done**: mojibake repair and Cyrillic/Latin homoglyph folding (glossAPI #99); line-level `$$ $$` and `Page N` residue; the N1–N5 detectors from [`NEXT_SAMPLE_WAVE_PLAN.md`](NEXT_SAMPLE_WAVE_PLAN.md); the deletion-% and `content_chars_kept` cutoffs that wave 1 opened.
- **Naming caveat**: C3's cleaner is recorded as the "wave-2 broad cleaner" on branch `codex/cleaner-audit-counters-20260506` ([`../../../docs/C3_TRAINING_DATASETS.md`](../../../docs/C3_TRAINING_DATASETS.md)), which is a later branch than this directory's April wave-2 branch `cleanup/cleaner-pipeline-20260425`. The two share a label, not a commit.

## Where things are

| Artifact | Role |
|---|---|
| [`THRESHOLDS.yaml`](THRESHOLDS.yaml) | The rejection config (`wave_2_20260426`): standard exclusions, upstream signals, charset filter, deletion threshold, min content size, open gaps. |
| [`scripts/reclean_canonical_to_parquet.py`](scripts/reclean_canonical_to_parquet.py) | The production re-clean driver used by waves 2–4 and by C3's build. |
| [`rust/glyphscan/`](rust/glyphscan) | The context-blind glyph/PostScript scanner written for wave 4. |
| [`HANDOFF_2026-04-28.md`](HANDOFF_2026-04-28.md) | The fullest single description of the loop, the instance layout and the branch map. |
| [`THRESHOLD_STUDY_PROTOCOL.md`](THRESHOLD_STUDY_PROTOCOL.md) | The reusable calibration protocol distilled from all four waves. |
| [`../../../legacy/corpus_clean_normalization/`](../../../legacy/corpus_clean_normalization/) | The companion rule-discovery pipeline (`NORMALIZATION_DESIGN_20260420.md` holds the A–M rule list the wave-3 plan narrows). |

## Working documents

Historical, kept for traceability. Every file here carries an archive banner added in `002bddc5`.

- **Plans:** [`CLEANER_PIPELINE_CLEANUP_PLAN_2026-04-25.md`](CLEANER_PIPELINE_CLEANUP_PLAN_2026-04-25.md), [`CORPUS_CLEAN_WAVE2_PLAN.md`](CORPUS_CLEAN_WAVE2_PLAN.md), [`WAVE3_CLEANER_PATCH_PLAN_2026-04-28.md`](WAVE3_CLEANER_PATCH_PLAN_2026-04-28.md), [`WAVE4_GLYPH_POSTSCRIPT_PLAN_AND_CHANGES_2026-04-29.md`](WAVE4_GLYPH_POSTSCRIPT_PLAN_AND_CHANGES_2026-04-29.md), [`NEXT_SAMPLE_WAVE_PLAN.md`](NEXT_SAMPLE_WAVE_PLAN.md), [`PER_LINE_CLEANER_BRANCH_PLAN_2026-05-04.md`](PER_LINE_CLEANER_BRANCH_PLAN_2026-05-04.md).
- **Status snapshots / handoffs:** [`CURRENT_STATUS.md`](CURRENT_STATUS.md) (three stacked states: a 2026-04-28 wave-3 override, the wave-2 landing, and a 2026-04-23 wave-1 block), [`HANDOFF_2026-04-28.md`](HANDOFF_2026-04-28.md), [`PRE_RESTART_CODE_AND_ANALYSIS_2026-04-28.md`](PRE_RESTART_CODE_AND_ANALYSIS_2026-04-28.md), [`TODO.md`](TODO.md).
- **Run logs:** [`WAVE2_PIPELINE_RUN_2026-04-26.md`](WAVE2_PIPELINE_RUN_2026-04-26.md), [`WAVE3_PRODUCTION_PROGRESS_2026-04-28.md`](WAVE3_PRODUCTION_PROGRESS_2026-04-28.md).
- **Reviews:** [`WAVE3_TOKENIZER_REVIEW_2026-04-28.md`](WAVE3_TOKENIZER_REVIEW_2026-04-28.md), [`GLYPH_POSTSCRIPT_MATCHER_ANALYSIS_2026-04-29.md`](GLYPH_POSTSCRIPT_MATCHER_ANALYSIS_2026-04-29.md), [`reports/user_review_notes.md`](reports/user_review_notes.md), [`reports/v6_review_notes.md`](reports/v6_review_notes.md), [`reports/v7_review_notes.md`](reports/v7_review_notes.md).
- **Script inventory:** [`scripts/README.md`](scripts/README.md) — describes the iteration-loop scripts, most of which live upstream in `eellak/glossAPI/cleaning_scripts/` and are not present in this directory.
