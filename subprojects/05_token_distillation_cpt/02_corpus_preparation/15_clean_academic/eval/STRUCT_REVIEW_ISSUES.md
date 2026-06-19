# v2 structure-annotation — review issues log

Observations from reviewing the 10-doc experiment in the struct-viz. Recorded (not silently auto-fixed)
per the project's observations-vs-requests practice; resolution noted inline.

## I1 — ΣΥΝΤΟΜΟΓΡΑΦΙΕΣ (abbreviations list) bucketed with the ToC  (doc d59e5d, tab 3)
**Observed:** the ToC span ran 46–223, swallowing the `## ΣΥΝΤΟΜΟΓΡΑΦΙΕΣ` abbreviations table (157–223),
a 2-column glossary (`ALT | alanine aminotransferase`, …). The real Περιεχόμενα ToC is 46–155.

**Two distinct causes:**
- **(a) merge bug — FIXED.** Opus had marked them as *two* separate `table_of_contents` sections
  (46–155 and 157–223); `merge_chunks.py` wrongly fused adjacent same-kind spans even within a single
  whole-doc annotation. Fixed to only stitch sections split across *chunk* boundaries. d59e5d now shows
  them separate.
- **(b) taxonomy — RESOLVED (user: leave as main text).** A list of abbreviations / glossary / list of
  figures/tables is **not marked** — it falls into derived main text and is kept. Only the chapter/section
  *table of contents* is `table_of_contents`. Spec §4/§6/§9 + the workflow prompt updated. `d59e5d`
  re-annotated under the corrected prompt to validate (its 157–223 abbreviations span should disappear).

## I2 — badly-extracted docs should be filtered before annotation  (greek_badness_score > 60)
**Observed:** some annotated docs are badly extracted (OCR garble / low Greek), wasting annotation on text
that won't be in the clean corpus anyway.
**Resolution — DONE (real gate).** `badness_filter.py` drops docs with `greek_badness_score > 60` at
unit-build time. greek_phd carries the column; **openarchives + kallipos lack it** and are scored on the fly
with the same Rust scorer (`glossapi_rs_noise.score_markdown_file_detailed`), verified consistent with the
column (ef7c29: 90.9 col vs 86 computed). Of the 10 experiment docs, **2 excluded**: ef7c29 (86, a 1.8%-Greek
chemistry thesis) and openarchives 6813ba (84). Viz now applies the gate + shows each doc's badness.
Note: the metric gates *extraction garble*, not Greek-ness — 7f58 (badness 26, only 3.8% Greek) passes; a
separate Greek-fraction floor would be a different decision (not added unprompted).
