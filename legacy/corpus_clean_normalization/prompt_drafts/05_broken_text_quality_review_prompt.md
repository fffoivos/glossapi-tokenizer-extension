# Broken-text quality review — Gemini prompt (2026-04-22)

**Purpose**: post-cleaning per-doc quality labelling. Intentionally
multi-axis and NO master binary — we collect orthogonal signals and
defer the rejection policy until we've seen the label-vs-stats
distribution (per user guidance 2026-04-22).

## Design intent

Each diagnostic question is engineered to correlate with one
quantifiable stat emitted by `clean_and_stats_rowsharded.py`
(`CleanStats` four-way char split + line counts + non-empty
measurements). The calibration workflow will, per question:

1. Bucket docs by the correlated stat (10 zones per
   `feedback_stratified_sampling.md`).
2. Plot the per-zone `yes` rate.
3. Set cutoffs where the shape shows a natural transition.

Not asking a master `suitable_for_pretraining` binary — that would let
Gemini smooth over partitioned-quality docs (e.g. first-half-good,
second-half-broken) that we'd want to split differently from uniformly
bad docs. Two dedicated axes handle that: `defect_rate_estimate` (how
much?) and `text_partition` (where?).

Dropping `cleaning_was_over_aggressive` per user direction — out of
scope; the cleaner is what it is, we're judging the OUTPUT for
pretraining fitness.

## Full prompt

```
[SYSTEM INSTRUCTION]
You are reviewing a document drawn from a Greek-language pretraining
corpus. The document has been through a deterministic cleaner that:
- strips PDF-extraction residue (PostScript glyph names like /hyphenminus,
  /uni03B1, /g302; Adobe font-subset markers like /XQDMQS+CenturyGothic)
- removes individual lines dominated by PDF glyph IDs
- canonicalizes repeated special chars (dots ........ → .....,
  separator lines of ---/___/\_\_\_ → ---)
- strips characters from scripts other than Greek, Latin, French-Spanish
  diacritics, punctuation, digits, common symbols
- collapses multi-space runs to bucketed canonical lengths

Where cleaning removed whole lines, the marker <!-- line-removed -->
appears inline. Where cleaning stripped a partial line's content, you may
see <!-- text-missing --> at line end. These markers are SIGNAL — they
tell you a discontinuity comes from cleaning, not from the source.

Some statistical variation is acceptable for pretraining: up to roughly
5% of the document can have isolated minor defects (missing punctuation,
small word-internal breaks, one-off awkward phrasings) AS LONG AS the
text is mostly sensible — the main subject can be understood end-to-end
and only minor details are lost. What is NOT acceptable is text where
the subject itself becomes unclear: pervasive incomplete sentences,
broken words mid-token, bad syntax, or disconnected fragments that leave
the reader unable to follow what the text is actually about.

Answer the structured questions below independently. There is no single
master "accept / reject" label — we collect orthogonal signals and set
the rejection policy later from the distribution shape.

[CONTEXT]
<cleaned doc, markers inline, ≤4000 chars; middle-truncate with
[...truncated...] if longer so start and end are both shown>

[QUESTIONS]

# --- Quantity / location axes ---------------------------------------

1. defect_rate_estimate: "≤5% (sensible, subject clear)" / "5-20%" /
                         "20-50%" / ">50%"
   What fraction of the text has cleaning defects (broken words,
   missing content, mid-thought sentences, non-sequitur jumps)?

2. text_partition: "uniformly_good" / "uniformly_bad" / "half_half" /
                   "mostly_good_with_bad_patches" /
                   "mostly_bad_with_good_patches"
   Are defects uniform or clustered? Helps distinguish "shred everything"
   vs "first half readable, second half gibberish" vs "occasional patches."

3. subject_clear_end_to_end (yes / no / uncertain)
   Can a Greek reader follow the main subject from start to finish,
   allowing for up to ~5% isolated minor defects?

# --- Per-symptom binaries (each maps to a quantifiable stat) --------

4. has_broken_words_mid_token (yes / no / uncertain)
   [→ chars_dropped_by_per_char_filter / non_empty_chars_in]
   Are multiple words visibly broken mid-token (chars missing inside a
   word, word split across a cleaner-inserted gap)?

5. has_narrative_jumps_from_line_drops (yes / no / uncertain)
   [→ lines_dropped_by_cleaner / non_empty_lines_in]
   Are there places where two consecutive surviving lines jump in topic
   or logic, suggesting intermediate content was removed?

6. has_mid_thought_sentences (yes / no / uncertain)
   [→ either filter stat]
   Are multiple sentences left grammatically incomplete, as if content
   after them was removed (trailing subject without verb, etc.)?

7. is_too_short_to_be_useful (yes / no / uncertain)
   [→ non_empty_chars_out absolute]
   Is the surviving text so short (< ~1-2 paragraphs of actual content)
   that it wouldn't meaningfully contribute to pretraining?

# --- Free-form explanation ------------------------------------------

8. short_reason (≤ 40 words)
   One-sentence summary — what's the dominant issue (or "clean" if none)?
```

## Post-hoc policy moves this prompt leaves on the table

Once we have 150 verdicts + per-doc stats joined:

- Reject if `defect_rate_estimate` ∈ {"20-50%", ">50%"} regardless of
  partition.
- For `5-20%`: reject only if `subject_clear_end_to_end=no`.
- Handle partitioned docs specially: `text_partition="half_half"` with
  `subject_clear=no` may warrant splitting the doc rather than dropping.
- Per-symptom cutoffs: find the metric threshold where each Q4-Q7
  crosses ~50% yes-rate — those become independent rejection gates if
  we want a conjunctive rule.
- `is_too_short_to_be_useful` → absolute-size floor (metric:
  `non_empty_chars_out`).

## Calibration contract

Per `feedback_stratified_sampling.md` + `feedback_dont_generalize_
beyond_test_parameters.md`:

- **Sampler**: stratified over each of the four correlated stats (char-
  strip ratio, line-drop ratio, non-empty-chars-out absolute, combined
  drop pct). 10 zones per metric. 150 total across zones.
- **Output**: per-doc verdicts.jsonl + per-metric `yes_rate_by_zone.md`
  histograms + a combined table showing Q1/Q2/Q3 distributions joined
  with the four stats.
- **No pre-committed thresholds** — the calibration RESULT is the
  threshold, not an input.

## Files to build

- `cleaning_scripts/sample_broken_text_candidates.py` — stratified
  sampler across the four metrics.
- `cleaning_scripts/gemini_broken_text_reviewer.py` — ThreadPoolExecutor
  driver, same shape as `sample_and_review_line_vs_span.py`, 150 calls,
  gemini-2.5-flash.
- `cleaning_scripts/analyze_broken_text_verdicts.py` — per-metric
  histograms + zone yes-rates + combined correlations.
