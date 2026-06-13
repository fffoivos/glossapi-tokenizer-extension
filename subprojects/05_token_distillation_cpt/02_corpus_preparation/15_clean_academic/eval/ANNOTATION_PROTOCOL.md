# Reference-detector evaluation — Opus annotation protocol

Goal: a **confusion matrix per decision the detector makes**, plus a **typed labelled dataset**
rich enough to tune the deterministic rules. Annotation is done by Opus 4.8 agents, **blind to the
detector's decision**, returning **structured JSON keyed by line span** (not inline tags, not free
text). We hold the source text; the annotator only emits coordinates + typed labels + a verbatim
evidence quote we can re-check for hallucination.

## Why these choices (answering the design questions)

- **Tags, not prose.** Every dimension is a controlled enum so the output is machine-readable and
  directly comparable to the detector. Free-text labels would not score.
- **Line spans into a line-numbered input**, not inline HTML. OCR docs are 10–100 k lines; asking a
  model to re-emit the doc with tags truncates and drifts. We give `Lnnnnn:` prefixes and the model
  returns the integers. Offsets map back to the detector's `line_start/line_end` exactly.
- **Blind annotation** (the detector's span/decision is withheld) so we measure **recall** (missed
  references), not just precision. The hidden detector decision lives in a separate manifest the
  scorer joins on.
- **Rich metadata, not just start/end.** We capture `n_entries`, style, language, script, subject,
  and per-region **syntax features** — because the point is to *tune the deterministic algo*, and
  those are exactly the features it can compute. The confusion matrix tells us where it's wrong; the
  typed features tell us *why* and which rule to add.
- **Evidence quote required** → the scorer verifies the quote occurs within the claimed line span
  (deterministic hallucination guard); units that fail are dropped from the matrix and reported.

## The three evals (each a distinct decision → its own matrix)

| Eval | Unit | Detector decision under test | Sources |
|---|---|---|---|
| **B — β-gate** | one `predicted_section==β` section | `bib` vs `kept-non-bib` | kallipos, pergamos |
| **A — end-matter boundary** | a doc's tail (last ~600 lines, line-numbered) | `endmatter_header_found` + the start line | greek_phd, openarchives |
| **C — footnote class** | a ~150-line body window | per-footnote `citation_only` vs `prose/hybrid` (+ is-it-a-footnote) | greek_phd, openarchives |

## Controlled vocabulary (shared)

`kind` — what the region IS:
`end_bibliography` · `chapter_bibliography` · `subdivided_sublist` (Ελληνόγλωσση/Ξενόγλωσση/etc.) ·
`archival_primary_sources` (ΑΡΧΕΙΑΚΕΣ ΠΗΓΕΣ) · `web_sources` (Δικτυογραφία/Ηλεκτρονικές Πηγές) ·
`further_reading` · `footnote_reference` · `intext_citation` · `colophon_citation` (Βιβλιογραφική
αναφορά / publisher cover) · `cv_publication_list` (author's own works, CV) · `not_reference`
(prose / TOC / appendix body / exercise / apparatus).

`citation_style` — `chicago_turabian` · `apa_harvard_authoryear` · `vancouver_numbered` ·
`footnote_humanities` · `iso690` · `mla` · `mixed` · `none` · `na`.

`language` — `greek` · `latin_foreign` · `mixed_greek_foreign`.
`script` — `monotonic_greek` · `polytonic_greek` · `latin` · `mixed`. *(polytonic is a KEEP signal.)*
`subject_register` — `humanities` · `social_science` · `stem` · `medical` · `law` · `theology` · `unknown`.

`syntax_features` (per region, booleans — the deterministic clues): `has_authors`, `has_year`,
`has_title`, `has_publisher_place`, `has_page_range`, `has_volume_issue`, `has_doi_or_url`,
`has_editors_επιμ`, `entries_numbered`, `entries_dash_bulleted`, `entries_hanging_indent`.

`position` (Eval A) — `front_matter` · `mid_body_after_chapter` · `end_before_appendix` ·
`end_after_appendix` · `document_end` · `interleaved_footnotes`. Plus `followed_by`:
`appendix` · `index` · `more_body` · `nothing_eof`.

## Return schemas (exact JSON the agent emits)

**Eval B (β-section):**
```json
{"unit_id":"…","is_reference_list":true,
 "kind":"end_bibliography","citation_style":"apa_harvard_authoryear","language":"mixed_greek_foreign",
 "script":"monotonic_greek","subject_register":"stem",
 "syntax_features":{"has_authors":true,"has_year":true,"has_title":true,"has_publisher_place":true,
   "has_page_range":false,"has_volume_issue":true,"has_doi_or_url":true,"has_editors_επιμ":false,
   "entries_numbered":false,"entries_dash_bulleted":true,"entries_hanging_indent":false},
 "n_entries":34,"confidence":"high","evidence_quote":"<verbatim first ~120 chars>","reasoning":"…"}
```
**Eval A (doc tail):**
```json
{"unit_id":"…","has_end_bibliography":true,"bibliography_start_line":7421,"bibliography_end_line":8003,
 "followed_by":"appendix","subdivided_sublists":[{"kind":"subdivided_sublist","start_line":7421,"header_quote":"Ελληνόγλωσση"},…],
 "main_kind":"end_bibliography","citation_style":"chicago_turabian","language":"mixed_greek_foreign",
 "script":"polytonic_greek","subject_register":"humanities","n_entries":210,
 "syntax_features":{…},"confidence":"high","evidence_quote":"<header + first entry>","reasoning":"…"}
```
**Eval C (body window):**
```json
{"unit_id":"…","footnotes":[
   {"start_line":108,"end_line":108,"content_type":"citation_only","language":"latin_foreign","has_year":true,"evidence_quote":"…"},
   {"start_line":110,"end_line":114,"content_type":"prose_commentary","language":"greek","has_year":false,"evidence_quote":"…"}],
 "confidence":"high"}
```
`content_type` ∈ `citation_only` · `prose_commentary` · `hybrid` · `not_a_footnote`.

## Sampling (stratified, deterministic — so the matrix has signal across the decision boundary)

- **Eval B**: stratify β by gate decision (`bib` / `kept-non-bib`) × feature bins (year_density,
  latin_fraction, positional_fraction) → oversample near the boundary (low-year `bib`, high-year
  `kept`) where errors live. ~90 sections (≈ balanced Kallipos/Pergamos).
- **Eval A**: stratify docs by `endmatter_header_found` (Y/N) × bib-char mass → **include the
  header-not-found docs** (that's where false negatives hide). ~40 doc tails.
- **Eval C**: stratify footnotes by `citation_only`/`prose` × greek-fraction; sample the *windows*
  containing them. ~50 windows.
- **15 % double-annotated** by a second agent → inter-annotator agreement (Cohen's κ) as a
  ground-truth quality floor. If κ is low, the "truth" is shaky and we say so.

## Scoring (deterministic)

For each eval, align annotation → detector and emit:
- **confusion matrix** (B: bib vs not; A: found vs not + boundary line error distribution; C: 3-way
  content_type) with precision / recall / F1 per class;
- **error breakdown by stratum** (style / language / script / subject / syntax) → *which conditions
  the detector fails on* = the tuning to-do list;
- the **hallucination check** (evidence_quote ⊂ claimed span) and **κ** on the double-annotated subset.

Outputs: `eval/units/`, `eval/annotations/`, `eval/CONFUSION_MATRIX.md` + `eval/results.json`.
