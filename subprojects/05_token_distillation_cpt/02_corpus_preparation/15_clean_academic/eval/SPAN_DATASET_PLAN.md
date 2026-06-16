# Bibliography-span dataset — annotation plan (what we're doing now)

**Goal.** A labelled dataset of **every bibliographic-list SPAN and its START + END line** in academic
documents, *wherever it sits* — end-of-document, end-of-chapter, or subdivided. Trains/evaluates a
**multi-span** bibliography detector (not a single end-boundary) and powers the Δstart/Δend boundary
eval at real scale.

**Decided (2026-06-13).**
- **Sources (3):** `greek_phd` + `openarchives` (native whole-doc) + `Kallipos` (reconstructed from its
  section rows into a document). **Pergamos dropped.** (Reconstructing Kallipos matches the production
  `selected` parquet, which is doc-level even for the section sources.)
- **Target:** **~2000 bibliography spans**, stratified across the 3 sources, oversampling multi-span
  (chapter-bibliography) documents so the set has span *variety*, not 2000 near-identical end lists.
- **Annotator:** **Opus 4.8.** (Haiku was tested and rejected: recall 0.63 — misses chapter bibs — and
  fragments spans; κ=0.56 vs Opus. Fine for easy end-matter, not for the hard/diverse spans we need.)
- **Granularity:** **lines** (char-level adds OCR-line-length noise without changing the picture).

**Unit.** A line-numbered document window (TRUE offsets, `L#####:` prefixes). Per document:
- a **tail** window (last ~420 non-empty lines) → the end-matter bibliography;
- **body** windows around entry-density spikes (≥0.4 entry-lines per 60-line block) → end-of-chapter
  bibliographies. Annotation is **blind**; the detector's hypothesis is withheld (kept in the manifest
  for scoring).

**Per-span schema (rich metadata).** Opus marks every contiguous bibliographic-list span and returns:
- `start_line`, `end_line` (the `L#####` numbers)
- `kind`: end_of_document · end_of_chapter · subdivided_sublist · archival_primary_sources · web_sources · further_reading
- `citation_style`: chicago_turabian · apa_harvard_authoryear · vancouver_numbered · footnote_humanities · iso690 · mixed · none
- `language`: greek · latin_foreign · mixed   ·   `script`: monotonic_greek · polytonic_greek · latin · mixed
- `subject_register`: humanities · social_science · stem · medical · law · theology · unknown
- `noise_level`: clean · light · heavy   (Docling/GlossAPI OCR + extraction noise inside the span)
- `n_entries` (int) · `has_header` (bool — a "Βιβλιογραφία"-type header vs header-less)

**Why this shape.** Start/end + `kind` train the multi-span detector and give the boundary eval weight;
`noise_level` / `style` / `script` let us slice failures (e.g. "does noise predict boundary error?",
"are polytonic Greek bibs the recall hole?"); `n_entries` is the list-vs-single discriminator that
already broke the β-gate plateau.

**Annotation throughput (2026-06-15).** The mass-concurrent Opus run mostly failed on a server-side
**tokens-per-minute throttle** (16 agents at once → 1–2 survive). Fix = annotate **sequentially**, one
agent at a time (`span_chunk_workflow.js`), driven by a paced loop (`span_loop_step.py`: merge → next
chunk of 10). Sequential clears the throttle (16/16) at ~5 min/batch. One batch (`batch_0055`) trips
the AUP content filter on a doc and is parked in `units/SPAN_skip.json` for an end-of-run per-unit
recovery pass.

**+5h unseen extension (2026-06-15).** On top of the base 178 batches, `build_span_units_ext.py`
appends ~60 batches (~840 windows, ~700 spans) of **unseen** documents — disjoint from the base sample
(fresh stride over not-yet-sampled greek_phd/openarchives ids; Kallipos hash residue `md5%30==7`). It
continues unit-id + batch numbering and appends atomically to the manifest + batchpaths so the live
loop flows straight into them. Total target ≈ 238 batches ≈ 2,800+ spans. After completion, measure
the true cross-source distribution and decide a *targeted* thin-strata top-up (candidates: polytonic
Greek, iso690, further_reading, theology).
