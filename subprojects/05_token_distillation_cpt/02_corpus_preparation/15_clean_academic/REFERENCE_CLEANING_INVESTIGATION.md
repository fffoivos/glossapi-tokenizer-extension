# Academic reference/citation cleaning for the Greek CPT corpus — investigation + design

**Date:** 2026-06-13 · **Status:** investigation complete, detector built + validated, drop-policy decisions open
**Scope:** the four academic sources of the ~60 B-new-token Apertus CPT mix.
**Method:** a 25-agent investigation (profile each source from real samples → evaluate each
essay method against the evidence → adversarially verify the load-bearing claims → synthesize),
then a Rust detector built and validated against the verified anchors. Raw agent output:
`investigation/_raw_synthesis.json`.

---

## 0. What this answers

The corpus question: **for the Greek academic CPT data, do the scholarly-corpus reference-handling
methods (peS2o/GROBID, MEDITRON, S2ORC) translate — and what do we actually do?** The corpus goal
is **plain Greek language modelling for Apertus continued-pretraining**, *not* citation graphs /
retrieval / scholarly QA. That goal collapses the essay's taxonomy cleanly (§2).

## 1. The dataset under treatment

New-Greek bucket of the 60 B mix (`PRODUCTION_MIX_DECISION_20260612.md`) draws academic text from
four sources inside `selected_after_apertus_and_internal_dedup.parquet` (`source_dataset` field):

| Source | ~Tokens | Storage at rest | Reference lever |
|---|---|---|---|
| **openarchives.gr** | 7.1 B | whole-doc Markdown (`.jsonl.zst`, field `text`) | none → detect in text |
| **greek_phd** | 5.0 B | whole-doc Markdown (`.jsonl.zst`, field `document`) | none → detect in text |
| **Apothetirio_Kallipos** | (tail) | **section parquet**, `predicted_section` label | **`β` = bibliography class** |
| **Apothetirio_Pergamos** | (tail) | **section parquet**, `predicted_section` label | **`β` = bibliography class** |

The GlossAPI cleaner has **no** reference/citation handling today (verified by grep) — this is net-new.
The section classifier (`gloss_section_classifier.py`) tags `π`=ToC, `β`=bibliography, `ε.σ.`=front,
`κ`=body, `α`=appendix; on the two parquet sources `β` is a *free* structural lever, but a **noisy** one.

## 2. Executive verdict — REMOVE / MASK / STRUCTURE

The goal is "don't trip over citations," not "use them," so:

- **STRUCTURE (S2ORC link / MEDITRON summary): rejected.** No downstream consumer; the marker→entry
  links are not reconstructable from OCR (humanities markers key to *per-page footnotes*, not the
  bibliography; numbered indices are OCR-corrupted/reordered, e.g. `1136811374`). A summary-injection
  also *fabricates* non-source Greek and needs per-marker LLM calls at 55 B-token scale (violates
  Rust-by-default).
- **MASK in-text → `<ref>` token: rejected.** Even the summary-free variant injects a new mid-prose
  token into the *existing* Apertus tokenizer (fertility risk) and corrupts OCR'd inline digits.
- **REMOVE end-matter + footnote reference mass: adopted — but as SEGMENT + per-family COUNT +
  reversible sidecar + user-controlled drop knob**, never a hard delete, never a baked threshold.
  In-text markers are **kept-and-counted**. "Drop harder" (the OCR-Greek posture) is applied
  **content-gated**, not uniformly — see the polytonic guardrail in §5.

### Method-translation table

| Essay method | Translates? | What we do instead |
|---|---|---|
| peS2o/MEDITRON: GROBID → keep sections, remove refs | **needs adaptation** | Goal transfers; tool does not (no source PDFs on disk; Docling already flattened layout). Substitute the in-house section classifier; replace REMOVE with SEGMENT+COUNT+EMIT. |
| MEDITRON: delete end-matter list | **needs adaptation** | Right concept, wrong verb (segment-not-delete) and **wrong scope** — in humanities the end list is the *smaller* sink (006: ~4.4 % end-bib vs **22 %** body footnote stream). |
| MEDITRON: mask in-text → summary special token | **translates poorly** | Summary is infeasible + fabricates text. Keep-and-count markers; never inject summaries/tokens. |
| S2ORC: parse refs into a linked field | **translates poorly** | No consumer; links unrecoverable from OCR. Segment-not-delete *spirit* already covered by the reversible sidecar. |
| sci-writing: random `\citeauthor` substitution | **translates poorly** | Premise absent — **zero** `\cite`-family commands across 100 samples + 82 k parquet rows + 3 raw shards. Keep in-prose author names; never substitute. |
| The REMOVE/MASK/STRUCTURE choice itself | **needs adaptation** | Collapses to REMOVE-end-matter (by segmentation) + keep-and-count in-text, per-text-type, user policy. |
| OCR-Greek: segment-not-delete, drop *harder* | **needs adaptation** | Posture correct + project-mandated, but "drop harder" is **content-gated** — 006's end-bib is dense polytonic/Katharevousa **Greek** to keep, not Latin noise. |

## 3. The Greek-specific crux (why a literal port fails)

**The dominant reference token sink in humanities theses is the body-interleaved footnote stream, not
the end-matter list.** Verified on greek_phd doc 006 by the detector this session:

- end-matter bibliography (`## ΠΗΓΕΣ -ΒΙΒΛΙΟΓΡΑΦΙΑ` at line 7397): **60,837 chars**, year-density 0.78;
- footnote stream: **586 footnotes**, of which **157 citation-only** and **429 prose/hybrid**
  (201,290 chars of genuine Greek commentary that must be **kept**);
- ano-teleia separator `·` (U+0387): **983** occurrences — *exactly* the investigation's hand count;
  the Latin middle-dot U+00B7 is 0 (the two are distinct codepoints, never folded).

So a peS2o/MEDITRON end-reference cut is necessary-but-far-from-sufficient (misses ~5× the mass),
and the footnote stream is **dual-content** — blanket removal would delete real prose. There are
**four** co-existing in-text regimes across the corpus (footnote stream; ano-teleia footnote lists;
parenthetical `(Author, year)` — APA/Harvard linguistics theses; bracket `[NN]` Vancouver STEM/medical),
which is why the counters are split per family and never summed.

## 4. Per-source mechanism

One Rust module (one detector, many consumers) — **DETECT → SEGMENT → EMIT SPANS → COUNT**, never
hard-delete. What differs is the structuring layer that feeds it:

- **greek_phd / openarchives** (whole-doc): (A) end-matter header boundary — take the **first**
  bib-family header in the lower portion (bibliographies are sub-divided into
  `Ελληνόγλωσση`/`Ξενόγλωσση`/`Δικτυογραφία`, so the *earliest* one begins the block), rejecting
  TOC-context rows, the `Βιβλιογραφική αναφορά` colophon, and `…ανασκόπηση` lit-review chapters;
  (B) footnote-stream detection with citation-vs-prose classification; (C) in-text family counters.
  openarchives additionally needs the existing mojibake/GFM-table gates to run **first** on the
  cp1253 / table-heavy subset.
- **Kallipos / Pergamos** (section parquet): consume `predicted_section==β` rows, then a **content +
  positional gate** — because `β` precision is only ~0.85–0.90. Verified false positives that the gate
  keeps: the `ΚΑΛΛΙΠΟΣ` colophon, `Προαπαιτούμενη γνώση` / `Λέξεις-κλειδιά` / `Απάντηση`-`Λύση`
  textbook apparatus, and **front-loaded CV `β`** on Pergamos theses (`Μεταπτυχιακή εκπαίδευση`…,
  36.7 % of β-docs). Kallipos references are *per-chapter interleaved* (real `κ` body after the last
  `β` in 24.5 % of docs), so the doc-level "end-of-doc" boundary mis-locates — the column read + gate
  is the right lever, not a header scan.

On a 40 k-row Kallipos slice the gate split **1,657 `bib` / 545 `kept-non-bib`**, with real
`Βιβλιογραφία` sections (latin-fraction 0.93–0.96, bullet-lists, year-density 1.0) flagged and the
colophon / solution sections kept — as intended.

## 5. Risks + guardrails (folded in from adversarial verification)

1. **Boundary over-segmentation.** 000's TOC row `## Βιβλιογραφία (σελ. 1029-1163)` would amputate
   99.4 % of the doc. Guards: lower-portion position + TOC/dot-leader rejection + colophon/lit-review
   deny + content-confirm year-density + **fail-closed** (keep text on disagreement). Unit-tested.
2. **`β` not clean enough to drop blindly** (~0.85–0.90). Guard: per-section citation-shape +
   positional gate; a `beta_kept_as_non_bib` audit counter; never drop on the raw label.
3. **Footnote dual-content.** ~21–64 % of footnote chars are genuine Greek prose. Guard: classify
   citation-only vs prose/hybrid, **default-keep** the prose, emit the split as separate counters.
4. **Polytonic/Katharevousa is a KEEP signal** the gate must protect: "drop harder" is gated on
   *Latin*-fraction, and a Greek-letter-fraction signal is computed so a polytonic Greek archival
   bibliography is not treated as Latin noise. *(Counter present; an explicit polytonic-codepoint
   sub-signal is a noted enhancement — see §7 open items.)*
5. **Fail-closed under-removes.** Every tie resolves toward KEEP, so reference mass the gates miss
   stays in training. This is a deliberate precision-over-recall default; the user sets drop policy
   knowing it under-removes rather than risks truncating body prose.

## 6. What was built (`./` layout)

| Path | What |
|---|---|
| `reference_detector/` | **Rust** crate — `reference_signals.rs` (regex/label/codepoint inventory) + `reference_module.rs` (`detect_doc` / `detect_sections`, split counters, 7 unit tests) + `reference_detect` CLI (zstd/JSONL streaming, rayon batches). Slots into `glossapi_rs_cleaner` as a module later. |
| `driver/run_reference_detect.py` | thin I/O driver — parquet→grouped-JSONL (sections) or `.jsonl.zst`→binary (wholedoc); writes spans + counters(+parquet). No per-doc text work in Python. |
| `review/sample_refspans.py` | review sampler — full-doc, post-cleaner, inline `<match kind=… gated=…>`, 10-zone stratified, metric-prefixed filenames; over-samples the risky gate decisions for audit. |
| `investigation/` | the multi-agent synthesis (`_raw_synthesis.json`), section-labelled samples, an example inline-`<match>` render. |

**Validation (this session):** `cargo test` 7/7 green; on real greek_phd 006 the detector reproduced
the verified anchors (U+0387 = 983; bib boundary at L7397; footnote-stream dominant); the Kallipos
section path produced the sensible gate split above; the sampler renders correct inline `<match>`.

## 7. Scope boundaries & open decisions (for you)

**Explicitly out of scope** (flagged by the completeness critic): the corpus also contains
`ellinika_dedomena_europaikou_koinovouliou` (EU-Parliament legislative text, 21 % of the audit pool)
whose reference forms are legal-instrument codes (`COM(2011)0809`, `2011/0401(COD)`) — **not**
academic-citation-shaped. It is *not* one of the four academic sources and is routed to a separate
legal-reference pass, not handled here.

**Open decisions — the detector emits + counts and keeps everything by default; you set policy:**
1. **Drop policy per family:** (a) end-matter bib span — drop or keep-and-down-weight? (b) footnote
   stream — drop citation-only, keep prose/hybrid (the default), or count-only? (c) in-text markers —
   keep-and-count (recommended) or strip the pure-numeric families?
2. **Gate thresholds** (year-density, latin-fraction, CV positional cutoff): set now from the counter
   distributions, or run detect-and-emit first and set them after reviewing the stratified samples?
3. **Section classifier on greek_phd/openarchives?** Run the sectionize-then-classify pass (adds the
   established-tool `β` candidate, β-precision ~0.85–0.90, needs a pre-segmentation pass), or rely on
   the bespoke Rust header-boundary detector for the two whole-doc sources (current default)?
4. **openarchives mojibake docs** (cp1253): quarantine-and-exclude, route to a re-decode pass, or keep?
5. **Pergamos `β` count** to budget the reference-text fraction: the brief's `β≈20 k` vs the measured
   full-parquet `β≈70 k` need reconciling before any quota claim.
6. **Polytonic sub-signal:** add an explicit polytonic-codepoint gate feature (recommended), or keep
   the Latin-fraction proxy?
7. **Build/land order:** kallipos (free labels, cleanest) → pergamos → greek_phd → openarchives — or
   prioritise the high-token humanities footnote sink (greek_phd) earlier despite the segmentation cost?

Per project rule, none of these keep/drop thresholds are baked in; the scripts detect, segment, count,
and expose them as knobs for you to drive.

## 8. Corpus-scale detect-and-emit results (2026-06-13)

Full run on the **raw source** data (home), all four sources — calibration distributions to set
policy from. (A production run on the CSCS `selected` parquet is the Clariden job
`clariden/run_academic_refs.sbatch`, which detects the same signals on the exact CPT text.)

| source | docs | bib-header | end-bib chars | footnotes | fn-cit % | β sections | β kept-non-bib | dominant in-text |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| greek_phd | 37,229 | **80 %** | 3.71 B | 2.38 M | 47 % | — | — | author-year 35 % |
| openarchives | 179,845 | **66 %** | 4.82 B | 3.18 M | 45 % | — | — | author-year 33 % |
| kallipos | 4,827 | — | — | — | — | 24,039 | **29 %** | author-year 44 % |
| pergamos | 15,241 | — | — | — | — | **70,243** | **36 %** | author-year 40 % |

What the full run confirms (vs the sample-based design):

- **The four in-text regimes are real and co-exist** — `(Author, year)` is the *dominant* form in all
  four sources (33–44 %), so a MEDITRON-style bracket-`[NN]` masking would target the wrong object
  for most docs. The split-counter design is vindicated at scale, not just on the sample.
- **End-matter mass is large**: ~3.71 B (greek_phd) + 4.82 B (openarchives) chars of bibliography,
  with a header detected in 80 % / 66 % of docs (the ~20–34 % header-miss rate is exactly why the
  detector *segments and fails closed* rather than hard-cutting).
- **β-gate false-positive rate matches the verified estimate**: 29 % (kallipos) / 36 % (pergamos) of
  `β` sections are kept-as-non-bib (colophons, CVs, textbook apparatus) — i.e. a blind `β`-drop would
  have removed that much real content. The content+positional gate is load-bearing, as predicted.
- **Open question #5 resolved**: the measured Pergamos `β` count is **70,243** (not the brief's ~20 k) —
  use this for any reference-fraction budgeting.

Outputs per source (auditable, nothing dropped): `out/<source>_full/refspans/<source>.spans.jsonl`,
`out/<source>_full/<source>.counters.jsonl`(+`.parquet`), `out/<source>_full/review_samples/`
(full-doc inline-`<match>`, stratified on `endmatter_bib_chars`).

### Production-text run (Clariden job 2527626, on the `selected` parquet)

The same detector, run whole-doc on the exact CPT text (`selected_after_apertus_and_internal_dedup.parquet`)
via `clariden/run_academic_refs.sbatch` — output under
`/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/academic_refs_20260613T105022Z/`:

| source | docs | bib-hdr | end-bib chars | note |
|---|---:|---:|---:|---|
| greek_phd | 35,790 | 81 % | 3.11 B | ≈ raw |
| openarchives | 147,652 | **80 %** | 3.47 B | header-rate up from 66 % raw (selection dropped OCR-broken docs) |
| kallipos | 4,818 | **0 %** | 1.5 M | whole-doc misses it |
| pergamos | 15,232 | **1 %** | 8.1 M | whole-doc misses it |

**This empirically confirms the central architectural split.** On the production parquet
Kallipos/Pergamos are doc-level (β labels gone), so whole-doc detection finds ~nothing (0–1 %
header) — their bibliographies are *per-chapter*, not a single end block. So **Kallipos/Pergamos must
use the section-`β` path** (the raw section parquets, `--mode sections`, which found 24,039 + 70,243
β sections), while greek_phd/openarchives detect cleanly in whole-doc mode on the production text
(80–81 % header). `openarchives` count = 147,652 matches the ROADMAP's 147 k academic figure exactly.

**Operational note for re-runs:** Clariden has no cargo/pyarrow by default — rust toolchain bootstrapped
at `$SC/rust/{cargo,rustup}`, a standalone pyarrow venv at `$SC/python_envs/refdetect_py` (py3.6 +
pyarrow 6.0.1; use `pa.table(dict)`, not `Table.from_pylist`). Connect with
`-o ControlMaster=no -o ControlPath=none` (the multiplexing socket breaks the ela ProxyJump).
