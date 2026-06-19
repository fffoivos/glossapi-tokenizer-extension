# Document-structure segmentation — annotation specification (v2)

## Decisions log (v2 final — read this first)
- **Annotation unit = front+tail, ONE unit/doc** (`build_fronttail.py`): whole-doc when it fits ≤420k chars,
  else front (≤140k chars) + tail (≤260k chars) with the middle elided. CHAR-bounded so long-line sources
  (kallipos) still fit context. This is bounded at the document count (~1,581 units) — no chunking explosion.
- **Annotation windows are DECOUPLED from the inference check.** The generous front+tail windows exist only
  to harvest ToC + end-bibliography training examples cheaply ("find more, faster"); the tight limits used to
  *detect* ToC/bib at inference are a separate concern. So front+tail's blind spot (mid-document chapter
  bibs) is acceptable: the model learns bib-entry features from the end-bibs and generalises at inference.
- **We retrain our OWN model** on the fresh gold (the project's line classifier), not GlossAPI's SVM.
- **GlossAPI reference** (`/home/foivos/glossAPI/src/glossapi/`, `corpus.section()` + `corpus.annotate()`):
  it does this exact job. Borrow its **tuned ToC/index gate at inference** — a section can be ToC only within
  `min(300 lines, 30% of doc)` (cumulative `section_length ≤ 300` AND `section_propo ≤ 300‰`). Its structural
  derivation is identical to ours (`ε.σ.` front-matter before first π, `κ` main text between, `α` appendix
  after last β); chapter mode keeps each chapter's β (bibliography). Matches our empirical ToC≤~370-line / ~20%.

## 0. What changed and why we are redoing this
v1 annotated *bibliography spans* on isolated windows. Two faults surfaced when we inspected the gold:
(a) **loose boundaries** — chapter-bibliography spans swept in the preceding figures/tables/equations/prose
(e.g. a span marked 196–274 whose real reference entries only start at 224); (b) **no notion of the rest of
the document**, so look-alikes (ToC, the author's own publication list) had no home and the classifier got a
muddy signal. v2 fixes both by annotating the **whole structure of the document**, not just its bibliography.

## 1. Why we annotate this (the purpose — give this to the annotator)
We are building a **clean plain-Greek training corpus** from Docling-extracted, GlossAPI-cleaned Greek
academic documents (journal **articles** and **books / theses**). A language model should learn from the
**MAIN TEXT** — the running scholarly prose — and NOT from structural scaffolding: tables of contents,
reference lists, appendices, title/author pages, or the author's own list of publications. Removing that
scaffolding **without deleting a single line of main text** requires knowing the document's STRUCTURE.

**Cost asymmetry (decisive):** deleting main text corrupts training and breaks coherence; leaving a stray
reference line is cheap and reversible. So **boundaries must be exact** — a section's span contains ONLY
that section, never a neighbouring line of main text. When unsure whether an edge line is scaffolding or
main text, **leave it as main text** (do not extend the scaffolding span over it).

## 2. The mental model: anchor on ToC and bibliography, segment the rest
A document's structure is largely pinned by two landmarks:
- the **table of contents** near the front — everything before it is **front-matter** (title page, authors,
  dedication, acknowledgements, author's note);
- the **bibliography** near the back — everything after the *last* reference list is the **appendix**.
Between the ToC and the bibliography is **main text**. In a **book/thesis** the main text is divided into
**chapters**, and a chapter may end in its **own (chapter) bibliography**; detecting those subdivides the
main text into chapters. In a **journal article** there is usually no ToC and no chapter bibliographies —
just front-matter (title/abstract) → main text → bibliography → optional appendix.

So segmentation = **find the non-main-text structural sections precisely; everything else is main text.**

## 3. Document type (annotator infers it first)
- **article** — journal/conference paper: title + authors + (abstract) → body sections → one bibliography →
  optional appendix. Rarely a ToC; no chapter bibliographies.
- **book** — book / thesis / monograph: front-matter → **table of contents** → chapters (each = main text +
  optional **chapter bibliography**) → an **end-of-document bibliography** → **appendices**.
Infer `doc_type` from the cues present (a ToC ⇒ book; an abstract+single end-list ⇒ article).

## 4. Section taxonomy (what the annotator MARKS; everything unmarked is `main_text`)
Mark only the **structural, non-main-text** sections below. Each is one **contiguous** line span.

| kind | what it is | boundary |
|---|---|---|
| `front_matter` | title page, authors/affiliations, dedication, acknowledgements, preface, author's note, declarations — the non-content material **before the main text / before the ToC**. (NOT the abstract — see §6.) | from the first content line to the line before the ToC or the first main-text line |
| `table_of_contents` | the ToC: a navigational list of **chapters/sections** with page numbers and/or dotted leaders. NOT lists of figures/tables/abbreviations (those are main text — see §6). | the listing itself + its header (`Περιεχόμενα`/`Contents`) |
| `chapter_bibliography` | a reference list at the **end of a chapter**, mid-document (book only). | header (if any) + the contiguous reference **entries** only |
| `bibliography` | the **end-of-document** reference list (`Βιβλιογραφία`/`Αναφορές`/`References`), incl. its subdivided parts (Ελληνόγλωσση/Ξενόγλωσση/Δικτυογραφία/archival). | header (if any) + the contiguous reference **entries** only |
| `appendix` | material **after the last reference list**: appendices, annexes, indexes, glossaries. | from after the bibliography to doc end |
| `author_publications` | the **author's own** publication/CV list (thesis-end "Δημοσιεύσεις / Publications arising from this thesis", conference/talk lists). Reference-shaped but it is NOT the document's bibliography. | header + the list of the author's own works |

## 5. Boundary rules (this is the clean-signal requirement — read twice)
1. A bibliography / chapter_bibliography span = **[its header line, if present] + the contiguous run of
   reference ENTRIES**. It must **NOT** include preceding figures, figure/table captions, equations,
   markdown tables, or running prose — **even if they sit between the chapter heading and the first
   reference entry**. Start at the **first reference entry** (or its header); end at the **last reference
   entry** (not trailing prose/appendix).
2. A reference **entry** = a bibliographic record: author(s) + year/title + venue/publisher/pages (it may
   wrap onto 2–3 physical lines — those wrapped lines are part of the entry). A figure caption
   ("Εικόνα 2.20 …"), a table row (`| … | … |`), or an equation is **never** an entry.
3. `table_of_contents` = the actual ToC listing; do not extend into the page that follows.
4. `front_matter` ends exactly where the ToC starts (book) or where the first main-text/abstract line starts
   (article).
5. `appendix` starts on the first line after the final bibliography's last entry.
6. Sections are **non-overlapping** and **contiguous**. If a structural section is interrupted by main text,
   emit two separate spans.
7. When unsure about an edge line → it is **main_text** (exclude it from the scaffolding span). Under-capture
   is safe; over-capture deletes main text.

## 6. Edge cases (resolve consistently)
- **Abstract / Περίληψη** → treat as **main_text** (it is content, keep it). Do not mark it.
- **In-body figures, tables, equations, footnotes, inline citations** → **main_text** (part of the body;
  not removable). Only END-MATTER / chapter-end reference *lists* are bibliographies.
- **Subdivided bibliography** (Ελληνόγλωσση / Ξενόγλωσση / Δικτυογραφία under one `Βιβλιογραφία`) → **one**
  `bibliography` span covering all the sub-lists (they are one bibliography).
- **Author's own publications** at a thesis end → `author_publications`, NOT `bibliography`.
- **Fragmented-OCR bibliography** (one token per line: `Aula`, `Medica,` …) → still mark the whole region as
  `bibliography`; the goal is clean gold even when extraction shattered it.
- **No ToC** (typical article) → emit no `table_of_contents`; front_matter is the title/authors block.
- **Multiple chapter bibliographies** (book) → one `chapter_bibliography` span each; the prose between them
  is main_text (chapters).
- **List of figures/tables/abbreviations, glossary (ΣΥΝΤΟΜΟΓΡΑΦΙΕΣ)** → **main text** (do NOT mark). Only
  the chapter/section *table of contents* is `table_of_contents`. These lists often sit *after* the ToC, so
  the "between ToC and bibliography = main text" derivation already keeps them — leave them unmarked.
- If the window does not contain any structural section (it is all main text) → return an empty section list.

## 6b. Extraction-quality gate (pre-filter, not annotation)
Docs with `greek_badness_score > 60` (Rust `glossapi_rs_noise` scorer) are badly extracted (OCR garble /
non-Greek-dominant) and are **dropped before annotation** — see `badness_filter.py`. greek_phd carries the
score as a shard column; openarchives + kallipos lack it and are scored on the fly with the same scorer.
(Note: this gates *extraction garble*, not Greek-ness — a cleanly-extracted Latin-heavy doc passes.)

## 7. What the annotator sees (the unit)
Each unit is one or more **line-numbered windows** of a document (`L#####:` true offsets), labelled with the
window's role and the document's total length so position can be reasoned about:
- a **front** window (first ~300 lines) — catches front_matter, ToC, start of main text;
- a **tail** window (last ~400 lines) — catches end of main text, bibliography, appendix, author_publications;
- **body** windows around reference-density spikes — catch chapter bibliographies.
The annotator marks the structural sections present in the window using **absolute** L##### line numbers.

## 8. Output schema (per unit)
```
{ "doc_type": "article" | "book" | "unknown",
  "sections": [ {
      "kind": "front_matter|table_of_contents|chapter_bibliography|bibliography|appendix|author_publications",
      "start_line": int, "end_line": int,        # absolute L##### of the section's first/last line
      "language": "greek|latin_foreign|mixed",   # (bibliographies/ToC) dominant language
      "script": "monotonic_greek|polytonic_greek|latin|mixed",
      "citation_style": "chicago_turabian|apa_harvard_authoryear|vancouver_numbered|footnote_humanities|iso690|mixed|none",
      "n_entries": int,                          # reference entries (0 for non-bibliography kinds)
      "has_header": bool,                         # a section header precedes it
      "noise_level": "clean|light|heavy",
      "confidence": "high|medium|low"
  } ] }
```

## 9. The annotator PROMPT — operative taxonomy (simplified after the experiment)
The 10-doc experiment simplified the taxonomy to the user's intent: **Opus marks only two kinds —
`table_of_contents` and `bibliography`** — with `is_chapter_bibliography` and `is_authors_own_works` as
*flags* on a bibliography rather than separate kinds. Everything unmarked is main text; front-matter / body
/ appendix are derived deterministically from the ToC and bibliography anchors (§2). The earlier multi-kind
taxonomy in §3–§8 above is the original design and is superseded by this.

**The canonical, verbatim prompt is single-sourced** in `eval/wf_struct_annotate.js` (the `INSTRUCTION`
constant) so the spec and the running annotator cannot drift. It is organised as:
1. **Precision overrides everything** — exact, line-level, conservative; under-capture over over-capture;
   if a block is ambiguous, don't mark it.
2. **Table of contents** — the chapter/section contents table + its header; **explicit non-ToC look-alikes**
   that must stay main text: abbreviation/glossary lists (Συντομογραφίες/Γλωσσάρι), lists of figures/tables
   (Κατάλογος Εικόνων/Πινάκων), the index (Ευρετήριο).
3. **Bibliography** — reference-entry lists (end-of-doc + per-chapter via `is_chapter_bibliography`; subdivided
   Ελληνόγλωσση/Ξενόγλωσση/Δικτυογραφία/Πηγές folded into one; author's-own/CV via `is_authors_own_works`;
   mark even if OCR-shattered); **explicit non-bibliography look-alikes**: footnotes/endnotes, inline
   citations in prose, data tables, and the "ΒΙΒΛΙΟΓΡΑΦΙΑ … 250" line *inside the ToC*.
4. **Boundary discipline** — bibliography starts at the header/first entry (never the preceding
   figures/captions/tables/equations/prose, even between a chapter heading and the first entry), ends at the
   last entry (never trailing chapter heading/appendix/prose); never cut a wrapped entry; split around any
   interrupting main text.

Extraction-quality gate (§6b) runs before annotation: docs with `greek_badness_score > 60` are dropped.
