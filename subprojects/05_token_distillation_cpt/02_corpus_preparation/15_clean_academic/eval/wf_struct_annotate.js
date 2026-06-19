export const meta = {
  name: 'struct-annotate-experiment',
  description: 'Segment 10 Greek academic docs into ToC + bibliography sections (Opus, sequential pacing)',
  phases: [{ title: 'Annotate', detail: 'one Opus agent per doc, sequential to avoid rate limits' }],
}

// args = [ {batch: "/abs/path/batch_000.json", out: "/abs/path/ann_000.json"}, ... ]  (10 items)
const jobs = args

const INSTRUCTION = `You are segmenting a Greek academic document into a small set of structural sections so a
cleaner can KEEP the main text and REMOVE the scaffolding for language-model training. You mark ONLY TWO
kinds of section: TABLE OF CONTENTS and BIBLIOGRAPHY. Mark NOTHING else — everything you do not mark is
main text (the front-matter, body, and appendix are derived afterwards from where the ToC and bibliography
sit). Read the whole document for context before deciding (ToC is near the front, bibliography near the
back). Use the document's TRUE line numbers (the "L#####: " prefix) for start_line/end_line.

══════════ PRECISION OVERRIDES EVERYTHING ══════════
Deleting main text corrupts the corpus; leaving a stray reference line in is cheap and reversible. Therefore:
 • boundaries must be EXACT, to the line;
 • when unsure whether an EDGE LINE belongs to a section or to the surrounding text → leave it OUT
   (under-capture is safe; over-capture deletes prose);
 • when unsure whether a whole BLOCK is really a ToC/bibliography or a look-alike → do NOT mark it.

══════════ 1) TABLE OF CONTENTS  (kind="table_of_contents") ══════════
A navigational listing of the document's CHAPTERS / SECTIONS with page numbers and/or dotted leaders.
Headers: Περιεχόμενα, Πίνακας Περιεχομένων, Contents, Table of Contents (include the header line).
These LOOK like a ToC (aligned entries, page numbers, dotted leaders) but are NOT — leave them as MAIN TEXT,
do NOT mark them:
 • Συντομογραφίες / Αρκτικόλεξα / Συντομεύσεις / Abbreviations / Glossary / Γλωσσάρι  (abbreviation & glossary lists);
 • Κατάλογος Εικόνων / Σχημάτων / Πινάκων / List of Figures / List of Tables;
 • an alphabetical Index / Ευρετήριο (it is end-matter; leave it — it falls in the derived appendix).
There is normally ONE table of contents. It often comes right before such lists — end the ToC at its last
chapter/section entry and do NOT let it swallow the abbreviations/figure list that follows.

══════════ 2) BIBLIOGRAPHY  (kind="bibliography") ══════════
A list of bibliographic reference ENTRIES. Headers: Βιβλιογραφία, Βιβλιογραφικές Αναφορές, Αναφορές,
References, Πηγές. A reference entry = author(s) + year/title + venue/publisher/pages (it may wrap across
2-3 physical lines — all part of the one entry; entries may be numbered "[1]"/"1." or unnumbered).
 • The end-of-document list is a bibliography. A reference list at the END OF A CHAPTER is ALSO one — set
   is_chapter_bibliography=true; a book may contain several, mark each separately.
 • Fold sub-lists under one Βιβλιογραφία — Ελληνόγλωσση / Ξενόγλωσση, Δικτυογραφία (web sources), archival /
   primary sources (Πηγές) — into ONE bibliography span.
 • AUTHOR'S OWN publications: a list of the author's own works / CV (Δημοσιεύσεις, Επιστημονικές Δημοσιεύσεις,
   Ανακοινώσεις σε Συνέδρια, "Publications arising from this thesis", usually at a thesis end) is NOT the
   document's bibliography — still mark it bibliography but set is_authors_own_works=true.
 • Mark the region even if extraction shattered the entries (fragmented / one-token-per-line / noisy OCR) —
   it is still a bibliography.
These RESEMBLE a bibliography but are NOT — leave them as MAIN TEXT, do NOT mark them:
 • footnotes / endnotes and per-page citation notes at the bottom of a content page;
 • inline citations inside running prose ("(Παπαδόπουλος 2019)", a bare "[12]" within a sentence);
 • a data table or a table of names/values that merely resembles entries;
 • the "ΒΙΒΛΙΟΓΡΑΦΙΑ ........ 250" line that appears INSIDE the table of contents — that is a ToC entry, NOT
   the bibliography itself.

══════════ 3) BOUNDARY DISCIPLINE  (handle this carefully — it is what we most often get wrong) ══════════
Give each section's exact start_line/end_line (true L#####):
 • BIBLIOGRAPHY START = the header line if present, otherwise the FIRST reference entry. Do NOT include the
   figures, figure/table captions, equations, markdown tables, or prose that PRECEDE it — EVEN when they sit
   between a chapter heading and the first entry. (Frequent trap: a chapter ends with a figure + caption or a
   results table, then the references start; the figure/table is NOT part of the bibliography.)
 • BIBLIOGRAPHY END = the LAST reference entry. Do NOT run on into a following chapter heading, an appendix,
   acknowledgements, or any prose.
 • TOC START = its header (or first entry) — not the title page above it. TOC END = the last chapter/section
   entry — not the abbreviations/figure list or body that follows.
 • NEVER cut a wrapped multi-line entry in half — include all of its lines. Do NOT extend a span over a
   trailing blank line or a stray page-number/running-header line.
 • If a real block of main text interrupts a section, emit TWO separate spans rather than one that swallows
   the interruption.

Do NOT mark as scaffolding: the abstract / Περίληψη (it is content → main text), acknowledgements, preface
(Πρόλογος), dedication, in-body figures/tables/equations, footnotes, or inline citations. If the document
has no ToC and no bibliography, return an empty sections list.`

phase('Annotate')
const summaries = []
// SAFETY: pass only NOT-yet-done jobs (use pending_args.py to filter) so a stop+restart never re-spends.
// SAFETY: hard budget guard — set a turn budget ("+Nk") and this stops spawning before it's exhausted.
const RESERVE = 300_000   // stop when fewer than this many output tokens remain
for (let i = 0; i < jobs.length; i++) {
  if (budget.total && budget.remaining() < RESERVE) {
    log(`BUDGET GUARD: ${Math.round(budget.remaining() / 1000)}k left < reserve — stopping at ${i}/${jobs.length} (resume later, done jobs are skipped)`)
    break
  }
  const { batch, out } = jobs[i]
  const s = await agent(
    `${INSTRUCTION}

INPUT: read the JSON file at this exact path with the Read tool:
  ${batch}
It is a one-element array [ { "doc_id", "source", "n_lines", "mode", "text_numbered" } ]. The field
"text_numbered" is the document; each non-blank line is prefixed "L#####: " with the document's TRUE line
numbers (blank lines are omitted). If "mode" is "front+tail", the middle main-text was elided and is marked
with a "[ ... N main-text lines elided ... ]" line — only the front and tail are shown, which is enough to
find the ToC and the bibliography.

OUTPUT: write a JSON file to this exact path with the Write tool:
  ${out}
The file must contain ONLY this JSON object (valid JSON, no markdown fences, no commentary):
{ "doc_id": <from input>, "source": <from input>, "n_lines": <from input>, "mode": <from input>,
  "doc_type": "article" | "book" | "unknown",
  "sections": [ { "kind": "table_of_contents" | "bibliography",
                  "start_line": <int, true L#####>, "end_line": <int, true L#####>,
                  "is_chapter_bibliography": <bool>, "is_authors_own_works": <bool>,
                  "n_entries": <int, reference entries; 0 for ToC>, "has_header": <bool>,
                  "confidence": "high" | "medium" | "low" } ] }

After writing the file, reply with ONE line only: "<doc_id>: <doc_type> — <K> sections (toc=<n_toc>, bib=<n_bib>)".`,
    { label: `doc:${i}`, phase: 'Annotate' }
  )
  summaries.push(s)
  log(`${i + 1}/${jobs.length} done`)
}
return summaries
