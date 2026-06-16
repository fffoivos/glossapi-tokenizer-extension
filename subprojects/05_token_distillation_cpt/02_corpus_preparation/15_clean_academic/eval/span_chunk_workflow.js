export const meta = {
  name: 'span-chunk',
  description: 'Annotate one modest chunk of bibliography-span batches with Opus (rich metadata) — paced loop step',
  phases: [{ title: 'Annotate' }],
}
const KIND = ['end_of_document','end_of_chapter','subdivided_sublist','archival_primary_sources','web_sources','further_reading']
const STYLE = ['chicago_turabian','apa_harvard_authoryear','vancouver_numbered','footnote_humanities','iso690','mixed','none']
const LANG = ['greek','latin_foreign','mixed']
const SCRIPT = ['monotonic_greek','polytonic_greek','latin','mixed']
const SUBJ = ['humanities','social_science','stem','medical','law','theology','unknown']
const NOISE = ['clean','light','heavy']
const SPAN = { type: 'object',
  required: ['start_line','end_line','kind','citation_style','language','script','subject_register','noise_level','n_entries','has_header'],
  properties: {
    start_line: { type: 'integer' }, end_line: { type: 'integer' },
    kind: { type: 'string', enum: KIND }, citation_style: { type: 'string', enum: STYLE },
    language: { type: 'string', enum: LANG }, script: { type: 'string', enum: SCRIPT },
    subject_register: { type: 'string', enum: SUBJ }, noise_level: { type: 'string', enum: NOISE },
    n_entries: { type: 'integer' }, has_header: { type: 'boolean' },
  } }
const ANNOT = { type: 'object', required: ['annotations'], properties: {
  annotations: { type: 'array', items: {
    type: 'object', required: ['unit_id','has_bib','spans','confidence'], properties: {
      unit_id: { type: 'string' }, has_bib: { type: 'boolean' },
      spans: { type: 'array', items: SPAN }, confidence: { type: 'string', enum: ['high','medium','low'] },
    },
  } },
} }
const GUIDE = `You are an expert annotator building a bibliography-span dataset. Each unit is a WINDOW from a Greek academic document — consecutive lines prefixed "L#####: " with TRUE line numbers (Docling-extracted, GlossAPI-cleaned, so possibly noisy/OCR'd).

Mark EVERY contiguous BIBLIOGRAPHIC-LIST span — a run of reference ENTRIES (author + year + title + publisher/place/journal + pages). For each span give start_line/end_line (the L##### numbers of its first and last entry line) and this metadata:
 • kind: end_of_document · end_of_chapter · subdivided_sublist (Ελληνόγλωσση/Ξενόγλωσση under a parent) · archival_primary_sources (Αρχειακές πηγές) · web_sources (Δικτυογραφία) · further_reading.
 • citation_style: chicago_turabian · apa_harvard_authoryear · vancouver_numbered (numbered [1]) · footnote_humanities · iso690 · mixed · none.
 • language: greek · latin_foreign · mixed   ·   script: polytonic_greek if you see πολυτονικά (ᾶ ἡ ὧ) else monotonic_greek · latin · mixed.
 • subject_register: humanities · social_science · stem · medical · law · theology · unknown.
 • noise_level: clean · light · heavy (broken words, dropped diacritics, glued/garbled OCR inside the span).
 • n_entries: count of distinct reference entries.  has_header: true if a "Βιβλιογραφία/ΠΗΓΕΣ/References"-type header precedes it.

DO NOT mark: running prose, table-of-contents (dotted leaders + page numbers), a SINGLE inline/footnote citation (only contiguous LISTS of ≥2 entries), figure/table captions, equations, the author's own CV publication list. No bibliographic list → has_bib=false, spans=[]. A list may run to the window edge — mark to the last visible entry.`
const paths = Array.isArray(args) ? args : JSON.parse(args)
phase('Annotate')
// SEQUENTIAL on purpose: one Opus agent at a time keeps us under the tokens-per-minute
// server throttle that mass-failed the concurrent runs. Gentleness lives here, not in spacing.
const flat = []
let ok = 0
for (const p of paths) {
  const r = await agent(`${GUIDE}\n\nRead this batch file (JSON array of {unit_id, source, window, text_numbered}): ${p}\nAnnotate EVERY unit. Return {"annotations":[ one object per unit ]}.`,
    { label: `span:${p.split('/').pop()}`, phase: 'Annotate', schema: ANNOT, model: 'opus' })
  if (r && r.annotations) { flat.push(...r.annotations); ok++ }
}
log(`chunk (sequential): ${flat.length} window annotations from ${ok}/${paths.length} batches`)
return { annotations: flat, batches_ok: ok, batches_total: paths.length }
