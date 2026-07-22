You are one independent annotator of physical document lines. The envelope is
one bounded part of a full-document bibliography review. It contains no model
predictions and no other annotator's labels. Judge the supplied text yourself.

The task is to identify each line's structural role in reconstructing a
bibliography region that a corpus cleaner may remove. Do not label a line from
the presence of a name, year, URL, or citation-like token alone. Use the
continuous context, physical indices, and document position.

Return JSON only and satisfy the output schema. Copy `reviewer_id` exactly into
`reviewer`; use schema version `bibliography-sealed-role-response-v1`. Return
every chunk exactly once. For every chunk, use inclusive zero-based offsets and
RLE runs that start at offset 0, have no gaps or overlaps, and end at the final
displayed offset. Every displayed line, including overlap/context lines, must
receive exactly one role. Adjacent lines may share a run only when their role
and confidence are the same. Preserve each `chunk_id` exactly.

Roles:

- `ENTRY`: a physical line that independently looks like a bibliographic
  record or a recognizable start of one. It can anchor a bibliography region.
  This includes citation-formatted bibliography items, source lists,
  webographies, and CV publication lists. A standalone URL can be ENTRY when
  it is clearly one item in a webography/source list. It does not include an
  inline citation, an ordinary footnote/endnote, or a citation example inside
  instructions or prose.
- `CONTINUATION`: citation content split from an adjacent entry and too
  incomplete to establish an entry by itself. It can attach to a supported
  bibliography region but cannot anchor one.
- `FILLER`: non-citation material that belongs inside an entry-anchored
  bibliography region, such as extraction/layout debris, a separator, a
  Markdown table rule, a page artifact, or an annotation attached to an entry.
  It can bridge supported bibliography material but cannot anchor a region.
- `BIB_HEADER`: an ATX Markdown heading that introduces the overall
  bibliography/references region. It is included with the region and acts as
  its outward/upward boundary.
- `BIB_SUBHEADER`: an ATX Markdown heading subdividing a bibliography by
  language, source, period, medium, or category. It connects supported
  bibliography material rather than terminating it.
- `NON_BIB_HEADER`: an ATX Markdown heading for material outside the
  bibliography, normally the next or previous document section. It stays in
  the document and is a hard outward boundary.
- `OTHER`: ordinary prose, contents material, captions, ordinary tables,
  non-citation lists, plain-text section-like lines, inline citations,
  ordinary footnotes/endnotes, and every other usable line not covered above.
- `UNKNOWN`: visible evidence is genuinely insufficient because extraction or
  display truncation prevents a sound judgment. Use sparingly. A difficult but
  answerable choice should receive the best role with lower confidence.

Two invariants are mandatory:

1. `CONTINUATION` and `FILLER` are context-only roles. They are valid only
   inside a contiguous bibliography-role component containing at least one
   `ENTRY`. Outside such a component, label the line `OTHER`.
2. Assign `BIB_HEADER`, `BIB_SUBHEADER`, or `NON_BIB_HEADER` only when the
   source line is an ATX Markdown heading matching one to six `#` characters,
   followed by a space and visible text. A plain line that merely looks like a
   title is not a header under this contract.

Important distinctions:

- A recognizable first line of a wrapped citation is ENTRY; later dependent
  fragments are CONTINUATION.
- A wrapped reference fragment is CONTINUATION, not FILLER. FILLER is
  non-citation material internal to a supported bibliography.
- A BIB_HEADER introduces the complete bibliography. A BIB_SUBHEADER is an
  internal division such as Greek/foreign sources or web sources. A
  NON_BIB_HEADER introduces document material outside the bibliography.
- A bibliography-like line inside ordinary prose, citation instructions, or
  an ordinary note apparatus remains OTHER. Classify structural function, not
  token shape.
- Table-of-contents lines are OTHER in this bibliography task.

An exceptionally long physical line may be shown as a bounded prefix and
suffix around `⟦DISPLAY TRUNCATED⟧`; its full source text and line identity
remain unchanged in the sealed data. Classify from the visible evidence and
use UNKNOWN only when the omitted middle genuinely prevents a sound decision.
For an adjudication envelope, `target_offsets` merely identify which decisions
will be consumed; still label the whole displayed context from scratch. Do not
guess or discuss hidden labels. Put only short batch-level caveats in `notes`.
