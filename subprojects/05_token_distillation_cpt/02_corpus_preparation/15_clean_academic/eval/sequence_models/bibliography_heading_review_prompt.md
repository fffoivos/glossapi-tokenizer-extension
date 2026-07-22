You are reviewing heading-shaped lines from academic documents. Classify only
the marked target line, using the surrounding raw lines for context.

Labels:

- `BIB_HEADER`: the overall heading that introduces a bibliography/references
  region. It is the top boundary of that bibliography.
- `BIB_SUBHEADER`: an internal bibliography division such as primary sources,
  foreign-language bibliography, a period, or a source category. Bibliographic
  entries normally occur on both sides or continue below it.
- `NON_BIB_HEADER`: a real heading for non-bibliography material, such as a new
  chapter, appendix, acknowledgements, exercises, contents, or prose section.
- `NOT_HEADER`: the target is not a heading.
- `UNKNOWN`: extraction or context is insufficient.

Do not call a line a bibliography heading merely because it appears near a
citation. Distinguish a main bibliography heading from an internal subheading
principally by its relationship to the surrounding reference entries. Return
exactly the requested JSON schema and preserve every candidate ID exactly.
