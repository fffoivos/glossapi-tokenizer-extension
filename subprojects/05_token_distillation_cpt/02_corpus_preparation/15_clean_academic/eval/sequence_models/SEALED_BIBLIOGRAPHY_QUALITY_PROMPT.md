You are independently screening extraction usability for a sealed bibliography
evaluation set. Review only the flagged documents in the envelope. You receive
source-text diagnostics, canonical GlossAPI Rust noise metrics, and a
deterministic head/middle/tail sample. You receive no bibliography predictions
or labels.

Return JSON only and satisfy the supplied schema. Copy `reviewer_id` exactly to
`reviewer` and use schema version `bibliography-sealed-quality-response-v1`.
Return each `document_alias` exactly once.

Choose `UNUSABLE` only when extraction corruption would make line-level
bibliography evaluation misleading: extreme one-token fragmentation,
character-spaced OCR, pervasive glyph placeholders/mojibake, or symbol-dominated
text. Choose `KEEP` for imperfect but readable material, unusual typography,
math, tables, multilingual text, or localized OCR defects. The automatic flags
nominate cases; they are not decisions. Give a confidence and concise reasons.
