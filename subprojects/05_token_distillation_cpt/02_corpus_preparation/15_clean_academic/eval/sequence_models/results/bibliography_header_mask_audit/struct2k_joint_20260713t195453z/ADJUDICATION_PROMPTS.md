# Header adjudication prompts

Both calls used `codex exec`, model `gpt-5.6-terra`, an ephemeral session,
read-only sandboxing, and the tracked
`bibliography_header_adjudication.schema.json` response schema. The complete
audit packet was appended as stdin.

## Reviewer A

> Act as independent bibliography-header adjudicator A. Do not run tools.
> Review every case in the appended JSON using the target text and its ordered
> context. ENTRY means a citation entry or any continuation/fragment belonging
> to an entry. BIB_HEADER means the overall bibliography/references heading.
> BIB_SUBHEADER means an internal grouping heading such as Greek-language,
> foreign-language, web sources, legislation, journals, or publications.
> OTHER_STRUCTURE means a structural line that is neither an entry nor
> bibliography header/subheader. UNCERTAIN only when context is genuinely
> insufficient. Candidate nomination and deterministic roles are untrusted
> hints, never labels. Return exactly one case per supplied candidate_id, in
> the supplied order. Set schema_version exactly
> bibliography-header-context-adjudication-v1 and reviewer to adjudicator_a.

## Reviewer B

> Act as independent bibliography-header adjudicator B. Do not run tools.
> Judge every case from scratch using the target line plus ordered context.
> Ignore nomination reasons and deterministic roles because they may be wrong.
> ENTRY includes complete citations, web references, citation-key lines, and
> wrapped or OCR-fragmented continuation lines. BIB_HEADER is only the overall
> references or bibliography title. BIB_SUBHEADER is an internal category
> dividing bibliography material, including language, source type,
> legislation, journals, or publication categories. OTHER_STRUCTURE is
> structural but not one of those. Use UNCERTAIN when evidence does not support
> a reliable distinction; do not force a header label merely because a line is
> short or starts a BIB-labelled block. Return one case for every candidate_id
> in the supplied order. Set schema_version exactly
> bibliography-header-context-adjudication-v1 and reviewer to adjudicator_b.

## Reproduce the summary

From the repository root:

```bash
PYTHONPATH="$PWD/subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval" \
  uv run python -m sequence_models.bibliography_header_adjudication \
  --packet outputs/bibliography-header-mask-audit-20260713t195453z/audit_sample.json \
  --review-a outputs/bibliography-header-mask-audit-20260713t195453z/review_a.json \
  --review-b outputs/bibliography-header-mask-audit-20260713t195453z/review_b.json
```
