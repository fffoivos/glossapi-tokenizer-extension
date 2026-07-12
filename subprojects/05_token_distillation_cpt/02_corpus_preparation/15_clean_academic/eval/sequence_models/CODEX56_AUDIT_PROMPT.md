You are performing a blinded structural-text audit for Greek pretraining data.

For every request, classify the line at `target_abs_idx`: decide whether that
target line belongs to a removable bibliography block, belongs to a removable
table-of-contents block, belongs to neither, or has genuinely insufficient
evidence. Judge the target using the complete supplied local context, not a
keyword or isolated citation. Ordinary prose, footnotes, CV publication lists,
legal-body enumerations, statistical tables, and ambiguous single references
must be kept.

The packet never joins the two sides of an unrepresented annotated-window
interval. Respect `context_coverage`; do not infer missing lines or extend a
span beyond the supplied contiguous context.

Return exactly one response per request using the supplied JSON schema. Copy
`request_id`, `request_sha256`, and `reviewer_model` exactly. Use absolute line
indices from the request. A BIB or TOC label requires
`should_remove=true` and a precise inclusive span that contains
`target_abs_idx`. OTHER or UNKNOWN requires `should_remove=false` and null span
bounds. Do not infer or mention any previous label or model prediction; none is
provided. Do not use tools or external information.

Requests follow as JSON:
