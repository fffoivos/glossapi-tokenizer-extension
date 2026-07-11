# Structural-span production and safety gate

This is the production path from the exact Stage50 post-source/post-PII text to
the optional structural-last Stage58 deletion. It does **not** require or claim
a human-gold corpus. Model-selection evidence remains explicitly LLM-silver;
the only new human work is a targeted review of 100 predicted deletions.

No job has been launched by adding this path.

## Execution boundary

The three stages are CPU-only Clariden jobs:

1. `52-structural-detect` streams every Stage50 `stable_uid` and exact `text`
   that is explicitly eligible for structural review into the existing Rust
   `reference_detect --mode structure-spans`. Eligibility requires both a
   cleaning profile in the frozen `allowed_apply_profiles` policy and row-level
   `structural_policy=apply_after_review`; academic sources marked `shadow` are
   excluded and receipt-counted. It writes
   one immutable raw sidecar set per Stage50 Parquet shard and resumes from
   verified completed shard receipts.
2. `53-structural-review-packet` reopens the exact Stage50 text, verifies every
   Rust offset as a Python Unicode-code-point slice, and chooses exactly 100
   source-balanced, highest-risk predicted deletions: exactly 50 ToC and 50
   bibliography cases, balanced by source within each head. The risk score is only a
   review-prioritisation heuristic; it is never a label or safety metric. This
   stage stops at a manual boundary and creates no adjudications.
3. `54-structural-promote` validates a separately completed manual audit,
   builds and validates `academic_structural_model_receipt_v1`, then rebinds
   the already-computed raw predictions to that final receipt hash. It never
   reruns the detector after seeing audit results.

The raw run manifest binds all of the following:

- the Stage50 cleaning manifest and its exact Parquet inventory;
- a per-document `stable_uid`, source, Stage50 text SHA-256, and code-point
  count;
- every raw Rust counter and prediction;
- the detector binary and immutable Clariden build receipt;
- the passed Python/Rust parity receipt;
- the bibliography/ToC model JSON, smoother, sequence configuration, and Rust
  model source files.

Rust offsets are Unicode scalar-value offsets. Python strings index the same
code-point units. Before an audit packet can be produced, every predicted span
must have matching whole-document character count, line boundaries, and the
same first-line trigger under Python slicing. A focused test includes astral
and polytonic characters so byte offsets cannot accidentally pass.

Raw ToC/bibliography overlaps are reported but never silently resolved. Any
overlap blocks the passed model receipt and final rebind, so Stage58 cannot see
an ambiguous deletion inventory.

Stage53 also writes `token_loss_report.json` and a receipt-bound per-document
Parquet before the manual decision. It uses the exact Stage50 tokenizer on
whole-document bibliography-only, ToC-only, and union counterfactuals. The
report separates all routed documents from training-eligible documents,
summarises every source, and expresses each loss against the full Stage50
training-token total. It does not estimate loss from isolated span tokens and
does not mutate corpus text. Cross-head overlaps are unioned only for this
counterfactual measurement; they still block promotion.

## Dry-run submission

`clariden/submit.sh` is dry-run first. Detection additionally needs the
immutable detector artifacts:

```bash
export PIPELINE_RUN_ID=<the-existing-phase04-run>
export REFERENCE_BIN=<detector-build-dir>/reference_detect
export DETECTOR_BUILD_RECEIPT=<detector-build-dir>/build_receipt.json

clariden/submit.sh structural-stage50-detect
clariden/submit.sh structural-review-packet
```

Use the chain helper, when available in the integrated submit script, to make
the second job depend on the first. It must stop at:

```text
stages/53-structural-review-packet/review_cases.jsonl
```

`CONFIRM_LAUNCH=1` is the normal explicit launch gate. These scripts request no
GPU/GRES and recheck the CPU-only Slurm request on-node.

A passed parity receipt may be supplied to Stage52 and will then be bound to
the raw run, but it is deliberately optional for audit-only detection. This
lets the 100-case safety review proceed even though the raw joint LLM-silver
artifact is currently absent. Stage54 promotion always requires the passed
joint-head parity receipt.

## Manual 100-case audit

The reviewer writes one JSONL adjudication per packet case. The required fields
are defined by
`schemas/structural_false_deletion_annotation.schema.json`:

```json
{
  "case_id": "<copied from packet>",
  "review_context_sha256": "<copied from packet>",
  "decision": "structural_only",
  "running_prose_chars_removed": 0,
  "main_text_chars_removed": 0,
  "catastrophic_document_deletion": false,
  "reviewer_notes": ""
}
```

For mixed or unsafe cases, the reviewer supplies the actual affected character
counts and notes; the validator never guesses them. The completed manual
receipt follows `schemas/structural_manual_audit_receipt.schema.json`, binds the
exact packet-manifest hash and annotations file receipt, states
`annotation_method=manual`, and must state
`automatic_adjudication_used=false`.

The validator recomputes all three configured safety metrics from those 100
manual decisions and the bound packet text. It does not trust declared metrics
and it does not turn LLM-silver labels into safety evidence.

Promotion is submitted separately:

```bash
export STRUCTURAL_MANUAL_AUDIT_RECEIPT=<manual-audit-receipt.json>
export STRUCTURAL_AUDIT_ANNOTATIONS=<manual-adjudications.jsonl>
export STRUCTURAL_SILVER_RECEIPT=<joint-bib-toc-LLM-silver-receipt.json>
export STRUCTURAL_SILVER_SPLIT=<joint-bib-toc-split-manifest.json>
export STRUCTURAL_PARITY_RECEIPT=<passed-joint-head-rust-parity-receipt.json>
clariden/submit.sh structural-promote
```

The final JSONL rows match `schemas/structural_span.schema.json` and bind the
exact Stage50 text plus the SHA-256 of the final model receipt. The receipt does
not hash the final spans; the spans hash the already-final receipt. This
one-way order avoids circular receipt hashing:

```text
raw predictions -> manual audit -> model receipt -> final bound spans
```

## Stage58 remains fail-closed

The ordinary cleaning chains stop before Stage58. Finalization is a separate,
irreversible operator choice:

- `chain-finalize-noop` requires `CONFIRM_STRUCTURAL_NOOP=1` and records the
  deterministic no-op copy;
- `chain-finalize-promoted` requests application and requires all of the
  following:

- a passed Stage54 model receipt;
- final receipt-bound, non-overlapping spans from that same Stage54 run;
- a manual confirmation matching the exact Stage54 model-receipt SHA-256;
- a tracked cleaning policy explicitly approved with both structural heads
  enabled and frozen before Stage10 of that run;
- the existing Stage58 receipt, artifact, split, audit-metric, and source
  admission checks.

The promoted path fails if any gate is absent; it never silently degrades to a
no-op. Stage58 freezes `no_op` or `apply` plus the exact promoted receipt path
and SHA-256 in `structural_finalization_request.json`, records the same binding
in the application decision, and revalidates it before any incomplete or
completed-stage resume can return. A changed choice or receipt requires a new
pipeline run.

The current tracked cleaning policy is still `audit_only` with both structural
materialisation flags disabled, so this CPT run uses the no-op path. Merely
producing raw predictions or even a clean manual audit cannot silently enable
deletions, and policy must not be edited mid-run. The no-op still requires the
separate explicit finalization confirmation; it is not queued by a cleaning
chain.

## Known remaining evidence gap

The reconstructable SPAN LLM-silver set supervises bibliography only. The
historically named `STRUCT_2K_gold.jsonl` was also LLM-silver, not human gold,
and its raw joint bibliography/ToC artifact is currently absent. Stage54
therefore rejects the bibliography-only silver receipt with an explicit error;
it will not invent ToC coverage. Recovering/importing that raw joint artifact
is still required for a two-head model receipt. No 2,000-line human annotation
effort is requested or implied.
