# Bibliography positional-role implementation — 2026-07-14

## Current outcome

The first implementation phase is complete. It does not fit a new classifier
or authorize text removal. It establishes the corrected role contract, the
lossless intra-line positional representation, and the complete prediction-
blind role-profile/review input needed before model fitting.

Authoritative implementation commits:

- `37bac1e` — role contract, overlay validation, sparse positional encoding,
  prediction-blind profiler, tests, and Clariden launcher;
- `ecc8799` — separate complete-profile coverage from full-document review
  eligibility; and
- `1ca4d75` — bound large review inputs with overlapping line/character chunks.

The frozen v1 entry table and B0/H0 decoder were not modified.

## Implemented artifacts

### Role contract and overlay safety

- `bibliography_role_contract_v1.json`
  - defines `ENTRY_ANCHOR`, `CONTINUATION`, `FILLER`, `HEADER`, `SUBHEADER`,
    `NON_BIB`, and `UNKNOWN`;
  - only `ENTRY_ANCHOR` may seed a block;
  - model-only labels remain `PROVISIONAL` and cannot become evaluation truth.
- `bibliography_role_dataset.py`
  - validates overlays against immutable document/work/line identity,
    coordinate, text SHA-256, and original regional label;
  - fails closed on stale or mismatched annotations;
  - derives the one-vs-rest entry-anchor target and the planned mask-in-block
    ablation without rewriting source labels.
- `bibliography_role_review.schema.json`
  - binds contextual reviewer output to the role and boundary inventories.

### Intra-line position encoding

`bibliography_positional_features.py` reuses the existing ownership-resolved
NFKC match spans. It provides:

- exact count/span parity checks for all 35 features;
- sparse feature/start/end events;
- normalized first, last, mean-centre, and union-coverage summaries;
- semantic-match complement intervals after excluding broad punctuation,
  prose-lead, and full-table-row detectors;
- unmatched prefix, suffix, total, longest-run, centre, run-count, and mean-run
  summaries;
- five positional nonmatch channels: letters, digits, whitespace,
  punctuation/symbols, and other/OCR;
- 8/16/32/64-bin rasterization plus explicit coordinate channels; and
- the exact 77-scalar count/gap and 210-scalar P1 representations from the
  plan.

The implementation stores/operates on sparse events. It does not materialize
the multi-gigabyte dense CNN map.

### Prediction-blind role profiling

`bibliography_role_profile.py`:

- profiles every training line and every silver bibliography block;
- records length, deterministic role, exact header status, feature-family
  presence, semantic match coverage, and unmatched-gap diagnostics;
- selects blocks without loading model predictions or outcomes;
- keeps complete profiling over all coverage modes while restricting the
  initial contextual-review sample to full documents;
- selects 20 work-distinct blocks per source; and
- writes two separate review artifacts:
  - a blinded packet with raw text and document position only; and
  - a provenance manifest containing nomination strata, source labels, and
    diagnostics that must not be passed to first-round reviewers.

Review inputs are bounded to 80 lines and 20,000 characters. Large blocks are
split into chunks with five overlapping context lines. The 60 selected blocks
therefore produce 104 review chunks.

## Authoritative Clariden run

Job: `2760129` (`COMPLETED`, exit `0:0`, elapsed `00:01:07`)

Code commit:

```text
1ca4d752b11a34935acc233f842564a50edf7109
```

Pinned input SHA-256:

```text
f18ef6bf3061d932ae0aaeb2349392a2e590f2778e3205cf7cbcb5c79dffa7c0
```

Remote result root:

```text
/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/experiments/bib_role_position_20260714/role_profile_1ca4d75_r3
```

Verified inventory:

- 1,118 training documents;
- 939,014 lines;
- 708 `full_document` and 410 `annotated_windows` documents;
- 1,632 silver bibliography blocks;
- source document counts: 373 Greek PhD, 369 Kallipos, 376 OpenArchives;
- source block counts: 634 Greek PhD, 591 Kallipos, 407 OpenArchives.

Block diagnostic inventory:

- 1,514 conventional-dense blocks;
- 573 blocks with an exact header candidate;
- 700 blocks containing long/wrapped lines;
- 384 sparse-internal blocks;
- 1,049 heterogeneous blocks.

These categories overlap by design.

Review selection:

- 60 work-distinct full-document blocks: 20 per source;
- 104 bounded review chunks;
- largest chunk: 80 lines and 19,989 characters;
- 26 selected blocks contain exact headers;
- 27 contain sparse-internal patterns;
- 36 contain long/wrapped lines;
- 54 contain conventional-dense evidence; and
- 51 are heterogeneous.

The receipt records both `prediction_inputs_loaded=false` and
`model_outcomes_used_for_selection=false`. Output hashes were verified after
the job. Important remote-only artifact:

```text
line_profile.jsonl
bytes: 757173125
sha256: bfae93390e400597ba9d9307b26a26d83382067009e86282328a3395fdab9e16
```

This large file remains on Clariden; it was not transferred to the MacBook.

## Local review handoff

Current verified local root:

```text
/Users/foivoskarounos-zamparloukos/presentations/train-apertus-with-glossapi/bibliography-role-pilot-20260714
```

- `blind/role_review_packet.blind.json`
  - 104 first-round contextual-review chunks;
  - SHA-256
    `aac8fedbb0b28b576f70acdd020d00b2eca977be0a43077ac8a966eca3d1afbd`.
- `provenance/selection_manifest.provenance.json`
  - source labels, strata, and line diagnostics; never pass this to a first
    reviewer.
- `provenance/block_inventory.jsonl`
  - all 1,632 block summaries.
- `receipts/receipt.json` and `receipts/profile.summary.json`
  - authoritative run receipt and summary.

The unchunked all-coverage r2 packet is retained under `superseded-r2/` only
for provenance. It is not the current review input.

## Corrected execution history

- Job `2760085` succeeded but profiled only 708 full documents / 368,421 lines.
  It exposed that profiling and review-eligibility coverage had been conflated
  and is superseded.
- Job `2760114` succeeded over all 1,118 documents / 939,014 lines, but its
  largest single review case was 1,027 lines / 128,276 characters. It is
  superseded by the bounded r3 packet.
- Job `2760129` is the authoritative complete and chunked result.

## Verification performed

- Python bytecode compilation passed for all new modules.
- Direct invariants passed for:
  - feature count/span parity;
  - raster dimensions and coordinate channels;
  - count/gap and positional-summary dimensions;
  - trusted/provisional entry-anchor target behavior;
  - source-bound overlay validation;
  - work-distinct review selection; and
  - chunk line/character bounds and overlap.
- An end-to-end synthetic profile CLI run passed, including blinding and
  multi-chunk output.
- Clariden verified the exact code-bundle hashes before execution.
- The authoritative receipt and transferred review/provenance artifacts were
  independently hash-checked.

The Mac system Python currently lacks `pytest`, so the new pytest-form tests in
`tests/test_bibliography_role_position.py` were exercised through equivalent
direct checks rather than installing a new local test environment.

## Next execution step

Do not fit P0-P3 yet. First:

1. build the schema-bound Codex contextual-review executor;
2. run a small three-block calibration sample to check role definitions,
   response completeness, and token cost before spending on all 104 chunks;
3. run two independent review passes only after the calibration passes;
4. adjudicate agreements/disagreements into the provenance-bound overlay;
5. jointly review the initial 30 blocks (10/source); and
6. only then materialize the corrected positional table and fit P0/P0D.
