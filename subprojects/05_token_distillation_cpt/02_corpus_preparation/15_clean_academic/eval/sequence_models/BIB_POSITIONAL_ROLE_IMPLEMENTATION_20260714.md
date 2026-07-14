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

## Dual-review and positional-table continuation

The dependency-safe continuation is implemented in commits:

- `b47e2bb` — schema-bound dual Codex executor, independent role/boundary
  adjudication, overlay v2, and tests;
- `4b5ee81` — prediction-blind-first 30-block human audit site;
- `6854ce4` — geometry-only positional table and cheap role target view;
- `ddfbde7` — nested work-grouped corrected P0/P0D/P1/P1G ladder;
- `05f274e` — 240-block source-balanced bootstrap and 30 zero-BIB controls;
- `1f9d866` — human decision importer/adjudicator;
- `e390944` — continue valid batches when another batch is rejected; and
- `b61630a` — explicit per-role/source readiness gates.

### Calibration result

The fixed three-source calibration is under:

```text
/Users/foivoskarounos-zamparloukos/presentations/train-apertus-with-glossapi/bibliography-role-pilot-20260714/calibration
```

It contains one Greek PhD, one Kallipos, and one OpenArchives block (107
unique displayed lines). Two ephemeral `gpt-5.6-sol` passes at high reasoning
covered every requested line. The independent executions agreed on all 107
roles, all entry-seed decisions, and all boundary flags. Calibration hashes
and the v2 overlay are preserved in `reviews/` and `adjudication/`.

The calibration did not contain `CONTINUATION` or `FILLER`; this is why it was
not accepted as a sufficient role dataset. Those roles occur in the full
packet and have explicit readiness gates.

### Full automatic pilot review

The two full executions use the same 104 blinded chunks but distinct pass IDs,
reviewer identities, and deterministic ordering. Their immutable batch roots
are:

```text
.../bibliography-role-pilot-20260714/full-review/pass-a/batches
.../bibliography-role-pilot-20260714/full-review/pass-b/batches
```

The executor fails closed on missing, duplicated, or invented identities. A
pure output permutation is canonicalized by the supplied IDs; a mistyped ID
is preserved as a rejection and retried without rerunning accepted batches.
Each Codex process receives only the blinded cases in an empty, read-only
workspace. Neither pass receives the other pass, the provenance manifest,
source labels, detector features, or model predictions.

After both passes finish, run `bibliography_role_adjudication.py` on the full
packet and its provenance manifest. It merges five-line chunk overlaps within
each reviewer before comparing reviewers. Role and boundary agreement are
resolved independently in `bibliography-role-overlay-v2`; disagreement in one
does not erase agreement in the other.

### One-time positional geometry

Clariden CPU job `2760592` completed in 57 seconds on commit `6854ce4` with
exit `0:0` and about 14 GB maximum resident memory. Authoritative root:

```text
/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/experiments/bib_role_position_20260714/positional_table_6854ce4_r1
```

The target-free extension contains:

- 939,014 aligned lines from 1,118 documents;
- 8,266,123 ownership-resolved detector spans;
- 52,500,315 typed non-match runs;
- exact count/span parity with the frozen v1 entry table;
- normalized 35 x 4 positional summaries; and
- seven unmatched-gap summaries.

The artifact is about 1.12 GB. Receipts, but not the bulk arrays, are mirrored
locally under `bibliography-role-pilot-20260714/positional-table/receipts`.
Role corrections therefore require only regenerating the small target view;
they do not trigger another all-line geometry pass.

### Remaining gate before fitting

Do not fit or report P0-P1G yet. The automatic full overlay must first exist,
then the 30-block audit site must be built and Foivos's decisions applied.
Only `AGREED_REVIEW` and `ADJUDICATED` roles can create entry targets.
Unreviewed silver `BIB` remains masked; silver `O`/`TOC` remains an entry
negative unless trusted review corrects it.
