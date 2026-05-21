# Review integration — 2026-05-18

Tracks how each finding from the review pass on 2026-05-18 was
addressed in `PLAN.md` and `README.md`. The audit has not yet
been executed; these are edits to the plan only.

## Findings and resolutions

### Finding 1 (HIGH) — "Released source" ≠ "seen by Apertus"

**Reviewer**: The plan dedups C3 against the full released FW2-HQ
Greek slice (4.35 M docs), not the subset actually consumed under
Apertus's curriculum (~10–33 % of release). For CPT recipe
purposes this over-removes C3 docs and understates the fresh
Greek budget.

**Status**: **INTEGRATED.**

**Changes**:
- `PLAN.md` §2.1 rewritten to explicitly report **two-axis**
  overlap: `source-universe` (conservative, against full release)
  and `consumed-estimate` (CPT-relevant, against
  Apertus-sampler-reconstructed subset). The sampler math is
  imported from `docs/APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md`
  §5.2.
- `PLAN.md` §5 outputs gained
  `artifacts/consumed_estimate/<a_source>_consumed_universe.parquet`
  — the Apertus-stage-weighted subset for each source.
- `PLAN.md` §9 success criterion 3 now requires both numbers in
  `REPORT.md`.

### Finding 2 (HIGH) — Held-out contamination uses md5 only

**Reviewer**: Given the plan correctly worries about normalization
drift across pipelines, the held-out leakage check should not use
only md5 — it should run the full ladder (strict + relaxed exact
+ MinHash near-dup + sentence-level for EuroParl).

**Status**: **INTEGRATED.**

**Changes**:
- `PLAN.md` §3.5 rewritten as "Held-out contamination check —
  full ladder." Per-stage detection (`strict_exact`,
  `relaxed_exact`, `near_signatures`, `near_clusters`) + sentence-
  level matching against EuroParl. Output now
  `holdout_contamination.parquet` (was JSON; was md5-only).
- `PLAN.md` §9 success criterion 2 reflects the full ladder.

### Finding 3 (HIGH/MEDIUM) — MinHash underspecified + storage too low

**Reviewer**: The plan said "256-permutation MinHash" without
specifying token vs char shingles, shingle size, short-doc
behavior, or normalization. The existing
`glossapi_corpus_cli/text_dedup.py` already has versioned defaults
(128 perms, token 5-shingles, threshold 0.85, skip-short-doc <20
tokens). Storage estimate of "<500 MB per source" and "~5 GB final"
is too low for 20 M docs × 256 sigs.

**Status**: **INTEGRATED.**

**Changes**:
- `PLAN.md` §3.1 reframed: the audit **drives the existing
  `text_dedup.py` pipeline in cross-corpus mode**, not a
  reimplementation. The methodology pins to a specific commit of
  that file (recorded in run manifest) and adopts its defaults
  verbatim. Verified defaults from the actual code:
  - `DEFAULT_NUM_PERM = 128` (not 256 — corrected)
  - `DEFAULT_SHINGLE_MODE = "token"`
  - `DEFAULT_SHINGLE_SIZE = 5`
  - `DEFAULT_NEAR_THRESHOLD = 0.85`
  - `SHORT_DOC_TOKEN_THRESHOLD = 20`
  - Greek diacritic policy `preserve`
  - blake3 (not md5) for shingle hashes
- `PLAN.md` §3.2 now describes the pipeline stages used (`strict_exact`,
  `relaxed_exact`, `near_signatures`, `near_candidates`,
  `near_clusters`) — these are stage names from the existing code,
  not new design.
- `PLAN.md` §6.2 added (new section): honest storage budget. Raw
  sigs ~20 GB, LSH bands ~5 GB, intermediate ~75 GB during run,
  ~15 GB final after sig cleanup. Strategy: discard full sigs
  post-LSH-join.

### Finding 4 (MEDIUM) — C3 source inventory inconsistency

**Reviewer**: Plan said 19 GlossAPI sub-datasets in §2.2 and "4 × 21
= 84 pairs" in §8. But the canonical
`source_dataset_summary.parquet` has 18 source rows (17 GlossAPI +
1 HPLT); the narrative doc lists 17 named GlossAPI sources too.
Build the audit source list from the actual train manifest, not
prose.

**Status**: **INTEGRATED.**

**Changes**:
- `PLAN.md` §2.2 rewritten: source list is now explicitly
  manifest-derived. The narrative inventory is labelled "sanity-
  check anchor only." Numbers corrected to 17 GlossAPI + 1 HPLT =
  18 sources, 4 × 18 = 72 pairs.
- `PLAN.md` §8 Risk 5 corrected from "84 pairs" to "72 pairs"
  with explicit note that the earlier number was wrong.

### Finding 5 (MEDIUM) — Sentence overlap missing action rule

**Reviewer**: Plan handles EuroParl by sentence-splitting C3 docs,
which is right, but doesn't define what to DO when matches are
found. Drop the whole doc? Mark partial contamination? Compute
overlap ratio?

**Status**: **INTEGRATED.**

**Changes**:
- `PLAN.md` §3.6 added (new section): per-doc `overlap_ratio =
  matched_chars / total_doc_chars`. Three-tier action rule:
  - `drop` ≥ 0.30 (treat as already-seen)
  - `partial` 0.05-0.30 (treat as half-weight, flag for review)
  - `trace` < 0.05 (treat as fresh)
- Thresholds configurable; raw `overlap_ratio` recorded per-doc
  for post-hoc re-classification without re-running the audit.
- `PLAN.md` §5 outputs gained `per_c3_doc_overlap.parquet` with
  the tier classification.

### Finding 6 (MEDIUM/LOW) — Long-context Greek skip needs framing

**Reviewer**: Excluding FineWeb-Long + Institutional Books is fine
for first pass, but `REPORT.md` must scope itself as "main
pretraining/cooldown only," not "everything the model saw." The
Institutional Books slice has Greek long-tail data per
`docs/APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md`.

**Status**: **INTEGRATED.**

**Changes**:
- `PLAN.md` §10 (open design choices) — the long-context exclusion
  now requires REPORT.md to state scope explicitly as
  "Apertus main-pretraining + cooldown only; long-context phase
  excluded." Cited link to the upstream Greek-share doc.

### Finding 7 (LOW) — Deliverable format mismatch

**Reviewer**: README promises `.json` artifacts; PLAN describes
Parquet directories. Parquet is the right call; align README.

**Status**: **INTEGRATED.**

**Changes**:
- `README.md` Deliverables section rewritten with all artifacts as
  Parquet (zstd, matching `text_dedup.py`'s `PARQUET_COMPRESSION`
  default). Explicit per-stage and per-output file paths.

## Other concrete actionable additions (from reviewer's closing summary)

The reviewer's closing paragraph said: *"make the final output
C3-side actionable: per-source fresh rows, fresh chars/tokens,
partial-overlap flags, and a recommended drop/replay policy."*

**Status**: **INTEGRATED.**

**Changes**:
- `PLAN.md` §5 outputs gained `per_c3_source_actionable.parquet`
  — one row per C3 source with `fresh_rows`, `partial_rows`,
  `seen_rows`, `fresh_chars`, `fresh_apertus_tokens`, and
  `recommended_action ∈ {include_full, include_half_weight,
  replay_only}`.
- `PLAN.md` §9 success criterion 4 elevates this artifact to a
  gating output for the audit.

## External validation noted

Reviewer confirmed via HF that `epfml/FineWeb2-HQ` and its
`ell_Grek` subset exist as referenced. Source identifiers validated;
audit can rely on them.

## Net effect on the plan

The audit's measurement contract is now tighter:
- Detection: drives versioned existing code, not new
  reimplementation. Pinned to a `text_dedup.py` commit.
- Reporting: two-axis (source-universe + consumed-estimate), not
  conflated.
- Held-out check: full ladder, not md5-only.
- C3-side actionable: `per_c3_source_actionable.parquet` with
  drop/half-weight/full-weight tiers and recommended actions.
- Source list: manifest-derived (17 GlossAPI + 1 HPLT = 18), not
  prose.
- Long-context scope: explicitly stated in REPORT.md.
- Storage: honest budget (~75 GB intermediate, ~15 GB final), not
  the earlier "<10 GB" understatement.
- Deliverables: Parquet everywhere; README + PLAN aligned.

No new evidence was added; no conclusions changed; the plan
remains unexecuted. The integrity of the measurement contract is
improved.
