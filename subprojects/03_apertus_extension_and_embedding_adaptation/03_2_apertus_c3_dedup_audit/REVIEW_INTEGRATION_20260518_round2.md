# Review integration — 2026-05-18 round 2

Second review pass on `PLAN.md`. Six findings, all integrated.

## Findings and resolutions

### Finding 1 (HIGH) — "consumed-estimate" overclaims document-level truth

**Reviewer**: The previous round renamed source-overlap to
"consumed-estimate" and said it answers "was this actually seen."
But the upstream Greek-share doc gives aggregate stage-weighted
token estimates with explicit uncertainty about exact mixing and
stage details — not a deterministic per-document consumed manifest.
The audit reintroduced the same truth-claim problem in a subtler
form.

**Status**: **INTEGRATED.**

**Changes** (PLAN.md §2.1):
- Renamed `consumed-estimate` → `consumed_exposure_estimate`
  throughout.
- Added an explicit honesty caveat block clarifying that this is
  **expected probability-weighted overlap**, treating Apertus's
  sampler as stochastic with reported marginal rates. NOT a
  per-document deterministic claim. We do not have Apertus's RNG
  seed, sample order, or per-document revision pins.
- The artifact `consumed_estimate/<a_source>_consumed_universe.parquet`
  is reframed: per-doc rows carry a `consumed_probability` column,
  not a binary `seen` flag.
- Added a "Note on naming" in §3.6 introducing **EXPOSED** as the
  tier-naming verb (not "SEEN"), to honour this uncertainty
  consistently in the action-rule artifact.

### Finding 2 (HIGH) — Stale md5 / 256-perm / JSON paths in execution sections

**Reviewer**: The methodology now correctly pins `text_dedup.py`
defaults (blake3 + 128-perm + Parquet), but the execution outline
in §6.4, §6.5, and §7 still said `text_md5`, `minhash256`,
`exact_overlap` paths, `<500 MB per source`, and
`holdout_contamination.json`. An implementer following the lower
half could build the wrong pipeline by accident.

**Status**: **INTEGRATED.**

**Changes**:
- §6.4 (Per-worker pipeline) rewritten: workers now emit
  `strict_exact_hash` + `relaxed_exact_hash` (blake3 8-byte) +
  `minhash_sig` (128 perms, token 5-shingles, blake3-hashed) +
  `lsh_band_hashes` (32 bands × 4 rows). Skip-short-doc threshold
  cited from `SHORT_DOC_TOKEN_THRESHOLD`. Output is zstd-Parquet.
- §6.5 (Coordinator joins) rewritten: three joins (strict_exact +
  relaxed_exact + LSH-near). Output paths corrected to
  `artifacts/overlap/{strict_exact,relaxed_exact,near}/<a>_x_<c>.parquet`,
  `holdout_contamination.parquet`, `per_c3_doc_overlap.parquet`,
  `per_c3_source_actionable.parquet`.
- §7 (Implementation outline) script names updated:
  `06_exact_overlap_join.py` joins both strict + relaxed exact via
  blake3; `07_minhash_overlap_lsh.py` does 128-perm LSH band join
  + Jaccard ≥ 0.85 validation. Worker `hash_pass.py` description
  expanded to spell out the three stages.

Verified: a final grep of the execution sections returns no stale
`md5 / 256-perm / exact_overlap/ / minhash_overlap/ /
holdout_contamination.json` references. Historical mentions in §3
(explicitly framed as "replaces my earlier md5 framing") are kept
as the correction trail.

### Finding 3 (MEDIUM) — Row counts not aligned with train artifact

**Reviewer**: Plan said 546,920 GlossAPI + 13,906,493 HPLT (the
pre-split mix counts). The actual train/firing-count summary has
517,791 GlossAPI + 13,883,763 HPLT — the post-split row counts.
The audit must be explicit about which universe each measurement
uses (train for CPT, val/test for held-out).

**Status**: **INTEGRATED.**

**Changes** (PLAN.md §2.2):
- Replaced the single-column "Docs in C3 mix" table with a
  **two-column** table: `docs in train split` vs `docs in mix
  (pre-split)`. Train numbers are now headline (517,791 +
  13,883,763 = 14,401,554); mix numbers retained for
  cross-reference only.
- Added explicit "Audit must be explicit about which universe each
  measurement uses" guidance, naming the three universes:
  - **CPT-planning** → train counts.
  - **Held-out contamination** → val (7,654) + test (7,282).
  - **Whole-mix** → useful only for cross-checking distribution;
    not CPT-actionable.

### Finding 4 (MEDIUM) — Per-doc overlap-ratio underspecified for non-sentence matches

**Reviewer**: §3.6 defines `matched_chars` as union of spans from
sentence-level matches. That works for EuroParl-style sentence
containment, but not for whole-document strict/relaxed/near matches
where no spans are produced. Need a rule before scripting.

**Status**: **INTEGRATED.**

**Changes** (PLAN.md §3.6):
- Added a per-match-type rule table:
  - `strict_exact` whole-doc match → `overlap_ratio = 1.0`
  - `relaxed_exact` whole-doc match → `1.0`
  - `near` whole-doc match (Jaccard ≥ 0.85) → `1.0` (the 0.85
    Jaccard threshold already implies near-duplicate; partial-credit
    math here would mislead)
  - `sentence-level` matches → `union(matched_char_spans) /
    total_doc_chars` (the original rule, kept for EuroParl-style
    cases)
- When a doc matches at multiple types, the final ratio is `max`
  across types — strict + sentence-fragments → 1.0.

### Finding 5 (MEDIUM/LOW) — Threshold policy is a default, not a truth

**Reviewer**: The drop ≥ 0.30 / partial 0.05-0.30 thresholds are
reasonable starts but should be framed as a default sensitivity
setting. REPORT.md should show fresh-token totals under at least
three threshold settings.

**Status**: **INTEGRATED.**

**Changes** (PLAN.md §3.6):
- Added an explicit "these thresholds are NOT a recipe truth — they
  are a default sensitivity setting" callout.
- Required REPORT.md to include a sensitivity-analysis table with
  three suggested grid points:
  - Strict: drop ≥ 0.10, partial ≥ 0.02
  - Default: drop ≥ 0.30, partial ≥ 0.05
  - Lenient: drop ≥ 0.50, partial ≥ 0.10
- Raw `overlap_ratio` is still recorded per-doc so the user can
  re-classify post-hoc.

### Finding 6 (LOW) — Stale prose references

**Reviewer**: §10 still asked "Which 19 GlossAPI sub-datasets…"
even though count was corrected to 17. §11 still said "anti-join by
Apertus-Greek-md5" even though the join key is now strict/relaxed/
near at varying strictness.

**Status**: **INTEGRATED.**

**Changes**:
- §10 rewritten: "Per-sub-dataset breakdown granularity within
  GlossAPI?" with manifest-derived 17 GlossAPI + 1 HPLT = 18
  source count.
- §11 (eval-slice integrity action) rewritten: "rebuild C3 val/test
  by anti-joining the Apertus side at the strictest matching level
  per §3.5 full-ladder check — typically `strict_exact` if hits are
  present there; else `relaxed_exact`; else `near` at Jaccard ≥ 0.85."
- §8 Risks #1 and #4 rephrased to reference the three-stage ladder
  (strict_exact / relaxed_exact / near) rather than `exact md5 vs
  MinHash`.

## Net effect on the plan

The conceptual / execution mismatch is closed:

- **Truth claim corrected**: consumed_exposure_estimate is now
  honestly framed as probability-weighted expectation, not
  deterministic per-document evidence. Tier labels use "EXPOSED"
  not "SEEN".
- **Execution consistency**: no stale md5/256-perm/JSON references
  in the coordinator or worker sections. Implementer can follow
  §6/§7 without falling back to the wrong pipeline.
- **Universe discipline**: train/val/test/mix universes named
  separately; each measurement attributed to one.
- **Per-doc ratio rules**: all four match types specified;
  scripting can proceed without judgment calls at runtime.
- **Threshold sensitivity**: REPORT must show three threshold
  settings; user picks the recipe-relevant one post-hoc.

The plan is **internally consistent** now. Scripting can start.

The single conceptual point worth re-emphasising before
implementation: **never report per-doc "seen by Apertus" claims
from this audit, only "exposed under Apertus's sampler with
probability p."** The artifacts and prose throughout the plan now
honour this distinction.
