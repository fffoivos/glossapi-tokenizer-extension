# Review integration — 2026-05-17

Tracks how each finding from the review-agent pass on 2026-05-17 was
addressed in the subproject. Capture preserved for traceability.

## Findings and resolutions

### Finding 1 (HIGH) — Internal scope split between README and tracker

**Reviewer**: `README.md` still presented `FAIRNESS_DEFINITION.md` +
`GREEK_BUDGET.md` as deliverables; `INVESTIGATIONS_TRACKER.md` said
they were out of scope; `_my_synthesis_set_aside/README.md` said they
were not the deliverable. Three docs disagreed.

**Status**: **INTEGRATED.**

**Changes**:
- [`README.md`](README.md) §Scope updated with a "Status
  (2026-05-17 scope-down)" block that defers to the tracker and
  explicitly marks the synthesis as set-aside, with the
  artifact-level reason (Apertus inherited Mistral's BPE table).
- [`README.md`](README.md) §Outputs replaced with the actual current
  file inventory and per-file status, cross-referencing the tracker.
- [`TODO.md`](TODO.md) gained a "Status (2026-05-17 scope-down)"
  banner clarifying which phases are done / deferred / active.

The tracker is now the single source of truth for scope.

### Finding 2 (HIGH) — `12_gini_optimization.md` overstates Apertus's criterion

**Reviewer**: doc said "only stated tokenizer-selection criterion"
and "sole stated criterion" but the quoted Apertus paragraph lists
four metrics (fertility, compression, vocab utilization, Gini) plus
a smaller-vocab preference. Also, the +3-5k prediction is from
`modern_greek_eval`, not FLORES+, and the doc presented it as if
the experiment had been run.

**Status**: **INTEGRATED.**

**Changes**:
- Header replaced. Now reads "experiment plan" not "report and
  plan". Adds an explicit STATUS banner at the top saying the sweep
  has not been run and the prediction is hypothesis from
  extrapolation.
- §1 rewritten to describe the **multi-criteria** framework Apertus
  actually used (Gini differentiator + fertility/compression/util
  verification + smaller-vocab secondary preference). The "Gini is
  the only criterion" framing is replaced with "Gini is the binding
  differentiator; the others still apply."
- §2.1 closing claim "Gini-on-FLORES+ is the sole stated criterion"
  changed to "Gini is the primary differentiator; the others are
  verification + secondary preference."
- §3.3 prediction explicitly reframed as a hypothesis pending
  FLORES+ measurement, with two failure modes spelled out (FLORES+
  Greek fertility could be lower than expected, in which case `N*`
  shifts toward 0; or higher than expected, in which case `N*`
  shifts past +11,264).

### Finding 3 (MEDIUM) — Tokenizer-provenance "no impact" claim unsupported

**Reviewer**: `11_tokenizer_provenance.md` claims Greek / Latin /
Arabic are unaffected by the 486 dropped BPE entries without a
tail-token audit.

**Status**: **INTEGRATED (text-level); audit added as TODO.**

**Changes**:
- The "impact: minimal" paragraph in §"Difference 1" softened to
  "likely minimal but unverified". Added explicit description of
  what a real audit would entail (decode the 486 dropped tokens,
  classify by script, check PMI-set overlap).
- TODO.md gained a "Tail-token audit" item in the new-work-in-scope
  list.

The audit itself is **not run**. To run it would require pulling
Mistral's tokenizer.json (~40 MB), decoding its last 486 BPE
entries, and cross-referencing against our PMI-promoted sets.
Cheap-but-not-zero work, deferred for now.

### Finding 4 (MEDIUM) — "47 custom tokens" paper claim vs. 58-tokens artifact

**Reviewer**: `01_explicit_goals.md` quotes "47 custom special
tokens" without flagging that the artifact-level diff in
`11_tokenizer_provenance.md` shows 58 newly named tokens plus 486
truncated BPE entries.

**Status**: **INTEGRATED.**

**Changes**:
- `01_explicit_goals.md` §A intro gained a "Paper claim vs.
  artifact-level reality" note distinguishing the verbatim paper
  claim ("47") from the measured artifact diff ("58 newly named +
  486 truncated BPE"), with a guidance line: use artifact numbers
  throughout the evidence corpus; quote 47 only as paper claim.

### Finding 5 (LOW/MEDIUM) — Math errors in set-aside synthesis

**Reviewer**:
- `FAIRNESS_DEFINITION.md` Operationalisation 2 says Greek at 2.41
  vs HQ-20 median 1.75 is "just outside" the ±50 % band, but 1.75 ×
  1.5 = 2.625 and 2.41 < 2.625 — Greek is *inside* the band.
- `GREEK_BUDGET.md` TL;DR quotes 5.03 % of vocab; the table later
  in the same doc correctly says 4.85 %.

**Status**: **DOCUMENTED, NOT FIXED.** The set-aside docs are
explicitly not the deliverable. Patching the math without
restructuring the argument would be misleading — the
Operationalisation 2 error in particular collapses the chain of
reasoning that leads to +5,120.

**Changes**:
- `_my_synthesis_set_aside/README.md` gained a "Known errors" section
  documenting both errors with the corrected arithmetic, plus a
  warning that fixing Op. 2's math undermines the +5,120
  recommendation and the synthesis should be reconsidered from
  scratch if revived.

### Finding 6 (LOW) — HPLT rank slip on Apertus-HQ-20

**Reviewer**: `03_evidence_HPLT3_FLORES_classical.md` §1.4 says
"Apertus-HQ-20 covers HPLT ranks {1, 2, 3, ...}" but English (HPLT
rank 1) is NOT in HQ-20 — Apertus handles English through separate
English-only datasets, not via FineWeb-2-HQ.

**Status**: **INTEGRATED.**

**Changes**:
- The rank set corrected to start at rank 2 (Russian). Added an
  explicit note that English / HPLT rank 1 is handled by separate
  English-only datasets, not FW2-HQ.

### Meta — git tracking

**Reviewer**: the whole restored directory is untracked, including
the 4.9 MB Apertus PDF.

**Status**: **NOT yet acted on.** Tracking decision is the user's.
Tracking the PDF would bloat the repo by 4.9 MB; reasonable
alternatives include `.gitignore`-ing `sources/` and keeping the
text outputs tracked.

Recommended user action: `git add` the markdown files (and the
investigation tracker) once content stabilises; consider whether
`sources/apertus_2509.14233v2.pdf` should be tracked or pulled
on-demand from arXiv.

## Net effect

The subproject's spine is unchanged — the evidence layer (01-04, 11)
and the Gini-experiment plan (12) are the same content. The review
findings primarily addressed **framing precision**:

- Scope claims now match across docs (Finding 1).
- Gini-optimization framed as hypothesis pending measurement, with
  multi-criteria framework acknowledged (Finding 2).
- Unverified claims marked as such, with TODOs for verification
  (Finding 3).
- Paper-vs-artifact discrepancies surfaced (Finding 4).
- Set-aside synthesis's math errors flagged (Finding 5).
- Counting slip corrected (Finding 6).

No new evidence was added; no conclusions were changed. The integrity
of the evidence corpus is improved; readers won't trip on stale or
overstated claims.
