# sequence_models — the ToC/bibliography sequence-model program

> **In one line:** twelve days and ~200 commits of ladders, decoders, feature audits and sealed annotation lanes that never produced a deployable model — and produced instead the two things that mattered: a correct problem definition (roles, not "bibliography lines") and a sealed cohort whose annotation actually agreed.
> **Period:** 2026-07-12 (STRUCT-2K import) → 2026-07-23 (sealed test v2). **Status:** completed as research. Every candidate was recorded `research_only` or `deployment_gate_passed: false`; the cohorts it built are what the model decision in [`../README.md`](../README.md) was made on.
> **Came from / led to:** [`../`](../README.md) (the two-head tagger) → this → the `heading_lexgate` bake-off and [`../../production`](../../production/README.md)

## Why this existed

The line-level model shipped in June scored each line independently. A bibliography is a
*sequence* — a header, entries, wrapped continuations, separators — so the hypothesis was that a
structured decoder over line scores would recover the recall the line model could not. The track
was CPU-only and explicitly forbidden from changing production Rust; the ladder was
`c0-rust-lr-hysteresis` / `c1-feature-bioes-crf` / `c2-char-ngram-feature-bioes-crf` /
`n1-bytecnn-tcn-masked-crf` ([`README` charter items now folded here](BIB_LADDER_RUNBOOK.md)).

Two constraints shaped everything. First, the supervision is **LLM silver**: the STRUCT-2K
importer authenticates all 2,000 documents but materializes only the **1,392 historical-train**
documents — the 608 historical-test documents are physically absent
([`JOINT_LADDER_RUNBOOK.md`](JOINT_LADDER_RUNBOOK.md)). Second, deployment needed a frozen
policy, a passed Stage54, and a receipt-bound 100-case false-deletion review — none of which was
ever reached.

## History

### 07-13 — deterministic baseline and inspection surfaces

[`BIBLIOGRAPHY_V2.md`](BIBLIOGRAPHY_V2.md) froze a coherence-rule detector on train and opened
validation once: **P 0.9919 / R 0.2169** coherent, against local-evidence-only 0.9361 / 0.4080.
The features carry signal; coherence buys precision at a heavy recall cost. In parallel a
35-feature explorer ([`BIBLIOGRAPHY_FEATURE_EXPLORER.md`](BIBLIOGRAPHY_FEATURE_EXPLORER.md))
was iterated v3→v6 over 14,815 lines, driving accidental feature-overlap events from 6,924 to 0.
A block audit established the target shape: **160 zero-BIB / 973 one-BIB / 259 multi-BIB**
documents. The header-mask audit
([`results/bibliography_header_mask_audit/…/ADJUDICATION_REPORT.md`](results/bibliography_header_mask_audit/struct2k_joint_20260713t195453z/ADJUDICATION_REPORT.md))
**rejected the broad candidate generator as a training mask**: two `gpt-5.6-terra` reviewers
agreed on 114/120 cases, cast **zero ENTRY votes** across all 60 exact-rule cases, but found
21/30 block-start probes and ≥27/30 internal sparse probes were real entries. Only 796 exact
structural lines were masked.

### 07-13/14 — the entry ladder

[`BIB_ENTRY_LADDER_RUN_20260713.md`](BIB_ENTRY_LADDER_RUN_20260713.md) (the 665-line master log)
ran L0–L4/D1 out-of-fold on 1,118 documents / 939,014 lines: best line PR-AUC **D1 0.891568**.
The predeclared gate was ≥0.99 line precision *and* ≤0.02 spurious blocks per silver-zero
document; **nothing passed**, and the freeze job recorded
`research_only_no_candidate_met_safety_gate` rather than moving the gate. A single retrospective
validation gave **line P 0.997976 / R 0.629953** — precision-first and recall-starved.

### 07-14 — separating extraction quality from model error

Prediction-blind screens excluded 6 validation and 5 train documents as unreadable (GLYPH
placeholder floods, Rust badness 68.10 > 60); an outcome-directed worst-50 review added one more.
Recall moved **0.629953 → 0.821625 → 0.841896** while the false-positive count stayed at 58 — the
recall hole was mostly unreadable text, not classifier failure. The 267-document figure is
explicitly branded a sensitivity analysis, not a headline.

### 07-14 — a chain of measured rejections

Proposal generation was shown not to be the bottleneck (candidate ceiling 99.8–99.99%); selection
was. Then, in order: component-gate v5 reaches 0.992836 P / 0.583454 R safely;
`component_expansion_r1` **rejected** (+2,329 lines bought +0.004 recall for precision
0.992836→0.971403); a role-sequence CRF is a good proposal model (0.960729/0.931406) but 0.308
spurious; the exact auxiliary-scope veto reaches 0.991728/0.804743; `rich_component_gate_r1`
**rejected** for fold-unstable feature signs; the signal TCN shows contextual ranking works but
per-line thresholding is the wrong decoder; the barrier arm **rejected** (480 configs, frontier
unchanged). Design record: [`BIBLIOGRAPHY_BLOCK_FEATURE_REFERENCE.md`](BIBLIOGRAPHY_BLOCK_FEATURE_REFERENCE.md).

One fail-closed bug is worth keeping: a `ΣΥΝΤΟΜΟΓΡΑΦΙΕΣ` heading enclosed 971 silver-BIB lines,
which forced the rule *exact negative scope is a **wall**, not a poison pill* — anchoring and
bridging run independently on each side.

### 07-14 — the label-completeness discovery

[`BIB_SIGNAL_FALSE50_REVIEW_20260714.md`](BIB_SIGNAL_FALSE50_REVIEW_20260714.md) audited the 50
worst apparent-false components: **only 9 of 50 are straightforward whole-block classifier
errors.** Twelve are silver omissions, 21 boundary overruns, seven policy-sensitive lists. One
Kallipos book has five genuinely annotated bibliographies (`Δ. 1`–`Δ. 52`) that the silver labels
call zero-BIB. Conclusion: a 99% silver-line-precision gate *"rewards suppressing real
bibliographies"*, and validation must stay closed until the label policy is resolved.

### 07-14 — the first human precision review

The owner reviewed 30 unseen documents / 5,518 predicted lines and marked **99 lines wrong**
([`BIB_UNSEEN_FP_REVIEW_20260714.md`](BIB_UNSEEN_FP_REVIEW_20260714.md)): 58 whole-block (56 of
them concentrated in two documents) and 41 boundary spill matching the 2-line expansion exactly.
Excluding one pathological document, 95 of 4,289 lines = **2.2%**. The signal-refinement run then
froze an asymmetric edge policy (all negative roles on the left, structural only on the right) and
**declined to advance the component gate**, requiring a fresh blinded packet instead of tuning on
the review.

### 07-14 — the pivot: positional roles

[`BIB_POSITIONAL_ROLE_PIPELINE_PLAN_20260714.md`](BIB_POSITIONAL_ROLE_PIPELINE_PLAN_20260714.md)
names the root problem: **treating every non-header line in a silver BIB region as an entry.** The
motivating number is D1's recall of ~0.556 on ≤110-character lines against ~0.935 on 111–220. It
introduces seven roles (`ENTRY_ANCHOR`, `CONTINUATION`, `FILLER`, `HEADER`, `SUBHEADER`,
`NON_BIB`, `UNKNOWN`) and a positional ladder P0→P3.

[`BIB_POSITIONAL_ROLE_IMPLEMENTATION_20260714.md`](BIB_POSITIONAL_ROLE_IMPLEMENTATION_20260714.md)
executes it. Two blind `gpt-5.6-sol` passes agreed on **97.764% exact seven-role**; the owner's
252-line human audit confirmed 226/241 of the automatic consensus, and **all 15 corrections stayed
inside ENTRY/CONTINUATION/FILLER** — no reviewer ever crossed the BIB/NON-BIB line. Targeted
Kallipos closure reached 99.614%, giving a combined overlay of 6,637 trusted role labels.
**Then the positional ladder failed its own gate:** PR-AUC P0 0.3521, **P0D 0.5125**, P1 0.4037,
P1G 0.4149, with a zero lower 95% bound on recall delta at precision ≥0.99. Location summaries
improve ranking but buy no secure recall, so the count-only **P0D was frozen** and P2/P3 never ran.

### 07-15 — the role pipeline, and the gate it missed

[`BIB_ROLE_PIPELINE_IMPLEMENTATION_20260715.md`](BIB_ROLE_PIPELINE_IMPLEMENTATION_20260715.md)
assembled a heading expert (any-heading PR-AUC 0.9690), a connector expert (front gate **0.6453**,
conditional subtype 0.9709) and a cost-sensitive semi-Markov block decoder. Result: **line
precision 0.999431, char precision 0.999698, zero hard-stop crossings, zero spurious blocks — and
recall 0.943087 / 0.943466 against a 0.95 gate.** Not deployment-approved, and the document says
so. It also records *"RL remains unnecessary."*

### 07-16/17 — what the two hard classes actually are

**FILLER is not a semantic class** ([`…FILLER_FEATURE_RESEARCH_20260716`](BIB_FILLER_FEATURE_RESEARCH_20260716.md)):
50.7% are ≤3-character fragments, 47.5% come from **one** malformed OpenArchives document, 95% sit
within five lines of an entry; line shape dominates (permutation drop 0.327554) and the
177-feature table is over-complete. **CONTINUATION is a citation fragment**
([`…CONTINUATION_FEATURE_RESEARCH_20260717`](BIB_CONTINUATION_FEATURE_RESEARCH_20260717.md)): only
7.5% clear the entry threshold alone, 31.2% are rescued by joining a neighbour, and the three
sources have three different geometries.

The **gap connector** ([`results/bibliography_gap_connect/20260717_v2/`](results/bibliography_gap_connect/20260717_v2/README.md))
lifted end-to-end line recall 0.4751 → 0.5588 at **2 false connects in 777** — and its own oracle
audit reframed the problem: of 23,292 missed gold lines only **~11% are internal gaps**, against
~49% outer edges and ~39% weak or unseeded blocks. Ordered sequence structure failed its
predeclared evidence gate (ordered − shuffled PR-AUC 0.002368, CI crossing zero), so the ordered
TCN and random-span augmentation were **rejected**. The dedicated continuation head raised line
PR-AUC 0.7820 → 0.8381 and **every** block integration was rejected — the best blend reached
1.000000 precision at 0.939597 recall, still under 0.95
([`…CONTINUATION_HEAD_RUN_20260719`](BIB_CONTINUATION_HEAD_RUN_20260719.md)).

### 07-18/19 — the sealed cohort that failed, and the successor

A 150-document sealed cohort (194,273 lines) was built prediction-blind. Mid-run the annotation
model switched: Sol lanes were stopped, accepted Sol batches **imported rather than re-annotated**,
and `gpt-5.6-terra` high completed the rest
([`SEALED_ANNOTATION_MODEL_SWITCH_20260718.md`](SEALED_ANNOTATION_MODEL_SWITCH_20260718.md)).
The merge then **failed closed**: A/B agreement 0.97758 against a 0.98 gate. The threshold was not
touched and no `FROZEN.receipt.json` was written.

Diagnosis found two **specification** faults, not annotator faults, each repaired conservatively
and independently per pass: contextual roles assigned in components containing no ENTRY (3,010
labels, 2,997 of them FILLER → filler/continuation detection 51.02% → 74.82%), and header roles on
non-Markdown lines (→ heading detection 85.91% → 88.22%). Contributor analysis then showed
**92.75% of the residual disagreement came from seven footnote-heavy documents**, with the root
cause named exactly: *"`ENTRY` was interpreted as 'citation-shaped line' instead of 'entry
belonging to a bibliography/list-of-references region'."* Those seven were dropped, giving a
**143-document consensus silver at 99.9133% membership agreement**, terminally sealed as a
*post-hoc* evaluation set — explicitly not a rewrite of the failed prediction-blind attempt
([`CONSENSUS_SILVER_20260719.md`](CONSENSUS_SILVER_20260719.md),
[`SEALED_BIBLIOGRAPHY_COMPLETION_AUDIT_20260719.md`](SEALED_BIBLIOGRAPHY_COMPLETION_AUDIT_20260719.md)).
A v1 seal that had used trusted coverage as if it were agreement was superseded and kept read-only.

Meanwhile the [`evolution/`](evolution/README.md) harness ran G1 and G2 and promoted nothing.

### 07-23 — the specification fix, proved

[`BIBLIOGRAPHY_SEALED_TEST_V2_20260723.md`](BIBLIOGRAPHY_SEALED_TEST_V2_20260723.md): a fresh
150-document cohort (210,704 lines), drawn after excluding all 650 prior documents,
quality-gated by two agreeing Terra reviewers (28 KEEP / 8 UNUSABLE), annotated by two blind
Terra-high passes with the V2 prompt. The proof that the fix worked is **zero contextual-role
violations in both passes**; membership agreement **99.7328%**, κ 0.9868. It passed the
predeclared gates on the first attempt, kept all 150 documents, needed no third annotator, and
masked its 563 disagreements. Trusted BIB lines: **23,694**. This is the cohort the
`heading_lexgate` decision was made on.
[`BIBLIOGRAPHY_ANNOTATION_REVIEW_RUNBOOK_20260723.md`](BIBLIOGRAPHY_ANNOTATION_REVIEW_RUNBOOK_20260723.md)
codifies the method and its eight lessons.

## Outcome

- **No sequence model was deployed.** The best block decoder reached 0.999431 line precision at
  0.943087 recall — precision to spare, recall short of gate, every time.
- **The problem definition changed and stuck:** roles, not "bibliography lines"; an ENTRY is an
  entry *in a reference region*, not a citation-shaped line. That definition, written into the V2
  annotation prompt, is what made the 07-23 cohort agree.
- **Cohorts, not models, were the durable output:** the 143-document consensus silver (99.9133%)
  and the 150-document `bibliography_150_20260723_v2` (99.7328%, 23,694 trusted BIB lines).
- **Measured negatives worth not repeating:** component expansion, the rich component gate, the
  barrier arm, the positional ladder P1/P1G, random-span augmentation, the ordered TCN, and all
  six continuation-head block integrations.
- **Method rules banked:** gates are not moved when a run fails them; a failed cohort is left
  permanently unfrozen rather than rewritten; blinded packets are rebuilt rather than tuned on a
  review; and label completeness is audited before a precision gate is trusted.

## Sub-units

| Dir | Role | Period | Status | Result |
|---|---|---|---|---|
| [`evolution/`](evolution/README.md) | Controlled generational search around the frozen decoder | 2026-07-18 → 07-19 | completed | G1 kept its parent; G2 promoted nothing; `g3_authorized: false` |
| `results/` | Per-run machine artifacts and review sites | 2026-07-13 → 07-19 | archive | Nine run directories, several with their own `README.md`/report |
| `fixtures/`, `tests/`, `clariden/`, `presentation_templates/` | Test fixtures, unit tests, Slurm jobs, HTML review-site templates | — | support | — |

## Working documents

- **Runbooks (procedure)** — `BIB_LADDER_RUNBOOK.md`, `JOINT_LADDER_RUNBOOK.md`,
  `TOC_BIB_AGENT2_RUNBOOK.md`, `SEALED_BIBLIOGRAPHY_TEST_RUNBOOK.md`,
  `BIBLIOGRAPHY_ANNOTATION_REVIEW_RUNBOOK_20260723.md`, `DETERMINISTIC_ABLATION_RUNNER.md`.
  `HUMAN_GOLD_RUNBOOK.md` specifies a 2,720-action human annotation campaign that was **never
  executed** — while `BIB_LADDER_RUNBOOK.md` states no human campaign is planned, a contradiction
  left unreconciled here.
- **Plans** — `BIB_LINE_TO_BLOCK_CLASSIFIER_PLAN.md` (superseded), `BIB_POSITIONAL_ROLE_PIPELINE_PLAN_20260714.md`.
- **Run logs** — `BIB_ENTRY_LADDER_RUN_20260713.md` (the master log), plus
  `BIB_SIGNAL_REFINEMENT_*`, `BIB_FRESH_EDGE_COMPONENT_*`, `BIB_POSITIONAL_ROLE_IMPLEMENTATION_*`,
  `BIB_ROLE_PIPELINE_IMPLEMENTATION_*`, `BIB_CONTINUATION_HEAD_RUN_*`.
- **Reviews and audits** — `BIB_TRAINING_EXTRACTION_REVIEW_*`, `BIB_VALIDATION_WORST50_REVIEW_*`,
  `BIB_SIGNAL_FALSE50_REVIEW_*`, `BIB_UNSEEN_FP_REVIEW_*`,
  `SEALED_BIBLIOGRAPHY_COMPLETION_AUDIT_*`, `POST_REPAIR_DISAGREEMENT_CONTRIBUTORS_*`.
- **Feature research** — `BIB_FILLER_FEATURE_RESEARCH_*`, `BIB_CONTINUATION_FEATURE_RESEARCH_*`,
  `BIBLIOGRAPHY_BLOCK_FEATURE_REFERENCE.md`, `BIBLIOGRAPHY_FEATURE_EXPLORER.md`, `BIBLIOGRAPHY_V2.md`.
- **Annotation repairs and provenance** — `CONTEXTUAL_ROLE_REPAIR_*`, `MARKDOWN_HEADER_ROLE_REPAIR_*`,
  `SEALED_ANNOTATION_MODEL_SWITCH_*`, `CONSENSUS_SILVER_*`, `SILVER_RECONSTRUCTION.md`,
  `TOC_BIB_FEEDBACK_LOG.md` (the 39-item feedback register).
- **Prompt templates** — `SEALED_BIBLIOGRAPHY_ROLE_PROMPT.md` / `_V2.md`,
  `SEALED_BIBLIOGRAPHY_QUALITY_PROMPT.md`, `bibliography_heading_review_prompt.md`,
  `bibliography_role_review_prompt.md`, `CODEX56_AUDIT_PROMPT.md`. V2 carries the ATX-Markdown and
  entry-anchored-component invariants that fixed the 07-19 failure.

### Known inconsistencies

- The header/gap agreement figures differ between `SEALED_ANNOTATION_MODEL_SWITCH` (85.78% /
  50.65%) and the repair documents (85.91% / 51.02%) for the same pre-repair state.
- The 150-document cohort's "original" membership agreement is quoted as **97.7583%** (the blocked
  merge gate, over 194,273 lines including UNKNOWN) and **98.04%** (the repair tables, over 193,718
  comparable lines). The denominators differ; no document states this side by side.
- The strictest annotation task is called "exact **seven**-role agreement" in
  `CONSENSUS_SILVER_20260719.md` and "exact **eight**-label agreement" in the 07-23 runbook; the
  taxonomy was renamed between contract v1 and v2 (`ENTRY_ANCHOR`≡`ENTRY`, `NON_BIB`→`OTHER`/
  `NON_BIB_HEADER`) and only the first equivalence is flagged.
- `SILVER_RECONSTRUCTION.md` records that **zero raw line rows are loadable from the checkout** for
  a fresh fit — all 240 `batch_*.json` files are a missing dependency. The coordinates survive; the
  text does not.
- Kallipos heading agreement rising to "100%" is empty (it has no retained Markdown headings in
  either pass); the caveat is stated in the repair document and dropped in the summary tables.
