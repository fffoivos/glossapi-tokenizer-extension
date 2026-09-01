# 10 — Clean HPLT (web)

> **In one line:** a two-day audit of the 46.5 M-row Greek HPLT slice that built a 21-class error taxonomy, hand-reviewed 947 rows, and concluded that **no broad automatic cleaning rule met the precision bar** — only the narrow E001 control-character fix and reviewed exact-boundary repairs were allowed near production.
> **Period:** 2026-06-05 19:31Z → 2026-06-06 13:20Z (report timestamps). All files entered git in one bulk commit, `a19c136f` (2026-06-11). **Status:** completed; conclusion carried into the launch policy.
> **Came from / led to:** the HPLT source selection in [`../../ROADMAP_20260611.md`](../../ROADMAP_20260611.md) → this → [`../15_clean_academic`](../15_clean_academic/README.md) and [`../20_dedup`](../20_dedup/README.md)

## Why this existed

HPLT 3 already applies its own extraction, LID thresholding, robots filtering, document dedup
and a Web-Docs-Scorer quality gate — and the corpus used here is `HPLT/ell_Grek_ge8_no_mt_clean60`,
i.e. *already* the high-quality bins. The question was whether a **high WDS score proves clean
text**. [`hplt3_script_library/script_index.tsv`](hplt3_script_library/script_index.tsv) answers
it by reading upstream's own pipeline stage by stage and recording, for each script, "role for
HPLT" against "role for us" — the recurring verdict being *"Use WDS as stratification, not
proof of clean text"* and *"the final HPLT clean is mostly filtering, not fine-grained repair."*
Everything downstream follows from that: local, in-row artifacts survive a document-level
filter.

## The operating constraint

Set at the start and never relaxed: **source HPLT rows are immutable**. Every report in this
directory carries a policy note to that effect — *"Audit only. This does not approve automatic
cleaning or mutate source rows."* Any repair had to be materialized as a **duplicate/shadow
row** with before/after checksums, never an in-place edit. The bar for promoting any
destructive action to automatic use was a **Wilson 95% lower bound ≥ 95%** on reviewed
precision, measured two ways: *intervention-warranted* (should this row be touched at all?) and
*exact-action* (is this specific action the right one?).

## History

### 2026-06-05 evening — inventory, review queue, pilot

`hplt_feature_inventory.py` does a metadata-only first pass to build statistically meaningful
samples, then reads text **only** for sampled documents. It maps coarse detector features onto
21 candidate catalog ids E000–E020 (control/replacement chars, escaped/entity residue,
markup remnants, URL dumps, symbol/encoding, top/bottom boilerplate, menu residue, split
candidates, internal repetition, language drift, badness, table-heavy, duplicated body, short
low-Greek), with the docstring caveat that *"these IDs are retrieval hooks, not final labels."*
Seven `review_queue_*` and seven `candidate_issue_summary_*` snapshots between 19:36Z and
20:46Z record the queue being rebuilt; six `pilot_review_pack_summary_*` files record the
pilot packs (36 → 57 records) stratified by task: `precision_E003`…`precision_E015`,
`destructive_action_review`, `false_negative_control`, `borderline_false_negative_probe`.

### 2026-06-05 21:00–23:35Z — the destructive-action and quarantine reviews

`destructive_span_policy_review_pack_20260605T210613Z` (45 annotations) and
`quarantine_precision_review_*_20260605T211220Z` establish the first precision numbers, and
`policy_gate_audit_*` starts running as a standing scoreboard (37 successive audits through
the next day). The E001 control-character scan
([`reports/e001_control_char_scan_20260605T224007Z_summary.json`](reports/e001_control_char_scan_20260605T224007Z_summary.json))
swept the whole corpus — **46,493,906 rows / 172,678,703,630 chars in 2,479 s** — and found
exactly **32 matching rows / 166 replacement characters** (match rate 6.88e-7). That narrowness
is what later made E001 the only class safe enough to act on.

### 2026-06-05 23:35Z — the shadow-overlay mechanism proved on E001

`e001_shadow_overlay_20260605T233528Z` materialized 32 shadow records removing 166 chars /
144 tokens, with a manifest validation reporting **0 violations** and no source row mutated.
The shadow-overlay gate is marked `met` from here on.

### 2026-06-06 00:00–08:30Z — the boundary-annotation grind

24 `boundary_*_materialization_*` summaries and 16 `pending_boundary_review_pack_*`
snapshots trace an hour-by-hour loop: build a pack of pending boundary cases → annotate →
materialize the *reviewed exact* spans as shadow rows → re-audit. The families are visible in
the filenames: suffix trims, footers, comment tails, whole-doc drops, small splits, safe
splits, normalization, freepen duplicate notes, holdout drops, URL-rescue splits (batches 1
and 2). Final tally in the last gate audit: **925 boundary materialization delta keys,
780,240 chars removed**, distributed as `split_doc` 176, `trim_suffix` 128, `quarantine` 116,
`drop_doc` 33, `trim_prefix` 9, `trim_span` 9, `normalize_or_trim_span` 5, `keep` 1. The
reviewed overlay at 06:10Z holds **382 decision rows / 241 shadow records / 330 unique source
docs**, again with **0 shadow-link issues and 0 validation violations**.

### 2026-06-06 00:00–06:00Z — hunting false negatives (the gate that never closed)

Four successive packs — control triage (32), representative (80), global-200 (200), global
remaining full-text — plus a 200-row candidate-*keep* control review. The keep-control result
is the one that mattered: **43 actionable false negatives and 12 serious ones** among rows the
detector proposed to keep. Serious FNs are dominated by missed archive/topic/category/list
pages (E010/E011). The `remaining_false_negatives` gate is recorded `not_met` to the end, with
the note *"Representative FN evidence is now present, but serious residual FNs remain and must
be reduced or explicitly accepted."*

### 2026-06-06 10:38–12:25Z — the full rule count and the held-out check

Five `full_rule_count_*` passes scored the rules over a ~100 k-row stratified slice. The last
([`reports/full_rule_count_20260606T122551Z_summary.json`](reports/full_rule_count_20260606T122551Z_summary.json))
gives `keep` 79,618, `quarantine` 18,707, `trim_suffix` 1,483, `drop_doc` 138, `trim_prefix`
47, `normalize_or_trim_span` 7 — i.e. **~19% of rows would be quarantined**, dominated by
E010 (15,628 hits) and E017 (9,308). Two heldout slices (2,295 → 459 rows) were drawn and
40 `drop_doc` rows blind-reviewed
([`reports/heldout_generalization_audit_20260606T121700Z.json`](reports/heldout_generalization_audit_20260606T121700Z.json)):
**exact-action precision 0.225** against intervention-warranted 0.900 — 23 of 40 proposed
whole-document drops should have been a `trim_suffix` instead. Two `full_rule_count_*_heldout_docs/`
directories keep 4,573 raw held-out document texts as the review substrate.

### 2026-06-06 12:20Z — the final policy gate audit and recommendation

[`reports/policy_gate_audit_20260606T122000Z.json`](reports/policy_gate_audit_20260606T122000Z.json)
and [`reports/policy_recommendation_20260606T122100Z.json`](reports/policy_recommendation_20260606T122100Z.json)
close the stage on **947 reviewed rows / 719 unique source docs** (labels: clean 400,
false_negative_found 179, partial_true_positive 169, true_positive 125, false_positive 70,
unclear 4). Reviewed evidence exists for all 20 candidate ids, so `issue_discovery` is `met`;
`shadow_overlay` is `met`; **`destructive_precision` and `remaining_false_negatives` are
`not_met`.**

| Action | Reviewed rows | Exact-action precision | Wilson 95% lower | Intervention-warranted |
|---|---:|---:|---:|---:|
| `normalize_or_trim_span` | 44 | 0.7727 | 63.0% | 0.9773 |
| `drop_doc` | 74 | 0.5676 | 45.4% | 0.8514 |
| `quarantine` | 198 | 0.3636 | 30.0% | 0.7929 |
| `trim_span` | 36 | **0.0833** | 2.9% | 0.6944 |

Not one action clears 95%. The recommendation is correspondingly uniform: *"Use reviewed
exact-boundary duplicate/shadow rows only; do not promote the broad detector."* On
`drop_doc`: *"Exclude only reviewed whole-doc drops from a derived stream; do not drop from
detector hits alone."* On `quarantine`: *"Use as a manual holdout/review state, not as
automatic exclusion."*

## Outcome

- **The broad HPLT cleaner was not shipped.** The standing decision in
  [`../../ARCHIVE.md`](../../ARCHIVE.md): *"only the confident E001 replacement-character /
  control residue cleanup is in the launch path. Broader HPLT cleaning categories were
  explored but not approved as destructive production overlays."*
- Two method rules survived as project-wide posture: only exact, observable, high-confidence
  artifacts are eligible for automatic transformation (avoid semantic rules), and **source
  text is immutable — cleaning happens through derived/shadow outputs.**
- The measured negative results are the real product: `trim_span` exact precision 0.0833,
  `quarantine` 0.3636, and held-out `drop_doc` 0.225 with 23/40 cases that were really suffix
  trims. Each is a rule someone would otherwise have shipped.
- Left open at the end (`next_required_evidence`): re-review action-specific precision after
  tightening split/drop/quarantine rules; reduce or explicitly bound the serious false
  negatives, *"especially E010/E011 archive/topic/category rows"*; compute chars/tokens
  removed by error type with good-text-loss estimates. None of this was done — the stage ends
  here and the effort moved to [`../15_clean_academic`](../15_clean_academic/README.md).

## Where things are

| Path | What |
|---|---|
| [`scripts/hplt_feature_inventory.py`](scripts/hplt_feature_inventory.py) | The detector: metadata-first sampling, cleaning-risk features, the E000–E020 mapping and the proposed-action logic. |
| [`scripts/build_policy_gate_audit.py`](scripts/build_policy_gate_audit.py) · [`build_policy_recommendation_report.py`](scripts/build_policy_recommendation_report.py) | The scoreboard and the final per-action recommendation. |
| [`scripts/scan_e001_control_chars.py`](scripts/scan_e001_control_chars.py) · [`materialize_e001_shadow_overlay.py`](scripts/materialize_e001_shadow_overlay.py) · [`validate_shadow_manifest.py`](scripts/validate_shadow_manifest.py) | The one class that shipped, and the shadow-overlay contract that made it safe. |
| [`scripts/build_full_rule_count_and_heldout_pack.py`](scripts/build_full_rule_count_and_heldout_pack.py) · [`summarize_heldout_generalization_annotations.py`](scripts/summarize_heldout_generalization_annotations.py) | The generalization check that produced the 0.225 exact-action precision. |
| [`hplt3_script_library/script_index.tsv`](hplt3_script_library/script_index.tsv) | Upstream HPLT-3 pipeline provenance, stage by stage, with what each stage does *not* guarantee. `script_checksums.sha256` pins them. |
| [`error_examples/`](error_examples) | 16 real document excerpts for the five C1 "strict action candidate" classes (E001, E002, E003, E013, E018), built by `build_error_example_library.py` with per-class definitions and false-positive boundaries. |

## Working documents

`reports/` holds 198 JSON/CSV run artifacts plus 4,573 held-out document texts (2,278 + 2,295). Nothing is
deleted; the useful groupings are:

- **Final state** — `policy_gate_audit_20260606T122000Z.json`, `policy_recommendation_20260606T122100Z.json`, `heldout_generalization_audit_20260606T121700Z.json`, `full_rule_count_20260606T122551Z_summary.json`. Read these four and you have the stage.
- **Superseded scoreboards** — the other 40 `policy_gate_audit_*` and 24 `policy_recommendation_*` snapshots are the same two reports rewritten as annotations arrived; they are an audit trail, not separate findings.
- **Review packs and their annotations** — `pilot_review_pack_*`, `review_queue_*`, `candidate_issue_summary_*`, `destructive_span_policy_review_pack_*`, `quarantine_precision_review_*`, `false_negative_*`, `global_prevalence_*`, `serious_fn_url_rescue_*`, `heldout_validation_slice_*`.
- **Boundary materialization trail** — 24 `boundary_*_materialization_*_summary.json` and 16 `pending_boundary_review_pack_*`; the aggregate is already in the final gate audit.
- **Shadow overlays** — `e001_shadow_overlay_20260605T233528Z_*` and five `reviewed_shadow_overlay_*` (summary + validation pairs).
- **Held-out corpora** — `full_rule_count_20260606T105018Z_heldout_docs/` and `…T122551Z_heldout_docs/`, raw `.txt` review substrate.

Note: the reports and scripts reference an `ERROR_CATALOG.md` that is **not present in this
tree**; the class definitions that survive are the ones embedded in
`build_error_example_library.py` and `hplt_feature_inventory.py`.
