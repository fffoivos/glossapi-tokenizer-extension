# Eval A — end-matter bibliography boundary: confusion matrix + the recall problem

497 doc tails (greek_phd + openarchives), Opus-annotated BLIND for the bibliography start line,
stratified to over-sample header-not-found docs. Detector = the whole-doc `endmatter_bib` boundary.

## Result (n=457 usable, weighted equally)

| | truth: has bib | truth: no bib |
|---|---:|---:|
| detector: found | 205 (TP) | 3 (FP) |
| detector: not found | 126 (FN) | 123 (TN) |

**precision = 0.986 · recall = 0.619 · F1 = 0.761.** Boundary localisation (both found, n=227):
**median |error| = 0 lines, 75% within 5 lines, no early/late bias.**

## Reading

The boundary detector is the **mirror image of the β-gate**: β over-flagged (precision-limited);
the boundary detector is **conservative — excellent precision (0.99), poor recall (0.62)**. When it
fires it is right and pin-points the line; the failure is **finding the bibliography at all**. By
source: openarchives recall **0.78**, greek_phd recall **0.50** — humanities theses (varied or absent
headers, footnote-heavy, sub-divided bibliographies) are where it misses.

This confirms the investigation's prediction at scale: the detector keys on a fixed set of header stems
(ΒΙΒΛΙΟΓΡΑΦΙΑ/ΠΗΓΕΣ/References…), but a large share of real bibliographies are **header-less**, carry a
non-matching header, or are archival/sub-divided (Αρχειακές πηγές, Αρχειακό υλικό), so the header scan
skips them. The 3 FP are negligible.

## The fix transfers directly from the β-gate

The β work already solved "recognise a reference list without a header": **entry-density + position**
(a bibliography is a dense run of reference entries at the document end). The boundary detector should
be upgraded from header-stem matching to **header-OR-entry-density detection at high position** — i.e.,
also open a bibliography span where a sustained run of `is_entry` lines begins in the document tail,
even with no header. That is the single highest-value change (it attacks all 126 FN) and reuses the
exact `is_entry` + position machinery now deployed for β. Localisation is already excellent, so the
segment-not-amputate property holds; this is purely a recall lift.

## Status

Both reference-cleaning decisions now have validated, leak-aware confusion matrices on held-out
Opus labels: **β-section classification** (P≈0.85/R≈0.91, deployed LR) and **boundary detection**
(P≈0.99/R≈0.62, recall-limited). Labelled sets (`annotations_scale/`, `annotations_scale_A/`) are
reusable regression harnesses. Next lift is boundary recall via entry-density+position.
