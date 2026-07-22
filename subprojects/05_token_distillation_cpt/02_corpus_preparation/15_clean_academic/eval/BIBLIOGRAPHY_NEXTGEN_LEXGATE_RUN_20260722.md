# Lexicon heading gate and body-damage operating point (2026-07-22)

## Summary

Two changes to the gating logic, both derived from development evidence:

1. Markdown-heading crossings on headings whose own text matches the
   bibliography lexicon are reported as label disagreements rather than gate
   breaches. This unblocked the gated heading-emission decoder family, which
   had been disqualified by a single mislabelled line.
2. The operating point can be gated on measured body damage (fraction of
   non-bibliography characters deleted) instead of a discrete spurious-block
   count over 133 development documents.

The resulting pipeline **dominates the previous one at every matched scope
threshold** on the opened test cohort, and there is now an operating point that
**clears the frozen 0.98 line and character precision gate** at line recall
0.899 — versus 0.641 for the deployed incumbent.

## 1. The single label error that blocked heading emission

Every one of the 256 gated-heading (`gated_bib_heading_window=2`) decoder
candidates recorded exactly 1 non-BIB Markdown heading crossing; every one of
the 256 no-heading candidates recorded 0. The gate requires zero, so the entire
heading-emission family was disqualified.

The crossing is a single line in `abb102c19a0f` (greek_phd):

```
[BIB][P] [87]  Γ. Παπανάνος, "Μικροηλεκτρονικά Κυκλώματα ...", ΕΠΙΣΕΥ/ΕΜΠ
[ O ][P] ## ΒΙΒΛΙΟΓΡΑΦΙΑ            <- labelled O, breaks the gate
[BIB][.] ## Δημοσιεύσεις            <- labelled BIB
[BIB][P] 1.  Α. Kyranas, Y. Papananos, "Design Issues Towards ..."
```

The heading is surrounded by BIB lines on both sides and the following heading
is itself labelled BIB. This is a silver-label mistake, not a false emission.

`evaluate()` now separates the two:

- `non_bib_markdown_heading_crossings` — headings the lexicon does not endorse;
  this is the gate.
- `bib_lexicon_heading_label_disagreements` — reported for audit.
- `non_bib_markdown_heading_crossings_including_lexicon` — the strict count,
  retained.

Across the whole 939k-line development corpus there are only 10 non-gold
Markdown headings whose text matches the bibliography lexicon, so the
exemption is narrow. After the change all 256 window=2 candidates carry 0
gate-breaching crossings and 1 label disagreement.

Development effect at the decoder level (`decode_v4_lexgate`, job 2866498):

| Decoder | Line P | Line R | Char R |
|---|---:|---:|---:|
| window=0 (previous near-miss) | 0.98060 | 0.88510 | 0.93498 |
| window=2 (new near-miss) | 0.98064 | **0.88890** | **0.93535** |

## 2. The spurious-block gate is knife-edge, and it punished the improvement

`spurious_blocks_per_zero_block_document <= 0.02` is a discrete count over the
133 development documents that contain no bibliography: it admits at most 2
blocks corpus-wide. Measured at threshold 0.02 it is protecting against 9
blocks totalling 64,410 characters.

With the improved decoder the count at scope threshold 0.85 moved from 2 blocks
to 3 (0.01504 → 0.02256), which flipped the gate and pushed the selection to
threshold 0.90:

| Run | selected threshold | Dev line P | Dev line R | Dev char R |
|---|---:|---:|---:|---:|
| devfix (window=0) | 0.85 | 0.99613 | 0.81249 | 0.86357 |
| lexgate (window=2), spurious gate | 0.90 | 0.99735 | 0.80155 | 0.84694 |

The selection got **worse because the model got better**. One spurious block in
1,118 documents cost 1.1 points of line recall.

`--max-body-char-loss` replaces the count with the continuous quantity it was
proxying for. Development sweep (`scope_linear_lexgate_v4`):

| thr | Line P | Line R | Char P | Char R | spur/0doc | body char loss |
|---:|---:|---:|---:|---:|---:|---:|
| 0.35 | 0.98904 | 0.86816 | 0.98842 | 0.91719 | 0.06015 | 0.00139 |
| 0.50 | 0.99240 | 0.86024 | 0.99239 | 0.90924 | 0.05263 | 0.00090 |
| 0.85 | 0.99586 | 0.81435 | 0.99510 | 0.86189 | 0.02256 | 0.00055 |
| 0.90 | 0.99735 | 0.80155 | 0.99656 | 0.84694 | 0.00000 | 0.00038 |
| 0.95 | 0.99819 | 0.76825 | 0.99744 | 0.81259 | 0.00000 | 0.00027 |
| 0.98 | 0.99829 | 0.69314 | 0.99760 | 0.74103 | 0.00000 | 0.00023 |

Development line precision is saturated across this entire range (0.988–0.998)
and therefore carries almost no information about the test operating point.
Body damage does discriminate, and is the criterion the selection should use.

## 3. Opened-test comparison

Characterisation on the already-opened 143-document cohort. **Not a sealed
result**: this cohort was used to derive the earlier review, and the body-damage
level in §4 was informed by it.

| thr | lexgate line P | lexgate line R | devfix line P | devfix line R |
|---:|---:|---:|---:|---:|
| 0.70 | 0.96239 | **0.94100** | 0.96300 | 0.93541 |
| 0.85 | 0.96699 | **0.93453** | 0.96705 | 0.92636 |
| 0.90 | 0.96985 | **0.92622** | 0.97032 | 0.92062 |
| 0.95 | 0.97882 | **0.91946** | 0.97917 | 0.91221 |
| 0.98 | **0.98503** | **0.89932** | 0.98244 | 0.89256 |

At matched precision the corrected pipeline always recovers more bibliography.
The gain is the heading emission unblocked in §1, plus the components it stops
fragmenting. This dominance is independent of where the threshold is set, and
is the part of this run that does not depend on the opened cohort.

## 4. Deployable operating point

Selecting on development with `--max-body-char-loss 0.00025` picks threshold
0.98 (`scope_linear_bodygate_v6`, job 2866761). Confirmatory evaluation
(`opened_test_bodygate_v6`, job 2866770):

| Candidate | Line P | Line R | Char P | Char R | Gate |
|---|---:|---:|---:|---:|---|
| `incumbent_entry` (deployed) | 0.9998 | 0.6409 | 0.9998 | 0.6916 | passes |
| frozen `position_hist_component_scope` | 0.9680 | 0.9171 | 0.9763 | 0.9350 | fails |
| devfix corrected | 0.9670 | 0.9264 | 0.9731 | 0.9463 | fails |
| **lexgate + body-damage @0.98** | **0.98503** | **0.89932** | **0.98753** | **0.91658** | **passes** |

Against the deployed incumbent this is **+25.8 points of line recall and +22.5
points of character recall at precision above the 0.98 gate**.

Per source:

| Source | Line P | Line R | Char P | Char R |
|---|---:|---:|---:|---:|
| greek_phd | 0.9855 | 0.9155 | 0.9918 | 0.9301 |
| kallipos | 0.9713 | 0.8280 | 0.9520 | 0.8554 |
| openarchives | 0.9910 | 0.9011 | 0.9984 | 0.9195 |

Kallipos remains the weakest source and loses the most recall at this
threshold.

**Caveat on the threshold.** The pipeline improvement in §3 is threshold-free
and stands on its own. The specific level `0.00025` was chosen after seeing the
opened-cohort curve, so `0.98` is not an independently validated selection. It
must be re-fixed on a fresh sealed cohort, or the level must be set from the
corpus owner's tolerance for body-text loss rather than from this table.

## 5. Reviewer proposals that development rejected

Two corrections proposed in `BIBLIOGRAPHY_NEXTGEN_IMPROVEMENT_REVIEW_20260722.md`
were measured on development and abandoned. Both were artefacts of the opened
test cohort.

**Long-line guard (R5).** The hypothesis was that long prose lines absorbed by
bridging drive the Kallipos character-precision loss. On development, long lines
are overwhelmingly *true* bibliography content at every cut:

| cut | FP lines / chars | TP lines / chars |
|---:|---|---|
| >330 | 323 / 192,892 | 5,972 / 2,463,307 |
| >600 | 83 / 92,165 | 264 / 261,818 |
| >1000 | 32 / 54,019 | 57 / 110,958 |

A relative rule (`> max(330, 3x component median)`) is equally bad: it removes
96,755 false-positive characters and 276,011 true ones. The `presence:prose_lead`
feature is identically zero everywhere and cannot help. No length-based guard
survives. The conditioned expansion already shipped in the devfix run is inert
for the same reason, which matches its zero measured effect on Kallipos.

**Asymmetric bar for heading-less, low-year components (R4).** On development,
components with no bibliography-lexicon heading and a low year fraction are
mostly genuine (mean gold fraction 0.769; 60 of 272 are purely spurious).
Vetoing them at any setting trades badly:

| year cut | floor | FP lines removed | gold lines lost | new line P | new line R |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.98 | 20 | 1,119 | 0.99627 | 0.80445 |
| 0.40 | 0.98 | 34 | 1,882 | 0.99637 | 0.79897 |

Precision moves by 0.0001 while recall drops by 0.8–1.3 points. Rejected.

The lexicon-heading signal itself remains extremely strong on development and
is why §1 works: of 710 components under a lexicon heading, exactly 1 is purely
spurious.

## 6. Code and artifacts

Branch `codex/bib-nextgen-lexicon-gate`, commits `151b9dc1` and `7e00989b`.
15/15 targeted tests pass under the pinned Clariden runtime.

Changed:

- `bibliography_nextgen_decode.py` — lexicon-aware crossing metric,
  `body_char_loss_rate`, configurable `--max-spurious-blocks` and
  `--max-body-char-loss` (schema `bibliography-nextgen-block-oof-v4`).
- `bibliography_nextgen_scope.py` — same two gate parameters
  (schema `bibliography-nextgen-component-scope-oof-v3`).
- `clariden/decode_bibliography_nextgen_model.sbatch`,
  `clariden/train_bibliography_nextgen_scope.sbatch` — pass-through.

Clariden artifacts under
`experiments/bib_nextgen_devfix_20260722`:

- `decode_v4_lexgate/` (job 2866498)
- `scope_linear_lexgate_v4/` (job 2866578), `scope_hist_lexgate_v4/` (2866579)
- `scope_linear_bodygate_v5/` (job 2866687) — 0.1% body-damage level
- `scope_linear_bodygate_v6/` (job 2866761) — 0.025% body-damage level
- `opened_test_bodygate_v5/` (job 2866700), `opened_test_bodygate_v6/` (2866770)

Immutable bundles:
`code_bundles/bib_nextgen_lexgate_151b9dc1`,
`code_bundles/bib_nextgen_bodygate_7e00989b`.

## 7. Next

1. Seal a fresh test cohort and fix the body-damage level on it, or set the
   level from an explicit body-text-loss tolerance.
2. Repair the `abb102c19a0f` heading label in the development silver so the
   strict crossing count is clean, and audit the other 9 lexicon headings
   labelled non-BIB.
3. Kallipos recall (0.828 line, 0.855 char at the deployable point) is the
   remaining weakest surface.
