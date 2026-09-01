# GreekMMLU response displacement by category

Date: 2026-08-11

## Scope and evidence

This analysis uses the exact decontaminated GreekMMLU predictions from all 19
evaluated Apertus-8B checkpoints: 16,159 fixed questions and 307,021
checkpoint-question response records. It reports the whole benchmark, all 31
native subject labels and all five native educational-level labels.

The compact response matrix was rebuilt from prediction payloads whose hashes
are bound by the frozen answer-drift receipt. Recomputed whole-benchmark
accuracy matches that receipt exactly. Recomputed choice NLL and correct-answer
BPB match the frozen metric series to maximum absolute differences of
`8.22e-15` and `2.11e-15`, respectively.

The displacement statistic is Absolute Historical Wrong-cell Displacement
(AHWD). Within every reported category, a nested accuracy-only baseline matches
the exact correct count at every checkpoint. Its question order is frozen from
iteration-0 choice NLL, with example ID as the deterministic tie-breaker. AHWD
therefore measures response-identity movement beyond a change in aggregate
accuracy. The checkpoint-order null uses 1,999 permutations. Subject and level
tests use Benjamini-Hochberg correction within their respective axes.

## Whole benchmark

| Quantity | Result |
| --- | ---: |
| Initial accuracy | 35.782% |
| Best accuracy | 56.810% at update 9,536 / 39.997B active tokens |
| Final accuracy | 54.855% |
| Peak to final | -1.956 percentage points / -316 net correct answers |
| Choice NLL | 1.45858 initially, 1.07399 best, 1.12213 final |
| Correct at every checkpoint | 3,018 / 18.68% |
| Wrong at every checkpoint | 3,832 / 23.71% |
| Correctness changed at least once | 9,309 / 57.61% |
| Selected answer changed at least once | 11,115 / 68.79% |
| Peak to final newly correct / newly wrong | 821 / 1,137 |
| Peak to final paired replacements | 821, plus 316 net additional errors |

Across the full run, AHWD is 1,415.40 equivalent wrong cells, or 19.50% of
available wrong mass. Its 97.5% permutation floor is 1,080.47, leaving 334.93
cells above that conservative threshold (`p=0.0005`). The strongest boundary is
update 3,576 to 4,768, or approximately 15.0B to 20.0B active tokens. Twenty-seven
of 31 subject labels select the same boundary. The dominant full-run effect is
therefore early response reorganization, not the late plateau.

Restricting the statistic to the ten checkpoints from the global accuracy peak
through the endpoint gives 906.42 raw displaced cells, or 12.60% of available
wrong mass. The 97.5% permutation floor is 871.14, leaving 35.27 cells above the
threshold (`p=0.0095`). The strongest boundary is update 14,627 to 15,496. Update
14,627 is the declared WSD cooldown start. This alignment is evidence of a
response-regime change near cooldown, but it does not by itself prove that the
learning-rate decay caused it.

## Educational levels

All five levels regress in both accuracy and choice NLL from the global accuracy
peak to the endpoint. All five also have significant post-peak historical
displacement after within-axis correction.

| Level | n | Peak to final accuracy | Peak to final NLL | Post-peak AHWD | Wrong-mass share | Adjusted q | Strongest boundary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Professional | 8,070 | -1.91 pp | +0.0423 | 569.25 | 15.1% | 0.0265 | 14,627 to 15,496 |
| Primary School | 4,390 | -1.48 pp | +0.0564 | 218.24 | 13.6% | 0.0219 | 11,920 to 13,112 |
| Secondary School | 1,523 | -4.73 pp | +0.0869 | 105.38 | 16.1% | 0.0219 | 11,920 to 13,112 |
| University | 911 | -0.33 pp | +0.0339 | 41.62 | 9.0% | 0.0219 | 11,920 to 13,112 |
| NA / Driving Rules | 1,265 | -1.74 pp | +0.0201 | 43.29 | 7.0% | 0.0219 | 11,920 to 13,112 |

Secondary-school questions show the largest aggregate late regression. The
Professional group controls half of the benchmark and selects the cooldown
boundary, which explains why the whole-benchmark late boundary is 14,627 to
15,496 even though four of five level groups select an earlier split.

## Subject findings

- Final choice NLL is worse than at update 9,536 in 29 of 31 subject labels.
  The only improvements are World Religions (`-0.0042`) and Greek Literature
  (`-0.0001`, only 14 questions). Thus the late degradation is broad in the
  continuous probabilities and is not merely an accuracy-argmax artefact.
- Six subjects finish with higher accuracy than at update 9,536, but four of
  those six still have worse choice NLL. Accuracy alone hides weakening answer
  margins.
- Among subjects with at least 100 questions, the largest final accuracy losses
  from update 9,536 are Art (`-7.37 pp`), Greek Traditions (`-5.33 pp`), General
  Knowledge (`-3.99 pp`), Modern Greek Language (`-3.77 pp`), Government and
  Politics (`-3.40 pp`) and Education (`-3.40 pp`).
- Modern Greek Language has the largest large-subject NLL deterioration
  (`+0.1136`) and 151.71 raw post-peak displaced cells. It contributes 22.63 of
  its own cells above its subject-specific 97.5% floor, the largest such
  category-specific excess. Subject-specific excesses are not additive with the
  whole-benchmark excess because each has a different null distribution.
- Full-history displacement is significant after within-subject correction for
  28 of 31 native labels. The three exceptions contain only 14, 20 and 21
  questions. Post-peak displacement is significant for 20 of 31 labels, but
  small-category results should be interpreted by effect size and sample size,
  not by the temporal-order p-value alone.
- Among subjects with at least 100 questions, post-peak displacement share has
  a weighted correlation of `-0.447` with peak-to-final accuracy change. More
  late identity movement is moderately associated with more forgetting. The
  corresponding full-history correlation is only `-0.149`, consistent with
  early adaptation being different from late regression.

## Interpretation

The score plateau is not a stable state. From the accuracy peak to the endpoint,
821 newly correct questions replace 821 formerly correct questions while a
further 316 formerly correct questions are lost. The continuous NLL worsens in
every educational level and nearly every subject, so the peak is not merely a
fortunate collection of argmax ties.

At the same time, most raw post-peak movement is compatible with unordered
checkpoint churn. The historically ordered excess is small compared with the
906-cell raw AHWD. The defensible conclusion is therefore:

1. early CPT causes a large, broad reorganization of what the model answers;
2. the late plateau contains substantial answer churn and broad probability
   degradation;
3. a smaller but measurable directional shift appears near cooldown start;
4. the single trajectory establishes alignment, not causality. A no-cooldown or
   different-decay control is needed to attribute the late shift to WSD.

The permutation test measures temporal ordering conditional on this fixed
benchmark. It is not a question-sampling confidence interval. Very small subject
labels remain descriptive even when their order-permutation test is small.

## Artifacts and execution

- Category JSON: `presentations/data/greekmmlu_response_displacement_20260811/greekmmlu_response_displacement_by_category.json`
- Flat category CSV: `presentations/data/greekmmlu_response_displacement_20260811/greekmmlu_response_displacement_by_category.csv`
- Compact exact response matrix: `presentations/data/greekmmlu_response_displacement_20260811/greekmmlu_exact_responses.npz` (locally retained; ignored by Git's `*.npz` rule)
- Extractor: `analysis/extract_greekmmlu_history_matrix.py`
- Analyzer: `analysis/analyze_greekmmlu_response_displacement_by_category.py`
- Debug-node wrapper: `clariden/analyze_greekmmlu_response_displacement.sbatch`
- Successful Clariden job: `3057942` (`COMPLETED`, 29 seconds)
- Exact response matrix SHA-256: `67641b0c206eb1e35b1cfdf8070d56c2076c83d08ebcabd6d7a1493247a65356`
- Final JSON SHA-256: `74d6f1f888f220aa85a753073f263dafe49b401b11ef557f8df35eecce35037e`
- Final CSV SHA-256: `4b57b0865d6d90118419d5eb85708fe2d76900a6276088b2f802580988e4b7ff`
