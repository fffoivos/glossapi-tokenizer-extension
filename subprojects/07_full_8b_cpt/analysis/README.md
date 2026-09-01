# 07 · analysis — why GreekMMLU peaked at 40B and then fell back

> **In one line:** a post-hoc investigation of the run's one uncomfortable result — the benchmark peaked mid-run — which found that the dominant effect is *early* response reorganization, with a secondary post-peak boundary sitting exactly at cooldown start.
> **Period:** 2026-08-11 → 2026-08-12. **Status:** completed; conclusions carried into subproject 09 and motivated subproject 10.
> **Came from / led to:** the 19 frozen GreekMMLU prediction payloads → this → [`../../09_full_8b_cpt_results_analysis/RESULTS.md`](../../09_full_8b_cpt_results_analysis/RESULTS.md) §2 and [`../../10_early_cooldown_causal_experiment/`](../../10_early_cooldown_causal_experiment/).

## Why this existed

The completed run's GreekMMLU accuracy rose from 35.782% to 56.810% at update 9,536 (≈40B active tokens) and then fell to 54.855% at the endpoint, while source-conditioned Greek loss kept improving. "The benchmark got worse" is not by itself a finding: accuracy can fall while the model's answers barely move, or stay flat while they churn. The question was whether the *responses* reorganized, when, and whether the change is bigger than an accuracy-matched null would produce.

## Method

[`analyze_greekmmlu_response_displacement_by_category.py`](analyze_greekmmlu_response_displacement_by_category.py), with [`extract_greekmmlu_history_matrix.py`](extract_greekmmlu_history_matrix.py) rebuilding a compact response matrix from the hash-bound prediction payloads. The statistic is **Absolute Historical Wrong-cell Displacement (AHWD)**: within each category a nested accuracy-only baseline matches the exact correct count at every checkpoint, with question order frozen from iteration-0 choice NLL and example ID as the deterministic tie-breaker, so AHWD measures response-identity movement *beyond* the change in aggregate accuracy. The checkpoint-order null uses 1,999 permutations; subject and level tests use Benjamini–Hochberg correction within their axes. The candidate statistics were compared first in the bake-off and simulation reports under [`../presentations/`](../presentations/).

Companion scripts: [`analyze_greekmmlu_answer_drift.py`](analyze_greekmmlu_answer_drift.py) (the frozen answer-drift receipt) and [`analyze_checkpoint_source_exposure.py`](analyze_checkpoint_source_exposure.py) (pairing drift against what each checkpoint had actually seen).

## Findings — [`GREEKMMLU_RESPONSE_DISPLACEMENT_BY_CATEGORY_20260811.md`](GREEKMMLU_RESPONSE_DISPLACEMENT_BY_CATEGORY_20260811.md)

Evidence base: 19 checkpoints × 16,159 fixed questions = 307,021 response records. Recomputed accuracy matches the frozen receipt exactly; recomputed choice NLL and correct-answer BPB match to `8.22e-15` and `2.11e-15`.

- **Churn is large.** 18.68% of questions were correct at every checkpoint and 23.71% wrong at every checkpoint, but 57.61% changed correctness at least once and **68.79% changed selected answer at least once**. Peak-to-final: 821 newly correct, 1,137 newly wrong — 821 paired replacements plus 316 net additional errors.
- **The dominant effect is early.** Full-run AHWD is 1,415.40 equivalent wrong cells (19.50% of available wrong mass) against a 97.5% permutation floor of 1,080.47 (`p = 0.0005`); the strongest boundary is updates 3,576 → 4,768, ≈15–20B active tokens, and **27 of 31 subject labels select that same boundary**.
- **The post-peak boundary is at cooldown.** Restricted to the ten checkpoints from the accuracy peak to the endpoint, AHWD is 906.42 cells (12.60%) against a floor of 871.14 (`p = 0.0095`), and the strongest boundary is 14,627 → 15,496 — update 14,627 is the declared WSD cooldown start. The document states plainly that this alignment "does not by itself prove that the learning-rate decay caused it."
- All five educational levels regress in both accuracy and choice NLL from peak to endpoint, and all five show significant post-peak displacement after within-axis correction.

## Outcome

- Subproject 09 adopted the operational conclusion: preserve the update-9,536 checkpoint as the observed leader, and do not select future checkpoints from GreekMMLU accuracy alone.
- The cooldown-aligned boundary is descriptive only, which is exactly why a matched control was needed — subproject 10's causal early-cooldown experiment.

## Working documents

Four scripts and one report. Their Slurm wrappers are `../clariden/analyze_greekmmlu_drift_and_exposure.sbatch`, `../clariden/analyze_greekmmlu_response_displacement.sbatch` and `../clariden/extract_greekmmlu_history_matrix.sbatch`.
