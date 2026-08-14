# Full 8B CPT results analysis

This subproject is the compact results authority for the completed sanitized
Apertus-8B CPT trajectory. It is organized around three questions only:

1. **Adaptation and retention:** what did the model learn, and what did it
   forget, over the full 76.685B-active-token run?
2. **Capability timing:** when did native-Greek benchmark performance peak,
   and did that agree with heldout loss?
3. **Proxy validity:** did the five 0.5B data-order arms predict the behavior
   of the 8B stationary-mix run?

[`RESULTS.md`](RESULTS.md) answers those questions. The three self-contained
HTML reports under [`presentations/`](presentations/) are the canonical visual
evidence. [`DATA_AND_LIMITATIONS.md`](DATA_AND_LIMITATIONS.md) records the
dataset and comparability boundaries that constrain interpretation.

## Ownership boundary

- `07_full_8b_cpt` remains authoritative for the training recipe, immutable
  bundles, Slurm orchestration, checkpoints, raw evaluation receipts and the
  completed CSCS run.
- This subproject owns post-hoc conclusions, compact analysis code and the
  canonical presentations.
- `06_dataset_scheduling_experiments` remains authoritative for the five 0.5B
  trajectories used in the scale comparison.
- `08_targeted_8b_cpt_experiments` owns proposed follow-up experiments. It is
  not evidence about this completed run.

Synthetic drift simulations, alternative displacement statistics, exploratory
prefix reports and progress snapshots are intentionally not copied here. They
did not change the three decisions above and remain historical working material
in subproject 07.

## Canonical artifacts

| Question | Presentation | Machine-readable payload |
| --- | --- | --- |
| Full 8B adaptation and retention | [`FULL8_RESULTS.html`](presentations/FULL8_RESULTS.html) | [`FULL8_RESULTS.data.json`](presentations/FULL8_RESULTS.data.json) |
| 8B versus five 0.5B arms | [`FULL8_VS_0P5B.html`](presentations/FULL8_VS_0P5B.html) | [`FULL8_VS_0P5B.data.json`](presentations/FULL8_VS_0P5B.data.json) |
| Checkpoint behavior and source exposure | [`CHECKPOINT_BEHAVIOR.html`](presentations/CHECKPOINT_BEHAVIOR.html) | [`CHECKPOINT_BEHAVIOR.data.json`](presentations/CHECKPOINT_BEHAVIOR.data.json) |
| Native-Greek three-checkpoint benchmark screen | [`NATIVE_GREEK_3CP_BENCHMARKS.html`](presentations/NATIVE_GREEK_3CP_BENCHMARKS.html) | [`NATIVE_GREEK_3CP_BENCHMARKS.data.json`](presentations/NATIVE_GREEK_3CP_BENCHMARKS.data.json) |

The token-aligned D0 0.5B replication of the native-Greek screen is reported
in [`D0_0P5B_VS_FULL8_NATIVE_GREEK_3CP_20260814.md`](D0_0P5B_VS_FULL8_NATIVE_GREEK_3CP_20260814.md),
with its machine-readable comparison in the adjacent `.data.json` file.

The payloads retain their original CSCS receipt paths and hashes. Verify the
copied package with:

```bash
python3 subprojects/09_full_8b_cpt_results_analysis/verify_bundle.py
```

## Next evaluation work

[`NATIVE_GREEK_BENCHMARKS.md`](NATIVE_GREEK_BENCHMARKS.md) defines a staged
native-Greek evaluation panel. Multiple-choice and classification datasets can
be scored over all 19 saved checkpoints. Free-generation and critic-based
benchmarks should first use seven decision checkpoints because their cost and
prompt sensitivity are materially higher.

The first expanded screen is frozen under [`evaluation/`](evaluation/): it
scores the iteration-zero, approximately 40B-token and terminal checkpoints on
DemosQA, Medical MCQA, ASEP MCQA, GPCR and the four OYXOY task views. Text-only
Protipa is contract-pinned but remains blocked on its manual Hugging Face gate.
The completed full-versus-filtered results are recorded in
[`NATIVE_GREEK_3CP_RESULTS_20260812.md`](NATIVE_GREEK_3CP_RESULTS_20260812.md),
and the exact evidence rule and 10,076 scored-example exclusions are frozen in
[`evaluation/CONTAMINATION_DROP_DECISION_20260812.md`](evaluation/CONTAMINATION_DROP_DECISION_20260812.md).
