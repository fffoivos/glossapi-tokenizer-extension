# TokenDistil-5B

Selected Token Distillation checkpoint after the 5.0B continuation. Bakeoff-final downstream winner.

Source HF-format copy on Clariden:

```text
/capstor/scratch/cscs/fffoivos/runs/eval/continuation_5b_td_vs_vanilla_20260525T142522Z_td_layer11/iter_0001192_hf
```

Technical notes:

- method: TokenDistil;
- target layer: `11`;
- training point: about `5.0B` tokens;
- exact iteration: `1192`;
- tokenizer: `greek-extension-tokenizer/` (ModernGreek-148k);
- dataset: `cpt-training-dataset/`.

Evaluation summary:

```text
../../benchmark-evals/bakeoff-final/
```

Continuation lineage:

- iter 476 (2.0B) → `TokenDistil-2B/`
- iter 834 (3.5B) → `TokenDistil-3.5B/`
- iter 1192 (5.0B) → this checkpoint

Caveat: at 5B, TokenDistil-5B leads the downstream Greek no-MT aggregate, English retention, and Multilingual aggregates over Vanilla-5B, but Vanilla still leads tokenizer-fair heldout BPB. The pre-commit decision-rule thresholds from the v0.12 experimental plan (X / M_progress / M_ext / M_van / T) were not locked before results came in, so the bakeoff produced data, not a rule-bound winner. See `../../supporting-material/provenance/decisions/PLAN_VS_RESULTS_RECONCILIATION_20260526.md`.
