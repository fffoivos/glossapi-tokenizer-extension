# Vanilla-5B

Original Apertus tokenizer control after the 5.0B continuation.

Source HF-format copy on Clariden:

```text
/capstor/scratch/cscs/fffoivos/runs/eval/continuation_5b_td_vs_vanilla_20260525T142522Z_vanilla/iter_0001192_hf
```

Technical notes:

- method: Vanilla (no tokenizer extension; base 131,072 vocab);
- training point: about `5.0B` tokens;
- exact iteration: `1192`;
- tokenizer: original Apertus tokenizer (`swiss-ai/Apertus-8B-2509`);
- dataset: `cpt-training-dataset/`.

Evaluation summary:

```text
../../benchmark-evals/bakeoff-final/
```

Continuation lineage:

- iter 476 (2.0B) → `Vanilla-2B/`
- iter 834 (3.5B) → `Vanilla-3.5B/`
- iter 1192 (5.0B) → this checkpoint
