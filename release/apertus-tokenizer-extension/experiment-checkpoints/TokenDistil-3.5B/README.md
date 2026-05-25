# TokenDistil-3.5B

Selected checkpoint for the Apertus Greek tokenizer-extension experiments.

Source HF-format copy on Clariden:

```text
/capstor/scratch/cscs/fffoivos/runs/eval/continuation_3p5b_20260524T143012Z_td_layer11/iter_0000834_hf
```

Technical notes:

- method: TokenDistil;
- target layer: `11`;
- training point: about `3.5B` tokens;
- exact iteration: `834`;
- tokenizer: `greek-extension-tokenizer/`;
- dataset: `cpt-training-dataset/`.

Evaluation summary:

```text
../../benchmark-evals/3.5B-comparison/
```
