# TokenDistil-3.5B

Selected checkpoint for the Apertus Greek tokenizer-extension experiments.

Status: HF weight upload pending from Clariden.

Source HF-format copy on Clariden:

```text
/capstor/scratch/cscs/fffoivos/runs/eval/continuation_3p5b_20260524T143012Z_td_layer11/iter_0000834_hf
```

Source Megatron checkpoint on Clariden:

```text
/capstor/scratch/cscs/fffoivos/runs/bakeoff/continuation_3p5b_20260524T143012Z_td_layer11/checkpoints/iter_0000834
```

Technical notes:

- method: Token Distillation;
- distillation target layer: `11`;
- continuation point: about `3.5B` tokens;
- exact iteration: `834`;
- run tag: `continuation_3p5b_20260524T143012Z_td_layer11`;
- tokenizer: `selected-tokenizer/`;
- dataset: `dataset/`.

Evaluation summary:

```text
../../evals/3.5B-comparison/
```
