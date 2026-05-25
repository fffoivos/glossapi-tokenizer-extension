# ReTok-3.5B

Human name: `ReTok-3.5B`.

Weights status: not uploaded in this HF release.

Clariden Megatron checkpoint:

```text
/capstor/scratch/cscs/fffoivos/runs/bakeoff/continuation_3p5b_20260524T143012Z_retok/checkpoints/iter_0000834
```

Clariden eval run root:

```text
/capstor/scratch/cscs/fffoivos/runs/eval/continuation_3p5b_20260524T143012Z_retok
```

Tokenizer: `ModernGreek-148k`.

Training data: `CPT-7B-mix`.

Technical notes:

- exact iteration: `834`;
- run tag: `continuation_3p5b_20260524T143012Z_retok`;
- format: Megatron `torch_dist`, TP=2;
- initialization: retokenization/subpiece-mean baseline.

Result summary:

```text
../results/3.5B-comparison/README.md
```
