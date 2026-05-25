# Vanilla-3.5B

Human name: `Vanilla-3.5B`.

Weights status: location only in this HF release.

Clariden Megatron checkpoint:

```text
/capstor/scratch/cscs/fffoivos/runs/bakeoff/continuation_3p5b_20260524T143012Z_vanilla/checkpoints/iter_0000834
```

Clariden eval run root:

```text
/capstor/scratch/cscs/fffoivos/runs/eval/continuation_3p5b_20260524T143012Z_vanilla
```

Tokenizer: original `swiss-ai/Apertus-8B-2509` tokenizer.

Training data: `CPT-7B-mix`.

Technical notes:

- exact iteration: `834`;
- run tag: `continuation_3p5b_20260524T143012Z_vanilla`;
- format: Megatron `torch_dist`, TP=2;
- this arm remains the heldout-BPC reference in the 3.5B comparison.

Result summary:

```text
../../evals/3.5B-comparison/README.md
```
