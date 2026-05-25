# TokenDistil-3.5B

Human name: `TokenDistil-3.5B`.

Weights status: selected payload path is `../TokenDistil-3.5B/`.

Intended HF weights path:

```text
checkpoints/TokenDistil-3.5B/
```

Clariden Megatron checkpoint:

```text
/capstor/scratch/cscs/fffoivos/runs/bakeoff/continuation_3p5b_20260524T143012Z_td_layer11/checkpoints/iter_0000834
```

Clariden HF-format eval copy:

```text
/capstor/scratch/cscs/fffoivos/runs/eval/continuation_3p5b_20260524T143012Z_td_layer11/iter_0000834_hf
```

Tokenizer: `ModernGreek-148k`.

Training data: `CPT-7B-mix`.

Technical notes:

- Token Distillation target layer: `11`;
- exact iteration: `834`;
- run tag: `continuation_3p5b_20260524T143012Z_td_layer11`;
- format: Megatron `torch_dist`, TP=2, plus HF eval conversion;
- initialized from `TokenDistil-Init`.

Result summary:

```text
../../evals/3.5B-comparison/README.md
```
