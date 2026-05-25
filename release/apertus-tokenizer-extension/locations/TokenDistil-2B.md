# TokenDistil-2B

Human name: `TokenDistil-2B`.

Weights status: not uploaded in this HF release.

Clariden Megatron checkpoint:

```text
/capstor/scratch/cscs/fffoivos/runs/bakeoff/td_full25_layer11_2b_20260523T165038Z/checkpoints/iter_0000476
```

Clariden eval run root:

```text
/capstor/scratch/cscs/fffoivos/runs/eval/td_full25_layer11_2b_20260523T165038Z
```

Tokenizer: `ModernGreek-148k`.

Training data: `CPT-7B-mix`.

Technical notes:

- Token Distillation target layer: `11`;
- exact iteration: `476`;
- run tag: `td_full25_layer11_2b_20260523T165038Z`;
- format: Megatron `torch_dist`, TP=2;
- initialized from `TokenDistil-Init`.

Detailed run evidence is in:

```text
../provenance/token-distillation/RUN_LOG_20260523.md
```
