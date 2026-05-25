# Hydration

This release entry is a pointer, not a checkpoint payload.

On Clariden, use:

```bash
ls -lh /capstor/scratch/cscs/fffoivos/runs/bakeoff/continuation_3p5b_20260524T143012Z_td_layer11/checkpoints/iter_0000834
ls -lh /capstor/scratch/cscs/fffoivos/runs/eval/continuation_3p5b_20260524T143012Z_td_layer11/iter_0000834_hf
```

Before publishing this as loadable HF weights, verify the HF eval copy contains
the expected config, tokenizer, and safetensor shards, then upload only that
inference-ready payload.

