# Checkpoints

This directory is reserved for actual model weights.

Current status: no checkpoint weight shards are uploaded in this HF release yet.
Pointer-only entries were moved to `../locations/` so `checkpoints/` does not
pretend to contain weights.

The first payload to upload should be:

```text
checkpoints/TokenDistil-3.5B/
```

Current source for that HF-format payload on Clariden:

```text
/capstor/scratch/cscs/fffoivos/runs/eval/continuation_3p5b_20260524T143012Z_td_layer11/iter_0000834_hf
```

Expected contents for a loadable HF checkpoint include:

```text
config.json
generation_config.json
model-00001-of-0000N.safetensors
model.safetensors.index.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
README.md
manifest.json
```
