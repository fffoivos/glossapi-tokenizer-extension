# Checkpoints

This is the top-level checkpoint area.

Primary checkpoint:

```text
TokenDistil-3.5B/
```

This folder is for the selected Token Distillation checkpoint after the 3.5B
continuation. The large weight shards are uploaded to Hugging Face only; they
are not mirrored in the GitHub source repo.

Supporting checkpoint locations:

```text
locations/
```

Those files record the Clariden source paths for `TokenDistil-Init`,
`TokenDistil-2B`, `Vanilla-3.5B`, and `ReTok-3.5B`.
