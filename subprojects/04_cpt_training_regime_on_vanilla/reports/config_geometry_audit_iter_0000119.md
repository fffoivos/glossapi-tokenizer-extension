# Config Geometry Audit - Vanilla-0.5B / Iter 119

Generated UTC: `2026-05-28T21:01:32Z`

This audit records the positional/RoPE geometry actually used by the running
04 Vanilla CPT experiment and the converted iter-119 HF checkpoint. It exists
because the adversarial review correctly flagged that the live checkpoint is
not in the same positional configuration as the official local HF base config.

## Evidence

| Source | Max position | RoPE theta | RoPE scaling | Notes |
| --- | ---: | ---: | --- | --- |
| `/iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509/config.json` | 65536 | 12000000 | llama3 factor 8.0 | Official local HF base config. |
| `goal/hyperparameters.json` | 65536 in architecture metadata; sequence length 4096 for this run | 500000 in architecture metadata | not specified | The run locks `sequence.length=4096`. |
| live `training_command.sh` | 4096 | 500000 | not specified | Uses `--max-position-embeddings 4096`, `--rotary-base 500000`, and `--seq-length 4096`. |
| `iter_0000119_hf/config.json` | 4096 | 500000 | null | Converted HF checkpoint config. |

The training template also states that `--max-position-embeddings` is the
pretrain value `4096`, not the 64K long-context extension, and declares:

```text
--max-position-embeddings $SEQ_LEN
--position-embedding-type rope
--rotary-base 500000
--seq-length $SEQ_LEN
```

## Conclusion

The iter-119 checkpoint is a base-vocabulary Vanilla continuation artifact, but
its geometry is best described as `bakeoff-local-4096-rope500k`, not the
official HF long-context geometry from `/iopsstor/.../apertus-8b-2509/config.json`.

This does not by itself stop the current run, because the active goal and
`goal/hyperparameters.json` lock the experiment to sequence length 4096.
However, the final 5B report must not compare this checkpoint to an official
HF Apertus-Base result as if only CPT changed. It must either:

1. compare against an Apertus-Base baseline evaluated under the same
   4096/RoPE-500K geometry and same harness, or
2. explicitly mark the positional geometry difference as a confound.

