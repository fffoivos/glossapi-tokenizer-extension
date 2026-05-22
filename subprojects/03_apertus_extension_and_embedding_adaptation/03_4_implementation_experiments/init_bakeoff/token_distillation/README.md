# Token Distillation Prep

This folder holds the bounded ReTok + Token Distillation challenger work. It is
not part of the default production path unless its gates beat the current
Vanilla decision state.

## First gate: CPU coverage prepass

Run this on Clariden `xfer`, not on a GPU partition:

```bash
cd /iopsstor/scratch/cscs/fffoivos/repo/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/token_distillation
sbatch td_coverage_prepass_xfer.sbatch
```

The prepass scans the mixed JSONL stream in order and stops after the first
2,000,000,000 extended-token emissions by default. A firing only counts when the
extended tokenizer actually emits a new token ID in `[131072, 148480)`.
Substring matches and BPE merge ancestry do not count.

Expected outputs:

- `td_coverage_prepass.jsonl` - one row per new token
- `td_coverage_summary.json` - aggregate thresholds and recommendation
- `td_snippet_index/snippets.jsonl` - sampled snippet references/text

Decision rule:

- `>= 90%` of new tokens with 100 usable snippets: run full TD at 100 snippets.
- `>= 90%` with 25 usable snippets: run the paper-fast TD setting and flag the
  tail.
- otherwise: do not launch full TD; inspect coverage/tokenizer mismatch first.

This prepass is intentionally CPU-only. Any dataset or snippet-building rerun
belongs on `xfer`.
