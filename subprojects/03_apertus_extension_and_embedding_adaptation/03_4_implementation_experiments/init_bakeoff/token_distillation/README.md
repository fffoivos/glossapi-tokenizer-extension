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

Runtime note: `xfer` nodes are x86_64 and do not expose the `uenv` CLI used on
GH200/normal nodes. The sbatch therefore uses a tiny xfer-built Python 3.11 venv
at `/iopsstor/scratch/cscs/fffoivos/python_envs/td_coverage_py311_xfer` with
`tokenizers==0.22.1`.

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

After the prepass finishes, select bounded smoke and layer-pilot token sets:

```bash
python3 select_td_pilot_tokens.py \
  --coverage-jsonl "$OUTPUT_DIR/td_coverage_prepass.jsonl" \
  --summary-json "$OUTPUT_DIR/td_coverage_summary.json" \
  --output-dir "$OUTPUT_DIR/pilot_selection"
```

This writes:

- `td_pilot_token_selection.json`
- `smoke_token_ids.txt`
- `layer_pilot_token_ids.txt`

Also render a reviewer-readable coverage report:

```bash
python3 summarize_td_coverage.py \
  --coverage-jsonl "$OUTPUT_DIR/td_coverage_prepass.jsonl" \
  --summary-json "$OUTPUT_DIR/td_coverage_summary.json" \
  --output-md "$OUTPUT_DIR/TD_COVERAGE_SUMMARY.md"
```

## Second gate: TD smoke, only if coverage passes

Do not submit this until `td_coverage_summary.json` recommends either
`run_full_td_100` or `run_td_25_with_flagged_tail`.

Example smoke launch:

```bash
cd /iopsstor/scratch/cscs/fffoivos/repo/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/token_distillation
OUTPUT_DIR=/iopsstor/scratch/cscs/fffoivos/token_distillation/retok_td_smoke_last_layer_$(date -u +%Y%m%dT%H%M%SZ)
COVERAGE_DIR=/iopsstor/scratch/cscs/fffoivos/token_distillation/coverage_2b_modern_20260523T012000Z_r2
sbatch --export=ALL,\
COVERAGE_JSONL="$COVERAGE_DIR/td_coverage_prepass.jsonl",\
SNIPPETS_JSONL="$COVERAGE_DIR/td_snippet_index/snippets.jsonl",\
TOKEN_IDS_FILE="$COVERAGE_DIR/pilot_selection/smoke_token_ids.txt",\
OUTPUT_DIR="$OUTPUT_DIR",\
TARGET_LAYER=-1,\
SNIPPETS_PER_TOKEN=25 \
train_retok_td.sbatch
```

The wrapper trains only selected new input/output rows. Every base row and every
unselected new row is gradient-zeroed and exact-checked by the vendored training
loop.

## Pinned Token Distillation code

The official implementation is vendored under
`external/token-distillation/` at upstream commit
`35702b5809599ecd68b7845eca27a0d7b7cec0da`; see
`external/token-distillation/PINNED_UPSTREAM.md`.

For Apertus, do not call the package's high-level
`TokenDistillation.run(...)` path because it appends tokens with
`add_tokens(...)`. Our ReTok tokenizer is already merge-extended with fixed
IDs. The Apertus adapter should load the exact student tokenizer/checkpoint and
call the lower-level training loop with an explicit
`base_phrase_ids -> new_token_id` mapping.
