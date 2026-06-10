# Token Distillation — initializing BOTH E and U the repo's way

Arm 2 adds 17,408 modern-Greek tokens (ids `[131072, 148480)`). Their input
embeddings **E** and output/unembedding rows **U** are initialized with **Token
Distillation** (Dobler et al., *Token Distillation*, ICLR 2026, arXiv:2505.20133),
the established method — not a reimplementation. Apertus has
`tie_word_embeddings=False`, so E and U are **separate matrices and both must be
produced**; the repo handles each with a different objective (its untied-head
recipe), and our wrapper calls that loop verbatim.

## Established code

- Repo: `…/init_bakeoff/token_distillation/external/token-distillation` @ upstream `35702b5`.
- Load-bearing function: `token_distillation.train_loop.train_embeddings`.
- Our adapter: `…/init_bakeoff/token_distillation/train_retok_td.py` — it
  `importlib`-loads and calls `train_embeddings` **unchanged**. The adapter
  exists only because upstream `TokenDistillation.run()` calls
  `target_tokenizer.add_tokens(...)`, which re-assigns token IDs and would break
  our fixed merge-rule extension `[131072,148480)`. The adapter routes the
  fixed-ID tokens; it does not touch the E/U math.

## How E and U are each produced (the untied-head recipe)

Both rows are made trainable, then updated by **two different objectives**, with
the original rows frozen by gradient surgery (`train_loop.py`):

| Matrix | Objective | Mechanism (line) |
|---|---|---|
| **E** (input `embed_tokens.weight`) | **MSE-on-hiddens distillation** at `target_layer=11`: match merged-sequence vs unmerged-sequence hidden states | `:288-300` compute MSE; `:352` `loss.backward(inputs=[get_input_embeddings().weight])` — flows ONLY into E |
| **U** (output `lm_head.weight`) | **next-token cross-entropy**, backpropped ONLY into the head | `:339-348` `learn_output_with_ce=True` → `autograd.backward([ce_loss], inputs=[get_output_embeddings().weight])` |
| both | trainable | `:245-246` `requires_grad=True` for input + output embeddings |
| originals | **frozen** | `:354-359` zero grads of `original_token_ids` on BOTH matrices; `:372-380` assert both original blocks byte-identical post-train |

`learn_output_with_ce=True` and `loss_methods=["MSE-on-hiddens"]` are fixed in
`train_retok_td.py:355` — so a launch can never silently disable U-learning.
This is exactly the paper's recommendation for untied heads: **distill E, learn U
by CE.** (The earlier "TD distills E and learns U" wording in
`CURRENT_HYPERPARAMETERS.md §4` is therefore operationally correct for our recipe,
even though TD-the-published-method names only the E-distillation step.)

## Command (layer 11, full 17,408-token list)

```bash
cd /iopsstor/scratch/cscs/fffoivos/repo/03_…/init_bakeoff/token_distillation
COVERAGE_DIR=/iopsstor/scratch/cscs/fffoivos/token_distillation/coverage_2b_modern_20260523T032424Z_nfc
sbatch --export=ALL,\
COVERAGE_JSONL="$COVERAGE_DIR/td_coverage_prepass.jsonl",\
SNIPPETS_JSONL="$COVERAGE_DIR/td_snippet_index/snippets.jsonl",\
TOKEN_IDS_FILE="$COVERAGE_DIR/pilot_selection/full_td_token_ids.txt",\
OUTPUT_ROOT=/iopsstor/scratch/cscs/fffoivos/token_distillation/retok_td_full_$(date -u +%Y%m%dT%H%M%SZ),\
TARGET_LAYERS="11",MAX_SELECTED_TOKENS=0,SNIPPETS_PER_TOKEN=25,\
MIN_ACCEPTED_SNIPPETS_PER_TOKEN=25,MIN_TRAINED_TOKEN_FRACTION=0.99,BATCH_SIZE=8 \
train_retok_td_layer_pilot_packed.sbatch
```

- `TARGET_LAYERS="11"` runs only the chosen layer (drops the last-layer
  comparison arm; frees 3 of 4 GPUs). Bump `--time` past the recorded ~4.5 h
  layer-11 wall.
- TD starts the new rows from the **ReTok** init checkpoint (subword-mean), then
  relearns them; ~15–31 tokens with too few base-phrase snippets keep their ReTok
  init (gate `MIN_TRAINED_TOKEN_FRACTION=0.99` passes at 17,377/17,408).
- After training, convert to the Megatron TP=2 init checkpoint:
  `…/megatron_patches/td_layer11_r17_roundtrip.sbatch` with
  `HF_DIR=$OUTPUT_ROOT/layer11` (forward HF→Megatron + R17 patch — see BUILD_PLAN §0/§1).

## Reuse shortcut

The existing `…/init_checkpoints/modern_only_148480/td/.../td_full25_layer11_r17_roundtrip_2357565/megatron_tp2_r17patched`
(roundtrip max-abs-diff 0.0, all 17,377 trained rows changed on **both** E and U,
preserved rows byte-exact) can be consumed **directly** by arm 2 **iff** its
tokenizer is byte-identical to `EXT_TOKENIZER_DIR`
(`apertus_greek_modern_only_148480`). Verify first; if the shipped tokenizer
changed, re-run TD with the command above.
