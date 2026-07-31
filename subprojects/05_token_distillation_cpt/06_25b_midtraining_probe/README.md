# 25B Greek midtraining probe

This subproject is the production replacement for the stale single-blend
`05_training_dataset_bridge` launch path. It pins the bibliography-cleaned HF
v2 corpus and the 148,992-token modern-plus-polytonic tokenizer, then implements
one continuous 25B optimization run over two randomized data phases.

## Frozen semantics

- Model: Apertus-8B-2509, layer-11 token-distilled initialization.
- Tokenizer: `fffoivos/apertus-tokenizer-extension` at
  `fcd33ec09fb7d86bc072b3a4b3e890efa6473b66`, subfolder
  `greek-modern-polytonic-tokenizer`.
- Dataset: `fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2` at
  `3f97cec48af502f4996cf8ff20b02660e2dd3d31`.
- Effective horizon: 5,960 iterations / 24,998,051,840 tokens.
- Phase 1, iterations 0-3,570: 79% HPLT, 20% foreign replay, 1% old Greek.
- Phase 2, iterations 3,570-5,960: HPLT and non-HPLT Greek are weighted so the
  whole run retains the corpus's natural HPLT share; replay remains 20% + 1%.
- Every phase is randomized with seed `20260609`.
- Document allocation is disjoint across phases. HPLT and replay use a
  domain-separated SHA-256 60/40 split; non-HPLT Greek is phase 2 only.

The switch at iteration 3,570 changes only the data blend. Optimizer,
scheduler, RNG and global iteration must resume normally. The runtime wrapper
converts the checkpoint's global consumed-sample count into a phase-relative
data index on every segment, fixing the historical first-segment-only reset.

## Gates before a training submission

1. Validate the frozen recipe locally:

   ```bash
   python3 scripts_validate_recipe.py
   python3 -m pytest -q tests
   ```

2. On Clariden, materialize and hash the exact gated dataset revision, tokenizer
   tree, GreekMMLU decontamination inputs, replay inputs, clean Megatron commit,
   and production initialization checkpoint.
3. Build heldouts before training binaries, exclude their document identities,
   and apply GreekMMLU decontamination to new Greek and replay pools.
4. Build the two phase manifests and binaries. The finalizer must prove there
   is no retained `doc_id` intersection across phases and that every blend sums
   exactly to one.
5. Run checkpoint-load and two-iteration GPU smokes, including a synthetic
   phase-boundary resume that proves the phase-relative data index.
6. Freeze all launch assets and checkpoint receipts. Only then prepare the
   16-node dry run. The 64-GPU training submission remains a separate explicit
   launch gate.

Iterations 1,785 and 3,570 are periodic-save boundaries. The exact final
iteration 5,960 is ten iterations after the last periodic save, so the final
segment must force and receipt a terminal checkpoint rather than silently
rounding the token horizon to 5,950.

The exact configuration is in
[`configs/recipe_25b_midtraining.json`](configs/recipe_25b_midtraining.json).

## Clariden preparation control

All operational work runs from a clean immutable checkout on Clariden. The
preparation submitter never launches the 64-GPU training run:

```bash
export REPO_ROOT=/iopsstor/scratch/cscs/fffoivos/repo/<clean-cpt-checkout>
export CPT_RUN_ID=<immutable-run-id>

# Inspect only.
DRY_RUN=1 clariden/submit_data_pipeline.sh prereqs

# Start the receipt-bound runtime, exact HF v2 materialization, replay restore,
# GreekMMLU freeze, uncpt TokenDistil-Init rebuild, and their dependent checks.
DRY_RUN=0 CONFIRM_PREPARATION=1 clariden/submit_data_pipeline.sh prereqs

# Once input_receipt.json exists, split the binary build into arrays of at most
# 1,001 tasks (Clariden's current MaxArraySize) and finalize both phase blends.
DRY_RUN=0 CONFIRM_PREPARATION=1 clariden/submit_data_pipeline.sh after-freeze

# After the bridge and zero-drift Megatron roundtrip both pass.
DRY_RUN=0 CONFIRM_PREPARATION=1 clariden/submit_data_pipeline.sh assets
```

`prereqs` deliberately rebuilds the appended 512 rows from the uncpt
`TokenDistil-Init` checkpoint. The earlier `TokenDistil-3.5B` cutoff-probe
checkpoint is CPT-trained and is explicitly forbidden as production
initialization. `clariden/submit_data_pipeline.sh status` reports every receipt
and live preparation job without changing state.
