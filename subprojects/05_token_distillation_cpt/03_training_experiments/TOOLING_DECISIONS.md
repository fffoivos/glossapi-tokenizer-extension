# Tooling decisions — established vs bespoke, per job

The rule: **use the most well-established script/library for each job**; keep a
bespoke piece only if it's a thin driver, an unavoidable gap-filler, or an
acceptance gate — never if it reimplements the established tool's core. Verdict
key: **KEEP** (thin/necessary), **JUSTIFY** (diverges from upstream for a stated
reason), **REPLACE** (retire for the established tool). Result: **no REPLACE** —
every heavy step already rides an established tool; the bespoke remainder is
plumbing and gates.

| Job | Established tool | Best existing script (adapt) | Bespoke pieces → verdict |
|---|---|---|---|
| **Train** | swiss-ai/Megatron-LM fork `pretrain_gpt.py` @ `c92402e` | trainer `…/init_bakeoff/bakeoff_training/bakeoff_train.sbatch`; config ref `…/04_…/train_config_04_vanilla.env`; chain ref `…/04_…/submit_training_5b_chain.sh` | `pretrain_gpt_te_guard.py` (TE-import shim, exec's stock script) → **KEEP**; arg-array + per-ARM routing → **KEEP**; single pre-weighted `--data-path` → **JUSTIFY** (replay blended upstream so both tokenizations share one doc stream); WSD cooldown window → **fixed** (was hardcoded `$TRAIN_SAMPLES`=100% decay; now `LR_WSD_DECAY_SAMPLES` knob, default unchanged) |
| **TD init** (arm 2) | Dobler **token-distillation** `train_embeddings` @ `35702b5` (arXiv:2505.20133) | `…/token_distillation/train_retok_td.py` + `train_retok_td_layer_pilot_packed.sbatch` | adapter `train_retok_td.py` (calls upstream loop verbatim; routes fixed merge-IDs because upstream `add_tokens` would renumber) → **KEEP**; `learn_output_with_ce=True` + `loss_methods=["MSE-on-hiddens"]` hardcode (the untied-head recipe — E by distill, U by CE) → **KEEP**; CPU snippet/coverage prepass replacing upstream Aho-Corasick builder → **KEEP** (same contract) |
| **Convert HF↔Megatron** | fork `tools/checkpoint/convert.py` + `saver_swissai_hf.py` / `saver_core.py` | `…/megatron_patches/td_layer11_r17_roundtrip.sbatch` (forward, TP=2) | `loader_apertus_hf.py` → **KEEP** (fork ships no Apertus HF loader); `patch_apertus_extras.py` (R17 — restores xIELU/QK-Norm params saver_core drops) → **KEEP**; `run_megatron_convert_with_pg.py` (1-rank PG for trained TP=2 back-leg) → **KEEP**; `verify_hf_roundtrip.py` (acceptance gate) → **KEEP** |
| **Data tokenize** | Megatron `tools/preprocess_data.py` | `…/bakeoff_training/preprocess_data.sbatch` | `mix_builder.py` (text-level token-fair interleaver) → **JUSTIFY** (one shared JSONL → byte-identical doc stream across the two vocabs; Megatron `BlendedDataset` mixes per-`.bin` and can't guarantee that invariant); `preprocess_hf_jsonl_to_megatron.py` (torch-free fallback for no-uenv nodes, byte-checked) → **JUSTIFY**; `normalize_jsonl_nfc.py` → **KEEP but VERIFY** (Apertus is `normalizer:null`); `prepare_greek_pool.sh` (duckdb dedup/decontam) → **KEEP** |
| **Eval** | lm-evaluation-harness (swiss-ai fork) | `…/04_…/submit_checkpoint_sidecars.sh` → 03 eval sbatches | retention `run_eval.sbatch` → **KEEP** (is lm-eval); native Greek MCQ `run_native_greek_mcq_eval.py` → **JUSTIFY** (no native YAMLs exist; avg-logprob argmax ≈ acc_norm); tokenizer-fair BPB `compute_tokenizer_fair_metrics.py` → **JUSTIFY** (byte-denominator comparability across 131072 vs 148480) |
| **Hyperparameters** | (policy doc) `subprojects/CURRENT_HYPERPARAMETERS.md` v1.0 | → `configs/common_cpt.env` | run-relative α/β3 warmup → **JUSTIFY** (Apertus's 100k > our run); Goldfish mask computed in-dataloader by the fork (`gpt_dataset.py`) → **KEEP** (no offline pass; hash uniform over 148480) |

## The one code change made

`bakeoff_train.sbatch`: `--lr-wsd-decay-samples` was hardcoded to `$TRAIN_SAMPLES`
(whole-run decay → no stable phase). Apertus uses a **separate** cooldown count
(`submit_apertus_8b.sh:250 --lr-wsd-decay-samples $COOLDOWN_SAMPLES`). Made it a
backward-compatible env knob `LR_WSD_DECAY_SAMPLES` (unset ⇒ old behavior, so no
past run changes); `common_cpt.env` sets it to 20% of the run. This generalizes
the established wrapper rather than forking it.

## What was verified, not assumed

- The trainer is a **thin wrapper** over the fork's `pretrain_gpt.py`, byte-mirroring Apertus's `submit_apertus_8b.sh` except marked CPT-diffs — not a bespoke trainer.
- TD trains **both** E (MSE-distill, layer 11) and U (CE into `lm_head`), originals frozen and asserted byte-identical — confirmed against `external/token-distillation/.../train_loop.py:245-380`.
- Geometry `rope θ=500k / seq 4096 / llama3 ×8` is Apertus's **main-pretraining** setting (`submit_apertus_8b.sh:188-192`, paper Table C.4) — scaling is **ON**, which the Vanilla-5B run omitted.
- Every flag in `common_cpt.env` traces to either `submit_apertus_8b.sh` (Apertus-faithful) or a marked, justified CPT change.
