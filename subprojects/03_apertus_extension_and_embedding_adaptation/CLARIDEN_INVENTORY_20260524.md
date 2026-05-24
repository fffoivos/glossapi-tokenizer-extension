# Clariden inventory — what's on disk and where

2026-05-24. Source-of-truth snapshot of the project's files on Clariden (user `fffoivos`, account `a0140`). Use this map to find: model checkpoints (Apertus base, init arms, TD, post-CPT), datasets (raw nanochat, replay, mixed JSONL, Megatron .bin/.idx), tokenizers, eval outputs, and code/envs. Read before scheduling new compute that might overlap or recreate any of these.

## TL;DR — top-level usage

| Filesystem | Path | Size | What |
|---|---|---:|---|
| capstor | `/capstor/scratch/cscs/fffoivos/runs/bakeoff/` | **5.1 TB** | Bakeoff + TD training runs (checkpoints every ~250 M tokens, ×4 arms) |
| iopsstor | `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/` | **654 GB** | Raw + intermediate + final corpus artifacts |
| capstor | `/capstor/scratch/cscs/fffoivos/runs/eval/` | **480 GB** | All eval outputs (V4 baselines + per-iter per-arm + TD) |
| iopsstor | `/iopsstor/scratch/cscs/fffoivos/repo/` | 273 GB | Mirrored repo + py cache |
| iopsstor | `/iopsstor/scratch/cscs/fffoivos/init_checkpoints/` | **168 GB** | Three init arms × {HF, Megatron, Megatron-TP2, Megatron-TP2-R17patched} |
| iopsstor | `/iopsstor/scratch/cscs/fffoivos/token_distillation/` | **125 GB** | TD prep, snippets, intrinsic evals, TD model outputs, R17-patched Megatron |
| iopsstor | `/iopsstor/scratch/cscs/fffoivos/code/` | 112 GB | Megatron-LM-Swiss-AI clone + pretrain-code |
| iopsstor | `/iopsstor/scratch/cscs/fffoivos/models/` | 16 GB | Apertus-8B-2509 base (HF safetensors) |
| iopsstor | `/iopsstor/scratch/cscs/fffoivos/python_envs/` | 1.5 GB | lm_eval target install + TD coverage Python envs |
| iopsstor | `/iopsstor/scratch/cscs/fffoivos/tokenizers/` | 9.2 MB | Extended modern-only 148,480 ship bundle |
| capstor | `/capstor/scratch/cscs/fffoivos/runs/preprocess/` | 287 MB | preprocess sbatch logs + small intermediates |
| capstor | `/capstor/scratch/cscs/fffoivos/runs/init/` | 440 KB | init sbatch logs |

Grand total: **~6.9 TB** of project state on Clariden.

## 1. Models + tokenizers

| Item | Path | Size | Notes |
|---|---|---:|---|
| Apertus-8B-2509 base (HF) | `/iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509/` | 16 GB | The teacher / V4-HF reference; pretrained Apertus checkpoint |
| Extended modern-only tokenizer | `/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_only_148480/` | 9.2 MB | 148,480-vocab merge-rule extension; the production extended tokenizer |

## 2. Corpus + datasets

Sequence from raw to canonical:
1. Raw HF pulls (Greek nanochat, replay-by-lang, code, math) → `nanochat/`, `replay/`, `code/`, `math/`.
2. Apertus-overlap drop applied → `cpt/selected_after_apertus_and_internal_dedup.parquet`.
3. NFC normalization in place (V9) → same parquet, atomically replaced.
4. Bucket-preserving mix-build (mix_builder.py) → `bulk_mix.jsonl` (also `bulk_mix.nfc.jsonl` post-NFC reroll).
5. Megatron preprocess → `bulk_mix_*_megatron/bulk_mix_text_document.{bin,idx}`.

### Raw + intermediate (still on disk)

| Item | Path | Size |
|---|---|---:|
| Greek nanochat raw parquets + dedup metadata | `cpt_corpus/nanochat/` | 144 GB |
| Selected post-Apertus-drop + post-internal-dedup pool (NFC) | `cpt_corpus/cpt/selected_after_apertus_and_internal_dedup.parquet` | 121 GB |
| Replay parquets (24 langs, single shard per lang) | `cpt_corpus/replay/` | 53 GB |
| Apertus-overlap drop overlay | `cpt_corpus/apertus_overlap_overlay/` | 116 MB |
| Heldout Greek eval set (500 docs, NFC) | `cpt_corpus/heldout/cpt_greek_heldout_500_20260522.jsonl` | 33 MB |

### Mix JSONLs

| Item | Path | Size | State |
|---|---|---:|---|
| Canonical NFC-safe mix JSONL | `cpt_corpus/bulk_mix.nfc.jsonl` | 41 GB | Production-grade |
| Pre-NFC canonical mix JSONL | `cpt_corpus/bulk_mix.jsonl` | 41 GB | Superseded by `.nfc.jsonl` |
| Per-shard bucket-fix part files | `cpt_corpus/bulk_mix_bucketfix_part_0[0-6].jsonl` | 6×5.9 GB | Intermediate shards (concatenated into bulk_mix.jsonl) |
| Code/local-replay shards | `cpt_corpus/bulk_mix_code_local_part_0[0-6].jsonl` | 7×~1.9 GB | Intermediate shards |
| Global-redistribution failed mix (preserved for audit) | `cpt_corpus/bulk_mix_global_redistribution_2338301.jsonl` | 40 GB | DEPRECATED — bucket-drift bug; kept for audit only |

### Final Megatron binaries

| Item | Path | .bin size |
|---|---|---:|
| **Production base-tokenized NFC binary** | `cpt_corpus/bulk_mix_base_nfc_megatron/bulk_mix_text_document.{bin,idx}` | 37 GB (9.83 B tokens) |
| Bakeoff base-tokenized (pre-NFC) | `cpt_corpus/bulk_mix_base_megatron/bulk_mix_text_document.{bin,idx}` | 37 GB |
| Bakeoff extended-tokenized | `cpt_corpus/bulk_mix_ext_megatron/bulk_mix_text_document.{bin,idx}` | 27 GB |

**Canonical production prefix:** `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_base_nfc_megatron/bulk_mix_text_document`.

## 3. Init checkpoints

All three init arms (vanilla, retok, centroid) exist as HF + three Megatron variants. Total ~56 GB per arm.

| Arm | HF safetensors (E + U) | Megatron TP=1 | Megatron TP=2 | **Megatron TP=2 R17-patched** |
|---|---|---|---|---|
| vanilla | `init_checkpoints/modern_only_148480/vanilla/` (~16 GB safetensors) | `…/vanilla/megatron/` (16 GB) | `…/vanilla/megatron_tp2/` (16 GB) | **`…/vanilla/megatron_tp2_r17patched/release/`** (16 GB) |
| retok | `init_checkpoints/modern_only_148480/retok/` (~15.5 GB safetensors) | `…/retok/megatron/` (16 GB) | `…/retok/megatron_tp2/` (16 GB) | **`…/retok/megatron_tp2_r17patched/release/`** (16 GB) |
| centroid | `init_checkpoints/modern_only_148480/centroid/` (~15.5 GB safetensors) | `…/centroid/megatron/` (16 GB) | `…/centroid/megatron_tp2/` (16 GB) | **`…/centroid/megatron_tp2_r17patched/release/`** (16 GB) |

R17 patch verification JSONs at `/capstor/scratch/cscs/fffoivos/runs/r17_patch_roundtrip_{vanilla,retok,centroid}_2341{182,239,241}/verification.json` — all three report `standard_max_abs_diff=0.0`, `r17_max_abs_diff=0.0`, `logit_max_abs_diff=0.0`.

**Production init checkpoint:** `/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/vanilla/megatron_tp2_r17patched/release/`.

R1 round-trip artifacts from earlier validation: `/capstor/scratch/cscs/fffoivos/runs/r1_roundtrip_2333864/{apertus_megatron,apertus_hf_roundtrip}/`.

## 4. Token Distillation artifacts

Major dirs under `/iopsstor/scratch/cscs/fffoivos/token_distillation/`:

| Item | Path | Size | Notes |
|---|---|---:|---|
| **Coverage prepass (2B-scan, NFC)** | `coverage_2b_modern_20260523T032424Z_nfc/` | (sub-GB) | 99.82 % of 17,408 new tokens cleared `enough_100`; includes `td_coverage_summary.json`, `td_coverage_prepass.jsonl`, `td_snippet_index/snippets.jsonl` |
| Coverage smoke variants | `coverage_smoke_50k_*/`, `coverage_2b_modern_20260523T0[012]*/` | (sub-GB each) | Iterations during prepass debugging |
| **Full TD HF outputs (layer-pilot, both layers)** | `retok_td_full25_layers_20260523T092602Z/{last,layer11}/` | 2 × 16 GB | The TD-trained HF checkpoints. `layer11` is the chosen one. |
| TD layer-pilot manifest | `retok_td_full25_layers_20260523T092602Z/layer_pilot_manifest.json` | — | Records config + hyperparameters for the layer pilot |
| **TD R17 round-trip / patched Megatron** | `td_full25_layer11_r17_roundtrip_2357565/megatron_tp2_r17patched/release/` | 16 GB | TD layer-11 init in Megatron TP=2 format, R17-patched, used as the 2B challenger init |
| TD R17 round-trip HF (verification copy) | `td_full25_layer11_r17_roundtrip_2357565/hf_roundtrip/` | 16 GB | Patched Megatron → HF; verified bit-identical to layer11 source |
| TD R17 round-trip raw Megatron | `td_full25_layer11_r17_roundtrip_2357565/megatron_tp2_raw/` | 16 GB | Pre-patch baseline for the round-trip comparison |
| TD pilot intrinsic eval | `td_pilot_intrinsics_20260523T091637Z/`, `td_full25_intrinsics_20260523T124000Z/` | (sub-GB) | BPC / NLL / new-token diag comparisons |
| Smoke-run intermediates | `retok_td_smoke_last_layer_*/`, `retok_td_layer_pilot_*/`, `td_wrapper_dry_run_*/` | (sub-GB each) | Debugging iterations |

**TD challenger init checkpoint:** `/iopsstor/scratch/cscs/fffoivos/token_distillation/td_full25_layer11_r17_roundtrip_2357565/megatron_tp2_r17patched/release/`.

## 5. Bakeoff + TD training runs

All four 2B-token training runs completed iter 476/476 with 0 NaN / 0 skipped. Each per-iter checkpoint is ~136-138 GB; final iter 476 is the most useful for downstream conversion.

| Arm | Run dir | Total checkpoints | Final iter | Final size |
|---|---|---:|---:|---:|
| Vanilla | `/capstor/scratch/cscs/fffoivos/runs/bakeoff/bakeoff_1node_chain_20260522_005620_vanilla/` | 9 (iter 65/130/195/260/316/325/390/455/476) | 476 | 136 GB |
| ReTok | `…_retok/` | 9 | 476 | 138 GB |
| Centroid | `…_centroid/` | 9 | 476 | 138 GB |
| **TD layer-11** | `/capstor/scratch/cscs/fffoivos/runs/bakeoff/td_full25_layer11_2b_20260523T165038Z/` | 10 (iter 65/130/195/260/320/325/390/455/476) | 476 | 138 GB |

Each run dir contains `checkpoints/iter_*/`, `tensorboard/`, `triggers/`, `run_metadata.json`, `training_command.sh`. Final-iter checkpoints (476) are torch_dist format with two .distcp files (TP=2 → 2 ranks).

Training stdout/err logs at `/capstor/scratch/cscs/fffoivos/runs/bakeoff/bakeoff_{vanilla,retok,centroid}-2341{822,824,826}.{out,err}` plus the per-arm `resume*` jobs at 2341823/5/7 + 2345082/3/4.

**Disk pressure note:** these four run dirs account for ~5 TB. Keep iter 0000476 per arm; the intermediate iter checkpoints (~136 GB × 8 × 4 ≈ 4.4 TB) can be deleted once their per-iter evals are extracted, but they're useful for the trajectory analysis until that workflow is locked.

## 6. Eval outputs (480 GB total)

### Single-baseline evals

| Item | Path |
|---|---|
| V4-HF corrected baseline | `/capstor/scratch/cscs/fffoivos/runs/eval/apertus_baseline_v4_corrected_20260521_121639/` |
| V4-post-conversion (corrected, then split retry) | `…/apertus_postconv_v4_corrected_20260521_121639/`, `…_retry_20260521_122535/` |
| V4-post-conversion retention retry | `…/apertus_postconv_v4_retention_retry_20260521_163240/` |
| V4-post-conversion Greek retry | `…/apertus_postconv_v4_greek_retry_20260521_163240/` |

### Bakeoff per-iter packed evals

| Arm | Path |
|---|---|
| Vanilla | `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_vanilla/iter_0000{065,130,195,260,325,390,455,476}_{full,greek_only}/` |
| ReTok | same pattern under `…_retok/` |
| Centroid | same pattern under `…_centroid/` |
| TD | `/capstor/scratch/cscs/fffoivos/runs/eval/td_full25_layer11_2b_20260523T165038Z/iter_0000{065,130,195,260,390,455,476}_{full,greek_only}/` |

Each per-iter dir contains: `results_<ts>.json`, `samples_<task>_<ts>.jsonl` (per-sample logs for bootstrap CIs), `run_metadata.json`, and packed-eval stdout/err. Plus sibling files `iter_<N>_tokenizer_fair_metrics.json` and `iter_<N>_new_token_diagnostics.json` for the §5.1 + §5.3 intrinsic suites.

### Compact summaries (the digest files referenced by handoffs)

`/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_iter*_summary.md` and the local `init_bakeoff/eval/live_summaries/` mirror.

## 7. Code + Python envs

| Item | Path | Notes |
|---|---|---|
| Rsync'd subproject + glossapi_corpus_cli | `/iopsstor/scratch/cscs/fffoivos/repo/` | The Clariden mirror of our local `subprojects/03_apertus_extension_*/` + `glossapi_corpus_cli/` |
| Megatron-LM-Swiss-AI | `/iopsstor/scratch/cscs/fffoivos/code/training/Megatron-LM-Swiss-AI/` | Commit `c92402e3`; loader_apertus_hf.py symlinked in |
| pretrain-code (Apertus reference) | `/iopsstor/scratch/cscs/fffoivos/code/training/pretrain-code/` | Commit `531cc8be`; reference only |
| swissai lm-eval-harness clone | `/iopsstor/scratch/cscs/fffoivos/code/eval/lm-evaluation-harness-swissai/` | Source for our target install |
| EleutherAI lm-evaluation-harness | `/iopsstor/scratch/cscs/fffoivos/code/eval/lm-evaluation-harness-eleuther/` | Fallback |
| lm_eval target install (the eval runtime) | `/iopsstor/scratch/cscs/fffoivos/python_envs/lm_eval/` | 802 MB; tokenizer-fair metrics + diagnostics live here |
| TD coverage Python envs (py3.11) | `/iopsstor/scratch/cscs/fffoivos/python_envs/td_coverage_py311{,_xfer}/` | Set up via `setup_td_coverage_py311_xfer.sbatch` |

## 8. Things NOT on Clariden yet (intentionally)

These are explicitly absent and should NOT be created without a plan:

- **15-20 B production CPT run output** — not yet launched. Launcher at `init_bakeoff/production_cpt/submit_vanilla_base_15b_chain.sh` is dry-run-validated; live launch will produce a new training run dir under `/capstor/.../runs/bakeoff/`.
- **Anneal-phase Megatron binary** — `recipes/anneal.json` is design-only; the corpus build for anneal is fresh xfer CPU work.
- **Polytonic TD corpus** — would need a separate snippet pull (modern-only CPT corpus has near-zero polytonic content).
- **ILSP Greek YAMLs in swissai harness** — PF5 (the only remaining task in the local task list); blocks adding `hellaswag_greek` / `winogrande_greek` / `mmlu_pro_greek` / `truthfulqa_greek` / `medical_mcqa_greek` to the eval suite.

## 9. Verification one-liner

Quick sanity check that the production-critical paths exist:

```bash
ssh clariden 'for p in \
  /iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509/config.json \
  /iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_only_148480/tokenizer.json \
  /iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_base_nfc_megatron/bulk_mix_text_document.bin \
  /iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_base_nfc_megatron/bulk_mix_text_document.idx \
  /iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/vanilla/megatron_tp2_r17patched/release \
  /iopsstor/scratch/cscs/fffoivos/code/training/Megatron-LM-Swiss-AI \
  /iopsstor/scratch/cscs/fffoivos/python_envs/lm_eval ; do
  if [ -e "$p" ]; then echo "OK  $p"; else echo "MISSING  $p"; fi
done'
```

All seven should report `OK`. If any report `MISSING`, do not launch production.

## 10. Quick reference — most-used paths

| Need | Path |
|---|---|
| Production training data (.bin) | `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_base_nfc_megatron/bulk_mix_text_document` |
| Production init checkpoint | `/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/vanilla/megatron_tp2_r17patched/release` |
| TD challenger init checkpoint | `/iopsstor/scratch/cscs/fffoivos/token_distillation/td_full25_layer11_r17_roundtrip_2357565/megatron_tp2_r17patched/release` |
| TD HF (un-converted) | `/iopsstor/scratch/cscs/fffoivos/token_distillation/retok_td_full25_layers_20260523T092602Z/layer11/` |
| Selected NFC pool (parquet) | `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/cpt/selected_after_apertus_and_internal_dedup.parquet` |
| Heldout Greek eval JSONL | `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/heldout/cpt_greek_heldout_500_20260522.jsonl` |
| Base tokenizer | `/iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509/` |
| Extended tokenizer | `/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_only_148480/` |
| Apertus base HF model (= teacher) | `/iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509/` |
| Megatron-LM commit | `/iopsstor/scratch/cscs/fffoivos/code/training/Megatron-LM-Swiss-AI/` (c92402e3) |
| lm-eval runtime | `PYTHONPATH=/iopsstor/scratch/cscs/fffoivos/python_envs/lm_eval` |
