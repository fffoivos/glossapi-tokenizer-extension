# 02_1_1 Tokenizer training

Sub-subproject of `02_1_tokenizer_experiments`. **Stage 1** of the
tokenizer-experiments pipeline:

```
[02_1_1 tokenizer training]
        → produces a full BPE tokenizer.json at the maximum target vocab
[02_1_2 cutoff variant builder]
        → derives Apertus-compatible merged variants at sub-vocabs
[02_1_3 fertility evaluation]
        → measures intrinsic + fertility metrics on each variant
[02_1_4 cutoff analysis]
        → combines fertility + glossary + lang-mask -> recommends a cutoff
```

## Goal

Train continuous-BPE tokenizer arms from Apertus base on a chosen
training mix, while preserving Apertus front-end behavior exactly.

The same scripts also produced the historical fresh-discovery arms
(F1, F2 — archived) and can be reused for future tokenizer arms once
the source mix, dedup plan, and extension-vs-fresh-tokenization path are
explicitly chosen.

## Inputs

- Apertus base snapshot: `~/data/glossapi_work/tokenizer_base_snapshots/apertus_8b_2509_<date>/`
- Training mix: `~/runs/<arm>/.../mix.parquet` (produced upstream by
  `subprojects/_archive/01_2_training_dataset_mix/`)
- Optional source-audit input for future arms. The active
  Ancient/Polytonic Greek source-selection work lives in the sibling
  project `../02_1_polytonic_greek_extension/`.

## Outputs

- `tokenizer.json` at the target vocab size
- `training_summary.json` (input rows, runtime, phase timings)
- `replication_check.json` + `front_end_contract_check.json` —
  Apertus front-end identity checks (special tokens, regex split,
  ByteLevel, first 1000 ids)
- Optionally: HF dataset entry via `publish_tokenizer_extension_repo.py`

## Scripts

- `scripts/train_continuous_bpe_tokenizer.py` — core trainer for the
  C-arms (continuous from Apertus). Runs in 4 phases:
  identity_check → count_segments → build_sequence_shards →
  aggregate_sequences → merge_loop → write_tokenizer.
- `scripts/train_discovery_tokenizer.py` — fresh-discovery trainer
  (produced F1/F2; archived but kept for reproducibility).
- `scripts/train_bpe_from_text_shards.py` — low-level BPE trainer used
  by both arms.
- `scripts/wait_for_tokenizer_mixes_and_launch_training.sh` — watcher
  that launches training when upstream `mix.parquet` lands.
- `scripts/watch_continuous_runs_and_publish.py` — monitors a
  continuous training run and publishes when complete.
- `scripts/inspect_bpe_vocab_denoising.py` — post-training inspection
  of the trained tokenizer's added units.
- `scripts/publish_tokenizer_extension_repo.py` — uploads the trained
  tokenizer artifacts to HF.

## Plans + history

- `CONTINUOUS_BPE_EXTENSION_PLAN.md` — original plan for the
  continuous-BPE arm. Sections §1 (cutoff grid), §6 (mergeback), §7
  (evaluation), §10 (acceptance) remain applicable; the four-arm
  §2–§5 / §8 are historical (archived).
- `CONTINUOUS_BPE_EXTENSION_TODO.md` — original work list.

The earlier production-strict-v2 run lives under
`../runs/_archive/production_strict_v2/` (F1 + F2 + C1; wave-3 strict).
