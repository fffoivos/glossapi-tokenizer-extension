# 02_1_2 Cutoff variant builder

Sub-subproject of `02_1_tokenizer_experiments`. **Stage 2**:

```
[02_1_1 tokenizer training] → full tokenizer.json at max vocab
       │
       ▼
[02_1_2 cutoff variant builder] → N tokenizers at total = base + cutoff
       │
       ▼
[02_1_3 fertility evaluation] → metrics per variant
```

## Goal

Derive Apertus-compatible **merged tokenizer variants** from a fully-
trained continuous-BPE arm by truncating its added vocab + merges to a
chosen cutoff `N` and writing out a loadable HF tokenizer dir.

Because the continuous-BPE arm preserves Apertus base ids
`0..131,071` unchanged and appends new ids `131,072..131,072+max_N-1`
in merge order, any prefix `131,072..131,072+N` is a valid Apertus-
compatible tokenizer.

## Inputs

- Apertus base tokenizer dir
- Full continuous-BPE arm output dir (`02_1_1` output)
- List of cutoffs `N` (must be 128-aligned; preferably 256-aligned to
  match the polytonic-split-plan grid)

## Outputs

Per cutoff `N`: a directory `<arm_prefix>_added_<N>/` containing
`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`.
Each variant has total vocab `131,072 + N`. Loadable with
`transformers.AutoTokenizer.from_pretrained`.

## Scripts

- `scripts/build_cutoff_variants.py` — main builder. Wraps
  `build_continuous_cutoff(...)` from
  `tokenizer_analysis/run_wave4_fertility_eval.py` and renames the
  hardcoded `c1_added_*` output prefix to whatever the caller passes
  via `--arm-prefix`.

## Example invocation (C3 sweep, 25 cutoffs at 1024 step)

```bash
python3 scripts/build_cutoff_variants.py \
  --arm-name C3_wave2_broad_glossapi_plus_hplt_50_50 \
  --arm-prefix c3 \
  --base-dir /home/foivos/data/glossapi_work/tokenizer_base_snapshots/apertus_8b_2509_20260415 \
  --full-dir /home/foivos/runs/c3_wave2_broad_latest_cleaner_20260506/50_50/tokenizers/C3_wave2_broad_glossapi_plus_hplt_50_50/tokenizer \
  --out-dir  /home/foivos/runs/c3_cutoff_eval_20260511/cutoff_tokenizers \
  --cutoffs $(seq 1024 1024 25600)
```

Each variant build is essentially a JSON slice + file copy; runtime is
seconds per cutoff.

## Contract checks each variant inherits from the full arm

- first 1,000 ids exactly preserved
- special tokens (`<s>`, `</s>`, `<pad>`, `<unk>`) exactly preserved
- regex split + ByteLevel pretokenizer untouched
- normalizer = null (Apertus front-end)

These are enforced by reusing the full arm's `tokenizer_config.json`
and `special_tokens_map.json` verbatim. The contract is verified on
the FULL arm during training (`02_1_1`); each cutoff variant inherits
because the front-end JSON files are byte-identical copies.

## What's archived

- The wave-3 strict run's C1 cutoff variants live under
  `~/runs/wave4_20260429/production_strict_v1/evaluation/.../cutoff_tokenizers/`
  on the gcloud instance. Archived as historical evidence of how the
  builder behaves on a different arm.
