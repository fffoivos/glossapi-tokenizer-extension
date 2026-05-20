# `arms/` — init-method implementations

Modules:

- `_common.py` — shared helpers (norm-match, Greek-block classification, centroid + std computation).
- `vanilla.py` — Vanilla arm: no init. Symlinks the base Apertus checkpoint.
- `retok.py` — ReTok init: per-new-token subpiece-mean + Phase A norm-match.
- `centroid.py` — Centroid init: per-script centroid of base Greek tokens + Gaussian noise + Phase A norm-match.
- `build_init_checkpoints.py` — production driver (Clariden-side; loads the full Apertus model and writes resized checkpoints for each arm).
- `test_init_logic.py` — home-side smoke test that validates `retok` and `centroid` algorithms against the local E/U matrices without needing a full model load.

## Local smoke test (no Clariden, no GPU)

```bash
cd init_bakeoff/arms
/home/foivos/.venvs/glossapi-merge-docling/bin/python3 test_init_logic.py
```

Validates:
- Greek-block classification (modern + polytonic + both sets) over the base 131,072 vocab
- ReTok produces norm-matched [200, 4096] new rows on the first 200 new IDs
- Centroid produces norm-matched [200, 4096] new rows with the polytonic-fallback path (since Apertus base has 0 polytonic tokens)
- Both methods agree on shape but produce near-orthogonal directions for the same new token (mean cos ≈ 0.03 — confirming they test different hypotheses)

Expected runtime: ~10–15 s after the E + U matrices are read into RAM (~9 s for 4.3 GB total).

## Clariden production build (3 arms × ~30 min total)

```bash
ssh clariden
salloc -A a0140 -p debug -N 1 -t 00:30:00 --gres=gpu:1
uenv start pytorch/v2.6.0:v1 --view=default

cd /capstor/scratch/cscs/fffoivos/code/init_bakeoff/arms
python3 build_init_checkpoints.py \
    --apertus-base /iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509 \
    --extended-tokenizer /iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_extended_153600 \
    --out-root /iopsstor/scratch/cscs/fffoivos/init_checkpoints \
    --arms vanilla retok centroid
```

Output: three subdirectories under `/iopsstor/.../init_checkpoints/`:
- `vanilla/` — symlinked from the base (no new safetensors)
- `retok/` — ~16 GB, vocab 153,600, ReTok-initialized new rows
- `centroid/` — ~16 GB, vocab 153,600, Centroid-initialized new rows

Plus an `init_build_summary.json` with per-arm stats and the sanity-check forward-pass results (V2: shape correct, no nan/inf).

## Conversion to Megatron format

The three HF checkpoints are then converted to Megatron-LM-Swiss-AI
format using [`swiss-ai/hfconverter`](https://github.com/swiss-ai/hfconverter)
at staging time. The Megatron checkpoints are what the training jobs
actually load.

## Phase A targets (used by both extension arms)

- E target norm: **5.05** (Greek-content tokens, from
  `runs/apertus_greek_diagnostic_20260511_v2/`)
- U target norm: **3.80**

These match the existing Greek-token distribution in Apertus base to
within 1 % (the smoke test confirms `E[modern].norm.p50 = 5.047`,
`U[modern].norm.p50 = 3.797`).
