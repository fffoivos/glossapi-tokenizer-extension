# `arms/` — init-method implementations

Modules:

- `_common.py` — shared helpers (norm-match, Greek-block classification, centroid + std computation).
- `vanilla.py` — Vanilla arm: no init. Symlinks the base Apertus checkpoint.
- `retok.py` — ReTok init: per-new-token subpiece-mean + Phase A norm-match.
- `centroid.py` — Centroid init: per-script centroid of base Greek tokens + Gaussian noise + Phase A norm-match.
- `build_init_checkpoints.py` — production driver (Clariden-side; loads the full Apertus model and writes resized checkpoints for each arm).
- `build_init_checkpoints.sbatch` — queueable Clariden job for the HF-format build.
- `convert_init_checkpoints.sbatch` — queueable Clariden job for HF -> Megatron `torch_dist` conversion.
- `submit_init_pipeline.sh` — submits build, then conversion with `afterok`.
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

## Clariden production build

```bash
ssh clariden

cd /iopsstor/scratch/cscs/fffoivos/repo/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/arms
INIT_CKPT_ROOT=/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480 \
VOCAB_SIZE=148480 \
bash submit_init_pipeline.sh
```

The init build/conversion jobs default to `INIT_UENV_IMAGE=pytorch/v2.9.1:v2`
because that uenv has a Transformers release new enough to recognize
`model_type=apertus`. The 2B training jobs still use the training recipe's
`pytorch/v2.6.0:v1`.

Output: three subdirectories under `/iopsstor/.../init_checkpoints/modern_only_148480/`:
- `vanilla/` — symlinked from the base (no new safetensors)
- `retok/` — ~16 GB, vocab 148,480, ReTok-initialized new rows
- `centroid/` — ~16 GB, vocab 148,480, Centroid-initialized new rows

Plus an `init_build_summary.json` with per-arm stats and the sanity-check forward-pass results (V2: shape correct, no nan/inf).

## Conversion to Megatron format

`submit_init_pipeline.sh` queues `convert_init_checkpoints.sbatch` after the
HF build. It uses Megatron-LM-Swiss-AI's checkpoint tool with our Apertus loader:

```bash
python3 tools/checkpoint/convert.py --model-type GPT \
    --loader apertus_hf --saver core \
    --load-dir <arm-hf-dir> --save-dir <arm-hf-dir>/megatron \
    --tokenizer-model <arm-hf-dir> --bf16 \
    --loader-transformer-impl transformer_engine
```

The job marks each converted checkpoint as a `release` checkpoint because
Megatron's `loader_core` rejects an iteration-0 `iter_0000000` checkpoint on the
roundtrip/training load path. The training jobs load:

```text
$INIT_CKPT_ROOT/{vanilla,retok,centroid}/megatron
```

## Phase A targets (used by both extension arms)

- E target norm: **5.05** (Greek-content tokens, from
  `runs/apertus_greek_diagnostic_20260511_v2/`)
- U target norm: **3.80**

These match the existing Greek-token distribution in Apertus base to
within 1 % (the smoke test confirms `E[modern].norm.p50 = 5.047`,
`U[modern].norm.p50 = 3.797`).
