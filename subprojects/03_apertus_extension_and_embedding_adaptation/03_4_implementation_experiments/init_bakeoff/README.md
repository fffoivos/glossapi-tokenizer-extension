# Init Bakeoff

Three closed-form init experiments per [`../../cpt_plan.md`](../../cpt_plan.md) v0.7 §5: **Vanilla / ReTok / Centroid**, 2 B tokens per arm. Modern-only (vocab 148,480) per the 2026-05-20 scope decision. Composite 153,600 path remains available behind a flag for the future polytonic specialization run.

See [`BAKEOFF_PLAN.md`](BAKEOFF_PLAN.md) for the setup plan.

## Layout

```
init_bakeoff/
├── BAKEOFF_PLAN.md            — overall plan: arms, fidelity constraints, sbatch sizing
├── README.md                  — this file
├── arms/                      — the three init methods + production driver + smoke test
│   ├── _common.py
│   ├── vanilla.py
│   ├── retok.py
│   ├── centroid.py
│   ├── build_init_checkpoints.py   (Clariden driver: load Apertus, resize, init, save)
│   ├── test_init_logic.py          (home-side smoke; runs in ~15 s without HF model load)
│   └── README.md
├── corpus_build/                      — corpus assembly
│   ├── MIX_RECIPE.md          — bucket allocations + per-source weights, both phases
│   ├── recipes/
│   │   ├── bulk.json          — 70 / 26 / 4 Greek / replay / code; 31 sources
│   │   └── anneal.json        — 85 / 12 / 3; final 10–20 % of production CPT (not bakeoff)
│   ├── mix_builder.py         — streaming interleaver → JSONL output
│   ├── pull_greek_corpus.sh   — pull our Greek nanochat + Apertus-overlap-drop overlay
│   └── pull_replay_datasets.sh — pull FW2 / FW2-HQ / FineWeb-Edu / StarCoder
└── eval/                      — V4 baseline + per-arm bakeoff eval
    ├── EVAL_RECIPE.md         — task lists, cadence, statistical methodology
    ├── pull_benchmarks.sh     — pull retention + ILSP Greek + safety benchmarks; clone harness
    ├── run_eval.sbatch        — parameterized sbatch (MODEL_PATH + OUTPUT_DIR + TASK_GROUP)
    ├── run_apertus_baseline.sh — thin wrapper: V4 baseline on unmodified Apertus-8B-2509
    ├── run_bakeoff_arm_eval.sh — thin wrapper: per-arm checkpoint eval
    └── compute_bootstrap_cis.py — post-process: bootstrap CIs over --log_samples
```

## End-to-end sequence

The bakeoff fires once these are complete (most are Clariden-side):

```
[home]     Done:
           ✓ ship/apertus_greek_modern_only_148480/   (verified loadable)
           ✓ arms/test_init_logic.py                  (smoke green)
           ✓ verify_and_normalize_nfc.py              (V9 enforcer)

[Clariden login]   bash corpus_build/pull_greek_corpus.sh    # ~30-60 min (now pulls wave2 dedup metadata too)
                   bash corpus_build/pull_replay_datasets.sh # ~1-3 h depending on bandwidth
                                                              # (now includes FineMath stage-1 alongside replay/code)
                   bash eval/pull_benchmarks.sh              # ~30-60 min

[Clariden xfer]    bash corpus_build/normalize_nfc.sh        # V9 enforcement (idempotent NFC pass)
                   bash corpus_build/prepare_greek_pool.sh   # Runbook step: Apertus-drop + drop_intra_and_inter
                                                              # produces $SELECTED parquet for mix_builder
                   export SELECTED=/iopsstor/.../cpt/selected_after_apertus_and_internal_dedup.parquet
                   python3 corpus_build/mix_builder.py \
                       --recipe corpus_build/recipes/bulk.json \
                       --target-tokens 7000000000 \
                       --tokenizer /iopsstor/.../tokenizers/apertus_greek_modern_only_148480 \
                       --output /iopsstor/.../cpt_corpus/bulk_mix.jsonl \
                       --seed 20260520                    # ~6-10 h
                   # Optionally also build anneal_mix.jsonl with recipes/anneal.json (not used in bakeoff)

                   # Then tokenize JSONL → Megatron binary indexed dataset
                   # via swiss-ai/pretrain-code (Megatron-LM's tools/preprocess_data.py)
                   # (see ../cpt_plan_v0.7_status.md V12 / V13 for the Megatron config flags)

[Clariden debug]   python3 arms/build_init_checkpoints.py \
                       --apertus-base /iopsstor/.../models/apertus-8b-2509 \
                       --extended-tokenizer /iopsstor/.../tokenizers/apertus_greek_modern_only_148480 \
                       --out-root /iopsstor/.../init_checkpoints \
                       --vocab-size 148480 \
                       --arms vanilla retok centroid       # ~30 min (covers V2 + V14 + V15 + V16)

[Clariden normal] # V4 baseline (gates §5.6 thresholds)
                   bash eval/run_apertus_baseline.sh      # ~3-4 h
                   python3 eval/compute_bootstrap_cis.py /capstor/.../runs/eval/apertus_baseline_v4_*/

                  # Bakeoff: three arms in parallel
                  for arm in vanilla retok centroid; do
                      sbatch --job-name=bakeoff_$arm \
                          --export=ARM=$arm,... \
                          ../bakeoff_training/$arm.sbatch           # 12 h × 1 node, ~2 B tokens each
                  done

                  # Per-checkpoint eval during training (every 500 M tokens)
                  bash eval/run_bakeoff_arm_eval.sh /capstor/.../runs/<arm>/checkpoint-<step>/
                  python3 eval/compute_bootstrap_cis.py /capstor/.../runs/eval/bakeoff_<arm>_*/

                  # Selection: windowed average across last 3-5 checkpoints in 80-100% range
                  # per v0.7 §5.6 hard gates + selection score
```

The `bakeoff_training/` directory (Megatron-LM-Swiss-AI sbatch templates per arm) is the missing piece between init-checkpoint build and bakeoff submission. It's gated on Q D1 (Apertus's Megatron-LM-Swiss-AI fork branch/commit) being resolved.

## Reference

- [`../../cpt_plan.md`](../../cpt_plan.md) v0.7 §5 (init spec), §6 (eval), §10 (open decisions)
- [`../../apertus_fidelity_checklist.md`](../../apertus_fidelity_checklist.md) (architectural constraints)
- [`../../cpt_plan_v0.7_status.md`](../../cpt_plan_v0.7_status.md) (V1–V16 verification status)
- [`../STORAGE_AND_EXISTING_WORK.md`](../STORAGE_AND_EXISTING_WORK.md) (storage paths + measured throughput)
- [`../AUTH_AND_NODE_FINDING.md`](../AUTH_AND_NODE_FINDING.md) (sbatch sizing)
