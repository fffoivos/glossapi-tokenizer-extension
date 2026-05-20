# 03.4 Implementation Experiments

## Scope

The hands-on sub-subproject. Where 03.3 plans, 03.4 runs.

This is where actual SLURM jobs get authored, submitted, monitored,
and analyzed on CSCS Clariden. The earlier sub-subprojects (03.1
diagnostic / 03.2 dedup audit / 03.3 kickoff planning) feed inputs
into this one.

## Files

- [`AUTH_AND_NODE_FINDING.md`](AUTH_AND_NODE_FINDING.md) — verified
  auth state, partition / QoS / queue probe results, expected start
  times at various job sizes, recommended-shape decision for the
  first calibration run.
- [`STORAGE_AND_EXISTING_WORK.md`](STORAGE_AND_EXISTING_WORK.md) —
  live storage map (iopsstor / capstor / users), CPU-cluster
  options (xfer partition + the "no Eiger access" finding), and the
  big finding: **p-skarvelis (in our a0140 project) has been running
  Apertus-Greek CPT + SFT since 2026-04-17** using HF Trainer +
  FineWeb-2-HQ. Their runs are tokenizer-incompatible with our C3-17,408
  ship but their setup is the most-likely scaffold we should adopt.
- [`ENVIRONMENT_AND_BENCHMARKS.md`](ENVIRONMENT_AND_BENCHMARKS.md) —
  the full inventory of (a) swiss-ai training/eval/init code repos
  (`apertus-finetuning-recipes`, `pretrain-code`, `lm-evaluation-harness`,
  `model-launch`, `perf-check`, `evals`, `token-distillation`, …),
  (b) Apertus's reported eval set (ARC / HellaSwag / WinoGrande /
  XNLI / XCOPA / PIQA), (c) the ILSP Greek Evaluation Suite (21+
  datasets on `ilsp/*` HF, all open), (d) safety benchmarks (ILSP +
  swiss-ai variants), and (e) the concrete CSCS deployment plan with
  paths, `huggingface-cli` commands, and login-node-vs-slurm
  scheduling.
- [`init_bakeoff/`](init_bakeoff/) — **active**: the three-arm init
  experiment per `../cpt_plan.md` v0.7 §5. Vanilla / ReTok / Centroid,
  2 B tokens per arm. **Modern-only (vocab 148,480)** per the
  2026-05-20 scope decision; composite 153,600 path remains in
  `build_init_checkpoints.py` behind `--vocab-size 153600` for the
  future polytonic specialization run. Three subdirectories:
  - [`arms/`](init_bakeoff/arms/) — the three init Python modules
    (`vanilla.py`, `retok.py`, `centroid.py`) + production driver
    (`build_init_checkpoints.py`) + local smoke test
    (`test_init_logic.py`). Smoke ran green: both extension arms
    produce norm-matched [17408, 4096] new rows.
  - [`data/`](init_bakeoff/corpus_build/) — corpus assembly:
    [`MIX_RECIPE.md`](init_bakeoff/corpus_build/MIX_RECIPE.md) (bucket
    allocations), `recipes/{bulk,anneal}.json` (31 sources for bulk,
    14 for anneal; weights sum to 1.0 verified),
    `mix_builder.py` (streaming interleaver → JSONL),
    `pull_greek_corpus.sh` + `pull_replay_datasets.sh` (login-node HF
    downloads). Bulk recipe = 70 % Greek / 26 % replay / 4 % code;
    anneal recipe = 85 / 12 / 3 (not used in bakeoff).
  - [`eval/`](init_bakeoff/eval/) — V4 baseline + per-arm eval:
    [`EVAL_RECIPE.md`](init_bakeoff/eval/EVAL_RECIPE.md) (task
    lists), `pull_benchmarks.sh` (login-node), `run_eval.sbatch`
    (parameterized: MODEL_PATH + OUTPUT_DIR + TASK_GROUP),
    `run_apertus_baseline.sh` (V4 wrapper), `run_bakeoff_arm_eval.sh`
    (per-arm wrapper), `compute_bootstrap_cis.py` (bootstrap CIs
    over `--log_samples` per v0.7 §6.1 methodology).
- (planned) `01_vanilla_calibration_v1/` — first concrete job: 1-node
  4×GH200 throughput calibration on Apertus-8B-2509 + the modern-only
  148,480 tokenizer (Vanilla arm; smallest setup-risk).
- (planned) `02_pilot_runs/` — three parallel arm pilots once
  calibration reports actual tokens/sec.
- (planned) `sbatch_templates/` — reusable sbatch wrappers for each
  partition + size + arm.
- (planned, accumulating) `job_log.jsonl` — append-only log of every
  job I submit, per the [CSCS workflow doc](../03_3_cscs_experiments_kickoff/CSCS_AUTH_WORKFLOW.md#cluster-job-log).

## Reads-from

- Authoritative tokenizer ship bundle:
  [`../03_3_cscs_experiments_kickoff/ship/apertus_greek_extended_153600/`](../03_3_cscs_experiments_kickoff/ship/apertus_greek_extended_153600/)
  (composite; the 148,480 modern-only base is at
  [`../../02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep/variants/c3_added_17408_curated_padded/`](../../02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep/variants/c3_added_17408_curated_padded/)).
- CPT corpus build recipe:
  [`../03_2_apertus_c3_dedup_audit/CPT_DATASET_BUILD_RUNBOOK.md`](../03_2_apertus_c3_dedup_audit/CPT_DATASET_BUILD_RUNBOOK.md).
- Curriculum + init-corpus decision:
  [`../03_3_cscs_experiments_kickoff/CURRICULUM_AND_INIT_CORPUS.md`](../03_3_cscs_experiments_kickoff/CURRICULUM_AND_INIT_CORPUS.md).
- Three-arm experimental design:
  [`../experiments_plan.md`](../experiments_plan.md) §5.

## Hard preconditions (verified 2026-05-20)

- CSCS cert at `~/.ssh/cscs-key-cert.pub` valid (`cscs-key list`); refresh via `cscs-key sign --headless --duration 1d`.
- Project account: `a0140` (live, confirmed via `sacctmgr show user fffoivos -s`).
- Login node tested: `ssh ela 'hostname'` → `ela5`. `ssh clariden 'hostname'` → `clariden-ln001`.
- pytorch `uenv` already on Clariden scratch: `pytorch/v2.6.0:v1` (8.2 GB, gh200 arch, pulled 2025-04-04).
