# LAUNCH RUNBOOK — two-arm 13.5B Greek CPT

> **Superseded as entry point by [`HANDOFF.md`](HANDOFF.md)** (2026-06-09): new
> Greek = 70% HPLT + 30% openarchives; 3 held-out val sets with per-set loss
> (patch in `dataset_build/EXTRA_VALID_README.md`, **required before launch**).
> This file remains the launch/observability detail.

The go-sequence from a clean Clariden to two running, auto-benchmarked arms.
Details for each build step are in [`BUILD_PLAN.md`](BUILD_PLAN.md); this is the
ordering + the launch/observability layer.

> **Current 2026-06-10 status:** dataset artifacts, init checkpoints, extra
> validation patch, and `gate_cpt2arm_artifacts.sh` are complete. Earlier
> multi-node AWS Libfabric/CXI Megatron smokes failed before iteration 1 with
> `NET/OFI ... NO_SPACE`; Socket/HSN proved functional but too slow at 16 nodes.
> The updated deep dive identifies trainer-forced `NCCL_NET_FORCE_FLUSH=1` as
> the root-cause candidate. The trainer now defaults it to `0`; 2-node and
> 4-node CXI no-flush smokes passed, and 16-node CXI no-flush smoke `2515665`
> is the launch-scale gate.
> Runtime report:
> [`../reports/CLARIDEN_MEGATRON_NCCL_NO_SPACE_20260610.md`](../reports/CLARIDEN_MEGATRON_NCCL_NO_SPACE_20260610.md).

## 0 · Status of inputs (all decisions made; remaining work is execution)

| # | item | status |
|---|---|---|
| Mixture | 10B new + 35%-of-new = 13.5B | ✅ resolved |
| Warmup | `2/(1−β2)` iters | ✅ resolved |
| Vocab | 148,480 modern-only (256-aligned) | ✅ resolved |
| Corpus-prep order / NFC / PII / HPLT scope | per `../02_corpus_preparation/PIPELINE.md` (clean→dedup→decontaminate→anonymize-last; no global NFC; HPLT = confident-only) | ✅ decided 2026-06-09 |
| Corpus-prep **execution** | full-corpus runs of clean(overlay)/decontaminate/anonymize on the existing 129 GB `SELECTED` | ✅ complete |
| 5% Greek-replay | from `apertus_overlap_drop_docs.parquet` | ✅ built |
| Stage-C order | replay/non-new slots preserved; new-Greek slots HPLT then openarchives | ✅ complete |
| Tokenization | base + extended Megatron binaries from the same ordered JSONL | ✅ complete |

## 1 · Sync to Clariden

Mirror the repo (incl. this `03_training_experiments/`) to
`$REPO_ROOT=/iopsstor/scratch/cscs/fffoivos/repo/glossapi-tokenizer-extension`
and confirm the Megatron fork + tokenizers + `apertus-8b-2509` model are staged.
Needs a fresh CSCS key (`cscs-key --headless sign`).

## 2 · Build the dataset (CPU-only on `xfer`)

Corpus content prep is the canonical pipeline — run
[`../02_corpus_preparation/PIPELINE.md`](../02_corpus_preparation/PIPELINE.md):
clean (confident-only overlay) → dedup (validate existing `SELECTED`) →
decontaminate (`correct_only`) → anonymize (mask email/IP/IBAN last). Then this
subproject's last two steps (BUILD_PLAN §2): mix to 10B-new + 35% replay (+ the
`apertus_overlap_drop` Greek-replay bucket), anonymize, Stage-C reorder only the
HPLT/openarchives slots while preserving replay/non-new-Greek line positions,
then tokenize the one shared mix **twice** →
`bulk_mix_ordered_replay_fixed_base_text_document` (131072) +
`bulk_mix_ordered_replay_fixed_ext_text_document` (148480). Both binaries from
the same ordered JSONL (byte-identical doc stream).

## 3 · Build init checkpoints (per BUILD_PLAN §0–§1)

- **arm1**: HF `main` → revert geometry (rope 12M/65536 → 500k/4096, keep `rope_scaling`) → convert TP=2 → R17-patch → `…/vanilla_base131072/megatron_tp2_r17patched`.
- **arm2**: Token-Distillation layer-11 (E by MSE-distill + U by CE) → R17 roundtrip → `…/modern_greek_td148480/megatron_tp2_r17patched`. (Reuse `td_full25_layer11_r17_roundtrip` iff its tokenizer == `apertus_greek_modern_only_148480`.)
- Gate every conversion: `verify_hf_roundtrip.py --require-r17-match --logits`.

Point the two `configs/arm*.env` `INIT_CKPT` + `*_DATA_PREFIX` at these.

## 4 · Launch BOTH arms in parallel + auto-benchmark

```bash
bash scripts/launch_all.sh                          # dry-run: prints both chains + watcher cmds
bash scripts/gate_cpt2arm_artifacts.sh              # checks force-flush disabled + artifacts
DRY_RUN=0 CONFIRM_LAUNCH=1 bash scripts/launch_all.sh
```
This submits, per arm: the full-run WSD training chain (`submit_two_arm_full_run.sh`,
walltime-bounded with `--exit-interval`; Socket/16-node launches default to 4
longer segments to avoid queue churn) **and** an eval-sidecar watcher fed the
benchmark cadence (every ~1B tokens + final = 14 checkpoints).
The two arms are independent Slurm chains → concurrent. `launch_all.sh` also
runs the artifact gate automatically for live launches by default; set
`RUN_ARTIFACT_GATE=0` only for a deliberately manual override.

Full-run default scale is **16 nodes per arm** (`64` GPUs/arm, `128` training
GPUs total for the two parallel arms). This preserves TP=2/checkpoint geometry
and scales data parallelism only. A 1-node launch is a diagnostic shape, not an
acceptable full-run shape.

At multi-node scale the trainer must use `LAUNCH_MODE=torchrun`: one Slurm task
per node launches `torchrun --nproc_per_node=4`. Direct multi-task Slurm launch
has reproduced an inter-node NCCL/OFI `NO_SPACE` failure before iteration 1.
The trainer keeps the CSCS Alps/uenv AWS Libfabric defaults and now disables
`NCCL_NET_FORCE_FLUSH` by default. Do not use the slower Socket fallback unless
the CXI no-flush validation fails. `scripts/gate_cpt2arm_artifacts.sh` checks
both the data/config artifacts and the force-flush-disabled runtime plumbing.

Pure PyTorch NCCL controls pass, including Megatron-shaped 40M and exact-size
67M bfloat16 all-reduce/reduce-scatter/all-gather. The latest evidence is that
the trainer-only `NCCL_NET_FORCE_FLUSH=1` caused the tiny `size:4` receive
failure in the AWS OFI NCCL SENDRECV path; with force-flush disabled, 2-node
and 4-node CXI Megatron smokes reach iteration 1.

If a run was accidentally launched at diagnostic scale, cancel the obsolete
continuation jobs and resume the existing run directory from its latest
checkpoint with:

```bash
RUN_TAG=cpt13b_vanilla_<STAMP> DRY_RUN=0 CONFIRM_LAUNCH=1 NODES=16 \
  bash scripts/submit_scaled_resume_chain.sh vanilla
RUN_TAG=cpt13b_td_<STAMP> DRY_RUN=0 CONFIRM_LAUNCH=1 NODES=16 \
  bash scripts/submit_scaled_resume_chain.sh td
```

## 5 · Observability & benchmarks (mostly automatic)

- **Loss / grad-norm / LR / throughput / params-norm / memory** — logged **every
  iteration** to `$OUTPUT_DIR/tensorboard` (`--log-interval 1 --log-throughput
  --log-params-norm --log-memory-to-tensorboard`) and to the `.out` files.
  `run_metadata.json` records the full as-run config (incl. the WSD cooldown +
  rope geometry) per segment.
- **Quick CLI + cross-arm comparison:**
  ```bash
  python3 scripts/collect_metrics.py \
    --arm vanilla:'$RUN_ROOT/cpt13b_vanilla_*/*.out' \
    --arm td:'$RUN_ROOT/cpt13b_td_*/*.out' --out metrics.csv
  ```
  (per-iter loss/lr/grad/TFLOPs + min/last-loss + NaN/skip alarms).
- **Benchmarks per checkpoint** (auto, via the watchers → `submit_*_checkpoint_sidecars.sh`):
  native Greek MCQ (`greekmmlu`=dascim, ilsp_medical, ilsp_asep, plutus),
  Greek-NLP suite, heldout Greek/code/math BPB (tokenizer-fair), multilingual
  retention, + a checksum manifest. Results land under each arm's `RUN_ROOT`.
- **TensorBoard:** `tensorboard --logdir $RUN_ROOT/cpt13b_vanilla_*/tensorboard` (and `_td_`).

## Watcher partition note

The sidecar watchers are 24h CPU pollers. They default to `xfer` (cheap and
CPU-only). Avoid `WATCHER_PARTITION=normal` unless explicitly forced: it bills a
full GH200 node for a poller. The training itself is unaffected (it's on
`normal`/GH200 by design).
