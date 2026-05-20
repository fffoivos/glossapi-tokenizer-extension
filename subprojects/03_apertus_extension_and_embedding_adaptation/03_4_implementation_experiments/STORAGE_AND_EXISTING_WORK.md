# CSCS Storage Map + Existing Apertus-Greek Work in `a0140`

*Drafted 2026-05-20 from live probes. Cert window 2026-05-20T15:03:46
→ 2026-05-21T15:03:46.*

The headline finding is in §3: **p-skarvelis (in our a0140 project)
has been running Apertus-Greek CPT + SFT since at least 2026-04-17.**
Their setup is HF Trainer + `ApertusForCausalLM` on
FineWeb-2-HQ Greek + FineWeb-HQ English at 90/10, with a non-C3
tokenizer. We should coordinate before submitting our own runs.

## 1. Storage map

All live measurements from `df -h` and `ls -la` on `clariden-ln001`,
2026-05-20.

| Mount | Total | Used | Avail | Used by us (fffoivos) | Role |
|---|---:|---:|---:|---:|---|
| `/users/fffoivos` | 2.0 PB (cluster-wide) | 7 % | — | ~38 KB (shell rcs only) | tiny per-user home, configs only |
| `/iopsstor/scratch/cscs/fffoivos` | 3.0 PB | 89 % (!) | 349 TB | **8.0 GB** | random-read scratch — datasets, uenv image cache, container layers |
| `/capstor/scratch/cscs/fffoivos` | 150 TB | 1 % | 150 TB | **24 KB** | large-sequential scratch — checkpoints, train output |
| `/capstor/store/cscs/swissai/` | 91 PB | 63 % | 35 PB | **N/A — `a0140/` does not exist yet** | project store; ~178 project dirs visible (a0060…a0140 absent) |
| `/users/cscs` NFS | 2.0 PB | 7 % | 1.8 PB | (read-only mount from ela) | login-node shared home |

**Important corrections from the earlier README understanding:**

1. The project-store path is **`/capstor/store/cscs/swissai/<project>`** (with `cscs/` in the middle), **not** `/capstor/store/swissai/...`. Our `a0140/` subdirectory does not exist there yet — sibling projects (a0060, a0072, a0081, etc.) do.
2. `iopsstor` is **89 % full at the cluster level**. Plan storage carefully on iopsstor. Our 8 GB usage is a rounding error in that.
3. `capstor/scratch` has plenty of room (150 TB total, 1 % used). Heavy outputs belong there.
4. `ela` does **not** mount the Alps scratch/store filesystems — only `/users`. Always probe storage from `clariden` (or any compute node), never from `ela`.

### 1.1 Our usage right now

```
/iopsstor/scratch/cscs/fffoivos/ — 8.0 GB
  ├── clariden-toy-pytorch/          (the March 28 smoke test)
  ├── .parallax_imagestore/
  └── .uenv-images/                  (pytorch/v2.6.0:v1 cache, 8.2 GB)

/capstor/scratch/cscs/fffoivos/ — 24 KB
  └── openarchives_ocr_20260401/     (an early placeholder, empty)
```

### 1.2 `p-skarvelis` usage (in our project — coordinator scope)

```
/capstor/scratch/cscs/p-skarvelis/ — multi-tens-of-GB
  ├── apertus-greek-cpt-probe-curated-1GB-100steps/      (May 1, smoke probe)
  ├── apertus-greek-cpt-full-1btok-500steps/             (May 1, 1 B-token CPT)
  ├── apertus-greek-cpt-full-1btok-500steps-continue-lr1e5-200steps-v2/  (May 1, continuation)
  ├── apertus-greek-cpt-prod-xielu-sdpa-nogc-curated-1GB-2048seq-1000steps/  (Apr 17 prod, 1000 steps)
  ├── apertus-greek-cpt-prod-xielu-sdpa-nogc-curated-1GB-2048seq-400steps/   (Apr 21 prod, 400 steps)
  ├── apertus-greek-sft/                                  (Apr 24, full HF SFT — 20 GB safetensors)
  ├── build_fixes/                                        (empty dir as of May 17)
  └── .enroot/                                            (enroot container cache)

/iopsstor/scratch/cscs/p-skarvelis/ — 106 GB
  ├── apertus-greek-init/                                 (custom CPT init checkpoint — used as model_path)
  ├── prepared-datasets/
  │   ├── apertus-greek-targeted-packed-2048/             (targeted CPT data, 2048-packed)
  │   ├── apertus-greek-full-packed-2048-1btok/           (full 1B-token packed)
  │   └── apertus-greek-sft-1024-left-val2048/            (SFT pre-tokenized)
  ├── podman-*/                                           (multiple container snapshots, modes 700; not readable)
  ├── .enroot/                                            (container cache)
  └── ...
```

The mode flag on most p-skarvelis directories is `drwxr-x---+` — group-readable with POSIX ACLs that grant `a0140` members read. We can read configs, checkpoints, and trainer state; we can't write into their tree.

## 2. CPU options (the user's question)

| Option | Where | Practical size | Walltime cap | Status / caveats |
|---|---|---|---|---|
| Clariden `xfer` partition | within Clariden, partition `xfer` | 2 nodes (`nid001154`, `nid001306`); **1 node per job max**; 256 vCPU, 500 GB RAM per node | 24 h | Designed for data transfer. `AllowAccounts=ALL`. Currently shows `maint` state (reservation-bound — same as the rest of the cluster). Usable for any CPU-only work that fits a single 256-vCPU node. |
| Clariden GH200 with no GPU use | normal/debug/low partitions | 1 node = 288 vCPU + ~856 GB RAM (4× GH200 idle on the side) | 12 h normal / 1.5 h debug | Wastes the GPUs (1340 of them available), but you get 288 fast Neoverse-V2 cores. |
| **CSCS Eiger** (AMD-CPU cluster, ~1,000+ nodes) | `eiger.cscs.ch` | per-job in the 10s of nodes | typical 12-24 h | **DNS doesn't resolve from clariden, and `sacctmgr show association` shows our `a0140` association is `clariden`-only.** To use Eiger we'd need a separate allocation request to CSCS — not currently available. |
| **CSCS Daint / Todi** | same story | — | — | Same: no association exists. Would need a separate request. |
| **Santis / Bristen** | resolves (`santis.cscs.ch` → 172.28.14.19, `bristen.alps.cscs.ch` in ssh config) | also Alps GH200 | — | These are sibling **GPU** clusters, not CPU. Same hardware shape, not a CPU option. |

**Practical answer**: for CPU-only data prep / dedup / tokenization
work we have **two real options today** — Clariden's `xfer` partition
(2 nodes max, single-job, 24 h), or burning GH200 nodes as 288-vCPU
hosts on `normal`/`low`. We have no CSCS CPU cluster like Eiger
through this project; that would need a separate allocation.

> **Constraint update 2026-05-20: we no longer have GCloud access.**
> Anything in the parent `CPT_DATASET_BUILD_RUNBOOK.md` that says "GCP scratch
> VM" now has to land on Clariden instead. Concretely:
> - **CPT corpus build** (download nanochat → hard-exclude Apertus overlap → replay internal dedup → write final pool) → run on a single-node `xfer` allocation (256 vCPU, 500 GB RAM, 24 h walltime). The runbook's compute requirement (c4-highmem-32 + 1-2 TB SSD) maps cleanly to one `xfer` node + iopsstor scratch.
> - **Tokenization of staged parquets** → same `xfer` allocation, or piggybacked on a GH200 training job during dataloader warm-up.
> - **Anything on the previously-running gcloud tokenizer instance** (held-out contamination check via the C3 mix manifest, late merged-variant builds, the active gcloud worker mentioned in earlier doc drafts) → see `ANALYSIS.md` §1.4 finding 6 for the alternatives.
>
> Home (this server) is now the only place with the GCloud-side mirror
> of glossapi_work artifacts; it has 354 GB of mirror under
> `/home/foivos/data/glossapi_work/` but only 126 GB free, so heavy
> staging belongs on CSCS, not here.

## 3. Existing Apertus-Greek work in `a0140` (p-skarvelis)

Six CPT runs + one full SFT, dating from 2026-04-17. **HF Trainer
based, not Megatron-LM, not nanotron.** Configs are group-readable.

### 3.1 The setup (from latest prod-run `run_config.json`)

```json
{
  "model_path": "/iopsstor/scratch/cscs/p-skarvelis/apertus-greek-init/",
  "torch_dtype": "bfloat16",
  "attn_implementation": "sdpa",
  "gradient_checkpointing": false,
  "max_seq_length": 2048,
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 16,
  "expected_world_size": 16,             // 4 nodes × 4 GH200
  "effective_global_batch_size": 256,
  "greek_dataset": "epfml/FineWeb2-HQ", "greek_config": "ell_Grek", "greek_probability": 0.9,
  "english_dataset": "epfml/FineWeb-HQ", "english_probability": 0.1,
  "lr_scheduler_type": "cosine",
  "phase_plan": {
    "warmup": {"max_steps": 300, "learning_rate": 1e-4},
    "full":   {"max_steps": 700, "learning_rate": 2e-5, "warmup_steps": 100}
  },
  "tokenizer_vocab_size": 142344
}
```

Apertus model config (from SFT output):

```json
{
  "architectures": ["ApertusForCausalLM"],
  "hidden_act": "xielu", "qk_norm": true, "post_norm": false,
  "tie_word_embeddings": false, "hidden_size": 4096,
  "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8,
  "intermediate_size": 21504, "max_position_embeddings": 65536,
  "rope_scaling": {"factor": 8.0, "rope_type": "llama3"},
  "transformers_version": "4.57.6",
  "vocab_size": 142344
}
```

So the setup is:
- **HF Trainer + ApertusForCausalLM** (transformers 4.57.6) — answers [Review checkpoint D in ANALYSIS.md](../03_3_cscs_experiments_kickoff/ANALYSIS.md#7-review-checkpoints--what-still-needs-your-explicit-sign-off): the in-house team chose option (c) HF Transformers, not Megatron-LM-Swiss-AI or nanotron.
- **SDPA attention** (not FA-2). Lower throughput than FA-2 but simpler.
- **No gradient checkpointing.** Saves time at the cost of memory; the GH200 96 GB has slack for Apertus-8B.
- **seq=2048**, not 4096. ~30 % faster per token; loses long-context capability for now.
- **Greek 90 / English 10 mix**, FineWeb-2-HQ vs FineWeb-HQ. **Not** GlossAPI, **not** HPLT-clean60, **not** the dedup-audited pool. Effectively a clean-room baseline using only HF-released corpora.
- **Tokenizer 142,344 vocab** — not C3-17,408 (which would be 148,480). 11,272 added units; **doesn't match** any of our planned ship variants. The continue-run at 136,072 uses a 5,000-token-added variant.
- **Cosine LR**, two-phase: warmup 300 steps @ 1e-4 → full 700 steps @ 2e-5.

### 3.2 Measured throughput (from `phase_metrics.json` of the 1000-step prod)

```
phase_token_budget:           367,001,600   (= 256 batch × 2048 seq × 700 steps)
cluster_tokens_per_second:    107,234
tokens_per_second_per_gpu:    6,702
train_runtime:                3,422 sec  ≈ 57 min
```

**This is real measured data on GH200 SDPA bf16 nogc seq=2048.** Use it to recalibrate the sizing table in [`AUTH_AND_NODE_FINDING.md § 6.1`](AUTH_AND_NODE_FINDING.md#6-how-big-a-job-do-we-actually-need):

| | their measurement | our seq=2048 estimate | seq=4096 (= Apertus pretraining) extrapolation* |
|---|---:|---:|---:|
| tok/s per GPU | **6,702** (seq 2048, SDPA, nogc, bf16) | 6,000 (estimate) | ~3,500-4,500 (FA-2 + grad-ckpt; seq^2 attention cost is the limiter) |
| 1 node (4 GPUs) tok/s | 26,808 | 24,000 | ~16,000 |
| 4 nodes (16 GPUs) tok/s | **107,234 (measured)** | 96,000 | ~64,000 |

*Going from seq=2048 to seq=4096 roughly halves throughput because attention is quadratic in seq length and the SwissAI Megatron-LM Apertus pretraining recipe uses gradient checkpointing. The exact factor depends on FA-2 availability.

For a 4-node 10 B-token pilot at seq=2048 SDPA nogc: **~26 h wall, in line with our previous estimate.** For seq=4096 with FA-2 + grad-ckpt: ~43 h, needs 4 chained 12-h jobs.

### 3.3 SFT outcome (from `apertus-greek-sft/`)

```
dataset: swiss-ai/apertus-sft-mixture  (3,787,981 train / 1,972 eval)
seq_length: 1024 (truncation_side=left)
epoch: 1.0
train_loss: 0.319
eval_loss: 0.654
train_runtime: 37,945 sec ≈ 10.5 h
samples/sec/step: 99.8  /  3.12 steps/sec
distributed_strategy: ddp
```

Full HF model output on scratch: ~16 GB across 4 safetensors shards, plus three intermediate checkpoints (117500, 118000, 118375 — sub-epoch saves). **This is a usable Apertus-Greek-CPT-then-SFT model checkpoint** sitting in our project scratch, on a 142,344-vocab tokenizer.

### 3.4 Implications for our plan

1. **The "training harness" review checkpoint (D in ANALYSIS.md) is effectively pre-resolved**: p-skarvelis has a working HF-Trainer-based CPT pipeline. Adopting their pipeline is much cheaper than building a new one. Tradeoff: lose the option of matching Apertus pretraining's exact recipe (AdEMAMix / WSD / 0.1 grad-clip) — but their results show the loss curve is well-behaved (final train_loss ~2.06, smooth cosine descent) and SFT lands at eval_loss 0.654.
2. **Their tokenizers are different from ours** (142,344 and 136,072 — neither is C3-17,408's 148,480). To use our ship bundle we'd need to:
   - Either re-do their `apertus-greek-init/` with our tokenizer and re-train (drop their existing checkpoints — they're tokenizer-locked).
   - Or use their existing checkpoints unchanged for downstream work, accepting the non-C3 tokenizer (gives up the C3-extension benefits).
   - **My read**: switch to our tokenizer, build a fresh init, run the three-arm comparison against our ship bundle. Their existing artifacts become a reference baseline ("here's what a tokenizer-different CPT did with this data + recipe") rather than the trunk.
3. **Their dataset choice is the simpler option** (FineWeb-2-HQ Greek + FineWeb-HQ English) — same as `Apertus_plan.md` baseline. **Our planned curriculum (HPLT-broad → GlossAPI register diversity → academic → dictionary) is a real upgrade** on this and would justify a fresh run; using their existing checkpoints with our curriculum would not work because the tokenizer is different.
4. **Throughput numbers in [`AUTH_AND_NODE_FINDING.md`](AUTH_AND_NODE_FINDING.md) are validated** by their measured 6,702 tok/s/GPU at seq=2048 SDPA nogc. The 1-node 1 B-token calibration we proposed (12 h walltime) is reasonable; we know it'll land at ~10 h actual.
5. **Storage strategy**: their `apertus-greek-init/` and `prepared-datasets/` are on `iopsstor` (high-IOPS, random-read for dataloader). Their checkpoints land on `capstor/scratch`. Our plan should mirror this: stage tokenized data on iopsstor, write checkpoints to capstor.

### 3.5 What we should NOT do yet

Read their files, learn the recipe, mirror the storage layout — yes.
**Do not modify, copy, or use their checkpoints in any of our runs
without their explicit OK.** Group-readable ≠ "use in production."
The right move is to coordinate with p-skarvelis (and / or whoever
authored `Apertus_plan.md`) before any new submission so we don't
spend GPU on a problem they already solved.

## 4. Coordination recommendation

Before we submit even a calibration run:

1. **Identify p-skarvelis** and confirm whether their work is part of the same project we're sub-subprojects of, or a parallel arm of someone else's. ([Apertus_plan.md](../Apertus_plan.md) signs off "Xronopoulos → Petros Stefaneas" — neither name matches `p-skarvelis` exactly, so this might be a third colleague.)
2. **Decide whether we adopt their pipeline as the trunk** (HF Trainer, FineWeb-2-HQ + FineWeb-HQ, cosine LR 1e-4 → 2e-5, seq=2048 SDPA nogc). My read: yes for the engineering scaffold, no for the corpus mix — keep our curriculum (HPLT + register diversity) and the C3 17,408 ship tokenizer.
3. **Negotiate write access** to `/capstor/store/cscs/swissai/a0140/` if the project store doesn't exist yet, or fall back to `/capstor/scratch/cscs/fffoivos/` (we have plenty of room there).
4. **Ask whether their `apertus-greek-init/` build process is documented**: how they built the 142,344-vocab init from the base Apertus checkpoint. We'd run the same process on our 148,480-vocab tokenizer.

## 5. Quick reference — the storage paths we'll use

```
# Datasets (random-read, high-IOPS)
/iopsstor/scratch/cscs/fffoivos/cpt_corpus_v1/             # post-internal-dedup HPLT+GlossAPI mix (to be staged)
/iopsstor/scratch/cscs/fffoivos/tokenizers/                # ship-bundle mirror
/iopsstor/scratch/cscs/fffoivos/.uenv-images/              # already in use, 8.2 GB

# Training outputs (large-sequential)
/capstor/scratch/cscs/fffoivos/runs/                       # per-run subdirectories
/capstor/scratch/cscs/fffoivos/runs/vanilla_calibration_v1/

# Permanent (after project store is created)
/capstor/store/cscs/swissai/a0140/                         # TODO — request creation
```

**For now, mirror p-skarvelis's pattern.** Datasets + init on
iopsstor; runs + checkpoints on capstor/scratch.
