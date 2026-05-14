# Phase 2 multi-GPU execution plan (gcloud)

Companion to [APERTUS_EMBEDDING_INIT_TEST_PLAN.md](APERTUS_EMBEDDING_INIT_TEST_PLAN.md).
Operational recipe for running **Phase 2 only** (the LOO behavioural
NLL evaluation) data-parallel across multiple A100-40GB GPUs on a
single `a2-highgpu-Ng` instance.

Other GPU work in the test plan — §2.7.9 (per-layer probe), Phase
1.5 (CTX cache), Phase 6.5 (mini-CPT) — stays on `a2-highgpu-1g`
because each is single-pass or sequential and gains nothing from
multi-GPU. This doc covers Phase 2 exclusively.

Date: 2026-05-12. Plan version: v1.

---

## 1. Scope and what this plan does NOT cover

**In scope:** the Phase 2 forward-pass workload only —
12 init methods × 2 modes × 100 LOO tokens × ~3.85 M Greek tokens
across ~5,262 docs from `hplt_el.parquet`. 24 independent
(method × mode) forward passes, distributed across N GPUs.

**Not in scope:**
- Phase 1 / Phase 1.5 / Phase 3 / Phase 6.5 — they stay on home or
  on a single-GPU instance per §7.4 of the test plan.
- The diagnostic layer (Phase 0 + §2.7.x + §6.1). It runs on home
  CPU plus the existing `a2-highgpu-1g` for §2.7.9 only.
- The decision of *whether* to commit to the init-benchmark layer.
  This plan presupposes the diagnostic layer has run and the
  project has decided to run the benchmark.

## 2. SKU choice

| SKU | A100s | Phase 2 wall | $ (eur-w4) | Notes |
|---|---|---|---|---|
| `a2-highgpu-2g` | 2 | ~90 min | ~$11 | Fallback if 4g is stocked-out in eur-w4-a / -b |
| `a2-highgpu-4g` | 4 | ~45 min | ~$11 | **Default.** Best wall-time at flat $ |
| `a2-highgpu-8g` | 8 | ~22 min | ~$11 | Diminishing returns; 24 pairs / 8 GPUs = 3 pairs per GPU + model-load amortisation gets less favourable |

Default: **`a2-highgpu-4g` in `europe-west4-a`** (same zone as the
existing `apertus-greek-gpu-phaseb` instance — keeps egress/PD-region
matching). Fallback zone: `europe-west4-b`.

## 3. Instance procurement

We create a **new instance** rather than resizing the existing
1g. Reasons:
- Resizing requires stop + machine-type change + start, and gcloud
  imposes constraints on which machine-type changes are allowed
  for instances with attached accelerators.
- A fresh instance from the deeplearning-platform image gives us
  a clean CUDA / NCCL stack, which matters for multi-GPU.
- The 1g instance is preserved for Phase 1.5 + Phase 6.5; we don't
  want to disrupt its state.

Procurement command (fill in `PROJECT_ID`):

```bash
gcloud compute instances create apertus-greek-gpu-phase2-4g \
  --project=PROJECT_ID \
  --zone=europe-west4-a \
  --machine-type=a2-highgpu-4g \
  --image-family=common-cu124-debian-11 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True" \
  --scopes=cloud-platform \
  --labels=owner=foivos,purpose=phase2-loo-eval
```

The `common-cu124-debian-11` deeplearning-platform image ships
CUDA 12.4 + NVIDIA driver + Python 3.11 + a generic PyTorch venv.
Confirm `nvidia-smi` shows 4 A100-40GBs before doing anything else.

## 4. File layout on the 4g instance

```
/home/foivos/runs/phase2_loo_20260512/
├── inputs/                              # uploaded from home
│   ├── hplt_el.parquet                  # eval corpus, ~30-50 MB
│   ├── loo_target_ids.json              # 100 LOO Greek token ids
│   ├── init_candidates/
│   │   ├── method=C-global.npz          # {ids: (100,), e_init: (100,4096), u_init: (100,4096)}
│   │   ├── method=C-group.npz
│   │   ├── method=C-group-normmatch.npz
│   │   ├── method=C-group-noise.npz
│   │   ├── method=A-aniso.npz
│   │   ├── method=A-PCs-only.npz
│   │   ├── method=R1.npz
│   │   ├── method=R2.npz
│   │   ├── method=R2-groupnormmatch.npz
│   │   ├── method=CTX.npz               # uses Phase 1.5 hidden-state cache
│   │   ├── method=R2-CG.npz
│   │   ├── method=Z.npz                 # zero-init baseline
│   │   ├── method=N0sigma2.npz          # Gaussian baseline
│   │   └── method=NormScrambled.npz     # norm-only baseline
│   └── assignments.json                 # {worker_rank: [(method, mode), ...]}
├── scripts/
│   ├── run_phase2_worker.py             # one worker, one GPU
│   ├── run_phase2_orchestrator.py       # spawns N workers via torch.multiprocessing
│   └── aggregate_phase2.py              # merges worker JSONs
├── workdir/                             # per-worker logs + intermediate state
│   ├── worker_0.log
│   ├── worker_1.log
│   ├── worker_2.log
│   └── worker_3.log
└── results/
    ├── worker_0.json                    # per-worker partial: {method:{mode:{nll_delta_per_token}}}
    ├── worker_1.json
    ├── worker_2.json
    ├── worker_3.json
    └── phase2_loo_results.json          # aggregated final
```

Inputs are pre-built on home in Phase 1; ~50–100 MB total, scp'd
to the 4g instance once.

## 5. Orchestration design

### 5.1 Why `torch.multiprocessing.spawn` over `torchrun`

`torchrun` is designed for collective training (DDP, all-reduce).
Phase 2 has **no inter-worker communication during the run** — each
worker handles its own (method × mode) pairs end-to-end. A simpler
`torch.multiprocessing.spawn(worker_fn, nprocs=N)` is enough and
avoids the NCCL init overhead.

Each spawned process:
1. Sets `CUDA_VISIBLE_DEVICES=<rank>` (so each sees only its assigned GPU).
2. Loads `swiss-ai/Apertus-8B-2509` in bf16 (~16 GB on the visible GPU).
3. Reads its assignment from `assignments.json`.
4. For each `(method, mode)` in assignment:
   - Loads `init_candidates/method=<m>.npz`.
   - Edits `model.get_output_embeddings().weight.data[loo_ids] = u_init`.
   - If mode == B: also edits `model.get_input_embeddings().weight.data[loo_ids] = e_init`.
   - Runs the per-doc forward pass over `hplt_el.parquet`, accumulating
     per-token NLL_delta only at positions where target id ∈ `loo_target_ids`.
   - Restores the original rows.
5. Writes `results/worker_<rank>.json` and exits.

### 5.2 Assignment generation

`assignments.json` is built deterministically before the run:
24 pairs sorted by descending expected cost (CTX-mode-B most
expensive, Z-mode-A cheapest) then round-robin allocated across
N workers. Pre-computed so the run is reproducible and so a failed
worker can be re-run in isolation.

For N=4: each worker gets 6 pairs ≈ 6 × ~7 min = ~42 min wall.
For N=2: each worker gets 12 pairs ≈ 12 × ~7 min = ~84 min wall.

### 5.3 Per-worker NLL machinery

Reuse the v5 NLL forward-pass code from
`/home/foivos/runs/apertus_greek_phase_b_v4_20260512/phase_b_v5_nll_triple.py`
verbatim for the actual forward pass + per-position cross-entropy.
The only additions for Phase 2:

- Weight-swap + restore around each pass.
- Position-mask to accumulate NLL only where target ∈ LOO ids.
- Baseline-NLL recording (one extra pass with the *original* weights
  per worker, to compute `NLL_delta = NLL_swapped - NLL_baseline`
  per token). The baseline pass is shared across all of the
  worker's assigned methods — one extra ~7 min upfront per worker.

### 5.4 Aggregation

`aggregate_phase2.py`:
1. Reads all `results/worker_*.json`.
2. Concatenates into `{method: {mode: {token_id: nll_delta}}}`.
3. Computes the §4.5 summary stats: median, p25/p75/p95, fraction
   beating C-group baseline, per-frequency-quartile breakdown.
4. Writes `phase2_loo_results.json` + `phase2_loo_summary.json` +
   `figures/nll_delta_box_per_method.png` +
   `figures/nll_delta_vs_token_frequency.png`.

## 6. Pre-flight (on home, before any GPU spend)

Run before creating the 4g instance:

1. **Phase 1 init catalog complete**: every `init_candidates/method=*.npz`
   built locally; spot-check `||u_init|| in (0.3, 0.8)` and
   `cos(u_init, mu_<greek>) > 0.3` for at least 2 random tokens per
   method.
2. **Round-trip swap test on 1 doc (CPU)**: load Apertus on CPU,
   swap rows, swap back, confirm `model.state_dict()` hashes match
   pre-swap. Catches reference-vs-copy mistakes before they corrupt
   the GPU run.
3. **`assignments.json` generated** and committed; one-line summary:
   "N=4 workers, 6 pairs each, 24 pairs total".

Then, on a **2-pair dry run** on the existing `a2-highgpu-1g` (NOT
on the 4g — we want zero $ on the multi-GPU instance until we know
the script works):
- Pick the 2 cheapest pairs from `assignments.json` (e.g.
  `(Z, mode_A)` and `(Z, mode_B)`).
- Run the worker script with `--max-docs 10 --pairs Z,A Z,B`.
- Check the output JSON has the expected schema.
- Compare per-token NLL_delta against a reference single-GPU run
  of the same 2 pairs. Tolerate < 1e-3 NLL difference (bf16
  non-determinism).

Only after dry-run passes: create the 4g instance.

## 7. Phase 2 execution on the 4g instance

```bash
# 1. From home, sync inputs to the new instance
gcloud compute scp --recurse \
  /home/foivos/runs/apertus_embedding_init_test_20260512/loo_inputs/ \
  apertus-greek-gpu-phase2-4g:/home/foivos/runs/phase2_loo_20260512/inputs/ \
  --zone=europe-west4-a

# 2. SSH in and start the run
gcloud compute ssh apertus-greek-gpu-phase2-4g --zone=europe-west4-a -- \
  "cd /home/foivos/runs/phase2_loo_20260512 && \
   nohup python scripts/run_phase2_orchestrator.py \
     --nprocs 4 \
     --assignments inputs/assignments.json \
     --eval-corpus inputs/hplt_el.parquet \
     --loo-ids inputs/loo_target_ids.json \
     --init-dir inputs/init_candidates \
     --results-dir results/ \
     > workdir/orchestrator.log 2>&1 &"

# 3. Tail progress (each worker logs roughly every 30s)
gcloud compute ssh apertus-greek-gpu-phase2-4g --zone=europe-west4-a -- \
  "tail -f /home/foivos/runs/phase2_loo_20260512/workdir/worker_0.log"
```

Expected timeline (wall, from `t0`):
- `t0 + 0`: orchestrator starts, spawns 4 workers.
- `t0 + 1 min`: all 4 workers have loaded Apertus (model is in the
  HF cache on the boot disk).
- `t0 + 8 min`: baseline pass complete on each worker.
- `t0 + 42 min`: all 4 workers finished their 6 (method × mode) pairs.
- `t0 + 45 min`: aggregator has produced `phase2_loo_results.json`.

If a worker dies mid-run (OOM, NCCL hiccup, transient driver
issue): each worker's progress is checkpointed after every
(method × mode) pair to `workdir/worker_<rank>_done.txt`.
Re-running the orchestrator skips completed pairs.

## 8. Stop and cleanup

When `phase2_loo_results.json` is written and visually sane:

```bash
# 1. Pull results back to home
gcloud compute scp --recurse \
  apertus-greek-gpu-phase2-4g:/home/foivos/runs/phase2_loo_20260512/results/ \
  /home/foivos/runs/apertus_embedding_init_test_20260512/loo/ \
  --zone=europe-west4-a

# 2. Stop the instance — paid compute compounds; per [feedback_instance_stop_decision] memory.
gcloud compute instances stop apertus-greek-gpu-phase2-4g \
  --zone=europe-west4-a

# 3. Delete (the 4g instance is single-purpose for Phase 2 only;
#    no Phase 6.5 / Phase 1.5 reason to keep it around).
#    Confirm results are on home first.
gcloud compute instances delete apertus-greek-gpu-phase2-4g \
  --zone=europe-west4-a --quiet
```

Decision per the three-step rule (memory `feedback_instance_stop_decision`):
1. Anything lost on delete? — The HF model cache (re-downloadable in
   ~5 min) and the venv (re-installable). Inputs were uploaded from
   home; results have been pulled back to home. **Nothing lost.**
2. Important enough to keep running? — No. Phase 6.5 uses the 1g
   instance.
3. Pause vs delete? — Delete. Single-purpose run, no future use,
   stop-state still bills for the boot disk.

## 9. Cost reality check

Multi-GPU does not reduce $ — only wall. Phase 2 cost stays ~$11
across SKU choices because hourly scales linearly with N.

| SKU | wall | $ | Phase 2 only |
|---|---|---|---|
| `a2-highgpu-1g` | ~3 h | ~$11 | — |
| `a2-highgpu-2g` | ~90 min | ~$11 | fallback if 4g stocked out |
| `a2-highgpu-4g` | ~45 min | ~$11 | **default** |

The full init-benchmark layer including Phase 1.5 and Phase 6.5 on
the 1g instance is still ~$22 / ~3–4 h wall (see test plan §7.2).

## 10. Failure modes worth handling explicitly

- **A100 stockout in eur-w4-a**: fall back to eur-w4-b (where the
  CPU tokenizer instance lives). If both are stocked out, drop to
  `a2-highgpu-2g`. If even 2g is stocked out, run sequentially on
  the 1g — accept ~3 h wall.
- **Driver mismatch on fresh image**: the `install-nvidia-driver=True`
  metadata flag triggers a first-boot install. Run `nvidia-smi` in
  the SSH session before launching the orchestrator; if it returns
  no devices, wait 2 min and retry. If still failing, reboot the
  instance once.
- **OOM on bf16 model + activations**: shouldn't happen — Apertus-8B
  bf16 is ~16 GB, leaving ~24 GB on a 40 GB A100 for activations.
  But if it does, drop the per-doc batch from 1 doc whole to
  doc-chunked-by-2048-tokens.
- **Worker hangs without progress**: the orchestrator runs a heartbeat
  check every 60 s on each worker's `workdir/worker_<rank>_progress.json`;
  if a worker hasn't updated in 300 s, kill it and re-spawn with
  remaining pairs from its assignment.
- **Aggregation mismatch with single-GPU dry run**: the dry-run
  reference (§6) is the source of truth. If aggregation diverges by
  > 1e-3 NLL on the 2 dry-run pairs, halt and investigate before
  trusting the full 24-pair output.

## 11. What to write into the Phase 2 results that lets §6.2 read off

`phase2_loo_results.json` schema (consumed by `aggregate_phase2.py`
and by §6.2.3 of the test plan's writeup):

```json
{
  "schema_version": "phase2_v1",
  "run_date": "2026-05-12",
  "instance": "apertus-greek-gpu-phase2-4g",
  "instance_sku": "a2-highgpu-4g",
  "n_workers": 4,
  "loo_target_ids": "<path to inputs/loo_target_ids.json>",
  "baseline_nll_per_token": {"<token_id>": <nll_baseline>, ...},
  "results": {
    "<method_name>": {
      "mode_A": {
        "nll_delta_per_token": {"<token_id>": <delta>, ...},
        "wall_seconds": <int>,
        "worker_rank": <int>
      },
      "mode_B": {...}
    },
    ...
  }
}
```

The aggregator further computes `phase2_loo_summary.json` with
per-(method, mode) median / p25 / p75 / p95 / frac_better_than_C_group,
+ per-frequency-quartile breakdown, which is what §6.2.3 publishes
in the decision-card table.

---

## Appendix A — assignment generator (Python sketch, ~30 LOC)

```python
import json
from pathlib import Path

METHODS = [
    "C-global", "C-group", "C-group-normmatch", "C-group-noise",
    "A-aniso", "A-PCs-only",
    "R1", "R2", "R2-groupnormmatch",
    "CTX", "R2-CG",
    "Z", "N0sigma2", "NormScrambled",
]
# NOTE: trim to 12 if budget-constrained per test plan §3.7.

MODES = ["mode_A", "mode_B"]

# Rough expected-cost ordering (most expensive first).
COST_RANK = {
    ("CTX", "mode_B"): 0, ("CTX", "mode_A"): 1,
    ("R2-CG", "mode_B"): 2, ("R2-CG", "mode_A"): 3,
    # ... fill in the rest; ties broken by alphabetical order.
}

def build_assignments(methods, modes, nprocs):
    pairs = [(m, mode) for m in methods for mode in modes]
    pairs.sort(key=lambda p: COST_RANK.get(p, 999))
    assignments = {r: [] for r in range(nprocs)}
    for i, p in enumerate(pairs):
        assignments[i % nprocs].append(p)
    return assignments

if __name__ == "__main__":
    a = build_assignments(METHODS, MODES, nprocs=4)
    Path("inputs/assignments.json").write_text(
        json.dumps(a, indent=2, sort_keys=True)
    )
```

## Appendix B — worker script outline (~80 LOC)

```python
# scripts/run_phase2_worker.py
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main(rank, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    device = "cuda:0"

    model = AutoModelForCausalLM.from_pretrained(
        "swiss-ai/Apertus-8B-2509",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained("swiss-ai/Apertus-8B-2509")

    loo_ids = json.loads(Path(args.loo_ids).read_text())
    loo_set = set(loo_ids)
    assignments = json.loads(Path(args.assignments).read_text())
    my_pairs = assignments[str(rank)]

    # Baseline NLL pass.
    baseline = run_nll(model, args.eval_corpus, loo_set, device)

    results = {}
    E = model.get_input_embeddings().weight.data
    U = model.get_output_embeddings().weight.data
    E_orig = E[loo_ids].clone()
    U_orig = U[loo_ids].clone()

    for method, mode in my_pairs:
        cand = np.load(Path(args.init_dir) / f"method={method}.npz")
        t0 = time.time()
        # Swap.
        U[loo_ids] = torch.from_numpy(cand["u_init"]).to(U.dtype).to(device)
        if mode == "mode_B":
            E[loo_ids] = torch.from_numpy(cand["e_init"]).to(E.dtype).to(device)

        swapped = run_nll(model, args.eval_corpus, loo_set, device)
        delta = {tid: swapped[tid] - baseline[tid] for tid in loo_set if tid in swapped}

        # Restore.
        U[loo_ids] = U_orig
        if mode == "mode_B":
            E[loo_ids] = E_orig

        results.setdefault(method, {})[mode] = {
            "nll_delta_per_token": delta,
            "wall_seconds": int(time.time() - t0),
            "worker_rank": rank,
        }
        # Checkpoint per pair.
        Path(args.workdir / f"worker_{rank}_done.txt").open("a").write(
            f"{method}\t{mode}\n"
        )

    out = {
        "schema_version": "phase2_v1",
        "worker_rank": rank,
        "baseline_nll_per_token": baseline,
        "results": results,
    }
    Path(args.results_dir / f"worker_{rank}.json").write_text(
        json.dumps(out, indent=2)
    )

# run_nll(...) is lifted from phase_b_v5_nll_triple.py and adapted
# to mask positions to LOO targets only.
```

## Appendix C — orchestrator (~40 LOC)

```python
# scripts/run_phase2_orchestrator.py
import argparse
from pathlib import Path
import torch.multiprocessing as mp
from run_phase2_worker import main as worker_main

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nprocs", type=int, default=4)
    p.add_argument("--assignments", required=True)
    p.add_argument("--eval-corpus", required=True)
    p.add_argument("--loo-ids", required=True)
    p.add_argument("--init-dir", required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--workdir", default="workdir/")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    mp.spawn(worker_main, args=(args,), nprocs=args.nprocs, join=True)
    # Aggregate.
    import subprocess
    subprocess.check_call([
        "python", "scripts/aggregate_phase2.py",
        "--results-dir", args.results_dir,
    ])
```
