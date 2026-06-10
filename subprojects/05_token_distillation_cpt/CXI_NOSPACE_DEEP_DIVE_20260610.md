# CXI `NO_SPACE` Deep Dive + Test Plan — 2026-06-10

> **UPDATE (evening, 5-angle re-investigation + 4 live tests): both the
> comm-multiplicity mechanism (§ below) and the allocator/VMM lead are now
> DEAD. See "Round 2" at the end of this doc for current state + ranked tests.**

Reviewer analysis of the multi-node `NET/OFI … Error 16 (NO_SPACE)` blocker.
Complements the execution agent's bisection in `RUN_LOG_20260609_CPT_2ARM.md`
(which correctly established: 1-node works, Socket/HSN works, CXI fails at the
first inter-node collective). This doc adds the **mechanism**, **eliminations**,
and a **ranked test plan**.

## Bottom line

- **Not disk, not Megatron, not data, not checkpoint, not NCCL itself, not the
  GPU count, not protocol/channels, not message buffers, not match-mode, not
  RDZV, not library versions.** All verified eliminated (see below).
- **Deepest supported cause:** aggregate **per-NIC CXI receive-resource
  exhaustion driven by the *number* of NCCL communicators** (~16/rank) that full
  Megatron creates, all sharing the 4 Cassini NICs at the first
  `DATA_PARALLEL_GROUP_WITH_CP` collective. The `FI_ENOSPC` lands on a **4-byte
  control RECV** (`size:4, state:CREATED, direction:RECV`) — a descriptor-alloc
  failure, not a payload problem.
- **The resource that's exhausted is a *hard* per-NIC/endpoint limit** (PtlTE /
  RX command-queue / plugin request pool), **not** one the user-space env can
  grow — proven by the fact that software-match + 256 MB request buffers +
  hybrid + alt_read all still fail.
- **Practical consequence:** this is most likely a CSCS/HPE-level config or an
  aws-ofi-nccl behavior at high communicator counts — i.e. a **support ticket**,
  not more knob-guessing. Meanwhile **Socket/HSN is the validated path** to run.

## The mechanism (why it's deeper than "CXI is broken")

The discriminating evidence isolates *communicator multiplicity*, not message
size or transport correctness:

| Configuration | Comms/rank | Fabric | Result |
|---|---|---|---|
| 1 node, 4 GPU | few | none (NVLink) | ✅ trains |
| Pure PyTorch, 2 node, Megatron-shaped TP/DP groups, **exact** failing size (67,108,864 bf16) | ~3 | CXI | ✅ passes (jobs 2514102/2514107) |
| **Full Megatron, 2 node** | **~16** | CXI | ❌ `NO_SPACE` at first DP collective |
| Full Megatron, 2/4 node, `NCCL_NET=Socket` | ~16 | TCP/HSN | ✅ trains |

The only variable that flips full-Megatron-CXI from the passing rows is the
**number of communicators concurrently initializing receive resources on the
shared NIC**. Pure-PyTorch with the identical collective *size and group shape*
passes because it instantiates ~3 comms, not ~16.

## Verified eliminations (don't re-test these)

- **Library versions are modern and matched** (this refutes the Apertus
  "version-drift" hypothesis): the NCCL plugin loads
  `libfabric 2.4.0-dev` (`/user-environment/.../libfabric-2.4.0-dev/lib/libfabric.so.1.30.0`),
  **aws-ofi-nccl 1.17.2**, **NCCL 2.29.2**, **libcxi 13.0.0 (SHS 13)**. libfabric
  is *above* the ≥2.2 the Apertus paper required. (The `/opt/cray/libfabric/1.22.0`
  is the unused system lib; the `EFA requires 1.22.0` string is an AWS-EFA path,
  irrelevant to the `cxi` provider in use.)
- **Message-buffer sizing**: `FI_CXI_REQ_BUF_SIZE` up to 256 MB +
  `FI_CXI_OFLOW_BUF_SIZE` 256 MB → still `NO_SPACE`.
- **Match mode**: software *and* hybrid (the uenv default) → still `NO_SPACE`.
- **Rendezvous**: `alt_read` + `RDZV_{GET_MIN,THRESHOLD,EAGER_SIZE}=0` (uenv
  default) → still `NO_SPACE`.
- **`FI_CXI_DEFAULT_RX_SIZE`** is a dead knob: max is 15360, anything higher
  ("Default RX size invalid. Setting to 1024") reverts to 1024.
- **Protocol / channels**: `NCCL_PROTO=^LL128` *and* `Simple`;
  `NCCL_NCHANNELS_PER_NET_PEER` 4 *and* 1 → still `NO_SPACE` (job 2514875).
- **Distributed-optimizer on/off, comm-overlap on/off, DDP bucket 40M/5M,
  goldfish vs ntp, mock vs real data, torchrun vs direct-srun, CPU affinity** →
  all still `NO_SPACE`.
- **`OFI_NCCL_PROTOCOL=RDMA`**: fails differently (`fi_writedata RC -38, not
  implemented`) — not usable here.

## Ranked test plan

Run each as a 2-node, mock-data, 1-iteration debug smoke
(`USE_MOCK_DATA=1 EXIT_INTERVAL=1 DISABLE_SAVE=1 ENABLE_EXTRA_VALID=0`,
`FI_LOG_LEVEL=warn FI_LOG_PROV=cxi`). Success = reaches `iteration 1/`.

### T1 — Hybrid *preemptive* flags (highest value, untried, HPE-documented) — ~3 min
The HPE remedy for "LE resources not recovered during flow control" is to force
the software-match transition **before** the hardware pool exhausts, not after.
These four are **forwarded by the trainer (lines 524-527) but have never been
set**:
```
FI_CXI_HYBRID_PREEMPTIVE=1
FI_CXI_HYBRID_RECV_PREEMPTIVE=1
FI_CXI_HYBRID_POSTED_RECV_PREEMPTIVE=1
FI_CXI_HYBRID_UNEXPECTED_MSG_PREEMPTIVE=1
```
(keep the uenv's hybrid mode; do **not** set `FI_CXI_DEFAULT_RX_SIZE`.)
This is the single most-likely env fix and directly targets the mechanism.

### T2 — Confirm the communicator-count mechanism — ~3 min + a 1-line edit
Add `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` to the trainer's uenv-forward
list (they are *not* currently forwarded — `NCCL_NCHANNELS_PER_NET_PEER` alone
was, which is why my earlier probe didn't actually cap total channels), then run
`NCCL_MAX_NCHANNELS=2 NCCL_MIN_NCHANNELS=1 NCCL_PROTO=Simple`. If fewer *total*
channels passes, the mechanism is confirmed and we have a production lever.

### T3 — Reduce communicator count via parallel layout — ~5 min
Put the inter-node dimension on **pipeline** (point-to-point sends) rather than
**data** parallel, which removes the large concurrent DP all-reduce from the
first step: 2-node `TP=2 PP=2 DP=1` (mock data, so no checkpoint-shape issue;
the TP2 checkpoint still loads since TP is unchanged). If `PP`-as-the-inter-node
dim passes where `DP` fails, it pins the cause to the DP-group receive fan-in.

### T4 — aws-ofi-nccl request-pool / chunk knobs — ~3 min each
`NCCL_BUFFSIZE` smaller (e.g. 1048576) and/or `NCCL_P2P_NET_CHUNKSIZE` smaller
to cut outstanding per-comm receives. Low confidence but cheap.

### T5 — CSCS/SwissAI support ticket (the decisive path if T1-T4 fail)
We now have a **clean, minimal reproducer and a modern verified stack**, which
is exactly what CSCS needs. Ticket contents:
- Stack: uenv `pytorch/v2.9.1:v2`, libfabric 2.4.0-dev, aws-ofi-nccl 1.17.2,
  NCCL 2.29.2, libcxi 13.0.0, GH200 / Slingshot-11 / Cassini.
- Symptom: `NET/OFI Request … RC:5 Error:16 (NO_SPACE)`, `size:4 RECV`, in
  `DATA_PARALLEL_GROUP_WITH_CP`, at the **first** Megatron collective, ≥2 nodes.
- A/B: `NCCL_NET=Socket` over HSN trains; `NCCL_NET="AWS Libfabric"` fails.
  Pure-PyTorch collectives (incl. the exact failing size + group shapes) pass.
- Tried-and-failed: software/hybrid match, REQ_BUF/OFLOW 256 MB, RDZV alt_read +
  zeros, `NCCL_PROTO` Simple/^LL128, channels-per-peer 1/4, no-distopt/no-overlap.
- Ask: the validated multi-node NCCL/libfabric recipe for Megatron-LM-Swiss-AI at
  production communicator counts (Apertus was trained on this exact cluster+fork).

## Recommendation

1. **Proceed on Socket/HSN now** (validated; ~1 day/arm projected) so the
   science is not blocked. Treat CXI as a throughput optimization (~2× upside).
2. **Run T1 immediately** (preemptive flags) — highest-odds cheap env fix.
3. If T1 fails, **run T2-T3** (mechanism confirmation + a real lever), then file
   **T5** in parallel rather than continuing to brute-force knobs. We have passed
   the point where more user-space env permutations are likely to pay off.

---

# Round 2 — 5-angle re-investigation + live tests (2026-06-10 evening)

A fresh multi-angle pass (plugin source, CXI provider internals, working Alps
configs, NCCL-2.29 regression, live log forensics) plus 4 live 2-node smokes.

## New decisive forensics (kills the comm-count mechanism above)

Failing runs die at **SeqNum=1, on a single communicator, including 1-element
allreduces** (verified in 2514815.err). Failure precedes any fan-out — so ALL
multiplicity/queue-depth theories (~16 comms × channels) are dead, and so are:
RX-size cap as root cause (legal max is 15360; tried 16384/65536 silently reset
to 1024 — config hygiene only), plugin CQ size 12288 (passing probes use the
same), hybrid non-preemptive transition (software fails identically),
device-LE-pool exhaustion (probe has 4 ranks/node too).

## The discriminator that led Round 2

Every failing run logged `cuda_set_sync_memops(): CUDA_ERROR_NOT_SUPPORTED` and
`cxip_map_cache ... returned -22` 3–330 lines before the first NO_SPACE; absent
from all passing runs (pure-PyTorch probes, Socket). Hypothesis: CXI provider
can't register cuMemMap/VMM GPU memory.

## Live tests run (2-node, mock, debug)

| Job | Change | sync_memops warns | NO_SPACE | iter 1 |
|---|---|---|---|---|
| 2515041 | `PYTORCH_ALLOC_CONF=native` | — | — | crash (bad syntax: needs `backend:native`) |
| 2515042 | `backend:native` (both vars) | **still 64** | yes (same) | no |
| 2515053 | + `NCCL_CUMEM_ENABLE=0` `NCCL_CUMEM_HOST_ENABLE=0` | **0** | same Error 16, size-4 RECV | no |
| 2515057 | + `NCCL_DEBUG=WARN` confirm | 0 | Error 16 size-4 RECV | no |

**Verdict: allocator/VMM lead also dead.** Even with cuMem fully disabled and
zero registration warnings, the identical 4-byte-RECV `NO_SPACE` fires at the
same place. The sync_memops noise was correlated, not causal.

## What remains unkilled

1. **`NCCL_NET_FORCE_FLUSH=1`** (trainer-only, never unset in any failing run;
   pure-PyTorch probes don't set it; the flush op IS a tiny tracked
   size-4 RECV — exact signature match). **Test: unset it (single var), 2-node.**
2. **Stack outlier**: uenv ships aws-ofi-nccl **1.18.0-dev snapshot** + NCCL
   2.29.2 (v11 net API) + libfabric 2.4.0-dev — an unreleased combo nobody
   validates; Apertus's proven path is the NGC container (NCCL 2.25.1, host
   libfabric 1.22-SHS) with zero exports. **Test: SwissAI NGC container EDF, no
   exports.** This is the proven exit ramp.
3. GIN init notice ("GIN only supports RDMA") appears on every rank in the
   1.18-dev plugin — wiring of the dev plugin's gin/flush path on SENDRECV
   protocol is suspect together with FORCE_FLUSH.
4. Ticket addendum (drafted in Round-2 cross-exam): symptom is not load/scale/
   comm-count dependent; probe passes on identical stack; suspect 1.18-dev
   plugin flush/GIN path under SENDRECV.

## RESOLVED — root cause was `NCCL_NET_FORCE_FLUSH=1`

Job **2515069** (2-node, CXI, identical config, **only** `NCCL_NET_FORCE_FLUSH=0`):
**COMPLETED 0:0, iteration 1, zero NO_SPACE, 72.5 s/iter (~12% faster than Socket).**

Follow-up job **2515691** (4-node / 16-GPU, CXI, same no-flush mechanism):
**COMPLETED 0:0, iteration 1, zero NO_SPACE, 40.0 s/iter.** This confirms the
fix beyond the 2-node boundary.

Full-scale job **2515665** (16-node / 64-GPU, CXI, same no-flush mechanism):
**COMPLETED 0:0, iteration 1, zero NO_SPACE, 15.6 s/iter.** Follow-up
real-data timing (`2515841`), validation timing (`2515891`), and save timing
(`2515966`) estimate production allocated runtime at about **8.3-8.5 h per
arm** with the 4-segment 16-node chain.

Mechanism: the trainer's hardcoded `NCCL_NET_FORCE_FLUSH=1` makes the plugin's
SENDRECV path issue a tiny tracked flush op after every receive — that is the
exact failing object (`size:4, direction:RECV`); on the 1.18-dev plugin (GIN
warns "only supports RDMA transport") it errors with C_RC_NO_SPACE at the very
first inter-node post. Pure-PyTorch probes never set it; Socket never reaches
the OFI flush path. Single env var, off by default at CSCS and Isambard.

Action plan:
1. Remove the hardcoded `NCCL_NET_FORCE_FLUSH=1` from bakeoff_train.sbatch
   (set to 0 / drop). It is not required on GH200 (PHB + C2C coherence).
2. Relaunch BOTH arms full-run on CXI (drop the Socket fallback).
3. Cosmetic later: lift RX_SIZE only to legal ≤15360 if ever desired; doc that
   FI_CXI_DEFAULT_RX_SIZE>15360 silently reverts to 1024.
