# Auth and Node Finding — Clariden Reality Check

*Drafted 2026-05-20, with live probes against `clariden-ln001` /
`ela5`. Numbers will drift; the methodology is what's load-bearing.*

## 1. Auth state — verified working

```
$ ssh-keygen -L -f ~/.ssh/cscs-key-cert.pub | grep -E 'Valid|Key ID|Principals'
Valid:        from 2026-05-20T15:03:46 to 2026-05-21T15:03:46
Key ID:       "fffoivos"
Principals:   fffoivos

$ cscs-key list
Serial Number       Valid     Expiration  Expire Time
202251777748338796  ✅ VALID  in a day    2026-05-21 15:03:46 +03:00

$ ssh ela 'hostname; whoami; pwd'
ela5
fffoivos
/users/fffoivos

$ ssh clariden 'hostname'
clariden-ln001
```

Daily refresh command (1-day max in `cscs-key 1.1.0`):

```bash
cscs-key sign --headless --duration 1d
```

The OIDC refresh token is cached locally so subsequent days usually
skip the device-code dance — just `cscs-key sign --headless` and it
mints a new cert silently.

## 2. Cluster posture — what `sinfo` actually tells us

A first read of `sinfo` is misleading. Here's the raw view:

```
PARTITION STATE NODES CPUS(A/I/O/T)              GRES
debug     maint  1363 0/392544/0/392544          gpu:4
debug     down$  21   0/0/6048/6048              gpu:4
normal*   maint  1340 0/385920/0/385920          gpu:4
normal*   down$  19   0/0/5472/5472              gpu:4
low       maint  1340 0/385920/0/385920          gpu:4
low       down$  19   0/0/5472/5472              gpu:4
xfer      maint  2    0/256/0/256                (null)
```

**Almost every node says `maint`** (1340 out of 1340 nodes on
`normal`). The first instinct is "the cluster is closed." That's
*wrong*. CSCS uses scheduled reservations heavily and the `maint`
flag really means "node has a reservation that covers it." The
reservations may exclude specific users/groups, run continuously,
or end at a known time.

What matters for *us* (project `a0140`) is whether the scheduler
will admit our jobs. The cleanest probe is `sbatch --test-only`,
which asks the scheduler to compute the *actual* expected start
time given current queue and reservation state.

## 3. Live probes (taken 2026-05-20 ~15:08 UTC)

```
$ sbatch --account=a0140 --partition=debug  --nodes=1 --time=01:00:00 --test-only --wrap="echo"
Job 2324204 to start at 16:00:00 a using 288 processors on nodes nid005394 in partition debug

$ sbatch --account=a0140 --partition=normal --nodes=1 --time=02:00:00 --test-only --wrap="echo"
Job 2324203 to start at Tomorr 03:33 a using 288 processors on nodes nid005394 in partition normal

$ sbatch --account=a0140 --partition=normal --nodes=1 --time=12:00:00 --test-only --wrap="echo"
Job 2324197 to start at Tomorr 03:33 a using 288 processors on nodes nid005394 in partition normal

$ sbatch --account=a0140 --partition=normal --nodes=2 --time=12:00:00 --test-only --wrap="echo"
Job 2324201 to start at Tomorr 03:33 a using 576 processors on nodes nid[005394-005395] in partition normal

$ sbatch --account=a0140 --partition=normal --nodes=4 --time=12:00:00 --test-only --wrap="echo"
Job 2324202 to start at Tomorr 03:33 a using 1152 processors on nodes nid[005394-005396,005398] in partition normal
```

Reading: **everything on `normal` would start at the same wall-clock instant — tomorrow ~03:33 UTC (~12 h from probe time).** That's
when the current cluster-wide maintenance reservation expires.
`debug` starts within ~1 h (today 16:00 UTC).

Practical implications:
- **Wait time on `normal` is reservation-bound, not queue-bound.** A 1-node job waits the same as a 4-node job.
- **For a "do something today" smoke test → `-p debug --time=00:30:00` lands within ~1 h.** Cap is 1.5 h, ≤4 nodes per user (`debug-qos`).
- **For "fire-and-forget overnight" → `-p normal --time=12:00:00` lands tomorrow morning UTC and runs to its time limit.**
- Probe `--test-only` before any real submission to know what you're committing to.

## 4. User + account state

```
$ sacctmgr show user fffoivos -s
  fffoivos  a0140  clariden                          QOS=normal
  fffoivos  root   clariden                          QOS=normal
$ sshare -u fffoivos
  root                   parent  1.000000    0  0.000000   1.000000  (fairshare = baseline)
$ sacctmgr show account a0140 -s
  a0140  apriftis                    QOS=normal
  a0140  fffoivos                    QOS=normal
  a0140  p-skarvel+                  QOS=normal
```

- Account: **`a0140`** (must be explicit: `-A a0140`; default is `root` which won't admit).
- QoS: **`normal`** (no explicit per-user TRES caps in this association).
- Co-users on the account: `apriftis`, `p-skarvel+` (truncated). Their submissions can compete for the account's share — relevant if we run anything large for long.
- Fairshare: I'm under `root` with no allocated shares (`RawShares=parent`). That puts us at the **lowest priority tier**; we ride on backfill rather than priority. So far the queue has been admitting us cleanly because Clariden has slack outside reserved windows.

## 5. QoS limits worth knowing

```
$ sacctmgr show qos format=Name,Priority,GrpTRES,MaxTRES,MaxTRESPU,MaxJobsPU,MaxWall
  Name        Priority  MaxTRES  MaxTRESPU       MaxJobsPU  MaxWall
  normal       1000     -        -               -          -
  large-sc-1     0     -        node=8         1          01:00:00
  large-sc-2     0     -        node=16 / 720 cluster total 1  02:00:00
  debug-qos      0     -        node=4         1          (debug partition default)
```

- **`normal` (our default)**: no explicit TRES cap. Partition cap = 12 h walltime; the QoS doesn't tighten this.
- **`large-sc-1`**: 8 nodes per user per job, 1 h. For "burst" jobs.
- **`large-sc-2`**: 16 nodes per user, 720 nodes cluster-wide, 2 h. Likely gated.
- **`debug-qos`**: 4 nodes per user, debug partition (1.5 h cap).

So **for our calibration + pilots**:
- Up to **4 nodes per job is friction-free** on `normal` (no extra QoS gymnastics, no allocation request).
- **8 nodes for 1 h** via `large-sc-1` if we wanted a burst with no escalation paperwork.
- **>8 nodes** is possible but we'd file an escalation; not needed for the pilots.

## 6. How big a job do we actually need

### 6.1 Apertus-8B throughput estimate on GH200

A reasonable working number for an 8B-class transformer with bf16,
FA-2, FSDP+optimizer-sharding, seq 4096 on a single GH200 GPU
is **~6,000 tokens/sec/GPU**. Apertus's actual training-recipe details
(AdEMAMix, 0.1 grad-clip, QK-Norm) shift this slightly but not in a
big way. Calibration's job is to nail this number for our specific
stack; until then:

| nodes | GPUs | total tok/s (est.) | tok/h        | 1 B tokens | 10 B tokens (pilot) | 30 B (3 arms) |
|------:|-----:|-------------------:|-------------:|----------:|---------------------:|--------------:|
|     1 |    4 |             24 k   |    86 M /h   |  ~11.6 h  |     ~5 days          |   ~14.5 days  |
|     2 |    8 |             48 k   |   173 M /h   |   ~5.8 h  |     ~2.4 days        |    ~7.2 days  |
|     4 |   16 |             96 k   |   346 M /h   |   ~2.9 h  |     ~1.2 days        |    ~3.6 days  |
|     8 |   32 |            192 k   |   691 M /h   |   ~1.4 h  |     ~14.5 h          |    ~1.8 days  |

Two facts shape sizing from this table:

- **`normal` partition's 12 h walltime cap.** Any pilot >12 h must be chained: checkpoint + resume after the 12 h timer fires. Chaining costs ~5-10 min per restart for context warm-up but doesn't lose tokens.
- **Wait time is the same regardless of size (probe section §3).** So *during the maintenance window* there's no reason to go small — larger is strictly faster end-to-end if a sized job fits in 12 h.

### 6.2 Recommended shape for the first calibration

At an estimated 24 k tok/s on a single GH200 node, 1 B tokens needs
~11.6 h of wall-clock — too tight for `--time=10:00:00`. Either
shrink the target or use the full 12 h partition cap. We use the full
cap below; if the actual throughput is lower than the estimate the
job will walltime out cleanly at the latest checkpoint, which is the
right failure mode for a calibration run.

| field | value | rationale |
|---|---|---|
| **arm** | Vanilla | zero init-method code needed; calibrates throughput + storage + scheduler |
| **partition** | `normal` | 12 h timer; `debug` is fine if a faster turnaround matters (1.5 h cap, ~1 h wait) |
| **account** | `a0140` | the only project we have |
| **nodes** | **1** | smallest unit, simplest to interpret |
| **tasks-per-node** | 1 (Slurm) | with `torchrun --nproc_per_node=4` inside |
| **GPUs** | 4× GH200 | implicit on Clariden |
| **time** | `--time=12:00:00` | full partition cap; at 24 k tok/s gives 1.04 B tokens — enough margin to hit a 1 B target |
| **token target** | **1 B tokens** | at 24 k tok/s → ~11.6 h, just under the 12 h cap; if throughput is higher we stop early at 1 B, if lower we walltime out at the last checkpoint |
| **seq length** | 4096 | matches Apertus pretraining; do NOT change here |
| **microbatch** | start at 4 per GPU | 4 × 4 GPUs × 4096 = 65k tokens / step; calibrate up to 8/GPU if memory allows |
| **global step size** | ~4 M tokens (with grad-accum) | matches Apertus's batch-size schedule |
| **checkpoint cadence** | every 250 steps OR every 1 B tokens (whichever first) | resilience to preemption + crash |
| **eval cadence** | every 100 steps on `virgin_hplt` + `glossapi_el_modern` (small slices) | early signal on training quality without burning wall |
| **workspace** | `/iopsstor/scratch/cscs/fffoivos/runs/vanilla_calibration_v1/` | per the CSCS storage convention (data on iopsstor, checkpoints on capstor) |
| **checkpoint dir** | `/capstor/scratch/cscs/fffoivos/runs/vanilla_calibration_v1/checkpoints/` | large sequential writes belong on capstor |
| **environment** | `uenv run pytorch/v2.6.0:v1 --view=default -- ...` | the verified-working stack from March 28 smoke |

### 6.3 After calibration → bakeoff shape, then production CPT

*v0.7 reframing*: what we earlier called "the 10 B-per-arm pilot" is
actually two distinct phases per v0.7 §5.4 + §9:

- **Bakeoff** (init-method discrimination only): 1.5–2 B tokens **per arm**, three arms = Vanilla / ReTok / **Centroid** (Distillation bracketed in v0.7 §13). Total 4.5–6 B tokens. Purpose: pick the winning init, not measure final quality.
- **Production CPT** (on the winning arm only): 10–20 B tokens per v0.7 §9 + Q A2 (with anneal in the final 10–20 %).

Sizing from p-skarvelis's measured throughput (107 k tok/s on 4 nodes,
seq=2048; expect ~½ of that at seq=4096 with FA-2 + grad-ckpt for an
Apertus-recipe-faithful Megatron run):

| phase | budget | nodes | walltime estimate |
|---|---|---|---|
| Bakeoff per arm | 2 B tokens | 4 | ~5.2 h at seq=2048, ~10 h at seq=4096 |
| All three arms in parallel | 6 B | 12 (3×4) | ~5.2 h at seq=2048, ~10 h at seq=4096 |
| Production CPT (winning arm) | 15 B | 4 (or 8) | ~6 days at 4 nodes / ~3 days at 8 nodes; chained via `afterok` through `normal`'s 12 h cap |

Each arm fits inside a single 12 h `normal` slot at the bakeoff size.
Production CPT requires chaining 2–4 sequential 12 h jobs via
`--dependency=afterok` plus dataloader-state preservation (v0.7 §2 +
V3 verification).

QoS `normal` doesn't gate concurrent jobs, so submitting the three
bakeoff arms in parallel (3 × 4 nodes = 12 nodes peak) is admissible.
Clariden has the slack outside reservation windows.

## 7. How to query this state yourself

The probes used here. Save them as one-liners.

```bash
# Daily auth health
ssh-keygen -L -f ~/.ssh/cscs-key-cert.pub | grep -E 'Valid|Key ID|Principals'
cscs-key list
ssh ela 'hostname; whoami'

# Cluster posture
ssh clariden 'sinfo -o "%P %t %D %C %G"'
ssh clariden 'squeue -u $USER'        # mine
ssh clariden 'squeue -h --format="%t %D" | sort | uniq -c'   # global queue depth

# Will my hypothetical job actually fit?
ssh clariden 'sbatch --account=a0140 --partition=normal --nodes=N --time=HH:MM:SS --test-only --wrap="echo"'
# Reads: "Job <id> to start at <when> using <X> processors on nodes <names> in partition <p>"

# Active reservations (what's currently blocking)
ssh clariden 'scontrol show res 2>&1 | head -40'

# Quotas + share
ssh clariden 'sacctmgr show user fffoivos -s'
ssh clariden 'sshare -u fffoivos'

# My job history (60 days back)
ssh clariden 'sacct -X -u fffoivos --starttime $(date -d "60 days ago" +%F) -o JobID,JobName,Partition,Account,AllocNodes,State,Elapsed,Start'
```

## 8. Open items before the first sbatch fires

| Item | Owner | Blocker? |
|---|---|---|
| Apertus-8B-2509 checkpoint mirrored onto `/iopsstor/scratch/cscs/fffoivos/apertus_8b_2509/` | I drive once you say go | yes |
| Composite tokenizer mirrored onto `/capstor/store/cscs/swissai/a0140/tokenizers/apertus_greek_extended_153600/` (or its 148,480 modern-only sibling for the comparison arms) | I drive | yes |
| Initial CPT corpus (Phase 0 HPLT-broad subset, fresh-only per [CURRICULUM_AND_INIT_CORPUS.md](../03_3_cscs_experiments_kickoff/CURRICULUM_AND_INIT_CORPUS.md)) staged on `/iopsstor/scratch/cscs/fffoivos/cpt_corpus_v1/` | needs the CPT corpus build to fire first. **GCloud access lost 2026-05-20** — the [runbook](../03_2_apertus_c3_dedup_audit/CPT_DATASET_BUILD_RUNBOOK.md) "GCP scratch VM" path no longer applies; run the same steps on a Clariden `xfer` allocation instead. | yes |
| Training harness picked: Swiss-AI Megatron / nanotron / HF+accelerate | you decide ([Review checkpoint D in ANALYSIS.md](../03_3_cscs_experiments_kickoff/ANALYSIS.md#7-review-checkpoints--what-still-needs-your-explicit-sign-off)) | yes |
| One eval harness wired (GreekMMLU public split as the gate) | I drive once auth + storage are staged | partially (can be parallel) |

When these are clear we write the sbatch and submit the calibration.
