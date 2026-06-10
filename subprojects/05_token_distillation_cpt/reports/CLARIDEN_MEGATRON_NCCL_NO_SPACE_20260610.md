# Clariden Megatron Multi-Node NCCL/OFI `NO_SPACE` Report

Date: 2026-06-10

## Summary

Dataset preparation and artifact gates are complete. The original blocker was
multi-node Megatron training on Clariden: Megatron runs failed before the first
iteration with `NET/OFI Request ... Error: 16 (NO_SPACE)`, even though the AWS
Libfabric NCCL plugin loaded and pure PyTorch NCCL diagnostics succeeded under
the same `pytorch/v2.9.1:v2` uenv.

The resolved root cause is trainer-forced `NCCL_NET_FORCE_FLUSH=1`. With
`NCCL_NET_FORCE_FLUSH=0`, CXI/AWS Libfabric Megatron smokes now pass at 2 nodes,
4 nodes, and 16 nodes. Real-data 16-node timing estimates **8.3-8.5h allocated
runtime per arm** with the 4-segment chain. See
`CPT_16NODE_CXI_TIMING_20260610.md`.

## Environment

- System: Clariden GH200
- Account: `a0140`
- Uenv: `pytorch/v2.9.1:v2`
- PyTorch observed in smoke: `2.9.1`, CUDA `12.9`
- Launcher shape: one Slurm task per node, `torchrun --nproc_per_node=4`
- NCCL/libfabric settings:
  - `NCCL_NET="AWS Libfabric"`
  - `NCCL_NET_GDR_LEVEL=PHB`
  - `NCCL_CROSS_NIC=1`
  - `NCCL_PROTO=^LL128`
  - `FI_CXI_DEFAULT_CQ_SIZE=131072`
  - `FI_CXI_DEFAULT_TX_SIZE=16384`
  - `FI_CXI_DISABLE_HOST_REGISTER=1`
  - `FI_CXI_RX_MATCH_MODE=software`
  - `FI_MR_CACHE_MONITOR=userfaultfd`
  - `FI_CXI_RDZV_GET_MIN=0`
  - `FI_CXI_RDZV_THRESHOLD=0`
  - `FI_CXI_RDZV_EAGER_SIZE=0`
  - `MPICH_GPU_SUPPORT_ENABLED=0` in diagnostics

## Positive Controls

Pure PyTorch 2-node all-reduce succeeds:

- Job: `2513892`
- Nodes: `nid006910,nid006931`
- Output: `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/torchrun_allreduce_u291_20260610T123750Z`
- Result: `COMPLETED`, exit `0:0`, elapsed `00:00:23`
- Key line: `world_size=8 value=28.0 expected=28.0 torch=2.9.1 cuda=12.9`

This confirms the plugin is available and basic inter-node NCCL works.

Pure PyTorch Megatron-style subgroup all-reduces also succeed:

- Job: `2513910`
- Nodes: `nid007436,nid007440`
- Output: `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/torchrun_megatron_groups_u291_20260610T124332Z`
- Result: `COMPLETED`, exit `0:0`, elapsed `00:00:25`
- Key line: `world_size=8 tp_size=2 groups=[('tp', [0, 1]), ('tp', [2, 3]), ('tp', [4, 5]), ('tp', [6, 7]), ('dp_with_cp', [0, 2, 4, 6]), ('dp_with_cp', [1, 3, 5, 7])] iters=8 torch=2.9.1 cuda=12.9`

This confirms the relevant 2-node TP=2 / DP-with-CP-shaped process groups can
be created and exercised outside full Megatron training.

Larger pure PyTorch Megatron-style collectives also succeed:

- 40M bfloat16 all-reduce:
  - Job: `2514008`
  - Output: `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/torchrun_megatron_groups_40m_bf16_20260610T130856Z`
  - Result: `COMPLETED`, exit `0:0`, elapsed `00:00:25`
- 40M bfloat16 reduce-scatter:
  - Job: `2514023`
  - Output: `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/torchrun_megatron_groups_rsag_40m_bf16_20260610T131049Z`
  - Result: `COMPLETED`, exit `0:0`, elapsed `00:00:25`
- 40M bfloat16 all-gather:
  - Job: `2514024`
  - Output: `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/torchrun_megatron_groups_ag_40m_bf16_20260610T131138Z`
  - Result: `COMPLETED`, exit `0:0`, elapsed `00:00:27`

Exact-size pure PyTorch controls matching the NTP Megatron failure also succeed:

- The failing NTP Megatron collective reports
  `NumelIn=268435456, NumelOut=67108864`.
- 67,108,864 bfloat16 reduce-scatter:
  - Job: `2514102`
  - Output: `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/torchrun_megatron_groups_rs_67108864_bf16_20260610T133611Z`
  - Result: `COMPLETED`, exit `0:0`, elapsed `00:00:26`
- 67,108,864 bfloat16 all-gather:
  - Job: `2514107`
  - Output: `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/torchrun_megatron_groups_ag_67108864_bf16_20260610T133703Z`
  - Result: `COMPLETED`, exit `0:0`, elapsed `00:00:25`

Single-node real-data Megatron smoke succeeds:

- Job: `2514129`
- Output: `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke1_vanilla_realdata_20260610T134150Z`
- Shape: 1 node / 4 GPUs, `LAUNCH_MODE=slurm`
- Data: real ordered base-tokenized CPT dataset, not mock data
- Objective/config: Goldfish, AdEMAMix, default distributed optimizer + overlap
- Result: `COMPLETED`, exit `0:0`, elapsed `00:05:49`
- Iteration lines:
  - iter 1: `lm loss: 1.476209E+00`, grad norm `4.252`
  - iter 2: `lm loss: 1.472929E+00`, grad norm `3.955`

## Failing Repros

1. Full-scale vanilla runtime smoke:
   - Job: `2513687`
   - Shape: 16 nodes / 64 GPUs
   - Output: `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke16_vanilla_20260610T115834Z`
   - Result: `FAILED`, exit `15:0`, elapsed `00:01:52`

2. 2-node Megatron debug smoke:
   - Job: `2513773`
   - Nodes: `nid006130,nid006148`
   - Output: `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke2_vanilla_nccldebug_20260610T121427Z`
   - Result: `FAILED`, exit `1:0`, elapsed `00:02:23`

3. 2-node with `NCCL_NCHANNELS_PER_NET_PEER=4`:
   - Job: `2513801`
   - Nodes: `nid006130,nid006139`
   - Output: `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke2_vanilla_ncclpeer4_20260610T122028Z`
   - Result: `FAILED`, exit `1:0`, elapsed `00:02:22`

4. 2-node with distributed optimizer and overlap disabled:
   - Job: `2513824`
   - Nodes: `nid006273,nid006302`
   - Output: `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke2_vanilla_no_distopt_20260610T122645Z`
   - Result: `FAILED`, exit `15:0`, elapsed `00:03:43`
   - Trainer confirmed `use_distributed_optimizer=0 use_comm_overlap=0`

5. `pytorch/v2.6.0:v1` comparison:
   - Job: `2513857`
   - Output: `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke2_vanilla_uenv260_20260610T123131Z`
   - Inconclusive: fails before NCCL with `ImportError: cannot import name 'SerializationFormat'`

6. 2-node with correct CPU affinity and distributed optimizer/overlap disabled:
   - Job: `2513920`
   - Nodes: `nid006033,nid006045`
   - Output: `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke2_vanilla_nodopt_cpu288_20260610T124611Z`
   - Result: `FAILED`, exit `1:0`, elapsed `00:03:51`
   - Slurm shape: `--cpus-per-task=288`, one `torchrun` task per node
   - Trainer confirmed `use_distributed_optimizer=0 use_comm_overlap=0`
   - CPU affinity was correct for all local workers before the failure

7. 2-node mock-data, production communication:
   - Job: `2513980`
   - Output: `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke2_vanilla_mockdata_prodcomm_20260610T130129Z`
   - Result: `FAILED`, exit `1:0`, elapsed `00:02:23`
   - `mock_data=True`, `extra_valid_data_path=None`
   - DDP config: `use_distributed_optimizer=True`, `overlap_grad_reduce=True`, `overlap_param_gather=True`, `bucket_size=40000000`

8. 2-node mock-data, production communication, peer-channel knob:
   - Job: `2513992`
   - Output: `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke2_vanilla_mockdata_prodcomm_peer4_20260610T130506Z`
   - Result: `FAILED`, exit `1:0`, elapsed `00:02:31`
   - `NCCL_NCHANNELS_PER_NET_PEER=4`

9. 2-node mock-data, NTP instead of Goldfish:
   - Job: `2514035`
   - Output: `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke2_vanilla_mockdata_ntp_20260610T131312Z`
   - Result: `FAILED`, exit `1:0`, elapsed `00:02:27`
   - First failing collective: `OpType=COALESCED, NumelIn=268435456, NumelOut=67108864`

10. 2-node mock-data, smaller DDP bucket:
    - Job: `2514047`
    - Output: `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke2_vanilla_mockdata_bucket5m_20260610T131655Z`
    - Result: `FAILED`, exit `1:0`, elapsed `00:02:29`
    - DDP config: `bucket_size=5000000`

11. 2-node mock-data, distributed optimizer kept, communication overlap disabled:
    - Job: `2514051`
    - Output: `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke2_vanilla_mockdata_distopt_nooverlap_20260610T132009Z`
    - Result: failed in log, then cancelled during watchdog cleanup
    - First failing collective: `OpType=COALESCED, NumelIn=4026810368, NumelOut=1006702592`

12. 2-node mock-data, direct Slurm rank launch instead of torchrun:
    - Job: `2514116`
    - Output: `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke2_vanilla_mockdata_slurm_20260610T133829Z`
    - Shape: 2 nodes, `--ntasks-per-node=4`, `--cpus-per-task=72`, `LAUNCH_MODE=slurm`
    - Result: `FAILED`, exit `15:0`, elapsed `00:02:37`
    - Same `DATA_PARALLEL_GROUP_WITH_CP` / `NET/OFI ... NO_SPACE` failure

## Failure Phase

The clearest phase evidence is from job `2513824`:

- TP/PP initialization succeeds: TP=2, PP=1.
- Model, optimizer, and LR scheduler build succeeds.
- Init checkpoint loads successfully at iteration 0.
- Training and extra validation datasets build successfully.
- Log reaches `training ...` and `[before the start of training step]`.
- First training step fails in `DATA_PARALLEL_GROUP_WITH_CP`.
- Mock-data repros confirm the same phase without real train/validation data.

Representative error:

```text
NET/OFI Request ... completed with error. RC: 5. Error: 16 (NO_SPACE).
[PG ID ... (DATA_PARALLEL_GROUP_WITH_CP) Rank ...] Process group watchdog thread terminated
NCCL error: unhandled system error, NCCL version 2.29.2
```

## 2026-06-10 Follow-Up CXI/OFI Diagnostics

Additional diagnostics after review-agent feedback:

13. 2-node mock-data with larger CXI request buffers:
    - Job: `2514355`
    - Output: `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke2_vanilla_mockdata_cxi_reqbuf_20260610T140452Z`
    - Env: `FI_CXI_REQ_BUF_SIZE=33554432`,
      `FI_CXI_REQ_BUF_MIN_POSTED=8`,
      `FI_CXI_DEFAULT_RX_SIZE=16384`,
      `FI_LOG_LEVEL=warn`, `FI_LOG_PROV=cxi`
    - Result: `FAILED`, exit `1:0`, elapsed `00:02:34`
    - Interpretation: 32 MiB request buffers are not sufficient.

14. 2-node mock-data with `FI_CXI_RX_MATCH_MODE=hybrid` and RDZV vars unset:
    - Job: `2514376`
    - Output: `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke2_vanilla_mockdata_cxi_hybrid_default_rdzv_20260610T140951Z`
    - Result: `FAILED`, exit `1:0`, elapsed `00:02:22`
    - Effective printed env had
      `FI_CXI_RX_MATCH_MODE=hybrid` and
      `FI_CXI_RDZV_{GET_MIN,THRESHOLD,EAGER_SIZE}=<unset>`
    - Still failed with `NET/OFI ... Error: 16 (NO_SPACE)`
    - Failing collective:
      `OpType=COALESCED, NumelIn=268435456, NumelOut=67108864`
    - Representative request:
      `Request: { dev: 2, size: 4, state: CREATED, direction: RECV }`

15. 2-node mock-data forcing AWS OFI NCCL RDMA:
    - Job: `2514396`
    - Output: `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke2_vanilla_mockdata_ofi_rdma_20260610T141443Z`
    - Env included `OFI_NCCL_PROTOCOL=RDMA`,
      `FI_CXI_RX_MATCH_MODE=hybrid`, `FI_CXI_RDZV_PROTO=alt_read`
    - Result: `FAILED`, exit `15:0`, elapsed `00:02:04`
    - NCCL confirmed `Using transport protocol RDMA (user set)` and
      `NET/AWS Libfabric/.../GDRDMA`
    - RDMA failed earlier with
      `fi_writedata failed; RC: -38, Error: Function not implemented`
    - Interpretation: forced RDMA is not a viable workaround in this uenv.

Important correction:

- `bakeoff_train.sbatch` sets `FI_CXI_RX_MATCH_MODE=software` before invoking
  `uenv run`, but `uenv run pytorch/v2.9.1:v2 --view=default` overrides the
  effective process environment back to:
  `FI_CXI_RX_MATCH_MODE=hybrid`,
  `FI_CXI_RDZV_{GET_MIN,THRESHOLD,EAGER_SIZE}=0`, and
  `FI_CXI_RDZV_PROTO=alt_read`.
- Therefore the original failure should be treated as an effective
  uenv-hybrid/SENDRECV failure, not a pure software-match-mode failure.
- The installed libfabric strings explicitly identify two relevant CXI
  mitigations for resource exhaustion:
  increase `FI_CXI_REQ_BUF_SIZE` for "request list full" and increase
  `FI_CXI_OFLOW_BUF_SIZE` for "overflow no match".
- Next unrun diagnostic, once CSCS auth is restored:
  large request buffers plus overflow buffers under the default SENDRECV path.

## Current Interpretation

This is not caused by:

- dataset build or tokenization;
- init checkpoint loading;
- missing AWS Libfabric plugin;
- generic two-node NCCL all-reduce;
- creation/use of Megatron-shaped TP and DP-with-CP process groups in pure
  PyTorch;
- 40M bfloat16 all-reduce, reduce-scatter, or all-gather on those groups;
- exact-size 67,108,864 bfloat16 reduce-scatter/all-gather matching the NTP
  failing collective shape;
- Megatron distributed optimizer alone;
- Megatron overlap flags alone.
- missing CPU affinity for local workers.
- real indexed dataset/dataloader or extra validation;
- Goldfish loss.
- torchrun specifically, since direct Slurm rank launch also fails.

The current recipe is healthy on a single node with the real ordered dataset.

The likely scope is a Megatron/SuisseAI training-step communication pattern in
the data-parallel-with-context-parallel process group on the
`pytorch/v2.9.1:v2` uenv's AWS-OFI-NCCL/CXI SENDRECV path.

## Recommended Next Steps

## 2026-06-10 Socket/HSN Fallback Update

Additional post-auth diagnostics changed the operational recommendation.

Further AWS Libfabric/CXI tests remained negative:

16. 2-node mock-data with environment re-applied inside `uenv run`:
    - Job: `2514693`
    - Output: `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke2_vanilla_mockdata_uenv_inside_software_20260610T150431Z`
    - Result: `FAILED` with `NET/OFI ... NO_SPACE`

17. 2-node mock-data with software match mode plus request-buffer sizing:
    - Job: `2514729`
    - Env included `FI_CXI_REQ_BUF_SIZE=33554432`,
      `FI_CXI_REQ_BUF_MIN_POSTED=8`, `FI_CXI_DEFAULT_RX_SIZE=16384`
    - Result: `FAILED` with `NET/OFI ... NO_SPACE`

18. 2-node mock-data with larger posted RX/request queue:
    - Job: `2514751`
    - Env included `FI_CXI_REQ_BUF_SIZE=16777216`,
      `FI_CXI_REQ_BUF_MIN_POSTED=32`, `FI_CXI_DEFAULT_RX_SIZE=65536`
    - Result: `FAILED` with `NET/OFI ... NO_SPACE`

19. 2-node mock-data with `NCCL_NCHANNELS_PER_NET_PEER=1`:
    - Job: `2514764`
    - Result: `FAILED` with `NET/OFI ... NO_SPACE`

20. Direct Slurm rank launch after inside-uenv env reapplication:
    - Job: `2514776`
    - Result: `FAILED` with `NET/OFI ... NO_SPACE`

21. No-distributed-optimizer/no-overlap retry:
    - Job: `2514815`
    - Result: `FAILED` with `NET/OFI ... NO_SPACE`

22. Review-agent `NCCL_PROTO=Simple` / channel-limited probe:
    - Job: `2514875`
    - Result: `FAILED`, exit `1:0`, elapsed `00:01:58`
    - Key evidence: failure occurred on an initial tiny default-process-group
      all-reduce, with
      `NET/OFI Request ... Error: 16 (NO_SPACE). ... size: 4 ... RECV`.

One attempted model-parallel workaround is not directly usable:

- Job `2514784` with `TENSOR_MODEL_PARALLEL_SIZE=4` failed at checkpoint load
  because the available init checkpoints are TP=2 shards. TP4 would require an
  explicit checkpoint conversion/reshard, not a runtime-only override.

Socket over HSN is now validated for this Megatron job shape:

- Job `2514830`: 2-node mock-data smoke with `NCCL_NET=Socket`,
  `NCCL_SOCKET_IFNAME=hsn`, `EXIT_INTERVAL=1`; `COMPLETED`, iteration 1 elapsed
  `82322.3 ms`, `tokens/sec/gpu: 6368.7`.
- Job `2514842`: 2-node real-data smoke with `NCCL_NET=Socket`,
  `NCCL_SOCKET_IFNAME=hsn`; `COMPLETED`, iteration 1 elapsed `81933.7 ms`,
  `lm loss: 1.476208E+00`.
- Job `2514854`: 4-node mock-data smoke with `NCCL_NET=Socket`,
  `NCCL_SOCKET_IFNAME=hsn`; `COMPLETED`, iteration 1 elapsed `50116.3 ms`,
  `tokens/sec/gpu: 5230.7`.
- Job `2514876`: 16-node mock-data smoke with `NCCL_NET=Socket`,
  `NCCL_SOCKET_IFNAME=hsn`; `COMPLETED`, iteration 1 elapsed `30046.7 ms`,
  `tokens/sec/gpu: 2181.1`, `throughput per GPU: 112.4 TFLOP/s/GPU`.

Current operational recommendation:

## 2026-06-10 Resolution Candidate: `NCCL_NET_FORCE_FLUSH=1`

The later deep dive in `CXI_NOSPACE_DEEP_DIVE_20260610.md` found a much better
explanation and successful validations:

- Failing jobs had the trainer-forced `NCCL_NET_FORCE_FLUSH=1`.
- Pure PyTorch probes did not set it, and Socket never enters the OFI flush
  path.
- The failing request shape is exactly a tiny tracked control receive:
  `size:4`, `direction:RECV`.
- Job `2515069` repeated the 2-node CXI Megatron smoke with
  `NCCL_NET_FORCE_FLUSH=0` and completed iteration 1.
- Job `2515691` repeated the no-flush Megatron smoke at 4 nodes / 16 GPUs and
  completed iteration 1:
  - output `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke4_vanilla_mockdata_cxi_noflush_20260610T185137Z`;
  - state `COMPLETED`, exit `0:0`, elapsed `00:01:46`;
  - runtime audit printed `NCCL_NET=AWS Libfabric` and
    `NCCL_NET_FORCE_FLUSH=0`;
  - iteration 1 elapsed `40043.2 ms`, `tokens/sec/gpu: 6546.5`;
  - no `NET/OFI ... NO_SPACE` failure.
- Job `2515665` repeated the no-flush Megatron smoke at 16 nodes / 64 GPUs and
  completed iteration 1:
  - output `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke16_vanilla_mockdata_cxi_noflush_20260610T183943Z`;
  - state `COMPLETED`, exit `0:0`, elapsed `00:01:48`;
  - runtime audit printed `WORLD_SIZE=64`, `NCCL_NET=AWS Libfabric`, and
    `NCCL_NET_FORCE_FLUSH=0`;
  - iteration 1 elapsed `15623.3 ms`, no `NET/OFI ... NO_SPACE` failure.

Action taken:

- `bakeoff_train.sbatch` now defaults `NCCL_NET_FORCE_FLUSH=0` and prints it in
  the runtime audit line.
- `gate_cpt2arm_artifacts.sh` now checks that the trainer has force-flush
  disabled.

Current operational recommendation:

1. Treat `NCCL_NET_FORCE_FLUSH=1` as the confirmed root cause of the earlier
   CXI `NO_SPACE` failure.
2. Launch production on AWS Libfabric/CXI with `NCCL_NET_FORCE_FLUSH=0`.
3. Keep Socket/HSN only as a functional but slower fallback.
4. Use the measured 16-node chain shape: `EXIT_INTERVAL=952`, `N_SEGMENTS=4`.
