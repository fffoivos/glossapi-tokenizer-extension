# Minimum defensible 12-hour geometry

This decision changes runtime geometry only. It preserves the model,
initialization, dataset identities and order, masks, 1,024-sequence global
batch, precision, optimizer, and LR schedule.

## Inputs

The remaining work is 1,790 updates for 1.5B (S2--S5) and 476 updates for 8B
(S4--S5). The conservative 12-hour limit is 43,200 seconds, including a
1,200-second checkpoint/exit reserve. The frozen timing receipts are
`receipts/production_timing_1p5b_v91r2.json` and
`receipts/production_timing_8b_v91r2.json` in the stage root.

## Exact-batch feasibility

For 1.5B (TP=1, microbatch=4), the accumulation count is
`1024 / (4 * nodes * 4)`; therefore 2 and 4 nodes are legal, while 3 is not.
For 8B (TP=2, microbatch=2), it is `1024 / (4 * nodes / 2 * 2)`; therefore
4 and 8 nodes are legal, while 3 is not.

## Corrected minimum-profile decision

The earlier version of this note incorrectly carried the 1.5B one-node
qualification wall time into the two-node candidate estimate.  That is not a
valid calculation: the qualification runs on the candidate profile and its
actual elapsed time must be charged to that same allocation.

Let `W` be the candidate's measured 256-update production-cadence wall time,
and `Q` the measured elapsed time for its restart checks plus that throughput
window.  The 12-hour gate is deliberately evaluated only after `Q` exists:

```
Q + 1.15 * remaining_256_update_blocks * W + 1,200 seconds <= 43,200 seconds
```

| Scale | Smallest exact-batch geometry | Remaining blocks | Pre-measurement screen | Decision |
| --- | --- | ---: | --- | --- |
| 1.5B | 2 nodes, TP=1, DP=8, microbatch=4, accumulation=32 | 8 | At the ideal extrapolation from the frozen one-node 256-update wall (`W=3,692s`), the theoretical screen requires about 89.7% scaling efficiency. | Candidate; continue only if the live `Q`/`W` budget gate passes. |
| 8B | 4 nodes, TP=2, DP=8, microbatch=2, accumulation=64 | 2 | At the ideal extrapolation from the measured 16-node wall (`W=10,472s`), the theoretical screen requires about 82.3% scaling efficiency. | Candidate; continue only if the live `Q`/`W` budget gate passes. |

For 8B, two nodes cannot fit even under ideal scaling and three nodes cannot
preserve the 1,024-sequence global batch.  A four-node candidate therefore is
the smallest possible 8B profile.  The candidates are not promoted from these
extrapolations: each first allocation measures `Q` and `W`, rejects itself if
the full remaining run cannot fit, and only then enters canonical training.
