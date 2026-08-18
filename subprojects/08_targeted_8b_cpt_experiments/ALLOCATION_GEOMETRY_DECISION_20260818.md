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

## Result

| Scale | Fewer-node result | Selected candidate | Reason |
| --- | --- | --- | --- |
| 1.5B | 2 nodes is legal but requires at least 98.1% of ideal scaling after retaining the first-allocation qualification and reserve. | 4 nodes, TP=1, DP=16, microbatch=4, accumulation=16 | 2 nodes has only 649 seconds of conservative slack, so it cannot be promised to fit 12 hours. |
| 8B | 2 nodes exceeds 12 hours even at ideal scaling; 3 nodes cannot preserve the global batch. Four nodes would require at least 82.3% of ideal scaling including first-allocation qualification. | 8 nodes, TP=2, DP=16, microbatch=2, accumulation=32 | 8 nodes needs only 41.1% of ideal scaling against the conservative budget. |

The 1.5B observed one-node tail was about 25.12 seconds/update; the 8B
16-node tail was about 8.67 seconds/update. These are inputs to the sizing
calculation, not evidence that an unmeasured new profile is promoted. Each
selected profile must qualify inside its first actual allocation before it
continues canonical training.
