# Bakeoff Training Summary

- run tag: `bakeoff_1node_chain_20260522_005620`
- generated at: `2026-05-22T19:13:55Z`
- parsed iteration lines: `1428` raw, `1428` deduplicated

| arm | points | latest iter | tokens B | lm loss | tok/s/gpu | skipped | nan | missing iters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vanilla | 476 | 476 | 1.996 | 1.7371 | 7983 | 0 | 0 | 0 |
| retok | 476 | 476 | 1.996 | 2.7040 | 7922 | 0 | 0 | 0 |
| centroid | 476 | 476 | 1.996 | 3.7875 | 7874 | 0 | 0 | 0 |

Source logs: final bakeoff launch, resume, and resume2 Slurm outputs on `/capstor/scratch/cscs/fffoivos/runs/bakeoff`. Rows are deduplicated by `(arm, iteration)`.
