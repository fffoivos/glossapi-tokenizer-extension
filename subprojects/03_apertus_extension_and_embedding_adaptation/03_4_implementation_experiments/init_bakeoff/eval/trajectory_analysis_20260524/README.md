# trajectory_analysis_20260524/

Per-arm metric-vs-token trajectory analysis from the four bakeoff arms (vanilla / retok / centroid / TD). Companion to the iter-476 digests; what the snapshots don't show.

Main doc: [`BAKEOFF_TRAJECTORY_ANALYSIS_20260524.md`](BAKEOFF_TRAJECTORY_ANALYSIS_20260524.md).

| File | What |
|---|---|
| `BAKEOFF_TRAJECTORY_ANALYSIS_20260524.md` | Full analysis: per-arm trajectories, three-window slope, TD-vs-Vanilla crossover projection |
| `trajectories.png` | Three-panel group-averaged trajectory (EN-ret / Multi / Greek) per arm |
| `trajectories_per_task.png` | Eight per-task panels (seven Greek tasks + English MMLU reference) |
| `regenerate_plots.py` | Self-contained reproduction script (reads `per_iter_results/`, regenerates PNGs, prints slopes) |
| `per_iter_results/` | 23 lm-eval `results.json` files: vanilla / retok / centroid at iter 130/260/325/390/455/476, TD at iter 130/260/390/455/476 |

Reproduction:

```bash
python3 regenerate_plots.py
```

Headline:

- Vanilla Greek-aggregate slope (mid-window) = **−17.9 m.p./B** (Vanilla is *losing* Greek performance during CPT).
- TD Greek-aggregate slope (mid-window) = **+8.2 m.p./B** (TD is *gaining*).
- Linear-extrapolation crossover: TD overtakes Vanilla on Greek aggregate at **~2.6-2.9 B tokens**, i.e., 0.6-0.9 B beyond the 2 B bakeoff budget.

This is a forward-looking signal, not a change to the in-flight Vanilla 15 B production launch decision. See [`PRODUCTION_DECISION_STATE.md`](../../../../PRODUCTION_DECISION_STATE.md) for the production decision and [`bakeoff_1node_chain_20260522_005620_iter0000476_digest.md`](../live_summaries/bakeoff_1node_chain_20260522_005620_iter0000476_digest.md) for the snapshot at iter 476.
