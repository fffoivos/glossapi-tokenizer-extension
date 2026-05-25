# trajectory_analysis_20260524/

Per-arm metric-vs-token trajectory analysis from the bakeoff arms. This
directory now contains both the original 2B analysis and the 3.5B continuation
analysis for Vanilla / ReTok / TD.

Main continuation doc:
[`CONTINUATION_3P5B_RESULTS_20260525.md`](CONTINUATION_3P5B_RESULTS_20260525.md).

Original 2B trajectory doc:
[`BAKEOFF_TRAJECTORY_ANALYSIS_20260524.md`](BAKEOFF_TRAJECTORY_ANALYSIS_20260524.md).

| File | What |
|---|---|
| `CONTINUATION_3P5B_RESULTS_20260525.md` | Final 3.5B continuation analysis: aggregate scoreboard, task winners, BPC, new-token diagnostics |
| `BAKEOFF_TRAJECTORY_ANALYSIS_20260524.md` | Full analysis: per-arm trajectories, three-window slope, TD-vs-Vanilla crossover projection |
| `plots/trajectories.png` | Three-panel group-averaged trajectory (EN-ret / Multi / Greek) per arm |
| `plots/trajectories_per_task.png` | Eight per-task panels (seven Greek tasks + English MMLU reference) |
| `plots/intrinsic_trajectories.png` | Tokenizer-fair BPC/NLL trajectory |
| `regenerate_plots.py` | Self-contained reproduction script (reads `per_iter_results/`, regenerates PNGs, prints slopes) |
| `summarize_3p5b_continuation.py` | Rebuilds the continuation markdown + JSON summary from local result snapshots |
| `per_iter_results/` | lm-eval `results.json` files, including continuation snapshots at iter 585/715/834 for Vanilla/ReTok/TD |

Reproduction:

```bash
python3 summarize_3p5b_continuation.py
python3 regenerate_plots.py
python3 plot_van_td.py
python3 plot_intrinsic_van_td.py
python3 plot_subcategories_van_td.py
```

Headline:

- At 3.5B, TD layer11 narrowly leads the Greek aggregate and clearly leads
  English-retention and multilingual aggregates.
- Vanilla remains best on tokenizer-fair heldout Greek BPC.
- ReTok improves fastest on BPC and wins Greek MMLU / INCLUDE-44 Greek, but it
  remains behind on the Greek aggregate.

See [`PRODUCTION_DECISION_STATE.md`](../../../../PRODUCTION_DECISION_STATE.md)
for the older production decision context and
[`bakeoff_1node_chain_20260522_005620_iter0000476_digest.md`](../live_summaries/bakeoff_1node_chain_20260522_005620_iter0000476_digest.md)
for the snapshot at iter 476.
