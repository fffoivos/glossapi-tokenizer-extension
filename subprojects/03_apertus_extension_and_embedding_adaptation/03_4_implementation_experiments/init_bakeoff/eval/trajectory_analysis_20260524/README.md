# trajectory_analysis_20260524/

Per-arm metric-vs-token trajectory analysis from the 4-arm bakeoff. Three
levels of result: the original 2B trajectory analysis, the 3.5B continuation
(Vanilla / ReTok / TD), and the 5B continuation (Vanilla / TD only).

Loss-reading rule: raw Megatron `lm loss` is dense but tokenizer-dependent.
Use it for within-arm health only. Cross-tokenizer comparisons should use
heldout BPB from intrinsic evals; newer training logs may additionally
provide dense `bpb`, `bpt`, `base_loss`, `new_loss`, and `n_new` fields.
Older metric JSONs may call BPB `BPC` / `bpc_bits_per_byte`.

**Canonical 4-arm result**:
[`BAKEOFF_FINAL_RESULTS_20260526.md`](BAKEOFF_FINAL_RESULTS_20260526.md) —
consolidated 4-arm bakeoff results at each arm's final iter (Vanilla/TD to
1192/5.0B; ReTok to 834/3.5B; Centroid to 476/2.0B).

3.5B continuation (iso-token Vanilla/ReTok/TD):
[`CONTINUATION_3P5B_RESULTS_20260525.md`](CONTINUATION_3P5B_RESULTS_20260525.md).

Original 2B trajectory doc:
[`BAKEOFF_TRAJECTORY_ANALYSIS_20260524.md`](BAKEOFF_TRAJECTORY_ANALYSIS_20260524.md).

| File | What |
|---|---|
| `BAKEOFF_FINAL_RESULTS_20260526.md` | **Canonical 4-arm result.** Aggregate scoreboard at iter 1192 (5.0B), iso-token comparisons at iter 834 and iter 476, per-task winners, BPC trajectory across all arms, TD new-token diagnostics, production-decision implications |
| `CONTINUATION_3P5B_RESULTS_20260525.md` | 3.5B continuation analysis: iso-token Vanilla/ReTok/TD at iter 834 |
| `BAKEOFF_TRAJECTORY_ANALYSIS_20260524.md` | Original 2B trajectory analysis: three-window slope, TD-vs-Vanilla crossover projection |
| `plots/trajectories.png` | Three-panel group-averaged trajectory (EN-ret / Multi / Greek) per arm |
| `plots/trajectories_per_task.png` | Eight per-task panels (seven Greek tasks + English MMLU reference) |
| `plots/intrinsic_trajectories.png` | Tokenizer-fair BPB/NLL trajectory |
| `regenerate_plots.py` | Self-contained reproduction script (reads `per_iter_results/`, regenerates PNGs, prints slopes) |
| `summarize_3p5b_continuation.py` | Rebuilds the continuation markdown + JSON summary from local result snapshots |
| `plot_training_loss.py` | Parses dense Megatron logs; raw LM plots are diagnostic-only, and dense BPB/base-new plots are emitted when patched fields exist |
| `plot_loss_comparison.py` | Fair-vs-unfair loss comparison; prefers measured dense BPB when present, otherwise marks the proxy as approximate |
| `per_iter_results/` | lm-eval `results.json` files, including continuation snapshots at iter 585/715/834 for Vanilla/ReTok/TD |

Reproduction (all plots include 5B data at iters 1013 and 1192 for Vanilla/TD):

```bash
python3 summarize_3p5b_continuation.py
python3 regenerate_plots.py
python3 plot_van_td.py
python3 plot_intrinsic_van_td.py
python3 plot_subcategories_van_td.py
python3 plot_training_loss.py
python3 plot_loss_comparison.py
```

Headline at 5B (Vanilla + TD only; iso-token comparisons at 3.5B / 2.0B in the canonical doc):

- TD layer11 leads all three downstream aggregates (Greek, English retention,
  multilingual) at iter 1192.
- Vanilla remains best on tokenizer-fair heldout BPC; the gap has narrowed
  to 0.027 (from 0.110 at iter 130) and TD's BPC slope is steeper.
- TD's Greek-aggregate lead widened from tied at 3.5B to +0.69 pp at 5B,
  driven primarily by xquad_el (+7.57 pp).
- ReTok was stopped at 3.5B (TD-dominated); Centroid at 2B (clearly broken).

See [`PRODUCTION_DECISION_STATE.md`](../../../../_archive/synthesis_sources_20260526/PRODUCTION_DECISION_STATE.md)
for the older production decision context and
[`bakeoff_1node_chain_20260522_005620_iter0000476_digest.md`](../live_summaries/bakeoff_1node_chain_20260522_005620_iter0000476_digest.md)
for the snapshot at iter 476.
