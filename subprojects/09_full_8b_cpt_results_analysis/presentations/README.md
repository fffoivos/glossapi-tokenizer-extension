# presentations/ — the canonical reports

> **In one line:** five self-contained HTML reports, each paired with the exact `.data.json` payload it renders; together they are the visual authority for the completed 8B run.
> **Period:** 2026-08-12 → 2026-08-19. **Status:** completed.

## History

| Date | Report | Why it was built |
| --- | --- | --- |
| 2026-08-12 (`2a413042`) | `FULL8_RESULTS.html` | the complete standalone 8B trajectory — learning panels, retention panels, per-document endpoints, the 19 GreekMMLU points, and the run's completion receipt |
| 2026-08-12 (`2a413042`) | `FULL8_VS_0P5B.html` | the scale / data-order comparison against the five 0.5 B arms |
| 2026-08-12 (`2a413042`) | `CHECKPOINT_BEHAVIOR.html` | GreekMMLU answer drift and per-checkpoint source exposure |
| 2026-08-12 (`f4855dfa`) | `NATIVE_GREEK_3CP_BENCHMARKS.html` | the first native-Greek screen, reported both full and contamination-filtered |
| 2026-08-14 (`2a7eb9d8`) | `D0_0P5B_VS_FULL8_CHECKPOINTS.html` | token-aligned 0.5 B-versus-8 B checkpoint performance (payload lives one level up, next to its prose verdict) |
| 2026-08-19 (`ae2acd30`) | `FULL8_ALL_CHECKPOINT_NATIVE_BENCHMARKS_20260819.html` | the complete 19-checkpoint matrix on one strict subset — the report that superseded the three-point screen |

The first three were copied from the completed-run workspace and are pinned by SHA-256 in [`../evidence/ARTIFACT_MANIFEST.json`](../evidence/ARTIFACT_MANIFEST.json). Earlier progress reports, previous-run-anchored comparisons and synthetic drift demonstrations were deliberately left in subproject 07 as historical working material.

The 2026-08-19 report is the one to read. It mixes no raw scores into the filtered trajectory: its three anchor points (initialization, 9,536, terminal) come from the strict rescoring CSVs in [`../evidence/`](../evidence/) and the other sixteen from two completed Clariden matrices — `full8_native_greek_peak_window_20260817` (four points around the peak) and `full8_remaining12_checkpoint_release_20260817` (twelve points, 252/252 shards verified). Its stated finding is that the GreekMMLU peak is real but that other Greek capabilities peak at different times.

## Regenerating

```bash
python3 build_native_greek_3cp_benchmarks_report.py
python3 build_d0_0p5b_vs_full8_checkpoints.py
python3 build_full8_all_checkpoint_benchmark_report.py   # reads two matrices over ssh
```

The last builder pulls the sixteen remote points from Clariden, so it only reproduces where those matrices are reachable; the committed `.data.json` is the frozen record of that read.

## Superseded

`NATIVE_GREEK_3CP_BENCHMARKS.html` is subsumed by the 19-checkpoint report on the same strict subset, but it is kept because it is the only place that carries the full-versus-filtered side-by-side comparison.
