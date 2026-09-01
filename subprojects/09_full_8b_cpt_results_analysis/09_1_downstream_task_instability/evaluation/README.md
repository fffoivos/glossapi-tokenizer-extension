# Peak-window native-Greek evaluation

This directory evaluates the two saved checkpoints immediately before and the
two immediately after the clean-GreekMMLU optimum at update 9,536. Together
with the already completed update-9,536 evaluation, the analysis window is:

| Position | Update | Token slots | Evaluation action |
| --- | ---: | ---: | --- |
| best - 2 | 7,152 | 29,997,662,208 | score now |
| best - 1 | 8,344 | 34,997,272,576 | score now |
| best | 9,536 | 39,996,882,944 | reuse authoritative result |
| best + 1 | 10,728 | 44,996,493,312 | score now |
| best + 2 | 11,920 | 49,996,103,680 | score now |

The scientific benchmark and scorer are unchanged from the frozen 8B
three-checkpoint screen: DemosQA, Medical MCQA, ASEP MCQA, GPCR, and OYXOY
NLI/WSD/WiC/metaphor, using the authoritative FP32 legacy candidate scorer.
Protipa remains excluded because its manual access gate was not approved.

Only the four missing checkpoints are evaluated. Because these benchmarks were
not removed from the completed training corpus, only the clean subset defined
by the existing original-corpus contamination audit is scored. The frozen
source questions and the audit's 10,076 strong-match exclusions are rebound
into a 73,894-row clean-question file before model loading. The receipt records
retained and dropped identities per benchmark; full-panel scores are neither
computed nor reported as valid results.

The initial plan used six sequential four-node debug segments, each owning
fourteen of the 84 independent scorer shards. Live execution showed that a
22-minute four-node envelope was insufficient for two long shards. Completed
receipts were preserved; the missing shards and remaining segments used a
receipt-verified two-node profile with longer wall time inside the same Clariden
node-minute ceiling. Update 9,536 was not rerun.

## Files

- `peak_window_checkpoint_bindings.json` - exact four missing model exports and
  their conversion receipts.
- `bind_peak_window_native_suite.py` - validates those exports, then rebinds
  the frozen examples without changing benchmark or scoring fields.
- `run_peak_window_segment.sbatch` - receipt-bound segmented evaluator.
- `finalize_peak_window.py` - joins the four new results with the existing
  update-9,536 result.
- `coordinate_peak_window_segments.sh` - bounded Mac-side submission and
  monitoring loop for Clariden's two-submitted-debug-job limit.
- `PEAK_WINDOW_CLEAN_RESULTS_20260817.md` - final clean-subset scores,
  artifact hashes, interpretation and execution audit.

The runtime scorer remains in the already qualified immutable bundle. These
files are an experiment-owned adapter for a new checkpoint selection; they do
not replace or modify that scorer.
