# Deterministic/hybrid ablation runner

`deterministic_ablation_runner` evaluates the deterministic structure rules and
the three C0/rule hybrids on one explicitly selected development split. It is a
receipt-oriented LLM-silver comparison, not a training or production command.

The runner requires five explicit paths and never searches for data:

- imported `academic-structure-gold-v1` LLM-silver JSONL;
- its locked split manifest;
- an exact C0 `academic-structure-predictions-v1` JSONL for the selected split;
- the matching sequence-model config;
- a new output directory that must not already exist.

The default and intended split is `validation`. `train` is accepted only when
named explicitly. Test, sealed, historical-test, holdout, and similar aliases
are rejected in the requested split, silver rows, manifest assignments, and C0
rows before `GoldLine` or adapter prediction objects are materialised.

Run this on an approved CPU worker from the academic-cleaning `eval/` directory:

```bash
python3 -m sequence_models.deterministic_ablation_runner \
  --silver /explicit/path/struct2k.train-validation.jsonl \
  --split-manifest /explicit/path/struct2k.train-validation.split.json \
  --base-c0 /explicit/path/c0.validation.predictions.jsonl \
  --config sequence_models/joint_config.json \
  --allowed-split validation \
  --output-dir /new/receipt/path/deterministic-ablation
```

The immutable output directory contains:

- `rules-only.predictions.jsonl`;
- `base-plus-rules.predictions.jsonl`;
- `base-rules-veto.predictions.jsonl`;
- `base-plus-rules-veto.predictions.jsonl`;
- `ablation.report.json`.

The report binds every input and prediction output by SHA-256, records the exact
Git commit only when Git is available and the worktree was clean before output,
and includes C0 plus per-mode LLM-silver metrics. Independent running-prose
safety values are intentionally `null`. The report explicitly records that no
model fitting, discovery, corpus mutation, sealed access, human-gold use, or
production authorization occurred.

Rerunning into the same output directory is an error. Use a new receipt path for
every execution; do not edit or replace an emitted prediction or report.

On Clariden, use `clariden/run_deterministic_ablation.sbatch`. It is plan-only
unless `CONFIRM_DETERMINISTIC_ABLATION=1`, pins a clean exact commit, verifies
the physically test-stripped source and the joint-ladder artifact inventory,
hides all accelerator runtimes, stages into a job-unique directory, and
publishes with no replacement. The ten-minute CPU-only allocation still
reserves a complete four-GH200 Clariden node, so do not extend or submit it
casually. The exact environment contract is documented in the Agent 2 runbook.
