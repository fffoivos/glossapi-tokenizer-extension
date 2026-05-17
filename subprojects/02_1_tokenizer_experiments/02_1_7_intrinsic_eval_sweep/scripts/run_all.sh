#!/usr/bin/env bash
# Canonical end-to-end pipeline for 02_1_7. Run order matches the actual
# evidence chain that produced REPORT.md + CHOSEN_CUTOFF.md.
#
# Order:
#   01_build_variants_inline.py     — build 25 raw cutoff variants locally
#   01b_build_curated_variants.py   — build 6 curated twins (ablation refs)
#   01c_build_curated_backfilled.py — build the SHIP artifact at cutoff 17,408
#   02_prep_eval_configs.py         — Apertus-55 + Greek + tokenizer configs
#   03a_run_tokeval.sh              — TokEval suite on home (Apertus-55 + Greek)
#   (gcloud side) — see CHOSEN_CUTOFF.md § Build reproduction for the
#                   02_1_3 fertility harness invocation used on the
#                   in-house Greek held-outs; outputs land in
#                   artifacts/our_suite_raw_gcloud/.
#   07_morphscore_greek.py          — MorphScore on the active variant set
#   04_aggregate.py                 — TokEval JSON → results.parquet
#   08_merge_all.py                 — merge TokEval + 02_1_3 + MorphScore
#   09_render_final_report.py       — REPORT.md + main plots
#   11_extended_4metric_plot.py     — extended 0→25.6k 4-metric plot
#   12_knee_analysis_plot.py        — knee-analysis plot
#
# All steps are deterministic and rerunnable. The build steps create
# variants under variants/ (active) and look for archived variants under
# variants/_archive/ when resolving configs (see find_variant() in
# scripts/02_prep_eval_configs.py).

set -euo pipefail
SSP="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SSP"
VENV="$SSP/vendor/tokenizer-intrinsic-evals/.venv/bin"

echo "==== [1/9] build raw cutoff variants (25 cutoffs) ===="
$VENV/python scripts/01_build_variants_inline.py

echo "==== [2/9] build curated twins (ablation references) ===="
$VENV/python scripts/01b_build_curated_variants.py

echo "==== [3/9] build CANONICAL SHIP ARTIFACT (curated + backfilled at 17,408) ===="
$VENV/python scripts/01c_build_curated_backfilled.py

echo "==== [4/9] prep eval configs ===="
$VENV/python scripts/02_prep_eval_configs.py

echo "==== [5/9] run TokEval (Apertus-55 jobs) ===="
bash scripts/03a_run_tokeval.sh

echo "==== [6/9] MorphScore Greek on the active variant set ===="
$VENV/python scripts/07_morphscore_greek.py

echo "==== [7/9] aggregate TokEval raw → parquet ===="
$VENV/python scripts/04_aggregate.py

echo "==== [8/9] merge TokEval + 02_1_3 harness + MorphScore ===="
$VENV/python scripts/08_merge_all.py

echo "==== [9/9] render REPORT.md + plots ===="
$VENV/python scripts/09_render_final_report.py
$VENV/python scripts/11_extended_4metric_plot.py
$VENV/python scripts/12_knee_analysis_plot.py

echo
echo "==== DONE ===="
ls -la CHOSEN_CUTOFF.md REPORT.md manifests/per_cutoff_metrics.json 2>/dev/null
echo
echo "Canonical ship artifact:"
echo "  variants/c3_added_17408_curated_padded/tokenizer.json"
