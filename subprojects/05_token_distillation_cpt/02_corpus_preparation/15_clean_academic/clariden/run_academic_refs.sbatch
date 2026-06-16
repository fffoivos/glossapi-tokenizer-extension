#!/bin/bash
#SBATCH --job-name=academic_refs
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Academic reference/citation detect-and-emit at corpus scale (CPU-only, Clariden `normal`).
# Runs the Rust detector over every academic source in the SELECTED CPT parquet, in parallel,
# emitting auditable spans + per-signal counters + stratified inline-<match> review samples.
# NOTHING is dropped — the keep/drop policy is a separate, user-driven step over these outputs.
#
# Compute justification:
#  - Parallelism: one streaming columnar split pass (I/O-bound) → 4 source detectors in parallel,
#    each `reference_detect` rayon-parallel across ~16 cores (64/4). Hot path is 100% Rust.
#  - CPU saturation: 4 procs × rayon ≈ 64 cores; the split overlaps with the first detector start.
#  - Mem ceiling: streaming everywhere (zstd/JSONL line-by-line, parquet iter_batches) → << 400G;
#    the 400G is headroom for the pyarrow split pass over the 47 M-row selected parquet.
#  - Known gaps: section-`β` lever is unavailable here (selected parquet is doc-level); use the
#    raw section parquets + ../driver --mode sections for the Kallipos/Pergamos β path.
#
# PREREQUISITE: build the aarch64 binary on a LOGIN node first (compute nodes lack crates.io):
#   cd $REPO/.../15_clean_academic/reference_detector && cargo build --release
# (bootstrap rust once with: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y)
set -euo pipefail

SC=${SC:-/iopsstor/scratch/cscs/fffoivos}
REPO_ROOT=${REPO_ROOT:-$SC/repo/glossapi-tokenizer-extension}
HERE=$REPO_ROOT/subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic
BIN=${BIN:-$HERE/reference_detector/target/release/reference_detect}
PY=${PY:-$SC/python_envs/refdetect_py/bin/python}   # standalone pyarrow venv (py3.6 + pyarrow 6.0.1)
SELECTED=${SELECTED:-$SC/cpt_corpus/cpt/selected_after_apertus_and_internal_dedup.parquet}
OUT=${OUT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/academic_refs_$(date -u +%Y%m%dT%H%M%SZ)}
SOURCES=${SOURCES:-greek_phd openarchives kallipos pergamos}

echo "=== academic_refs: preflight ==="
for f in "$BIN" "$PY" "$SELECTED"; do
  [ -e "$f" ] || { echo "MISSING: $f" >&2; exit 1; }
done
mkdir -p "$OUT"/split
echo "BIN=$BIN"; echo "SELECTED=$SELECTED"; echo "OUT=$OUT"; echo "SOURCES=$SOURCES"

echo "=== step 1: split selected parquet → per-source jsonl (single streaming pass) ==="
"$PY" "$HERE/clariden/split_selected_by_source.py" --selected "$SELECTED" --out-dir "$OUT/split"

echo "=== step 2: run the Rust detector per source IN PARALLEL ==="
pids=()
for s in $SOURCES; do
  in="$OUT/split/$s.jsonl"
  [ -s "$in" ] || { echo "  (skip $s: empty $in)"; continue; }
  mkdir -p "$OUT/$s/refspans"
  # NOTE: --mode wholedoc is the OLD header→EOF detector (endmatter_bib). The eval decided to ship the
  # new line-LR span model (eval/MODEL_TRADEOFF.md) but it is NOT yet ported to Rust; this still emits
  # the known-worse arm. Do not treat this output as the new model until a `--mode spans` path lands.
  ( RAYON_NUM_THREADS=16 "$BIN" --mode wholedoc --source "$s" \
        --input "$in" --text-field text --id-field id \
        --out-spans "$OUT/$s/refspans/$s.spans.jsonl" \
        --out-counters "$OUT/$s/$s.counters.jsonl" \
      && "$PY" -c "import json,pyarrow as pa,pyarrow.parquet as pq;rs=[json.loads(l) for l in open('$OUT/$s/$s.counters.jsonl') if l.strip()];ks=sorted({k for r in rs for k in r});pq.write_table(pa.table({k:[r.get(k) for r in rs] for k in ks}),'$OUT/$s/$s.counters.parquet') if rs else None" ) &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -eq 0 ] || { echo "a detector failed" >&2; exit 1; }

echo "=== step 3: stratified inline-<match> review samples per source ==="
for s in $SOURCES; do
  cj="$OUT/$s/$s.counters.jsonl"; sp="$OUT/$s/refspans/$s.spans.jsonl"
  [ -s "$cj" ] || continue
  "$PY" "$HERE/review/sample_refspans.py" --source "$s" --mode wholedoc \
      --text-input "$OUT/split/$s.jsonl" --spans "$sp" --counters "$cj" \
      --metric footnote_citation_only --out-dir "$OUT/$s/review_samples" --total 150 || true
done

echo "=== done → $OUT ==="
for s in $SOURCES; do
  c="$OUT/$s/$s.counters.jsonl"
  [ -s "$c" ] && echo "  $s: $(wc -l < "$c") docs, spans=$(wc -l < "$OUT/$s/refspans/$s.spans.jsonl")"
done
