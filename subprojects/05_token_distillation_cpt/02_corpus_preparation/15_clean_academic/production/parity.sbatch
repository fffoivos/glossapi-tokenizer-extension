#!/usr/bin/env bash
#SBATCH --job-name=bibclean-v2-parity
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=160G
#SBATCH --time=01:00:00

set -euo pipefail
: "${RUN_BOOTSTRAP:?RUN_BOOTSTRAP is required}"
: "${COHORT:?COHORT is required}"
: "${REFERENCE_PROBABILITIES:?REFERENCE_PROBABILITIES is required}"
: "${PARITY_OUTPUT:?PARITY_OUTPUT is required}"
: "${TRAIN_COMMIT:?TRAIN_COMMIT is required}"
: "${GLOSSAPI_COMMIT:?GLOSSAPI_COMMIT is required}"
TRAIN_SRC="$RUN_BOOTSTRAP/train"
GLOSSAPI_SRC="$RUN_BOOTSTRAP/glossapi"
CRATE="$GLOSSAPI_SRC/rust/glossapi_rs_bib"
ARTIFACTS="$GLOSSAPI_SRC/src/glossapi/models/bibliography"
TARGET="$RUN_BOOTSTRAP/target-cli"
WHEELS="$RUN_BOOTSTRAP/wheels"
CANDIDATE="$RUN_BOOTSTRAP/evidence/current_line_prob.npy"
mkdir -p "$RUN_BOOTSTRAP/evidence" "$WHEELS"

export CRATE TARGET WHEELS
export CARGO_HOME="$RUN_BOOTSTRAP/cargo-home"
unset RUSTFLAGS CARGO_BUILD_TARGET RUSTC_WRAPPER || true
uenv run pytorch/v2.9.1:v2 --view=default -- bash -lc '
  set -euo pipefail
  cargo build --release --locked --manifest-path "$CRATE/Cargo.toml" \
    --target-dir "$TARGET" --no-default-features --features cli
  "$HOME/.venvs/hfupload/bin/python" -m maturin build --release \
    --manifest-path "$CRATE/Cargo.toml" --out "$WHEELS"
'
RAYON_NUM_THREADS=64 "$TARGET/release/bib_line_detect" detect \
  --artifacts "$ARTIFACTS" --input "$COHORT" --out "$CANDIDATE"
WHEEL=$(find "$WHEELS" -maxdepth 1 -name 'glossapi_rs_bib-*.whl' -print -quit)
test -n "$WHEEL"
test ! -e "$RUN_BOOTSTRAP/runtime"
"$HOME/.venvs/hfupload/bin/python" -m pip install --quiet --no-deps \
  --target "$RUN_BOOTSTRAP/runtime" "$WHEEL"

export PYTHONPATH="$TRAIN_SRC${PYTHONPATH:+:$PYTHONPATH}"
"$HOME/.venvs/hfupload/bin/python" -m \
  subprojects.05_token_distillation_cpt.02_corpus_preparation.15_clean_academic.production.parity \
  --reference "$REFERENCE_PROBABILITIES" --candidate "$CANDIDATE" \
  --cohort "$COHORT" --binary "$TARGET/release/bib_line_detect" \
  --artifact-manifest "$ARTIFACTS/manifest.json" \
  --train-commit "$TRAIN_COMMIT" --glossapi-commit "$GLOSSAPI_COMMIT" \
  --output "$PARITY_OUTPUT"
