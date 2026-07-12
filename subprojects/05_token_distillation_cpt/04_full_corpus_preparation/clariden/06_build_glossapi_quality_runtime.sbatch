#!/usr/bin/env bash
#SBATCH --job-name=cpt4_build_rust_quality
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail

HERE=${PHASE04_CLARIDEN_DIR:?submit through clariden/submit.sh}
source "$HERE/paths.env"
source "$HERE/common.sh"
phase04_require_compute_partition "GlossAPI PyO3 quality-runtime build"
phase04_require_cpu_request
phase04_require_uenv_python
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"
phase04_require_file "$DATASET_QUALITY_SCRIPT"

[[ "$(git -C "$GLOSSAPI_QUALITY_ROOT" rev-parse --is-inside-work-tree 2>/dev/null || true)" == "true" ]] || {
    echo "ERROR: prepare the clean pinned GlossAPI checkout first: $GLOSSAPI_QUALITY_ROOT" >&2
    exit 120
}
[[ "$(git -C "$GLOSSAPI_QUALITY_ROOT" rev-parse HEAD)" == "$GLOSSAPI_QUALITY_COMMIT" ]] || {
    echo "ERROR: GlossAPI checkout is not at $GLOSSAPI_QUALITY_COMMIT" >&2
    exit 121
}
[[ -z "$(git -C "$GLOSSAPI_QUALITY_ROOT" status --porcelain --untracked-files=normal)" ]] || {
    echo "ERROR: GlossAPI quality checkout must be clean" >&2
    exit 122
}

mkdir -p "$(dirname "$GLOSSAPI_QUALITY_RUNTIME_ROOT")"
exec 9>"${GLOSSAPI_QUALITY_RUNTIME_ROOT}.lock"
flock -n 9 || { echo "ERROR: another quality-runtime build is active" >&2; exit 123; }

if [[ -s "$GLOSSAPI_RUST_QUALITY_BUILD_RECEIPT" && "${REBUILD_GLOSSAPI_QUALITY_RUNTIME:-0}" != "1" ]]; then
    export PYTHONPATH="$GLOSSAPI_QUALITY_MODULE_DIR${PYTHONPATH:+:$PYTHONPATH}"
    uenv run "$PHASE04_UENV" --view=default -- \
        "$RUNTIME_VENV/bin/python" "$DATASET_QUALITY_SCRIPT" validate-build-receipt \
        --receipt "$GLOSSAPI_RUST_QUALITY_BUILD_RECEIPT" \
        --expected-commit "$GLOSSAPI_QUALITY_COMMIT"
    echo "GLOSSAPI_QUALITY_RUNTIME_ALREADY_VALID=$GLOSSAPI_QUALITY_RUNTIME_ROOT"
    exit 0
fi

case "$GLOSSAPI_QUALITY_RUNTIME_ROOT" in
    "$DATA_ROOT"/runtime/glossapi-rust-quality-*) ;;
    *) echo "ERROR: refusing unsafe quality-runtime path: $GLOSSAPI_QUALITY_RUNTIME_ROOT" >&2; exit 124 ;;
esac
if [[ -e "$GLOSSAPI_QUALITY_RUNTIME_ROOT" ]]; then
    [[ "${REBUILD_GLOSSAPI_QUALITY_RUNTIME:-0}" == "1" ]] || {
        echo "ERROR: incomplete runtime exists; inspect it, then set REBUILD_GLOSSAPI_QUALITY_RUNTIME=1" >&2
        exit 125
    }
    rm -rf "$GLOSSAPI_QUALITY_RUNTIME_ROOT"
fi

SCRATCH_ROOT=${SLURM_TMPDIR:-$DATA_ROOT/tmp/glossapi-quality-build/$SLURM_JOB_ID}
PARTIAL="${GLOSSAPI_QUALITY_RUNTIME_ROOT}.partial-$SLURM_JOB_ID"
[[ ! -e "$PARTIAL" ]] || { echo "ERROR: stale build partial: $PARTIAL" >&2; exit 126; }
mkdir -p "$SCRATCH_ROOT" "$PARTIAL/modules" "$PARTIAL/wheels"
cleanup() { rm -rf "$PARTIAL" "$SCRATCH_ROOT"; }
trap cleanup EXIT

MATURIN_VERSION=${MATURIN_VERSION:-1.9.4}
TOOL_VENV="$SCRATCH_ROOT/maturin-venv"
uenv run "$PHASE04_UENV" --view=default -- bash -lc '
set -euo pipefail
python3 -m venv "$1"
"$1/bin/python" -m pip install --disable-pip-version-check "maturin==$2"
' bash "$TOOL_VENV" "$MATURIN_VERSION"

for crate in glossapi_rs_noise glossapi_rs_cleaner; do
    manifest="$GLOSSAPI_QUALITY_ROOT/rust/$crate/Cargo.toml"
    phase04_require_file "$manifest"
    phase04_require_file "$GLOSSAPI_QUALITY_ROOT/rust/$crate/Cargo.lock"
    CARGO_BUILD_JOBS="$SLURM_CPUS_PER_TASK" \
    CARGO_TARGET_DIR="$SCRATCH_ROOT/target-$crate" \
    uenv run "$PHASE04_UENV" --view=default -- \
        "$TOOL_VENV/bin/maturin" build \
        --release --locked --interpreter "$RUNTIME_VENV/bin/python" \
        --manifest-path "$manifest" --out "$PARTIAL/wheels"
done

mapfile -t wheels < <(find "$PARTIAL/wheels" -maxdepth 1 -type f -name '*.whl' | sort)
[[ ${#wheels[@]} -eq 2 ]] || { echo "ERROR: expected exactly two Rust wheels" >&2; exit 127; }
uenv run "$PHASE04_UENV" --view=default -- \
    "$RUNTIME_VENV/bin/python" -m pip install \
    --disable-pip-version-check --no-deps --target "$PARTIAL/modules" "${wheels[@]}"

# Build the receipt against staged modules but record their future paths.  The
# entire directory, including that receipt, is then published by one rename.
export PYTHONPATH="$PARTIAL/modules${PYTHONPATH:+:$PYTHONPATH}"
uenv run "$PHASE04_UENV" --view=default -- \
    "$RUNTIME_VENV/bin/python" "$DATASET_QUALITY_SCRIPT" build-receipt \
    --glossapi-root "$GLOSSAPI_QUALITY_ROOT" \
    --expected-commit "$GLOSSAPI_QUALITY_COMMIT" \
    --module-root "$PARTIAL/modules" \
    --published-module-root "$GLOSSAPI_QUALITY_MODULE_DIR" \
    --maturin-version "$MATURIN_VERSION" \
    --output "$PARTIAL/build_receipt.json"

mv "$PARTIAL" "$GLOSSAPI_QUALITY_RUNTIME_ROOT"
trap - EXIT
rm -rf "$SCRATCH_ROOT"
export PYTHONPATH="$GLOSSAPI_QUALITY_MODULE_DIR${PYTHONPATH:+:$PYTHONPATH}"
uenv run "$PHASE04_UENV" --view=default -- \
    "$RUNTIME_VENV/bin/python" "$DATASET_QUALITY_SCRIPT" validate-build-receipt \
    --receipt "$GLOSSAPI_RUST_QUALITY_BUILD_RECEIPT" \
    --expected-commit "$GLOSSAPI_QUALITY_COMMIT"
echo "GLOSSAPI_QUALITY_RUNTIME_COMPLETE=$GLOSSAPI_QUALITY_RUNTIME_ROOT"
