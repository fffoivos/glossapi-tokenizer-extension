#!/usr/bin/env bash
# Build and publish the detector from a clean exact Clariden checkout.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/paths.env"
source "$HERE/common.sh"
phase04_require_compute_partition "detector build"
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"
phase04_require_uenv_python
[[ "$(uname -m)" == "aarch64" ]] || { echo "ERROR: detector build host must be aarch64" >&2; exit 88; }

for variable in CARGO_BUILD_TARGET RUSTFLAGS CARGO_ENCODED_RUSTFLAGS RUSTC_WRAPPER RUSTC_WORKSPACE_WRAPPER; do
    [[ -z "${!variable:-}" ]] || {
        echo "ERROR: unset $variable; Phase-04 detector builds do not accept inherited compiler overrides" >&2
        exit 89
    }
done

if ! command -v cargo >/dev/null; then
    echo "Installing the user-scoped Rust toolchain with rustup."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
    source "$HOME/.cargo/env"
fi

MANIFEST="$ACADEMIC_DIR/reference_detector/Cargo.toml"
CARGO_BIN=$(command -v cargo)
RUSTC_BIN=$(command -v rustc)
CODE_COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD)
BUILD_DIR=${BUILD_DIR:-$RUN_ROOT/detector_builds/${CODE_COMMIT}-${SLURM_JOB_ID}}
PARTIAL="${BUILD_DIR}.partial-${SLURM_JOB_ID}"
BUILD_RECEIPT=${BUILD_RECEIPT:-$BUILD_DIR/build_receipt.json}
[[ "$BUILD_RECEIPT" == "$BUILD_DIR/"* ]] || {
    echo "ERROR: BUILD_RECEIPT must live under the immutable BUILD_DIR" >&2
    exit 90
}
[[ ! -e "$BUILD_DIR" && ! -e "$PARTIAL" ]] || {
    echo "ERROR: refusing to overwrite detector build path: $BUILD_DIR or $PARTIAL" >&2
    exit 91
}
mkdir -p "$(dirname "$BUILD_DIR")" "$PARTIAL"
trap 'rm -rf "$PARTIAL"' EXIT

export CARGO_TARGET_DIR="$PARTIAL/cargo-target"
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-${SLURM_CPUS_PER_TASK:-32}}
cargo test --locked --manifest-path "$MANIFEST"
cargo build --locked --release --manifest-path "$MANIFEST"
phase04_require_file "$CARGO_TARGET_DIR/release/reference_detect"
install -m 0555 "$CARGO_TARGET_DIR/release/reference_detect" "$PARTIAL/reference_detect"
rm -rf "$CARGO_TARGET_DIR"
PUBLISHED_BINARY="$BUILD_DIR/reference_detect"
RECEIPT_RELATIVE=${BUILD_RECEIPT#"$BUILD_DIR/"}
PARTIAL_RECEIPT="$PARTIAL/$RECEIPT_RELATIVE"
mkdir -p "$(dirname "$PARTIAL_RECEIPT")"
uenv run "$PHASE04_UENV" --view=default -- \
    "$RUNTIME_VENV/bin/python" "$PHASE04_DIR/scripts/write_detector_build_receipt.py" \
    --repo "$REPO_ROOT" \
    --binary "$PARTIAL/reference_detect" \
    --published-binary-path "$PUBLISHED_BINARY" \
    --cargo-lock "$ACADEMIC_DIR/reference_detector/Cargo.lock" \
    --cargo-toml "$MANIFEST" \
    --cargo-bin "$CARGO_BIN" \
    --rustc-bin "$RUSTC_BIN" \
    --code-commit "$CODE_COMMIT" \
    --output "$PARTIAL_RECEIPT"
touch "$PARTIAL/COMPLETED"
mv "$PARTIAL" "$BUILD_DIR"
trap - EXIT

echo "REFERENCE_BIN=$PUBLISHED_BINARY"
echo "DETECTOR_BUILD_RECEIPT=$BUILD_RECEIPT"
