#!/usr/bin/env bash
# Script-preparation gate. No SSH, downloads, Slurm submission or corpus processing.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PHASE04_DIR=$(cd "$HERE/.." && pwd)
REPO_ROOT=$(git -C "$PHASE04_DIR" rev-parse --show-toplevel)
ACADEMIC_DIR="$REPO_ROOT/subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic"
source "$HERE/paths.env"

python3 "$PHASE04_DIR/scripts/validate_configs.py"
python3 "$PHASE04_DIR/scripts/validate_source_license_adjudication.py" \
    --sources "$SOURCE_CONFIG" \
    --adjudication "$SOURCE_LICENSE_ADJUDICATION"
python3 -m compileall -q "$PHASE04_DIR/scripts" "$PHASE04_DIR/clariden" "$ACADEMIC_DIR/driver"

while IFS= read -r shell_file; do
    bash -n "$shell_file"
done < <(find "$PHASE04_DIR/clariden" -type f \( -name '*.sh' -o -name '*.sbatch' \) | sort)

cargo test --locked --manifest-path "$ACADEMIC_DIR/reference_detector/Cargo.toml"
cargo build --locked --manifest-path "$ACADEMIC_DIR/reference_detector/Cargo.toml"

if command -v uv >/dev/null; then
    uv run --python 3.12 --with pytest --with pyarrow --with tokenizers --with numpy \
        pytest -q "$PHASE04_DIR/tests"
else
    echo "WARNING: uv unavailable; Phase-04 pytest suite was not run." >&2
fi

echo "PREPARE OK"
echo "commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
echo "branch=$(git -C "$REPO_ROOT" branch --show-current)"
git -C "$REPO_ROOT" status --short
