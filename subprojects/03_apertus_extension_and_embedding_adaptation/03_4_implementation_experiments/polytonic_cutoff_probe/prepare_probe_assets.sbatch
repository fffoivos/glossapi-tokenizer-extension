#!/usr/bin/env bash
#SBATCH --job-name=poly_cut_assets
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/%x-%j.err

set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/iopsstor/scratch/cscs/fffoivos/tokenizer_finalization/20260729T094000Z-poly512-1024}"
CODE_ROOT="${CODE_ROOT:-$RUN_ROOT/code}"
PROBE_CODE="$CODE_ROOT/polytonic_cutoff_probe"
TD_CODE="$CODE_ROOT/token_distillation"
HF_CLI="${HF_CLI:-/users/fffoivos/.venvs/hfupload/bin/hf}"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-/usr/bin/python3.11}"
COVERAGE_ENV="${COVERAGE_ENV:-$RUN_ROOT/python_envs/coverage_py311}"
COVERAGE_PYTHON="$COVERAGE_ENV/bin/python"
PYARROW_PYTHON="${PYARROW_PYTHON:-$COVERAGE_PYTHON}"

ASSET_ROOT="$RUN_ROOT/assets"
HF_REPO_ROOT="$ASSET_ROOT/apertus-tokenizer-extension"
BASE_MODEL="$HF_REPO_ROOT/experiment-checkpoints/TokenDistil-3.5B"
PARQUET_ROOT="$ASSET_ROOT/fineweb2"
PROBE_DATA="$RUN_ROOT/probe_data"

CANDIDATE_512="$RUN_ROOT/candidates/c3p_poly_added_0512"
CANDIDATE_1024="$RUN_ROOT/candidates/c3p_poly_added_1024"
EXPECTED_512_SHA="bbb08e71929b519c5c2362338b0fc6a0e99955cb8fdbf0729ae1311117e6561b"
EXPECTED_1024_SHA="3465350b74e186ce22e5aefcfb67c33ac9cab7d90bc42596b52c82c7bd93fbff"

mkdir -p "$ASSET_ROOT" "$PARQUET_ROOT" "$PROBE_DATA" "$RUN_ROOT/logs"

echo "=== polytonic cutoff asset/coverage job ==="
date -u
echo "host:       $(hostname)"
echo "RUN_ROOT:   $RUN_ROOT"
echo "CODE_ROOT:  $CODE_ROOT"
echo "BASE_MODEL: $BASE_MODEL"

test -f "$PROBE_CODE/prepare_probe_jsonl.py"
test -f "$TD_CODE/td_coverage_prepass.py"
test -f "$CANDIDATE_512/tokenizer.json"
test -f "$CANDIDATE_1024/tokenizer.json"

actual_512_sha="$(sha256sum "$CANDIDATE_512/tokenizer.json" | awk '{print $1}')"
actual_1024_sha="$(sha256sum "$CANDIDATE_1024/tokenizer.json" | awk '{print $1}')"
test "$actual_512_sha" = "$EXPECTED_512_SHA"
test "$actual_1024_sha" = "$EXPECTED_1024_SHA"

"$HF_CLI" download fffoivos/apertus-tokenizer-extension \
  --include 'experiment-checkpoints/TokenDistil-3.5B/*' \
  --local-dir "$HF_REPO_ROOT"

test -f "$BASE_MODEL/config.json"
test -f "$BASE_MODEL/tokenizer.json"
test -f "$BASE_MODEL/model.safetensors.index.json"

if [ ! -x "$COVERAGE_PYTHON" ]; then
    "$PYTHON_BOOTSTRAP" -m venv "$COVERAGE_ENV"
fi
if ! "$COVERAGE_PYTHON" -c 'from tokenizers import Tokenizer; import pyarrow.parquet' 2>/dev/null; then
    "$COVERAGE_PYTHON" -m pip install --disable-pip-version-check \
      "tokenizers==0.22.1" "pyarrow==25.0.0"
fi
"$COVERAGE_PYTHON" -c \
  'from tokenizers import Tokenizer; import pyarrow; print("xfer Python runtime ready", pyarrow.__version__)'

download_parquet() {
    local url="$1"
    local output="$2"
    if [ ! -s "$output" ]; then
        curl -L --fail --retry 5 --retry-delay 5 "$url" -o "$output.part"
        mv "$output.part" "$output"
    fi
}

download_parquet \
  "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/main/data/grc_Grek/train/000_00000.parquet?download=true" \
  "$PARQUET_ROOT/grc_train_000_00000.parquet"
download_parquet \
  "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/main/data/grc_Grek/test/000_00000.parquet?download=true" \
  "$PARQUET_ROOT/grc_test_000_00000.parquet"
download_parquet \
  "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/main/data/ell_Grek/test/000_00000.parquet?download=true" \
  "$PARQUET_ROOT/ell_test_000_00000.parquet"

"$PYARROW_PYTHON" "$PROBE_CODE/prepare_probe_jsonl.py" \
  --ancient-train-parquet "$PARQUET_ROOT/grc_train_000_00000.parquet" \
  --ancient-test-parquet "$PARQUET_ROOT/grc_test_000_00000.parquet" \
  --modern-test-parquet "$PARQUET_ROOT/ell_test_000_00000.parquet" \
  --output-dir "$PROBE_DATA" \
  --max-ancient-test-docs 1000 \
  --max-modern-test-docs 1000 \
  --max-eval-chars-per-doc 3000

"$PYARROW_PYTHON" - "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
specs = {
    512: {148924, 148979, 148987},
    1024: {148924, 148979, 148987, 149436},
}
for cutoff, excluded in specs.items():
    end = 148480 + cutoff
    path = root / f"train_token_ids_{cutoff:04d}.txt"
    path.write_text(
        "".join(f"{token_id}\n" for token_id in range(148480, end) if token_id not in excluded),
        encoding="utf-8",
    )
    manifest = {
        "new_id_range": [148480, end],
        "selected_count": cutoff - len(excluded),
        "excluded_structural_bytelevel_ids": sorted(excluded),
        "exclusion_reason": (
            "not standalone UTF-8; retained and merge-chain initialized, "
            "but not independently token-distilled"
        ),
    }
    (root / f"train_token_ids_{cutoff:04d}.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
PY

run_coverage() {
    local cutoff="$1"
    local end_id="$2"
    local tokenizer="$3"
    local out_dir="$RUN_ROOT/coverage_$(printf '%04d' "$cutoff")"
    if [ -s "$out_dir/td_coverage_summary.json" ]; then
        echo "coverage already present for cutoff=$cutoff; reusing"
        return 0
    fi
    mkdir -p "$out_dir"
    TOKENIZERS_PARALLELISM=false "$COVERAGE_PYTHON" "$TD_CODE/td_coverage_prepass.py" \
      --input-jsonl "$PROBE_DATA/ancient_train.jsonl" \
      --base-tokenizer "$BASE_MODEL" \
      --student-tokenizer "$tokenizer" \
      --output-dir "$out_dir" \
      --new-id-start 148480 \
      --new-id-end "$end_id" \
      --target-extended-tokens 30000000 \
      --snippets-per-token 100 \
      --snippet-token-radius 50 \
      --progress-token-interval 5000000 \
      --require-nfc \
      >"$RUN_ROOT/logs/coverage_$(printf '%04d' "$cutoff").out" \
      2>"$RUN_ROOT/logs/coverage_$(printf '%04d' "$cutoff").err"
}

run_coverage 512 148992 "$CANDIDATE_512" &
pid_512=$!
run_coverage 1024 149504 "$CANDIDATE_1024" &
pid_1024=$!
wait "$pid_512"
wait "$pid_1024"

"$PYARROW_PYTHON" - "$RUN_ROOT" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
coverage = {}
for cutoff in (512, 1024):
    path = root / f"coverage_{cutoff:04d}" / "td_coverage_summary.json"
    data = json.loads(path.read_text())
    fraction = data["enough_25_fraction"]
    if fraction < 0.90:
        raise SystemExit(f"cutoff {cutoff}: enough_25_fraction={fraction:.4f} < 0.90")
    coverage[str(cutoff)] = {
        "summary": str(path),
        "tokens_scanned": data["tokens_scanned"],
        "enough_25_fraction": fraction,
        "status_counts": data["status_counts"],
    }

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

files = [
    root / "assets/fineweb2/grc_train_000_00000.parquet",
    root / "assets/fineweb2/grc_test_000_00000.parquet",
    root / "assets/fineweb2/ell_test_000_00000.parquet",
    root / "probe_data/ancient_train.jsonl",
    root / "probe_data/ancient_test.jsonl",
    root / "probe_data/modern_test.jsonl",
    root / "assets/apertus-tokenizer-extension/experiment-checkpoints/TokenDistil-3.5B/tokenizer.json",
]
manifest = {
    "schema_version": "polytonic-cutoff-assets-v1",
    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "files": {
        str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in files
    },
    "coverage": coverage,
}
(root / "asset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY

date -u
echo "asset and coverage preparation complete"
