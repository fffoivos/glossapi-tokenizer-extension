#!/usr/bin/env bash
# Build and attest the pinned GlossAPI Rust quality modules for one v3 run.
# This is a Phase 0 action: publication is one no-replace directory rename and
# its immutable build receipt is later bound into the frozen run contract.
set -euo pipefail

: "${PHASE04_DIR:?PHASE04_DIR must be exported by the v3 dispatcher}"
: "${AGENT1_V3_UENV:?AGENT1_V3_UENV must be exported by the v3 dispatcher}"
: "${AGENT1_V3_RUNTIME_VENV:?AGENT1_V3_RUNTIME_VENV must be exported by the v3 dispatcher}"
: "${AGENT1_V3_DATA_ROOT:?AGENT1_V3_DATA_ROOT must be exported by the v3 dispatcher}"
: "${AGENT1_V3_GLOSSAPI_ROOT:?AGENT1_V3_GLOSSAPI_ROOT must be exported by the v3 dispatcher}"
: "${AGENT1_V3_GLOSSAPI_COMMIT:?AGENT1_V3_GLOSSAPI_COMMIT must be exported by the v3 dispatcher}"
: "${AGENT1_V3_GLOSSAPI_RUNTIME_ROOT:?AGENT1_V3_GLOSSAPI_RUNTIME_ROOT must be exported by the v3 dispatcher}"
: "${AGENT1_V3_GLOSSAPI_BUILD_RECEIPT:?AGENT1_V3_GLOSSAPI_BUILD_RECEIPT must be exported by the v3 dispatcher}"
: "${AGENT1_V3_GLOSSAPI_MODULE_DIR:?AGENT1_V3_GLOSSAPI_MODULE_DIR must be exported by the v3 dispatcher}"
: "${AGENT1_V3_RUN_ROOT:?AGENT1_V3_RUN_ROOT must be exported by the v3 dispatcher}"
: "${AGENT1_V3_RUN_ID:?AGENT1_V3_RUN_ID must be exported by the v3 dispatcher}"
: "${AGENT1_V3_ATTEMPT_ID:?AGENT1_V3_ATTEMPT_ID must be exported by the v3 dispatcher}"
: "${AGENT1_V3_ATTEMPT_DIR:?AGENT1_V3_ATTEMPT_DIR must be exported by the v3 dispatcher}"
: "${AGENT1_V3_PHASE0_DIR:?AGENT1_V3_PHASE0_DIR must be exported by the v3 dispatcher}"
: "${AGENT1_V3_ALLOCATION_EVIDENCE:?AGENT1_V3_ALLOCATION_EVIDENCE must be exported by the v3 dispatcher}"
: "${AGENT1_V3_EXPECTED_COMMIT:?AGENT1_V3_EXPECTED_COMMIT must be exported by the v3 dispatcher}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
: "${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK is required}"

quality_script="$PHASE04_DIR/scripts/profile_dataset_quality_rust.py"
test -s "$quality_script" || {
    echo "ERROR: missing GlossAPI quality receipt tool: $quality_script" >&2
    exit 120
}

# Do not allow an environment override to decouple the contract-bound paths
# from the pinned runtime naming convention.
expected_runtime_root="$AGENT1_V3_DATA_ROOT/runtime/glossapi-rust-quality-$AGENT1_V3_GLOSSAPI_COMMIT"
[[ "$AGENT1_V3_GLOSSAPI_RUNTIME_ROOT" == "$expected_runtime_root" ]] || {
    echo "ERROR: GlossAPI runtime root must equal $expected_runtime_root" >&2
    exit 121
}
[[ "$AGENT1_V3_GLOSSAPI_BUILD_RECEIPT" == "$expected_runtime_root/build_receipt.json" ]] || {
    echo "ERROR: GlossAPI build receipt must live under the immutable runtime root" >&2
    exit 121
}
[[ "$AGENT1_V3_GLOSSAPI_MODULE_DIR" == "$expected_runtime_root/modules" ]] || {
    echo "ERROR: GlossAPI module directory must live under the immutable runtime root" >&2
    exit 121
}
expected_attempt_dir="$AGENT1_V3_RUN_ROOT/phase0/attempts/$AGENT1_V3_ATTEMPT_ID"
[[ "$AGENT1_V3_ATTEMPT_DIR" == "$expected_attempt_dir" ]] || {
    echo "ERROR: GlossAPI build must use its Phase 0 attempt directory" >&2
    exit 121
}
test -s "$AGENT1_V3_ALLOCATION_EVIDENCE" || {
    echo "ERROR: CPU allocation evidence is missing" >&2
    exit 121
}
[[ -z "${CUDA_VISIBLE_DEVICES:-}" && -z "${ROCR_VISIBLE_DEVICES:-}" && -z "${HIP_VISIBLE_DEVICES:-}" ]] || {
    echo "ERROR: GlossAPI quality runtime build must not expose GPUs" >&2
    exit 122
}

[[ "$(git -C "$AGENT1_V3_GLOSSAPI_ROOT" rev-parse --is-inside-work-tree 2>/dev/null || true)" == "true" ]] || {
    echo "ERROR: prepare the clean pinned GlossAPI checkout first: $AGENT1_V3_GLOSSAPI_ROOT" >&2
    exit 123
}
[[ "$(git -C "$AGENT1_V3_GLOSSAPI_ROOT" rev-parse HEAD)" == "$AGENT1_V3_GLOSSAPI_COMMIT" ]] || {
    echo "ERROR: GlossAPI checkout is not at $AGENT1_V3_GLOSSAPI_COMMIT" >&2
    exit 124
}
[[ -z "$(git -C "$AGENT1_V3_GLOSSAPI_ROOT" status --porcelain --untracked-files=normal)" ]] || {
    echo "ERROR: GlossAPI quality checkout must be clean" >&2
    exit 125
}

for crate in glossapi_rs_noise glossapi_rs_cleaner; do
    test -s "$AGENT1_V3_GLOSSAPI_ROOT/rust/$crate/Cargo.toml" || {
        echo "ERROR: missing Cargo manifest for $crate" >&2
        exit 126
    }
    test -s "$AGENT1_V3_GLOSSAPI_ROOT/rust/$crate/Cargo.lock" || {
        echo "ERROR: missing Cargo lock for $crate" >&2
        exit 126
    }
done

phase0_receipt="$AGENT1_V3_PHASE0_DIR/glossapi_quality_runtime_receipt.json"
attempt_receipt="$AGENT1_V3_ATTEMPT_DIR/glossapi_quality_runtime_attempt_receipt.json"
for immutable_path in \
    "$AGENT1_V3_GLOSSAPI_RUNTIME_ROOT" \
    "$AGENT1_V3_GLOSSAPI_BUILD_RECEIPT" \
    "$AGENT1_V3_GLOSSAPI_MODULE_DIR" \
    "$phase0_receipt" \
    "$attempt_receipt"; do
    [[ ! -e "$immutable_path" && ! -L "$immutable_path" ]] || {
        echo "ERROR: immutable GlossAPI quality-runtime output already exists: $immutable_path" >&2
        exit 127
    }
done

mkdir -p "$(dirname "$AGENT1_V3_GLOSSAPI_RUNTIME_ROOT")"
command -v flock >/dev/null || { echo "ERROR: flock is required" >&2; exit 128; }
exec 9>"${AGENT1_V3_GLOSSAPI_RUNTIME_ROOT}.lock"
flock -n 9 || { echo "ERROR: another GlossAPI quality-runtime build is active" >&2; exit 128; }

# Re-check after taking the cooperative publication lock.
for immutable_path in \
    "$AGENT1_V3_GLOSSAPI_RUNTIME_ROOT" \
    "$AGENT1_V3_GLOSSAPI_BUILD_RECEIPT" \
    "$AGENT1_V3_GLOSSAPI_MODULE_DIR" \
    "$phase0_receipt" \
    "$attempt_receipt"; do
    [[ ! -e "$immutable_path" && ! -L "$immutable_path" ]] || {
        echo "ERROR: immutable GlossAPI quality-runtime output appeared during setup: $immutable_path" >&2
        exit 129
    }
done

partial="$AGENT1_V3_GLOSSAPI_RUNTIME_ROOT.partial-$SLURM_JOB_ID"
[[ ! -e "$partial" && ! -L "$partial" ]] || {
    echo "ERROR: stale GlossAPI quality-runtime partial exists: $partial" >&2
    exit 130
}
scratch_parent=${SLURM_TMPDIR:-"$AGENT1_V3_DATA_ROOT/tmp"}
scratch_root="$scratch_parent/agent1-v3-glossapi-quality-$SLURM_JOB_ID"
[[ ! -e "$scratch_root" && ! -L "$scratch_root" ]] || {
    echo "ERROR: stale GlossAPI build scratch directory exists: $scratch_root" >&2
    exit 130
}
mkdir -p "$(dirname "$scratch_root")"
mkdir "$scratch_root"
mkdir -p "$partial/modules" "$partial/wheels"

MATURIN_VERSION=${MATURIN_VERSION:-1.9.4}
tool_venv="$scratch_root/maturin-venv"
uenv run "$AGENT1_V3_UENV" --view=default -- bash -lc '
set -euo pipefail
test "$(uname -m)" = aarch64
python3 -m venv "$1"
"$1/bin/python" -m pip install --disable-pip-version-check "maturin==$2"
' bash "$tool_venv" "$MATURIN_VERSION"

for crate in glossapi_rs_noise glossapi_rs_cleaner; do
    manifest="$AGENT1_V3_GLOSSAPI_ROOT/rust/$crate/Cargo.toml"
    CARGO_BUILD_JOBS="$SLURM_CPUS_PER_TASK" \
    CARGO_TARGET_DIR="$scratch_root/target-$crate" \
    uenv run "$AGENT1_V3_UENV" --view=default -- \
        "$tool_venv/bin/maturin" build \
        --release --locked --interpreter "$AGENT1_V3_RUNTIME_VENV/bin/python" \
        --manifest-path "$manifest" --out "$partial/wheels"
done

mapfile -t wheels < <(find "$partial/wheels" -maxdepth 1 -type f -name '*.whl' -print | sort)
[[ ${#wheels[@]} -eq 2 ]] || {
    echo "ERROR: expected exactly two GlossAPI Rust wheels, found ${#wheels[@]}" >&2
    exit 131
}
uenv run "$AGENT1_V3_UENV" --view=default -- \
    "$AGENT1_V3_RUNTIME_VENV/bin/python" -m pip install \
    --disable-pip-version-check --no-deps --target "$partial/modules" "${wheels[@]}"

# Attest staged imports while recording their future immutable location.  The
# staged root is renamed only after this receipt has been written successfully.
export PYTHONPATH="$partial/modules${PYTHONPATH:+:$PYTHONPATH}"
uenv run "$AGENT1_V3_UENV" --view=default -- \
    "$AGENT1_V3_RUNTIME_VENV/bin/python" "$quality_script" build-receipt \
    --glossapi-root "$AGENT1_V3_GLOSSAPI_ROOT" \
    --expected-commit "$AGENT1_V3_GLOSSAPI_COMMIT" \
    --module-root "$partial/modules" \
    --published-module-root "$AGENT1_V3_GLOSSAPI_MODULE_DIR" \
    --maturin-version "$MATURIN_VERSION" \
    --output "$partial/build_receipt.json"

# GNU mv's --no-clobber preserves a racing output; the partial-directory check
# turns its otherwise-successful no-op into a hard failure.  Both paths share a
# parent filesystem, so a successful rename is atomic.
mv -T -n -- "$partial" "$AGENT1_V3_GLOSSAPI_RUNTIME_ROOT"
[[ ! -e "$partial" && ! -L "$partial" ]] || {
    echo "ERROR: refusing to replace an existing GlossAPI quality-runtime root" >&2
    exit 132
}

export PYTHONPATH="$AGENT1_V3_GLOSSAPI_MODULE_DIR${PYTHONPATH:+:$PYTHONPATH}"
uenv run "$AGENT1_V3_UENV" --view=default -- \
    "$AGENT1_V3_RUNTIME_VENV/bin/python" "$quality_script" validate-build-receipt \
    --receipt "$AGENT1_V3_GLOSSAPI_BUILD_RECEIPT" \
    --expected-commit "$AGENT1_V3_GLOSSAPI_COMMIT"

# The build receipt attests the Rust modules.  This second immutable receipt
# binds that build to the exact Phase 0 Slurm attempt and CPU allocation.
uenv run "$AGENT1_V3_UENV" --view=default -- \
    "$AGENT1_V3_RUNTIME_VENV/bin/python" - \
    "$attempt_receipt" \
    "$AGENT1_V3_RUN_ID" \
    "$AGENT1_V3_ATTEMPT_ID" \
    "$AGENT1_V3_EXPECTED_COMMIT" \
    "$AGENT1_V3_GLOSSAPI_ROOT" \
    "$AGENT1_V3_GLOSSAPI_COMMIT" \
    "$AGENT1_V3_GLOSSAPI_RUNTIME_ROOT" \
    "$AGENT1_V3_GLOSSAPI_BUILD_RECEIPT" \
    "$AGENT1_V3_GLOSSAPI_MODULE_DIR" \
    "$AGENT1_V3_ALLOCATION_EVIDENCE" \
    "$AGENT1_V3_UENV" \
    "$MATURIN_VERSION" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    output_value,
    run_id,
    attempt_id,
    code_commit,
    glossapi_root_value,
    glossapi_commit,
    runtime_root_value,
    build_receipt_value,
    module_root_value,
    allocation_evidence_value,
    uenv,
    maturin_version,
) = sys.argv[1:]

output = Path(output_value)
build_receipt = Path(build_receipt_value)
module_root = Path(module_root_value).resolve()
allocation_evidence = Path(allocation_evidence_value)


def file_binding(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise SystemExit(f"missing receipt input: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


build = json.loads(build_receipt.read_text(encoding="utf-8"))
if (
    build.get("schema_version") != "glossapi_rust_quality_build_receipt_v1"
    or build.get("status") != "passed"
    or build.get("source", {}).get("commit") != glossapi_commit
):
    raise SystemExit("validated GlossAPI build receipt drift")
modules = build.get("modules")
if not isinstance(modules, list):
    raise SystemExit("validated GlossAPI build receipt has no module list")
by_name = {str(row.get("name")): row for row in modules if isinstance(row, dict)}
if set(by_name) != {"glossapi_rs_noise", "glossapi_rs_cleaner"}:
    raise SystemExit("validated GlossAPI build receipt does not bind both modules")
for name, row in by_name.items():
    path = Path(str(row.get("path", ""))).resolve()
    try:
        path.relative_to(module_root)
    except ValueError as exc:
        raise SystemExit(f"{name}: module path escapes published module root") from exc
    if not path.is_file():
        raise SystemExit(f"{name}: published module is missing")

payload = {
    "schema_version": "agent1_full_corpus_v3_glossapi_quality_runtime_attempt_receipt_v1",
    "status": "passed",
    "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "phase": "phase0",
    "run_id": run_id,
    "attempt_id": attempt_id,
    "code_commit": code_commit,
    "glossapi": {"root": str(Path(glossapi_root_value).resolve()), "commit": glossapi_commit},
    "runtime": {
        "root": str(Path(runtime_root_value).resolve()),
        "module_root": str(module_root),
        "uenv": uenv,
        "maturin": maturin_version,
    },
    "build_receipt": file_binding(build_receipt),
    "allocation_evidence": file_binding(allocation_evidence),
    "modules": [by_name[name] for name in sorted(by_name)],
}

encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
temporary = output.with_name(f".{output.name}.partial-{os.getpid()}")
fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
with os.fdopen(fd, "wb") as handle:
    handle.write(encoded)
    handle.flush()
    os.fsync(handle.fileno())
try:
    os.link(temporary, output)
except FileExistsError as exc:
    raise SystemExit(f"immutable Phase 0 attempt receipt already exists: {output}") from exc
os.unlink(temporary)
directory_fd = os.open(output.parent, os.O_RDONLY)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
PY

ln -s "attempts/$AGENT1_V3_ATTEMPT_ID/$(basename "$attempt_receipt")" "$phase0_receipt"
echo "AGENT1_V3_GLOSSAPI_QUALITY_RUNTIME=$AGENT1_V3_GLOSSAPI_RUNTIME_ROOT"
echo "AGENT1_V3_GLOSSAPI_BUILD_RECEIPT=$AGENT1_V3_GLOSSAPI_BUILD_RECEIPT"
echo "AGENT1_V3_GLOSSAPI_PHASE0_RECEIPT=$attempt_receipt"
