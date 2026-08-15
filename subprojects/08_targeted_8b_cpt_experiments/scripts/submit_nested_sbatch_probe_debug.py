#!/usr/bin/env python3
"""Submit the targeted debug child from the controller's exact uenv runtime."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import tempfile
from pathlib import Path


def atomic_write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--code-bundle-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    code_root = args.code_root.resolve()
    receipt = args.code_bundle_receipt.resolve()
    output = args.output_root.resolve()

    verifier = code_root / "subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py"
    subprocess.run(
        ["/usr/bin/python3.11", str(verifier), "--root", str(code_root), "--receipt", str(receipt), "--kind", "scientific"],
        check=True,
    )
    parent = os.environ["SLURM_JOB_ID"]
    exports = "ALL," + ",".join(
        (
            f"FULL8_CODE_ROOT={code_root}",
            f"FULL8_CODE_BUNDLE_RECEIPT={receipt}",
            f"FULL8_NESTED_SMOKE_ROOT={output}",
            f"FULL8_NESTED_PARENT_JOB_ID={parent}",
        )
    )
    command = [
        "sbatch",
        "--uenv-passthrough=ignore",
        "--parsable",
        "--partition=debug",
        f"--output={output}/logs/%x-%j.out",
        f"--error={output}/logs/%x-%j.err",
        f"--export={exports}",
        str(code_root / "subprojects/08_targeted_8b_cpt_experiments/clariden/nested_sbatch_child_debug.sbatch"),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    child = result.stdout.strip().split(";", 1)[0]
    atomic_write_json(
        output / "nested_sbatch_submission.json",
        {
            "schema_version": "apertus_full_8b_nested_sbatch_submission_v1",
            "status": "submitted",
            "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "parent_job_id": parent,
            "child_job_id": child,
            "code_root": str(code_root),
            "code_bundle_receipt": str(receipt),
            "parent_runtime": "uenv run pytorch/v2.9.1:v2 --view=default -- python3",
            "nested_submit_flag": "--uenv-passthrough=ignore",
            "partition": "debug",
            "command": command,
        },
    )
    print(json.dumps({"ok": True, "parent_job_id": parent, "child_job_id": child}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
