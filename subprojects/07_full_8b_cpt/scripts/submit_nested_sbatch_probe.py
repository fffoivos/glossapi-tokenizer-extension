#!/usr/bin/env python3
"""Submit a real child job from the supervisor's exact uenv runtime."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
from pathlib import Path

from contract import atomic_write_json, verify_code_bundle_receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--code-bundle-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.code_root = args.code_root.resolve()
    args.code_bundle_receipt = args.code_bundle_receipt.resolve()
    args.output_root = args.output_root.resolve()
    bundle = verify_code_bundle_receipt(args.code_bundle_receipt, args.code_root)
    parent = os.environ["SLURM_JOB_ID"]
    exports = "ALL," + ",".join(
        (
            f"FULL8_CODE_ROOT={args.code_root}",
            f"FULL8_CODE_BUNDLE_RECEIPT={args.code_bundle_receipt}",
            f"FULL8_NESTED_SMOKE_ROOT={args.output_root}",
            f"FULL8_NESTED_PARENT_JOB_ID={parent}",
        )
    )
    command = [
        "sbatch", "--uenv-passthrough=ignore", "--parsable",
        f"--output={args.output_root}/logs/%x-%j.out",
        f"--error={args.output_root}/logs/%x-%j.err",
        f"--export={exports}",
        str(args.code_root / "subprojects/07_full_8b_cpt/clariden/nested_sbatch_child.sbatch"),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    child = result.stdout.strip().split(";", 1)[0]
    atomic_write_json(
        args.output_root / "nested_sbatch_submission.json",
        {
            "schema_version": "apertus_full_8b_nested_sbatch_submission_v1",
            "status": "submitted",
            "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "parent_job_id": parent,
            "child_job_id": child,
            "code_root": str(args.code_root),
            "code_bundle_tree_sha256": bundle["tree_sha256"],
            "command_prefix": command[:3],
        },
    )
    print(json.dumps({"ok": True, "parent_job_id": parent, "child_job_id": child}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
