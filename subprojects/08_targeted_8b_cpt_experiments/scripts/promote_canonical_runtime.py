#!/usr/bin/env python3
"""Promote the exact candidate runtime bound by a canonical qualification."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

from contract_utils import file_binding, read_json, require, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled-candidate", type=Path, required=True)
    parser.add_argument("--qualification-receipt", type=Path, required=True)
    parser.add_argument("--canonical-runner-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    canonical_src = args.canonical_runner_root.resolve() / "src"
    require(canonical_src.is_dir(), "canonical runner source root missing")
    sys.path.insert(0, str(canonical_src))
    from apertus_cscs_campaign.contracts import (
        qualification_target,
        validate_runtime,
    )
    from apertus_cscs_campaign.engine import verify_compiled
    from apertus_cscs_campaign.receipts import digest, verify_binding

    manifest = verify_compiled(args.compiled_candidate.resolve())
    runtime = manifest["runtime"]
    require(runtime.get("status") == "candidate", "runtime is not a candidate")
    qualification_binding = file_binding(args.qualification_receipt)
    qualification = read_json(Path(verify_binding(qualification_binding)))
    qualification_schema = qualification.get("schema_version")
    require(
        qualification_schema
        in {"apertus_runtime_qualification_v2", "apertus_runtime_qualification_v3"}
        and qualification.get("status") == "passed",
        "canonical qualification receipt did not pass",
    )
    source_digest_field = (
        "source_candidate_contract_digest"
        if qualification_schema == "apertus_runtime_qualification_v3"
        else "candidate_contract_digest"
    )
    require(
        qualification.get("campaign_id") == manifest["campaign_id"]
        and qualification.get("profile_id") == runtime["profile_id"]
        and qualification.get(source_digest_field)
        == manifest["contract_digest"]
        and qualification.get("code_tree_sha256")
        == manifest["code_bundle"]["tree_sha256"]
        and qualification.get("target_digest") == digest(qualification_target(runtime)),
        "qualification does not bind the exact candidate runtime",
    )
    gate_set = read_json(Path(verify_binding(qualification["gate_set_receipt"])))
    require(
        gate_set.get("schema_version") == "apertus_campaign_gate_set_v2"
        and gate_set.get("status") == "passed"
        and gate_set.get("campaign_id") == manifest["campaign_id"]
        and gate_set.get("contract_digest") == manifest["contract_digest"]
        and gate_set.get("code_tree_sha256") == manifest["code_bundle"]["tree_sha256"]
        and bool(gate_set.get("gates"))
        and all(row.get("passed") is True for row in gate_set["gates"]),
        "canonical gate set is incomplete or foreign",
    )
    proven = copy.deepcopy(runtime)
    proven["status"] = "proven"
    proven["qualification_receipt"] = qualification_binding
    validate_runtime(proven)
    write_json_atomic(args.output, proven)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
