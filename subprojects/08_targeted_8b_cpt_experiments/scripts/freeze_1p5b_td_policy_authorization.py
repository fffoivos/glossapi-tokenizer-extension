#!/usr/bin/env python3
"""Freeze explicit owner approval for the prospective 1.5B TD policy."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

from contract_utils import file_binding, require, write_json_atomic


def digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--approval-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable authorization exists: {args.output}")
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    source = json.loads(args.approval_source.read_text(encoding="utf-8"))
    require(
        policy.get("schema_version") == "apertus_1p5b_td_acceptance_policy_v2",
        "policy schema drift",
    )
    require(
        source.get("schema_version") == "apertus_owner_scientific_amendment_approval_v1"
        and source.get("status") == "approved",
        "owner approval source is not approved",
    )
    require(
        source.get("amendment_id") == policy.get("amendment_id"),
        "owner approval amendment drift",
    )
    require(source.get("approved_by") == "user", "approval is not owner-authored")
    require(
        source.get("policy_sha256") == file_binding(args.policy)["sha256"],
        "owner approval policy digest drift",
    )
    receipt = {
        "schema_version": "apertus_1p5b_td_policy_authorization_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "amendment_id": policy["amendment_id"],
        "approved_by": "user",
        "approval_predates_new_td_job": True,
        "policy": file_binding(args.policy),
        "approval_source": file_binding(args.approval_source),
        "approval_scope_sha256": digest(
            {
                "amendment_id": policy["amendment_id"],
                "policy_sha256": file_binding(args.policy)["sha256"],
            }
        ),
    }
    write_json_atomic(args.output, receipt)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
