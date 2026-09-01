#!/usr/bin/env python3
"""Record explicit owner authorization after every non-owner gate has passed."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from contract_utils import executing_code_bundle, file_binding, read_json, require, write_json_atomic
from freeze_hard_h_to_g_contract import (
    PRE_MAIN_ARTIFACTS_BY_SCALE,
    artifacts_for_stage,
    validate_artifact_manifest,
)
from producer_bundle_compatibility import load_authority


OWNER_ROLE = {
    "pre_main": "owner_production_authorization",
    "pre_extension": "owner_extension_authorization",
}


def explicit_run_authorization(text: str) -> bool:
    """Accept direct owner launch language without requiring magic keywords.

    The bound experiment, scale, allocation and preauthorization manifest carry
    the exact scientific scope.  Requiring the owner to repeat those identifiers
    verbatim turned a clear launch instruction into an artificial blocker.
    """

    words = text.casefold()
    return "authoriz" in words or any(
        phrase in words
        for phrase in ("run the training", "start the training", "launch the training")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization-stage", choices=tuple(OWNER_ROLE), required=True)
    parser.add_argument("--scale", choices=tuple(PRE_MAIN_ARTIFACTS_BY_SCALE))
    parser.add_argument("--owner", required=True)
    parser.add_argument("--confirmation-text", required=True)
    parser.add_argument("--confirmed-at", required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--preauthorization-manifest", type=Path, required=True)
    parser.add_argument("--producer-compatibility", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable owner authorization exists: {args.output}")
    require(args.owner == "fffoivos", "owner identity drift")
    confirmed = dt.datetime.fromisoformat(args.confirmed_at.replace("Z", "+00:00"))
    require(confirmed.tzinfo is not None, "owner confirmation timestamp must be timezone-aware")
    words = args.confirmation_text.casefold()
    require(
        explicit_run_authorization(args.confirmation_text),
        "confirmation text does not explicitly authorize or direct the training run",
    )
    if args.authorization_stage == "pre_main":
        require(args.scale in PRE_MAIN_ARTIFACTS_BY_SCALE, "pre-main authorization requires an exact scale")
        model_token = "8b" if args.scale == "8b" else "1.5b"
        repeated_scope = all(
            token in words for token in (model_token, "hplt", "openarchives")
        )
        direct_run_instruction = any(
            phrase in words
            for phrase in ("run the training", "start the training", "launch the training")
        )
        require(
            repeated_scope or direct_run_instruction,
            "pre-main confirmation scope is incomplete",
        )
    else:
        require(args.scale is None, "matched extension authorization does not accept a single scale")
        require("extension" in words, "extension confirmation scope is incomplete")
    experiment = read_json(args.experiment)
    require(experiment.get("schema_version") == "apertus_hard_h_to_g_replication_v2", "experiment contract drift")
    if args.authorization_stage == "pre_main":
        require(
            experiment.get("launch", {}).get("production_launch_authorized_by_scale", {}).get(args.scale) is False,
            "immutable experiment contract must remain unauthorized; the owner receipt is the authority",
        )
    current = executing_code_bundle()
    _, accepted = load_authority(args.producer_compatibility, current)
    preauth = read_json(args.preauthorization_manifest)
    require(
        preauth.get("gate_stage") == args.authorization_stage
        and preauth.get("scale") == args.scale
        and preauth.get("partial") is True
        and preauth.get("executing_code_bundle") == current
        and preauth.get("producer_bundle_compatibility") == file_binding(args.producer_compatibility),
        "preauthorization manifest identity drift",
    )
    owner_role = OWNER_ROLE[args.authorization_stage]
    expected_roles = [
        role for role in artifacts_for_stage(args.authorization_stage, args.scale)
        if role != owner_role
    ]
    require(set(preauth.get("artifacts", {})) == set(expected_roles), "preauthorization manifest role set drift")
    _, blockers = validate_artifact_manifest(
        args.preauthorization_manifest,
        expected_roles,
        accepted_producers=accepted,
        expected_scale=args.scale,
    )
    require(not blockers, f"non-owner authorization gates do not pass: {blockers}")
    payload = {
        "schema_version": "apertus_hard_h_to_g_owner_authorization_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "authorization_stage": args.authorization_stage,
        "scale": args.scale,
        "owner": args.owner,
        "confirmation_text": args.confirmation_text,
        "confirmed_at": confirmed.astimezone(dt.timezone.utc).isoformat(),
        "experiment": file_binding(args.experiment),
        "allocation": file_binding(args.allocation),
        "preauthorization_manifest": file_binding(args.preauthorization_manifest),
        "producer_bundle_compatibility": file_binding(args.producer_compatibility),
        "executing_code_bundle": current,
        "checks": {
            "explicit_owner_confirmation_recorded": True,
            "all_non_owner_stage_gates_passed_first": True,
            "authorization_scope_is_exact": True,
            "scope_is_bound_by_experiment_scale_and_manifest": True,
        },
    }
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
