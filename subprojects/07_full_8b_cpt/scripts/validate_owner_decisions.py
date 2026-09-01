#!/usr/bin/env python3
"""Validate the owner's explicit D0, libduth and launch decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contract import require_passing


REQUIRED_ACCEPTED = {
    "D0_selection_confirmed_after_per_document_rerun_or_explicit_point_estimate_acceptance",
    "libduth_permission_evidence_conflict_is_reconciled_or_explicitly_accepted",
    "explicit_production_launch_authorization_is_received",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--recipe-id", required=True)
    args = parser.parse_args()
    value = require_passing(args.decisions, "apertus_full_8b_owner_decisions_v1")
    if value.get("recipe_id") != args.recipe_id:
        raise ValueError("owner decision/recipe drift")
    decisions = value.get("decisions", {})
    missing = sorted(
        name for name in REQUIRED_ACCEPTED if decisions.get(name, {}).get("accepted") is not True
    )
    if missing:
        raise ValueError(f"owner decisions not accepted: {missing}")
    libduth = decisions[
        "libduth_permission_evidence_conflict_is_reconciled_or_explicitly_accepted"
    ]
    if libduth.get("include_libduth") is not True or libduth.get("legal_conclusion_claimed") is not False:
        raise ValueError("libduth decision must record inclusion without manufacturing a legal conclusion")
    if decisions.get("checkpoint_averaging", {}).get("accepted") is not False:
        raise ValueError("checkpoint averaging exclusion drift")
    print(json.dumps({"ok": True, "recipe_id": args.recipe_id, "accepted": sorted(REQUIRED_ACCEPTED)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
