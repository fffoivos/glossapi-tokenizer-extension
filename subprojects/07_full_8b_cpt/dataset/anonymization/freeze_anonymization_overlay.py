#!/usr/bin/env python3
"""Freeze the parent corpus, raw validation, masker, and executing code."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from anonymization_common import (
    OVERLAY_SCHEMA,
    REPO_ROOT,
    absolute_receipt,
    canonical_sha256,
    read_json,
    sha256_file,
    utc_now,
    write_json_atomic,
)


CODE_NAMES = (
    "anonymization_common.py",
    "freeze_anonymization_overlay.py",
    "build_anonymization_inventory.py",
    "finalize_postmask_dedup.py",
    "build_sanitized_binary_shard.py",
    "finalize_sanitized_bridge.py",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-input-receipt", type=Path, required=True)
    parser.add_argument("--parent-heldout-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    args = parser.parse_args()

    stage = args.stage_root.resolve()
    stage.mkdir(parents=True, exist_ok=False)
    parent_path = args.parent_input_receipt.resolve()
    parent = read_json(parent_path)
    parent_sha = sha256_file(parent_path)
    if (
        parent.get("schema_version") != "full_cpt_training_bridge_input_receipt_v1"
        or parent.get("status") != "frozen"
    ):
        raise ValueError("parent input receipt is not frozen")
    heldout_path = args.parent_heldout_manifest.resolve()
    heldout = read_json(heldout_path)
    if (
        heldout.get("schema_version") != "full_cpt_training_heldouts_v1"
        or heldout.get("status") != "completed"
        or heldout.get("input_receipt_sha256") != parent_sha
    ):
        raise ValueError("parent heldout manifest is not bound to the parent input")
    validation_path = args.validation_manifest.resolve()
    validation = read_json(validation_path)
    if (
        validation.get("schema_version") != "apertus_full_8b_validation_manifest_v1"
        or validation.get("status") != "frozen"
        or len(validation.get("panels", [])) != 13
        or not validation.get("all_panels_training_exact_content_disjoint")
    ):
        raise ValueError("the corrected 13-panel validation manifest is not frozen")

    here = Path(__file__).resolve().parent
    bridge = REPO_ROOT / "subprojects/05_token_distillation_cpt/05_training_dataset_bridge/scripts"
    masker = REPO_ROOT / "subprojects/05_token_distillation_cpt/02_corpus_preparation/40_anonymize/scripts/pii_masker.py"
    phase = REPO_ROOT / "subprojects/05_token_distillation_cpt/06_25b_midtraining_probe/dataset/phase_partition.py"
    code_paths = [here / name for name in CODE_NAMES] + [
        bridge / "bridge_common.py",
        bridge / "build_binary_shard.py",
        masker,
        phase,
    ]
    code_receipts = [absolute_receipt(path) for path in code_paths]
    overlay_path = stage / "anonymization_overlay.json"
    heldout_output = stage / "heldouts" / "heldout_manifest.json"
    payload = {
        "schema_version": OVERLAY_SCHEMA,
        "status": "frozen",
        "created_at": utc_now(),
        "parent_input_receipt": absolute_receipt(parent_path),
        "parent_heldout_manifest": absolute_receipt(heldout_path),
        "validation_manifest": absolute_receipt(validation_path),
        "tasks": parent["tasks"],
        "tasks_sha256": canonical_sha256(parent["tasks"]),
        "task_count": len(parent["tasks"]),
        "tokenizer": parent["tokenizer"],
        "decontamination": parent["decontamination"],
        "repository": {
            "root": str(REPO_ROOT),
            "commit": parent["repository"]["commit"],
            "code_files": code_receipts,
        },
        "anonymization": {
            "policy": "apertus_email_ip_plus_validated_country_length_iban_v1",
            "implementation": absolute_receipt(masker),
            "replacement_tokens": ["<email-pii>", "<ip-pii>", "<iban-pii>"],
            "applies_to": [
                "hplt_new_greek",
                "non_hplt_new_greek",
                "foreign_replay",
                "old_greek_replay",
            ],
            "operation_order": [
                "heldout_exclusion",
                "greekmmlu_decontamination",
                "pii_masking",
                "global_exact_postmask_deduplication",
                "tokenization",
            ],
            "validation_text": "raw_frozen_not_masked",
            "pii_values_logged": False,
        },
    }
    write_json_atomic(overlay_path, payload)
    overlay_sha = sha256_file(overlay_path)
    derived = copy.deepcopy(heldout)
    derived["completed_at"] = utc_now()
    derived["input_receipt"] = str(overlay_path)
    derived["input_receipt_sha256"] = overlay_sha
    derived["derivation"] = {
        "policy": "reuse_raw_frozen_heldouts_for_anonymized_training_v1",
        "parent": absolute_receipt(heldout_path),
        "validation_is_not_anonymized": True,
    }
    write_json_atomic(heldout_output, derived)

    # Publish the heldout binding in a sidecar instead of creating a hash cycle.
    sidecar = {
        "schema_version": "full_cpt_anonymization_overlay_bindings_v1",
        "status": "frozen",
        "overlay": absolute_receipt(overlay_path),
        "heldout_manifest": absolute_receipt(heldout_output),
    }
    write_json_atomic(stage / "overlay_bindings.json", sidecar)
    print(json.dumps({"ok": True, "tasks": len(parent["tasks"]), "stage": str(stage)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
