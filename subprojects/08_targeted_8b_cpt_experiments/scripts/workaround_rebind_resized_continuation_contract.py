#!/usr/bin/env python3
"""Compile a candidate continuation on a smaller H-to-G runtime profile.

This is an experiment-owned adapter for issue #119.  It preserves the frozen
continuation segments, checkpoints, phase paths, training argv semantics and
evaluation plan, replacing only the executable science bundle and the runtime
geometry that must be qualified in the first held allocation.  The canonical
runner still owns allocation, claims, promotion, training and handoff.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from contract_utils import file_binding, read_json, require, write_json_atomic


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def replace_string(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return new if value == old else value
    if isinstance(value, list):
        return [replace_string(row, old, new) for row in value]
    if isinstance(value, dict):
        return {key: replace_string(row, old, new) for key, row in value.items()}
    return value


def replace_binding(rows: list[dict[str, Any]], item_id: str, path: Path) -> None:
    matches = [row for row in rows if row.get("id") == item_id]
    require(len(matches) == 1, f"immutable input is not unique: {item_id}")
    matches[0].clear()
    matches[0].update({"id": item_id, **file_binding(path), "verify_at_submit": True})


def profile_for(scale: str, allocation: dict[str, Any]) -> dict[str, Any]:
    if scale == "8b":
        profile = allocation["profiles"]["8b"]
    else:
        candidates = allocation["profiles"]["1p5b_candidates"]
        profile = next(
            (row for row in candidates if int(row["nodes"]) == 2), None
        )
        require(profile is not None, "two-node 1.5B candidate is absent")
    require(
        str(profile["status"]).startswith("conditional_minimum_candidate"),
        "selected profile is not the minimum conditional candidate",
    )
    return profile


def qualification_argv(
    *,
    scale: str,
    code_root: Path,
    code_receipt: Path,
    canonical_runner_root: Path,
    stage_root: Path,
    qualification_contract: Path,
    allocation: Path,
    megatron_root: Path,
    megatron_receipt: Path,
    validation_root: Path,
    validation_receipt: Path,
) -> list[str]:
    remaining_blocks = 8 if scale == "1p5b" else 2
    return [
        "/usr/bin/env",
        f"PYTHONPATH={code_root / 'subprojects/08_targeted_8b_cpt_experiments/scripts'}:{code_root / 'subprojects/08_targeted_8b_cpt_experiments/evaluation'}:{canonical_runner_root / 'src'}:{canonical_runner_root / 'src/_vendor/campaign_pydeps'}",
        # The frozen canonical schema wheel is CPython-3.12/aarch64 and the
        # held worker runs inside the declared uenv.  Do not bypass it with
        # the login host's /usr/bin Python.
        "python3.12",
        str(code_root / "subprojects/08_targeted_8b_cpt_experiments/scripts/workaround_parameterized_profile_qualification.py"),
        "--manifest", "{manifest}", "--scale", scale,
        "--allocation", str(allocation),
        "--canonical-runner-root", str(canonical_runner_root),
        "--qualification-contract", str(qualification_contract),
        "--qualification-root", "{qualification_root}",
        "--qualification-context", "{qualification_context}",
        "--run-root", "{run_root}", "--campaign-id", "{campaign_id}",
        "--contract-digest", "{contract_digest}",
        "--code-root", str(code_root), "--code-receipt", str(code_receipt),
        "--stage-root", str(stage_root), "--megatron-root", str(megatron_root),
        "--megatron-receipt", str(megatron_receipt),
        "--validation-root", str(validation_root),
        "--validation-receipt", str(validation_receipt),
        "--extra-valid-sets", "hplt openarchives greek_phd english de ru zh code old_greek",
        "--new-greek-valid-sets", "hplt openarchives greek_phd",
        "--remaining-conservative-blocks", str(remaining_blocks),
        "--checkpoint-reserve-seconds", "1200",
        "--maximum-allocation-seconds", "43200",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("8b", "1p5b"), required=True)
    parser.add_argument("--base-campaign", type=Path, required=True)
    parser.add_argument("--base-runtime", type=Path, required=True)
    parser.add_argument("--base-evaluation", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--code-receipt", type=Path, required=True)
    parser.add_argument("--canonical-runner-root", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--qualification-contract", type=Path, required=True)
    parser.add_argument("--producer-compatibility", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    code_root = args.code_root.resolve()
    runner_root = args.canonical_runner_root.resolve()
    allocation_path = args.allocation.resolve()
    code_receipt = args.code_receipt.resolve()
    qualification_contract = args.qualification_contract.resolve()
    producer_compatibility = args.producer_compatibility.resolve()
    for path in (code_receipt, allocation_path, qualification_contract, producer_compatibility):
        require(path.is_file(), f"required file is missing: {path}")
    require((runner_root / "src").is_dir(), "canonical runner root is missing src")
    require((code_root / "subprojects/08_targeted_8b_cpt_experiments").is_dir(), "scientific code root drift")
    allocation = read_json(allocation_path)
    require(allocation.get("schema_version") == "apertus_hard_h_to_g_allocation_v1", "allocation schema drift")
    profile = profile_for(args.scale, allocation)
    contract = read_json(qualification_contract)
    require(
        contract.get("schema_version") == "apertus_hard_h_to_g_prelaunch_benchmark_contract_v1"
        and contract.get("status") == "frozen"
        and contract.get("scale") == args.scale
        and contract.get("profile_id") == profile["profile_id"],
        "qualification contract does not bind selected profile",
    )
    require(
        (int(contract["nodes"]), int(contract["data_parallel"]), int(contract["microbatch"]))
        == (int(profile["nodes"]), int(profile["data_parallel"]), int(profile["microbatch"])),
        "qualification geometry mismatch",
    )
    base_campaign = read_json(args.base_campaign.resolve())
    base_runtime = read_json(args.base_runtime.resolve())
    base_evaluation = read_json(args.base_evaluation.resolve())
    require(base_campaign.get("schema_version") == "apertus_campaign_v3", "base campaign schema drift")
    require(base_runtime.get("schema_version") == "apertus_runtime_profile_v2", "base runtime schema drift")
    require(base_evaluation.get("schema_version") == "apertus_evaluation_plan_v2", "base evaluation schema drift")
    campaign = copy.deepcopy(base_campaign)
    runtime = copy.deepcopy(base_runtime)
    evaluation = copy.deepcopy(base_evaluation)
    old_code_row = next(row for row in campaign["science"]["immutable_inputs"] if row["id"] == "scientific_code_receipt")
    # Receipt suffix removal is only valid for the frozen-bundle naming convention.
    require(old_code_row["path"].endswith(".receipt.json"), "base scientific receipt name drift")
    old_code_root = old_code_row["path"][: -len(".receipt.json")]
    campaign = replace_string(campaign, old_code_root, str(code_root))
    campaign = replace_string(campaign, old_code_row["path"], str(code_receipt))
    inputs = campaign["science"]["immutable_inputs"]
    replace_binding(inputs, "scientific_code_receipt", code_receipt)
    replace_binding(inputs, "allocation_contract", allocation_path)
    replace_binding(inputs, "qualification_contract", qualification_contract)
    replace_binding(inputs, "producer_compatibility", producer_compatibility)
    runtime["profile_id"] = str(profile["profile_id"])
    runtime["slurm"]["nodes"] = int(profile["nodes"])
    runtime["parallelism"] = {
        "tensor": int(profile["tensor_parallel"]),
        "pipeline": int(profile["pipeline_parallel"]),
        "context": int(profile["context_parallel"]),
        "data": int(profile["data_parallel"]),
    }
    runtime["qualification_scope"]["micro_batch"] = int(profile["microbatch"])
    runtime["submission_policy"]["max_nodes"] = max(int(profile["nodes"]), 4)
    campaign["runtime_profile_id"] = runtime["profile_id"]
    campaign["science"]["runtime_requirements_sha256"] = digest(runtime["qualification_scope"])
    stage_root = Path(next(arg for arg in campaign["science"]["train_argv"] if "/cpt_runs/hard_h2g_matched/" in arg)).resolve()
    # The chosen argv element is the stage root itself; reject a malformed historical train argv.
    require(stage_root.is_dir() and stage_root.name.endswith("v14"), "stage root cannot be recovered from base argv")
    megatron_index = campaign["science"]["train_argv"].index("--megatron-root") + 1
    validation_index = campaign["science"]["train_argv"].index("--validation-root") + 1
    megatron_receipt_index = campaign["science"]["train_argv"].index("--megatron-receipt") + 1
    validation_receipt_index = campaign["science"]["train_argv"].index("--validation-receipt") + 1
    campaign["qualification"] = {"argv": qualification_argv(
        scale=args.scale, code_root=code_root, code_receipt=code_receipt,
        canonical_runner_root=runner_root, stage_root=stage_root,
        qualification_contract=qualification_contract, allocation=allocation_path,
        megatron_root=Path(campaign["science"]["train_argv"][megatron_index]),
        megatron_receipt=Path(campaign["science"]["train_argv"][megatron_receipt_index]),
        validation_root=Path(campaign["science"]["train_argv"][validation_index]),
        validation_receipt=Path(campaign["science"]["train_argv"][validation_receipt_index]),
    )}
    output_root = args.output_root.resolve()
    require(not output_root.exists(), "immutable candidate output already exists")
    output_root.mkdir(parents=True)
    # The v3 data manifest is intentionally portable: its path and each
    # prepared gate are relative to the contract directory.  Preserve the
    # already-frozen bytes by hard-linking that small closure beside the new
    # candidate, rather than rebuilding or altering the data contract.
    base_root = args.base_campaign.resolve().parent
    portable_data_files = [
        "training_data_manifest.json",
        "readiness_plan.json",
        "prepared_gate_hplt.json",
        "prepared_gate_openarchives.json",
        "prepared_gate_foreign_replay.json",
        "prepared_gate_old_greek_replay.json",
    ]
    for name in portable_data_files:
        source = base_root / name
        require(source.is_file() and not source.is_symlink(), f"base portable data file is missing: {source}")
        os.link(source, output_root / name)
    campaign_path = output_root / "campaign.json"
    runtime_path = output_root / "runtime-candidate.json"
    evaluation_path = output_root / "evaluation.json"
    compiled_path = output_root / "compiled-candidate.json"
    write_json_atomic(campaign_path, campaign)
    write_json_atomic(runtime_path, runtime)
    write_json_atomic(evaluation_path, evaluation)
    # The canonical bundle carries its schema dependency as a vendored tree;
    # compile on the login/control path must use the same import closure as
    # the eventual held-allocation wrapper.
    sys.path.insert(0, str(runner_root / "src/_vendor/campaign_pydeps"))
    sys.path.insert(0, str(runner_root / "src"))
    from apertus_cscs_campaign.contracts import compile_contracts  # pylint: disable=import-outside-toplevel
    write_json_atomic(compiled_path, compile_contracts(campaign_path, runtime_path, evaluation_path))
    print(compiled_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
