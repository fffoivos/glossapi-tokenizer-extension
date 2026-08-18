#!/usr/bin/env python3
"""Atomically bind immutable companion-dataset revisions into private model cards.

This adapter deliberately performs a metadata-only commit: one replacement
``README.md`` and one new ``provenance/training_data.json`` per already
verified private checkpoint branch.  It refuses to run unless the branch's
previous full Hub-inventory receipt passed, and records both parent and child
revisions.  It never uploads, converts, replaces, or deletes weight files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


MODEL_REPO = "fffoivos/apertus-8b-greek-cpt"
PUBLIC_REPO = "fffoivos/apertus-8b-greek-cpt-modern-greek-train"
PRIVATE_REPO = "fffoivos/apertus-8b-greek-cpt-d0-full-mix"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path}: expected JSON object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def upload_receipt(path: Path, *, repo_id: str, private: bool) -> dict[str, Any]:
    value = read_json(path)
    require(value.get("schema_version") == "apertus_full8_frozen_dataset_hf_upload_v1", "dataset receipt schema drift")
    require(value.get("status") == "completed", "dataset upload is incomplete")
    require(value.get("repo_id") == repo_id and bool(value.get("private")) is private, "dataset repository or visibility drift")
    revision = str(value.get("revision", ""))
    require(len(revision) == 40 and value.get("training_must_pin_revision") == revision, "dataset receipt lacks immutable revision")
    return {"repo_id": repo_id, "revision": revision, "private": private, "receipt": {"path": str(path.resolve()), "sha256": sha256_file(path)}}


def pinned_card(original: str, *, public: dict[str, Any], private: dict[str, Any]) -> str:
    marker = "## Training-data provenance\n"
    require(marker in original, "staging card lacks training-data provenance section")
    prefix = original.split(marker, 1)[0].rstrip()
    return "\n".join((
        prefix,
        "",
        "## Frozen training-data provenance",
        "",
        f"- Exact public Modern-Greek train-only snapshot: [`{public['repo_id']}`](https://huggingface.co/datasets/{public['repo_id']}/tree/{public['revision']}) at `{public['revision']}`.",
        f"- Exact packed 79/20/1 D0 mixture: [`{private['repo_id']}`](https://huggingface.co/datasets/{private['repo_id']}/tree/{private['revision']}) at `{private['revision']}` (private because constrained replay sources are not authorized for redistribution).",
        "- These companion snapshots document the frozen run; this metadata-only commit does not alter the trained token order, masking, tokenizer, or checkpoint weights.",
        "",
    ))


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    plan = read_json(args.private_release_plan)
    require(plan.get("schema_version") == "apertus_full8_private_branch_release_plan_v1", "private branch plan schema drift")
    require(plan.get("status") == "ready_for_private_xfer_release" and plan.get("model_repo") == MODEL_REPO, "private branch plan is not the expected staging release")
    public = upload_receipt(args.public_train_upload_receipt, repo_id=PUBLIC_REPO, private=False)
    private = upload_receipt(args.private_full_mix_upload_receipt, repo_id=PRIVATE_REPO, private=True)
    output = args.output_root.resolve()
    require(not output.exists(), f"refusing to overwrite metadata release plan: {output}")
    branches = plan.get("branches")
    require(isinstance(branches, list) and len(branches) == 18 and len({str(row.get('branch')) for row in branches}) == 18, "private branch scope drift")
    rows: list[dict[str, Any]] = []
    for row in branches:
        branch = str(row["branch"])
        contract = read_json(Path(str(row["contract"]["path"])))
        source_card = Path(str(contract["sources"]["model_card"]))
        require(source_card.is_file(), f"missing frozen staging card: {branch}")
        inventory_path = args.private_release_plan.parent / "receipts" / branch / "hub_inventory.json"
        inventory = read_json(inventory_path)
        require(inventory.get("status") == "passed" and inventory.get("repo_id") == MODEL_REPO, f"branch lacks passed full Hub inventory: {branch}")
        base_revision = str(inventory.get("revision", ""))
        require(len(base_revision) == 40, f"branch inventory revision drift: {branch}")
        baseline_files = sorted(str(item["relative_path"]) for item in inventory.get("files", []) if item.get("relative_path") != ".gitattributes")
        require("README.md" in baseline_files and baseline_files, f"branch baseline inventory lacks README: {branch}")
        branch_root = output / "branches" / branch
        card_path = branch_root / "README.md"
        card_path.parent.mkdir(parents=True, exist_ok=True)
        card_path.write_text(pinned_card(source_card.read_text(encoding="utf-8"), public=public, private=private), encoding="utf-8")
        provenance_path = branch_root / "provenance" / "training_data.json"
        write_json(provenance_path, {
            "schema_version": "apertus_full8_checkpoint_training_data_provenance_v1",
            "status": "passed",
            "branch": branch,
            "base_weight_revision": base_revision,
            "datasets": {"public_modern_greek_train": public, "private_d0_full_mix": private},
            "metadata_only": True,
        })
        rows.append({
            "branch": branch,
            "base_revision": base_revision,
            "prior_full_inventory": {"path": str(inventory_path.resolve()), "sha256": sha256_file(inventory_path)},
            "baseline_payload_paths": baseline_files,
            "metadata_files": {
                "README.md": {"path": str(card_path.resolve()), "sha256": sha256_file(card_path)},
                "provenance/training_data.json": {"path": str(provenance_path.resolve()), "sha256": sha256_file(provenance_path)},
            },
        })
    result = {"schema_version": "apertus_full8_private_branch_dataset_binding_plan_v1", "status": "ready_for_metadata_only_commit", "model_repo": MODEL_REPO, "datasets": {"public_modern_greek_train": public, "private_d0_full_mix": private}, "branches": rows}
    write_json(output / "binding_plan.json", result)
    return result


def apply(args: argparse.Namespace) -> dict[str, Any]:
    token = os.environ.get("HF_TOKEN")
    require(bool(token), "HF_TOKEN must be injected per command")
    plan = read_json(args.binding_plan)
    require(plan.get("schema_version") == "apertus_full8_private_branch_dataset_binding_plan_v1" and plan.get("status") == "ready_for_metadata_only_commit", "binding plan is not ready")
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    output = args.output.resolve()
    require(not output.exists(), f"refusing to overwrite binding receipt: {output}")
    api = HfApi(token=token)
    committed: list[dict[str, Any]] = []
    for row in plan["branches"]:
        branch, base = str(row["branch"]), str(row["base_revision"])
        info = api.repo_info(repo_id=MODEL_REPO, repo_type="model", revision=branch, files_metadata=True)
        require(info.sha == base, f"branch moved since full-inventory verification: {branch}")
        local = row["metadata_files"]
        operations = [
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=local["README.md"]["path"]),
            CommitOperationAdd(path_in_repo="provenance/training_data.json", path_or_fileobj=local["provenance/training_data.json"]["path"]),
        ]
        commit = api.create_commit(repo_id=MODEL_REPO, repo_type="model", revision=branch, operations=operations, commit_message="Bind immutable CPT training-data provenance")
        revision = str(commit.oid)
        require(len(revision) == 40 and revision != base, f"metadata commit failed: {branch}")
        post = api.repo_info(repo_id=MODEL_REPO, repo_type="model", revision=revision, files_metadata=True)
        require(post.sha == revision and bool(post.private), f"post-commit visibility/revision drift: {branch}")
        observed = {str(item.rfilename) for item in post.siblings}
        expected = set(row["baseline_payload_paths"]) | {"provenance/training_data.json", ".gitattributes"}
        require(observed == expected, f"metadata commit changed branch inventory: {branch}; missing={sorted(expected-observed)}, extra={sorted(observed-expected)}")
        cache = args.cache_root / branch
        card = Path(hf_hub_download(repo_id=MODEL_REPO, repo_type="model", revision=revision, filename="README.md", token=token, local_dir=cache))
        provenance = Path(hf_hub_download(repo_id=MODEL_REPO, repo_type="model", revision=revision, filename="provenance/training_data.json", token=token, local_dir=cache))
        require(sha256_file(card) == local["README.md"]["sha256"], f"remote card hash drift: {branch}")
        require(sha256_file(provenance) == local["provenance/training_data.json"]["sha256"], f"remote provenance hash drift: {branch}")
        committed.append({"branch": branch, "base_weight_revision": base, "metadata_revision": revision, "metadata_only_paths": ["README.md", "provenance/training_data.json"], "prior_full_inventory": row["prior_full_inventory"]})
    result = {"schema_version": "apertus_full8_private_branch_dataset_binding_receipt_v1", "status": "completed", "model_repo": MODEL_REPO, "datasets": plan["datasets"], "branches": committed}
    write_json(output, result)
    print(json.dumps({"ok": True, "branches": len(committed)}, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("prepare")
    create.add_argument("--private-release-plan", type=Path, required=True)
    create.add_argument("--public-train-upload-receipt", type=Path, required=True)
    create.add_argument("--private-full-mix-upload-receipt", type=Path, required=True)
    create.add_argument("--output-root", type=Path, required=True)
    commit = commands.add_parser("apply")
    commit.add_argument("--binding-plan", type=Path, required=True)
    commit.add_argument("--cache-root", type=Path, required=True)
    commit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args) if args.command == "prepare" else apply(args)
    if args.command == "prepare":
        print(json.dumps({"ok": True, "branches": len(result["branches"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
