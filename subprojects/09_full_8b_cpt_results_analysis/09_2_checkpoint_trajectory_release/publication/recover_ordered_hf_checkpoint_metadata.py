#!/usr/bin/env python3
"""Finish an interrupted metadata-only ordered-ref release safely.

It requires the original pre-change dry-run receipt, re-verifies every ordered
ref's README and non-metadata model inventory before deleting any surviving
old alias, and records progress after every alias deletion.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from update_ordered_hf_checkpoint_metadata import DATA, REFS, REPO, card, model_inventory, point_map, require, sha256_bytes, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dry-run", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    require(os.environ.get("HF_TOKEN"), "HF_TOKEN must be supplied per command")
    from huggingface_hub import HfApi, hf_hub_download

    source = json.loads(args.source_dry_run.read_text())
    require(source.get("status") == "dry_run_passed" and source.get("repo") == REPO, "source receipt is not the expected dry run")
    baseline = source.get("old_refs")
    require(isinstance(baseline, dict) and set(baseline) == {ref.old for ref in REFS}, "source receipt ref coverage drift")
    data = json.loads(DATA.read_text())
    native, greek = point_map(data)
    expected_cards = {ref.new: card(ref, native[ref.iteration], greek[ref.iteration]).encode() for ref in REFS}
    api = HfApi(token=os.environ["HF_TOKEN"])
    refs = {str(row.name): str(row.target_commit) for row in api.list_repo_refs(REPO, repo_type="model").branches}
    require({ref.new for ref in REFS} <= set(refs), "ordered-ref coverage is incomplete")
    verified = []
    for ref in REFS:
        expected_inventory = {k: tuple(v) for k, v in baseline[ref.old]["model_inventory"].items()}
        require(model_inventory(api, ref.new) == expected_inventory, f"model inventory drift: {ref.new}")
        downloaded = Path(hf_hub_download(repo_id=REPO, repo_type="model", revision=ref.new, filename="README.md", token=os.environ["HF_TOKEN"], force_download=True))
        observed = downloaded.read_bytes()
        require(observed == expected_cards[ref.new], f"README content drift: {ref.new}")
        if ref.old in refs:
            require(refs[ref.old] == refs[ref.new], f"old/new target mismatch: {ref.old}")
        verified.append({"old": ref.old, "new": ref.new, "commit": refs[ref.new], "readme_sha256": sha256_bytes(observed)})
    index_path = Path(hf_hub_download(repo_id=REPO, repo_type="model", revision="main", filename="checkpoint-index.json", token=os.environ["HF_TOKEN"], force_download=True))
    index = json.loads(index_path.read_text())
    require(index.get("ordering") == "zero_padded_release_ordinal_ascending", "index ordering drift")
    require([row.get("revision") for row in index.get("checkpoints", [])] == [ref.new for ref in REFS], "index revision ordering drift")
    remaining = [ref.old for ref in REFS if ref.old != "main" and ref.old in refs]
    plan = {"schema_version": "apertus_ordered_hf_checkpoint_metadata_recovery_v1", "status": "recovery_dry_run_passed" if not args.apply else "recovery_in_progress", "repo": REPO, "verified": verified, "remaining_old_aliases": remaining, "index_sha256": sha256_bytes(index_path.read_bytes())}
    write_json(args.receipt, plan)
    if not args.apply:
        print(json.dumps({"ok": True, "status": plan["status"], "remaining_old_aliases": len(remaining)}, sort_keys=True))
        return 0
    deleted = []
    for name in remaining:
        api.delete_branch(REPO, branch=name, repo_type="model")
        deleted.append(name)
        write_json(args.receipt, {**plan, "deleted_old_aliases": deleted})
    final_refs = {str(row.name): str(row.target_commit) for row in api.list_repo_refs(REPO, repo_type="model").branches}
    require({ref.new for ref in REFS} <= set(final_refs) and "main" in final_refs, "final ordered-ref coverage drift")
    require(not ({ref.old for ref in REFS if ref.old != "main"} & set(final_refs)), "old aliases remain after recovery")
    write_json(args.receipt, {**plan, "status": "completed", "deleted_old_aliases": deleted, "final_refs": final_refs})
    print(json.dumps({"ok": True, "status": "completed", "deleted_old_aliases": len(deleted)}, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
