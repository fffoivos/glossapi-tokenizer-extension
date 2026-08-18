#!/usr/bin/env python3
"""Stage the exact full-8B D0 packed mixture as a portable private dataset.

The source stage is never modified.  This adapter follows the completed
packing receipt, hard-links exactly the training-reader payload and its
provenance into a new immutable directory, then writes a manifest.  Payload
hash verification is a separate explicit pass so a long hash sweep cannot be
mistaken for a lightweight preparation step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


EXPECTED_POOLS = {"hplt_new_greek", "non_hplt_new_greek", "foreign_replay", "old_greek_replay"}
EXPECTED_ACTIVE = {
    "hplt_new_greek": 41_512_804_679,
    "non_hplt_new_greek": 19_068_732_797,
    "foreign_replay": 15_337_098_095,
    "old_greek_replay": 766_854_905,
}


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
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def portable_relative(stage: Path, path: Path) -> str:
    resolved = path.resolve()
    require(stage == resolved or stage in resolved.parents, f"payload escapes source stage: {path}")
    return resolved.relative_to(stage).as_posix()


def source_file(stage: Path, value: str) -> Path:
    path = Path(value).resolve()
    portable_relative(stage, path)
    require(path.is_file(), f"source file is missing: {path}")
    return path


def add_bound_file(files: dict[str, dict[str, Any]], stage: Path, path: Path, *, expected_sha256: str | None = None, expected_bytes: int | None = None, hash_origin: str) -> None:
    relative = portable_relative(stage, path)
    size = path.stat().st_size
    if expected_bytes is not None:
        require(size == expected_bytes, f"source byte size drift: {relative}")
    if expected_sha256 is not None:
        require(len(expected_sha256) == 64, f"invalid expected checksum: {relative}")
    row = {"relative_path": relative, "bytes": size, "sha256": expected_sha256, "hash_origin": hash_origin}
    prior = files.get(relative)
    require(prior is None or prior == row, f"conflicting payload binding: {relative}")
    files[relative] = row


def payload_inventory(stage: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    packed_path = stage / "inventory" / "packed_corpus_receipt.json"
    packed = read_json(packed_path)
    require(packed.get("schema_version") == "apertus_packed_sequence_corpus_v1", "packed-corpus schema drift")
    require(packed.get("status") == "completed", "packed-corpus receipt is not completed")
    pools = packed.get("pools")
    require(isinstance(pools, dict) and set(pools) == EXPECTED_POOLS, "D0 pool set drift")
    observed_active = {name: int(row["active_tokens"]) for name, row in pools.items()}
    require(observed_active == EXPECTED_ACTIVE, f"D0 active-token accounting drift: {observed_active}")
    files: dict[str, dict[str, Any]] = {}
    add_bound_file(files, stage, packed_path, hash_origin="release_finalizer")
    plan = packed.get("packing_plan")
    require(isinstance(plan, dict), "packed-corpus lacks packing plan")
    add_bound_file(files, stage, source_file(stage, str(plan["path"])), expected_sha256=str(plan["sha256"]), hash_origin="packing_plan_receipt")
    for row in packed.get("packing_task_manifests", []):
        manifest_path = source_file(stage, str(row["manifest_path"]))
        add_bound_file(files, stage, manifest_path, expected_sha256=str(row["manifest_sha256"]), hash_origin="packed_corpus_receipt")
        manifest = read_json(manifest_path)
        for binding in manifest.get("outputs", {}).values():
            add_bound_file(files, stage, source_file(stage, str(binding["path"])), expected_sha256=str(binding["sha256"]), expected_bytes=int(binding["bytes"]), hash_origin="canonical_packer_write_time_sha256")
    for pool in pools.values():
        catalog = pool.get("sequence_catalog")
        require(isinstance(catalog, dict), "pool lacks sequence catalog")
        add_bound_file(files, stage, source_file(stage, str(catalog["path"])), expected_sha256=str(catalog["sha256"]), expected_bytes=int(catalog["bytes"]), hash_origin="packed_corpus_receipt")

    schedule_path = stage / "schedules" / "schedule_manifest.json"
    schedule = read_json(schedule_path)
    require(schedule.get("status") == "completed", "schedule manifest is not completed")
    arms = schedule.get("arms")
    require(isinstance(arms, list), "schedule manifest has no arm list")
    d0_arms = [row for row in arms if isinstance(row, dict) and row.get("arm_id") == "D0_mixed"]
    require(len(d0_arms) == 1, "schedule manifest does not contain exactly one D0 arm")
    arm = d0_arms[0]
    require({key: int(value) for key, value in arm["pool_active_tokens"].items()} == {"H": EXPECTED_ACTIVE["hplt_new_greek"], "G": EXPECTED_ACTIVE["non_hplt_new_greek"], "F": EXPECTED_ACTIVE["foreign_replay"], "O": EXPECTED_ACTIVE["old_greek_replay"]}, "D0 schedule active tokens drift")
    add_bound_file(files, stage, schedule_path, hash_origin="release_finalizer")
    for key in ("active_tokens", "sequence_ids"):
        binding = arm[key]
        add_bound_file(files, stage, source_file(stage, str(binding["path"])), expected_sha256=str(binding["sha256"]), expected_bytes=int(binding["bytes"]), hash_origin="schedule_manifest")

    for relative in (
        "contracts/recipe_8b_full_mixed.sanitized.json",
        "contracts/execution_profiles.sanitized.json",
        "evidence/selected_training_content/selected_training_content_receipt.json",
        "evidence/selected_training_content/selected_training_content.sorted.unique.sha32",
        "evidence/selected_training_content/foreign_replay.selected.sha32",
        "evidence/selected_training_content/old_greek_replay.selected.sha32",
        "inventory/raw/modern.content57",
        "inventory/catalog/hplt_new_greek.source_local_selected.catalog45",
        "inventory/catalog/non_hplt_new_greek.source_local_selected.catalog45",
        "inventory/catalog/foreign_replay.source_local_selected.catalog45",
        "inventory/catalog/old_greek_replay.source_local_selected.catalog45",
    ):
        add_bound_file(files, stage, stage / relative, hash_origin="release_finalizer")
    return files, {"packed": packed, "schedule": schedule, "active_tokens": observed_active}


def hardlink_tree(stage: Path, output: Path, files: dict[str, dict[str, Any]]) -> None:
    for relative in sorted(files):
        source = stage / relative
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError as error:
            raise RuntimeError(f"hard-link failed for {relative}; source and output must share a filesystem") from error
        require(source.stat().st_ino == target.stat().st_ino, f"hard-link identity mismatch: {relative}")


def readme(active_tokens: dict[str, int]) -> str:
    total = sum(active_tokens.values())
    return "\n".join((
        "# Exact Apertus 8B D0 full training mixture (private)", "",
        "This private portable dataset is the exact packed source-local D0 training payload used for the full 8B Greek CPT run. It is not a general redistribution artifact and does not grant redistribution rights for replay sources.", "",
        "## Mix", "",
        f"- HPLT Modern Greek: {active_tokens['hplt_new_greek']:,} active tokens",
        f"- GlossAPI/non-HPLT Modern Greek: {active_tokens['non_hplt_new_greek']:,} active tokens",
        f"- Foreign replay: {active_tokens['foreign_replay']:,} active tokens",
        f"- Old-Greek replay: {active_tokens['old_greek_replay']:,} active tokens",
        f"- Total active tokens: {total:,}", "",
        "The manifest binds the packed corpus, source-local catalogs, D0 schedule, training recipe, and post-mask selected-content evidence. Every payload path is portable and relative to this dataset root. A separate verification pass records a fresh SHA-256 for every payload before upload.", "",
    ))


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    stage = args.source_stage.resolve()
    output = args.output_stage.resolve()
    require(stage.is_dir(), f"source stage missing: {stage}")
    require(not output.exists(), f"refusing to overwrite immutable output: {output}")
    files, facts = payload_inventory(stage)
    output.mkdir(parents=True)
    hardlink_tree(stage, output, files)
    (output / "README.md").write_text(readme(facts["active_tokens"]), encoding="utf-8")
    readme_path = output / "README.md"
    manifest = {"schema_version": "apertus_full8_d0_private_portable_dataset_v1", "status": "prepared_unverified_payload_hashes", "visibility": "private", "redistribution": "not_authorized", "source_stage": str(stage), "training_identity": {"arm": "D0_mixed", "active_tokens": facts["active_tokens"], "total_active_tokens": sum(facts["active_tokens"].values()), "sequence_count": int(facts["packed"]["global"]["sequence_count"])}, "upload_payload_inventory": [{"relative_path": "README.md", "bytes": readme_path.stat().st_size, "sha256": sha256_file(readme_path), "hash_origin": "release_finalizer"}, *[files[key] for key in sorted(files)]], "hash_verification": {"required_before_upload": True, "receipt": "payload_sha256_verification.json"}}
    write_json(output / "manifest.json", manifest)
    return manifest


def verify(args: argparse.Namespace) -> dict[str, Any]:
    stage = args.output_stage.resolve()
    manifest_path = stage / "manifest.json"
    manifest = read_json(manifest_path)
    require(manifest.get("schema_version") == "apertus_full8_d0_private_portable_dataset_v1", "private stage schema drift")
    require(manifest.get("status") in {"prepared_unverified_payload_hashes", "verified_payload_hashes"}, "private stage status drift")
    rows = manifest.get("upload_payload_inventory")
    require(isinstance(rows, list) and rows, "private stage has no payload inventory")
    verified: list[dict[str, Any]] = []
    for row in rows:
        relative = Path(str(row["relative_path"]))
        require(not relative.is_absolute() and ".." not in relative.parts, "nonportable payload path")
        path = stage / relative
        require(path.is_file() and path.stat().st_size == int(row["bytes"]), f"payload size drift: {relative}")
        observed = sha256_file(path)
        expected = row.get("sha256")
        if expected is not None:
            require(observed == expected, f"payload checksum drift: {relative}")
        verified.append({"relative_path": relative.as_posix(), "bytes": path.stat().st_size, "sha256": observed, "prior_hash_origin": row.get("hash_origin")})
    receipt = {"schema_version": "apertus_full8_d0_private_payload_hash_verification_v1", "status": "passed", "manifest_sha256": sha256_file(manifest_path), "files": verified}
    write_json(args.output, receipt)
    manifest["status"] = "verified_payload_hashes"
    manifest["hash_verification"] = {"required_before_upload": True, "receipt": str(args.output.resolve()), "sha256": sha256_file(args.output)}
    write_json(manifest_path, manifest)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--source-stage", type=Path, required=True)
    prepare_parser.add_argument("--output-stage", type=Path, required=True)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--output-stage", type=Path, required=True)
    verify_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args) if args.command == "prepare" else verify(args)
    print(json.dumps({"ok": True, "status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
