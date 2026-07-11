#!/usr/bin/env python3
"""Freeze every input needed to rebuild the three full-corpus Megatron pools.

This is deliberately an exact, expensive receipt pass.  It accepts only a
passed Phase-04 local-release validation, hashes every private training shard
and replay source shard, checks the tokenizer identity used by Phase-04, and
emits a deterministic task plan.  Later array jobs consume this immutable
receipt rather than re-resolving globs or dataset names.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from bridge_common import (
    HEX_COMMIT,
    canonical_sha256,
    file_tree_receipt,
    read_json,
    resolve_relative,
    safe_name,
    sha256_file,
    tokenizer_tree_receipt,
    utc_now,
    write_json_atomic,
)


EXPECTED_RELEASE_SCHEMA = "full_cpt_release_manifest_v1"
EXPECTED_VALIDATION_SCHEMA = "full_cpt_release_validation_v1"
EXPECTED_INTEGRITY = "full_cpt_release_integrity_v1"
EXPECTED_DECONTAM_SCHEMA = "full_cpt_greekmmlu_decontamination_v1"
EXPECTED_DECONTAM_POLICY = "greekmmlu_decontamination_v1"
HISTORICAL_SCRATCH = "/iopsstor/scratch/cscs/fffoivos"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _parquet_receipt(path: Path, *, relative: str) -> dict[str, Any]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    return {
        "path": str(path.resolve()),
        "relative_path": relative,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
        "columns": parquet.schema_arrow.names,
    }


def _assert_receipt(
    actual: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    for field in ("sha256", "bytes", "rows", "row_groups"):
        if actual.get(field) != expected.get(field):
            raise ValueError(f"{label}: {field} differs from the upstream receipt")


def validate_phase04(
    stage: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    manifest_path = (stage / "release_manifest.json").resolve()
    validation_path = (stage / "validation.json").resolve()
    manifest = read_json(manifest_path)
    validation = read_json(validation_path)
    if manifest.get("schema_version") != EXPECTED_RELEASE_SCHEMA:
        raise ValueError("Phase-04 release manifest schema is unsupported")
    if manifest.get("integrity_contract_version") != EXPECTED_INTEGRITY:
        raise ValueError("Phase-04 release integrity contract is unsupported")
    if validation.get("schema_version") != EXPECTED_VALIDATION_SCHEMA:
        raise ValueError("Phase-04 validation schema is unsupported")
    if validation.get("integrity_contract_version") != EXPECTED_INTEGRITY:
        raise ValueError("Phase-04 validation integrity contract is unsupported")
    if validation.get("status") != "passed" or validation.get("failed_checks") != []:
        raise ValueError("Phase-04 local release has not passed validation")
    manifest_sha = sha256_file(manifest_path)
    if validation.get("release_manifest_sha256") != manifest_sha:
        raise ValueError(
            "Phase-04 validation is bound to different release-manifest bytes"
        )
    if Path(str(validation.get("release_manifest", ""))).resolve() != manifest_path:
        raise ValueError("Phase-04 validation points at a different release manifest")
    release_root = (stage / "release").resolve()
    if Path(str(manifest.get("output", ""))).resolve() != release_root:
        raise ValueError(
            "Phase-04 release manifest output is not the selected local release"
        )
    if Path(str(validation.get("release", ""))).resolve() != release_root:
        raise ValueError("Phase-04 validation points at a different local release")

    expected: dict[str, Mapping[str, Any]] = {}
    for entry in manifest.get("files", []):
        receipt = entry.get("training") if isinstance(entry, dict) else None
        if not isinstance(receipt, dict):
            raise ValueError("Phase-04 release has a malformed training receipt")
        relative = str(receipt.get("path", ""))
        path = Path(relative)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not relative.startswith("training/data/")
        ):
            raise ValueError(f"unsafe Phase-04 training receipt path: {relative!r}")
        if relative in expected:
            raise ValueError(f"duplicate Phase-04 training receipt: {relative}")
        expected[relative] = receipt
    actual_paths = sorted(
        path
        for path in (release_root / "training" / "data").rglob("*.parquet")
        if path.is_file()
    )
    actual_relatives = {
        path.relative_to(release_root).as_posix() for path in actual_paths
    }
    if actual_relatives != set(expected):
        raise ValueError(
            "Phase-04 private training inventory differs from the release manifest"
        )
    receipts: list[dict[str, Any]] = []
    for path in actual_paths:
        relative = path.relative_to(release_root).as_posix()
        actual = _parquet_receipt(path, relative=relative)
        _assert_receipt(actual, expected[relative], label=relative)
        receipts.append(actual)
    if sum(row["rows"] for row in receipts) != int(
        manifest.get("counts", {}).get("training_rows", -1)
    ):
        raise ValueError("Phase-04 private training row accounting does not reconcile")
    return manifest, validation, receipts


def resolve_decontamination(release_manifest: Mapping[str, Any]) -> dict[str, Any]:
    upstream = release_manifest.get("upstream_manifests", {}).get("decontamination", {})
    path = Path(str(upstream.get("path", ""))).resolve()
    if not path.is_file() or sha256_file(path) != upstream.get("sha256"):
        raise ValueError("Phase-04 decontamination manifest receipt drift")
    value = read_json(path)
    if (
        value.get("schema_version") != EXPECTED_DECONTAM_SCHEMA
        or value.get("status") != "completed"
    ):
        raise ValueError("Phase-04 decontamination manifest is not completed")
    policy = value.get("policy", {})
    if policy.get("policy_version") != EXPECTED_DECONTAM_POLICY:
        raise ValueError("Phase-04 decontamination policy is not the production policy")
    benchmark = value.get("benchmark", {})
    queries = Path(str(benchmark.get("queries_path", ""))).resolve()
    benchmark_manifest = Path(str(benchmark.get("manifest_path", ""))).resolve()
    if not queries.is_file() or sha256_file(queries) != benchmark.get("queries_sha256"):
        raise ValueError("Phase-04 GreekMMLU query bytes drift")
    if not benchmark_manifest.is_file() or sha256_file(
        benchmark_manifest
    ) != benchmark.get("manifest_sha256"):
        raise ValueError("Phase-04 GreekMMLU benchmark-manifest bytes drift")
    return {
        "manifest": {"path": str(path), "sha256": sha256_file(path)},
        "queries": {"path": str(queries), "sha256": sha256_file(queries)},
        "benchmark_manifest": {
            "path": str(benchmark_manifest),
            "sha256": sha256_file(benchmark_manifest),
        },
        "policy": policy,
    }


def validate_tokenizer(
    config: Mapping[str, Any], release_manifest: Mapping[str, Any], tokenizer_dir: Path
) -> dict[str, Any]:
    expected = config["tokenizer"]
    if not HEX_COMMIT.fullmatch(str(expected.get("revision", ""))):
        raise ValueError("tokenizer revision must be a full immutable commit")
    tokenizer_json = (tokenizer_dir / "tokenizer.json").resolve()
    if not tokenizer_json.is_file():
        raise FileNotFoundError(tokenizer_json)
    actual_sha = sha256_file(tokenizer_json)
    if actual_sha != expected.get("tokenizer_json_sha256"):
        raise ValueError("local training tokenizer differs from the frozen recipe")
    cleaning_receipt = release_manifest.get("upstream_manifests", {}).get(
        "cleaning", {}
    )
    cleaning_path = Path(str(cleaning_receipt.get("path", ""))).resolve()
    if not cleaning_path.is_file() or sha256_file(
        cleaning_path
    ) != cleaning_receipt.get("sha256"):
        raise ValueError("Phase-04 cleaning manifest receipt drift")
    cleaning = read_json(cleaning_path)
    if cleaning.get("tokenizer_sha256") != actual_sha:
        raise ValueError(
            "training tokenizer differs from the tokenizer bound by Phase-04 cleaning"
        )
    from tokenizers import Tokenizer

    backend = Tokenizer.from_file(str(tokenizer_json))
    vocab_size = backend.get_vocab_size(with_added_tokens=True)
    if vocab_size != int(expected.get("vocab_size", -1)):
        raise ValueError(
            "training tokenizer vocabulary size differs from the frozen recipe"
        )
    tokenizer_config_path = tokenizer_dir / "tokenizer_config.json"
    if not tokenizer_config_path.is_file():
        raise FileNotFoundError(
            "tokenizer_config.json is required to bind the EOD token"
        )
    tokenizer_config = read_json(tokenizer_config_path)
    eos_value = tokenizer_config.get("eos_token")
    eos_token = eos_value.get("content") if isinstance(eos_value, dict) else eos_value
    if not isinstance(eos_token, str) or not eos_token:
        raise ValueError("tokenizer_config.json has no usable eos_token")
    eos_id = backend.token_to_id(eos_token)
    if eos_id is None or eos_id < 0 or eos_id >= vocab_size:
        raise ValueError("tokenizer EOD/EOS identity is invalid")
    tree = tokenizer_tree_receipt(tokenizer_dir)
    return {
        **expected,
        "root": str(tokenizer_dir.resolve()),
        "tokenizer_json": str(tokenizer_json),
        "tree_sha256": tree["tree_sha256"],
        "files": tree["files"],
        "eos_token": eos_token,
        "eos_token_id": eos_id,
        "phase04_cleaning_manifest": {
            "path": str(cleaning_path),
            "sha256": sha256_file(cleaning_path),
        },
    }


def _source_paths(raw: str, scratch_root: Path) -> list[Path]:
    rendered = raw.format(scratch_root=str(scratch_root.resolve()))
    rendered = rendered.replace(HISTORICAL_SCRATCH, str(scratch_root.resolve()))
    paths = sorted(
        Path(value).resolve() for value in glob.glob(rendered, recursive=True)
    )
    paths = [path for path in paths if path.is_file() and path.suffix == ".parquet"]
    if not paths:
        raise FileNotFoundError(f"source glob matched no local Parquet: {rendered}")
    return paths


def _choose_id_column(columns: list[str], explicit: str | None) -> str:
    if explicit:
        if explicit not in columns:
            raise ValueError(f"configured id column {explicit!r} is absent")
        return explicit
    for candidate in ("doc_id", "id", "source_doc_id", "source_row_id"):
        if candidate in columns:
            return candidate
    return ""


def _task(
    *,
    index: int,
    pool: str,
    source_name: str,
    source_weight: str,
    path: Path,
    receipt: Mapping[str, Any],
    text_column: str,
    identity_columns: list[str],
    identity_scope: str,
    filter_field: str | None,
    filter_min: float | None,
    decontaminate: bool,
    exclusion_file: str | None,
    exclusion_key: str | None,
) -> dict[str, Any]:
    return {
        "task_index": index,
        "task_id": f"{index:05d}-{safe_name(pool)}-{safe_name(source_name)}",
        "kind": "training",
        "pool": pool,
        "source_name": source_name,
        "source_weight_within_pool": source_weight,
        "input_path": str(path.resolve()),
        "input_relative": str(receipt["relative_path"]),
        "input_sha256": receipt["sha256"],
        "input_bytes": receipt["bytes"],
        "input_rows": receipt["rows"],
        "text_column": text_column,
        "identity_columns": identity_columns,
        "identity_scope": identity_scope,
        "filter_field": filter_field or "",
        "filter_min": filter_min,
        "decontaminate_greekmmlu": decontaminate,
        "exclusion_file": exclusion_file or "",
        "exclusion_key": exclusion_key or "",
        "requires_heldout_exclusion": bool(exclusion_file),
        "output_prefix": f"train/{safe_name(pool)}/{safe_name(source_name)}/{index:05d}_text_document",
    }


def resolve_replay_sources(
    config_path: Path, config: Mapping[str, Any], scratch_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    foreign_config = config["foreign_replay"]
    legacy_path = resolve_relative(config_path, str(foreign_config["legacy_recipe"]))
    if sha256_file(legacy_path) != foreign_config.get("legacy_recipe_sha256"):
        raise ValueError("frozen pilot recipe bytes have drifted")
    legacy = read_json(legacy_path)
    include = set(foreign_config["include_buckets"])
    replacements = foreign_config.get("source_replacements", {})
    overrides = foreign_config.get("path_overrides", {})
    heldout_sources = {
        str(row["source_name"]) for row in config["heldouts"]["foreign_replay"]
    }
    specs: list[dict[str, Any]] = []
    for original in legacy.get("sources", []):
        if original.get("bucket") not in include:
            continue
        spec = dict(original)
        if spec["name"] in replacements:
            replacement = dict(replacements[spec["name"]])
            replacement["weight"] = spec["weight"]
            spec = replacement
        if spec["name"] in overrides:
            spec["local_parquet"] = overrides[spec["name"]]
        if spec.get("fallback_id"):
            spec["id"] = spec["fallback_id"]
            spec["config"] = spec.get("fallback_config")
        if not spec.get("local_parquet"):
            raise ValueError(
                f"foreign source is not locally materialized: {spec['name']}"
            )
        if (
            foreign_config.get("expand_local_recipe_file_to_sibling_parquets")
            and spec["name"] not in overrides
            and not any(character in str(spec["local_parquet"]) for character in "*?[")
        ):
            spec["local_parquet"] = str(
                Path(str(spec["local_parquet"])).with_name("*.parquet")
            )
        specs.append(spec)
    if not specs:
        raise ValueError("frozen foreign replay recipe resolved no sources")
    total_weight = sum(float(spec["weight"]) for spec in specs)
    if total_weight <= 0:
        raise ValueError("foreign replay weights are not positive")

    tasks: list[dict[str, Any]] = []
    inventories: list[dict[str, Any]] = []
    for spec in specs:
        source_name = str(spec["name"])
        paths = _source_paths(str(spec["local_parquet"]), scratch_root)
        file_receipts: list[dict[str, Any]] = []
        for path in paths:
            try:
                source_relative = path.relative_to(scratch_root.resolve()).as_posix()
            except ValueError:
                source_relative = f"external/{sha256_file(path)[:16]}/{path.name}"
            receipt = _parquet_receipt(
                path, relative=f"foreign/{source_name}/{source_relative}"
            )
            if str(spec["text_column"]) not in receipt["columns"]:
                raise ValueError(f"{source_name}: text column is absent from {path}")
            id_column = _choose_id_column(
                receipt["columns"], spec.get("id_column") or spec.get("doc_key_field")
            )
            identity_columns = [
                str(value)
                for value in spec.get("identity_columns", [id_column])
                if value
            ]
            if any(value not in receipt["columns"] for value in identity_columns):
                raise ValueError(
                    f"{source_name}: identity columns are absent from {path}: "
                    f"{identity_columns}"
                )
            identity_scope = str(spec.get("identity_scope", "file"))
            if identity_scope not in {"file", "global"}:
                raise ValueError(f"{source_name}: unsupported identity scope")
            if (
                spec.get("filter_field")
                and spec["filter_field"] not in receipt["columns"]
            ):
                raise ValueError(f"{source_name}: filter field is absent from {path}")
            file_receipts.append(receipt)
            tasks.append(
                _task(
                    index=-1,
                    pool="foreign_replay",
                    source_name=source_name,
                    source_weight=format(float(spec["weight"]) / total_weight, ".17g"),
                    path=path,
                    receipt=receipt,
                    text_column=str(spec["text_column"]),
                    identity_columns=identity_columns,
                    identity_scope=identity_scope,
                    filter_field=spec.get("filter_field"),
                    filter_min=float(spec["filter_min"])
                    if spec.get("filter_min") is not None
                    else None,
                    decontaminate=bool(foreign_config["decontaminate_greekmmlu"]),
                    exclusion_file=(
                        f"heldouts/exclusions/{safe_name(source_name)}.jsonl"
                        if source_name in heldout_sources
                        else None
                    ),
                    exclusion_key=source_name
                    if source_name in heldout_sources
                    else None,
                )
            )
        inventories.append(
            {
                "source_name": source_name,
                "bucket": spec["bucket"],
                "repo_id": spec.get("repo_id") or spec.get("id"),
                "config": spec.get("config"),
                "weight_within_foreign": format(
                    float(spec["weight"]) / total_weight, ".17g"
                ),
                "text_column": spec["text_column"],
                "files": file_receipts,
            }
        )
    return (
        tasks,
        inventories,
        {
            "path": str(legacy_path),
            "sha256": sha256_file(legacy_path),
            "name": legacy.get("name"),
            "version": legacy.get("version"),
            "derivation": "replay+code+math weights renormalized within the 20% foreign pool; CodeParrot replaced by StarCoderData at the same code-bucket weight",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase04-stage", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--megatron-dir", type=Path, required=True)
    parser.add_argument("--replay-acquisition-receipt", type=Path, required=True)
    parser.add_argument("--old-greek-build-receipt", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-dirty-repo", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = read_json(config_path)
    if config.get("schema_version") != "full_cpt_training_bridge_config_v1":
        raise ValueError("unsupported training-bridge config")
    release_manifest, validation, release_files = validate_phase04(
        args.phase04_stage.resolve()
    )
    decontamination = resolve_decontamination(release_manifest)
    tokenizer = validate_tokenizer(
        config, release_manifest, args.tokenizer_dir.resolve()
    )

    repo_commit = _git(args.repo_root.resolve(), "rev-parse", "HEAD")
    if not HEX_COMMIT.fullmatch(repo_commit):
        raise ValueError("repository HEAD is not a full Git commit")
    if not args.allow_dirty_repo and _git(
        args.repo_root.resolve(), "status", "--porcelain"
    ):
        raise ValueError(
            "refusing to freeze a dirty repository; commit the bridge first"
        )
    megatron_commit = _git(args.megatron_dir.resolve(), "rev-parse", "HEAD")
    expected_megatron = config["builder"]["expected_megatron_commit"]
    if megatron_commit != expected_megatron:
        raise ValueError(
            f"Megatron commit drift: {megatron_commit} != {expected_megatron}"
        )
    if _git(
        args.megatron_dir.resolve(), "status", "--porcelain", "--untracked-files=all"
    ):
        raise ValueError("refusing a dirty or untracked Megatron training tree")
    megatron_tree = file_tree_receipt(
        args.megatron_dir.resolve(), exclude_top_level=(".git",)
    )

    foreign_tasks, foreign_inventory, legacy_recipe = resolve_replay_sources(
        config_path, config, args.scratch_root.resolve()
    )
    old = config["old_greek_replay"]
    old_paths = _source_paths(str(old["local_parquet"]), args.scratch_root.resolve())
    old_inventory: list[dict[str, Any]] = []
    old_tasks: list[dict[str, Any]] = []
    for path in old_paths:
        receipt = _parquet_receipt(path, relative=f"old_greek/{path.name}")
        if (
            old["text_column"] not in receipt["columns"]
            or old["id_column"] not in receipt["columns"]
            or any(
                value not in receipt["columns"]
                for value in old.get("identity_columns", [old["id_column"]])
            )
        ):
            raise ValueError(f"old-Greek replay schema drift: {path}")
        old_inventory.append(receipt)
        old_tasks.append(
            _task(
                index=-1,
                pool="old_greek_replay",
                source_name=str(old["name"]),
                source_weight="1",
                path=path,
                receipt=receipt,
                text_column=str(old["text_column"]),
                identity_columns=[
                    str(value)
                    for value in old.get("identity_columns", [old["id_column"]])
                ],
                identity_scope=str(old.get("identity_scope", "file")),
                filter_field=None,
                filter_min=None,
                decontaminate=bool(old["decontaminate_greekmmlu"]),
                exclusion_file=f"heldouts/exclusions/{safe_name(str(old['name']))}.jsonl",
                exclusion_key=str(old["name"]),
            )
        )

    new_tasks: list[dict[str, Any]] = []
    for receipt in release_files:
        path = Path(receipt["path"])
        required = {"text", "stable_uid", "source_dataset", "source_family_id"}
        if missing := required - set(receipt["columns"]):
            raise ValueError(
                f"Phase-04 training shard misses bridge columns {sorted(missing)}: {path}"
            )
        new_tasks.append(
            _task(
                index=-1,
                pool="new_greek",
                source_name="phase04_release",
                source_weight="1",
                path=path,
                receipt=receipt,
                text_column="text",
                identity_columns=["stable_uid"],
                identity_scope="global",
                filter_field=None,
                filter_min=None,
                decontaminate=False,
                exclusion_file="heldouts/exclusions/new_greek.jsonl",
                exclusion_key="new_greek",
            )
        )

    tasks = new_tasks + foreign_tasks + old_tasks
    for index, task in enumerate(tasks):
        task["task_index"] = index
        task["task_id"] = (
            f"{index:05d}-{safe_name(task['pool'])}-{safe_name(task['source_name'])}"
        )
        task["output_prefix"] = (
            f"train/{safe_name(task['pool'])}/{safe_name(task['source_name'])}/{index:05d}_text_document"
        )
    if len(tasks) != len({task["task_id"] for task in tasks}):
        raise ValueError("training task identities are not unique")

    bridge_dir = Path(__file__).resolve().parents[1]
    code_files = [
        bridge_dir / "scripts" / name
        for name in (
            "bridge_common.py",
            "freeze_inputs.py",
            "build_heldouts.py",
            "build_binary_shard.py",
            "finalize_bridge.py",
            "freeze_training_assets.py",
            "freeze_resume_checkpoint.py",
            "verify_launch_assets.py",
        )
    ]
    decontam_script = (
        bridge_dir.parent
        / "04_full_corpus_preparation"
        / "scripts"
        / "decontaminate_full_corpus.py"
    )
    code_files.append(decontam_script)
    for path in code_files:
        if not path.is_file():
            raise FileNotFoundError(f"bridge implementation is incomplete: {path}")
    code_receipts = [
        {"path": str(path), "sha256": sha256_file(path)} for path in code_files
    ]
    config_sha = sha256_file(config_path)
    mix = config["probe"]["mix_numerators"]
    denominator = int(config["probe"]["mix_denominator"])
    if (
        set(mix) != {"new_greek", "foreign_replay", "old_greek_replay"}
        or sum(mix.values()) != denominator
    ):
        raise ValueError("top-level mix must be an exact 79/20/1 partition")
    if mix != {"new_greek": 79, "foreign_replay": 20, "old_greek_replay": 1}:
        raise ValueError("frozen top-level mix drift")
    probe = config["probe"]
    effective_steps = int(probe["nominal_tokens"]) // (
        int(probe["sequence_length"]) * int(probe["global_batch_sequences"])
    )
    effective_samples = effective_steps * int(probe["global_batch_sequences"])
    effective_tokens = effective_samples * int(probe["sequence_length"])
    if (
        effective_steps != int(probe["effective_steps"])
        or effective_samples != int(probe["effective_training_samples"])
        or effective_tokens != int(probe["effective_training_tokens"])
        or int(probe["nominal_tokens"]) - effective_tokens
        != int(probe["nominal_token_floor_residual"])
    ):
        raise ValueError("effective 5,960-step probe accounting drift")

    acquisition_path = args.replay_acquisition_receipt.resolve()
    acquisition = read_json(acquisition_path)
    if (
        acquisition.get("schema_version") != "full_cpt_replay_acquisition_receipt_v1"
        or acquisition.get("status") != "completed"
    ):
        raise ValueError("replay acquisition is not completed")
    acquisition_impl = Path(str(acquisition.get("implementation", ""))).resolve()
    acquisition_config = Path(str(acquisition.get("config", ""))).resolve()
    if (
        not acquisition_impl.is_file()
        or sha256_file(acquisition_impl) != acquisition.get("implementation_sha256")
        or not acquisition_config.is_file()
        or sha256_file(acquisition_config) != acquisition.get("config_sha256")
    ):
        raise ValueError("replay acquisition implementation/config receipt drift")
    phase04_pin_receipt = acquisition.get("phase04_sources_config", {})
    phase04_pin_path = Path(str(phase04_pin_receipt.get("path", ""))).resolve()
    if not phase04_pin_path.is_file() or sha256_file(
        phase04_pin_path
    ) != phase04_pin_receipt.get("sha256"):
        raise ValueError("Phase-04 source pins used by replay acquisition drifted")
    acquired_rows = [
        row
        for row in acquisition.get("outputs", [])
        if row.get("role") == "foreign_replay"
    ]
    acquired = {Path(str(row["path"])).resolve(): row for row in acquired_rows}
    if len(acquired) != len(acquired_rows):
        raise ValueError("replay acquisition receipt has duplicate output paths")
    replay_paths = {Path(task["input_path"]).resolve() for task in foreign_tasks}
    if set(acquired) != replay_paths:
        raise ValueError("foreign replay inventory differs from acquisition receipt")
    for path, row in acquired.items():
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            raise ValueError(f"acquired replay payload drift: {path}")

    old_build_path = args.old_greek_build_receipt.resolve()
    old_build = read_json(old_build_path)
    if (
        old_build.get("schema_version") != "full_cpt_old_greek_build_receipt_v1"
        or old_build.get("status") != "completed"
        or old_build.get("acquisition_receipt_sha256") != sha256_file(acquisition_path)
    ):
        raise ValueError("old-Greek replay build is not bound to this acquisition")
    old_impl = Path(str(old_build.get("implementation", ""))).resolve()
    if (
        not old_impl.is_file()
        or sha256_file(old_impl) != old_build.get("implementation_sha256")
        or old_build.get("config_sha256") != acquisition.get("config_sha256")
    ):
        raise ValueError("old-Greek implementation/config receipt drift")
    old_output_rows = old_build.get("outputs", [])
    old_outputs = {Path(str(row["path"])).resolve(): row for row in old_output_rows}
    if len(old_outputs) != len(old_output_rows):
        raise ValueError("old-Greek build receipt has duplicate output paths")
    if set(old_outputs) != set(old_paths):
        raise ValueError("old-Greek replay inventory differs from build receipt")
    for path, row in old_outputs.items():
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            raise ValueError(f"old-Greek replay payload drift: {path}")

    payload = {
        "schema_version": "full_cpt_training_bridge_input_receipt_v1",
        "status": "completed",
        "completed_at": utc_now(),
        "recipe_id": config["recipe_id"],
        "config": {"path": str(config_path), "sha256": config_sha},
        "repository": {
            "root": str(args.repo_root.resolve()),
            "commit": repo_commit,
            "code_files": code_receipts,
            "code_sha256": canonical_sha256(code_receipts),
        },
        "megatron": {
            "root": str(args.megatron_dir.resolve()),
            "commit": megatron_commit,
            "tree": megatron_tree,
        },
        "replay_staging": {
            "acquisition_receipt": {
                "path": str(acquisition_path),
                "sha256": sha256_file(acquisition_path),
            },
            "old_greek_build_receipt": {
                "path": str(old_build_path),
                "sha256": sha256_file(old_build_path),
            },
        },
        "phase04": {
            "stage": str(args.phase04_stage.resolve()),
            "release_root": str((args.phase04_stage / "release").resolve()),
            "release_manifest": {
                "path": str((args.phase04_stage / "release_manifest.json").resolve()),
                "sha256": sha256_file(args.phase04_stage / "release_manifest.json"),
            },
            "validation": {
                "path": str((args.phase04_stage / "validation.json").resolve()),
                "sha256": sha256_file(args.phase04_stage / "validation.json"),
                "status": validation["status"],
            },
            "training_files": release_files,
            "training_rows": sum(row["rows"] for row in release_files),
        },
        "decontamination": {
            **decontamination,
            "implementation": {
                "path": str(decontam_script.resolve()),
                "sha256": sha256_file(decontam_script),
            },
        },
        "tokenizer": tokenizer,
        "mix": {
            "nominal_tokens": config["probe"]["nominal_tokens"],
            "numerators": mix,
            "denominator": denominator,
            "application": "Megatron --data-path sampling weights; source documents are never duplicated during binary construction",
        },
        "source_inventories": {
            "new_greek": release_files,
            "foreign_replay": foreign_inventory,
            "old_greek_replay": old_inventory,
        },
        "legacy_foreign_recipe": legacy_recipe,
        "tasks": tasks,
        "task_count": len(tasks),
        "task_plan_sha256": canonical_sha256(tasks),
    }
    write_json_atomic(args.output.resolve(), payload)
    print(
        json.dumps(
            {"ok": True, "output": str(args.output), "tasks": len(tasks)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
