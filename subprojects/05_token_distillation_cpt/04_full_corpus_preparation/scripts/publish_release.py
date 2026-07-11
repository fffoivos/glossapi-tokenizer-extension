#!/usr/bin/env python3
"""Dry-run by default; publish only an exactly validated public release."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from finalization_io import read_json_object, sha256_file, utc_now, write_json_atomic
from materialize_release import INTEGRITY_CONTRACT_VERSION


PUBLICATION_SCHEMA = "full_cpt_publication_receipt_v1"
ALLOWED_REMOTE_SYSTEM_FILES = {".gitattributes"}


def _field(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _assert_no_symlink_path(path: Path, *, stop: Path) -> None:
    current = path
    stop = stop.resolve()
    while True:
        if current.is_symlink():
            raise ValueError(f"publication path contains a symlink: {current}")
        if current.resolve() == stop:
            return
        if current.parent == current:
            raise ValueError(f"publication path {path} is not below release root {stop}")
        current = current.parent


def validated_public_root(release: Path, manifest: Mapping[str, Any], validation: Mapping[str, Any]) -> Path:
    if release.is_symlink():
        raise ValueError("release root may not be a symlink")
    release_real = release.resolve(strict=True)
    if manifest.get("redistribution_root") != "redistribution/data":
        raise ValueError("manifest redistribution_root must be exactly 'redistribution/data'")
    expected = release / "redistribution" / "data"
    _assert_no_symlink_path(expected, stop=release)
    public_root = expected.resolve(strict=True)
    if public_root != release_real / "redistribution" / "data":
        raise ValueError("resolved public root is not exactly <release>/redistribution/data")
    siblings = list(expected.parent.iterdir())
    if len(siblings) != 1 or siblings[0].name != "data" or siblings[0].is_symlink():
        raise ValueError("<release>/redistribution may contain only the validated data directory")
    inventory = validation.get("publication_inventory")
    if not isinstance(inventory, Mapping):
        raise ValueError("validation receipt lacks publication_inventory")
    if Path(str(inventory.get("root", ""))).resolve() != public_root:
        raise ValueError("validation receipt is bound to a different public root")
    return public_root


def _walk_public_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for directory in directories:
            candidate = current_path / directory
            if candidate.is_symlink():
                raise ValueError(f"public data tree contains a symlink: {candidate}")
        for name in names:
            candidate = current_path / name
            if candidate.is_symlink():
                raise ValueError(f"public data tree contains a symlink: {candidate}")
            if name.startswith(".") or candidate.suffix != ".parquet":
                raise ValueError(f"public data tree contains a non-Parquet file: {candidate}")
            files.append(candidate)
    if not files:
        raise ValueError("public data tree has no Parquet files")
    return sorted(files)


def verify_local_public_inventory(
    public_root: Path, validation: Mapping[str, Any]
) -> list[dict[str, Any]]:
    inventory = validation.get("publication_inventory")
    if not isinstance(inventory, Mapping) or not isinstance(inventory.get("files"), list):
        raise ValueError("validation receipt has no publication file inventory")
    expected: dict[str, dict[str, Any]] = {}
    for row in inventory["files"]:
        if not isinstance(row, dict):
            raise ValueError("publication inventory row is not an object")
        relative = str(row.get("path", ""))
        path = Path(relative)
        if not relative or path.is_absolute() or ".." in path.parts or path.suffix != ".parquet":
            raise ValueError(f"unsafe publication inventory path: {relative!r}")
        if relative in expected:
            raise ValueError(f"duplicate publication inventory path: {relative}")
        if row.get("remote_path") != f"data/{path.as_posix()}":
            raise ValueError(f"publication inventory remote path mismatch: {relative}")
        expected[relative] = row

    actual = {path.relative_to(public_root).as_posix(): path for path in _walk_public_files(public_root)}
    if set(actual) != set(expected):
        raise ValueError(
            "public data inventory differs from validation receipt: "
            f"extra={sorted(set(actual) - set(expected))}, missing={sorted(set(expected) - set(actual))}"
        )
    verified: list[dict[str, Any]] = []
    for relative in sorted(expected):
        path = actual[relative]
        row = expected[relative]
        # This is deliberately a full byte pass immediately before upload.
        actual_sha = sha256_file(path)
        actual_bytes = path.stat().st_size
        if actual_sha != row.get("sha256") or actual_bytes != int(row.get("bytes", -1)):
            raise ValueError(f"public file drifted after validation: {path}")
        verified.append(
            {
                "path": relative,
                "remote_path": str(row["remote_path"]),
                "sha256": actual_sha,
                "bytes": actual_bytes,
                "rows": int(row.get("rows", -1)),
            }
        )
    if sum(row["bytes"] for row in verified) != int(inventory.get("bytes", -1)):
        raise ValueError("publication inventory byte total does not reconcile")
    if sum(row["rows"] for row in verified) != int(inventory.get("rows", -1)):
        raise ValueError("publication inventory row total does not reconcile")
    return verified


def validated_token_waterfall(manifest: Mapping[str, Any]) -> Path:
    """Return the manifest-bound waterfall only when its current bytes match."""

    path = Path(str(manifest.get("token_waterfall", "")))
    expected = manifest.get("token_waterfall_sha256")
    if (
        not path.is_file()
        or path.is_symlink()
        or not isinstance(expected, str)
        or len(expected) != 64
        or sha256_file(path) != expected
    ):
        raise ValueError("token waterfall drifted after release validation")
    return path


def _build_hardlink_upload_tree(
    *, public_root: Path, inventory: Iterable[Mapping[str, Any]], staging_root: Path
) -> Path:
    """Create an exact data/ upload tree without mutating the release.

    ``upload_large_folder`` stores restart metadata below its folder root.  A
    temporary hardlink tree keeps that cache outside the immutable release and
    avoids a second full corpus copy.
    """

    for row in inventory:
        relative = Path(str(row["path"]))
        source = public_root / relative
        destination = staging_root / "data" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination, follow_symlinks=False)
        except OSError as exc:
            raise RuntimeError(
                "publication staging requires same-filesystem hardlinks; no bulk copy fallback is permitted"
            ) from exc
    return staging_root


def _remote_files(api: Any, *, repo_id: str, revision: str | None = None) -> dict[str, object]:
    kwargs: dict[str, Any] = {"repo_id": repo_id, "repo_type": "dataset", "recursive": True, "expand": True}
    if revision is not None:
        kwargs["revision"] = revision
    rows = api.list_repo_tree(**kwargs)
    result: dict[str, object] = {}
    for row in rows:
        path = _field(row, "path")
        if not isinstance(path, str):
            continue
        # RepoFolder objects have no size; files do.
        if _field(row, "size") is None:
            continue
        result[path] = row
    return result


def _remote_lfs_sha(row: object) -> str | None:
    lfs = _field(row, "lfs")
    value = _field(lfs, "sha256") if lfs is not None else None
    return value if isinstance(value, str) and value else None


def _download_remote_sha256(
    *, repo_id: str, remote_path: str, revision: str, token: str, temporary_directory: Path
) -> str:
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=remote_path,
        repo_type="dataset",
        revision=revision,
        token=token,
        local_dir=temporary_directory,
    )
    return sha256_file(Path(downloaded))


def verify_remote_inventory(
    api: Any,
    *,
    repo_id: str,
    commit_sha: str,
    token: str,
    expected: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    expected_by_path = {str(row["remote_path"]): row for row in expected}
    remote = _remote_files(api, repo_id=repo_id, revision=commit_sha)
    extras = set(remote) - set(expected_by_path) - ALLOWED_REMOTE_SYSTEM_FILES
    missing = set(expected_by_path) - set(remote)
    if extras or missing:
        raise RuntimeError(f"remote inventory mismatch: extra={sorted(extras)}, missing={sorted(missing)}")
    verified: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="full-cpt-publish-verify-") as temporary:
        temp_root = Path(temporary)
        for remote_path in sorted(expected_by_path):
            expected_row = expected_by_path[remote_path]
            remote_row = remote[remote_path]
            size = int(_field(remote_row, "size", -1))
            if size != int(expected_row["bytes"]):
                raise RuntimeError(f"remote file size mismatch: {remote_path}")
            actual_sha = _remote_lfs_sha(remote_row)
            verification = "lfs_sha256"
            if actual_sha is None:
                actual_sha = _download_remote_sha256(
                    repo_id=repo_id,
                    remote_path=remote_path,
                    revision=commit_sha,
                    token=token,
                    temporary_directory=temp_root,
                )
                verification = "downloaded_sha256"
            if actual_sha != expected_row["sha256"]:
                raise RuntimeError(f"remote file checksum mismatch: {remote_path}")
            verified.append(
                {
                    "path": remote_path,
                    "bytes": size,
                    "sha256": actual_sha,
                    "verification": verification,
                }
            )
    return verified


def require_new_empty_remote(api: Any, *, repo_id: str) -> None:
    """Fail closed on any payload, including remnants of a partial upload."""

    existing = _remote_files(api, repo_id=repo_id)
    stale = set(existing) - ALLOWED_REMOTE_SYSTEM_FILES
    if stale:
        raise RuntimeError(
            "new-empty publication mode refuses an existing remote payload: "
            f"{sorted(stale)}. Partial uploads are not resumed because the "
            "temporary uploader cache is intentionally discarded. Inspect and "
            "delete/recreate the dataset repository, or choose a new empty "
            "--repo-id; the publisher performs no remote cleanup."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--validation-receipt", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--gate-mode", choices=["manual"], default="manual")
    parser.add_argument("--remote-mode", choices=["new-empty"], default="new-empty")
    parser.add_argument("--output", type=Path, required=True, help="Immutable local publication receipt")
    parser.add_argument("--execute", action="store_true", help="Actually create and upload")
    parser.add_argument("--token", default=None, help="Defaults to HF_TOKEN; never written to receipts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable publication receipt: {args.output}")
    manifest = read_json_object(args.release_manifest)
    validation = read_json_object(args.validation_receipt)
    if (
        manifest.get("schema_version") != "full_cpt_release_manifest_v1"
        or manifest.get("integrity_contract_version") != INTEGRITY_CONTRACT_VERSION
    ):
        raise ValueError("release manifest schema/integrity contract is unsupported")
    if (
        validation.get("schema_version") != "full_cpt_release_validation_v1"
        or validation.get("integrity_contract_version") != INTEGRITY_CONTRACT_VERSION
        or validation.get("status") != "passed"
        or validation.get("failed_checks") != []
    ):
        raise ValueError("publication requires a passed, zero-failure full-corpus validation receipt")
    manifest_sha = sha256_file(args.release_manifest)
    if validation.get("release_manifest_sha256") != manifest_sha:
        raise ValueError("validation receipt is bound to a different release manifest")
    if Path(str(manifest.get("output"))).resolve() != args.release.resolve():
        raise ValueError("release root differs from the immutable manifest")
    if Path(str(validation.get("release"))).resolve() != args.release.resolve():
        raise ValueError("validation receipt is bound to a different release root")

    license_receipt = manifest.get("source_license_adjudication")
    if not isinstance(license_receipt, Mapping):
        raise ValueError("release manifest lacks a source-license adjudication receipt")
    if validation.get("source_license_adjudication") != license_receipt:
        raise ValueError("validation receipt is bound to a different source-license adjudication")
    if (
        license_receipt.get("schema_version")
        != "full_cpt_source_license_adjudication_v1"
        or license_receipt.get("status") != "technical_audit_complete"
    ):
        raise ValueError("publication requires a completed technical source-license audit")
    license_path = Path(str(license_receipt.get("path", "")))
    if not license_path.is_file() or sha256_file(license_path) != license_receipt.get("sha256"):
        raise ValueError("source-license adjudication drifted after release validation")

    public_root = validated_public_root(args.release, manifest, validation)
    data_inventory = verify_local_public_inventory(public_root, validation)
    metadata_claim = validation.get("publication_metadata_inventory")
    manifest_card = manifest.get("dataset_card")
    if not isinstance(metadata_claim, list) or len(metadata_claim) != 1:
        raise ValueError("validation receipt lacks the exact dataset-card inventory")
    if not isinstance(manifest_card, Mapping) or metadata_claim[0] != manifest_card:
        raise ValueError("validation dataset-card inventory differs from the release manifest")
    card_path = args.release / str(manifest_card.get("path", ""))
    if (
        not card_path.is_file()
        or card_path.is_symlink()
        or sha256_file(card_path) != manifest_card.get("sha256")
        or card_path.stat().st_size != int(manifest_card.get("bytes", -1))
        or manifest_card.get("remote_path") != "README.md"
    ):
        raise ValueError("dataset card drifted after release validation")
    metadata_inventory = [
        {
            "path": str(card_path.resolve()),
            "remote_path": "README.md",
            "sha256": sha256_file(card_path),
            "bytes": card_path.stat().st_size,
        }
    ]
    token_waterfall_path = validated_token_waterfall(manifest)
    provenance_paths: dict[str, Path] = {
        "provenance/release_manifest.json": args.release_manifest,
        "provenance/validation_receipt.json": args.validation_receipt,
        "provenance/token_waterfall.json": token_waterfall_path,
        "provenance/source_license_adjudication.json": license_path,
    }
    upstream = manifest.get("upstream_manifests")
    if not isinstance(upstream, Mapping):
        raise ValueError("release manifest has no upstream manifest bindings")
    for name in ("cleaning", "decontamination", "dedup"):
        row = upstream.get(name)
        if not isinstance(row, Mapping):
            raise ValueError(f"release manifest lacks upstream {name!r} receipt")
        path = Path(str(row.get("path", "")))
        if not path.is_file() or sha256_file(path) != row.get("sha256"):
            raise ValueError(f"upstream {name!r} manifest drifted after release validation")
        provenance_paths[f"provenance/upstream/{name}_manifest.json"] = path

    provenance_inventory: list[dict[str, Any]] = []
    for remote_path, path in sorted(provenance_paths.items()):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        actual_sha256 = sha256_file(path)
        if (
            remote_path == "provenance/token_waterfall.json"
            and actual_sha256 != manifest["token_waterfall_sha256"]
        ):
            raise ValueError("token waterfall drifted while preparing publication")
        provenance_inventory.append(
            {
                "path": str(path.resolve()),
                "remote_path": remote_path,
                "sha256": actual_sha256,
                "bytes": path.stat().st_size,
            }
        )
    expected_remote = [*data_inventory, *metadata_inventory, *provenance_inventory]
    base_receipt: dict[str, Any] = {
        "schema_version": PUBLICATION_SCHEMA,
        "integrity_contract_version": INTEGRITY_CONTRACT_VERSION,
        "completed_at": utc_now(),
        "status": "dry_run" if not args.execute else "passed",
        "repo_id": args.repo_id,
        "repo_type": "dataset",
        "gate_mode": "manual",
        "remote_mode": args.remote_mode,
        "storage_mode": "git_lfs_sha256_verifiable",
        "release": str(args.release.resolve()),
        "release_manifest": str(args.release_manifest.resolve()),
        "release_manifest_sha256": manifest_sha,
        "validation_receipt": str(args.validation_receipt.resolve()),
        "validation_receipt_sha256": sha256_file(args.validation_receipt),
        "redistribution_root": str(public_root),
        "counts": {
            "files": len(data_inventory),
            "metadata_files": len(metadata_inventory),
            "provenance_files": len(provenance_inventory),
            "bytes": sum(int(row["bytes"]) for row in data_inventory),
            "rows": int(manifest["counts"]["redistribution_rows"]),
            "training_rows_not_uploaded": int(manifest["counts"]["training_rows"])
            - int(manifest["counts"]["redistribution_rows"]),
        },
        "local_inventory": expected_remote,
        "execute": args.execute,
        "commit_sha": None,
        "remote_inventory": [],
    }
    if not args.execute:
        write_json_atomic(args.output, base_receipt)
        print(json.dumps({"ok": True, "dry_run": True, "output": str(args.output)}, sort_keys=True))
        return 0

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("--execute requires --token or HF_TOKEN")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=False, exist_ok=True)
    require_new_empty_remote(api, repo_id=args.repo_id)
    api.update_repo_settings(
        repo_id=args.repo_id,
        repo_type="dataset",
        gated="manual",
        private=False,
        xet_enabled=False,
    )
    # Upload from a temporary hardlink tree so the uploader's restart cache
    # cannot mutate the immutable release tree.
    with tempfile.TemporaryDirectory(prefix="full-cpt-upload-", dir=args.output.parent) as temporary:
        upload_root = _build_hardlink_upload_tree(
            public_root=public_root,
            inventory=data_inventory,
            staging_root=Path(temporary),
        )
        api.upload_large_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=str(upload_root),
            allow_patterns=["**/*.parquet"],
        )
    for row in metadata_inventory:
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="dataset",
            path_or_fileobj=row["path"],
            path_in_repo=row["remote_path"],
        )
    for row in provenance_inventory:
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="dataset",
            path_or_fileobj=row["path"],
            path_in_repo=row["remote_path"],
        )
    info = api.repo_info(repo_id=args.repo_id, repo_type="dataset", files_metadata=True)
    commit_sha = _field(info, "sha")
    if not isinstance(commit_sha, str) or not commit_sha:
        raise RuntimeError("Hugging Face did not return a final dataset commit SHA")
    if (
        _field(info, "gated") != "manual"
        or bool(_field(info, "private", False))
        or bool(_field(info, "xet_enabled", True))
    ):
        raise RuntimeError(
            "final Hugging Face repository is not public/manual-gated with SHA-verifiable Git LFS storage"
        )
    remote_inventory = verify_remote_inventory(
        api,
        repo_id=args.repo_id,
        commit_sha=commit_sha,
        token=token,
        expected=expected_remote,
    )
    base_receipt["commit_sha"] = commit_sha
    base_receipt["remote_inventory"] = remote_inventory
    write_json_atomic(args.output, base_receipt)
    print(
        json.dumps(
            {
                "ok": True,
                "dry_run": False,
                "repo_id": args.repo_id,
                "commit_sha": commit_sha,
                "receipt": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
