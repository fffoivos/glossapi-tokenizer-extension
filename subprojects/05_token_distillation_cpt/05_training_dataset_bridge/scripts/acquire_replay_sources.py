#!/usr/bin/env python3
"""Receipt-bound Clariden restaging for every Phase-05 replay prerequisite.

The command resolves only full 40-hex Hugging Face revisions, stages the exact
matching file inventory, deterministically rewrites the pinned StarCoder subset,
and optionally restores a clean detached Megatron checkout.  It never resolves
``main`` and never silently replaces a non-matching local payload.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

from bridge_common import (
    HEX_COMMIT,
    file_tree_receipt,
    read_json,
    sha256_file,
    utc_now,
    validate_file_tree_receipt,
    write_json_atomic,
)


def _render(raw: str, scratch_root: Path) -> Path:
    return Path(raw.format(scratch_root=str(scratch_root.resolve()))).resolve()


def _copy_atomic(source: Path, destination: Path, *, replace: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise ValueError(f"unsafe existing staging target: {destination}")
        if destination.stat().st_size == source.stat().st_size and sha256_file(
            destination
        ) == sha256_file(source):
            return
        if not replace:
            raise ValueError(
                f"existing staging target differs; explicit replacement required: {destination}"
            )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.unlink()
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _parquet_receipt(
    path: Path,
    *,
    repo_id: str,
    revision: str,
    remote_path: str,
    source_name: str,
    role: str,
    transform: str = "identity",
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
        "columns": parquet.schema_arrow.names,
        "repo_id": repo_id,
        "revision": revision,
        "remote_path": remote_path,
        "source_name": source_name,
        "role": role,
        "transform": transform,
    }


def _glob_prefix(pattern: str) -> Path:
    parts = []
    for part in Path(pattern).parts:
        if any(character in part for character in "*?["):
            break
        parts.append(part)
    return Path(*parts)


def _rewrite_starcoder(
    source: Path,
    destination: Path,
    remote_path: str,
    *,
    replace: bool,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise ValueError(f"unsafe existing StarCoder staging target: {destination}")
    if destination.is_file():
        if not replace:
            raise ValueError(
                "an unreceipted transformed StarCoder shard cannot be trusted; "
                f"explicit replacement is required: {destination}"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    temporary.unlink(missing_ok=True)
    source_parquet = pq.ParquetFile(source)
    if "content" not in source_parquet.schema_arrow.names:
        raise ValueError(f"StarCoder source has no content column: {remote_path}")
    columns = [
        column
        for column in ("content", "id")
        if column in source_parquet.schema_arrow.names
    ]
    writer = None
    absolute_row = 0
    try:
        for batch in source_parquet.iter_batches(columns=columns, batch_size=20_000):
            data = batch.to_pydict()
            content = data["content"]
            identities = data.get("id")
            if identities is None:
                identities = list(range(absolute_row, absolute_row + len(content)))
            table = pa.table(
                {
                    "content": content,
                    "id": [str(value) for value in identities],
                    "doc_id": [
                        f"starcoderdata:{remote_path}:{value}" for value in identities
                    ],
                    "source_file": [remote_path] * len(content),
                }
            )
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
            absolute_row += len(content)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError(f"StarCoder source is empty: {remote_path}")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _parse_plan(raw: str) -> list[tuple[str, int]]:
    result: list[tuple[str, int]] = []
    for value in raw.split(","):
        language, count = value.split(":", 1)
        result.append((language.strip(), int(count)))
    if not result or any(not language or count <= 0 for language, count in result):
        raise ValueError("invalid StarCoder staging plan")
    return result


def _run_git(*args: str, cwd: Path | None = None) -> str:
    command = ["git", *(args if cwd is None else ("-C", str(cwd), *args))]
    result = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _stage_git_source(
    spec: Mapping[str, Any], scratch_root: Path, *, replace: bool
) -> dict[str, Any]:
    destination = _render(str(spec["destination"]), scratch_root)
    expected = str(spec["commit"])
    if not HEX_COMMIT.fullmatch(expected):
        raise ValueError("Git acquisition commit must be full immutable SHA-1")
    valid = False
    if destination.is_dir() and (destination / ".git").is_dir():
        try:
            valid = _run_git(
                "rev-parse", "HEAD", cwd=destination
            ) == expected and not _run_git(
                "status", "--porcelain", "--untracked-files=all", cwd=destination
            )
        except subprocess.CalledProcessError:
            valid = False
    if not valid:
        if destination.exists() and not replace:
            raise ValueError(
                "Megatron destination is not the clean pinned checkout; set the explicit "
                f"replacement gate after reviewing it: {destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.restage.partial")
        if temporary.exists():
            shutil.rmtree(temporary)
        _run_git("clone", "--no-checkout", str(spec["url"]), str(temporary))
        _run_git("checkout", "--detach", expected, cwd=temporary)
        if _run_git("rev-parse", "HEAD", cwd=temporary) != expected:
            raise ValueError("restaged Git checkout resolved the wrong commit")
        if destination.exists():
            backup = destination.with_name(f".{destination.name}.replaced")
            if backup.exists():
                raise FileExistsError(backup)
            os.replace(destination, backup)
            try:
                os.replace(temporary, destination)
            except BaseException:
                os.replace(backup, destination)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(temporary, destination)
    tree = file_tree_receipt(destination, exclude_top_level=(".git",))
    return {
        "name": spec["name"],
        "url": spec["url"],
        "commit": expected,
        "destination": str(destination),
        "clean": True,
        "tree": tree,
    }


def _validate_existing(receipt_path: Path, *, config_sha: str, script_sha: str) -> bool:
    if not receipt_path.is_file():
        return False
    value = read_json(receipt_path)
    if (
        value.get("schema_version") != "full_cpt_replay_acquisition_receipt_v1"
        or value.get("status") != "completed"
        or value.get("config_sha256") != config_sha
        or value.get("implementation_sha256") != script_sha
    ):
        raise ValueError("existing acquisition receipt is bound to different inputs")
    for row in value.get("outputs", []):
        path = Path(str(row["path"]))
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(row["bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            raise ValueError(f"existing acquired payload drift: {path}")
    phase04 = value.get("phase04_sources_config", {})
    phase04_path = Path(str(phase04.get("path", "")))
    if not phase04_path.is_file() or sha256_file(phase04_path) != phase04.get("sha256"):
        raise ValueError("tracked Phase-04 source pins drifted after acquisition")
    for row in value.get("git_sources", []):
        destination = Path(str(row["destination"]))
        if _run_git("rev-parse", "HEAD", cwd=destination) != row["commit"]:
            raise ValueError(f"existing acquired Git commit drift: {destination}")
        if _run_git("status", "--porcelain", "--untracked-files=all", cwd=destination):
            raise ValueError(f"existing acquired Git tree is dirty: {destination}")
        validate_file_tree_receipt(row["tree"])
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    if config.get("schema_version") != "full_cpt_replay_acquisition_config_v1":
        raise ValueError("unsupported replay acquisition config")
    if config.get("status") != "ready_for_receipt_bound_acquisition":
        raise ValueError(
            "replay acquisition config is not operator-adjudicated; set every "
            "historical revision and then explicitly mark status "
            "ready_for_receipt_bound_acquisition"
        )
    phase04_config_path = (
        args.config.resolve().parent / str(config["phase04_sources_config"])
    ).resolve()
    phase04_sources = read_json(phase04_config_path)
    for repository in config.get("repositories", []):
        section = repository.get("phase04_pin")
        if not section:
            continue
        pinned = phase04_sources.get(str(section), {})
        if pinned.get("repo_id") != repository.get("repo_id") or pinned.get(
            "revision"
        ) != repository.get("revision"):
            raise ValueError(
                f"replay acquisition disagrees with tracked Phase-04 pin: {section}"
            )
    unresolved = [
        str(row.get("repo_id"))
        for row in config.get("repositories", [])
        if not HEX_COMMIT.fullmatch(str(row.get("revision") or ""))
    ]
    if unresolved:
        raise ValueError(
            "operator must adjudicate historically intended immutable dataset "
            "revisions before acquisition: " + ", ".join(unresolved)
        )
    config_sha = sha256_file(args.config.resolve())
    script_sha = sha256_file(Path(__file__).resolve())
    if _validate_existing(args.output, config_sha=config_sha, script_sha=script_sha):
        print(json.dumps({"ok": True, "resumed": True, "output": str(args.output)}))
        return 0

    from huggingface_hub import HfApi, hf_hub_download

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    outputs: list[dict[str, Any]] = []
    repository_receipts: list[dict[str, Any]] = []
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    for repository in config["repositories"]:
        repo_id = str(repository["repo_id"])
        revision = str(repository["revision"])
        if not HEX_COMMIT.fullmatch(revision):
            raise ValueError(f"dataset revision is not immutable: {repo_id}")
        resolved = api.repo_info(
            repo_id, repo_type="dataset", revision=revision, token=token
        ).sha
        if resolved != revision:
            raise ValueError(f"dataset revision resolution drift: {repo_id}")
        files = sorted(
            api.list_repo_files(
                repo_id, repo_type="dataset", revision=revision, token=token
            )
        )
        selected_for_repo: list[str] = []
        if repository.get("transform") == "starcoder_stable_doc_id_v1":
            selected: list[str] = []
            for language, count in _parse_plan(str(repository["plan"])):
                candidates = sorted(
                    value
                    for value in files
                    if value.startswith(f"{language}/") and value.endswith(".parquet")
                )
                if len(candidates) < count:
                    raise ValueError(
                        f"{repo_id}/{language}: requested {count}, found {len(candidates)}"
                    )
                selected.extend(candidates[:count])
            destination_root = _render(
                str(repository["destination_root"]), args.scratch_root
            )
            expected_paths = {destination_root / remote for remote in selected}
            actual_paths = (
                set(destination_root.rglob("*.parquet"))
                if destination_root.exists()
                else set()
            )
            unexpected = actual_paths - expected_paths
            if unexpected and not args.replace_existing:
                raise ValueError(
                    "unexpected StarCoder staging files require explicit replacement: "
                    + str(sorted(unexpected)[:5])
                )
            for path in unexpected:
                path.unlink()
            for remote_path in selected:
                cached = Path(
                    hf_hub_download(
                        repo_id=repo_id,
                        repo_type="dataset",
                        filename=remote_path,
                        revision=revision,
                        cache_dir=str(cache_dir),
                        token=token,
                    )
                )
                destination = destination_root / remote_path
                _rewrite_starcoder(
                    cached,
                    destination,
                    remote_path,
                    replace=args.replace_existing,
                )
                outputs.append(
                    _parquet_receipt(
                        destination,
                        repo_id=repo_id,
                        revision=revision,
                        remote_path=remote_path,
                        source_name=str(repository["source_name"]),
                        role=str(repository["role"]),
                        transform="starcoder_stable_doc_id_v1",
                    )
                )
            selected_for_repo.extend(selected)
        else:
            for mapping in repository.get("mappings", []):
                pattern = str(mapping["remote_glob"])
                selected = sorted(
                    value for value in files if fnmatch.fnmatch(value, pattern)
                )
                if not selected:
                    raise ValueError(f"{repo_id}: no files match {pattern!r}")
                destination_root = _render(
                    str(mapping["destination_root"]), args.scratch_root
                )
                prefix_root = destination_root / _glob_prefix(pattern)
                expected_paths = {destination_root / remote for remote in selected}
                actual_paths = (
                    set(prefix_root.rglob("*.parquet"))
                    if prefix_root.exists()
                    else set()
                )
                unexpected = actual_paths - expected_paths
                if unexpected and not args.replace_existing:
                    raise ValueError(
                        f"{repo_id}/{pattern}: unexpected local Parquets: "
                        + str(sorted(unexpected)[:5])
                    )
                for path in unexpected:
                    path.unlink()
                for remote_path in selected:
                    cached = Path(
                        hf_hub_download(
                            repo_id=repo_id,
                            repo_type="dataset",
                            filename=remote_path,
                            revision=revision,
                            cache_dir=str(cache_dir),
                            token=token,
                        )
                    )
                    destination = destination_root / remote_path
                    _copy_atomic(cached, destination, replace=args.replace_existing)
                    outputs.append(
                        _parquet_receipt(
                            destination,
                            repo_id=repo_id,
                            revision=revision,
                            remote_path=remote_path,
                            source_name=str(mapping["source_name"]),
                            role=str(mapping["role"]),
                        )
                    )
                selected_for_repo.extend(selected)
        repository_receipts.append(
            {
                "repo_id": repo_id,
                "revision": revision,
                "selected_files": sorted(selected_for_repo),
                "selected_file_count": len(selected_for_repo),
            }
        )

    git_receipts = [
        _stage_git_source(spec, args.scratch_root, replace=args.replace_existing)
        for spec in config.get("git_sources", [])
    ]
    payload = {
        "schema_version": "full_cpt_replay_acquisition_receipt_v1",
        "status": "completed",
        "completed_at": utc_now(),
        "config": str(args.config.resolve()),
        "config_sha256": config_sha,
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": script_sha,
        "scratch_root": str(args.scratch_root.resolve()),
        "phase04_sources_config": {
            "path": str(phase04_config_path),
            "sha256": sha256_file(phase04_config_path),
        },
        "repositories": repository_receipts,
        "outputs": sorted(outputs, key=lambda row: row["path"]),
        "output_count": len(outputs),
        "git_sources": git_receipts,
    }
    write_json_atomic(args.output.resolve(), payload)
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output.resolve()),
                "files": len(outputs),
                "git_sources": len(git_receipts),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
